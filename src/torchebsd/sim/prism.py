"""PRISM scattering-matrix backend: batched, GPU-native, no eigendecomposition.

The reference :class:`~torchebsd.sim.dynamical.BlochWaveBackend` diagonalizes the
dynamical matrix per direction; that eigendecomposition is complex, small, and
non-Hermitian, so it neither batches nor GPU-accelerates. This backend replaces
it with the physically equivalent scattering-matrix propagation:

    psi(z) = exp(2 pi i * A / (2 kn) * z) psi(0),   psi(0) = e_gzero
    L_gh   = (dz / thick) * sum_z lambda_E[z] conj(psi_g(z)) psi_h(z)

The propagator is one batched ``matrix_exp`` (scaling-and-squaring; pure matmul)
followed by a depth recurrence -- all operations the GPU is fast at. The whole
per-direction pipeline is vectorized: excitation errors, the Bethe strong/weak
partition, the dynamical-matrix assembly (with weak-beam folding), the
propagation, and the Sgh reduction all run batched, bucketed by strong-beam
count so each batch is a uniform ``[B, S, S]``.

It reproduces ``BlochWaveBackend`` to machine precision in float64 (rel err
~1e-13) and to ~1e-4 in float32. Because it is exact against the reference, it
inherits the reference's correlation against EMsoft -- it is the practical way
to run a native-resolution master (~10^6 directions) that the per-direction
float64 reference cannot. Measured on a Pascal TITAN X: a full npx-500 Si master
(~2e6 directions) in ~50 s at r = 1.0000 vs EMsoft's ``Si_MC.h5``.
"""

from __future__ import annotations

import numpy as np
import torch

import torchebsd.sim.reflections as _R
from torchebsd.sim.dynamical import BlochWaveBackend, SimulationInputs
from torchebsd.sim.reflections import excitation_errors, master_reflection_list


class BlochPrismBackend(BlochWaveBackend):
    """Scattering-matrix (PRISM) Bloch backend; see the module docstring.

    ``cdtype`` selects the working precision: ``torch.complex128`` (default,
    exact against the reference) or ``torch.complex64`` (faster, ~1e-4).

    The whole intensity path is torch (``matrix_exp`` + a depth recurrence), so
    it is autograd-differentiable: with a ``requires_grad`` structure-factor LUT
    (``u_lut``) gradients flow through the Bloch solve back to ``U_g`` (matching
    finite differences; M4). Connecting ``U_g`` to crystal parameters needs torch
    structure factors -- the remaining M4 piece.
    """

    name = "bloch_prism"

    def __init__(self, cdtype: torch.dtype = torch.complex128):
        self.cdtype = cdtype
        self.rdtype = torch.float64 if cdtype == torch.complex128 else torch.float32
        # class_chunk bounds the [chunk, Cmax, Cmax] candidate-Bethe tensor; block
        # is classified up front so buckets span it (large, GPU-efficient batches);
        # matmul_batch caps the [B, S, S] matrix_exp batch.
        self.class_chunk = 1024
        self.block = 65536
        self.matmul_batch = 4096

    # ---- setup ---------------------------------------------------------- #
    def prepare(self, inputs: SimulationInputs) -> None:
        super().prepare(inputs)  # sf, bethe, depth profile, Sgh LUT, kn, gzero, n_atoms, hmax
        dev = torch.device(self.device if self.device is not None else "cpu")
        self.tdev = dev
        self.off = 2 * self.hmax
        self.u_lut = torch.as_tensor(self.sf.ucg_lut, device=dev, dtype=self.cdtype)
        self.sgh_t = torch.as_tensor(self.sgh_lut, device=dev, dtype=self.cdtype)
        self.lam = float(self.sf.wavelength)
        self.upz0 = float(self.sf.upz0)
        self.lambda_t = torch.as_tensor(self.lambda_e, device=dev, dtype=self.rdtype)
        self.n_depths = self.lambda_t.shape[0]

        # fixed, direction-independent tables (g = 0 prepended at index 0)
        master = master_reflection_list(self.sf)
        z1 = np.zeros((1, 3), np.int64)
        self.hkl_aug = torch.as_tensor(np.concatenate([z1, master.hkl]), device=dev)
        self.n_beams = self.hkl_aug.shape[0]
        u0 = float(np.abs(self.sf.lookup(np.zeros(3, np.int64))))
        self.u_mod = torch.as_tensor(
            np.concatenate([[u0], master.u_mod]), device=dev, dtype=self.rdtype)
        self.forbidden = torch.as_tensor(
            np.concatenate([[False], master.forbidden]), device=dev)
        self.g_cart = torch.as_tensor(
            self.sf.crystal.g_cartesian(master.hkl), device=dev, dtype=self.rdtype)

    def _lookup(self, lut: torch.Tensor, hkl: torch.Tensor) -> torch.Tensor:
        i = hkl + self.off
        return lut[i[..., 0], i[..., 1], i[..., 2]]

    # ---- depth integration via the scattering matrix -------------------- #
    def _calc_lgh(self, a: torch.Tensor) -> torch.Tensor:
        """Batched L_gh for a [B, S, S] stack of dynamical matrices."""
        phase = 2.0 * np.pi * self.depthstep / (2.0 * self.kn)
        step = torch.complex(torch.zeros((), device=self.tdev, dtype=self.rdtype),
                             torch.as_tensor(phase, device=self.tdev, dtype=self.rdtype))
        sigma = torch.linalg.matrix_exp(step * a)          # exp(i phase A)
        psi = sigma[:, :, self.gzero].clone()              # psi_1 = Sigma e_gzero
        lgh = torch.zeros_like(a)
        for n in range(self.n_depths):
            lgh = lgh + self.lambda_t[n] * (psi.conj()[:, :, None] * psi[:, None, :])
            psi = torch.matmul(sigma, psi[:, :, None]).squeeze(-1)
        return lgh * (self.depthstep / self.thick)

    # ---- one intensity per incident direction --------------------------- #
    def intensities(self, directions: torch.Tensor) -> torch.Tensor:
        k_all = directions.to(self.tdev, self.rdtype)
        total = k_all.shape[0]
        out = torch.empty(total, device=self.tdev, dtype=self.out_dtype)
        for m0 in range(0, total, self.block):
            k = k_all[m0:m0 + self.block]
            strong, weak, sg = self._classify(k)
            counts = strong.sum(1)
            for s in torch.unique(counts).tolist():
                sub = torch.nonzero(counts == s, as_tuple=False).squeeze(1)
                for b0 in range(0, sub.shape[0], self.matmul_batch):
                    self._bucket(int(s), sub[b0:b0 + self.matmul_batch],
                                 strong, weak, sg, out, m0)
        return out

    def _classify(self, k: torch.Tensor):
        """Strong/weak boolean masks [D, N] and excitation errors sg [D, N].

        The Bethe strength ``m_g = min_h |s_g| / (lam |U_{g-h}|)`` ranges only
        over candidate beams h, so the partition is done on the candidate subset
        (~tens of beams) gathered to ``[chunk, Cmax, Cmax]`` rather than the full
        ``[chunk, N, N]`` -- exact, but ~40x cheaper. Masks are accumulated over
        the whole block so the caller can bucket globally.
        """
        bethe = self.bethe
        d = k.shape[0]
        strong = torch.zeros(d, self.n_beams, dtype=torch.bool, device=self.tdev)
        weak = torch.zeros_like(strong)
        sg_all = torch.empty(d, self.n_beams, device=self.tdev, dtype=self.rdtype)
        for c0 in range(0, d, self.class_chunk):
            kc = k[c0:c0 + self.class_chunk]
            c = kc.shape[0]
            sg = excitation_errors(self.g_cart, kc, self.lam)
            sg = torch.cat(
                [torch.zeros(c, 1, device=self.tdev, dtype=self.rdtype), sg], dim=1)
            absg = sg.abs()
            cand = (absg / (self.lam * self.u_mod.clamp_min(1e-300))) <= bethe.cutoff
            cand[:, self.forbidden] = absg[:, self.forbidden] <= bethe.sg_double
            cand[:, 0] = True

            # gather the candidate beams to a padded [c, Cmax] index list
            cc = cand.sum(1)
            cmax = int(cc.max().item())
            nz = cand.nonzero(as_tuple=False)                  # [T, 2], row-sorted
            rows = nz[:, 0]
            starts = torch.zeros(c, dtype=torch.long, device=self.tdev)
            starts[1:] = cc.cumsum(0)[:-1]
            within = torch.arange(rows.shape[0], device=self.tdev) - starts[rows]
            cidx = torch.zeros(c, cmax, dtype=torch.long, device=self.tdev)
            creal = torch.zeros(c, cmax, dtype=torch.bool, device=self.tdev)
            cidx[rows, within] = nz[:, 1]
            creal[rows, within] = True

            # Bethe min over candidate h only
            absg_c = absg.gather(1, cidx)                      # [c, Cmax]
            hkl_c = self.hkl_aug[cidx]                         # [c, Cmax, 3]
            u = self._lookup(
                self.u_lut, hkl_c[:, :, None, :] - hkl_c[:, None, :, :]).abs().to(self.rdtype)
            good = (u > _R._FORBIDDEN_TOL) & creal[:, None, :]
            rh = torch.where(good, (absg_c[:, :, None] / self.lam) / u.clamp_min(1e-300),
                             torch.full_like(u, 1.0e4))
            m_c = rh.min(dim=2).values                         # [c, Cmax]
            st_c = (m_c <= bethe.strong) & creal
            wk_c = (m_c > bethe.strong) & (m_c <= bethe.weak) & creal

            st = torch.zeros(c, self.n_beams, dtype=torch.bool, device=self.tdev)
            wk = torch.zeros_like(st)
            st.scatter_(1, cidx, st_c)
            wk.scatter_(1, cidx, wk_c)
            st[:, 0] = True
            st[:, self.forbidden] = False
            wk[:, self.forbidden] = False
            wk[:, 0] = False
            strong[c0:c0 + c] = st
            weak[c0:c0 + c] = wk
            sg_all[c0:c0 + c] = sg
        return strong, weak, sg_all

    def _bucket(self, s, sub, strong, weak, sg, out, c0):
        """Process the directions whose strong-beam count is exactly ``s``."""
        b = sub.shape[0]
        st, wk, sgs = strong[sub], weak[sub], sg[sub]
        s_idx = st.nonzero(as_tuple=False)[:, 1].reshape(b, s)   # ascending -> g0 first
        hkl_s = self.hkl_aug[s_idx]                              # [b, s, 3]
        sg_s = sgs.gather(1, s_idx)                              # [b, s]

        wc = wk.sum(1)
        w_max = int(wc.max().item()) if b else 0
        a = self._lookup(self.u_lut, hkl_s[:, :, None, :] - hkl_s[:, None, :, :]).clone()
        weak_diag = torch.zeros(b, s, device=self.tdev, dtype=self.rdtype)
        if w_max > 0:
            nz = wk.nonzero(as_tuple=False)                     # [T, 2], row-sorted
            rows, cols = nz[:, 0], nz[:, 1]
            starts = torch.zeros(b, dtype=torch.long, device=self.tdev)
            starts[1:] = wc.cumsum(0)[:-1]
            within = torch.arange(rows.shape[0], device=self.tdev) - starts[rows]
            w_idx = torch.zeros(b, w_max, dtype=torch.long, device=self.tdev)
            w_ok = torch.zeros(b, w_max, dtype=torch.bool, device=self.tdev)
            w_idx[rows, within] = cols
            w_ok[rows, within] = True
            hkl_w = self.hkl_aug[w_idx]
            sg_w = sgs.gather(1, w_idx)
            inv_sgw = torch.where(w_ok, 1.0 / sg_w, torch.zeros_like(sg_w))
            u_gw = self._lookup(self.u_lut, hkl_s[:, :, None, :] - hkl_w[:, None, :, :])
            u_wh = self._lookup(self.u_lut, hkl_w[:, :, None, :] - hkl_s[:, None, :, :])
            a = a - 0.5 * self.lam * torch.matmul(u_gw * inv_sgw[:, None, :], u_wh)
            weak_diag = 0.5 * self.lam * (u_gw.abs() ** 2 * inv_sgw[:, None, :]).sum(2)

        di = torch.arange(s, device=self.tdev)
        diag = (2.0 * sg_s / self.lam - weak_diag).to(self.cdtype) + 1j * self.upz0
        a[:, di, di] = diag

        lgh = self._calc_lgh(a)
        sgh = self._lookup(self.sgh_t, hkl_s[:, None, :, :] - hkl_s[:, :, None, :])  # [g,h]=h-g
        inten = (lgh * sgh).sum((1, 2)).real / self.n_atoms
        out[sub + c0] = inten.to(self.out_dtype)
