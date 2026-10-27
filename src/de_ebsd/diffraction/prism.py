"""Scattering-matrix Bloch backend: the batched, differentiable form of ``bloch``.

Replaces the per-direction dynamical matrix with batched solves of the
scattering matrix, much faster for dense direction grids and equal to
:class:`~de_ebsd.diffraction.backends.BlochWaveBackend` in double precision.
"""

from __future__ import annotations

import numpy as np
import torch

import de_ebsd.diffraction.reflections as _R
from de_ebsd.diffraction.backends import BlochWaveBackend, SimulationInputs
from de_ebsd.diffraction.reflections import excitation_errors, master_reflection_list


class BlochPrismBackend(BlochWaveBackend):
    """Scattering-matrix (PRISM) Bloch backend: one shared eigen-decomposition.

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
        super().prepare(
            inputs
        )  # structure_factors, bethe, depth profile, Sgh LUT, kn, gzero, n_atoms, hmax
        dev = torch.device(self.device if self.device is not None else "cpu")
        self.tdev = dev
        self.off = 2 * self.hmax
        self.u_lut = torch.as_tensor(self.structure_factors.ucg_lut, device=dev, dtype=self.cdtype)
        self.sgh_t = torch.as_tensor(self.sgh_lut, device=dev, dtype=self.cdtype)
        self.lam = float(self.structure_factors.wavelength)
        self.upz0 = float(self.structure_factors.upz0)
        self.lambda_t = torch.as_tensor(self.lambda_e, device=dev, dtype=self.rdtype)
        self.n_depths = self.lambda_t.shape[0]

        # fixed, direction-independent tables (g = 0 prepended at index 0)
        master = master_reflection_list(self.structure_factors)
        z1 = np.zeros((1, 3), np.int64)
        self.hkl_aug = torch.as_tensor(np.concatenate([z1, master.hkl]), device=dev)
        self.n_beams = self.hkl_aug.shape[0]
        u0 = float(np.abs(self.structure_factors.lookup(np.zeros(3, np.int64))))
        self.u_mod = torch.as_tensor(
            np.concatenate([[u0], master.u_mod]), device=dev, dtype=self.rdtype
        )
        self.forbidden = torch.as_tensor(np.concatenate([[False], master.forbidden]), device=dev)
        self.g_cart = torch.as_tensor(
            self.structure_factors.crystal.g_cartesian(master.hkl), device=dev, dtype=self.rdtype
        )

    def _pairs(self) -> None:
        """Every pair of beams' U(h_g - h_h) and S(h_h - h_g): the Bethe test and
        each bucket gather from these small matrices instead of building
        (directions, beams, beams, 3) index differences to look up in the 3D
        tables, which was most of their time. Taken from the tables as they are
        when :meth:`intensities` is called, so a ``u_lut`` swapped in for its
        gradient is the one used."""
        hkl = self.hkl_aug
        self.u_pair = self._lookup(self.u_lut, hkl[:, None, :] - hkl[None, :, :])
        self.u_pair_abs = self.u_pair.detach().abs().to(self.rdtype)
        self.sgh_pair = self._lookup(self.sgh_t, hkl[None, :, :] - hkl[:, None, :])  # [g,h]=h-g

    def _lookup(self, lut: torch.Tensor, hkl: torch.Tensor) -> torch.Tensor:
        i = hkl + self.off
        return lut[i[..., 0], i[..., 1], i[..., 2]]

    # ---- depth integration via the scattering matrix -------------------- #
    def _depth_intensity(self, a: torch.Tensor, sgh: torch.Tensor) -> torch.Tensor:
        """Each direction's depth-integrated intensity ``sum_n lambda_n psi_n^H S
        psi_n``, the wave ``psi_n = Sigma^n e_gzero`` after n depth steps and S the
        ``Sgh`` matrix: what ``sum(L_gh * S_gh)`` is, with ``L_gh = sum_n lambda_n
        conj(psi_g) psi_h``, without forming L one depth at a time (a batched
        outer product per depth was most of the bucket's time). Unnormalized by the
        atom count."""
        phase = 2.0 * np.pi * self.depthstep / (2.0 * self.kn)
        step = torch.complex(
            torch.zeros((), device=self.tdev, dtype=self.rdtype),
            torch.as_tensor(phase, device=self.tdev, dtype=self.rdtype),
        )
        sigma = torch.linalg.matrix_exp(step * a)  # exp(i phase A)
        psi = sigma[:, :, self.gzero]  # psi_1 = Sigma e_gzero
        waves = []
        for _ in range(self.n_depths):
            waves.append(psi)
            psi = torch.matmul(sigma, psi[:, :, None]).squeeze(-1)
        waves = torch.stack(waves, 1)  # (b, depths, beams)
        scattered = torch.matmul(waves, sgh.transpose(1, 2))  # (S psi_n)^T, (b, depths, beams)
        per_depth = (waves.conj() * scattered).sum(-1)  # (b, depths)
        weights = self.lambda_t.to(per_depth.real.dtype)
        return (per_depth * weights).sum(-1).real * (self.depthstep / self.thick)

    # ---- one intensity per incident direction --------------------------- #
    def intensities(self, directions: torch.Tensor) -> torch.Tensor:
        self._pairs()
        k_all = directions.to(self.tdev, self.rdtype)
        total = k_all.shape[0]
        out = torch.empty(total, device=self.tdev, dtype=self.out_dtype)
        for m0 in range(0, total, self.block):
            k = k_all[m0 : m0 + self.block]
            strong, weak, sg = self._classify(k)
            counts = strong.sum(1)
            for s in torch.unique(counts).tolist():
                sub = torch.nonzero(counts == s, as_tuple=False).squeeze(1)
                for b0 in range(0, sub.shape[0], self.matmul_batch):
                    self._bucket(
                        int(s), sub[b0 : b0 + self.matmul_batch], strong, weak, sg, out, m0
                    )
        return out

    def _classify(self, k: torch.Tensor):
        bethe = self.bethe
        d = k.shape[0]
        strong = torch.zeros(d, self.n_beams, dtype=torch.bool, device=self.tdev)
        weak = torch.zeros_like(strong)
        sg_all = torch.empty(d, self.n_beams, device=self.tdev, dtype=self.rdtype)
        for c0 in range(0, d, self.class_chunk):
            kc = k[c0 : c0 + self.class_chunk]
            c = kc.shape[0]
            sg = excitation_errors(self.g_cart, kc, self.lam)
            sg = torch.cat([torch.zeros(c, 1, device=self.tdev, dtype=self.rdtype), sg], dim=1)
            absg = sg.abs()
            cand = (absg / (self.lam * self.u_mod.clamp_min(1e-300))) <= bethe.cutoff
            cand[:, self.forbidden] = absg[:, self.forbidden] <= bethe.sg_double
            cand[:, 0] = True

            # gather the candidate beams to a padded [c, Cmax] index list
            cc = cand.sum(1)
            cmax = int(cc.max().item())
            nz = cand.nonzero(as_tuple=False)  # [T, 2], row-sorted
            rows = nz[:, 0]
            starts = torch.zeros(c, dtype=torch.long, device=self.tdev)
            starts[1:] = cc.cumsum(0)[:-1]
            within = torch.arange(rows.shape[0], device=self.tdev) - starts[rows]
            cidx = torch.zeros(c, cmax, dtype=torch.long, device=self.tdev)
            creal = torch.zeros(c, cmax, dtype=torch.bool, device=self.tdev)
            cidx[rows, within] = nz[:, 1]
            creal[rows, within] = True

            # Bethe min over candidate h only
            absg_c = absg.gather(1, cidx)  # [c, Cmax]
            u = self.u_pair_abs[cidx[:, :, None], cidx[:, None, :]]
            good = (u > _R._FORBIDDEN_TOL) & creal[:, None, :]
            rh = torch.where(
                good,
                (absg_c[:, :, None] / self.lam) / u.clamp_min(1e-300),
                torch.full_like(u, 1.0e4),
            )
            m_c = rh.min(dim=2).values  # [c, Cmax]
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
            strong[c0 : c0 + c] = st
            weak[c0 : c0 + c] = wk
            sg_all[c0 : c0 + c] = sg
        return strong, weak, sg_all

    def _bucket(self, s, sub, strong, weak, sg, out, c0):
        b = sub.shape[0]
        st, wk, sgs = strong[sub], weak[sub], sg[sub]
        s_idx = st.nonzero(as_tuple=False)[:, 1].reshape(b, s)  # ascending -> g0 first
        sg_s = sgs.gather(1, s_idx)  # [b, s]

        wc = wk.sum(1)
        w_max = int(wc.max().item()) if b else 0
        a = self.u_pair[s_idx[:, :, None], s_idx[:, None, :]]
        weak_diag = torch.zeros(b, s, device=self.tdev, dtype=self.rdtype)
        if w_max > 0:
            nz = wk.nonzero(as_tuple=False)  # [T, 2], row-sorted
            rows, cols = nz[:, 0], nz[:, 1]
            starts = torch.zeros(b, dtype=torch.long, device=self.tdev)
            starts[1:] = wc.cumsum(0)[:-1]
            within = torch.arange(rows.shape[0], device=self.tdev) - starts[rows]
            w_idx = torch.zeros(b, w_max, dtype=torch.long, device=self.tdev)
            w_ok = torch.zeros(b, w_max, dtype=torch.bool, device=self.tdev)
            w_idx[rows, within] = cols
            w_ok[rows, within] = True
            sg_w = sgs.gather(1, w_idx)
            inv_sgw = torch.where(w_ok, 1.0 / sg_w, torch.zeros_like(sg_w))
            u_gw = self.u_pair[s_idx[:, :, None], w_idx[:, None, :]]
            u_wh = self.u_pair[w_idx[:, :, None], s_idx[:, None, :]]
            a = a - 0.5 * self.lam * torch.matmul(u_gw * inv_sgw[:, None, :], u_wh)
            weak_diag = 0.5 * self.lam * (u_gw.abs() ** 2 * inv_sgw[:, None, :]).sum(2)

        di = torch.arange(s, device=self.tdev)
        diag = (2.0 * sg_s / self.lam - weak_diag).to(self.cdtype) + 1j * self.upz0
        a[:, di, di] = diag

        sgh = self.sgh_pair[s_idx[:, :, None], s_idx[:, None, :]]  # [g,h]=h-g
        inten = self._depth_intensity(a, sgh) / self.n_atoms
        out[sub + c0] = inten.to(self.out_dtype)
