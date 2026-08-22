"""End-to-end spherical indexer (I4).

``SphericalIndexer`` ties the four indexing primitives together into a working
single-pattern (and batched-orientation) indexer that recovers a crystal
orientation from a detector pattern:

1. build **one** shared spherical-harmonic-transform plan
   (:class:`torchebsd.index.harmonics.SphericalHarmonicTransform`) so the master
   and every pattern are expanded on the *same* Gauss-Legendre x equiangular grid
   at the same device/dtype;
2. precompute the master spectrum ``a_lm`` once in :meth:`__init__` by sampling
   the continuous master on the plan's quadrature directions and transforming;
3. per pattern, back-project it to the sphere
   (:func:`torchebsd.index.backproject.pattern_to_sphere`), window it over the
   detector field of view, transform to ``b_lm``;
4. cross-correlate the two spectra over all of SO(3)
   (:func:`torchebsd.index.correlation.so3_cross_correlation`), take the peak, and
   read the orientation off the SOFT grid with
   :func:`torchebsd.index.wigner.grid_to_rotation`.

**Convention (pinned).** ``grid_to_rotation(argmax)`` is *directly* the sample
orientation that was passed to :meth:`torchebsd.sim.MasterPattern.project` -- no
inverse, no ``~R``, no transpose, no ``(alpha, gamma)`` swap. The correlation peak
sits at ``R^-1`` at the active-matrix level, but the orix ZYZ->Bunge bridge in
:func:`torchebsd.index.wigner.euler_zyz_to_orix` folds in a transpose that exactly
cancels it, so the returned orix ``Rotation`` is ``R`` itself. The round-trip
acceptance test (``tests/index/test_indexer.py``) pins this end to end.

**Field-of-view windowing (settled policy).** Each back-projected pattern is
standardized to zero-mean, unit-variance *over the field of view* and zeroed
outside it. Zeroing the FOV mean makes the pattern's ``l = 0`` (DC) harmonic
vanish, which kills the DC-alignment bias and makes the master's own DC irrelevant
to the masked correlation. The correlation itself is the plain (unnormalized)
masked cross-correlation; full masked *normalized* cross-correlation (a
``g``-dependent, EMSphInx-style normalization) is a deferred I5 enhancement. The
unnormalized form is sufficient to recover orientation but can bias toward
orientations that place bright master regions inside the field of view.

See ``INDEXING_DESIGN.md`` (milestone I4). Scan -> ``CrystalMap`` assembly (I5),
gradient refinement (I6), and visualization (I7) are separate milestones.
"""

from __future__ import annotations

import torch

from torchebsd.index.backproject import pattern_to_sphere
from torchebsd.index.correlation import so3_cross_correlation
from torchebsd.index.harmonics import SphericalHarmonicTransform
from torchebsd.index.wigner import grid_to_rotation


class SphericalIndexer:
    """Index EBSD patterns against a master pattern by spherical cross-correlation.

    Build once (the master spectrum is precomputed here), then call :meth:`index`
    on a single pattern or a stack. Everything is torch-native and batched, so a
    whole stack of orientations is indexed in one shot.

    Parameters
    ----------
    master
        A :class:`torchebsd.sim.MasterPattern` (or one imported from EMsoft) -- the
        continuous "dictionary" ``M(n)`` on the sphere, sampled via ``master.sample``.
    detector
        :class:`torchebsd.sim.detector.EBSDDetector` geometry; its pixel -> direction
        map is inverted to back-project each pattern to the sphere.
    bandwidth
        Spherical-harmonic band limit ``L`` (the accuracy/speed knob). The SOFT grid
        step is ``360 / (2L + 2)`` degrees, so ``L`` sets the achievable orientation
        resolution before refinement. The default (``53``) is a sane
        accuracy/cost balance for EBSD; note that
        :func:`torchebsd.index.wigner.wigner_d` materializes a
        ``(N, L+1, 2L+1, 2L+1)`` tensor whose memory grows ~``O(L^4)``, so large
        ``L`` is costly in both time and memory.
    device
        Torch device for the transforms (``"cuda"`` recommended for scans). Resolved
        as: explicit ``device`` argument > ``master.device`` > CPU. The shared SHT
        plan is pinned to this device and to the master's real dtype (the plan
        otherwise defaults to CPU/float64 and does not auto-move to the sample
        device).

    Attributes
    ----------
    bandwidth : int
        The band limit ``L``.
    device : torch.device
        Device the transforms run on.
    dtype : torch.dtype
        Real floating dtype of the transforms (inherited from ``master.data``).
    sht : SphericalHarmonicTransform
        The single shared transform plan (master and patterns both go through it).
    a_lm : torch.Tensor
        The precomputed master spectrum, complex, shape ``((L+1)**2,)``.
    """

    def __init__(self, master, detector, bandwidth=53, device=None):
        self.master = master
        self.detector = detector
        self.bandwidth = int(bandwidth)

        # device: explicit arg > master.device > cpu
        if device is not None:
            self.device = torch.device(device)
        else:
            master_device = getattr(master, "device", None)
            self.device = torch.device(master_device) if master_device is not None else (
                torch.device("cpu")
            )

        # dtype coherence: pin the plan to the master's real dtype so master and
        # patterns are transformed identically. The harmonics plan requires a real
        # float type; fall back to float64 if the master is stored as something else.
        master_dtype = getattr(getattr(master, "data", None), "dtype", torch.float64)
        self.dtype = master_dtype if master_dtype in (torch.float32, torch.float64) else (
            torch.float64
        )
        self._eps = torch.finfo(self.dtype).eps

        # one shared SHT plan, explicitly pinned to this device/dtype (it does NOT
        # auto-move from its CPU/float64 default onto the sample device)
        self.sht = SphericalHarmonicTransform(
            self.bandwidth, device=self.device, dtype=self.dtype
        )

        # quadrature directions M = (L+1)*(2L+2): N_theta = L+1, N_phi = 2L+2
        self._dirs = self.sht.grid_directions().reshape(-1, 3)  # (M, 3) on self.device

        # master spectrum, precomputed once: sample the continuous master on the
        # plan's directions and forward-transform. master.sample runs on the master's
        # own device; move the samples onto the plan's device/dtype before forward.
        m_samples = master.sample(self._dirs)  # (M,), on master.device
        m_samples = m_samples.to(device=self.device, dtype=self.dtype)
        self.a_lm = self.sht.forward(
            m_samples.reshape(self.sht.n_theta, self.sht.n_phi)
        )  # ((L+1)**2,) complex

    def index(self, patterns):
        """Index one pattern or a stack, returning orix orientations and quality.

        Parameters
        ----------
        patterns
            A single detector image ``(H, W)`` or a stack ``(Np, H, W)``, as a numpy
            array or a torch tensor (``H = detector.nrows``, ``W = detector.ncols``).
            Plain arrays are accepted directly, so no kikuchipy import is needed --
            a kikuchipy signal's ``.data`` can be passed straight in.

        Returns
        -------
        rotations : orix.quaternion.Rotation
            The best-matching sample orientation of each pattern, length ``Np``.
            ``grid_to_rotation(argmax)`` is returned *as is*: it is the orientation
            that was passed to :meth:`torchebsd.sim.MasterPattern.project` (no
            inverse/transpose -- see the module docstring).
        quality : torch.Tensor
            The peak correlation value of each pattern, a length-``Np`` real tensor.
            Higher means a better match; use it as a per-pattern confidence metric.

        Notes
        -----
        I5 will wrap this into an orix :class:`~orix.crystal_map.CrystalMap` (one
        orientation + quality per scan point); here the raw ``Rotation`` and quality
        are returned so the single-pattern path stays dependency-light.
        """
        patt = torch.as_tensor(patterns, device=self.device, dtype=self.dtype)
        single = patt.ndim == 2
        if single:
            patt = patt[None]  # (1, H, W)
        if patt.ndim != 3:
            raise ValueError(
                f"patterns must be (H, W) or (Np, H, W), got shape {tuple(patt.shape)}"
            )

        # back-project the stack to the sphere on the shared quadrature grid
        samples, mask = pattern_to_sphere(patt, self.detector, self._dirs)  # (Np, M), (M,)

        # window over the field of view: standardize to zero-mean, unit-variance over
        # the FOV and zero outside it. Zeroing the FOV mean makes E's l=0 (DC) harmonic
        # vanish, which removes the DC-alignment bias and makes the master's own DC
        # irrelevant to the masked correlation (b_00 = 0 => the (l=0) cross term drops).
        w = mask  # (M,), 1 inside FOV else 0
        denom = w.sum().clamp_min(self._eps)
        mean = (samples * w).sum(-1, keepdim=True) / denom
        var = ((samples - mean) ** 2 * w).sum(-1, keepdim=True) / denom
        e = ((samples - mean) / var.clamp_min(self._eps).sqrt()) * w  # (Np, M), 0 outside FOV
        b_lm = self.sht.forward(
            e.reshape(*e.shape[:-1], self.sht.n_theta, self.sht.n_phi)
        )  # (Np, (L+1)**2)

        # cross-correlate over SO(3): C[np, alpha, beta, gamma], N = 2L+2
        c = so3_cross_correlation(self.a_lm, b_lm, self.bandwidth)  # (Np, N, N, N)
        n_grid = 2 * self.bandwidth + 2

        # per-pattern peak over the three orientation axes
        flat = c.reshape(c.shape[0], -1)
        quality, peak_flat = flat.max(dim=-1)  # (Np,), (Np,)
        j, k, p = (idx.cpu() for idx in torch.unravel_index(peak_flat, (n_grid, n_grid, n_grid)))

        # read the orientation directly off the SOFT grid (the sample orientation R)
        rotations = grid_to_rotation(self.bandwidth, j, k, p)
        return rotations, quality.detach()
