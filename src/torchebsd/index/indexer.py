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
   (:func:`torchebsd.index.correlation.so3_cross_correlation`), **locally normalize
   the correlation over the field of view** (the default; ``normalize=False`` for the
   raw baseline), take the peak, and read the orientation off the SOFT grid with
   :func:`torchebsd.index.wigner.grid_to_rotation`.

**Convention (pinned).** ``grid_to_rotation(argmax)`` is *directly* the sample
orientation that was passed to :meth:`torchebsd.sim.MasterPattern.project` -- no
inverse, no ``~R``, no transpose, no ``(alpha, gamma)`` swap. The correlation peak
sits at ``R^-1`` at the active-matrix level, but the orix ZYZ->Bunge bridge in
:func:`torchebsd.index.wigner.euler_zyz_to_orix` folds in a transpose that exactly
cancels it, so the returned orix ``Rotation`` is ``R`` itself. The round-trip
acceptance test (``tests/index/test_indexer.py``) pins this end to end.

**Field-of-view windowing.** Each back-projected pattern is standardized to
zero-mean, unit-variance *over the field of view* (FOV) and zeroed outside it (call
the result ``E``). Zeroing the FOV mean makes ``E``'s ``l = 0`` (DC) harmonic vanish,
which kills the DC-alignment bias and makes the master's own DC irrelevant to the
masked correlation (``b_00 = 0`` => the ``l = 0`` cross term drops).

**Locally-normalized (masked) cross-correlation -- the default.** The masked
cross-correlation ``X(g) = int_W E(n) M(g^-1 n) dn`` that I4 computes is *unnormalized*:
it carries the ``g``-dependent master energy over the FOV, so it can be outscored by a
spurious orientation that merely places a bright master region inside the FOV (I4
measured, on a bright-spot master, a true-cell score below a far-off spurious peak).
The robustness fix I4 deferred -- a ``g``-dependent, EMSphInx-style normalization -- is
now applied **by default** (``normalize=True``). The score becomes the spherical
zero-mean masked normalized cross-correlation

    gamma(g) = X(g) / [ sqrt(P) * sqrt(S2(g) - S1(g)^2 / A_W) ],

a correlation-coefficient-like number (roughly in ``[-1, 1]``). Here ``W(n)`` is the
detector's fixed FOV indicator (``1`` inside, ``0`` outside), ``M_g(n) = M(g^-1 n)``,
and

    X(g)  = int E M_g dn   = so3_cross_correlation(master=a_lm,  pattern=b_lm)
    S1(g) = int W M_g  dn   = so3_cross_correlation(master=a_lm,  pattern=w_lm)
    S2(g) = int W M_g^2 dn  = so3_cross_correlation(master=a2_lm, pattern=w_lm)

with ``a2_lm`` the spectrum of ``M^2`` and ``w_lm`` that of ``W``. The denominator
``sqrt(S2 - S1^2 / A_W)`` is the local norm of the master inside the *rotated* FOV
(``A_W = int W dn`` its spherical area), which divides out exactly the ``g``-dependent
master energy that biased ``X``. ``A_W`` and ``P = int W E^2 dn`` (the pattern's FOV
energy) are ``g``-independent scalars, so they set only the quality scale, not the
argmax; both are the spherical quadrature integral of a field, ``sqrt(4 pi) Re(f_00)``.

The three terms are the **same** :func:`so3_cross_correlation` evaluated three ways on
one SO(3) grid (same ``(alpha, beta, gamma)`` axis order), so they are aligned
elementwise and ``gamma`` is a straightforward elementwise combination followed by the
*identical* argmax + :func:`grid_to_rotation` chain -- normalization does **not** touch
the rotation convention. ``w_lm``, ``S1``, ``S2``, ``A_W`` and the whole denominator
field depend only on the (fixed) detector geometry and the master, so they are computed
**once** and cached on the indexer, shared across every pattern in a batch; the only
per-pattern extra cost is forming ``gamma`` from ``X`` and ``P``.

``normalize=False`` keeps the honest **unnormalized** baseline ``X(g)``: sufficient for
near-uniform-norm masters (dense Kikuchi bands) but biased for masters with localized
bright regions -- it is retained so the bias, and that normalization removes it, stay
testable.

See ``INDEXING_DESIGN.md`` (milestone I4). Scan -> ``CrystalMap`` assembly (I5),
gradient refinement (I6), and visualization (I7) are separate milestones.
"""

from __future__ import annotations

import math

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
    a2_lm : torch.Tensor
        The precomputed spectrum of the *squared* master ``M^2``, complex, shape
        ``((L+1)**2,)`` -- the ``S2`` term of the masked normalization (see the module
        docstring). Precomputed here because it depends only on the master.

    Notes
    -----
    The geometry-derived masked-normalization terms (the FOV mask spectrum ``w_lm``,
    the FOV area ``A_W`` and the denominator field ``sqrt(S2 - S1^2 / A_W)``) are
    computed once on the first ``normalize=True`` call and cached, since they depend
    only on the master and the *fixed* detector pattern center -- they are shared
    across every pattern in a scan. If the pattern center were later made variable (I6
    refinement), this cache would have to be invalidated.
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
        grid_shape = (self.sht.n_theta, self.sht.n_phi)
        self.a_lm = self.sht.forward(m_samples.reshape(grid_shape))  # ((L+1)**2,) complex
        # spectrum of M^2 for the S2 term of the masked normalization; master-only, so
        # precomputed here alongside a_lm.
        self.a2_lm = self.sht.forward((m_samples**2).reshape(grid_shape))

        # flat quadrature weights (M,), used for the g-independent FOV integrals A_W, P
        self._weights_flat = self.sht.weights.reshape(-1)
        # geometry-derived normalizer, cached lazily on the first normalize=True call
        # (depends only on the master and the fixed detector pc -- see class Notes).
        self._w_lm = None  # FOV mask spectrum
        self._a_w = None  # FOV spherical area A_W
        self._denom_geo = None  # sqrt(clamp(S2 - S1^2 / A_W)), shape (N, N, N)

    def _geometry_normalizer(self):
        """Compute (once) and return the cached masked-NCC denominator field.

        Returns the ``g``-dependent local-norm field ``sqrt(S2(g) - S1(g)^2 / A_W)``
        (clamped), shape ``(N, N, N)`` with ``N = 2L + 2``, on the same SO(3) grid as
        the per-pattern cross term ``X``. It reuses :func:`so3_cross_correlation` twice
        -- ``S1`` against the FOV mask spectrum ``w_lm`` and ``S2`` against ``a2_lm``.
        Everything here depends only on the master and the fixed detector pattern
        center, so it is computed once and cached (``_w_lm``, ``_a_w``, ``_denom_geo``)
        and shared across the whole batch. (If ``pc`` were made variable for I6
        refinement, this cache would need invalidating.)
        """
        if self._denom_geo is not None:
            return self._denom_geo
        # The FOV mask W is pattern-independent (fixed by detector geometry), so recover
        # it by back-projecting a zero pattern and keeping only the mask output.
        zero = torch.zeros(
            (1, self.detector.nrows, self.detector.ncols),
            device=self.device,
            dtype=self.dtype,
        )
        _, w = pattern_to_sphere(zero, self.detector, self._dirs)  # (M,), 1 in FOV else 0
        w_lm = self.sht.forward(w.reshape(self.sht.n_theta, self.sht.n_phi))
        # S1, S2: the same so3_cross_correlation as X, against w_lm (master mean and
        # second moment over the rotated FOV). Aligned elementwise with X.
        s1 = so3_cross_correlation(self.a_lm, w_lm, self.bandwidth)  # (N, N, N)
        s2 = so3_cross_correlation(self.a2_lm, w_lm, self.bandwidth)  # (N, N, N)
        # A_W = int W dn = sqrt(4 pi) Re(w_00): the FOV spherical area (a scalar).
        a_w = math.sqrt(4.0 * math.pi) * w_lm[0].real
        denom = (s2 - s1**2 / a_w.clamp_min(self._eps)).clamp_min(self._eps).sqrt()
        self._w_lm = w_lm
        self._a_w = a_w
        self._denom_geo = denom
        return denom

    def index(self, patterns, normalize=True):
        """Index one pattern or a stack, returning orix orientations and quality.

        Parameters
        ----------
        patterns
            A single detector image ``(H, W)`` or a stack ``(Np, H, W)``, as a numpy
            array or a torch tensor (``H = detector.nrows``, ``W = detector.ncols``).
            Plain arrays are accepted directly, so no kikuchipy import is needed --
            a kikuchipy signal's ``.data`` can be passed straight in.
        normalize
            When ``True`` (default) the score is the **locally-normalized (masked)
            cross-correlation** ``gamma(g)`` (see the module docstring): the raw masked
            cross term ``X(g)`` divided by the master's local FOV norm, which removes
            the bias toward orientations that place bright master regions in the field
            of view. When ``False`` the score is the raw **unnormalized** masked
            cross-correlation ``X(g)`` -- the honest I4 baseline, unbiased only for
            near-uniform-norm masters. Both share the identical argmax + rotation
            convention chain; only the per-pattern scaling of the surface differs.

        Returns
        -------
        rotations : orix.quaternion.Rotation
            The best-matching sample orientation of each pattern, length ``Np``.
            ``grid_to_rotation(argmax)`` is returned *as is*: it is the orientation
            that was passed to :meth:`torchebsd.sim.MasterPattern.project` (no
            inverse/transpose -- see the module docstring).
        quality : torch.Tensor
            The peak score of each pattern, a length-``Np`` real tensor. For
            ``normalize=True`` this is the peak of ``gamma`` -- a
            correlation-coefficient-like number, roughly in ``[-1, 1]``, comparable
            across patterns and FOVs. For ``normalize=False`` it is the raw peak of
            ``X`` (arbitrary scale). Higher means a better match; use it as a
            per-pattern confidence metric.

        Notes
        -----
        The scan-indexing milestone (I5) will wrap this into an orix
        :class:`~orix.crystal_map.CrystalMap` (one orientation + quality per scan
        point); here the raw ``Rotation`` and quality are returned so the
        single-pattern path stays dependency-light.
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

        # cross-correlate over SO(3): the masked cross term X[np, alpha, beta, gamma].
        # This is exactly so3_cross_correlation's X(g); the S1/S2 terms of the
        # normalization reuse the same call against w_lm / a2_lm.
        c = so3_cross_correlation(self.a_lm, b_lm, self.bandwidth)  # (Np, N, N, N)
        n_grid = 2 * self.bandwidth + 2

        if normalize:
            # locally-normalized (masked) correlation gamma = X / (sqrt(P) * denom_geo).
            # denom_geo = sqrt(S2 - S1^2 / A_W) is geometry+master only (cached, shared
            # across the batch); P = int W E^2 dn is the per-pattern FOV energy, the
            # quadrature integral sqrt(4 pi) Re((E^2)_00) = sum_grid weights * E^2. Both
            # P and A_W are g-independent, so they scale the surface but not the argmax.
            denom_geo = self._geometry_normalizer()  # (N, N, N)
            p_energy = (e**2 * self._weights_flat).sum(-1)  # (Np,)
            scale = p_energy.clamp_min(self._eps).sqrt().reshape(-1, 1, 1, 1)
            surface = c / (scale * denom_geo[None])  # (Np, N, N, N)
        else:
            surface = c

        # per-pattern peak over the three orientation axes
        flat = surface.reshape(surface.shape[0], -1)
        quality, peak_flat = flat.max(dim=-1)  # (Np,), (Np,)
        j, k, p = (idx.cpu() for idx in torch.unravel_index(peak_flat, (n_grid, n_grid, n_grid)))

        # read the orientation directly off the SOFT grid (the sample orientation R)
        rotations = grid_to_rotation(self.bandwidth, j, k, p)
        return rotations, quality.detach()

    def refine(
        self,
        patterns,
        rotations=None,
        pc=None,
        refine_pc=False,
        n_iters=100,
        lr=0.05,
        pc_lr=None,
        return_history=False,
    ):
        """Gradient-refine indexed orientations (and optionally the pattern center).

        Convenience wrapper around :func:`torchebsd.index.refine.refine_orientation` that
        supplies this indexer's own master, detector, device and -- crucially -- its SHT
        quadrature grid directions (``self._dirs``), so the refinement scores over the
        *same* grid the coarse index used. If ``rotations`` is ``None`` the seeds are
        obtained by calling :meth:`index` first, so a single call goes straight from raw
        patterns to sharpened orientations (I6).

        Parameters
        ----------
        patterns
            A single detector image ``(H, W)`` or a stack ``(Np, H, W)`` (numpy or torch).
        rotations
            Seed orientations, an :class:`orix.quaternion.Rotation` of length ``Np``. If
            ``None`` (default), :meth:`index` is run on ``patterns`` to obtain them.
        pc, refine_pc, n_iters, lr, pc_lr, return_history
            Passed through to :func:`~torchebsd.index.refine.refine_orientation` (see its
            docstring). ``pc`` defaults to the indexer's ``detector.pc``.

        Returns
        -------
        tuple
            ``(rotation, score)`` -- the refined orix ``Rotation`` and final NCC of each
            pattern; ``pc`` appended when ``refine_pc``, ``history`` appended when
            ``return_history`` (same order as
            :func:`~torchebsd.index.refine.refine_orientation`).
        """
        from torchebsd.index.refine import refine_orientation

        if rotations is None:
            rotations, _ = self.index(patterns)
        return refine_orientation(
            patterns,
            self.master,
            self.detector,
            rotations,
            pc=pc,
            refine_pc=refine_pc,
            n_iters=n_iters,
            lr=lr,
            pc_lr=pc_lr,
            directions=self._dirs,
            return_history=return_history,
        )

    def index_scan(
        self,
        patterns,
        shape=None,
        step=1.0,
        phase=None,
        batch_size=None,
        normalize=True,
        refine=False,
        refine_iters=100,
        refine_lr=0.05,
    ):
        """Index a whole scan and assemble the result into an orix ``CrystalMap``.

        Wraps :meth:`index` over an entire EBSD scan: it processes the patterns in
        mini-batches (to bound memory -- see below), concatenates the per-batch
        orientations and quality, and hands them to
        :func:`torchebsd.index.io.to_crystal_map`, which lays them out on the scan
        grid as a single-phase :class:`~orix.crystal_map.CrystalMap`.

        Parameters
        ----------
        patterns
            The scan patterns, as a numpy array or torch tensor, in either layout:

            * ``(rows, cols, H, W)`` -- the scan grid is read off directly and
              ``shape`` is inferred (any explicit ``shape`` must then match);
            * ``(N, H, W)`` -- a flat, row-major stack, in which case ``shape``
              **must** be given so the ``N = rows * cols`` points can be reshaped
              onto the grid.

            Plain arrays are accepted (no kikuchipy needed -- a kikuchipy signal's
            ``.data`` reshaped to the scan passes straight in). ``H = detector.nrows``,
            ``W = detector.ncols``.
        shape
            ``(rows, cols)`` of the scan. Required for ``(N, H, W)`` input; optional
            (and cross-checked) for ``(rows, cols, H, W)`` input.
        step
            Scan step size (same for both axes), written onto the map's ``x``/``y``.
        phase
            The map's crystal symmetry, as an :class:`orix.crystal_map.Phase` or a
            torchebsd :class:`~torchebsd.sim.Crystal` (converted via
            :func:`torchebsd.index.io.phase_from_crystal`). If ``None``, it is
            derived from the master's metadata (``master.metadata["crystal"]`` if a
            :class:`~torchebsd.sim.Crystal` is carried there); if neither is
            available a clear error tells the caller to pass ``phase``.
        batch_size
            Number of patterns indexed per mini-batch (default ``64``). The SO(3)
            correlation surface is ``(batch_size, N, N, N)`` with ``N = 2L + 2``,
            which at production bandwidth dominates memory, so this bounds the peak
            working set at the cost of a Python-level loop over the scan. The result
            is **identical** to a single batch (the argmax per pattern is
            independent); it only trades memory for iterations.
        normalize
            Passed through to :meth:`index` (default ``True``, the locally-normalized
            masked cross-correlation).
        refine
            If ``True``, each mini-batch's coarse orientations are gradient-refined
            (I6) with :meth:`refine` before assembly, and ``prop["quality"]`` becomes
            the refined NCC score. The pattern center is *not* refined here (a scan
            shares one detector geometry; joint pattern-center fitting over a whole scan
            is left for later) -- only the orientations are sharpened. Default ``False``.
        refine_iters, refine_lr
            Adam iterations and rotation learning rate for the per-batch refinement when
            ``refine`` is ``True`` (see
            :func:`~torchebsd.index.refine.refine_orientation`).

        Returns
        -------
        orix.crystal_map.CrystalMap
            One orientation + ``prop["quality"]`` per scan point, single phase
            (``phase_id`` all zero), on the ``(rows, cols)`` grid. With ``refine=True``
            the orientations are gradient-sharpened and the quality is the refined NCC.
        """
        import numpy as np
        from orix.quaternion import Rotation

        from torchebsd.index.io import phase_from_crystal, to_crystal_map

        patt = torch.as_tensor(patterns, device=self.device, dtype=self.dtype)
        if patt.ndim == 4:  # (rows, cols, H, W)
            rows, cols = int(patt.shape[0]), int(patt.shape[1])
            if shape is not None and (int(shape[0]), int(shape[1])) != (rows, cols):
                raise ValueError(
                    f"shape {tuple(shape)} does not match the (rows, cols, H, W) "
                    f"pattern grid {(rows, cols)}"
                )
            patt = patt.reshape(rows * cols, patt.shape[2], patt.shape[3])
        elif patt.ndim == 3:  # (N, H, W) -- shape required
            if shape is None:
                raise ValueError(
                    "shape=(rows, cols) is required for (N, H, W) patterns "
                    "(pass (rows, cols, H, W) to infer it instead)"
                )
            rows, cols = int(shape[0]), int(shape[1])
            if rows * cols != patt.shape[0]:
                raise ValueError(
                    f"shape {(rows, cols)} implies {rows * cols} patterns but got "
                    f"{patt.shape[0]}"
                )
        else:
            raise ValueError(
                "patterns must be (rows, cols, H, W) or (N, H, W), got shape "
                f"{tuple(patt.shape)}"
            )

        n = rows * cols
        bs = int(batch_size) if batch_size is not None else 64
        if bs < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # index in mini-batches to bound the (bs, N, N, N) SO(3) surface memory;
        # concatenate the per-batch orix Rotations and quality back to length n.
        rot_chunks = []
        qual_chunks = []
        for start in range(0, n, bs):
            chunk = patt[start : start + bs]
            rot_c, qual_c = self.index(chunk, normalize=normalize)
            if refine:
                # sharpen the coarse seeds; the refined NCC replaces quality
                rot_c, qual_c = self.refine(
                    chunk, rotations=rot_c, n_iters=refine_iters, lr=refine_lr
                )
            rot_chunks.append(np.asarray(rot_c.data))
            qual_chunks.append(qual_c)
        rotations = Rotation(np.concatenate(rot_chunks, axis=0))
        quality = torch.cat(qual_chunks)

        # resolve the phase: explicit arg (Phase or Crystal) wins; else derive from
        # the master's metadata; else a clear error.
        if phase is None:
            crystal = None
            meta = getattr(self.master, "metadata", None)
            if isinstance(meta, dict):
                crystal = meta.get("crystal")
            if crystal is None:
                raise ValueError(
                    "no phase given and master.metadata has no 'crystal' to derive "
                    "one from; pass phase=<orix Phase or torchebsd Crystal> to "
                    "index_scan (the Phase gives the CrystalMap its crystal symmetry)"
                )
            phase = phase_from_crystal(crystal)

        return to_crystal_map(rotations, quality, (rows, cols), phase, step=step)
