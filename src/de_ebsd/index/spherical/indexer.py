"""End-to-end spherical indexer.

Back-projects each pattern to the sphere, expands it in spherical harmonics,
cross-correlates against the precomputed master spectrum over all of SO(3), and
reads the orientation off the peak. ``normalize=True`` (the default) divides out
the rotation-dependent master energy inside the field of view, which keeps a
bright master region from outscoring the true orientation.

``grid_to_rotation(argmax)`` is directly the sample orientation passed to
``MasterPattern.project`` -- no inverse or transpose.
"""

from __future__ import annotations

import math

import torch

from de_ebsd.index.backproject import pattern_to_sphere
from de_ebsd.index.spherical.correlation import so3_cross_correlation
from de_ebsd.index.spherical.harmonics import SphericalHarmonicTransform
from de_ebsd.index.spherical.wigner import grid_to_rotation, so3_grid, wigner_d


class SphericalIndexer:
    """Index EBSD patterns against a master pattern by spherical cross-correlation.

    Build once (the master spectrum is precomputed here), then call :meth:`index`
    on a single pattern or a stack.

    Parameters
    ----------
    master
        A :class:`de_ebsd.sim.MasterPattern`, sampled via ``master.sample``.
    detector
        :class:`~de_ebsd.sim.detector.EBSDDetector` geometry, inverted to
        back-project each pattern to the sphere.
    bandwidth
        Band limit ``L``, the accuracy/speed knob: the SOFT grid step is
        ``360 / (2L + 2)`` degrees. Cost grows ~``O(L^4)`` in both time and memory,
        since :func:`~de_ebsd.index.spherical.wigner.wigner_d` materializes a
        ``(N, L+1, 2L+1, 2L+1)`` tensor.
    device
        Resolved as explicit argument > ``master.device`` > CPU.
    energy_kev
        Energy bin to sample the master at; ``None`` (default) selects the highest.

    Attributes
    ----------
    sht : SphericalHarmonicTransform
        The shared transform plan; master and patterns both go through it.
    a_lm, a2_lm : torch.Tensor
        Spectra of the master and of ``M^2``, complex, shape ``((L+1)**2,)``.
    """

    def __init__(self, master, detector, bandwidth=53, device=None, energy_kev=None):
        self.master = master
        self.detector = detector
        self.bandwidth = int(bandwidth)
        self.energy_kev = energy_kev

        if device is not None:
            self.device = torch.device(device)
        else:
            master_device = getattr(master, "device", None)
            self.device = torch.device(master_device) if master_device is not None else (
                torch.device("cpu")
            )

        # pin the plan to the master's real dtype so master and patterns transform
        # identically; the plan requires a real float type
        master_dtype = getattr(getattr(master, "data", None), "dtype", torch.float64)
        self.dtype = master_dtype if master_dtype in (torch.float32, torch.float64) else (
            torch.float64
        )
        self._eps = torch.finfo(self.dtype).eps

        # the plan does NOT auto-move from its CPU/float64 default, so pin it here
        self.sht = SphericalHarmonicTransform(
            self.bandwidth, device=self.device, dtype=self.dtype
        )

        # quadrature directions M = (L+1)*(2L+2)
        self._dirs = self.sht.grid_directions().reshape(-1, 3)

        m_samples = master.sample(self._dirs, energy_kev=self.energy_kev)
        m_samples = m_samples.to(device=self.device, dtype=self.dtype)
        grid_shape = (self.sht.n_theta, self.sht.n_phi)
        self.a_lm = self.sht.forward(m_samples.reshape(grid_shape))
        self.a2_lm = self.sht.forward((m_samples**2).reshape(grid_shape))

        self._weights_flat = self.sht.weights.reshape(-1)

        # cache the O(L^4) Wigner-d block: so3_cross_correlation otherwise rebuilds it
        # per call, so a mini-batched scan would pay for it per chunk
        _, beta_grid, _ = so3_grid(self.bandwidth, device=self.device, dtype=self.dtype)
        self._wig_d = wigner_d(self.bandwidth, beta_grid)

        # normalizer terms, cached on the first normalize=True call; they depend only
        # on the master and the fixed detector pattern_center, so a variable pattern_center must invalidate them
        self._w_lm = None
        self._a_w = None
        self._denom_geo = None

    def _geometry_normalizer(self):
        if self._denom_geo is not None:
            return self._denom_geo
        # the FOV mask is fixed by detector geometry: recover it by back-projecting a
        # zero pattern and keeping only the mask output
        zero = torch.zeros(
            (1, self.detector.nrows, self.detector.ncols),
            device=self.device,
            dtype=self.dtype,
        )
        _, w = pattern_to_sphere(zero, self.detector, self._dirs)
        w_lm = self.sht.forward(w.reshape(self.sht.n_theta, self.sht.n_phi))
        # master mean and second moment over the rotated FOV, aligned elementwise with X
        s1 = so3_cross_correlation(
            self.a_lm, w_lm, self.bandwidth, wigner_d_cached=self._wig_d
        )
        s2 = so3_cross_correlation(
            self.a2_lm, w_lm, self.bandwidth, wigner_d_cached=self._wig_d
        )
        a_w = math.sqrt(4.0 * math.pi) * w_lm[0].real  # FOV spherical area
        denom = (s2 - s1**2 / a_w.clamp_min(self._eps)).clamp_min(self._eps).sqrt()
        self._w_lm = w_lm
        self._a_w = a_w
        self._denom_geo = denom
        return denom

    def index(self, patterns, normalize=True, return_surface=False):
        """Index one pattern or a stack, returning orix orientations and quality.

        Parameters
        ----------
        patterns
            A detector image ``(H, W)`` or stack ``(Np, H, W)``, numpy or torch.
        normalize
            Divide out the master's local FOV norm (default), removing the bias
            toward orientations that place bright master regions in the field of
            view. ``False`` keeps the raw masked cross-correlation, unbiased only
            for near-uniform-norm masters.
        return_surface
            Also return the SO(3) score surface that was arg-maxed, as a third
            element. It feeds :func:`de_ebsd.viz.maps.show_correlation` directly.

        Returns
        -------
        rotations : orix.quaternion.Rotation
            Best-matching sample orientation per pattern, length ``Np``.
        quality : torch.Tensor
            Peak score per pattern. Normalized, this is roughly in ``[-1, 1]`` and
            comparable across patterns; unnormalized it has arbitrary scale.
        surface : torch.Tensor
            Only when ``return_surface``: shape ``(Np, N, N, N)``, ``N = 2L + 2``,
            on the ``(alpha, beta, gamma)`` grid. Detached.
        """
        patt = torch.as_tensor(patterns, device=self.device, dtype=self.dtype)
        single = patt.ndim == 2
        if single:
            patt = patt[None]
        if patt.ndim != 3:
            raise ValueError(
                f"patterns must be (H, W) or (Np, H, W), got shape {tuple(patt.shape)}"
            )

        samples, mask = pattern_to_sphere(patt, self.detector, self._dirs)

        # standardize over the FOV and zero outside it. Zeroing the FOV mean makes E's
        # l=0 harmonic vanish, which drops the DC term from the masked correlation.
        w = mask
        denom = w.sum().clamp_min(self._eps)
        mean = (samples * w).sum(-1, keepdim=True) / denom
        var = ((samples - mean) ** 2 * w).sum(-1, keepdim=True) / denom
        e = ((samples - mean) / var.clamp_min(self._eps).sqrt()) * w
        b_lm = self.sht.forward(
            e.reshape(*e.shape[:-1], self.sht.n_theta, self.sht.n_phi)
        )

        c = so3_cross_correlation(
            self.a_lm, b_lm, self.bandwidth, wigner_d_cached=self._wig_d
        )
        n_grid = 2 * self.bandwidth + 2

        if normalize:
            # rho = X / (sqrt(P) * denom_geo). P and A_W are g-independent, so they
            # scale the surface but not the argmax.
            denom_geo = self._geometry_normalizer()
            p_energy = (e**2 * self._weights_flat).sum(-1)
            scale = p_energy.clamp_min(self._eps).sqrt().reshape(-1, 1, 1, 1)
            surface = c / (scale * denom_geo[None])
        else:
            surface = c

        flat = surface.reshape(surface.shape[0], -1)
        quality, peak_flat = flat.max(dim=-1)
        j, k, p = (idx.cpu() for idx in torch.unravel_index(peak_flat, (n_grid, n_grid, n_grid)))

        rotations = grid_to_rotation(self.bandwidth, j, k, p)
        if return_surface:
            return rotations, quality.detach(), surface.detach()
        return rotations, quality.detach()

    def refine(
        self,
        patterns,
        rotations=None,
        pattern_center=None,
        refine_pattern_center=False,
        n_iters=100,
        learning_rate=0.02,
        pattern_center_learning_rate=None,
        keep_best=True,
        return_history=False,
    ):
        """Gradient-refine indexed orientations (and optionally the pattern center).

        Wraps :func:`de_ebsd.index.refine.orientation.refine_orientation` with this
        indexer's master, detector, device and SHT quadrature directions, so
        refinement scores over the same grid the coarse index used. With
        ``rotations=None`` the seeds come from :meth:`index`, taking raw patterns
        straight to sharpened orientations.

        ``pattern_center`` defaults to ``detector.pattern_center``; the remaining arguments pass through.
        Returns ``(rotation, score)``, with ``pattern_center`` appended when ``refine_pattern_center`` and
        ``history`` when ``return_history``.
        """
        from de_ebsd.index.refine.orientation import refine_orientation

        if rotations is None:
            rotations, _ = self.index(patterns)
        return refine_orientation(
            patterns,
            self.master,
            self.detector,
            rotations,
            pattern_center=pattern_center,
            refine_pattern_center=refine_pattern_center,
            n_iters=n_iters,
            learning_rate=learning_rate,
            pattern_center_learning_rate=pattern_center_learning_rate,
            directions=self._dirs,
            device=self.device,
            keep_best=keep_best,
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
        refine_lr=0.02,
    ):
        """Index a whole scan and assemble it into an orix ``CrystalMap``.

        Parameters
        ----------
        patterns
            Either ``(rows, cols, H, W)``, from which the scan grid is inferred, or
            a flat row-major ``(N, H, W)``, which requires ``shape``.
        shape
            ``(rows, cols)`` of the scan. Cross-checked against 4-D input.
        step
            Scan step size, written onto the map's ``x``/``y``.
        phase
            An orix :class:`~orix.crystal_map.Phase` or a de_ebsd
            :class:`~de_ebsd.crystal.Crystal`. If ``None``, derived from
            ``master.metadata["crystal"]``.
        batch_size
            Patterns per mini-batch (default 64). The SO(3) surface is
            ``(batch_size, N, N, N)``, which dominates memory at production
            bandwidth. Results are identical to a single batch.
        refine
            Gradient-refine each mini-batch before assembly, making
            ``prop["quality"]`` the refined NCC. Orientation only -- a scan shares
            one detector geometry, and joint pattern-center fitting over a whole
            scan is not done here.
        refine_iters, refine_lr
            Adam iterations and rotation learning rate used when ``refine``.

        Returns
        -------
        orix.crystal_map.CrystalMap
            One orientation + ``prop["quality"]`` per scan point, single phase.
        """
        import numpy as np
        from orix.quaternion import Rotation

        from de_ebsd.index.io import phase_from_crystal, to_crystal_map

        # shape inference and reshaping stay on the stack's own device (view-only);
        # only each mini-batch moves to self.device, so a 10^6-pattern scan never
        # lands wholesale in GPU memory
        patt = torch.as_tensor(patterns, dtype=self.dtype)
        if patt.ndim == 4:
            rows, cols = int(patt.shape[0]), int(patt.shape[1])
            if shape is not None and (int(shape[0]), int(shape[1])) != (rows, cols):
                raise ValueError(
                    f"shape {tuple(shape)} does not match the (rows, cols, H, W) "
                    f"pattern grid {(rows, cols)}"
                )
            patt = patt.reshape(rows * cols, patt.shape[2], patt.shape[3])
        elif patt.ndim == 3:
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

        rot_chunks = []
        qual_chunks = []
        for start in range(0, n, bs):
            chunk = patt[start : start + bs].to(self.device)
            rot_c, qual_c = self.index(chunk, normalize=normalize)
            if refine:
                rot_c, qual_c = self.refine(
                    chunk, rotations=rot_c, n_iters=refine_iters, learning_rate=refine_lr
                )
            rot_chunks.append(np.asarray(rot_c.data))
            qual_chunks.append(qual_c)
        rotations = Rotation(np.concatenate(rot_chunks, axis=0))
        quality = torch.cat(qual_chunks)

        if phase is None:
            crystal = None
            meta = getattr(self.master, "metadata", None)
            if isinstance(meta, dict):
                crystal = meta.get("crystal")
            if crystal is None:
                raise ValueError(
                    "no phase given and master.metadata has no 'crystal' to derive "
                    "one from; pass phase=<orix Phase or de_ebsd Crystal> to "
                    "index_scan (the Phase gives the CrystalMap its crystal symmetry)"
                )
            phase = phase_from_crystal(crystal)

        return to_crystal_map(rotations, quality, (rows, cols), phase, step=step)
