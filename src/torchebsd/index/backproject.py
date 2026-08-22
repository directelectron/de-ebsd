"""Detector -> sphere back-projection (I3).

Spherical indexing matches two functions on the unit sphere: the dynamical
master pattern ``M(n)`` and the experimental pattern ``E(n)`` back-projected off
the detector. This module builds ``E(n)``: it resamples a detector image onto a
set of sphere directions (typically the SHT quadrature grid, see
``torchebsd.index.harmonics``) so it can be spherical-harmonic-transformed (I1)
and cross-correlated against the master over SO(3) (I2). See ``INDEXING_DESIGN.md``
(milestone I3).

**No orientation is applied.** For spherical indexing the pattern is back-projected
to the *scattering sphere in the detector/sample frame* -- each detector pixel
records the intensity scattered into one direction (its "direction cosine"), so a
sphere direction ``n`` in the detector's field of view simply reads the pixel that
looks along ``n``. The crystal orientation is exactly what indexing solves for, so
it is *not* folded in here; back-projecting ``master.project(detector, R)`` recovers
the master rotated by ``R`` over the field of view.

**Geometry.** This is the inverse of ``torchebsd.sim``'s own ``EBSDDetector`` pixel ->
direction map, so torchebsd stays self-contained -- no kikuchipy dependency. The
detector's ``direction_cosines`` builds, for pixel (row, col), the gnomonic point
``(x, y) = (gn_x, gn_y)``, forms the detector-frame ray ``(gn_x, gn_y, 1) * pcz``,
and rotates it into the sample frame with the orthonormal sample->detector matrix
``Om`` (rows = detector basis ``X_d, Y_d, Z_d`` in sample coordinates); the unit vector
of that ray is the direction cosine. We invert it:

1. project a unit sphere direction ``n`` onto the detector basis,
   ``(d_x, d_y, d_z) = (n . X_d, n . Y_d, n . Z_d) = n @ Om.T`` (``Om`` orthonormal,
   so this is the inverse rotation); ``d_z > 0`` means ``n`` points toward the
   detector (in front of it);
2. gnomonic projection back to the detector plane, ``gn_x = d_x / d_z``,
   ``gn_y = d_y / d_z`` (independent of the pattern center -- the ray's direction
   alone fixes it);
3. the affine gnomonic -> fractional-pixel map, inverting the node placement in
   ``direction_cosines``: ``col = (gn_x - x_min) / x_scale - 0.5`` and
   ``row = (y_max - gn_y) / y_scale - 0.5`` (row 0 = top, matching the detector),
   where the gnomonic bounds ``[x_min, x_max, y_min, y_max]`` and the scales come
   from the pattern center ``pc = (pcx, pcy, pcz)`` exactly as in
   ``EBSDDetector.gnomonic_bounds``.

The pattern center enters only in step 3 (rebuilt here from ``pc`` as torch ops), so
the whole map is autograd-differentiable in the pattern center -- this is what makes
joint pattern-center refinement (I6) possible. The intensity read-off is a bilinear
``grid_sample`` (``align_corners=True``), differentiable in the pattern. Torch-native
(CPU/CUDA) and batched over an arbitrary stack of patterns.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchebsd.sim.detector import EBSDDetector, _sample_to_detector_matrix

__all__ = ["directions_to_pixels", "pattern_to_sphere"]


def _sample_to_detector_tensor(
    detector: EBSDDetector, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """The orthonormal sample->detector matrix ``Om`` as a torch tensor.

    Rows are the detector basis ``(X_d, Y_d, Z_d)`` expressed in sample coordinates;
    this is exactly ``EBSDDetector``'s ``_sample_to_detector_matrix`` (a function of
    the tilt geometry only, not the pattern center), so the frame convention matches
    the forward projection bit for bit.
    """
    om = _sample_to_detector_matrix(
        detector.sample_tilt, detector.tilt, detector.azimuthal, detector.twist
    )
    return torch.as_tensor(om, device=device, dtype=dtype)


def _pc_components(
    detector: EBSDDetector,
    pc: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three pattern-center components ``(pcx, pcy, pcz)`` as torch scalars.

    If ``pc`` is given (a length-3 tensor) it overrides ``detector.pc`` and its
    autograd graph is preserved, so gradients flow back into the pattern center.
    Otherwise the detector's fixed (non-differentiable) ``pc`` is used.
    """
    if pc is None:
        pc = torch.as_tensor(detector.pc, device=device, dtype=dtype)
    else:
        if pc.shape[-1] != 3:
            raise ValueError(f"pc must have last dim 3, got shape {tuple(pc.shape)}")
        pc = pc.to(device=device, dtype=dtype)
    return pc[0], pc[1], pc[2]


def directions_to_pixels(
    directions: torch.Tensor,
    detector: EBSDDetector,
    pc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map sphere directions to fractional detector pixels (inverse of the DC map).

    Parameters
    ----------
    directions
        Unit vectors ``(M, 3)`` in the sample frame (need not be exactly normalized;
        the gnomonic ratios are scale-invariant).
    detector
        The ``torchebsd.sim`` :class:`~torchebsd.sim.detector.EBSDDetector` geometry.
    pc
        Optional pattern-center override ``(3,)`` (``pcx, pcy, pcz``). When supplied
        with ``requires_grad``, the returned pixel coordinates are differentiable in
        the pattern center. Defaults to ``detector.pc``.

    Returns
    -------
    col, row
        Fractional pixel coordinates ``(M,)``, where integer values land on pixel
        centers (``col = 0`` is the leftmost column, ``row = 0`` the top row) --
        ready for :func:`torch.nn.functional.grid_sample` with ``align_corners=True``.
    in_fov
        Boolean field-of-view mask ``(M,)``: ``True`` where the direction points at
        the detector (``d_z > 0``) and lands within the gnomonic bounds.
    """
    device = directions.device
    om = _sample_to_detector_tensor(detector, device, directions.dtype)
    dtype = directions.dtype
    pcx, pcy, pcz = _pc_components(detector, pc, device, dtype)

    # detector-frame components d = (n . X_d, n . Y_d, n . Z_d) = directions @ Om.T
    comp = directions @ om.transpose(-2, -1)
    d_x, d_y, d_z = comp[..., 0], comp[..., 1], comp[..., 2]
    # gnomonic projection back onto the detector plane (pattern-center independent)
    d_z_safe = torch.where(d_z.abs() < 1e-12, torch.full_like(d_z, 1e-12), d_z)
    gn_x = d_x / d_z_safe
    gn_y = d_y / d_z_safe

    # gnomonic bounds and node scales from the pattern center (torch, differentiable),
    # matching EBSDDetector.gnomonic_bounds / direction_cosines exactly
    aspect = detector.ncols / detector.nrows
    x_min = -aspect * pcx / pcz
    x_max = aspect * (1.0 - pcx) / pcz
    y_min = -(1.0 - pcy) / pcz
    y_max = pcy / pcz
    x_scale = (x_max - x_min) / detector.ncols
    y_scale = (y_max - y_min) / detector.nrows

    col = (gn_x - x_min) / x_scale - 0.5
    row = (y_max - gn_y) / y_scale - 0.5

    in_fov = (
        (d_z > 0) & (gn_x >= x_min) & (gn_x <= x_max) & (gn_y >= y_min) & (gn_y <= y_max)
    )
    return col, row, in_fov


def pattern_to_sphere(
    patterns: torch.Tensor,
    detector: EBSDDetector,
    directions: torch.Tensor,
    pc: torch.Tensor | None = None,
    padding_mode: str = "border",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project detector ``patterns`` onto ``directions`` on the unit sphere.

    Each sphere direction reads the detector pixel that scatters along it (no crystal
    orientation is applied -- see the module docstring), bilinearly interpolating the
    pattern. The result is ``E(n)`` sampled on ``directions``, ready to feed
    :meth:`torchebsd.index.harmonics.SphericalHarmonicTransform.forward` (reshape the
    ``(N_theta, N_phi)`` grid directions to ``(M, 3)`` and the samples back).

    Parameters
    ----------
    patterns
        Detector images ``(..., H, W)`` (``H = detector.nrows``, ``W = detector.ncols``),
        batched over arbitrary leading dimensions. Real floating dtype.
    detector
        The ``torchebsd.sim`` :class:`~torchebsd.sim.detector.EBSDDetector` geometry
        (fixed pattern center). Its pixel -> direction map is inverted here.
    directions
        Target sphere directions ``(M, 3)`` in the sample/detector frame, e.g.
        ``SphericalHarmonicTransform.grid_directions().reshape(-1, 3)``.
    pc
        Optional pattern-center override ``(3,)`` (``pcx, pcy, pcz``); pass a tensor
        with ``requires_grad`` to differentiate the samples in the pattern center
        (I6 refinement). Defaults to ``detector.pc``.
    padding_mode
        ``grid_sample`` padding for directions that fall (just) outside the pixel
        grid; ``"border"`` (default) clamps to the edge. Off-FOV samples are flagged
        by the returned mask regardless.

    Returns
    -------
    samples
        Sphere samples ``(..., M)`` (same leading dims and dtype as ``patterns``).
        Entries outside the field of view are the ``padding_mode`` read-off and
        should be excluded via ``mask``. Differentiable in ``patterns`` (and, via
        ``pc``, in the pattern center).
    mask
        Field-of-view weight ``(M,)`` in ``patterns``' dtype: ``1`` where the
        direction lands on the detector, ``0`` otherwise (the detector's gnomonic
        bounds). Shared across the batch since the geometry is.
    """
    if patterns.shape[-2:] != (detector.nrows, detector.ncols):
        raise ValueError(
            f"patterns trailing shape {tuple(patterns.shape[-2:])} != detector "
            f"{(detector.nrows, detector.ncols)}"
        )
    if directions.shape[-1] != 3 or directions.ndim != 2:
        raise ValueError(f"directions must have shape (M, 3), got {tuple(directions.shape)}")

    device = patterns.device
    dirs = directions.to(device=device, dtype=patterns.dtype)
    col, row, in_fov = directions_to_pixels(dirs, detector, pc)

    # normalized sampling coordinates for grid_sample(align_corners=True):
    # col in [0, W-1] -> gx in [-1, 1]; row in [0, H-1] -> gy in [-1, 1]
    gx = 2.0 * col / (detector.ncols - 1) - 1.0
    gy = 2.0 * row / (detector.nrows - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (M, 2)

    m = dirs.shape[0]
    leading = patterns.shape[:-2]
    batch = patterns.reshape(-1, 1, detector.nrows, detector.ncols)
    n = batch.shape[0]
    grid = grid.view(1, 1, m, 2).expand(n, 1, m, 2)
    sampled = F.grid_sample(
        batch, grid, mode="bilinear", align_corners=True, padding_mode=padding_mode
    )  # (N, 1, 1, M)
    samples = sampled.reshape(*leading, m)
    mask = in_fov.to(patterns.dtype)
    return samples, mask
