"""Strain against a reference pattern, by fitting the whole pattern.

:func:`~de_ebsd.index.crosscorr.hrebsd` locates a grid of regions and fits the
shift equation to their centers; here every pixel inside the screen's margin
enters one least-squares fit instead. The model of a test pattern is the
reference re-projected through the gnomonic geometry into the deformed
configuration -- the remap :func:`~de_ebsd.index.crosscorr.remap_reference`
makes -- scaled and offset::

    test(x) = a + b * reference(pixel_ref(F^-1 ray(x)))

with ``F`` the lattice deformation from the reference point to the test point
(eight trace-free numbers: the hydrostatic part is invisible to a projection)
and ``a``, ``b`` the contrast, free because the two points' backgrounds and
doses differ. ``F`` is found by Gauss-Newton -- each iteration samples the
reference and its gradient once at the current warp, solves the 10 x 10 normal
equations, and composes the step onto ``F`` -- on a pyramid, coarse to fine, so
that a seed a degree or two off still converges. This is integrated digital
image correlation (Vermeij and Hoefnagels, Ultramicroscopy 2018), taken against
a real reference rather than a simulation: nothing about how well a master
matches the camera limits it, and it measures only differences from the
reference, so it holds within the reference's grain (:func:`within_grain`).

The seed is the lattice rotation between the two points' indexed orientations
(:func:`seed_from_orientations`); the region cross-correlation is the
cross-check. The strain reported is the left stretch of ``F``'s polar
decomposition, less its trace: the test lattice's deviatoric strain as it
sits in the sample frame, exact at any rotation (``sym(F - I)``, what the
region correlation reports, is off by about half the square of the rotation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from de_ebsd.core.constants import DIVISION_FLOOR
from de_ebsd.core.device import resolve_device
from de_ebsd.core.geometry import detector_rays, pixels_from_rays, sample_to_detector_tensor

__all__ = [
    "REFERENCE_GRAIN_DEG",
    "ReferenceFit",
    "fit_reference",
    "lattice_turns",
    "prepare_patterns",
    "seed_from_orientations",
    "within_grain",
]

#: How far from the reference's orientation a point may be and still have its
#: strain measured against it, in degrees. Seeded from the orientations, a
#: synthetic 2e-3 strain on a 128 screen came back within 2.2e-4 at 5 degrees
#: with the seed 0.3 degrees off; at 8 degrees the fit stopped converging.
REFERENCE_GRAIN_DEG = 5.0

#: The pyramid's binning factors, coarse to fine: the coarse levels pull a seed
#: that is up to a few pixels off at full resolution in, the last one fits.
PYRAMID = (4, 2, 1)

#: Gauss-Newton iterations at each level, at most; a level stops sooner once
#: every pattern's largest step is below :data:`TOLERANCE`.
ITERATIONS = 25

#: The largest step, of the deformation's eight orthonormal numbers, that
#: counts as converged: 1e-7 is a hundredth of the best floor measured.
TOLERANCE = 1e-7

#: The largest step one iteration may take. A step is the solve of a local
#: linear model; on a pattern unlike the reference (a boundary, another grain
#: let through) it can point far outside the elastic range.
MAX_STEP = 2e-2

#: The share of each side left out of the fit at the screen's edges, where a
#: warped reference pulls in pixels it does not have.
MARGIN = 0.05

#: The degree of the polynomial surfaces the offset and the scale are, over
#: the screen: the scale is the ratio of the test point's background to the
#: reference's where its bands came from (:func:`prepare_patterns`). On a
#: synthetic pattern turned 1.5 degrees under a Gaussian background that moved
#: with the beam, degree 2 left 1e-3 of strain, degree 4 4e-6.
CONTRAST_DEGREE = 4

#: Patterns times pixels fitted at once: the memory of one batch is about
#: forty bytes a pattern pixel.
BATCH_PIXELS = 1 << 22


@dataclass(frozen=True)
class ReferenceFit:
    """What :func:`fit_reference` measured, per test pattern."""

    deformation: torch.Tensor
    """``(N, 3, 3)`` ``F`` in the detector frame, determinant one."""

    strain: torch.Tensor
    """``(N, 3, 3)`` deviatoric strain in the sample frame: the left stretch of
    ``F`` less the identity, less its trace."""

    rotation_vector: torch.Tensor
    """``(N, 3)`` lattice rotation from the reference, axis times angle in
    radians, in the sample frame."""

    score: torch.Tensor
    """``(N,)`` the normalized correlation of the test with the fitted model."""

    converged: torch.Tensor
    """``(N,)`` whether the finest level's last step was below its tolerance."""


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _smooth(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """``(N, H, W)`` blurred by a separable Gaussian of *sigma* pixels."""
    if sigma <= 0:
        return images
    kernel = _gaussian_kernel(sigma, images.device, images.dtype)
    pad = len(kernel) // 2
    x = images[:, None]
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="replicate"), kernel.view(1, 1, -1, 1))
    return x[:, 0]


def _binned(frames, shape) -> torch.Tensor:
    x = torch.as_tensor(frames)
    x = x if x.is_floating_point() else x.float()
    x = x[None] if x.ndim == 2 else x
    if shape is not None and tuple(shape) != tuple(x.shape[-2:]):
        factor = x.shape[-2] // int(shape[0])
        if factor * int(shape[0]) != x.shape[-2] or factor * int(shape[1]) != x.shape[-1]:
            raise ValueError(f"{tuple(x.shape[-2:])} does not bin evenly to {tuple(shape)}")
        x = F.avg_pool2d(x[:, None], factor)[:, 0]
    return x


def _monomials(rows: int, cols: int, degree: int, device, dtype) -> torch.Tensor:
    """``(M, rows * cols)``: the monomials of the screen position up to
    *degree*, each running -1 to 1 across the screen."""
    v = (2.0 * torch.arange(rows, device=device, dtype=dtype) + 1.0) / rows - 1.0
    u = (2.0 * torch.arange(cols, device=device, dtype=dtype) + 1.0) / cols - 1.0
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    return torch.stack([uu**i * vv**j for i in range(degree + 1) for j in range(degree + 1 - i)])


def prepare_patterns(frames, shape=None, *, smoothing: float = 0.0) -> torch.Tensor:
    """*frames* ``(H, W)`` or ``(N, H, W)`` ready for :func:`fit_reference`:
    as float, averaged down to *shape* (an integer binning; the frame's own
    when None) and blurred by *smoothing* pixels -- and NOT divided by any
    background.

    The background is the detector's: it stays where it is while the bands
    move, so a test pattern is its reference warped and then multiplied by the
    ratio of the two points' backgrounds, a smooth field the fit's contrast
    surface (:data:`CONTRAST_DEGREE`) takes up exactly. Dividing first by a
    fitted background does not help: a surface fitted to a pattern takes up
    some of its bands' slow variation too, which moves with the lattice, and
    on a synthetic pattern turned 1.5 degrees under a moving background that
    left 1.5e-4 of strain, each pattern divided by its own surface 1.7e-4;
    fitting the raw patterns left 4e-6."""
    return _smooth(_binned(frames, shape), smoothing)


# ---------------------------------------------------------------------------
# the grain and the seed
# ---------------------------------------------------------------------------


def _operators(symmetry) -> np.ndarray:
    return np.asarray(
        symmetry.to_matrix() if hasattr(symmetry, "to_matrix") else symmetry, np.float64
    ).reshape(-1, 3, 3)


def lattice_turns(rotations, reference_rotation, symmetry) -> np.ndarray:
    """The lattice rotation ``(N, 3, 3)`` from the reference point to each test
    point, in the sample frame: ``g_test^T S g_ref`` for the crystal symmetry
    ``S`` that makes it smallest, so a test indexed as another symmetry variant
    of the same orientation still turns by its small physical angle.

    *rotations* ``(N, 3, 3)`` and *reference_rotation* ``(3, 3)`` take sample
    to crystal, as the indexers give them."""
    g = np.asarray(rotations, np.float64).reshape(-1, 3, 3)
    ref = np.asarray(reference_rotation, np.float64)
    operators = _operators(symmetry)
    turns = np.einsum("nji,kjl,lm->nkim", g, operators, ref)
    best = np.trace(turns, axis1=-2, axis2=-1).argmax(1)
    return turns[np.arange(len(g)), best]


def within_grain(
    rotations, reference_rotation, symmetry, max_deg: float = REFERENCE_GRAIN_DEG
) -> np.ndarray:
    """``(N,)`` whether each orientation is within *max_deg* of the reference's,
    over the crystal's symmetry: the points whose strain is measured against
    it. A NaN orientation is outside."""
    turns = lattice_turns(rotations, reference_rotation, symmetry)
    cosine = np.clip((np.trace(turns, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return np.isfinite(angle) & (angle <= max_deg)


def seed_from_orientations(rotations, reference_rotation, symmetry, detector) -> torch.Tensor:
    """The deformation ``(N, 3, 3)`` in the detector frame that the indexed
    orientations alone predict -- the lattice rotation between the points, no
    strain -- to start :func:`fit_reference` from."""
    turns = torch.as_tensor(lattice_turns(rotations, reference_rotation, symmetry))
    om = sample_to_detector_tensor(detector, turns.device, turns.dtype)
    return om @ turns @ om.T


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------


def _basis(device, dtype) -> torch.Tensor:
    """The eight trace-free 3x3 matrices, orthonormal: rotations about x, y, z,
    then the five deviatoric strains."""
    root2, root6 = math.sqrt(2.0), math.sqrt(6.0)
    basis = torch.zeros(8, 3, 3, dtype=torch.float64)
    for k, (i, j) in enumerate(((1, 2), (2, 0), (0, 1))):
        basis[k, i, j], basis[k, j, i] = -1 / root2, 1 / root2
        basis[3 + k, i, j] = basis[3 + k, j, i] = 1 / root2
    basis[6] = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)) / root2
    basis[7] = torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=torch.float64)) / root6
    return basis.to(device, dtype)


#: The cubic B-spline's padding about the image: its 4 x 4 support reaches one
#: pixel before and two after, and the mirrored prefilter settles within it.
_SPLINE_PAD = 16


def _spline_pad(rows: int, cols: int) -> int:
    """:data:`_SPLINE_PAD`, or less on an image too small to mirror that far."""
    return min(_SPLINE_PAD, rows - 1, cols - 1)


def _spline_coefficients(image: torch.Tensor) -> torch.Tensor:
    """The cubic B-spline coefficients ``(H + 2p, W + 2p)`` that interpolate
    *image* ``(H, W)`` exactly at its pixels, *p* :func:`_spline_pad`: the image
    mirrored out by *p*, divided in Fourier space by the spline's sampled
    kernel ``(1, 4, 1) / 6`` along each axis."""
    pad = _spline_pad(*image.shape)
    padded = F.pad(image[None, None], (pad, pad, pad, pad), mode="reflect")[0, 0]
    rows, cols = padded.shape
    spectrum = torch.fft.rfft2(padded.double())
    fr = torch.fft.fftfreq(rows, device=image.device, dtype=torch.float64)
    fc = torch.fft.rfftfreq(cols, device=image.device, dtype=torch.float64)
    kernel_r = (4.0 + 2.0 * torch.cos(2 * math.pi * fr)) / 6.0
    kernel_c = (4.0 + 2.0 * torch.cos(2 * math.pi * fc)) / 6.0
    spectrum = spectrum / (kernel_r[:, None] * kernel_c[None, :])
    return torch.fft.irfft2(spectrum, s=(rows, cols)).to(image.dtype)


def _spline_weights(t: torch.Tensor):
    """The four cubic B-spline weights at fraction *t* and their derivatives."""
    t2, t3 = t * t, t * t * t
    one = 1.0 - t
    weights = (
        one * one * one / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    )
    slopes = (-0.5 * one * one, 1.5 * t2 - 2.0 * t, -1.5 * t2 + t + 0.5, 0.5 * t2)
    return weights, slopes


class _Level:
    """One level of the pyramid: the test pixels inside the margin and their
    rays, and the reference's cubic B-spline, ready to be sampled.

    The reference is interpolated by its cubic B-spline, and its gradient is
    the same spline's: torch's bicubic ``grid_sample`` measured 5e-3 off a
    smooth image at a whole-pixel shift (its bilinear 1e-3), which on a
    synthetic pattern held a 2e-3 strain 2.5e-4 away from the truth."""

    def __init__(self, test, reference, detector, pattern_center, margin, degree):
        rows, cols = test.shape[-2:]
        self.detector = detector
        self.shape = (rows, cols)
        inset_rows, inset_cols = int(round(margin * rows)), int(round(margin * cols))
        r = torch.arange(inset_rows, rows - inset_rows, device=test.device, dtype=test.dtype)
        c = torch.arange(inset_cols, cols - inset_cols, device=test.device, dtype=test.dtype)
        rr, cc = torch.meshgrid(r, c, indexing="ij")
        self.values = test[:, rr.long(), cc.long()].reshape(len(test), -1)  # (N, K)
        self.rays = detector_rays(detector, pattern_center, rr.reshape(-1), cc.reshape(-1))
        #: The contrast's surface, ``(M, K)``: the monomials of the screen
        #: position up to :data:`CONTRAST_DEGREE`, each -1 to 1 across it.
        surface = _monomials(rows, cols, degree, test.device, test.dtype)
        self.surface = surface.reshape(-1, rows, cols)[:, rr.long(), cc.long()].reshape(
            len(surface), -1
        )
        self.coefficients = _spline_coefficients(reference.reshape(rows, cols))
        self.pad = _spline_pad(rows, cols)
        self.width = self.coefficients.shape[1]
        self.flat = self.coefficients.reshape(-1)

    def sample(self, row, col):
        """The reference, its column and its row gradient at fractional pixels
        ``(N, K)``, each ``(N, K)``, and whether each is inside the reference."""
        rows, cols = self.shape
        inside = (row >= 1) & (row <= rows - 2) & (col >= 1) & (col <= cols - 2)
        row = row.clamp(0, rows - 1)
        col = col.clamp(0, cols - 1)
        i, j = torch.floor(row), torch.floor(col)
        wr, dr = _spline_weights(row - i)
        wc, dc = _spline_weights(col - j)
        base = (i.long() - 1 + self.pad) * self.width + (j.long() - 1 + self.pad)
        value = torch.zeros_like(row)
        grad_col = torch.zeros_like(row)
        grad_row = torch.zeros_like(row)
        for a in range(4):
            line = torch.zeros_like(row)
            line_slope = torch.zeros_like(row)
            for b in range(4):
                c = self.flat[base + (a * self.width + b)]
                line = line + wc[b] * c
                line_slope = line_slope + dc[b] * c
            value = value + wr[a] * line
            grad_col = grad_col + wr[a] * line_slope
            grad_row = grad_row + dr[a] * line
        return value, grad_col, grad_row, inside


def _fit_level(level: _Level, f, contrast, reference_pattern_center, iterations, tolerance):
    """Gauss-Newton at one level: *f* ``(N, 3, 3)`` and *contrast* ``(N, 2, M)``
    (the offset's and the scale's coefficients over :attr:`_Level.surface`)
    refined, returned as ``(f, contrast, score, last_step)``."""
    n = len(f)
    dtype, device = f.dtype, f.device
    basis = _basis(device, dtype)
    surface = level.surface  # (M, K)
    m = len(surface)
    nrows = level.detector.nrows
    pcz = reference_pattern_center[:, 2:3]
    t = level.values
    eye = torch.eye(3, device=device, dtype=dtype)
    last = torch.full((n,), torch.inf, device=device, dtype=dtype)
    for _ in range(iterations):
        inverse = torch.linalg.inv(f)
        q = torch.einsum("nij,nkj->nki", inverse, level.rays)
        row, col = pixels_from_rays(level.detector, reference_pattern_center, q)
        value, gx, gy, inside = level.sample(row, col)
        w = inside.to(dtype)
        q0, q1, q2 = q[..., 0], q[..., 1], q[..., 2]
        scale_col = nrows * pcz / (q2 * q2)
        offset = contrast[:, 0] @ surface  # (N, K)
        scale = contrast[:, 1] @ surface
        # F <- (I + d B_k) F turns F^-1 r into F^-1 (I - d B_k) r
        moves = -torch.einsum("nij,kjl->nkil", inverse, basis)  # (N, 8, 3, 3)
        columns = [surface.expand(n, -1, -1), surface[None] * value[:, None]]
        slopes = []
        for k in range(8):
            dq = torch.einsum("nij,nkj->nki", moves[:, k], level.rays)
            dcol = scale_col * (dq[..., 0] * q2 - q0 * dq[..., 2])
            drow = -scale_col * (dq[..., 1] * q2 - q1 * dq[..., 2])
            slopes.append(scale * (gx * dcol + gy * drow))
        columns.append(torch.stack(slopes, 1))
        jacobian = torch.cat(columns, 1) * w[:, None]  # (N, 2M + 8, K)
        residual = (t - offset - scale * value) * w
        normal = (jacobian @ jacobian.transpose(1, 2)).double()
        gradient = (jacobian @ residual[..., None])[..., 0].double()
        damping = 1e-9 * normal.diagonal(dim1=-2, dim2=-1)
        normal = normal + torch.diag_embed(damping)
        step, singular = torch.linalg.solve_ex(normal, gradient)
        usable = (singular == 0)[:, None] & step.isfinite().all(-1, keepdim=True)
        step = torch.where(usable, step, torch.zeros_like(step)).to(dtype)
        move = step[:, 2 * m :]
        reach = move.abs().amax(-1, keepdim=True)
        move = move * (MAX_STEP / reach.clamp_min(MAX_STEP))
        contrast = contrast + step[:, : 2 * m].reshape(n, 2, m)
        f = (eye + torch.einsum("nk,kij->nij", move, basis)) @ f
        last = move.abs().amax(-1)
        if bool((last < tolerance).all()):
            break
    # the score: the correlation of the pattern with the model at the final warp
    q = torch.einsum("nij,nkj->nki", torch.linalg.inv(f), level.rays)
    row, col = pixels_from_rays(level.detector, reference_pattern_center, q)
    value, _, _, inside = level.sample(row, col)
    model = contrast[:, 0] @ surface + (contrast[:, 1] @ surface) * value
    w = inside.to(dtype)
    count = w.sum(-1, keepdim=True).clamp_min(1.0)
    a = (t - (t * w).sum(-1, keepdim=True) / count) * w
    b = (model - (model * w).sum(-1, keepdim=True) / count) * w
    score = (a * b).sum(-1) / ((a * a).sum(-1) * (b * b).sum(-1)).sqrt().clamp_min(DIVISION_FLOOR)
    return f, contrast, score, last


def _polar_strain(f: torch.Tensor, om: torch.Tensor):
    """``F = V R`` in the detector frame -> the deviatoric ``V - I`` and the
    rotation vector of ``R``, both in the sample frame."""
    f64 = f.double()
    u, s, vt = torch.linalg.svd(f64)
    rotation = u @ vt
    stretch = u @ torch.diag_embed(s) @ u.transpose(-1, -2)
    eye = torch.eye(3, device=f.device, dtype=torch.float64)
    strain = stretch - eye
    strain = strain - torch.diagonal(strain, dim1=-2, dim2=-1).sum(-1)[:, None, None] / 3.0 * eye
    om = om.double()
    strain = om.T @ strain @ om
    rotation = om.T @ rotation @ om
    cosine = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.arccos(cosine)
    axis = torch.stack(
        [
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ],
        -1,
    )
    sine = torch.sin(angle)
    # angle / (2 sin angle) -> 1/2 as the angle -> 0
    factor = torch.where(sine > 1e-8, angle / (2.0 * sine), torch.full_like(sine, 0.5))
    return strain, axis * factor[:, None]


def _as_centers(value, default, n, device, dtype) -> torch.Tensor:
    value = default if value is None else value
    centers = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1, 3)
    return centers.expand(n, 3) if centers.shape[0] == 1 else centers


def fit_reference(
    patterns,
    reference,
    detector,
    pattern_center=None,
    reference_pattern_center=None,
    *,
    seed=None,
    pyramid: tuple[int, ...] = PYRAMID,
    iterations: int = ITERATIONS,
    tolerance: float = TOLERANCE,
    margin: float = MARGIN,
    contrast_degree: int = CONTRAST_DEGREE,
    batch_pixels: int = BATCH_PIXELS,
    device=None,
) -> ReferenceFit:
    """The deformation of each of *patterns* relative to *reference*, fitted
    over the whole pattern.

    Parameters
    ----------
    patterns
        ``(H, W)`` or ``(N, H, W)``, prepared (:func:`prepare_patterns`).
    reference
        ``(H, W)``, prepared the same way; ideally summed from several times a
        test point's dose, since its noise is in every fit.
    detector
        :class:`~de_ebsd.sim.detector.EBSDDetector` at the patterns' shape.
    pattern_center, reference_pattern_center
        Where the test points' and the reference's patterns are centered,
        ``(3,)`` or ``(N, 3)``; both default to the detector's. A difference is
        exact in the model, so the centers' change across a scan is not strain.
    seed
        ``(N, 3, 3)`` or ``(3, 3)`` starting ``F`` in the detector frame
        (:func:`seed_from_orientations`); the identity when None.
    pyramid
        Binning factors, coarse to fine; each must divide the shape.
    iterations, tolerance
        Gauss-Newton iterations per level, at most, and the largest step that
        ends a level.
    margin
        The share of each side left out at the screen's edges.
    contrast_degree
        The degree of the offset's and the scale's surfaces over the screen.
    batch_pixels
        Patterns times pixels fitted at once, which bounds memory.
    """
    test = torch.as_tensor(patterns)
    device = resolve_device(device, test.device)
    dtype = torch.float32
    test = test.to(device=device, dtype=dtype)
    test = test[None] if test.ndim == 2 else test
    if tuple(test.shape[-2:]) != (detector.nrows, detector.ncols):
        raise ValueError(
            f"patterns are {tuple(test.shape[-2:])} but the detector is "
            f"{(detector.nrows, detector.ncols)}"
        )
    ref = torch.as_tensor(reference, device=device, dtype=dtype)
    if tuple(ref.shape) != tuple(test.shape[-2:]):
        raise ValueError(f"the reference is {tuple(ref.shape)}, the patterns {test.shape[-2:]}")
    n = len(test)
    centers = _as_centers(pattern_center, detector.pattern_center, n, device, dtype)
    reference_centers = _as_centers(
        reference_pattern_center, detector.pattern_center, n, device, dtype
    )
    if seed is None:
        f = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3).clone()
    else:
        f = torch.as_tensor(seed, device=device, dtype=dtype).reshape(-1, 3, 3)
        f = f.expand(n, 3, 3).clone() if len(f) == 1 else f.clone()

    pixels = detector.nrows * detector.ncols
    per_batch = max(1, batch_pixels // pixels)
    if n > per_batch:
        parts = [
            fit_reference(
                test[lo : lo + per_batch],
                ref,
                detector,
                centers[lo : lo + per_batch],
                reference_centers[lo : lo + per_batch],
                seed=f[lo : lo + per_batch],
                pyramid=pyramid,
                iterations=iterations,
                tolerance=tolerance,
                margin=margin,
                contrast_degree=contrast_degree,
                batch_pixels=batch_pixels,
                device=device,
            )
            for lo in range(0, n, per_batch)
        ]
        return ReferenceFit(
            **{
                name: torch.cat([getattr(part, name) for part in parts])
                for name in ("deformation", "strain", "rotation_vector", "score", "converged")
            }
        )

    from dataclasses import replace

    contrast = None
    score = last = None
    for factor in pyramid:
        if detector.nrows % factor or detector.ncols % factor:
            raise ValueError(f"a pyramid factor of {factor} does not divide {detector.shape}")
        if factor > 1:
            level_test = F.avg_pool2d(test[:, None], factor)[:, 0]
            level_ref = F.avg_pool2d(ref[None, None], factor)[:, 0]
        else:
            level_test, level_ref = test, ref[None]
        level_detector = replace(
            detector, shape=(detector.nrows // factor, detector.ncols // factor)
        )
        level = _Level(level_test, level_ref, level_detector, centers, margin, contrast_degree)
        if contrast is None:
            # no offset, unit scale: the constant term is the first monomial
            contrast = torch.zeros(n, 2, len(level.surface), device=device, dtype=dtype)
            contrast[:, 1, 0] = 1.0
        f, contrast, score, last = _fit_level(
            level, f, contrast, reference_centers, iterations, tolerance
        )
    f = f / torch.linalg.det(f).abs().clamp_min(DIVISION_FLOOR)[:, None, None] ** (1.0 / 3.0)
    om = sample_to_detector_tensor(detector, device, dtype)
    strain, rotation = _polar_strain(f, om)
    return ReferenceFit(
        deformation=f,
        strain=strain.to(dtype),
        rotation_vector=rotation.to(dtype),
        score=score,
        converged=last < tolerance * 10,
    )
