"""HR-EBSD: deviatoric strain and lattice rotation by pattern cross-correlation.

Where :func:`~de_ebsd.index.refine.highres.refine_hr` matches a pattern against a
dynamical master, HR-EBSD is purely differential: it cross-correlates regions of a
test pattern against a *reference* pattern of the same phase and fits the
displacement gradient tensor that maps one onto the other. Nothing about the master
pattern enters, so the model-fidelity ceiling that caps absolute pattern-center
accuracy near 1e-3 does not apply here.

The shift equation
------------------
A lattice displacement gradient ``beta = F - I`` sends a crystal direction ``r`` to
``(I + beta) r``. In gnomonic screen coordinates ``(x, y, z)`` with ``z = pcz``, a
feature therefore moves by, to first order,

    ``qx = b11 x + b12 y + b13 z - (x/z)(b31 x + b32 y + b33 z)``
    ``qy = b21 x + b22 y + b23 z - (y/z)(b31 x + b32 y + b33 z)``

Two equations per region of interest, linear in ``beta``. Adding ``c I`` to ``beta``
leaves both *exactly* unchanged -- the hydrostatic component is invisible to a
projection -- so only the 8 trace-free components are measurable and :func:`hrebsd`
solves under ``trace(beta) = 0``. Recovering the hydrostatic part needs an elastic
boundary condition (usually ``sigma_33 = 0`` at the free surface) and the stiffness
tensor; that is left to the caller.

Reference remapping
-------------------
The first-order shift equation degrades once the lattice rotation is large, and a
pattern-center error coupled to a lattice rotation ``w`` leaks in as a phantom strain
of order ``(dpc / pcz) * w``. So :func:`hrebsd` defaults to two passes: measure
``F``, re-project the reference *through the gnomonic geometry* into the deformed
configuration (:func:`remap_reference`), then re-correlate and compose. The residual
rotation entering the second pass is only as large as the first pass's error, which
relaxes the pattern-center requirement by one to two orders of magnitude. The remap
is exact rather than first-order, and it absorbs any pattern-center difference
between the reference and test points as well.

How much of this is verified
----------------------------
The geometry: the design matrix reproduces the exact projective displacement of a
region to ~1e-4 px, and the solver inverts it to ~1e-3 of ``beta``.

The locator, on real 2048px direct-detection patterns (64 regions on a uniform grid,
5% independent noise added, true shifts from 0.04 to 13 px, known exactly by Fourier
shift): **median error 0.002-0.003 px, worst 0.006-0.009 px** at 128-256 px regions.
Plain correlation with one refinement iteration measured best; whitening was neither
needed nor helpful. Feeding that precision through the shift equation puts ``beta``
around 1e-5 with ~64 regions, and the error falls as ``1 / sqrt(M)``.

After Wilkinson, Meaden & Dingley (2006); remapping after Britton & Wilkinson (2012).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from de_ebsd.core.constants import DIVISION_FLOOR
from de_ebsd.core.device import resolve_device
from de_ebsd.core.geometry import detector_rays, sample_to_detector_tensor

__all__ = ["HREBSDResult", "hrebsd", "remap_reference", "roi_centers"]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _pixels_from_ray(detector, pattern_center: torch.Tensor, ray: torch.Tensor):
    """The fractional ``(row, col)`` a ray lands on: the inverse of ``detector_rays``.

    ``col + 0.5 = ncols * pcx + nrows * pcz * (r1 / r3)`` and
    ``row + 0.5 = nrows * pcy - nrows * pcz * (r2 / r3)``. Both gnomonic axes scale
    by ``nrows``, so a pattern-center change is a pure scale-and-shift of the pattern
    in pixel coordinates -- the magnification-about-the-pattern-center that the
    moving-screen method exploits.
    """
    pcx, pcy, pcz = pattern_center[..., 0:1], pattern_center[..., 1:2], pattern_center[..., 2:3]
    inv_z = 1.0 / ray[..., 2]
    col = detector.ncols * pcx + detector.nrows * pcz * (ray[..., 0] * inv_z) - 0.5
    row = detector.nrows * pcy - detector.nrows * pcz * (ray[..., 1] * inv_z) - 0.5
    return row, col


def roi_centers(detector, grid: int = 8, roi_size: int = 128, margin: float = 0.05):
    """Region-of-interest centers on a ``grid x grid`` layout, as ``(M, 2)`` row/col.

    Regions are inset far enough that a full ``roi_size`` box fits, plus ``margin``
    (a fraction of each side) to stay off the detector edge, where the pattern is
    weakest and the gnomonic lever arm is most sensitive to pattern-center error.
    """
    half = roi_size // 2
    axes = []
    for length in (detector.nrows, detector.ncols):
        lo = half + margin * length
        hi = length - half - margin * length
        if hi <= lo:
            raise ValueError(
                f"roi_size={roi_size} with margin={margin} leaves no room on a "
                f"{detector.nrows}x{detector.ncols} detector"
            )
        axes.append(np.linspace(lo, hi, grid))
    rr, cc = np.meshgrid(axes[0], axes[1], indexing="ij")
    return np.stack([rr.reshape(-1), cc.reshape(-1)], axis=-1)


# ---------------------------------------------------------------------------
# sub-pixel cross-correlation
# ---------------------------------------------------------------------------


def _roi_pixels(centers: torch.Tensor, roi_size: int):
    """``(M, 2)`` centers -> ``(M, roi, roi)`` row and col indices, integer-aligned."""
    half = roi_size // 2
    offs = torch.arange(roi_size, device=centers.device, dtype=centers.dtype) - half
    rows = centers[:, 0, None] + offs[None, :]  # (M, roi)
    cols = centers[:, 1, None] + offs[None, :]
    return rows[:, :, None].expand(-1, -1, roi_size), cols[:, None, :].expand(-1, roi_size, -1)


def _extract(patterns: torch.Tensor, centers: torch.Tensor, roi_size: int):
    """``(N, H, W)`` and ``(M, 2)`` -> ``(N, M, roi, roi)`` boxes at integer pixels."""
    rows, cols = _roi_pixels(centers, roi_size)
    r = rows[:, :, 0].long().clamp(0, patterns.shape[-2] - 1)  # (M, roi)
    c = cols[:, 0, :].long().clamp(0, patterns.shape[-1] - 1)
    return patterns[:, r[:, :, None], c[:, None, :]]


def _window(roi_size: int, device, dtype) -> torch.Tensor:
    w = torch.hann_window(roi_size, periodic=False, device=device, dtype=dtype)
    return w[:, None] * w[None, :]


def _prepare(rois: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Mean-subtract, window, then unit-normalize, so peaks are comparable."""
    x = rois - rois.mean(dim=(-2, -1), keepdim=True)
    x = x * win
    return x / x.flatten(-2).norm(dim=-1)[..., None, None].clamp_min(DIVISION_FLOOR)


def _spectral_mask(roi_size: int, bandpass, device, dtype):
    """Gaussian band-pass over the sampling frequency, or ``None`` for flat.

    ``bandpass`` is ``(low, high)`` in cycles per pixel; Nyquist is 0.5. The
    high-frequency roll-off matters because whitening (see :func:`_measure_shifts`)
    gives every frequency equal weight, including the ones carrying only noise.
    """
    if bandpass is None:
        return None
    low, high = bandpass
    f = torch.fft.fftfreq(roi_size, device=device, dtype=dtype)
    r = (f[:, None] ** 2 + f[None, :] ** 2).sqrt()
    mask = torch.ones_like(r)
    if low and low > 0:
        mask = mask * torch.exp(-((low / r.clamp_min(DIVISION_FLOOR)) ** 2))
    if high and high > 0:
        mask = mask * torch.exp(-((r / high) ** 2))
    return mask


def _fourier_shift(rois: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Shift each region by a sub-pixel ``(drow, dcol)``, band-limited and exact."""
    r = rois.shape[-1]
    f = torch.fft.fftfreq(r, device=rois.device, dtype=rois.dtype)
    phase = torch.exp(
        -2j
        * torch.pi
        * (shift[..., 0, None, None] * f[:, None] + shift[..., 1, None, None] * f[None, :])
    )
    return torch.fft.ifft2(torch.fft.fft2(rois) * phase).real


def _upsampled_peak(cps: torch.Tensor, upsample: int, radius: float = 1.0):
    """Sub-pixel cross-correlation peak by local upsampled DFT.

    ``cps`` is the cross-power spectrum ``(B, r, r)``. Takes the integer peak of the
    inverse transform, then evaluates the inverse DFT on a ``1 / upsample`` grid
    within ``+-radius`` pixels of it by matrix multiply (after Guizar-Sicairos,
    Thurman & Fienup, 2008), which costs a small matmul rather than a padded FFT.

    The integer peak is within half a pixel of the true one, so a radius of one
    pixel is enough. The grid's cost goes as the radius squared: at 2.5 pixels and
    an upsample of 50 it was most of :func:`hrebsd`'s time and 9-15 GB of memory
    on 256 patterns of 36-64 regions.

    Returns ``(shift, peak)``: ``(B, 2)`` sub-pixel ``(drow, dcol)`` and ``(B,)``
    correlation peak height.
    """
    b, m, n = cps.shape
    real = cps.real.dtype
    cc = torch.fft.ifft2(cps).real
    flat = cc.flatten(-2).argmax(-1)
    peak = cc.flatten(-2).gather(-1, flat[:, None])[:, 0]
    py, px = torch.div(flat, n, rounding_mode="floor"), flat % n
    # a peak past the midpoint wraps to a negative shift
    iy = torch.where(py > m // 2, py - m, py).to(real)
    ix = torch.where(px > n // 2, px - n, px).to(real)

    n_up = int(np.ceil(radius * upsample)) * 2 + 1
    step = (torch.arange(n_up, device=cps.device, dtype=real) - n_up // 2) / upsample
    dy = iy[:, None] + step[None, :]  # (B, n_up)
    dx = ix[:, None] + step[None, :]
    fy = torch.fft.fftfreq(m, device=cps.device, dtype=real)
    fx = torch.fft.fftfreq(n, device=cps.device, dtype=real)
    two_pi_i = 2j * torch.pi
    ky = torch.exp(two_pi_i * dy[:, :, None] * fy[None, None, :])  # (B, n_up, m)
    kx = torch.exp(two_pi_i * dx[:, :, None] * fx[None, None, :])  # (B, n_up, n)
    fine = torch.einsum("bpu,buv,bqv->bpq", ky, cps, kx).real / (m * n)

    flat = fine.flatten(-2).argmax(-1)
    sy = dy.gather(-1, torch.div(flat, n_up, rounding_mode="floor")[:, None])[:, 0]
    sx = dx.gather(-1, (flat % n_up)[:, None])[:, 0]
    return torch.stack([sy, sx], dim=-1), peak


def _measure_shifts(test_rois, ref_rois, win, upsample, mask=None, refine=True, whiten=0.0):
    """``(N, M, r, r)`` pairs -> ``(N, M, 2)`` shifts in pixels and ``(N, M)`` NCC.

    ``refine`` shifts the reference by the first estimate and re-measures the
    residual. That second look is what buys the precision: measured on real 2048px
    direct-detection patterns it roughly halves the error, to a median of 0.002-0.003
    px. It is skipped for any region whose shift exceeds an eighth of the region --
    the shift is circular, and a large one would wrap real content into the
    comparison.

    ``whiten`` divides the cross-power spectrum by ``|cps| ** whiten``, 1.0 being full
    phase correlation. It defends against a specific failure: a taper is unavoidable
    (regions are not periodic), and the window's own autocorrelation peaks at zero lag
    and drags a plain correlation peak toward it, in proportion to how broad the
    pattern's autocorrelation is. On a pathologically smooth pattern that bias reaches
    whole pixels. Real Kikuchi patterns are nowhere near that regime -- so the default
    is 0, plain correlation, which measured *better* than whitening on real data and
    does not amplify noise-dominated frequencies the way whitening does.
    """
    n, m, r, _ = test_rois.shape
    raw_ref = ref_rois.reshape(n * m, r, r)
    ft = torch.fft.fft2(_prepare(test_rois, win).reshape(n * m, r, r))

    def measure(ref):
        cps = ft * torch.fft.fft2(_prepare(ref, win)).conj()
        if whiten:
            cps = cps / cps.abs().clamp_min(DIVISION_FLOOR) ** whiten
        if mask is not None:
            cps = cps * mask
        return _upsampled_peak(cps, upsample)

    shift, _ = measure(raw_ref)
    if refine:
        moved = _fourier_shift(raw_ref, shift)
        residual, _ = measure(moved)
        ok = shift.norm(dim=-1) < r / 8.0
        shift = shift + torch.where(ok[:, None], residual, torch.zeros_like(residual))

    # quality is the plain normalized cross-correlation of the aligned regions,
    # which stays interpretable in [-1, 1] even though the locator is whitened
    aligned = _prepare(_fourier_shift(raw_ref, shift), win)
    ncc = (_prepare(test_rois, win).reshape(n * m, r, r) * aligned).flatten(-2).sum(-1)
    return shift.reshape(n, m, 2), ncc.reshape(n, m)


# ---------------------------------------------------------------------------
# shifts -> displacement gradient
# ---------------------------------------------------------------------------


def _design(detector, pattern_center: torch.Tensor, centers: torch.Tensor):
    """Design matrix ``(N, M, 2, 8)`` for the trace-free shift equation.

    Free parameters are ``[b11, b12, b13, b21, b22, b23, b31, b32]`` with
    ``b33 = -(b11 + b22)``; rows are the gnomonic ``(qx, qy)`` of each region.
    """
    ray = detector_rays(detector, pattern_center, centers[:, 0], centers[:, 1])  # (N, M, 3)
    x, y, z = ray[..., 0], ray[..., 1], ray[..., 2]
    o = torch.zeros_like(x)
    a9 = torch.stack(
        [
            torch.stack([x, y, z, o, o, o, -x * x / z, -x * y / z, -x], dim=-1),
            torch.stack([o, o, o, x, y, z, -y * x / z, -y * y / z, -y], dim=-1),
        ],
        dim=-2,
    )  # (N, M, 2, 9)
    a = a9[..., :8].clone()
    a[..., 0] -= a9[..., 8]  # b33 = -(b11 + b22) folds into the b11 ...
    a[..., 4] -= a9[..., 8]  # ... and b22 columns
    return a


def _solve_beta(a: torch.Tensor, q: torch.Tensor, w: torch.Tensor):
    """Weighted least squares ``(N, M, 2, 8)`` x ``(N, M, 2)`` -> ``(N, 3, 3)``, rms."""
    n = q.shape[0]
    a_flat = a.reshape(n, -1, 8)
    q_flat = q.reshape(n, -1, 1)
    ww = w.repeat_interleave(2, dim=-1).clamp_min(0.0)  # (N, 2M)
    sw = ww[..., None].sqrt()
    sol = torch.linalg.lstsq(a_flat * sw, q_flat * sw).solution[..., 0]  # (N, 8)
    resid = (a_flat @ sol[..., None] - q_flat)[..., 0]
    rms = ((resid * resid * ww).sum(-1) / ww.sum(-1).clamp_min(DIVISION_FLOOR)).sqrt()

    b33 = -(sol[:, 0] + sol[:, 4])
    beta = torch.stack(
        [
            torch.stack([sol[:, 0], sol[:, 1], sol[:, 2]], dim=-1),
            torch.stack([sol[:, 3], sol[:, 4], sol[:, 5]], dim=-1),
            torch.stack([sol[:, 6], sol[:, 7], b33], dim=-1),
        ],
        dim=1,
    )
    return beta, rms


# ---------------------------------------------------------------------------
# remapping
# ---------------------------------------------------------------------------


def _reference_grid(detector, f, pattern_center, reference_pattern_center, rows, cols):
    """Test-frame fractional pixels ``(N, K)`` -> ``(N, K, 2)`` grid into the reference.

    A feature at crystal direction ``r`` in the reference sits at ``F r`` once the
    lattice is deformed, so the remapped image at ray ``r`` carries the reference's
    content at ``F^-1 r``. Leaving through the test pattern center and re-entering
    through the reference's also absorbs their pattern-center difference, exactly.
    """
    ray = detector_rays(detector, pattern_center, rows, cols)  # (N, K, 3)
    ray_ref = torch.einsum("nij,nkj->nki", torch.linalg.inv(f), ray)
    row_ref, col_ref = _pixels_from_ray(detector, reference_pattern_center, ray_ref)
    gx = 2.0 * (col_ref + 0.5) / detector.ncols - 1.0
    gy = 2.0 * (row_ref + 0.5) / detector.nrows - 1.0
    return torch.stack([gx, gy], dim=-1)


def _sample(ref: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.grid_sample(
        ref[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[:, 0]


def _warped_rois(ref, detector, f, pattern_center, reference_pattern_center, centers, roi_size):
    """Only the regions of the remapped reference, never the whole warped frame.

    On a 4k detector a full remap is ~17x the pixels this actually needs.
    """
    n = f.shape[0]
    rows, cols = _roi_pixels(centers, roi_size)  # (M, roi, roi)
    rr = rows.reshape(1, -1).expand(n, -1)
    cc = cols.reshape(1, -1).expand(n, -1)
    grid = _reference_grid(detector, f, pattern_center, reference_pattern_center, rr, cc)
    m = centers.shape[0]
    grid = grid.reshape(n, m * roi_size, roi_size, 2)
    return _sample(ref, grid).reshape(n, m, roi_size, roi_size)


def remap_reference(
    reference: torch.Tensor, detector, f: torch.Tensor, pattern_center, reference_pattern_center
) -> torch.Tensor:
    """Re-project a whole reference pattern into the deformed configuration.

    :func:`hrebsd` uses a region-only path internally; this is the full-frame version,
    for inspecting a remap or for building one by hand.

    Parameters
    ----------
    reference
        ``(H, W)`` or ``(N, H, W)``, broadcast against ``f``.
    detector
        Geometry, for the pixel <-> gnomonic conversion.
    f
        ``(N, 3, 3)`` deformation gradient in the detector frame.
    pattern_center, reference_pattern_center
        Test and reference pattern centers, ``(3,)`` or ``(N, 3)``.

    Returns
    -------
    torch.Tensor
        ``(N, H, W)``, bilinearly resampled; rays falling outside the reference come
        back as zero.
    """
    device, dtype = reference.device, reference.dtype
    f = torch.as_tensor(f, device=device, dtype=dtype).reshape(-1, 3, 3)
    n = f.shape[0]
    pattern_center = (
        torch.as_tensor(pattern_center, device=device, dtype=dtype).reshape(-1, 3).expand(n, 3)
    )
    reference_pattern_center = (
        torch.as_tensor(reference_pattern_center, device=device, dtype=dtype)
        .reshape(-1, 3)
        .expand(n, 3)
    )
    ref = reference.reshape(-1, detector.nrows, detector.ncols)
    if ref.shape[0] == 1:
        ref = ref.expand(n, -1, -1)

    rows = torch.arange(detector.nrows, device=device, dtype=dtype)
    cols = torch.arange(detector.ncols, device=device, dtype=dtype)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    grid = _reference_grid(
        detector,
        f,
        pattern_center,
        reference_pattern_center,
        rr.reshape(1, -1).expand(n, -1),
        cc.reshape(1, -1).expand(n, -1),
    )
    return _sample(ref, grid.reshape(n, detector.nrows, detector.ncols, 2))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HREBSDResult:
    """What :func:`hrebsd` measured, per test pattern, in the **detector** frame.

    Use :meth:`to_sample` for the sample frame.
    """

    beta: torch.Tensor
    """``(N, 3, 3)`` displacement gradient, trace-free: the hydrostatic part is
    invisible to the projection (see the module docstring)."""

    strain: torch.Tensor
    """``(N, 3, 3)`` deviatoric strain, ``sym(beta)``."""

    rotation: torch.Tensor
    """``(N, 3, 3)`` lattice rotation, ``skew(beta)``."""

    rotation_vector: torch.Tensor
    """``(N, 3)`` axial vector of :attr:`rotation`, in radians."""

    shifts: torch.Tensor
    """``(N, M, 2)`` region shifts ``(drow, dcol)`` in pixels, from the final pass.
    After remapping these are residuals, so they should be small."""

    centers: torch.Tensor
    """``(M, 2)`` region-of-interest centers as ``(row, col)``."""

    peak: torch.Tensor
    """``(N, M)`` correlation peak height; low values flag an unusable region."""

    residual: torch.Tensor
    """``(N,)`` weighted rms of the shift-equation fit, in gnomonic units -- multiply
    by ``detector.nrows`` for pixels. The honest per-point error bar: divide by the
    gnomonic half-width to read it as a strain floor."""

    n_passes: int
    """Passes run, remapping between each."""

    def to_sample(self, detector):
        """``(beta, strain, rotation)`` rotated into the sample frame."""
        om = sample_to_detector_tensor(detector, self.beta.device, self.beta.dtype)

        def rot(t):
            return torch.einsum("ji,njk,kl->nil", om, t, om)

        return rot(self.beta), rot(self.strain), rot(self.rotation)


def hrebsd(
    patterns,
    reference,
    detector,
    pattern_center=None,
    reference_pattern_center=None,
    *,
    roi_size: int = 128,
    grid: int = 8,
    margin: float = 0.05,
    upsample: int = 50,
    n_passes: int = 2,
    bandpass: tuple[float, float] | None = None,
    whiten: float = 0.0,
    min_peak: float = 0.0,
    batch: int = 256,
    device=None,
) -> HREBSDResult:
    """Deviatoric strain and lattice rotation relative to a reference pattern.

    Cross-correlates a grid of regions between each test pattern and the reference,
    fits the gnomonic shift equation for the trace-free displacement gradient, then
    remaps the reference into the measured configuration and repeats. That second
    pass is what makes the result trustworthy once lattice rotations exceed a degree:
    it drops the residual rotation to first-pass accuracy, and pattern-center error
    reaches the strain only through that residual.

    Parameters
    ----------
    patterns
        Test patterns ``(H, W)`` or ``(N, H, W)``, background-corrected; see
        :mod:`de_ebsd.index.preprocess`.
    reference
        ``(H, W)``, or ``(N, H, W)`` for one reference per test point. Strain is
        measured *relative* to whatever state this pattern is in.
    detector
        :class:`~de_ebsd.sim.detector.EBSDDetector`, for the gnomonic geometry.
    pattern_center, reference_pattern_center
        Pattern center at the test and reference points, ``(3,)`` or ``(N, 3)``; both
        default to ``detector.pattern_center``. Passing them separately is worth doing over any
        appreciable scan -- a millimetre of travel moves the pattern center by
        several percent of the screen -- and the remap absorbs the difference.
    roi_size
        Region side in pixels. Larger regions correlate more precisely but average
        the deformation over a wider gnomonic span.
    grid
        Regions per side, so ``grid ** 2`` in total, against 8 free parameters.
    margin
        Edge inset as a fraction of each side.
    upsample
        Sub-pixel resolution of the correlation peak, ``1 / upsample`` pixels.
    n_passes
        Correlation passes; ``1`` disables remapping and is the classic single-pass
        formulation. A pattern-center difference is still applied on pass 1.
    bandpass
        ``(low, high)`` Gaussian band-pass in cycles per pixel (Nyquist is 0.5), or
        ``None`` for flat. Rarely needed on real patterns; raise ``low`` above zero to
        suppress slow background gradients that survived preprocessing.
    whiten
        Spectral whitening exponent, 1.0 being full phase correlation. Defaults to 0
        (plain correlation), which measured better than whitening on real patterns.
        See :func:`_measure_shifts` for the failure mode it exists to defend against.
    min_peak
        Regions whose normalized cross-correlation falls below this are dropped from
        the fit. Regions are weighted by that correlation regardless.
    batch
        Patterns correlated at once, which bounds memory: it scales as
        ``batch * grid**2 * roi_size**2``.
    device
        Defaults to the test patterns'.

    Returns
    -------
    HREBSDResult
        Detector-frame tensors, the raw region shifts, and a per-point residual.
    """
    patt = torch.as_tensor(patterns)
    device = resolve_device(device, patt.device)
    dtype = patt.dtype if patt.is_floating_point() else torch.float32
    patt = patt.to(device=device, dtype=dtype)
    if patt.ndim == 2:
        patt = patt[None]
    if patt.ndim != 3:
        raise ValueError(f"patterns must be (H, W) or (N, H, W), got {tuple(patt.shape)}")
    if tuple(patt.shape[-2:]) != (detector.nrows, detector.ncols):
        raise ValueError(
            f"patterns are {tuple(patt.shape[-2:])} but the detector is "
            f"{(detector.nrows, detector.ncols)}"
        )
    n = patt.shape[0]

    ref = torch.as_tensor(reference, device=device, dtype=dtype)
    if ref.ndim == 2:
        ref = ref[None]
    if ref.shape[0] not in (1, n):
        raise ValueError(f"reference must be (H, W) or (N, H, W) with N={n}")
    if ref.shape[0] == 1:
        ref = ref.expand(n, -1, -1)
    if n_passes < 1:
        raise ValueError("n_passes must be at least 1")

    pattern_center = detector.pattern_center if pattern_center is None else pattern_center
    reference_pattern_center = (
        pattern_center if reference_pattern_center is None else reference_pattern_center
    )
    pattern_center = torch.as_tensor(pattern_center, device=device, dtype=dtype).reshape(-1, 3)
    reference_pattern_center = torch.as_tensor(
        reference_pattern_center, device=device, dtype=dtype
    ).reshape(-1, 3)
    pattern_center = pattern_center.expand(n, 3) if pattern_center.shape[0] == 1 else pattern_center
    reference_pattern_center = (
        reference_pattern_center.expand(n, 3)
        if reference_pattern_center.shape[0] == 1
        else reference_pattern_center
    )

    if n > batch:
        parts = [
            hrebsd(
                patt[lo : lo + batch],
                ref[lo : lo + batch],
                detector,
                pattern_center[lo : lo + batch],
                reference_pattern_center[lo : lo + batch],
                roi_size=roi_size,
                grid=grid,
                margin=margin,
                upsample=upsample,
                n_passes=n_passes,
                bandpass=bandpass,
                whiten=whiten,
                min_peak=min_peak,
                batch=batch,
                device=device,
            )
            for lo in range(0, n, batch)
        ]
        return HREBSDResult(
            **{
                name: torch.cat([getattr(part, name) for part in parts])
                for name in (
                    "beta",
                    "strain",
                    "rotation",
                    "rotation_vector",
                    "shifts",
                    "peak",
                    "residual",
                )
            },
            centers=parts[0].centers,
            n_passes=n_passes,
        )

    centers = torch.as_tensor(
        roi_centers(detector, grid, roi_size, margin), device=device, dtype=dtype
    ).round()
    win = _window(roi_size, device, dtype)
    mask = _spectral_mask(roi_size, bandpass, device, dtype)
    test_rois = _extract(patt, centers, roi_size)  # fixed; the test never moves
    # shifts are measured in the test frame, so the design matrix uses the test pattern_center
    a = _design(detector, pattern_center, centers)

    eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
    f_total = eye.clone()
    shifts = peak = residual = None
    for _ in range(n_passes):
        ref_rois = _warped_rois(
            ref, detector, f_total, pattern_center, reference_pattern_center, centers, roi_size
        )
        shifts, peak = _measure_shifts(test_rois, ref_rois, win, upsample, mask, whiten=whiten)
        # pixels -> gnomonic: both axes scale by 1 / nrows, y increasing upward
        q = torch.stack([shifts[..., 1], -shifts[..., 0]], dim=-1) / detector.nrows
        w = torch.where(peak >= min_peak, peak.clamp_min(0.0), torch.zeros_like(peak))
        beta_step, residual = _solve_beta(a, q, w)
        f_total = (eye + beta_step) @ f_total

    beta = f_total - eye
    trace = torch.diagonal(beta, dim1=-2, dim2=-1).sum(-1)
    beta = beta - trace[:, None, None] / 3.0 * eye  # the gauge the projection fixes
    rot = 0.5 * (beta - beta.transpose(-1, -2))
    return HREBSDResult(
        beta=beta,
        strain=0.5 * (beta + beta.transpose(-1, -2)),
        rotation=rot,
        rotation_vector=torch.stack([rot[:, 2, 1], rot[:, 0, 2], rot[:, 1, 0]], dim=-1),
        shifts=shifts,
        centers=centers,
        peak=peak,
        residual=residual,
        n_passes=n_passes,
    )
