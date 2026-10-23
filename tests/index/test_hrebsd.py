"""Cross-correlation HR-EBSD (:mod:`de_ebsd.index.crosscorr`).

Deviatoric strain and lattice rotation measured differentially against a reference
pattern, with the reference remapped into the deformed configuration between passes.

The tests are deliberately split by what each one can actually pin down, because the
geometry and the correlation front-end fail in completely different ways:

1. **Projective gauge** -- ``F`` and ``lambda F`` are the same deformation to a
   gnomonic projection, so :func:`remap_reference` returns a bit-identical image for
   both, and the reported ``beta`` is always the trace-free representative.
2. **The remap is the identity when it should be** -- ``F = I`` with equal pattern
   centers reproduces a plain pixel extraction.
3. **The shift equation is the real projection, linearized** -- the design matrix
   agrees with the exact projective displacement of each region to ~1e-4 px.
4. **The solver inverts it** -- fed analytically exact shifts, the weighted fit
   returns ``beta`` to ~1e-3 of itself, which is the linearization residual.
5. **Shift noise propagates as advertised** -- the test that says what per-region
   precision a given strain resolution costs.
6. **The remap reproduces a deformed pattern** -- including at 2 degrees, where the
   first-order shift equation on its own has visibly broken down.
7. **Sub-pixel locator** -- a known Fourier shift comes back to the peak grid.

What these do *not* establish is end-to-end accuracy on real patterns. The synthetic
band master here is far smoother than a real Kikuchi pattern (region autocorrelation
half-width ~10 px in a 64 px region), which is the regime where a tapered correlation
is worst behaved, so ``test_end_to_end_*`` carries a deliberately loose tolerance and
is a smoke test, not a specification. The correlation stage wants calibrating against
real patterns; the geometry above does not.

fp64 on CPU at a small detector throughout.
"""

import numpy as np
import pytest
import torch

from band_master import make_band_master
from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.geometry import detector_rays, sample_to_detector_tensor
from de_ebsd.index.crosscorr import (
    _design,
    _measure_shifts,
    _pixels_from_ray,
    _solve_beta,
    _window,
    hrebsd,
    remap_reference,
    roi_centers,
)

DT = torch.float64


@pytest.fixture(scope="module")
def master() -> MasterPattern:
    """A band master. Smoother than a real pattern -- see the module docstring."""
    return make_band_master(npx=220, width=0.05, n_bands=18, seed=11)


@pytest.fixture(scope="module")
def detector() -> EBSDDetector:
    """Wide field of view: the pattern-center/rotation degeneracy breaks as (angle)^2."""
    return EBSDDetector(shape=(256, 256), pc=(0.5, 0.5, 0.6), sample_tilt=70.0, tilt=0.0)


_R = torch.tensor(
    [
        [0.5335, -0.8080, 0.2504],
        [0.8080, 0.4184, -0.4148],
        [0.2504, 0.4148, 0.8750],
    ],
    dtype=DT,
)

_BETA = torch.tensor([[1.0, 0.5, -0.3], [0.2, -0.4, 0.6], [-0.1, 0.3, -0.6]], dtype=DT)


def _trace_free(b):
    return b - torch.eye(3, dtype=DT) * torch.diagonal(b).sum() / 3.0


def _render(master, detector, f=None, pc=None, rot=_R):
    """Pattern with the lattice deformed by ``f``, rendered straight from the master.

    A feature at ray ``r`` moves to ``F r``, so the deformed pattern at ray ``r``
    carries the undeformed content at ``F^-1 r``. Rendering it this way keeps the
    ground truth independent of :func:`remap_reference`, which shares geometry code
    with the thing under test.
    """
    pc = torch.as_tensor(detector.pc if pc is None else pc, dtype=DT)
    rows = torch.arange(detector.nrows, dtype=DT)
    cols = torch.arange(detector.ncols, dtype=DT)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    ray = detector_rays(detector, pc, rr.reshape(-1), cc.reshape(-1))
    if f is not None:
        ray = ray @ torch.linalg.inv(torch.as_tensor(f, dtype=DT)).T
    om = sample_to_detector_tensor(detector, ray.device, DT)
    dc = ray @ om
    dc = dc / dc.norm(dim=-1, keepdim=True)
    return master.sample(dc @ rot.T).reshape(detector.nrows, detector.ncols).to(DT)


def _rotation(axis, angle):
    axis = torch.as_tensor(axis, dtype=DT)
    axis = axis / axis.norm()
    k = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=DT,
    )
    return torch.eye(3, dtype=DT) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _exact_shifts(detector, f, centers, pc=None):
    """Exact projective displacement ``(M, 2)`` of each region center, in pixels."""
    pc = torch.as_tensor(detector.pc if pc is None else pc, dtype=DT)[None]
    ray = detector_rays(detector, pc, centers[:, 0], centers[:, 1])
    row1, col1 = _pixels_from_ray(detector, pc, torch.einsum("ij,nkj->nki", f, ray))
    return torch.stack([row1[0] - centers[:, 0], col1[0] - centers[:, 1]], dim=-1)


def _centers(detector, grid=5, roi=64):
    return torch.as_tensor(roi_centers(detector, grid, roi, 0.02), dtype=DT).round()


def _px_to_gnomonic(detector, shifts):
    return torch.stack([shifts[..., 1], -shifts[..., 0]], dim=-1) / detector.nrows


def _run(test, ref, detector, **kw):
    kw.setdefault("roi_size", 64)
    kw.setdefault("grid", 5)
    kw.setdefault("margin", 0.02)
    kw.setdefault("upsample", 100)
    return hrebsd(test, ref, detector, **kw)


# --------------------------------------------------------------------------- #
# 1-2. the remap
# --------------------------------------------------------------------------- #


def test_remap_is_invariant_under_scaling_f(master, detector):
    """``F`` and ``lambda F`` project identically -- the hydrostatic blind spot, exactly."""
    ref = _render(master, detector)[None]
    f = (torch.eye(3, dtype=DT) + 3e-3 * _BETA)[None]
    pc = torch.as_tensor(detector.pc, dtype=DT)
    a = remap_reference(ref, detector, f, pc, pc)
    b = remap_reference(ref, detector, 1.37 * f, pc, pc)
    assert torch.allclose(a, b, atol=1e-12, rtol=0)


def test_identity_remap_matches_plain_extraction(master, detector):
    """``F = I`` with equal pattern centers must not move a single pixel."""
    ref = _render(master, detector)[None]
    pc = torch.as_tensor(detector.pc, dtype=DT)
    same = remap_reference(ref, detector, torch.eye(3, dtype=DT)[None], pc, pc)
    assert float((same - ref).abs().max()) < 1e-12


@pytest.mark.parametrize(("deg", "min_gain"), [(0.2, 3.0), (2.0, 20.0)])
def test_remap_reproduces_the_deformed_pattern(master, detector, deg, min_gain):
    """The remap is exact rather than first-order, so it keeps working at 2 degrees.

    Asserted as the gain over *not* remapping, since the absolute residual is set by
    bilinear resampling. The gain grows with rotation -- 3x at 0.2 degrees, >20x at
    2 -- which is the whole argument for the second pass.
    """
    f = _rotation([0.2, 0.9, -0.4], np.deg2rad(deg)) @ (torch.eye(3, dtype=DT) + 5e-4 * _BETA)
    ref = _render(master, detector)
    want = _render(master, detector, f)
    pc = torch.as_tensor(detector.pc, dtype=DT)
    got = remap_reference(ref[None], detector, f[None], pc, pc)[0]
    inner = slice(16, -16)  # the remap pulls zeros in at the border
    remapped = float((got[inner, inner] - want[inner, inner]).abs().max())
    raw = float((ref[inner, inner] - want[inner, inner]).abs().max())
    assert raw / remapped > min_gain, f"remap gain only {raw / remapped:.1f}x"


# --------------------------------------------------------------------------- #
# 3-4. the shift equation and its inverse
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scale", [2e-3, 8e-3])
def test_shift_equation_matches_exact_projection(detector, scale):
    """The design matrix is the true projective displacement, linearized."""
    centers = _centers(detector)
    beta = _trace_free(scale * _BETA)
    exact = _exact_shifts(detector, torch.eye(3, dtype=DT) + beta, centers)

    pc = torch.as_tensor(detector.pc, dtype=DT)[None]
    b8 = torch.tensor(
        [
            beta[0, 0],
            beta[0, 1],
            beta[0, 2],
            beta[1, 0],
            beta[1, 1],
            beta[1, 2],
            beta[2, 0],
            beta[2, 1],
        ],
        dtype=DT,
    )
    q = (_design(detector, pc, centers)[0] @ b8) * detector.nrows
    linear = torch.stack([-q[:, 1], q[:, 0]], dim=-1)
    # second-order in beta, so the 4x larger deformation should be ~16x worse
    assert float((linear - exact).abs().max()) < 1.0 * scale**2 * detector.nrows


def test_solver_inverts_exact_shifts(detector):
    """Fed exact shifts, the fit returns beta to the linearization residual."""
    centers = _centers(detector)
    beta = _trace_free(2e-3 * _BETA)
    q = _px_to_gnomonic(detector, _exact_shifts(detector, torch.eye(3, dtype=DT) + beta, centers))
    pc = torch.as_tensor(detector.pc, dtype=DT)[None]
    got, rms = _solve_beta(
        _design(detector, pc, centers), q[None], torch.ones(1, centers.shape[0], dtype=DT)
    )
    assert float((got[0] - beta).abs().max()) < 1e-5
    assert float(rms[0]) < 1e-6


@pytest.mark.parametrize("deg", [0.2, 0.5, 2.0])
def test_single_pass_phantom_strain_scales_as_theta_squared(detector, deg):
    """A pure rotation fits as a rotation *plus* a phantom strain of order theta^2.

    ``sym(Q - I) = (cos(theta) - 1)(I - n n^T)``, so a single-pass fit of a purely
    rotated lattice reports a deviatoric strain of 0.41 theta^2 on this geometry, with
    no strain present at all. That is 1.6e-5 at half a degree -- already at a 1e-5
    noise floor -- and it is exactly what remapping removes, by driving the rotation
    seen by the linear fit down to the first pass's residual.
    """
    centers = _centers(detector, grid=8, roi=32)
    theta = np.deg2rad(deg)
    q = _px_to_gnomonic(
        detector, _exact_shifts(detector, _rotation([0.3, -0.8, 0.5], theta), centers)
    )
    pc = torch.as_tensor(detector.pc, dtype=DT)[None]
    got, _ = _solve_beta(
        _design(detector, pc, centers), q[None], torch.ones(1, centers.shape[0], dtype=DT)
    )
    strain = 0.5 * (got[0] + got[0].T)
    rot = 0.5 * (got[0] - got[0].T)
    axial = float((rot[2, 1] ** 2 + rot[0, 2] ** 2 + rot[1, 0] ** 2) ** 0.5)
    assert abs(axial - theta) < 0.02 * theta  # the rotation itself comes back
    assert abs(float(strain.abs().max()) / (0.408 * theta**2) - 1.0) < 0.1


@pytest.mark.parametrize(("sigma_px", "tol"), [(0.005, 2e-5), (0.01, 4e-5), (0.05, 2e-4)])
def test_shift_noise_propagates_to_strain(detector, sigma_px, tol):
    """What per-region shift precision a strain resolution costs.

    With 64 regions on this geometry, independent shift noise of ``sigma_px`` lands
    in ``beta`` at roughly ``3e-3 * sigma_px``. Reaching 1e-5 strain therefore wants
    per-region precision of a few thousandths of a pixel, or proportionally more
    regions -- the error falls as ``1 / sqrt(M)``.
    """
    centers = _centers(detector, grid=8, roi=32)
    beta = _trace_free(2e-3 * _BETA)
    q = _px_to_gnomonic(detector, _exact_shifts(detector, torch.eye(3, dtype=DT) + beta, centers))
    torch.manual_seed(0)
    q = q + torch.randn(q.shape, dtype=DT) * sigma_px / detector.nrows
    pc = torch.as_tensor(detector.pc, dtype=DT)[None]
    got, _ = _solve_beta(
        _design(detector, pc, centers), q[None], torch.ones(1, centers.shape[0], dtype=DT)
    )
    assert float((got[0] - beta).abs().max()) < tol


# --------------------------------------------------------------------------- #
# 5. the sub-pixel locator
# --------------------------------------------------------------------------- #


def _shift_error(img, true, whiten):
    f = torch.fft.fftfreq(img.shape[-1], dtype=DT)
    phase = torch.exp(-2j * torch.pi * (f[:, None] * true[0] + f[None, :] * true[1]))
    shifted = torch.fft.ifft2(torch.fft.fft2(img) * phase).real
    win = _window(img.shape[-1], img.device, DT)
    got, ncc = _measure_shifts(
        shifted[None, None], img[None, None], win, upsample=100, whiten=whiten
    )
    return float((got[0, 0] - torch.tensor(true, dtype=DT)).abs().max()), float(ncc[0, 0])


@pytest.mark.parametrize("true", [(0.0, 0.0), (0.37, -0.62), (-1.24, 0.08)])
def test_subpixel_shift_recovery(master, detector, true):
    """A band-limited Fourier shift comes back to the peak grid resolution.

    Run with ``whiten=1`` because this fixture is the smooth pathology -- see
    :func:`test_whitening_is_what_rescues_a_pathologically_smooth_pattern`.
    """
    img = _render(master, detector)[96:160, 96:160]
    err, ncc = _shift_error(img, true, whiten=1.0)
    assert ncc > 0.9
    assert err < 0.02


def test_whitening_is_what_rescues_a_pathologically_smooth_pattern(master, detector):
    """Why ``whiten`` exists, and why it is nevertheless off by default.

    A taper is unavoidable, and the window's own autocorrelation peaks at zero lag,
    dragging a plain correlation peak toward it in proportion to how broad the
    *pattern's* autocorrelation is. This synthetic master is far smoother than a real
    Kikuchi pattern, so the pull is enormous -- whole pixels on a 1.24 px shift -- and
    whitening removes it. On real 2048px direct-detection patterns the pull is not
    measurable and plain correlation wins, so the default is ``whiten=0``.
    """
    img = _render(master, detector)[96:160, 96:160]
    plain, _ = _shift_error(img, (-1.24, 0.08), whiten=0.0)
    whitened, _ = _shift_error(img, (-1.24, 0.08), whiten=1.0)
    assert plain > 0.2, f"fixture stopped being pathological ({plain:.3f} px)"
    assert whitened < 0.02
    assert plain > 10 * whitened


# --------------------------------------------------------------------------- #
# 6. end to end -- a smoke test, see the module docstring
# --------------------------------------------------------------------------- #


def test_reported_beta_is_trace_free(master, detector):
    """Whatever gauge the input is in, the answer comes back in the measurable one."""
    f = 1.21 * (torch.eye(3, dtype=DT) + 2e-3 * _BETA)
    res = _run(_render(master, detector, f), _render(master, detector), detector)
    assert abs(float(torch.diagonal(res.beta[0]).sum())) < 1e-12


def test_end_to_end_runs_and_is_the_right_order(master, detector):
    """Images in, a deformation of the right size and sign out.

    Tolerance is loose on purpose: this fixture's regions are far smoother than real
    Kikuchi patterns, which is the worst case for a tapered correlation. It catches
    a broken pipeline, not an accuracy regression.
    """
    beta = _trace_free(2e-3 * _BETA)
    ref = _render(master, detector)
    test = _render(master, detector, torch.eye(3, dtype=DT) + beta)
    res = _run(test, ref, detector)
    got = res.beta[0]
    assert torch.isfinite(got).all()
    assert float(got.abs().max()) < 5.0 * float(beta.abs().max())
    assert float(res.peak.min()) > 0.5  # the regions did actually match each other


def test_batched_and_frames(master, detector):
    """A stack of test patterns, and the sample-frame rotation of the result."""
    ref = _render(master, detector)
    betas = [_trace_free(1e-3 * _BETA), _trace_free(-2e-3 * _BETA)]
    stack = torch.stack(
        [_render(master, detector, torch.eye(3, dtype=DT) + b) for b in betas], dim=0
    )
    res = _run(stack, ref, detector)
    assert res.beta.shape == (2, 3, 3)
    assert res.shifts.shape[0] == 2 and res.centers.shape[1] == 2
    assert res.peak.shape == (2, 25) and res.residual.shape == (2,)

    beta_s, strain_s, rot_s = res.to_sample(detector)
    om = sample_to_detector_tensor(detector, res.beta.device, DT)
    assert torch.allclose(beta_s[0], om.T @ res.beta[0] @ om, atol=1e-14)
    # a frame change is a similarity transform, so the invariants survive
    assert torch.allclose(
        torch.diagonal(strain_s, dim1=-2, dim2=-1).sum(-1), torch.zeros(2, dtype=DT), atol=1e-12
    )
    assert abs(float(rot_s[0].norm() - res.rotation[0].norm())) < 1e-14


def test_a_stack_in_batches_is_the_stack_at_once(master, detector):
    """Correlated a pattern at a time, a stack gives what it gives all at once."""
    ref = _render(master, detector)
    betas = [_trace_free(1e-3 * _BETA), _trace_free(-2e-3 * _BETA), _trace_free(5e-4 * _BETA)]
    stack = torch.stack(
        [_render(master, detector, torch.eye(3, dtype=DT) + b) for b in betas], dim=0
    )
    whole, batched = _run(stack, ref, detector), _run(stack, ref, detector, batch=1)
    assert torch.allclose(whole.beta, batched.beta, atol=1e-12)
    assert torch.allclose(whole.residual, batched.residual, atol=1e-12)
    assert batched.peak.shape == whole.peak.shape


def test_pattern_center_difference_changes_the_geometry(master, detector):
    """``pc_ref`` is actually plumbed through the remap, not silently ignored."""
    pc = np.array(detector.pc, dtype=float)
    pc_ref = pc + np.array([4e-3, 5e-3, 3e-3])
    ref = _render(master, detector, pc=pc_ref)
    test = _render(master, detector, pc=pc)
    good = _run(test, ref, detector, pc=pc, pc_ref=pc_ref)
    bad = _run(test, ref, detector, pc=pc)
    assert float(good.beta[0].abs().max()) < float(bad.beta[0].abs().max())


def test_rejects_mismatched_shapes(master, detector):
    ref = _render(master, detector)
    with pytest.raises(ValueError, match="detector"):
        hrebsd(torch.zeros(16, 16, dtype=DT), ref, detector)
    with pytest.raises(ValueError, match="n_passes"):
        hrebsd(ref, ref, detector, n_passes=0)
    with pytest.raises(ValueError, match="no room"):
        roi_centers(detector, 4, roi_size=512)
