"""Detector -> sphere back-projection (I3): round-trip, FOV mask, autograd, batching.

Validates :mod:`torchebsd.index.backproject`. The money test is the round trip: a
``torchebsd.sim`` master pattern is projected onto an :class:`EBSDDetector` and then
back-projected onto sphere directions; over the detector's field of view it must
recover the master evaluated on those same directions to bilinear-interpolation
accuracy. The rotated case recovers the *rotated* master, pinning the frame/orientation
convention (back-projection applies no rotation). The field-of-view mask matches the
detector's gnomonic bounds, and gradients flow into both the pattern and the
pattern center (fp64, matched to finite differences). fp64 throughout.
"""

import numpy as np
import pytest
import torch

from torchebsd import EBSDDetector, MasterPattern
from torchebsd.index.backproject import directions_to_pixels, pattern_to_sphere
from torchebsd.index.harmonics import SphericalHarmonicTransform
from torchebsd.sim.lambert import master_pattern_grid
from torchebsd.sim.rotation import quaternion_to_matrix

IDENTITY_Q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
# a generic, non-axis-aligned unit quaternion (so the rotation genuinely mixes axes)
GENERIC_Q = torch.tensor([[0.80, 0.30, -0.45, 0.25]], dtype=torch.float64)
GENERIC_Q = GENERIC_Q / GENERIC_Q.norm()


def _smooth_field(v: torch.Tensor) -> torch.Tensor:
    """A smooth, non-degenerate scalar field on the sphere (band-limited, low order).

    Being smooth on the scale of a detector pixel is what makes the back-projection
    round trip exact to bilinear accuracy; being genuinely three-dimensional makes the
    rotated round trip a real test of the frame convention.
    """
    nx, ny, nz = v[..., 0], v[..., 1], v[..., 2]
    return (
        1.5 + 0.6 * nx + 0.3 * ny + 0.2 * nz
        + 0.4 * nx * ny - 0.25 * (nx**2 - ny**2) + 0.15 * ny * nz
    )


@pytest.fixture(scope="module")
def master() -> MasterPattern:
    """A synthetic master whose ``.sample`` reproduces ``_smooth_field`` closely.

    Both hemispheres are filled from the same field on a fine square-Lambert grid, so
    the Lambert discretization error is negligible and the round-trip residual is due
    to the detector-plane bilinear interpolation alone.
    """
    npx = 300
    north = master_pattern_grid(npx, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    data = torch.stack([_smooth_field(north), _smooth_field(south)], dim=0)[None]
    return MasterPattern(data=data, energies_kev=np.array([20.0]))


@pytest.fixture(scope="module")
def detector() -> EBSDDetector:
    return EBSDDetector(shape=(180, 180), pc=(0.5, 0.5, 0.62), sample_tilt=70.0, tilt=5.0)


@pytest.fixture(scope="module")
def directions() -> torch.Tensor:
    """SHT quadrature grid directions reshaped to ``(M, 3)`` (the I1 sampling grid)."""
    plan = SphericalHarmonicTransform(48, dtype=torch.float64)
    return plan.grid_directions().reshape(-1, 3)


def test_roundtrip_identity(master, detector, directions):
    """Project at identity, back-project: recovers the master over the FOV (bilinear)."""
    pattern = master.project(detector, IDENTITY_Q, rescale=False)  # (1, H, W)
    samples, mask = pattern_to_sphere(pattern, detector, directions)
    expected = master.sample(directions)

    fov = mask > 0
    assert 0.05 < fov.float().mean() < 0.95  # a sensible partial field of view
    err = (samples[0][fov] - expected[fov]).abs()
    assert err.max().item() < 4e-3
    assert err.pow(2).mean().sqrt().item() < 5e-4


def test_roundtrip_rotated(master, detector, directions):
    """Project at a generic orientation R: back-projection recovers the R-rotated master.

    ``project`` samples the master at ``R . n`` for the pixel looking along ``n``, and
    back-projection applies *no* rotation, so ``E(n)`` must equal ``M(R . n)`` over the
    field of view. This pins the frame/orientation convention.
    """
    pattern = master.project(detector, GENERIC_Q, rescale=False)
    samples, mask = pattern_to_sphere(pattern, detector, directions)

    rot = quaternion_to_matrix(GENERIC_Q)[0]  # (3, 3), rotates column vectors
    rotated_dirs = directions @ rot.T  # rows hold R . n
    expected = master.sample(rotated_dirs)

    fov = mask > 0
    err = (samples[0][fov] - expected[fov]).abs()
    assert err.max().item() < 4e-3
    assert err.pow(2).mean().sqrt().item() < 5e-4

    # the recovered field is the *rotated* master, not the identity one: the wrong
    # convention (no rotation) is off by a large margin, so the convention is genuine
    identity_expected = master.sample(directions)
    wrong = (samples[0][fov] - identity_expected[fov]).abs()
    assert wrong.max().item() > 0.05


def test_fov_mask_matches_gnomonic_bounds(master, detector, directions):
    """The mask is exactly ``d_z > 0`` and inside the detector's gnomonic bounds."""
    _, mask = pattern_to_sphere(
        torch.zeros(detector.nrows, detector.ncols, dtype=torch.float64), detector, directions
    )

    # recompute the gnomonic coordinates independently from the public bounds
    from torchebsd.index.backproject import _sample_to_detector_tensor

    om = _sample_to_detector_tensor(detector, directions.device, directions.dtype)
    comp = directions @ om.transpose(-2, -1)
    gn_x = comp[..., 0] / comp[..., 2]
    gn_y = comp[..., 1] / comp[..., 2]
    x_min, x_max, y_min, y_max = detector.gnomonic_bounds
    expected = (
        (comp[..., 2] > 0)
        & (gn_x >= x_min) & (gn_x <= x_max)
        & (gn_y >= y_min) & (gn_y <= y_max)
    )
    assert torch.equal(mask > 0, expected)

    # every pixel-center direction (the detector's own direction cosines) is in the FOV
    dc = detector.direction_cosines(dtype=torch.float64)
    _, _, dc_in = directions_to_pixels(dc, detector)
    assert bool(dc_in.all())

    # directions pointing away from the detector (behind it) are masked out
    _, _, behind = directions_to_pixels(-dc, detector)
    assert not bool(behind.any())


def test_roundtrip_at_pixel_centers_is_exact(master, detector):
    """Back-projecting exactly at the pixel-center directions returns the pixels.

    With no sub-pixel interpolation the round trip is exact to floating point, isolating
    the geometry (direction -> pixel) from the bilinear read-off.
    """
    dc = detector.direction_cosines(dtype=torch.float64)  # (H*W, 3), row-major
    pattern = master.project(detector, IDENTITY_Q, rescale=False)  # (1, H, W)
    samples, mask = pattern_to_sphere(pattern, detector, dc)
    assert bool((mask > 0).all())
    assert torch.allclose(samples[0], pattern[0].reshape(-1), atol=1e-9)


def test_autograd_into_pattern(master, detector, directions):
    """Gradients flow into the pattern and match finite differences at sampled pixels."""
    pattern = master.project(detector, IDENTITY_Q, rescale=False)[0].clone()
    torch.manual_seed(0)
    weights = torch.randn(directions.shape[0], dtype=torch.float64)

    def functional(p):
        s, mask = pattern_to_sphere(p, detector, directions)
        return (s * (mask > 0).to(s.dtype) * weights).sum()

    p = pattern.clone().requires_grad_(True)
    functional(p).backward()
    grad = p.grad

    eps = 1e-6
    for r, c in [(90, 90), (60, 120), (120, 60)]:
        pp, pm = pattern.clone(), pattern.clone()
        pp[r, c] += eps
        pm[r, c] -= eps
        fd = (functional(pp) - functional(pm)).item() / (2 * eps)
        assert grad[r, c].item() == pytest.approx(fd, abs=1e-6, rel=1e-6)


def test_autograd_into_pattern_center_matches_fd(master, detector, directions):
    """Gradients flow into the pattern center and match central finite differences.

    This is the differentiability that enables joint PC refinement (I6): the whole
    direction -> pixel map is autograd-differentiable in ``pc = (pcx, pcy, pcz)``.
    """
    pattern = master.project(detector, IDENTITY_Q, rescale=False)  # (1, H, W)
    pc0 = torch.tensor(detector.pc, dtype=torch.float64)
    torch.manual_seed(1)
    weights = torch.randn(directions.shape[0], dtype=torch.float64)
    # Bilinear grid_sample has derivative kinks at integer pixel gridlines and at the
    # border-clamp edge; a finite-difference step that straddles one gives a wrong
    # slope there. Restrict the functional to directions that sit in the *smooth
    # interior* of a bilinear cell so autograd and FD agree to floating point (the
    # kink itself is the bilinear scheme, not the pattern-center derivative).
    col, row, in_fov = directions_to_pixels(directions, detector, pc0)
    margin = 0.05
    interior = (
        in_fov
        & ((col - col.round()).abs() > margin) & ((row - row.round()).abs() > margin)
        & (col > margin) & (col < detector.ncols - 1 - margin)
        & (row > margin) & (row < detector.nrows - 1 - margin)
    )
    weights = (weights * interior.to(torch.float64)).detach()

    def functional(pc):
        s, _ = pattern_to_sphere(pattern, detector, directions, pc=pc)
        return (s[0] * weights).sum()

    pc = pc0.clone().requires_grad_(True)
    functional(pc).backward()
    grad = pc.grad

    eps = 1e-6
    for k in range(3):
        pp, pm = pc0.clone(), pc0.clone()
        pp[k] += eps
        pm[k] -= eps
        fd = (functional(pp) - functional(pm)).item() / (2 * eps)
        assert grad[k].item() == pytest.approx(fd, abs=1e-5, rel=1e-5)


def test_batched_equals_loop(master, detector, directions):
    """A batch of patterns equals the per-pattern loop; the mask is shared."""
    p0 = master.project(detector, IDENTITY_Q, rescale=False)[0]
    p1 = master.project(detector, GENERIC_Q, rescale=False)[0]
    batch = torch.stack([p0, p1, 0.5 * p0 + 0.1])  # (3, H, W)

    samples, mask = pattern_to_sphere(batch, detector, directions)
    assert samples.shape == (3, directions.shape[0])
    assert mask.shape == (directions.shape[0],)
    for i in range(3):
        si, mi = pattern_to_sphere(batch[i], detector, directions)
        assert torch.allclose(samples[i], si, atol=1e-14)
        assert torch.equal(mask, mi)

    # leading dims of arbitrary rank are preserved
    stacked = batch.reshape(1, 3, detector.nrows, detector.ncols)
    s2, _ = pattern_to_sphere(stacked, detector, directions)
    assert s2.shape == (1, 3, directions.shape[0])
    assert torch.allclose(s2[0], samples)


def test_shape_validation(detector, directions):
    """Mismatched pattern / direction shapes raise."""
    with pytest.raises(ValueError):
        pattern_to_sphere(torch.zeros(10, 10), detector, directions)
    with pytest.raises(ValueError):
        pattern_to_sphere(
            torch.zeros(detector.nrows, detector.ncols), detector, directions[:, :2]
        )
