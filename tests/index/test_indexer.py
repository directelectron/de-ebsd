"""End-to-end spherical indexing (I4): the closed round trip through the real sim.

Validates :class:`torchebsd.index.indexer.SphericalIndexer`. The money test is the
round trip: a ``torchebsd.sim`` master pattern is projected onto an
:class:`EBSDDetector` at a *known* orientation ``R`` with
:meth:`~torchebsd.sim.MasterPattern.project`, then indexed back; the disorientation
between the recovered orientation and ``R`` must land within the SOFT grid cell
(``< 1.5x`` the ``360 / (2L + 2)`` degree grid step). This is the test that catches
any inversion / transpose bug in the convention chain -- ``grid_to_rotation(argmax)``
is asserted to be the orientation passed to ``project`` *directly* (no ``~R``, no
transpose), so a wrong convention fails here loudly (the recovered orientation would
sit ~90-180 degrees away).

The synthetic master is a sum of bright **great-circle bands** (a stand-in for real
Kikuchi bands): bands tile the whole sphere, so the master has roughly uniform local
norm and the *unnormalized* masked cross-correlation of I4 is unbiased and recovers
the orientation reliably over the detector's partial field of view. (A master with a
few isolated bright spots would instead exercise the documented I4 bias toward
orientations that place those spots in the field of view -- the ``g``-dependent
normalization that removes it is the deferred I5 enhancement.) The bands are smooth
on the pixel scale, so the field is band-limited well within ``L = 16``. fp64 on CPU
throughout for an exact-ish check that runs fast and fits in memory.
"""

import math

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation

from torchebsd import EBSDDetector, MasterPattern
from torchebsd.index.backproject import pattern_to_sphere
from torchebsd.index.correlation import so3_cross_correlation
from torchebsd.index.indexer import SphericalIndexer
from torchebsd.sim.lambert import master_pattern_grid

BANDWIDTH = 16
# SOFT grid step in degrees: alpha/gamma are spaced 360 / (2L + 2). The peak should
# land in the correct grid cell, so we allow up to 1.5x the step (per INDEXING_DESIGN
# and the correlation-level round-trip test).
GRID_STEP_DEG = 360.0 / (2 * BANDWIDTH + 2)
TOL_DEG = 1.5 * GRID_STEP_DEG


def _band_field(v: torch.Tensor, poles: torch.Tensor, amps: torch.Tensor, width: float):
    """A smooth Kikuchi-band-like scalar field on the sphere.

    Each pole contributes a bright Gaussian band about its great circle
    (``exp(-((n . pole) / width) ** 2)`` peaks where ``n`` is perpendicular to the
    pole). Bands cover the whole sphere, giving the field near-uniform local norm --
    exactly what lets the unnormalized I4 correlation recover orientation without the
    bright-region bias. Band-limited (smooth) for ``width`` not too small.
    """
    d = v.to(torch.float64) @ poles.T  # (..., n_bands)
    return (amps * torch.exp(-(d / width) ** 2)).sum(-1)


@pytest.fixture(scope="module")
def master() -> MasterPattern:
    """A synthetic band master, filled on a fine square-Lambert grid (both hemispheres).

    Twelve great-circle bands at fixed pseudo-random poles: distinctive and asymmetric
    (a genuine test of the frame convention) yet uniform enough in local norm that the
    unnormalized correlation is unbiased.
    """
    g = torch.Generator().manual_seed(7)
    n_bands = 12
    width = 0.11
    poles = torch.randn(n_bands, 3, generator=g, dtype=torch.float64)
    poles = poles / poles.norm(dim=-1, keepdim=True)
    amps = 0.5 + torch.rand(n_bands, generator=g, dtype=torch.float64)

    npx = 160
    north = master_pattern_grid(npx, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    data = torch.stack(
        [_band_field(north, poles, amps, width), _band_field(south, poles, amps, width)],
        dim=0,
    )[None]  # (1, 2, n, n)
    return MasterPattern(data=data, energies_kev=np.array([20.0]))


@pytest.fixture(scope="module")
def detector() -> EBSDDetector:
    """A realistic tilted detector with a generous (~38%) field of view."""
    return EBSDDetector(shape=(90, 90), pc=(0.5, 0.5, 0.20), sample_tilt=70.0, tilt=5.0)


def _disorientation_deg(r_a: Rotation, r_b: Rotation) -> float:
    """Angle (degrees) between two orix Rotations, no crystal symmetry applied."""
    return float(np.asarray((r_a * (~r_b)).angle).ravel()[0]) * 180.0 / math.pi


# a general / asymmetric orientation plus a few others (Bunge Euler angles, degrees)
_KNOWN_EULER_DEG = np.array(
    [
        [10.0, 20.0, 30.0],
        [45.0, 55.0, 15.0],
        [123.0, 47.0, 205.0],  # general / asymmetric
        [200.0, 100.0, 300.0],  # general / asymmetric
    ]
)


def test_roundtrip_recovers_known_orientation(master, detector):
    """THE acceptance test: sim a pattern at known R, index it back to < grid cell.

    Renders each known orientation with ``master.project`` and indexes the stack; the
    recovered orientation must sit within ``1.5x`` the SOFT grid step of the true one.
    A wrong inversion / transpose convention would land ~90-180 degrees off and fail.
    """
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)  # (4, H, W)

    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    rot_out, quality = indexer.index(patterns)

    assert rot_out.size == r_true.size
    for i in range(r_true.size):
        dis = _disorientation_deg(r_true[i], rot_out[i])
        # the direct convention must be right: the inverse hypothesis lands far away
        dis_inv = _disorientation_deg(r_true[i], ~rot_out[i])
        assert dis < TOL_DEG, f"orientation {i}: {dis:.2f} deg > {TOL_DEG:.2f} deg tol"
        assert dis < dis_inv  # grid_to_rotation(argmax) is R itself, not R^-1
    assert torch.isfinite(quality).all()


def test_batched_indexing_shapes_and_recovery(master, detector):
    """A stack of orientations is indexed in one shot; shapes and each R come back."""
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)

    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    rot_out, quality = indexer.index(patterns)

    assert rot_out.size == 4
    assert tuple(quality.shape) == (4,)
    # batched result equals indexing each pattern on its own
    for i in range(4):
        ri, qi = indexer.index(patterns[i])
        assert ri.size == 1
        assert _disorientation_deg(rot_out[i], ri[0]) < 1e-6
        assert float(qi[0]) == pytest.approx(float(quality[i]), rel=1e-9)


def test_single_pattern_input(master, detector):
    """A bare ``(H, W)`` pattern indexes to a length-1 Rotation + length-1 quality."""
    r_true = Rotation.from_euler(np.deg2rad([[30.0, 40.0, 50.0]]))
    pattern = master.project(detector, r_true, rescale=True)[0]  # (H, W)
    assert pattern.ndim == 2

    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    rot_out, quality = indexer.index(pattern)
    assert rot_out.size == 1
    assert tuple(quality.shape) == (1,)
    assert _disorientation_deg(r_true[0], rot_out[0]) < TOL_DEG


def test_numpy_input_is_accepted(master, detector):
    """Plain numpy arrays index without any torch/kikuchipy wrapping."""
    r_true = Rotation.from_euler(np.deg2rad([[70.0, 25.0, 110.0]]))
    pattern = master.project(detector, r_true, rescale=True)[0].cpu().numpy()

    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    rot_out, quality = indexer.index(pattern)
    assert rot_out.size == 1
    assert torch.isfinite(quality).all()
    assert _disorientation_deg(r_true[0], rot_out[0]) < TOL_DEG


def _windowed_pattern_coeffs(indexer, pattern):
    """Reproduce the indexer's FOV windowing + SHT to score a pattern off-peak.

    Mirrors the (settled) I4 windowing so a test can evaluate the correlation surface
    directly (using only public modules): standardize over the FOV, zero outside it,
    forward-transform.
    """
    dirs = indexer.sht.grid_directions().reshape(-1, 3)
    samples, mask = pattern_to_sphere(pattern[None], indexer.detector, dirs)
    w = mask
    denom = w.sum().clamp_min(1e-12)
    mean = (samples * w).sum(-1, keepdim=True) / denom
    var = ((samples - mean) ** 2 * w).sum(-1, keepdim=True) / denom
    e = ((samples - mean) / var.clamp_min(1e-12).sqrt()) * w
    return indexer.sht.forward(e.reshape(*e.shape[:-1], indexer.sht.n_theta, indexer.sht.n_phi))


def test_quality_is_finite_and_peaks_above_random_orientation(master, detector):
    """Quality is finite and the peak beats the score at a random (wrong) orientation.

    ``quality`` is the peak of the SO(3) correlation surface; for a genuine pattern it
    is the maximum over the grid, so it must exceed the surface's value at a randomly
    chosen orientation cell (and it does so with a clear margin for the band master).
    """
    r_true = Rotation.from_euler(np.deg2rad([[15.0, 35.0, 95.0]]))
    pattern = master.project(detector, r_true, rescale=True)[0]

    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    _, quality = indexer.index(pattern)
    assert torch.isfinite(quality).all()

    # the full correlation surface for this pattern
    b_lm = _windowed_pattern_coeffs(indexer, pattern)
    c = so3_cross_correlation(indexer.a_lm, b_lm, BANDWIDTH)[0]  # (N, N, N)
    peak = float(c.max())
    assert float(quality[0]) == pytest.approx(peak, rel=1e-9)  # quality is that peak

    # score at a handful of random (wrong) grid orientations: all below the peak
    n = 2 * BANDWIDTH + 2
    torch.manual_seed(0)
    idx = torch.randint(0, n, (20, 3))
    random_scores = c[idx[:, 0], idx[:, 1], idx[:, 2]]
    assert peak > float(random_scores.max())
    # and the peak stands clearly above the bulk of the surface (a real match)
    assert peak > float(c.median()) + 2.0 * float(c.std())


def test_device_and_dtype_plumbing_cpu(master, detector):
    """The shared SHT plan is pinned to the resolved device and the master's dtype."""
    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    assert indexer.device == torch.device("cpu")
    # master is fp64 -> plan is fp64, master spectrum is complex128
    assert indexer.dtype == torch.float64
    assert indexer.sht.device == torch.device("cpu")
    assert indexer.sht.dtype == torch.float64
    assert indexer.a_lm.dtype == torch.complex128
    assert indexer.a_lm.shape == ((BANDWIDTH + 1) ** 2,)
    # default device resolution falls back to the master's device when device=None
    assert SphericalIndexer(master, detector, bandwidth=BANDWIDTH).device == master.device


def test_float32_master_propagates_dtype(master, detector):
    """A float32 master yields a float32 plan and a complex64 spectrum, and still indexes."""
    master32 = MasterPattern(
        data=master.data.to(torch.float32), energies_kev=master.energies_kev
    )
    indexer = SphericalIndexer(master32, detector, bandwidth=BANDWIDTH, device="cpu")
    assert indexer.dtype == torch.float32
    assert indexer.sht.dtype == torch.float32
    assert indexer.a_lm.dtype == torch.complex64

    r_true = Rotation.from_euler(np.deg2rad([[80.0, 30.0, 160.0]]))
    pattern = master32.project(detector, r_true, rescale=True)[0]
    rot_out, quality = indexer.index(pattern)
    assert torch.isfinite(quality).all()
    assert _disorientation_deg(r_true[0], rot_out[0]) < TOL_DEG


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_path_matches_cpu(master, detector):
    """When CUDA is present, the indexer runs on it and recovers the orientation."""
    master_cuda = master.to("cuda")
    indexer = SphericalIndexer(master_cuda, detector, bandwidth=BANDWIDTH, device="cuda")
    assert indexer.device.type == "cuda"
    r_true = Rotation.from_euler(np.deg2rad([[123.0, 47.0, 205.0]]))
    pattern = master_cuda.project(detector, r_true, rescale=True)[0]
    rot_out, quality = indexer.index(pattern)
    assert torch.isfinite(quality).all()
    assert _disorientation_deg(r_true[0], rot_out[0]) < TOL_DEG
