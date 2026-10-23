"""End-to-end spherical indexing (I4): the closed round trip through the real sim.

Validates :class:`de_ebsd.index.spherical.indexer.SphericalIndexer`. The money test is the
round trip: a ``de_ebsd.sim`` master pattern is projected onto an
:class:`EBSDDetector` at a *known* orientation ``R`` with
:meth:`~de_ebsd.sim.MasterPattern.project`, then indexed back; the disorientation
between the recovered orientation and ``R`` must land within the SOFT grid cell
(``< 1.5x`` the ``360 / (2L + 2)`` degree grid step). This is the test that catches
any inversion / transpose bug in the convention chain -- ``grid_to_rotation(argmax)``
is asserted to be the orientation passed to ``project`` *directly* (no ``~R``, no
transpose), so a wrong convention fails here loudly (the recovered orientation would
sit ~90-180 degrees away).

The synthetic band master is a sum of bright **great-circle bands** (a stand-in for
real Kikuchi bands): bands tile the whole sphere, so the master has roughly uniform
local norm and the *unnormalized* masked cross-correlation of I4 is unbiased and
recovers the orientation reliably over the detector's partial field of view. The bands
are smooth on the pixel scale, so the field is band-limited well within ``L = 16``.

The **locally-normalized (masked) cross-correlation** (``normalize=True``, now the
default) is exercised by a second master with a few isolated bright **spots** on top of
the bands: its local norm varies strongly across the sphere, so the *unnormalized*
score is biased toward orientations that merely place a bright spot in the field of
view -- and misindexes some orientations by ~100-180 degrees. The money test
(:func:`test_normalized_cc_fixes_bright_spot_bias`) reproduces that failure with
``normalize=False`` and shows ``normalize=True`` recovers every one of the same
orientations to within the SOFT grid cell. The bands underneath guarantee non-zero
local variance in every FOV, so the normalized correlation is well-posed. fp64 on CPU
throughout for an exact-ish check that runs fast and fits in memory.
"""

import math

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation

from band_master import band_field, make_band_master
from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.lambert import master_pattern_grid
from de_ebsd.crystal import Crystal
from de_ebsd.index.backproject import pattern_to_sphere
from de_ebsd.index.spherical.correlation import so3_cross_correlation
from de_ebsd.index.spherical.indexer import SphericalIndexer
from de_ebsd.index.spherical.wigner import grid_to_rotation

BANDWIDTH = 16
# SOFT grid step in degrees: alpha/gamma are spaced 360 / (2L + 2). The peak should
# land in the correct grid cell, so we allow up to 1.5x the step (see also the
# correlation-level round-trip test).
GRID_STEP_DEG = 360.0 / (2 * BANDWIDTH + 2)
TOL_DEG = 1.5 * GRID_STEP_DEG

# The bright-spot / normalization tests use a slightly larger bandwidth so the narrow
# spots are well represented; still modest (fp64 CPU, fast).
BW_SPOT = 20
GRID_STEP_SPOT_DEG = 360.0 / (2 * BW_SPOT + 2)
TOL_SPOT_DEG = 1.5 * GRID_STEP_SPOT_DEG


@pytest.fixture(scope="module")
def master() -> MasterPattern:
    """A synthetic band master, filled on a fine square-Lambert grid (both hemispheres).

    Twelve great-circle bands at fixed pseudo-random poles: distinctive and asymmetric
    (a genuine test of the frame convention) yet uniform enough in local norm that the
    unnormalized correlation is unbiased.
    """
    return make_band_master()


@pytest.fixture(scope="module")
def detector() -> EBSDDetector:
    """A realistic tilted detector with a generous (~38%) field of view."""
    return EBSDDetector(shape=(90, 90), pattern_center=(0.5, 0.5, 0.20), sample_tilt=70.0, tilt=5.0)


def _spot_field(
    v: torch.Tensor,
    poles: torch.Tensor,
    pamps: torch.Tensor,
    bwidth: float,
    centers: torch.Tensor,
    camps: torch.Tensor,
    cwidth: float,
) -> torch.Tensor:
    """Great-circle bands PLUS a few isolated bright Gaussian spots.

    The band part (as in :func:`_band_field`) gives every field of view non-zero local
    variance so the normalized correlation is well-posed; the spots (``camps`` large)
    make the *local norm* vary strongly across the sphere, which is exactly what biases
    the unnormalized masked correlation toward orientations that put a spot in the FOV.
    """
    v = v.to(torch.float64)
    d = v @ poles.T
    bands = (pamps * torch.exp(-((d / bwidth) ** 2))).sum(-1)
    ang = torch.arccos((v @ centers.T).clamp(-1.0, 1.0))  # angle to each spot center
    spots = (camps * torch.exp(-0.5 * (ang / cwidth) ** 2)).sum(-1)
    return bands + spots


@pytest.fixture(scope="module")
def spot_master() -> MasterPattern:
    """A bright-spot master: 12 bands + 3 very bright, narrow, isolated spots.

    The spots (base amplitude ~12x the bands) dominate the *unnormalized* correlation
    wherever they fall in the FOV, so it misindexes some orientations grossly; the
    ``g``-dependent normalization divides that local energy out and fixes it.
    """
    g = torch.Generator().manual_seed(3)
    poles = torch.randn(12, 3, generator=g, dtype=torch.float64)
    poles = poles / poles.norm(dim=-1, keepdim=True)
    pamps = 0.5 + torch.rand(12, generator=g, dtype=torch.float64)
    centers = torch.randn(3, 3, generator=g, dtype=torch.float64)
    centers = centers / centers.norm(dim=-1, keepdim=True)
    camps = 12.0 * (0.6 + 0.4 * torch.rand(3, generator=g, dtype=torch.float64))

    npx = 200
    north = master_pattern_grid(npx, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    data = torch.stack(
        [
            _spot_field(north, poles, pamps, 0.11, centers, camps, 0.16),
            _spot_field(south, poles, pamps, 0.11, centers, camps, 0.16),
        ],
        dim=0,
    )[None]  # (1, 2, n, n)
    return MasterPattern(data=data, energies_kev=np.array([20.0]))


@pytest.fixture(scope="module")
def detector_partial() -> EBSDDetector:
    """A tilted detector with a partial (~35%) field of view for the bias test."""
    return EBSDDetector(shape=(90, 90), pattern_center=(0.5, 0.5, 0.25), sample_tilt=70.0, tilt=5.0)


# eight known orientations (Bunge Euler, degrees) for the bright-spot money test
_SPOT_EULER_DEG = np.array(
    [
        [10.0, 20.0, 30.0],
        [45.0, 55.0, 15.0],
        [123.0, 47.0, 205.0],
        [200.0, 100.0, 300.0],
        [300.0, 80.0, 40.0],
        [15.0, 35.0, 95.0],
        [250.0, 60.0, 150.0],
        [88.0, 120.0, 260.0],
    ]
)


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

    Pins the ``normalize=False`` (unnormalized) baseline: ``quality`` is then the peak
    of the raw ``X`` SO(3) correlation surface, so it must equal that surface's maximum
    and exceed its value at a randomly chosen orientation cell (with a clear margin for
    the band master). (The ``normalize=True`` default instead returns the peak of the
    locally-normalized ``gamma`` surface -- a different, correlation-coefficient-like
    number -- which is checked in the bright-spot tests below.)
    """
    r_true = Rotation.from_euler(np.deg2rad([[15.0, 35.0, 95.0]]))
    pattern = master.project(detector, r_true, rescale=True)[0]

    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    _, quality = indexer.index(pattern, normalize=False)
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
    master32 = MasterPattern(data=master.data.to(torch.float32), energies_kev=master.energies_kev)
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


# --------------------------------------------------------------------------- #
# I5: locally-normalized (masked) cross-correlation -- the bright-spot bias fix
# --------------------------------------------------------------------------- #
def test_normalized_cc_fixes_bright_spot_bias(spot_master, detector_partial):
    """THE money test: unnormalized misindexes the bright-spot master; normalized fixes it.

    On a master with a few very bright localized spots and a partial FOV, the
    *unnormalized* masked correlation (``normalize=False``) is biased toward
    orientations that place a spot in the field of view, so it misindexes at least one
    of the known orientations by ~100-180 degrees (a gross, far-off spurious peak, not a
    near-miss). The **locally-normalized** correlation (``normalize=True``, the default)
    divides out that ``g``-dependent local energy and recovers every one of the *same*
    orientations to within ~1.5x the SOFT grid cell.
    """
    r_true = Rotation.from_euler(np.deg2rad(_SPOT_EULER_DEG))
    patterns = spot_master.project(detector_partial, r_true, rescale=True)  # (8, H, W)

    indexer = SphericalIndexer(spot_master, detector_partial, bandwidth=BW_SPOT, device="cpu")

    # (1) unnormalized baseline is biased -> at least one gross misindexing
    rot_u, _ = indexer.index(patterns, normalize=False)
    dis_u = [_disorientation_deg(r_true[i], rot_u[i]) for i in range(r_true.size)]
    n_fail_u = sum(d > TOL_SPOT_DEG for d in dis_u)
    assert n_fail_u >= 1, (
        "expected the unnormalized bright-spot bias to misindex >= 1 orientation "
        f"(so normalization is what fixes it); got disorientations {dis_u}"
    )
    assert max(dis_u) > 90.0  # the failure is a far-off spurious peak, not a near-miss

    # (2) normalized correlation recovers every one of the same orientations
    rot_n, quality = indexer.index(patterns, normalize=True)
    dis_n = [_disorientation_deg(r_true[i], rot_n[i]) for i in range(r_true.size)]
    for i, dis in enumerate(dis_n):
        assert dis < TOL_SPOT_DEG, (
            f"orientation {i}: normalized {dis:.2f} deg > {TOL_SPOT_DEG:.2f} deg tol "
            f"(unnormalized was {dis_u[i]:.2f} deg)"
        )
    assert torch.isfinite(quality).all()


def test_normalized_quality_in_correlation_range(spot_master, detector_partial):
    """``normalize=True`` quality is finite and correlation-coefficient-like (~[-1, 1])."""
    r_true = Rotation.from_euler(np.deg2rad(_SPOT_EULER_DEG))
    patterns = spot_master.project(detector_partial, r_true, rescale=True)

    indexer = SphericalIndexer(spot_master, detector_partial, bandwidth=BW_SPOT, device="cpu")
    _, quality = indexer.index(patterns, normalize=True)

    assert torch.isfinite(quality).all()
    # gamma is a proper correlation-coefficient-like number: for genuine matches it is
    # clearly positive and around [0, 1] (small band-limiting overshoot above 1 is fine).
    assert float(quality.min()) > 0.0
    assert float(quality.max()) < 1.3


def test_geometry_normalizer_computed_once_and_shared(spot_master, detector_partial):
    """The geometry-derived denominator is built once, shaped (N, N, N), and reused.

    ``a2_lm`` is precomputed in ``__init__``; the FOV-mask normalizer is built lazily on
    the first ``normalize=True`` call, cached, and shared across the batch (and across
    later calls) -- it is a function of the master + fixed detector geometry only.
    """
    n = 2 * BW_SPOT + 2
    indexer = SphericalIndexer(spot_master, detector_partial, bandwidth=BW_SPOT, device="cpu")
    # a2_lm (the S2 spectrum) is precomputed alongside a_lm in __init__
    assert indexer.a2_lm.shape == ((BW_SPOT + 1) ** 2,)
    assert indexer.a2_lm.dtype == torch.complex128
    # the geometry normalizer is not built until the first normalized index
    assert indexer._denom_geo is None

    r_true = Rotation.from_euler(np.deg2rad(_SPOT_EULER_DEG[:3]))
    patterns = spot_master.project(detector_partial, r_true, rescale=True)
    indexer.index(patterns, normalize=True)

    denom = indexer._denom_geo
    assert denom is not None
    assert tuple(denom.shape) == (n, n, n)  # the local-norm field over the SO(3) grid
    assert torch.isfinite(denom).all()
    assert float(denom.min()) > 0.0  # bands keep the local norm off zero everywhere
    assert indexer._a_w is not None and float(indexer._a_w) > 0.0

    # a second (single-pattern) call reuses the SAME cached tensor -- not recomputed
    indexer.index(patterns[0], normalize=True)
    assert indexer._denom_geo is denom

    # the unnormalized path neither needs nor builds the normalizer
    idx_u = SphericalIndexer(spot_master, detector_partial, bandwidth=BW_SPOT, device="cpu")
    idx_u.index(patterns, normalize=False)
    assert idx_u._denom_geo is None


def test_normalized_batched_and_single_agree(spot_master, detector_partial):
    """``normalize=True`` works for a stack and a bare ``(H, W)``, and they agree."""
    r_true = Rotation.from_euler(np.deg2rad(_SPOT_EULER_DEG))
    patterns = spot_master.project(detector_partial, r_true, rescale=True)

    indexer = SphericalIndexer(spot_master, detector_partial, bandwidth=BW_SPOT, device="cpu")
    rot_b, q_b = indexer.index(patterns, normalize=True)  # batched
    assert rot_b.size == r_true.size
    assert tuple(q_b.shape) == (r_true.size,)

    for i in range(r_true.size):
        rot_s, q_s = indexer.index(patterns[i], normalize=True)  # single (H, W)
        assert rot_s.size == 1
        assert tuple(q_s.shape) == (1,)
        assert _disorientation_deg(rot_b[i], rot_s[0]) < 1e-6
        assert float(q_s[0]) == pytest.approx(float(q_b[i]), rel=1e-9)


def test_band_master_roundtrip_normalized(master, detector):
    """``normalize=True`` still passes the band-master round trip (the default flip is safe).

    On the near-uniform-norm band master both modes land in the same grid cell; the
    normalized quality is finite and in the correlation-coefficient range.
    """
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)

    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    rot_n, q_n = indexer.index(patterns, normalize=True)
    rot_u, _ = indexer.index(patterns, normalize=False)
    for i in range(r_true.size):
        assert _disorientation_deg(r_true[i], rot_n[i]) < TOL_DEG
        assert _disorientation_deg(rot_n[i], rot_u[i]) < TOL_DEG  # same cell, uniform norm
    assert torch.isfinite(q_n).all()
    assert float(q_n.min()) > 0.0
    assert float(q_n.max()) < 1.3


# --------------------------------------------------------------------------- #
# A1: return_surface -- expose the arg-maxed SO(3) score surface
# --------------------------------------------------------------------------- #
def test_return_surface_matches_argmax_and_feeds_show_correlation(master, detector):
    """``index(return_surface=True)`` returns the arg-maxed SO(3) surface as element 3.

    The per-pattern argmax of the surface must map (via ``grid_to_rotation``) to the
    exact ``Rotation`` ``index`` returns, its peak value equals ``quality``, and the
    surface feeds :func:`show_correlation` directly.
    """
    r_true = Rotation.from_euler(np.deg2rad(_KNOWN_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)  # (4, H, W)
    n = 2 * BANDWIDTH + 2

    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    # default is still a 2-tuple
    assert len(indexer.index(patterns)) == 2

    out = indexer.index(patterns, return_surface=True)
    assert len(out) == 3
    rotations, quality, surface = out
    assert tuple(surface.shape) == (r_true.size, n, n, n)

    for i in range(r_true.size):
        # the surface's peak value is exactly the returned quality
        assert float(surface[i].max()) == pytest.approx(float(quality[i]), rel=1e-9)
        # and its argmax cell maps to the exact returned rotation
        j, k, p = (int(x) for x in torch.unravel_index(torch.argmax(surface[i]), (n, n, n)))
        r_cell = grid_to_rotation(BANDWIDTH, j, k, p)
        assert _disorientation_deg(rotations[i], r_cell) < 1e-9

    # normalize=False returns the raw X surface as the third element
    _, q_u, surface_u = indexer.index(patterns, normalize=False, return_surface=True)
    assert tuple(surface_u.shape) == (r_true.size, n, n, n)
    assert float(surface_u[0].max()) == pytest.approx(float(q_u[0]), rel=1e-9)

    # the surface feeds show_correlation directly (skip cleanly without anyplotlib)
    pytest.importorskip("anyplotlib")
    from de_ebsd.viz import show_correlation

    fig = show_correlation(surface)  # a (Np, N, N, N) stack -> first pattern shown
    j0, k0, p0 = (int(x) for x in torch.unravel_index(torch.argmax(surface[0]), (n, n, n)))
    assert fig.peak_index == (j0, k0, p0)
    assert fig.to_html()


# --------------------------------------------------------------------------- #
# A2: the O(L^4) Wigner-d block is cached on the indexer and reused
# --------------------------------------------------------------------------- #
def test_wigner_d_cache_shape_and_identical_result(spot_master, detector_partial):
    """The Wigner-d block is precomputed once, has the right shape, and is reused.

    The indexed result must be identical whether the cache path is taken (default) or
    forced off (``wigner_d_cached=None`` recomputes), and ``index_scan`` must not
    rebuild the cached block across chunks.
    """
    n = 2 * BW_SPOT + 2
    indexer = SphericalIndexer(spot_master, detector_partial, bandwidth=BW_SPOT, device="cpu")
    assert tuple(indexer._wig_d.shape) == (n, BW_SPOT + 1, 2 * BW_SPOT + 1, 2 * BW_SPOT + 1)

    r_true = Rotation.from_euler(np.deg2rad(_SPOT_EULER_DEG[:3]))
    patterns = spot_master.project(detector_partial, r_true, rescale=True)

    rot_c, q_c, surf_c = indexer.index(patterns, return_surface=True)

    # force the no-cache path (recompute wigner_d every call) -- must be byte-identical
    cached = indexer._wig_d
    indexer._wig_d = None
    try:
        rot_n, q_n, surf_n = indexer.index(patterns, return_surface=True)
    finally:
        indexer._wig_d = cached
    assert torch.equal(surf_c, surf_n)
    assert torch.equal(q_c, q_n)
    assert np.array_equal(rot_c.data, rot_n.data)

    # index_scan reuses the SAME cached tensor across its mini-batches (not rebuilt)
    crystal = Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
    )
    wig_before = indexer._wig_d
    indexer.index_scan(patterns, shape=(1, 3), phase=crystal, batch_size=1)
    assert indexer._wig_d is wig_before


# --------------------------------------------------------------------------- #
# A5: energy_kev selects which master energy bin a_lm / a2_lm are built from
# --------------------------------------------------------------------------- #
def test_energy_kev_selects_master_bin(detector):
    """``energy_kev`` picks the master energy bin used to precompute a_lm and a2_lm."""
    npx = 80
    north = master_pattern_grid(npx, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    g = torch.Generator().manual_seed(11)
    poles = torch.randn(6, 3, generator=g, dtype=torch.float64)
    poles = poles / poles.norm(dim=-1, keepdim=True)
    amps0 = 0.5 + torch.rand(6, generator=g, dtype=torch.float64)
    amps1 = 2.0 + torch.rand(6, generator=g, dtype=torch.float64)  # markedly different bin
    bin0 = torch.stack(
        [band_field(north, poles, amps0, 0.11), band_field(south, poles, amps0, 0.11)], dim=0
    )
    bin1 = torch.stack(
        [band_field(north, poles, amps1, 0.11), band_field(south, poles, amps1, 0.11)], dim=0
    )
    data = torch.stack([bin0, bin1], dim=0)  # (2 energy, 2 hemi, n, n)
    multi = MasterPattern(data=data, energies_kev=np.array([10.0, 20.0]))

    idx_default = SphericalIndexer(multi, detector, bandwidth=BANDWIDTH, device="cpu")
    idx_low = SphericalIndexer(multi, detector, bandwidth=BANDWIDTH, device="cpu", energy_kev=10.0)
    assert idx_default.energy_kev is None
    assert idx_low.energy_kev == 10.0

    # default samples the highest bin (20 keV = bin 1); energy_kev=10 samples bin 0
    assert not torch.allclose(idx_default.a_lm, idx_low.a_lm)
    assert not torch.allclose(idx_default.a2_lm, idx_low.a2_lm)

    # the low-energy a_lm equals a plan built on a master that holds *only* bin 0
    only_bin0 = MasterPattern(data=bin0[None], energies_kev=np.array([10.0]))
    idx_bin0 = SphericalIndexer(only_bin0, detector, bandwidth=BANDWIDTH, device="cpu")
    assert torch.allclose(idx_low.a_lm, idx_bin0.a_lm)
    assert torch.allclose(idx_low.a2_lm, idx_bin0.a2_lm)
