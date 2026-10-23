"""Scan indexing -> orix ``CrystalMap`` (I5b): the ecosystem-handoff output.

Validates the two orix bridges (:func:`de_ebsd.index.io.phase_from_crystal`,
:func:`de_ebsd.index.io.to_crystal_map`) and the scan wrapper
:meth:`de_ebsd.index.spherical.indexer.SphericalIndexer.index_scan`.

The synthetic master is the same great-circle **band** master used by
``test_indexer.py`` (a stand-in for real Kikuchi bands): it tiles the whole
sphere, so it has near-uniform local norm and is *not* itself point-group
symmetric -- the recovered orientation is therefore the exact ``R`` that was
projected (no symmetry-equivalent ambiguity), and a raw disorientation check is
correct. The attached :class:`~de_ebsd.sim.Crystal` (cubic Si, ``Fd-3m``) is
metadata that gives the ``CrystalMap`` its symmetry; it does not make the band
master symmetric. Small ``L = 16``, CPU, fp64 throughout -- exact-ish and fast.
"""

import math

import numpy as np
import pytest
import torch
from orix.crystal_map import CrystalMap, Phase
from orix.quaternion import Rotation

from band_master import band_field, make_band_master
from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.lambert import master_pattern_grid
from de_ebsd.crystal import Crystal
from de_ebsd.index.io import phase_from_crystal, to_crystal_map
from de_ebsd.index.spherical.indexer import SphericalIndexer

BANDWIDTH = 16
GRID_STEP_DEG = 360.0 / (2 * BANDWIDTH + 2)
TOL_DEG = 1.5 * GRID_STEP_DEG

# a (3, 4) grid of 12 distinct orientations (Bunge Euler, degrees), row-major
_SCAN_EULER_DEG = np.array(
    [
        [10.0, 20.0, 30.0],
        [45.0, 55.0, 15.0],
        [123.0, 47.0, 205.0],
        [200.0, 100.0, 300.0],
        [300.0, 80.0, 40.0],
        [15.0, 35.0, 95.0],
        [250.0, 60.0, 150.0],
        [88.0, 120.0, 260.0],
        [30.0, 40.0, 50.0],
        [70.0, 25.0, 110.0],
        [160.0, 70.0, 20.0],
        [220.0, 90.0, 330.0],
    ]
)
_SCAN_SHAPE = (3, 4)


def _disorientation_deg(r_a: Rotation, r_b: Rotation) -> float:
    """Angle (degrees) between two orix Rotations, no crystal symmetry applied."""
    return float(np.asarray((r_a * (~r_b)).angle).ravel()[0]) * 180.0 / math.pi


@pytest.fixture(scope="module")
def crystal() -> Crystal:
    """Cubic silicon (Fd-3m, space group 227) -- a non-trivial point group (m-3m)."""
    return Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
    )


@pytest.fixture(scope="module")
def master(crystal) -> MasterPattern:
    """A synthetic band master carrying the crystal in metadata (for auto-derivation)."""
    return make_band_master(npx=140, metadata={"crystal": crystal})


@pytest.fixture(scope="module")
def detector() -> EBSDDetector:
    """A realistic tilted detector with a generous field of view."""
    return EBSDDetector(shape=(90, 90), pattern_center=(0.5, 0.5, 0.20), sample_tilt=70.0, tilt=5.0)


@pytest.fixture(scope="module")
def scan_patterns(master, detector) -> torch.Tensor:
    """The 12 known orientations projected to a flat ``(12, H, W)`` pattern stack."""
    r_true = Rotation.from_euler(np.deg2rad(_SCAN_EULER_DEG))
    return master.project(detector, r_true, rescale=True)  # (12, H, W)


# --------------------------------------------------------------------------- #
# phase_from_crystal
# --------------------------------------------------------------------------- #
def test_phase_from_crystal_carries_symmetry(crystal):
    """The bridge returns an orix Phase with the right space group + a real point group."""
    phase = phase_from_crystal(crystal)
    assert isinstance(phase, Phase)
    assert phase.space_group.number == 227
    # orix derives the point group from the space group -- must be set + non-trivial
    assert phase.point_group is not None
    assert phase.point_group.name == "m-3m"
    assert phase.point_group.size > 1
    # lattice passed through (nm), atoms added from the asymmetric unit
    lat = phase.structure.lattice
    assert (lat.a, lat.b, lat.c) == pytest.approx((0.5431, 0.5431, 0.5431))
    assert (lat.alpha, lat.beta, lat.gamma) == pytest.approx((90.0, 90.0, 90.0))
    assert len(phase.structure) == 1  # one asymmetric-unit site (Si)


# --------------------------------------------------------------------------- #
# to_crystal_map
# --------------------------------------------------------------------------- #
def test_to_crystal_map_layout_and_roundtrip(crystal):
    """A known flat Rotation + quality + shape assembles with the right orix layout."""
    n = _SCAN_SHAPE[0] * _SCAN_SHAPE[1]
    rot = Rotation.from_euler(np.deg2rad(_SCAN_EULER_DEG))
    quality = np.arange(n, dtype=np.float64)  # tag each point by its flat index
    phase = phase_from_crystal(crystal)

    xmap = to_crystal_map(rot, quality, _SCAN_SHAPE, phase, step=0.5)

    assert isinstance(xmap, CrystalMap)
    assert xmap.shape == _SCAN_SHAPE
    assert xmap.size == n
    # rotations come back in the same row-major order
    assert np.allclose(xmap.rotations.data, rot.data)
    # quality lands in prop as a plain numpy array
    assert isinstance(xmap.prop["quality"], np.ndarray)
    assert np.allclose(xmap.prop["quality"], quality)
    # phase carries the symmetry
    assert xmap.phases[0].point_group.name == "m-3m"
    # single phase: every point phase_id == 0
    assert np.all(xmap.phase_id == 0)
    # step honoured: x = column index * step, y = row index * step
    assert np.isclose(_step(np.unique(xmap.x)), 0.5)
    assert np.isclose(_step(np.unique(xmap.y)), 0.5)

    # 2-D indexing: map[r, c] is the row-major flat index r*cols + c
    rows, cols = _SCAN_SHAPE
    for r in range(rows):
        for c in range(cols):
            flat = r * cols + c
            assert float(xmap[r, c].prop["quality"][0]) == flat
            assert _disorientation_deg(xmap[r, c].rotations[0], rot[flat]) < 1e-9


def test_to_crystal_map_accepts_crystal_and_torch_quality(crystal):
    """``to_crystal_map`` accepts a de_ebsd Crystal for phase and a torch quality tensor."""
    n = _SCAN_SHAPE[0] * _SCAN_SHAPE[1]
    rot = Rotation.from_euler(np.deg2rad(_SCAN_EULER_DEG))
    quality = torch.arange(n, dtype=torch.float64)

    xmap = to_crystal_map(rot, quality, _SCAN_SHAPE, crystal)  # Crystal, not Phase
    assert xmap.phases[0].point_group.name == "m-3m"
    assert isinstance(xmap.prop["quality"], np.ndarray)
    assert np.allclose(xmap.prop["quality"], np.arange(n))


def test_to_crystal_map_length_mismatch_raises(crystal):
    """A rotation/quality length that does not match rows*cols is a clear error."""
    rot = Rotation.from_euler(np.deg2rad(_SCAN_EULER_DEG))  # 12
    with pytest.raises(ValueError, match="expected rows.cols"):
        to_crystal_map(rot, np.arange(12.0), (3, 5), phase_from_crystal(crystal))


def _step(unique_sorted: np.ndarray) -> float:
    return float(unique_sorted[1] - unique_sorted[0])


# --------------------------------------------------------------------------- #
# index_scan
# --------------------------------------------------------------------------- #
def test_index_scan_recovers_grid_both_layouts(master, detector, scan_patterns):
    """A synthetic scan indexes back to a CrystalMap; every cell recovered, both layouts."""
    r_true = Rotation.from_euler(np.deg2rad(_SCAN_EULER_DEG))
    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")

    # layout (N, H, W) + shape
    xmap_flat = indexer.index_scan(scan_patterns, shape=_SCAN_SHAPE)
    # layout (rows, cols, H, W) -- shape inferred
    grid = scan_patterns.reshape(*_SCAN_SHAPE, *scan_patterns.shape[1:])
    xmap_grid = indexer.index_scan(grid)

    for xmap in (xmap_flat, xmap_grid):
        assert isinstance(xmap, CrystalMap)
        assert xmap.shape == _SCAN_SHAPE
        assert xmap.size == r_true.size
        assert np.isfinite(xmap.prop["quality"]).all()
        assert xmap.phases[0].point_group.name == "m-3m"  # auto-derived from metadata
        dis = [
            _disorientation_deg(r_true[i], xmap.rotations[i]) for i in range(r_true.size)
        ]
        assert max(dis) < TOL_DEG, f"worst cell {max(dis):.2f} deg > {TOL_DEG:.2f} deg"

    # both layouts give the identical map
    assert np.allclose(xmap_flat.rotations.data, xmap_grid.rotations.data)
    assert np.allclose(xmap_flat.prop["quality"], xmap_grid.prop["quality"])

    # 2-D indexing lines up with the row-major flat order
    rows, cols = _SCAN_SHAPE
    for r in range(rows):
        for c in range(cols):
            flat = r * cols + c
            assert (
                _disorientation_deg(xmap_flat[r, c].rotations[0], r_true[flat]) < TOL_DEG
            )


def test_index_scan_batch_chunking_matches_single_batch(master, detector, scan_patterns):
    """``batch_size`` only bounds memory: chunked and unchunked results are identical."""
    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    xmap_full = indexer.index_scan(scan_patterns, shape=_SCAN_SHAPE, batch_size=None)
    xmap_chunk = indexer.index_scan(scan_patterns, shape=_SCAN_SHAPE, batch_size=5)

    assert np.array_equal(xmap_full.rotations.data, xmap_chunk.rotations.data)
    assert np.array_equal(xmap_full.prop["quality"], xmap_chunk.prop["quality"])


def test_index_scan_accepts_cpu_numpy_array(master, detector, scan_patterns):
    """A plain CPU numpy scan indexes correctly with ``device='cpu'``.

    ``index_scan`` moves each mini-batch to ``self.device`` inside the loop rather than
    the whole stack up front (so a huge scan never lands wholesale in device memory);
    the logic is device-agnostic, so a CPU numpy array indexes identically to a tensor.
    """
    r_true = Rotation.from_euler(np.deg2rad(_SCAN_EULER_DEG))
    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")

    patterns_np = scan_patterns.cpu().numpy()  # plain CPU numpy array, not a tensor
    assert isinstance(patterns_np, np.ndarray)

    xmap = indexer.index_scan(patterns_np, shape=_SCAN_SHAPE, batch_size=5)
    assert isinstance(xmap, CrystalMap)
    assert xmap.shape == _SCAN_SHAPE
    dis = [_disorientation_deg(r_true[i], xmap.rotations[i]) for i in range(r_true.size)]
    assert max(dis) < TOL_DEG

    # identical to indexing the torch stack (device-agnostic per-chunk move)
    xmap_t = indexer.index_scan(scan_patterns, shape=_SCAN_SHAPE, batch_size=5)
    assert np.array_equal(xmap.rotations.data, xmap_t.rotations.data)


def test_index_scan_explicit_phase_paths(master, detector, scan_patterns, crystal):
    """Explicit ``phase`` accepts both an orix Phase and a de_ebsd Crystal."""
    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    phase = phase_from_crystal(crystal)

    xmap_p = indexer.index_scan(scan_patterns, shape=_SCAN_SHAPE, phase=phase)
    xmap_c = indexer.index_scan(scan_patterns, shape=_SCAN_SHAPE, phase=crystal)

    assert xmap_p.phases[0].point_group.name == "m-3m"
    assert xmap_c.phases[0].point_group.name == "m-3m"
    assert np.allclose(xmap_p.rotations.data, xmap_c.rotations.data)


def test_index_scan_requires_shape_for_flat_input(master, detector, scan_patterns):
    """``(N, H, W)`` input without ``shape`` is a clear error."""
    indexer = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    with pytest.raises(ValueError, match="shape=.rows, cols. is required"):
        indexer.index_scan(scan_patterns)


def test_index_scan_errors_when_no_phase_available(detector, scan_patterns):
    """No ``phase`` and no crystal in master metadata -> a clear, actionable error."""
    # a master with NO crystal metadata
    g = torch.Generator().manual_seed(7)
    poles = torch.randn(12, 3, generator=g, dtype=torch.float64)
    poles = poles / poles.norm(dim=-1, keepdim=True)
    amps = 0.5 + torch.rand(12, generator=g, dtype=torch.float64)
    npx = 140
    north = master_pattern_grid(npx, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    data = torch.stack(
        [band_field(north, poles, amps, 0.11), band_field(south, poles, amps, 0.11)],
        dim=0,
    )[None]
    bare = MasterPattern(data=data, energies_kev=np.array([20.0]))  # empty metadata

    indexer = SphericalIndexer(bare, detector, bandwidth=BANDWIDTH, device="cpu")
    with pytest.raises(ValueError, match="no phase given"):
        indexer.index_scan(scan_patterns, shape=_SCAN_SHAPE)
