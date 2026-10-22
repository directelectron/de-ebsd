"""Live indexing: the fused kernels against the torch path they replace, and a
round trip through :class:`~de_ebsd.index.live.LiveIndexer`.

The round trip projects a point-group-invariant band master (the one
``test_dictionary.py`` builds) at known orientations and indexes the patterns
back: the search must land in the right cell of a coarse dictionary and the
refine must take it to well under that cell's size.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from orix.quaternion import Orientation, Rotation
from test_dictionary import _KNOWN_EULER_DEG, _symmetric_band_master

from de_ebsd import EBSDDetector
from de_ebsd.core.rotation import quaternion_to_matrix
from de_ebsd.crystal import Crystal
from de_ebsd.index import _kernels
from de_ebsd.index.io import phase_from_crystal
from de_ebsd.index.live import LiveIndexer, _normalized_rows, _texture

fused = pytest.mark.skipif(
    not _kernels.available("cuda") or not torch.cuda.is_available(),
    reason="the fused kernels need CUDA and CuPy",
)


@pytest.fixture(scope="module")
def point_group():
    crystal = Crystal.from_parameters(a=0.352, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))])
    return phase_from_crystal(crystal).point_group


@pytest.fixture(scope="module")
def master(point_group):
    crystal = Crystal.from_parameters(a=0.352, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))])
    return _symmetric_band_master(crystal, point_group)


@pytest.fixture(scope="module")
def detector():
    return EBSDDetector(shape=(64, 64), pc=(0.5, 0.4, 0.55), sample_tilt=70.0, tilt=0.0)


def _known():
    return Rotation.from_euler(np.radians(_KNOWN_EULER_DEG))


def test_round_trip_recovers_known_orientations(master, point_group, detector):
    """Patterns projected at known orientations come back within 0.3°, from a 10°
    dictionary: the refine, not the search, sets the accuracy."""
    rotations = _known()
    patterns = master.project(detector, rotations)
    indexer = LiveIndexer(
        [master],
        [point_group],
        detector,
        resolution=10.0,
        search_shape=(32, 32),
        refine_shape=(64, 64),
        refine_steps_deg=(5, 2.5, 1.2, 0.6, 0.3, 0.15),
        device="cpu",
    )
    phase, g, score, entry = indexer.index(patterns)
    found = Rotation.from_matrix(g.numpy())
    error = Orientation(found, point_group).angle_with(
        Orientation(rotations, point_group), degrees=True
    )
    assert (phase == 0).all()
    assert (indexer.phase_of[entry] == phase).all()
    assert np.max(error) < 0.3, error
    assert (score > 0.95).all(), score


def test_refine_moves_a_seed_toward_its_orientation(master, point_group, detector):
    """A seed 2° off is moved to within a tenth of that."""
    rotations = _known()
    patterns = master.project(detector, rotations)
    indexer = LiveIndexer(
        [master],
        [point_group],
        detector,
        resolution=10.0,
        refine_shape=(64, 64),
        device="cpu",
    )
    _, rows = indexer.prepare(patterns)
    off = rotations * Rotation.from_axes_angles([1, 2, 3], 2, degrees=True)
    seed = quaternion_to_matrix(torch.as_tensor(off.data, dtype=torch.float32))
    g, _ = indexer.refine(rows, torch.zeros(len(seed), dtype=torch.int64), seed)
    error = Rotation.from_matrix(g.numpy()).angle_with(rotations, degrees=True)
    assert np.max(error) < 0.2, error


def test_pattern_center_is_refined_from_the_patterns(master, point_group, detector):
    """Patterns projected at a known center: indexed from that center and from
    one 0.006 off in every coordinate, the refine arrives at the same center.

    That center is a few thousandths from the true one here, because this band
    master is mostly smooth background, which the patterns lose in preparation
    and the simulations keep. On twin scans rendered from dynamical masters it
    stays within 0.0002 of the truth (``PATTERN_CENTER_STEPS``)."""
    patterns = master.project(detector, _known())
    found = []
    for start in (detector.pc, tuple(v + 0.006 for v in detector.pc)):
        indexer = LiveIndexer(
            [master],
            [point_group],
            replace(detector, pc=start),
            resolution=10.0,
            refine_shape=(64, 64),
            refine_steps_deg=(5, 2.5, 1.2, 0.6, 0.3, 0.15),
            device="cpu",
        )
        found.append(indexer.refine_pattern_center(patterns))
        assert indexer.detector.pc == found[-1]
    assert np.allclose(found[0], found[1], atol=0.001), found
    assert np.allclose(found[1], detector.pc, atol=0.005), found


@fused
def test_fused_projection_matches_the_master_projection(master, detector):
    texture = _texture([master], "cuda")
    q = torch.as_tensor(_known().data, dtype=torch.float32, device="cuda")
    got = _kernels.project(
        texture,
        quaternion_to_matrix(q),
        torch.zeros(len(q), dtype=torch.int64, device="cuda"),
        detector.direction_cosines(device="cuda"),
    )
    expected = master.to("cuda").project(detector, q, rescale=False).flatten(1)
    # fast-math atan: a few parts in a thousand of the master's range
    assert torch.allclose(got, expected, atol=3e-3 * float(expected.abs().max()))


@fused
def test_fused_score_is_the_normalized_cross_correlation(master, detector):
    texture = _texture([master], "cuda")
    g = quaternion_to_matrix(
        torch.as_tensor(Rotation.random(28).data, dtype=torch.float32, device="cuda")
    )
    bases = torch.zeros(28, dtype=torch.int64, device="cuda")
    directions = detector.direction_cosines(device="cuda")
    rows = _normalized_rows(torch.randn(4, len(directions), device="cuda"))
    got = _kernels.score(texture, g, bases, directions, rows, 7)
    simulated = _normalized_rows(_kernels.project(texture, g, bases, directions)).view(4, 7, -1)
    expected = (simulated * rows[:, None]).sum(-1).flatten()
    assert torch.allclose(got, expected, atol=1e-5)


def test_center_maps_move_the_directions_to_another_center(master, point_group, detector):
    """The refine's directions at one center, mapped and normalized, are exactly
    the directions at another."""
    indexer = LiveIndexer([master], [point_group], detector, resolution=10.0, device="cpu")
    other = (0.507, 0.392, 0.561)
    (m,) = indexer.center_maps([other])
    moved = indexer.refine_directions @ m.T
    moved = moved / moved.norm(dim=-1, keepdim=True)
    expected = replace(detector, pc=other, shape=indexer.refine_shape).direction_cosines()
    assert torch.allclose(moved, expected, atol=1e-5)


def test_strain_is_refined_with_the_orientation(master, point_group, detector):
    """Patterns projected with a known trace-free strain and a seed 0.2° off: the
    deformation refine returns the strain and the orientation.

    The patterns come from the indexer's own projection, so this checks the fit and
    its conventions -- ``F = R U`` in the sample frame, the orientation that of the
    deformed lattice -- not how closely a master matches a camera; on twin scans
    see :data:`DEFORMATION_ITERATIONS`."""
    rotations = _known()
    indexer = LiveIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    g = quaternion_to_matrix(torch.as_tensor(rotations.data, dtype=torch.float32))
    rng = np.random.default_rng(3)
    e = rng.normal(0.0, 2e-3, (len(g), 3, 3))
    e = 0.5 * (e + np.swapaxes(e, -1, -2))
    e -= np.trace(e, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    strain = torch.as_tensor(e, dtype=torch.float32)
    deformed = g @ torch.linalg.inv(torch.eye(3) + strain)
    phase = torch.zeros(len(g), dtype=torch.int64)
    rows = _normalized_rows(
        indexer._project(indexer.texture, deformed, phase, indexer.refine_directions)
    )
    off = rotations * Rotation.from_axes_angles([1, 2, 3], 0.2, degrees=True)
    seed = quaternion_to_matrix(torch.as_tensor(off.data, dtype=torch.float32))
    found, found_strain, score = indexer.refine_deformation(rows, phase, seed, iterations=5)
    error = Rotation.from_matrix(found.numpy()).angle_with(rotations, degrees=True)
    assert np.max(error) < 0.02, error
    assert torch.allclose(found_strain, strain, atol=2e-4), (found_strain - strain).abs().max()
    assert (score > 0.999).all(), score
