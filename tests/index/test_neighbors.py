"""An orientation map against its own neighborhood: misorientations, speckle and
the seeds to retry it with (:mod:`de_ebsd.index.neighbors`)."""

import numpy as np
import pytest
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import Oh

from de_ebsd.index.neighbors import (
    NEIGHBORS,
    neighbor_misorientation,
    neighbor_seeds,
    speckle_fraction,
    speckled,
)


def _random(count, seed):
    q = np.random.default_rng(seed).normal(size=(count, 4))
    return Rotation(q / np.linalg.norm(q, axis=1, keepdims=True))


def _grain(shape, seed=0, spread_deg=0.5):
    """One grain, ``(rows, cols)`` quaternions within *spread_deg* of one orientation."""
    rng = np.random.default_rng(seed)
    center = _random(1, seed + 100)
    count = shape[0] * shape[1]
    axes = rng.normal(size=(count, 3))
    jitter = Rotation.from_axes_angles(axes, rng.uniform(0, spread_deg, count), degrees=True)
    return (jitter * center).data.reshape(*shape, 4), center


def test_neighbor_misorientation_is_orix_disorientation():
    """Each point's angle to each neighbor is orix's, smallest over the point group,
    with NaN off the map's edges."""
    q = _random(12, 1).data.reshape(3, 4, 4)
    angles = neighbor_misorientation(q, Oh)
    for number, (dr, dc) in enumerate(NEIGHBORS):
        for r in range(3):
            for c in range(4):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < 3 and 0 <= cc < 4):
                    assert np.isnan(angles[r, c, number])
                    continue
                want = Orientation(q[r, c], Oh).angle_with(Orientation(q[rr, cc], Oh), degrees=True)
                assert angles[r, c, number] == pytest.approx(float(want[0]), abs=1e-4)


def test_matrices_and_quaternions_give_the_same_angles():
    q = _random(20, 2).data.reshape(4, 5, 4)
    matrices = Rotation(q.reshape(-1, 4)).to_matrix().reshape(4, 5, 3, 3)
    assert np.allclose(
        neighbor_misorientation(q, Oh), neighbor_misorientation(matrices, Oh), equal_nan=True
    )


def test_a_symmetric_equivalent_is_no_speckle():
    """A point turned by one of the crystal's own symmetries is the same orientation."""
    q, _ = _grain((5, 5))
    q[2, 2] = (Rotation.from_axes_angles([0, 0, 1], 90, degrees=True) * Rotation(q[2, 2])).data
    assert neighbor_misorientation(q, Oh)[2, 2].max() < 2.0
    assert speckle_fraction(q, Oh) == 0.0


def test_speckle_is_the_points_that_agree_with_no_neighbor():
    """Three planted wrong points are speckle, a corner one included, and a wrong
    pair agreeing with each other is not: each has a neighbor that agrees."""
    q, _ = _grain((8, 10))
    wrong = _random(5, 7).data
    q[3, 4], q[0, 0], q[7, 5] = wrong[:3]
    q[5, 8], q[5, 9] = wrong[3], wrong[3]
    mask = speckled(q, Oh)
    assert set(zip(*np.nonzero(mask), strict=True)) == {(3, 4), (0, 0), (7, 5)}
    assert speckle_fraction(q, Oh) == pytest.approx(3 / 80)


def test_a_neighbor_of_another_phase_never_agrees():
    q, _ = _grain((4, 4))
    phase = np.zeros((4, 4), dtype=int)
    phase[1, 1] = 1
    mask = speckled(q, [Oh, Oh], phase)
    assert mask[1, 1] and mask.sum() == 1


def test_the_seeds_are_the_point_then_its_neighbors():
    """Each speckled point's seeds: its own orientation first, then its four
    neighbors', with their phases and entries, and invalid off the edge."""
    q, _ = _grain((4, 5))
    q[0, 2] = _random(1, 3).data
    entry = np.arange(20).reshape(4, 5)
    phase = np.zeros((4, 5), dtype=int)
    seeds = neighbor_seeds(q, Oh, phase, entry)
    assert seeds.points.tolist() == [2]
    matrices = Rotation(q.reshape(-1, 4)).to_matrix().reshape(4, 5, 3, 3)
    assert np.allclose(seeds.rotations[0, 0], matrices[0, 2], atol=1e-6)
    assert seeds.valid[0].tolist() == [True, False, True, True, True]
    for number, (dr, dc) in enumerate(NEIGHBORS):
        if seeds.valid[0, 1 + number]:
            assert np.allclose(seeds.rotations[0, 1 + number], matrices[dr, 2 + dc], atol=1e-6)
            assert seeds.entries[0, 1 + number] == entry[dr, 2 + dc]
    assert seeds.entries[0, 0] == 2


def test_no_speckle_no_seeds():
    q, _ = _grain((3, 3))
    seeds = neighbor_seeds(q, Oh)
    assert len(seeds) == 0
    assert seeds.rotations.shape == (0, 5, 3, 3)
