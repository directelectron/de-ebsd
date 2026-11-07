"""The maps read off an indexed scan: crystal map, similarity, phase merge, ADP.

The similarity map gets the most attention: it measures agreement between
NEIGHBORS' ranked match lists, which exposes a confidently wrong match that a score
map cannot. Ported from SpyDE's ``test_ebsd_crystal_map``.
"""

import numpy as np
import pytest
import torch
from orix.crystal_map import Phase
from orix.quaternion import Rotation

from de_ebsd import KinematicMaster
from de_ebsd.index import (
    DictionaryIndexer,
    average_dot_product_map,
    merge_phases,
    orientation_similarity_map,
    remove_background,
    sample_orientations,
    to_crystal_map,
)
from de_ebsd.viz import ipf_colors
from kinematic_scan import kinematic_scan, scan_detector

SHAPE = (32, 32)
FCC = Phase(name="fcc", space_group=225)


@pytest.fixture(scope="module")
def scan():
    return kinematic_scan(nav=(8, 8), shape=SHAPE)


class TestCrystalMapFromEuler:
    def test_takes_the_scan_shape_from_the_angles(self, scan):
        _, euler, _ = scan
        xmap = to_crystal_map(euler, phase=FCC)
        assert xmap.size == 64 and xmap.shape == (8, 8)
        assert "quality" not in xmap.prop

    def test_is_the_map_of_the_same_rotations(self, scan):
        _, euler, _ = scan
        from_euler = to_crystal_map(euler, np.linspace(0, 1, 64), phase=FCC)
        rotations = Rotation.from_euler(euler.reshape(-1, 3))
        from_rotations = to_crystal_map(rotations, np.linspace(0, 1, 64), (8, 8), FCC)
        assert np.allclose(from_euler.rotations.data, from_rotations.rotations.data)
        assert from_euler.prop["quality"].shape == (64,)

    def test_flat_angles_need_a_shape(self, scan):
        _, euler, _ = scan
        with pytest.raises(ValueError, match="shape"):
            to_crystal_map(euler.reshape(-1, 3), phase=FCC)
        assert to_crystal_map(euler.reshape(-1, 3), shape=(8, 8), phase=FCC).shape == (8, 8)

    def test_needs_a_phase(self, scan):
        with pytest.raises(ValueError, match="phase"):
            to_crystal_map(scan[1])

    def test_the_step_scales_the_coordinates(self, scan):
        assert to_crystal_map(scan[1], phase=FCC, step=0.5).x.max() == pytest.approx(3.5)


class TestOrientationSimilarityMap:
    def test_uniform_indexing_scores_one(self):
        osm = orientation_similarity_map(np.tile([[3, 7, 1]], (16, 1)), (4, 4))
        np.testing.assert_allclose(osm, 1.0)

    def test_a_boundary_shows_as_low_agreement(self):
        grid = np.zeros((4, 4, 2), int)
        grid[:, :2] = [1, 2]
        grid[:, 2:] = [8, 9]
        osm = orientation_similarity_map(grid.reshape(16, 2), (4, 4))
        assert osm[:, 1].mean() < osm[:, 0].mean()
        assert osm[:, 2].mean() < osm[:, 3].mean()

    def test_partial_overlap_is_between_zero_and_one(self):
        grid = np.array([[[1, 2], [1, 9]], [[1, 2], [1, 2]]])
        osm = orientation_similarity_map(grid.reshape(4, 2), (2, 2))
        assert 0.0 < osm[0, 1] < 1.0

    def test_detects_a_confidently_wrong_position(self):
        """What a score map cannot show: one point indexed to something unrelated."""
        idx = np.tile([[3, 7, 1]], (25, 1))
        idx[12] = [40, 41, 42]
        osm = orientation_similarity_map(idx, (5, 5))
        assert osm[2, 2] == pytest.approx(0.0)
        assert osm[0, 0] > 0.9

    def test_rejects_a_mismatched_shape(self):
        with pytest.raises(ValueError, match="does not match"):
            orientation_similarity_map(np.zeros((10, 3), int), (4, 4))

    def test_requires_a_shape(self):
        with pytest.raises(ValueError, match="shape"):
            orientation_similarity_map(np.zeros((16, 3), int), None)

    def test_works_on_a_real_match(self, scan):
        patterns, _, _ = scan
        indexer = DictionaryIndexer(
            KinematicMaster(),
            scan_detector(SHAPE),
            rotations=sample_orientations(12.0),
            dynamic_background=True,
        )
        indices, _ = indexer.match(remove_background(patterns.reshape(-1, *SHAPE)), keep=5)
        osm = orientation_similarity_map(indices, (8, 8))
        assert osm.shape == (8, 8)
        assert (osm >= 0).all() and (osm <= 1).all()
        assert osm.std() > 0, "a grain boundary must show"


class TestMergePhases:
    def test_picks_the_higher_scoring_phase_per_position(self):
        a = np.tile([0.1, 0.2, 0.3], (4, 1))
        b = np.tile([1.0, 1.1, 1.2], (4, 1))
        euler, phase_id, best = merge_phases(
            [a, b], [np.array([0.9, 0.1, 0.9, 0.1]), np.array([0.2, 0.8, 0.2, 0.8])]
        )
        assert phase_id.tolist() == [0, 1, 0, 1]
        np.testing.assert_allclose(euler[0], a[0])
        np.testing.assert_allclose(euler[1], b[1])
        np.testing.assert_allclose(best, [0.9, 0.8, 0.9, 0.8])

    def test_takes_rotations_and_tensors(self):
        a, b = Rotation.identity(4), Rotation.from_euler(np.full((4, 3), 0.5))
        merged, phase_id, _ = merge_phases(
            [a, b], [torch.tensor([0.9, 0.1, 0.9, 0.1]), torch.tensor([0.2, 0.8, 0.2, 0.8])]
        )
        assert isinstance(merged, Rotation)
        assert np.allclose(merged[1].data, b[1].data)
        assert phase_id.tolist() == [0, 1, 0, 1]

    def test_reshapes_to_the_scan(self):
        euler, phase_id, best = merge_phases(
            [np.zeros((4, 3)), np.ones((4, 3))], [np.zeros(4), np.ones(4)], shape=(2, 2)
        )
        assert euler.shape == (2, 2, 3) and phase_id.shape == (2, 2) and best.shape == (2, 2)

    def test_mismatched_coverage_is_caught(self):
        with pytest.raises(ValueError, match="same positions"):
            merge_phases([np.zeros((4, 3)), np.zeros((5, 3))], [np.zeros(4), np.zeros(5)])


class TestAverageDotProductMap:
    def test_is_one_where_neighbors_are_identical(self):
        patterns = np.tile(np.random.default_rng(0).random((1, 1, 6, 6)), (3, 4, 1, 1))
        np.testing.assert_allclose(average_dot_product_map(patterns), 1.0, atol=1e-6)

    def test_dips_at_the_grain_boundary(self, scan):
        patterns, _, mask = scan
        adp = average_dot_product_map(remove_background(patterns)).numpy()
        boundary = np.zeros_like(mask)
        boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
        boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
        assert adp[boundary].mean() < adp[~boundary & ~mask].mean()

    def test_needs_a_scan(self):
        with pytest.raises(ValueError, match="scan"):
            average_dot_product_map(np.zeros((4, 6, 6)))

    def test_holds_the_lock(self):
        import threading

        lock = threading.RLock()
        entered = []

        class Counting:
            def __enter__(self):
                entered.append(1)
                return lock.__enter__()

            def __exit__(self, *exc):
                return lock.__exit__(*exc)

        average_dot_product_map(np.zeros((2, 2, 4, 4)), lock=Counting())
        assert entered == [1]


class TestEndToEnd:
    def test_index_then_color_recovers_the_grain_structure(self, scan):
        """Indexing and IPF coloring reproduce the two-grain layout the scan was made
        with: the closest headless proxy for "the map looks right"."""
        patterns, _, mask = scan
        indexer = DictionaryIndexer(
            KinematicMaster(),
            scan_detector(SHAPE),
            rotations=sample_orientations(10.0),
            dynamic_background=True,
        )
        rotations, quality = indexer.index(remove_background(patterns.reshape(-1, *SHAPE)))
        rgb = ipf_colors(to_crystal_map(rotations, quality, (8, 8), FCC))
        assert rgb.shape == (8, 8, 3)
        assert not np.allclose(rgb[mask].mean(0), rgb[~mask].mean(0), atol=0.05)
