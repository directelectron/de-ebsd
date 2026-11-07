"""Dictionary indexing against kinematic bands, and the one match it shares.

The test that matters is recovering known orientations: the synthetic scan knows
the exact Euler angles it was drawn from. The rest pins the chunking (a chunked
top-k must equal one unchunked product, or results depend on the chunk size) and,
against kikuchipy, the similarity itself. Ported from SpyDE's
``test_ebsd_indexing``; ``test_dictionary`` covers the dynamical master.
"""

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation

from de_ebsd import KinematicMaster
from de_ebsd.index import DictionaryIndexer, remove_background, sample_orientations
from kinematic_scan import detector_background, kinematic_patterns, kinematic_scan, scan_detector

SHAPE = (40, 40)


@pytest.fixture(scope="module")
def scan():
    return kinematic_scan(nav=(8, 8), shape=SHAPE)


@pytest.fixture(scope="module")
def exact(scan):
    """A dictionary holding exactly the scan's orientations."""
    _, euler, _ = scan
    rotations = Rotation.from_euler(euler.reshape(-1, 3))
    return DictionaryIndexer(KinematicMaster(), scan_detector(SHAPE), rotations=rotations)


@pytest.fixture(scope="module")
def euler_grid():
    """Indexers over Euler grids, built once per (step, background)."""
    cache = {}

    def get(step, dynamic_background=True):
        key = (step, dynamic_background)
        if key not in cache:
            cache[key] = DictionaryIndexer(
                KinematicMaster(),
                scan_detector(SHAPE),
                rotations=sample_orientations(step),
                dynamic_background=dynamic_background,
            )
        return cache[key]

    return get


def flat(patterns):
    return np.asarray(patterns).reshape(-1, *SHAPE)


class TestRecoversKnownOrientations:
    def test_exact_dictionary_recovers_every_orientation(self, scan, exact):
        """Put the true orientations IN the dictionary and find each one. Grain 2 is
        uniform, so several entries share a pattern: compare patterns, not indices."""
        patterns, euler, _ = scan
        indices, _ = exact.match(flat(patterns))
        dictionary = kinematic_patterns(exact.dictionary_rotations, SHAPE)
        for i, j in enumerate(indices[:, 0].tolist()):
            assert np.allclose(dictionary[j], dictionary[i], atol=1e-5), f"position {i}"

    def test_background_removal_is_what_makes_scores_near_one(self, scan, exact):
        """NCC is invariant to gain and offset but not to a spatial gradient, which is
        in every experimental pattern and in no simulated one. Removing it from the
        experimental side alone would leave the dictionary carrying low frequencies
        its counterpart lost, so the dictionary goes through the same filter."""
        patterns, euler, _ = scan
        raw = float(exact.index(flat(patterns))[1].min())
        filtered = DictionaryIndexer(
            KinematicMaster(),
            scan_detector(SHAPE),
            rotations=exact.dictionary_rotations,
            dynamic_background=True,
        )
        clean = float(filtered.index(remove_background(flat(patterns)))[1].min())
        assert clean > raw, f"background removal did not help ({clean} vs {raw})"
        assert clean > 0.99, f"an exact match still only scores {clean:.4f}"

    def test_a_finer_dictionary_matches_better(self, scan, euler_grid):
        """A realistic dictionary lands on a nearby sample, closer the finer it is."""
        patterns, _, _ = scan
        clean = remove_background(flat(patterns))
        scores = {step: float(euler_grid(step).index(clean)[1].mean()) for step in (15, 10, 5)}
        assert scores[5] > scores[10] > scores[15], scores
        assert scores[5] > 0.9, scores

    def test_the_two_grains_index_differently(self, scan, euler_grid):
        """Disjoint index sets for the two grains, and one entry for uniform grain 2."""
        patterns, _, mask = scan
        best = euler_grid(10, False).match(flat(patterns))[0][:, 0].numpy().reshape(mask.shape)
        assert set(best[mask].tolist()).isdisjoint(set(best[~mask].tolist()))
        assert len(set(best[mask].tolist())) == 1


class TestTheMatch:
    def test_chunked_equals_unchunked(self, scan, euler_grid, monkeypatch):
        """The best entry and the k best scores do not depend on the chunk size. The
        ranked list is compared on scores: cubic symmetry makes many entries identical,
        and which of a tie comes first is not meaningful."""
        import de_ebsd.index.dictionary as dictionary

        patterns, _, _ = scan
        indexer = euler_grid(12, False)
        whole = indexer.match(flat(patterns), keep=5)
        monkeypatch.setattr(dictionary, "_MAX_SCORE_ELEMS", indexer.n_dict * 3)
        pieces = indexer.match(flat(patterns), keep=5)
        np.testing.assert_array_equal(whole[0][:, 0], pieces[0][:, 0])
        np.testing.assert_allclose(whole[1], pieces[1], atol=1e-6)

    def test_scores_are_sorted_best_first(self, scan, euler_grid):
        _, scores = euler_grid(15, False).match(flat(scan[0]), keep=4)
        assert (np.diff(scores.numpy(), axis=1) <= 1e-6).all()

    def test_the_best_of_k_is_the_index(self, scan, euler_grid):
        indexer = euler_grid(15, False)
        indices, scores = indexer.match(flat(scan[0]), keep=3)
        rotations, quality = indexer.index(flat(scan[0]))
        np.testing.assert_allclose(scores[:, 0], quality, atol=1e-6)
        assert np.allclose(rotations.data, indexer.dictionary_rotations[indices[:, 0]].data)

    def test_keep_larger_than_the_dictionary_is_clamped(self, scan):
        _, euler, _ = scan
        rotations = Rotation.from_euler(euler.reshape(-1, 3)[:3])
        indexer = DictionaryIndexer(KinematicMaster(), scan_detector(SHAPE), rotations=rotations)
        indices, _ = indexer.match(flat(scan[0]), keep=99)
        assert indices.shape[1] == 3

    def test_progress_reports_completion(self, scan, exact, monkeypatch):
        import de_ebsd.index.dictionary as dictionary

        monkeypatch.setattr(dictionary, "_MAX_SCORE_ELEMS", exact.n_dict * 10)
        seen = []
        exact.match(flat(scan[0]), progress=lambda done, total: seen.append((done, total)))
        assert len(seen) > 1 and seen[-1] == (64, 64)

    def test_a_raising_progress_stops_the_match(self, scan, exact):
        def cancel(done, total):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            exact.match(flat(scan[0]), progress=cancel)


class TestNormalization:
    def test_invariant_to_gain_and_offset(self, scan, exact):
        patterns = flat(scan[0])
        base = exact.match(patterns)
        altered = exact.match(patterns.astype(np.float64) * 3.7 + 120.0)
        np.testing.assert_array_equal(base[0], altered[0])
        np.testing.assert_allclose(base[1], altered[1], atol=1e-4)

    def test_survives_a_detector_background(self, scan, exact):
        """An uncorrected pattern still indexes: background removal is an improvement,
        not a prerequisite."""
        _, euler, _ = scan
        raw = kinematic_patterns(Rotation.from_euler(euler.reshape(-1, 3)), SHAPE)
        with_background = raw / raw.max() + detector_background(SHAPE)
        assert float(exact.index(with_background)[1].min()) > 0.9


class TestFromPatterns:
    def test_matches_the_indexer_built_from_the_master(self, scan, exact):
        dictionary = kinematic_patterns(exact.dictionary_rotations, SHAPE)
        given = DictionaryIndexer.from_patterns(dictionary, exact.dictionary_rotations)
        a, b = exact.match(flat(scan[0])), given.match(flat(scan[0]))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], atol=1e-6)

    def test_takes_flattened_or_image_shaped_patterns(self, scan, exact):
        dictionary = kinematic_patterns(exact.dictionary_rotations, SHAPE)
        images = DictionaryIndexer.from_patterns(dictionary, exact.dictionary_rotations)
        rows = DictionaryIndexer.from_patterns(
            dictionary.reshape(len(dictionary), -1), exact.dictionary_rotations
        )
        np.testing.assert_array_equal(
            images.match(flat(scan[0]))[0], rows.match(flat(scan[0]).reshape(64, -1))[0]
        )

    def test_a_mismatched_detector_is_caught(self, scan, exact):
        small = kinematic_patterns(exact.dictionary_rotations, (20, 20))
        indexer = DictionaryIndexer.from_patterns(small, exact.dictionary_rotations)
        with pytest.raises(ValueError, match="does not match"):
            indexer.match(flat(scan[0]))

    def test_rotations_must_match_the_entries(self):
        with pytest.raises(ValueError, match="rotations"):
            DictionaryIndexer.from_patterns(np.zeros((3, 4, 4)), Rotation.identity(2))

    def test_has_no_master_to_refine(self):
        indexer = DictionaryIndexer.from_patterns(
            np.random.default_rng(0).random((2, 4, 4)), Rotation.identity(2)
        )
        with pytest.raises(ValueError, match="no master"):
            indexer.refine(np.zeros((1, 4, 4)))


class TestSampleOrientations:
    def test_the_euler_grid_covers_the_cubic_region(self):
        euler = sample_orientations(15.0).to_euler()
        assert euler[:, 0].max() <= np.deg2rad(360)
        assert euler[:, 1].max() <= np.deg2rad(90) + 1e-6

    def test_a_finer_step_gives_more(self):
        assert sample_orientations(10.0).size > sample_orientations(20.0).size

    def test_a_point_group_samples_its_fundamental_zone(self):
        from orix.quaternion.symmetry import Oh

        assert sample_orientations(10.0, Oh).size < sample_orientations(10.0).size


class TestKikuchipyParity:
    """The bar is kikuchipy's numbers, at the level of the similarity metric: our
    ``E @ D`` must reproduce its normalized cross-correlation."""

    @staticmethod
    def _reference(exp, dic):
        kp_indexing = pytest.importorskip("kikuchipy.indexing")
        p, d = len(exp), len(dic)
        metric = kp_indexing.NormalizedCrossCorrelationMetric(p, d)
        e = metric.prepare_experimental(exp.reshape(p, -1))
        dd = metric.prepare_dictionary(dic.reshape(d, -1))
        return np.asarray(metric.match(e, dd))

    @pytest.fixture(scope="class")
    def pair(self):
        """Random patterns, a few dictionary entries near-copies of experimental ones,
        so the ranking is decided by signal rather than by how ties break."""
        rng = np.random.default_rng(0)
        p, d, h, w = 37, 53, 12, 10
        exp = rng.normal(50, 12, (p, h, w)).astype(np.float32)
        dic = rng.normal(50, 12, (d, h, w)).astype(np.float32)
        dic[:8] = exp[:8] + rng.normal(0, 0.4, (8, h, w))
        return exp, DictionaryIndexer.from_patterns(dic, Rotation.identity(d), device="cpu"), dic

    def test_every_similarity_matches(self, pair):
        exp, indexer, dic = pair
        ref = self._reference(exp, dic)
        indices, scores = indexer.match(exp, keep=len(dic))
        mine = np.empty_like(ref)
        for p in range(len(exp)):
            mine[p, indices[p].numpy()] = scores[p].numpy()
        np.testing.assert_allclose(mine, ref, atol=1e-5)

    def test_the_same_pattern_wins(self, pair):
        exp, indexer, dic = pair
        ref = self._reference(exp, dic)
        indices, scores = indexer.match(exp)
        assert (indices[:, 0].numpy() == ref.argmax(1)).all()
        np.testing.assert_allclose(scores[:, 0], ref.max(1), atol=1e-5)

    @pytest.mark.skipif(
        not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        reason="no Apple-MPS device",
    )
    def test_mps_matches_the_reference_too(self, pair):
        exp, _, dic = pair
        ref = self._reference(exp, dic)
        indexer = DictionaryIndexer.from_patterns(dic, Rotation.identity(len(dic)), device="mps")
        indices, scores = indexer.match(exp)
        assert (indices[:, 0].cpu().numpy() == ref.argmax(1)).all()
        np.testing.assert_allclose(scores[:, 0].cpu(), ref.max(1), atol=1e-5)
