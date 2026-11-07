"""Orientation refinement against kinematic bands: the same refine as the dynamical one.

Indexing can only return a dictionary entry, so its accuracy is capped by the
sampling step. Refinement must lift that cap, landing measurably closer to the true
orientation the synthetic data knows. :func:`refine_orientation` takes any master,
so these are SpyDE's ``test_ebsd_refine`` run through it with a
:class:`~de_ebsd.sim.KinematicMaster`.
"""

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation

from de_ebsd import KinematicMaster
from de_ebsd.index import (
    DictionaryIndexer,
    refine_orientation,
    remove_background,
    sample_orientations,
)
from kinematic_scan import kinematic_patterns, kinematic_scan, pattern_similarity, scan_detector

SHAPE = (32, 32)


@pytest.fixture(scope="module")
def master():
    return KinematicMaster()


@pytest.fixture(scope="module")
def detector():
    return scan_detector(SHAPE)


def euler(degrees_or_radians):
    return Rotation.from_euler(np.atleast_2d(degrees_or_radians))


def refine(patterns, master, detector, seeds, **kw):
    kw.setdefault("method", "adam")
    return refine_orientation(patterns, master, detector, seeds, device="cpu", **kw)


class TestTheObjective:
    def test_the_seed_score_is_the_pattern_ncc(self, master, detector):
        """Every detector pixel is in the field of view at this size, so the spherical
        NCC at the seed is the plain NCC of the pattern with the projected one."""
        truth = euler([[0.3, 0.6, 1.0], [1.2, 0.5, 0.4]])
        seed = euler([[0.35, 0.62, 1.05], [1.1, 0.55, 0.4]])
        patterns = kinematic_patterns(truth, SHAPE)
        _, score = refine(patterns, master, detector, seed, n_iterations=0)
        projected = kinematic_patterns(seed, SHAPE).reshape(2, -1).astype(np.float64)
        flat = patterns.reshape(2, -1).astype(np.float64)
        flat -= flat.mean(1, keepdims=True)
        projected -= projected.mean(1, keepdims=True)
        want = (flat * projected).sum(1) / np.linalg.norm(flat, axis=1)
        want /= np.linalg.norm(projected, axis=1)
        np.testing.assert_allclose(score.numpy(), want, atol=1e-5)


class TestRefinementImprovesOnIndexing:
    @pytest.mark.parametrize("method", ["adam", "lm"])
    def test_recovers_a_perturbed_orientation(self, master, detector, method):
        """Nudge known orientations a few degrees off and walk them back."""
        truth_euler = np.array([[0.3, 0.6, 1.0], [1.2, 0.5, 0.4]])
        truth = euler(truth_euler)
        start = euler(truth_euler + np.deg2rad(4.0))
        patterns = kinematic_patterns(truth, SHAPE)
        refined, _ = refine(patterns, master, detector, start, method=method)
        before = pattern_similarity(start, truth)
        after = pattern_similarity(refined, truth)
        assert (after > before).all(), f"{after} vs {before}"
        assert after.min() > 0.99, f"only reached {after}"

    def test_beats_a_coarse_dictionary(self, master):
        """Index a background-corrected scan against a coarse dictionary, then refine:
        refinement lands closer to the truth than the dictionary entry could."""
        patterns, truth, _ = kinematic_scan(nav=(4, 4), shape=SHAPE)
        clean = remove_background(patterns.reshape(-1, *SHAPE))
        indexer = DictionaryIndexer(
            master,
            scan_detector(SHAPE),
            rotations=sample_orientations(15.0),
            dynamic_background=True,
        )
        indexed, _ = indexer.index(clean)
        refined, _ = indexer.refine(clean, rotations=indexed)
        truth = euler(truth.reshape(-1, 3))
        before = pattern_similarity(indexed, truth)
        after = pattern_similarity(refined, truth)
        assert after.mean() > before.mean(), f"{after.mean():.4f} vs {before.mean():.4f}"

    def test_never_returns_a_worse_orientation(self, master, detector):
        """An absurd learning rate makes Adam diverge on purpose; the seed comes back."""
        truth = euler([[0.3, 0.6, 1.0]])
        patterns = kinematic_patterns(truth, SHAPE)
        refined, score, history = refine(
            patterns,
            master,
            detector,
            truth,
            n_iterations=40,
            learning_rate=5.0,
            return_history=True,
        )
        assert np.allclose(np.abs((refined.data * truth.data).sum(-1)), 1.0, atol=1e-6)
        assert (score >= history[0] - 1e-6).all()


class TestBatching:
    def test_each_pattern_gets_its_own_orientation(self, master, detector):
        """A broadcasting mistake would drive every position to one answer."""
        truth_euler = np.array([[0.2, 0.5, 0.9], [1.4, 0.8, 0.3], [2.6, 0.3, 1.2]])
        truth = euler(truth_euler)
        patterns = kinematic_patterns(truth, SHAPE)
        refined, _ = refine(patterns, master, detector, euler(truth_euler + np.deg2rad(3.0)))
        assert (pattern_similarity(refined, truth) > 0.98).all()
        assert len({tuple(np.round(q, 4)) for q in refined.data}) == 3

    def test_a_subset_refines_as_it_does_in_the_whole(self, master, detector):
        """Each pattern's optimization is its own, so a stack's chunk equals itself."""
        truth_euler = np.array([[0.2, 0.5, 0.9], [1.4, 0.8, 0.3], [2.6, 0.3, 1.2], [0.9, 0.6, 0.7]])
        patterns = kinematic_patterns(euler(truth_euler), SHAPE)
        start = euler(truth_euler + np.deg2rad(3.0))
        whole, _ = refine(patterns, master, detector, start, n_iterations=30)
        pieces, _ = refine(patterns[:2], master, detector, start[:2], n_iterations=30)
        np.testing.assert_allclose(whole.data[:2], pieces.data, atol=1e-5)

    def test_shape_mismatch_is_caught(self, master, detector):
        patterns = kinematic_patterns(euler([[0.3, 0.6, 1.0]]), SHAPE)
        with pytest.raises(ValueError, match="rotations length"):
            refine(patterns, master, detector, Rotation.identity(5), n_iterations=2)


class TestProgress:
    @pytest.mark.parametrize("method", ["adam", "lm"])
    def test_is_called_after_every_step(self, master, detector, method):
        """Inside the loop, not between stages, or an interface freezes for seconds."""
        truth = euler([[0.3, 0.6, 1.0]])
        seen = []
        refine(
            kinematic_patterns(truth, SHAPE),
            master,
            detector,
            truth,
            method=method,
            n_iterations=6,
            progress=lambda done, total: seen.append((done, total)),
        )
        assert seen == [(i, 6) for i in range(1, 7)]

    def test_a_raising_progress_stops_the_refinement(self, master, detector):
        truth = euler([[0.3, 0.6, 1.0]])

        def stop(done, total):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            refine(kinematic_patterns(truth, SHAPE), master, detector, truth, progress=stop)


def test_the_kinematic_master_moves_to_the_refine_device(detector):
    """refine_orientation reads the master's device and dtype, not a tensor of it."""
    master = KinematicMaster(dtype=torch.float64)
    truth = euler([[0.3, 0.6, 1.0]])
    _, score = refine(kinematic_patterns(truth, SHAPE), master, detector, truth, n_iterations=1)
    assert score.dtype == torch.float64
