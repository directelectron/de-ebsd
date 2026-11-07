"""Several phases indexed live, against kinematic masters as well as dynamical ones.

:class:`~de_ebsd.index.live.LiveIndexer` searches every phase's dictionary in one
product and the best score gives the phase. Here two kinematic masters, a cubic
and a hexagonal phase, each draw a few patterns at known orientations, on a
detector background, and the indexer must give every pattern its own phase and
orientation back. Ported from Autopilot's multi-phase ``Indexer`` and
``DynamicalIndexer`` tests (``tests/test_ebsd.py``, ``tests/test_dynamical_indexer_seam.py``),
whose wrapper this replaces.
"""

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation
from scipy.spatial.transform import Rotation as ScipyRotation

from de_ebsd import EBSDDetector, KinematicMaster, MasterPattern
from de_ebsd.core.rotation import misorientation_deg
from de_ebsd.crystal import Crystal
from de_ebsd.index.io import phase_from_crystal
from de_ebsd.index.live import (
    FITTED_REFINE_STEPS_DEG,
    LiveIndexer,
    dynamical_indexing_available,
    fitted_refine_steps_deg,
    fused_fit_available,
    lambert_masters,
)
from de_ebsd.index.live.common import as_frames, shape_in_aspect
from de_ebsd.sim.kinematic_bands import reflectors_from_phase

PER_PHASE = 6
SHAPE = (60, 80)


def _crystals():
    nickel = Crystal.from_parameters(a=0.352, space_group=225, atoms=[("Ni", (0, 0, 0))])
    titanium = Crystal.from_parameters(
        a=0.295, c=0.468, gamma=120, space_group=194, atoms=[("Ti", (1 / 3, 2 / 3, 0.25))]
    )
    return [nickel, titanium]


@pytest.fixture(scope="module")
def point_groups():
    return [phase_from_crystal(c).point_group for c in _crystals()]


@pytest.fixture(scope="module")
def masters():
    return [KinematicMaster(reflectors_from_phase(c, max_bands=30)) for c in _crystals()]


@pytest.fixture(scope="module")
def detector():
    return EBSDDetector(shape=SHAPE, pattern_center=(0.5, 0.45, 0.6), sample_tilt=70.0)


@pytest.fixture(scope="module")
def truth():
    """``PER_PHASE`` random orientations per phase, as matrices."""
    return [
        ScipyRotation.random(PER_PHASE, random_state=1 + number).as_matrix() for number in range(2)
    ]


@pytest.fixture(scope="module")
def frames(masters, detector, truth):
    """Each phase's patterns, phase 0's first, on a sloping detector background."""
    patterns = [
        m.project(detector, Rotation.from_matrix(g), rescale=False).numpy()
        for m, g in zip(masters, truth, strict=True)
    ]
    background = 0.2 + 0.1 * np.linspace(0.0, 1.0, SHAPE[1], dtype=np.float32)
    return (np.concatenate(patterns) + background).astype(np.float32)


@pytest.fixture(scope="module")
def indexer(masters, point_groups, detector):
    return LiveIndexer(
        masters,
        point_groups,
        detector,
        resolution=5.0,
        search_shape=32,
        refine_shape=64,
        device="cpu",
    )


@pytest.fixture(scope="module")
def indexed(indexer, frames):
    return indexer.index(frames)


class TestTwoPhases:
    def test_the_best_score_gives_each_pattern_its_own_phase(self, indexed):
        phase = indexed[0].numpy()
        assert (phase == np.repeat([0, 1], PER_PHASE)).all(), phase

    def test_known_orientations_come_back(self, indexed, truth, point_groups):
        """Within 0.15 degrees from a 5 degree dictionary (measured: 0.098 at worst)."""
        g = indexed[1].numpy()
        for number in range(2):
            rows = slice(number * PER_PHASE, (number + 1) * PER_PHASE)
            error = [
                misorientation_deg(found[None], true, point_groups[number])[0]
                for found, true in zip(g[rows], truth[number], strict=True)
            ]
            assert max(error) < 0.15, (number, error)

    def test_scores_are_high_on_clean_patterns(self, indexed):
        assert indexed[2].min() > 0.9, indexed[2]

    def test_a_phase_entry_is_counted_within_its_own_phase(self, indexer, indexed):
        phase, _, _, entry = indexed
        assert list(indexer.offsets) == [0, indexer.sizes[0]]
        local = indexer.phase_entries(phase, entry)
        assert (local >= 0).all()
        for number in range(2):
            rows = phase == number
            assert (local[rows] < indexer.sizes[number]).all()
            seeds = indexer.phase_rotations(number)[local[rows]]
            assert torch.equal(seeds, indexer.rotations[entry[rows]])
        on_host = indexer.phase_entries(phase.numpy(), entry.numpy())
        assert np.array_equal(on_host, local.numpy())

    def test_the_comparison_shapes_keep_the_detectors_aspect(self, indexer):
        assert indexer.search_shape == (24, 32)
        assert indexer.refine_shape == (48, 64)


def test_one_frame_indexes_as_a_stack_of_one(indexer, frames, indexed):
    phase, g, score, entry = indexer.index(frames[PER_PHASE])
    assert phase.shape == score.shape == entry.shape == (1,)
    assert int(phase[0]) == int(indexed[0][PER_PHASE])
    assert torch.allclose(g[0], indexed[1][PER_PHASE], atol=1e-5)


def test_torch_and_dlpack_frames_index_as_numpy_ones(indexer, frames, indexed):
    class OnlyDLPack:
        """An array that offers nothing but DLPack, as a CuPy one does."""

        def __init__(self, tensor):
            self._tensor = tensor

        def __dlpack__(self, **kwargs):
            return self._tensor.__dlpack__(**kwargs)

        def __dlpack_device__(self):
            return self._tensor.__dlpack_device__()

    tensor = torch.from_numpy(frames)
    for given in (tensor, OnlyDLPack(tensor)):
        phase, g, _, _ = indexer.index(given)
        assert torch.equal(phase, indexed[0])
        assert torch.allclose(g, indexed[1], atol=1e-6)


def test_frames_are_taken_where_they_are_without_a_copy():
    frames = np.zeros((2, 4, 4), np.uint16)
    assert np.shares_memory(as_frames(frames).numpy(), frames)
    tensor = torch.zeros(2, 4, 4)
    assert as_frames(tensor) is tensor


def test_a_shape_by_its_long_side_keeps_the_aspect():
    assert shape_in_aspect(32, (24, 30)) == (26, 32)
    assert shape_in_aspect(128, (480, 640)) == (96, 128)
    assert shape_in_aspect(16, (4, 400)) == (8, 16)  # never below 8
    assert shape_in_aspect((20, 30), (480, 640)) == (20, 30)


def test_a_kinematic_master_on_the_lambert_grid_samples_as_itself(masters):
    master = masters[1]
    lambert = master.to_master_pattern(200)
    assert lambert.data.shape == (1, 2, 401, 401)
    directions = torch.nn.functional.normalize(torch.randn(500, 3, generator=torch.manual_seed(0)))
    exact, stored = master.sample(directions), lambert.sample(directions)
    # bilinear between grid points 0.45 degrees apart, on bands a degree or two wide
    assert torch.allclose(stored, exact, atol=0.03 * float(exact.max()))


def test_kinematic_masters_join_dynamical_ones_on_their_grid(masters):
    dynamical = MasterPattern(torch.rand(1, 2, 31, 31), np.array([20.0]), {})
    lambert = lambert_masters([masters[0], dynamical])
    assert lambert[1] is dynamical
    assert lambert[0].npx == dynamical.npx == 15


def test_indexing_needs_a_phase(detector):
    with pytest.raises(ValueError, match="at least one phase"):
        LiveIndexer([], [], detector, device="cpu")


def test_the_capabilities_on_a_cpu():
    assert not dynamical_indexing_available("cpu")
    assert dynamical_indexing_available("cuda")
    assert not fused_fit_available("cpu")
    assert fitted_refine_steps_deg("cpu") == (*FITTED_REFINE_STEPS_DEG, 0.5)


# --- the calls an application makes, on the CPU (Autopilot's seam test) ----------------

FRAMES = 4
FRAME_SHAPE = (24, 30)


@pytest.fixture(scope="module")
def random_indexer():
    """A random master stands in for a simulated one: the calls are what is checked,
    not the indexing."""
    from de_ebsd import Setup

    generator = torch.Generator().manual_seed(0)
    master = MasterPattern(torch.rand(1, 2, 31, 31, generator=generator), np.array([20.0]), {})
    detector = Setup.default("SEM", shape=FRAME_SHAPE).detector
    point_group = phase_from_crystal(_crystals()[0]).point_group
    return LiveIndexer(
        [master],
        [point_group],
        detector,
        resolution=15.0,
        search_shape=32,
        refine_shape=128,
        refine_steps_deg=fitted_refine_steps_deg("cpu"),
        device="cpu",
    )


@pytest.fixture(scope="module")
def random_frames():
    return np.random.default_rng(0).random((FRAMES, *FRAME_SHAPE), dtype=np.float32)


def test_indexing_returns_a_row_per_frame(random_indexer, random_frames):
    phase, g, score, entry = random_indexer.index(random_frames, fit=True)
    assert phase.shape == entry.shape == score.shape == (FRAMES,)
    assert g.shape == (FRAMES, 3, 3)


def test_indexing_with_strain_returns_a_tensor_per_frame(random_indexer, random_frames):
    phase, g, strain, score, entry = random_indexer.index_with_strain(random_frames)
    assert g.shape == strain.shape == (FRAMES, 3, 3)
    assert phase.shape == score.shape == entry.shape == (FRAMES,)


def test_the_static_background_restarts_from_frames_blurred(random_indexer, random_frames):
    random_indexer.restart_static(random_frames, blur=0.1)
    assert random_indexer.index(random_frames)[2].shape == (FRAMES,)
    random_indexer.restart_static()


def test_strain_from_a_reference_returns_a_tensor_per_frame(random_indexer, random_frames):
    strain = random_indexer.strain_from_reference(random_frames, random_frames[0])[0]
    assert strain.shape == (FRAMES, 3, 3)


def test_a_frame_is_corrected_at_its_own_resolution(random_indexer, random_frames):
    assert random_indexer.corrected(random_frames[0]).shape == (1, *FRAME_SHAPE)


def test_frames_pool_to_the_refine_shape_for_a_center_search(random_indexer, random_frames):
    assert random_indexer.refine_shape == (102, 128)
    assert random_indexer.pooled(random_frames).shape == (FRAMES, 102, 128)
