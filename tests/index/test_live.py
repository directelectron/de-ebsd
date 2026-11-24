"""Live indexing: the fused kernels against the torch path they replace, and a
round trip through :class:`~de_ebsd.index.live.LiveIndexer`.

The round trip projects a point-group-invariant band master (the one
``test_dictionary.py`` builds) at known orientations and indexes the patterns
back: the search must land in the right cell of a coarse dictionary and the
refine must take it to well under that cell's size.
"""

import functools
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from orix.quaternion import Orientation, Rotation
from test_dictionary import _KNOWN_EULER_DEG, _symmetric_band_master

from de_ebsd import EBSDDetector
from de_ebsd.core.rotation import quaternion_to_matrix, rotation_vector_to_matrix
from de_ebsd.crystal import Crystal
from de_ebsd.index import _kernels
from de_ebsd.index.io import phase_from_crystal
from de_ebsd.index.live import LiveIndexer, retry_speckle
from de_ebsd.index.live import background as background_module
from de_ebsd.index.live import indexer as live_indexer
from de_ebsd.index.live.background import (
    LOG_POLYNOMIAL,
    MIN_FRAMES_FOR_STATIC,
    ScanBackground,
    polynomial_basis,
)
from de_ebsd.index.live.common import pick, pool_into
from de_ebsd.index.live.deformation import DEFORMATION_LINE_SEARCH, DEFORMATION_MAX_STEP
from de_ebsd.index.live.indexer import texture_of
from de_ebsd.index.neighbors import _operators, _pair_angles
from de_ebsd.index.preprocess import normalize_rows

# The band masters' patterns run from -1 to 1: not intensities to take the log of,
# so these indexers remove the dynamic background alone, as the static kind does
# until it has seen frames. The log-polynomial background has its own tests.
BandIndexer = functools.partial(LiveIndexer, background="static")

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
    return EBSDDetector(shape=(64, 64), pattern_center=(0.5, 0.4, 0.55), sample_tilt=70.0, tilt=0.0)


def _known():
    return Rotation.from_euler(np.radians(_KNOWN_EULER_DEG))


def test_round_trip_recovers_known_orientations(master, point_group, detector):
    """Patterns projected at known orientations come back within 0.3°, from a 10°
    dictionary: the refine, not the search, sets the accuracy."""
    rotations = _known()
    patterns = master.project(detector, rotations)
    indexer = BandIndexer(
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


def _small_indexer(master, point_group, detector, **kwargs):
    return BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), **kwargs
    )


def test_a_search_in_chunks_finds_what_one_product_finds(
    master, point_group, detector, monkeypatch
):
    """Scored a few entries at a time, with a ragged last chunk, the search picks
    the entries and scores that one product over the whole dictionary does."""
    indexer = _small_indexer(master, point_group, detector, device="cpu")
    search_rows, _ = indexer.prepare(master.project(detector, _known()))
    whole_score, whole_entry = (search_rows @ indexer.dictionary).max(1)
    monkeypatch.setattr(live_indexer, "SEARCH_CHUNK", 10)
    assert indexer.dictionary.shape[1] % 10 != 0
    _, _, score, entry = indexer.search(search_rows)
    assert torch.equal(entry, whole_entry)
    assert torch.allclose(score, whole_score)


def test_a_new_pattern_center_rebuilds_the_dictionary_in_place(master, point_group, detector):
    """A center change overwrites the dictionary where it is, rather than
    building a second one beside it, and gives what building it fresh would."""
    indexer = _small_indexer(master, point_group, detector, device="cpu")
    where = indexer.dictionary.data_ptr()
    moved = tuple(v + 0.02 for v in detector.pattern_center)
    indexer.set_pattern_center(moved)
    assert indexer.dictionary.data_ptr() == where
    fresh = _small_indexer(
        master, point_group, replace(detector, pattern_center=moved), device="cpu"
    )
    assert torch.allclose(indexer.dictionary, fresh.dictionary)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a float16 product needs CUDA")
def test_a_half_precision_dictionary_indexes_as_a_single_precision_one(
    master, point_group, detector
):
    """The dictionary in float16 is half the size and seeds the same entries: the
    search only picks seeds, which the refine then takes from."""
    master = master.to("cuda")
    patterns = master.project(detector, _known())
    found = {}
    for dtype in (torch.float32, torch.float16):
        indexer = _small_indexer(
            master,
            point_group,
            detector,
            refine_steps_deg=(5, 2.5, 1.2, 0.6, 0.3, 0.15),
            device="cuda",
            dtype=dtype,
        )
        assert indexer.dictionary.dtype == dtype
        found[dtype] = indexer.index(patterns)
    assert torch.equal(found[torch.float16][3], found[torch.float32][3])
    found_rotations = Rotation.from_matrix(found[torch.float16][1].cpu().numpy())
    error = Orientation(found_rotations, point_group).angle_with(
        Orientation(_known(), point_group), degrees=True
    )
    assert np.max(error) < 0.3, error


def test_refine_moves_a_seed_toward_its_orientation(master, point_group, detector):
    """A seed 2° off is moved to within a tenth of that."""
    rotations = _known()
    patterns = master.project(detector, rotations)
    indexer = BandIndexer(
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
    for start in (detector.pattern_center, tuple(v + 0.006 for v in detector.pattern_center)):
        indexer = BandIndexer(
            [master],
            [point_group],
            replace(detector, pattern_center=start),
            resolution=10.0,
            refine_shape=(64, 64),
            refine_steps_deg=(5, 2.5, 1.2, 0.6, 0.3, 0.15),
            device="cpu",
        )
        found.append(indexer.refine_pattern_center(patterns))
        assert indexer.detector.pattern_center == found[-1]
    assert np.allclose(found[0], found[1], atol=0.001), found
    assert np.allclose(found[1], detector.pattern_center, atol=0.005), found


@fused
def test_fused_projection_matches_the_master_projection(master, detector):
    texture = texture_of([master], "cuda")
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
    texture = texture_of([master], "cuda")
    g = quaternion_to_matrix(
        torch.as_tensor(Rotation.random(28).data, dtype=torch.float32, device="cuda")
    )
    bases = torch.zeros(28, dtype=torch.int64, device="cuda")
    directions = detector.direction_cosines(device="cuda")
    rows = normalize_rows(torch.randn(4, len(directions), device="cuda"))
    got = _kernels.score(texture, g, bases, directions, rows, 7)
    simulated = normalize_rows(_kernels.project(texture, g, bases, directions)).view(4, 7, -1)
    expected = (simulated * rows[:, None]).sum(-1).flatten()
    assert torch.allclose(got, expected, atol=1e-5)


def test_center_maps_move_the_directions_to_another_center(master, point_group, detector):
    """The refine's directions at one center, mapped and normalized, are exactly
    the directions at another."""
    indexer = BandIndexer([master], [point_group], detector, resolution=10.0, device="cpu")
    other = (0.507, 0.392, 0.561)
    (m,) = indexer.center_maps([other])
    moved = indexer.refine_directions @ m.T
    moved = moved / moved.norm(dim=-1, keepdim=True)
    expected = replace(
        detector, pattern_center=other, shape=indexer.refine_shape
    ).direction_cosines()
    assert torch.allclose(moved, expected, atol=1e-5)


def test_strain_is_refined_with_the_orientation(master, point_group, detector):
    """Patterns projected with a known trace-free strain and a seed 0.2° off: the
    deformation refine returns the strain and the orientation.

    The patterns come from the indexer's own projection, so this checks the fit and
    its conventions -- ``F = R U`` in the sample frame, the orientation that of the
    deformed lattice -- not how closely a master matches a camera; on a real scan
    see :data:`DEFORMATION_ITERATIONS`."""
    rotations = _known()
    indexer = BandIndexer(
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
    rows = normalize_rows(
        indexer.project(indexer.texture, deformed, phase, indexer.refine_directions)
    )
    off = rotations * Rotation.from_axes_angles([1, 2, 3], 0.2, degrees=True)
    seed = quaternion_to_matrix(torch.as_tensor(off.data, dtype=torch.float32))
    found, found_strain, score = indexer.refine_deformation(rows, phase, seed, iterations=5)
    error = Rotation.from_matrix(found.numpy()).angle_with(rotations, degrees=True)
    assert np.max(error) < 0.02, error
    assert torch.allclose(found_strain, strain, atol=2e-4), (found_strain - strain).abs().max()
    assert (score > 0.999).all(), score


def test_the_orientation_only_fit_refines_the_orientation_and_no_strain(
    master, point_group, detector
):
    """Unstrained patterns and a seed 0.3° off: fitted without the strains, the
    orientation comes back to within a few hundredths of a degree, the strain is
    zero, and :meth:`LiveIndexer.index` with *fit* runs it after the refine."""
    rotations = _known()
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    g = quaternion_to_matrix(torch.as_tensor(rotations.data, dtype=torch.float32))
    phase = torch.zeros(len(g), dtype=torch.int64)
    rows = normalize_rows(indexer.project(indexer.texture, g, phase, indexer.refine_directions))
    off = rotations * Rotation.from_axes_angles([1, 2, 3], 0.3, degrees=True)
    seed = quaternion_to_matrix(torch.as_tensor(off.data, dtype=torch.float32))
    found, strain, score = indexer.refine_deformation(rows, phase, seed, iterations=5, strain=False)
    error = Rotation.from_matrix(found.numpy()).angle_with(rotations, degrees=True)
    assert np.max(error) < 0.02, error
    assert strain.abs().max() < 1e-6
    assert (score > 0.999).all(), score


@fused
@pytest.mark.parametrize(
    ("background", "blur"), [("static", None), ("static", 0.05), ("log-polynomial", None)]
)
def test_the_fused_prepare_is_the_torch_one(master, point_group, detector, background, blur):
    """The frames' preparation in one launch (log-polynomial or static background,
    dynamic background, both shapes, normalized) against the torch path: the same
    rows to float32 rounding, with a plain and a blurred static background."""
    indexer = LiveIndexer(
        [master],
        [point_group],
        detector,
        resolution=10.0,
        refine_shape=(64, 64),
        device="cuda",
        background=background,
    )
    assert indexer.fused
    generator = torch.Generator(device="cuda").manual_seed(0)
    frames = 100 + 20 * torch.rand(40, 256, 256, device="cuda", generator=generator)
    indexer.restart_static(frames, blur=blur)
    assert indexer._fused_prepare(frames) is not None
    fused_search, fused_refine = indexer.prepare(frames)
    indexer.fused = False
    search, refine = indexer.prepare(frames)
    assert torch.allclose(fused_search, search, atol=1e-5), (fused_search - search).abs().max()
    assert torch.allclose(fused_refine, refine, atol=1e-5), (fused_refine - refine).abs().max()


@fused
def test_the_fused_stepped_refine_is_the_torch_one(master, point_group, detector):
    """The stepped refine in one launch (one block per pattern, every level) against
    the torch refine on the same seeds: the same orientations and scores."""
    rotations = _known()
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cuda"
    )
    assert indexer.fused
    g = quaternion_to_matrix(torch.as_tensor(rotations.data, dtype=torch.float32)).cuda()
    phase = torch.zeros(len(g), dtype=torch.int64, device="cuda")
    rows = normalize_rows(indexer.project(indexer.texture, g, phase, indexer.refine_directions))
    off = rotations * Rotation.from_axes_angles([1, 2, 3], 1.5, degrees=True)
    seed = quaternion_to_matrix(torch.as_tensor(off.data, dtype=torch.float32)).cuda()
    fused, fused_score = indexer.refine(rows, phase, seed)
    plain, plain_score = indexer.refine(rows, phase, seed, directions=indexer.refine_directions)
    difference = Rotation.from_matrix(fused.cpu().numpy()).angle_with(
        Rotation.from_matrix(plain.cpu().numpy()), degrees=True
    )
    assert np.max(difference) < 0.005, difference
    assert torch.allclose(fused_score, plain_score, atol=1e-4), fused_score - plain_score
    error = Rotation.from_matrix(fused.cpu().numpy()).angle_with(rotations, degrees=True)
    assert np.max(error) < 0.2, error


@fused
@pytest.mark.parametrize(("analytic", "tolerance"), [(False, 0.005), (True, 0.02)])
def test_the_fused_orientation_fit_is_the_torch_one(
    master, point_group, detector, monkeypatch, analytic, tolerance
):
    """The orientation-only fit on the fused kernels (one launch, one block per
    pattern) against the torch path, seeds 0.3° off and a ghost in the patterns.
    With central differences, the same orientations and scores to float32
    rounding; with the Jacobian from the master's gradient
    (:data:`~de_ebsd.index.live.deformation.ANALYTIC_JACOBIAN`), the same fit to a
    few hundredths of a degree, as it differs from a central difference."""
    from de_ebsd.index.live import deformation

    monkeypatch.setattr(deformation, "ANALYTIC_JACOBIAN", analytic)
    rotations = _known()
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cuda"
    )
    assert indexer.fused
    g = quaternion_to_matrix(torch.as_tensor(rotations.data, dtype=torch.float32)).cuda()
    phase = torch.zeros(len(g), dtype=torch.int64, device="cuda")
    patterns = normalize_rows(indexer.project(indexer.texture, g, phase, indexer.refine_directions))
    rows = normalize_rows(patterns - 0.5 * patterns.mean(0))
    ghost = normalize_rows(patterns.mean(0, keepdim=True))[0]
    off = rotations * Rotation.from_axes_angles([1, 2, 3], 0.3, degrees=True)
    seed = quaternion_to_matrix(torch.as_tensor(off.data, dtype=torch.float32)).cuda()
    fused, _, fused_score = indexer.refine_deformation(rows, phase, seed, ghost=ghost, strain=False)
    indexer.fused = False
    plain, _, plain_score = indexer.refine_deformation(rows, phase, seed, ghost=ghost, strain=False)
    difference = Rotation.from_matrix(fused.cpu().numpy()).angle_with(
        Rotation.from_matrix(plain.cpu().numpy()), degrees=True
    )
    assert np.max(difference) < tolerance, difference
    assert torch.allclose(fused_score, plain_score, atol=1e-4), fused_score - plain_score
    error = Rotation.from_matrix(fused.cpu().numpy()).angle_with(rotations, degrees=True)
    assert np.max(error) < 0.05, error


@fused
@pytest.mark.parametrize("moved_center", [False, True])
def test_the_fused_strain_fit_is_the_torch_one(master, point_group, detector, moved_center):
    """The strain fit on the fused kernels (one launch, one block per pattern, the
    Jacobian from the master's own slope) against the torch path, on patterns with a
    known strain, a ghost and seeds 0.2° off, at the indexer's pattern center and
    at a center of each pattern's own: both recover the strain and the
    orientation, and agree with each other to well within the strain's size."""
    from de_ebsd.index.live import deformation

    rotations = _known()
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cuda"
    )
    assert indexer.fused and deformation.STRAIN_SCHEDULE
    g = quaternion_to_matrix(torch.as_tensor(rotations.data, dtype=torch.float32)).cuda()
    rng = np.random.default_rng(3)
    e = rng.normal(0.0, 2e-3, (len(g), 3, 3))
    e = 0.5 * (e + np.swapaxes(e, -1, -2))
    e -= np.trace(e, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    strain = torch.as_tensor(e, dtype=torch.float32).cuda()
    deformed = g @ torch.linalg.inv(torch.eye(3, device="cuda") + strain)
    maps = None
    if moved_center:
        centers = np.array(detector.pattern_center) + rng.normal(0.0, 0.01, (len(g), 3))
        maps = indexer.center_maps(centers)
        deformed = deformed @ maps
    phase = torch.zeros(len(g), dtype=torch.int64, device="cuda")
    patterns = normalize_rows(
        indexer.project(indexer.texture, deformed, phase, indexer.refine_directions)
    )
    rows = normalize_rows(patterns - 0.5 * patterns.mean(0))
    ghost = normalize_rows(patterns.mean(0, keepdim=True))[0]
    off = rotations * Rotation.from_axes_angles([1, 2, 3], 0.2, degrees=True)
    seed = quaternion_to_matrix(torch.as_tensor(off.data, dtype=torch.float32)).cuda()
    options = {"ghost": ghost, "center_maps": maps, "iterations": 5}
    fused, fused_strain, fused_score = indexer.refine_deformation(rows, phase, seed, **options)
    indexer.fused = False
    plain, plain_strain, plain_score = indexer.refine_deformation(rows, phase, seed, **options)
    for found, found_strain in ((fused, fused_strain), (plain, plain_strain)):
        error = Rotation.from_matrix(found.cpu().numpy()).angle_with(rotations, degrees=True)
        assert np.max(error) < 0.02, error
        assert torch.allclose(found_strain, strain, atol=2e-4), (found_strain - strain).abs().max()
    assert torch.allclose(fused_strain, plain_strain, atol=1e-4), fused_strain - plain_strain
    assert torch.allclose(fused_score, plain_score, atol=1e-4), fused_score - plain_score


def test_the_static_ghost_bends_no_strain(master, point_group, detector):
    """Unstrained patterns each carrying the negative of the scan's mean pattern, as
    a frame divided by a static background made of the scan's own frames does: fitted
    with the :attr:`~LiveIndexer.ghost`, their strain is zero; fitted without, the
    ghost of the other grains is taken for strain."""
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    g = quaternion_to_matrix(torch.as_tensor(_known().data, dtype=torch.float32))
    phase = torch.zeros(len(g), dtype=torch.int64)
    patterns = normalize_rows(indexer.project(indexer.texture, g, phase, indexer.refine_directions))
    rows = normalize_rows(patterns - 0.5 * patterns.mean(0))
    indexer.background.static = torch.ones(indexer.refine_shape)
    indexer.add_fits(phase, g)
    _, with_ghost, score = indexer.refine_deformation(rows, phase, g, ghost=indexer.ghost)
    _, without, _ = indexer.refine_deformation(rows, phase, g)
    assert with_ghost.abs().max() < 1e-4, with_ghost.abs().max()
    assert without.abs().max() > 5 * with_ghost.abs().max(), without.abs().max()
    assert (score > 0.999).all(), score


def test_a_fit_cannot_run_away(master, point_group, detector):
    """A pattern of noise beside a real one: the noise's fit moves at most
    DEFORMATION_MAX_STEP times the line search's longest multiple an iteration,
    and the real pattern's fit is untouched."""
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    g = quaternion_to_matrix(torch.as_tensor(_known()[:2].data, dtype=torch.float32))
    phase = torch.zeros(2, dtype=torch.int64)
    rows = normalize_rows(indexer.project(indexer.texture, g, phase, indexer.refine_directions))
    noise = torch.randn(1, rows.shape[1], generator=torch.Generator().manual_seed(1))
    rows[1] = normalize_rows(noise)[0]
    _, strain, score = indexer.refine_deformation(rows, phase, g, iterations=10)
    assert strain[0].abs().max() < 1e-5 and score[0] > 0.999
    # strain is the symmetric part of eight numbers that each moved this far at most
    assert strain[1].abs().max() <= 10 * DEFORMATION_MAX_STEP * max(DEFORMATION_LINE_SEARCH)


def test_a_pattern_the_fit_cannot_move_keeps_its_seed(master, point_group, detector):
    """A simulation no step changes leaves the fit's normal matrix singular, as a
    blank frame did on a real camera: that pattern keeps its seed and no strain,
    and the batch is not lost to the exception."""
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    g = quaternion_to_matrix(torch.as_tensor(_known()[:2].data, dtype=torch.float32))
    phase = torch.zeros(2, dtype=torch.int64)
    rows = normalize_rows(indexer.project(indexer.texture, g, phase, indexer.refine_directions))
    flat = rows[0].clone()
    indexer.project = lambda texture, matrices, phases, directions: flat.expand(
        len(matrices), -1
    ).clone()
    found, strain, _ = indexer.refine_deformation(rows, phase, g, iterations=3)
    assert torch.allclose(found, g)
    assert torch.nan_to_num(strain, 0.0).abs().max() == 0


def test_the_line_search_closes_a_soft_patterns_fit_sooner(master, point_group, detector):
    """Strained patterns blurred softer than the master, as a camera's are: a plain
    Gauss-Newton step then falls short, and two iterations with the line search
    end several times nearer the converged fit than two without (a pattern whose
    plain fit has nearly converged may tie)."""
    from de_ebsd.index.preprocess import gaussian_blur

    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    g = quaternion_to_matrix(torch.as_tensor(_known().data, dtype=torch.float32))
    rng = np.random.default_rng(4)
    e = rng.normal(0.0, 3e-3, (len(g), 3, 3))
    e = 0.5 * (e + np.swapaxes(e, -1, -2))
    e -= np.trace(e, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    deformed = g @ torch.linalg.inv(torch.eye(3) + torch.as_tensor(e, dtype=torch.float32))
    phase = torch.zeros(len(g), dtype=torch.int64)
    sharp = indexer.project(indexer.texture, deformed, phase, indexer.refine_directions)
    rows = normalize_rows(gaussian_blur(sharp.view(-1, 64, 64), 2.0, 4.0).flatten(1))

    def fit(**options):
        return indexer.refine_deformation(rows, phase, g, **options)[1]

    converged = fit(iterations=20, line_search=())
    plain = (fit(iterations=2, line_search=()) - converged).abs().amax((1, 2))
    searched = (fit(iterations=2) - converged).abs().amax((1, 2))
    assert searched.median() < plain.median() / 3, (searched, plain)


def _camera_frames(master, detector, shape=(256, 256), device="cuda"):
    """The known orientations' patterns as a camera shows them: at *shape*, under a
    backscatter falloff, with counting noise."""
    camera = replace(detector, shape=shape)
    patterns = master.to(device).project(camera, _known(), rescale=True)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, shape[0], device=device),
        torch.linspace(-1, 1, shape[1], device=device),
        indexing="ij",
    )
    backscatter = torch.exp(-(x**2 + (y - 0.3) ** 2) / 0.8)
    generator = torch.Generator(device=device).manual_seed(0)
    frames = 400 * backscatter * (0.6 + 0.4 * patterns)
    return torch.poisson(frames.repeat(8, 1, 1), generator=generator)


def test_frames_at_the_refine_shape_are_pooled_as_they_are(master, point_group, detector):
    """Frames already at the refine shape are not copied, and :func:`pool_into`
    averages into a buffer exactly as :meth:`LiveIndexer.pooled` does."""
    indexer = _small_indexer(master, point_group, detector, device="cpu")
    frames = _camera_frames(master, detector, device="cpu")
    pooled = torch.empty(len(frames), *indexer.refine_shape)
    assert pool_into(frames, pooled) is pooled
    assert torch.equal(pooled, indexer.pooled(frames))
    assert indexer.pooled(pooled) is pooled


def test_index_writes_into_the_buffers_it_is_given(master, point_group, detector):
    indexer = _small_indexer(master, point_group, detector, device="cpu")
    patterns = master.project(detector, _known())
    expected = indexer.index(patterns)
    out = tuple(torch.full((len(patterns) + 3, *c.shape[1:]), -1, dtype=c.dtype) for c in expected)
    found = indexer.index(patterns, out=out)
    for column, target, want in zip(found, out, expected, strict=True):
        assert column.data_ptr() == target.data_ptr()
        assert torch.equal(column, want)


def _index_twice(indexer, frames, **options):
    """``index(frames, fit=True)`` from the same ghost twice: in the workspace,
    then through the allocating steps."""
    fit_sum = indexer.background._fit_sum.clone()
    fit_count = indexer.background._fit_count
    in_workspace = tuple(c.clone() for c in indexer.index(frames, fit=True, **options))
    indexer.background._fit_sum.copy_(fit_sum)
    indexer.background._fit_count = fit_count
    workspace, indexer.workspace = indexer.workspace, None
    allocating = indexer.index(frames, fit=True)
    indexer.workspace = workspace
    return in_workspace, allocating


@fused
def test_the_workspace_indexes_as_the_allocating_steps(master, point_group, detector):
    """Indexing in the workspace is the same arithmetic as the public steps, so the
    same phases, orientations, scores and entries to the bit."""
    indexer = _small_indexer(master, point_group, detector, device="cuda")
    assert indexer.workspace is not None
    frames = _camera_frames(master, detector)
    indexer.restart_static(frames)
    indexer.index(frames, fit=True)  # a ghost to fit with
    for found, expected in zip(*_index_twice(indexer, frames), strict=True):
        assert torch.equal(found, expected)


@fused
def test_pooled_frames_index_as_the_frames(master, point_group, detector):
    """Frames pooled once (:func:`pool_into`) and given to :meth:`observe` and
    :meth:`index` in place of the frames give the same static background and the
    same results to the bit: the refine shape is a whole-number bin of the
    frames, which averages the same however it is done."""
    frames = _camera_frames(master, detector)
    results = []
    for given in ("frames", "pooled"):
        indexer = _small_indexer(master, point_group, detector, device="cuda")
        if given == "pooled":
            pooled = torch.empty(len(frames), *indexer.refine_shape, device="cuda")
            frames = pool_into(frames, pooled)
        indexer.restart_static(frames)
        results.append((indexer.static.clone(), *indexer.index(frames, fit=True)))
    for from_frames, from_pooled in zip(*results, strict=True):
        assert torch.equal(from_frames, from_pooled)


@fused
def test_index_allocates_nothing_once_it_has_run(master, point_group, detector):
    """With buffers given for its results, a second :meth:`index` allocates nothing
    of its own on the device: every step writes into the workspace. Only the two
    column sums (the static background's and the ghost's) may ask torch for a
    reduction's staging buffer, kept so that both stay exact."""
    indexer = _small_indexer(master, point_group, detector, device="cuda")
    frames = _camera_frames(master, detector)
    indexer.restart_static(frames)
    first = indexer.index(frames, fit=True)
    out = tuple(torch.empty_like(column) for column in first)
    indexer.index(frames, fit=True, out=out)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    indexer.index(frames, fit=True, out=out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] - before <= 2


def test_uint16_frames_are_picked_as_they_are():
    frames = torch.randint(0, 2**16, (6, 4, 4), dtype=torch.int32).to(torch.uint16)
    rows = torch.tensor([4, 0, 4])
    assert torch.equal(pick(frames, rows).int(), frames.int()[rows])


def test_the_static_background_divides_out(master, point_group, detector):
    """Bands multiplied by the backscattered intensity, as on a real screen: divided
    by a static background of that intensity, they are as if it were even."""
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    bands = 1.0 + 0.1 * torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(0))
    y, x = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    backscatter = torch.exp(-(x**2 + (y - 0.3) ** 2) / 0.5)
    indexer.background.static = torch.ones(64, 64)
    even = indexer.corrected(bands)
    indexer.background.static = backscatter
    assert torch.allclose(indexer.corrected(backscatter * bands), even, atol=1e-5)


def test_a_search_starts_the_static_background_and_a_single_crystal_blurs_it(
    master, point_group, detector
):
    """A search's frames are the static background from a scan's first frame. One
    crystal's frames all carry its band, so their mean does; blurred, it does not,
    and there is no ghost."""
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    y, x = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    backscatter = torch.exp(-(x**2 + (y - 0.3) ** 2) / 2.0)
    band = torch.zeros(64, 64)
    band[:, 31:33] = 0.5
    frames = (backscatter * (1 + band)).expand(MIN_FRAMES_FOR_STATIC, 64, 64)
    # away from the top and bottom, where the blur meets the edge
    middle = (slice(16, 48), slice(31, 33))
    indexer.restart_static(frames)
    assert (indexer.static / backscatter - 1)[middle].min() > 0.45
    indexer.restart_static(frames, blur=0.05)
    assert (indexer.static / backscatter - 1)[middle].max() < 0.15
    indexer.add_fits(torch.zeros(1, dtype=torch.int64), torch.eye(3)[None])
    assert indexer.ghost is None


def test_strain_from_a_reference_pattern(master, point_group, detector):
    """Patterns of one orientation with known strains, against an unstrained
    pattern of it: the strain comes back relative to that reference."""
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    g = quaternion_to_matrix(torch.as_tensor(_known()[:1].data, dtype=torch.float32))
    rng = np.random.default_rng(5)
    e = rng.normal(0.0, 2e-3, (4, 3, 3))
    e = 0.5 * (e + np.swapaxes(e, -1, -2))
    e -= np.trace(e, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    strain = torch.as_tensor(e, dtype=torch.float32)
    deformed = torch.cat([g, g @ torch.linalg.inv(torch.eye(3) + strain)])
    phase = torch.zeros(len(deformed), dtype=torch.int64)
    frames = indexer.project(indexer.texture, deformed, phase, indexer.refine_directions)
    frames = frames.view(-1, *indexer.refine_shape)
    found, rotation, residual = indexer.strain_from_reference(frames[1:], frames[0])
    # the whole-pattern fit of a 64-pixel pattern: 6e-5 when this was written
    assert torch.allclose(found, strain, atol=2e-4), (found - strain).abs().max()
    assert 0.9 < float((found * strain).sum() / (strain * strain).sum()) < 1.1
    assert rotation.abs().max() < 2e-4


def test_strain_from_a_reference_pattern_a_few_degrees_turned(master, point_group, detector):
    """Points whose lattice is turned 3 degrees from the reference's, seeded
    from their orientations: the strain comes back, and the turn with it."""
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    g = quaternion_to_matrix(torch.as_tensor(_known()[:1].data, dtype=torch.float32))[0]
    rng = np.random.default_rng(6)
    e = rng.normal(0.0, 2e-3, (3, 3, 3))
    e = 0.5 * (e + np.swapaxes(e, -1, -2))
    e -= np.trace(e, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    strain = torch.as_tensor(e, dtype=torch.float32)
    axes = torch.as_tensor(rng.normal(size=(3, 3)), dtype=torch.float32)
    vectors = axes / axes.norm(dim=1, keepdim=True) * math.radians(3.0)
    turns = torch.linalg.matrix_exp(
        torch.stack(
            [torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]]) for x, y, z in vectors.tolist()]
        )
    )
    turned = g @ turns.transpose(-1, -2)  # the lattice turned by each turn
    deformed = torch.cat([g[None], turned @ torch.linalg.inv(torch.eye(3) + strain)])
    phase = torch.zeros(len(deformed), dtype=torch.int64)
    frames = indexer.project(indexer.texture, deformed, phase, indexer.refine_directions)
    frames = frames.view(-1, *indexer.refine_shape)
    found, rotation, _ = indexer.strain_from_reference(
        frames[1:],
        frames[0],
        rotations=turned.numpy(),
        reference_rotation=g.numpy(),
        symmetry=point_group,
    )
    assert torch.allclose(found, strain, atol=3e-4), (found - strain).abs().max()
    assert torch.allclose(rotation, vectors, atol=3e-4), (rotation - vectors).abs().max()


def test_a_center_error_shows_as_the_strain_strain_per_center_predicts(
    master, point_group, detector
):
    """Patterns projected at a center a few thousandths off and fitted at the
    indexer's: their strain is what the geometry alone predicts."""
    indexer = BandIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    g = quaternion_to_matrix(torch.as_tensor(_known().data, dtype=torch.float32))
    phase = torch.zeros(len(g), dtype=torch.int64)
    error = np.array([0.003, -0.002, 0.003])  # the indexer's center minus the true one
    true_center = np.asarray(detector.pattern_center) - error
    directions = indexer.directions_at(torch.as_tensor(true_center), indexer.refine_shape)
    rows = normalize_rows(indexer.project(indexer.texture, g, phase, directions))
    _, found, _ = indexer.refine_deformation(rows, phase, g, iterations=5)
    predicted = np.tensordot(error, indexer.strain_per_center(), 1)
    assert np.abs(predicted).max() > 2e-3  # the test means something
    assert np.allclose(found.numpy(), predicted, atol=2e-4), np.abs(found.numpy() - predicted).max()


def test_a_scans_center_plane_is_read_back_from_its_strain(master, point_group, detector):
    """A scan fitted at one wrong center, its center drifting with the beam, a
    real strain over one corner and noise: the robust plane finds the center and
    its drift; a plain least-squares plane is pulled toward the corner."""
    from de_ebsd.index.live.calibration import centers_from_strain

    indexer = _small_indexer(master, point_group, detector, device="cpu")
    response = indexer.strain_per_center()
    side = np.arange(12) * 30.0 - 165.0
    positions = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)  # um
    middle = np.array([0.5, 0.4, 0.55])
    drift = np.array([[6.5e-5, 0.0, 0.0], [0.0, -5.6e-5, 3.3e-5]])  # per um
    true = middle + positions @ drift
    fitted_at = middle + np.array([0.010, -0.008, 0.006])
    strain = np.einsum("nk,kij->nij", fitted_at - true, response)
    rng = np.random.default_rng(0)
    corner = (positions[:, 0] < -40) & (positions[:, 1] < -40)
    local = np.array([[3e-3, 1e-3, 0.0], [1e-3, -1e-3, 2e-3], [0.0, 2e-3, -2e-3]])
    strain[corner] += local
    noise = rng.normal(0.0, 3e-4, strain.shape)
    strain += 0.5 * (noise + np.swapaxes(noise, -1, -2))
    strain[[5, 77]] = np.nan  # refused fits

    plane = centers_from_strain(strain, positions, fitted_at, response)
    robust_error = np.abs(np.subtract(plane.center, middle)).max()
    assert robust_error < 1.5e-4, plane.center
    assert np.abs(plane.drift - drift).max() < 3e-6, plane.drift
    assert (plane.weights[corner] < 0.1).mean() > 0.9  # the corner was left out
    assert np.allclose(plane.at(positions), true, atol=5e-4)

    plain = centers_from_strain(strain, positions, fitted_at, response, robust=False)
    assert np.abs(np.subtract(plain.center, middle)).max() > 2 * robust_error
    known = centers_from_strain(strain, positions, fitted_at, response, use=~corner, robust=False)
    assert np.abs(np.subtract(known.center, middle)).max() < 3e-4
    fixed = centers_from_strain(strain, positions, fitted_at, response, drift=drift)
    assert np.abs(np.subtract(fixed.center, middle)).max() < 3e-4


# --- several candidates, and the retry -------------------------------------------


def _errors(g, rotations, point_group):
    found = Rotation.from_matrix(g.detach().cpu().numpy())
    return Orientation(found, point_group).angle_with(
        Orientation(rotations, point_group), degrees=True
    )


def _everything_rechecked(indexer):
    indexer.recheck_margin = indexer.recheck_score = float("inf")
    return indexer


def test_the_candidates_start_with_the_best_entry_and_are_distinct(master, point_group, detector):
    """The first candidate is the best entry the search finds, and no two kept ones
    of a pattern are within the separation of each other, symmetry reduced."""
    indexer = _small_indexer(master, point_group, detector, device="cpu", candidates=4)
    search_rows, _ = indexer.prepare(master.project(detector, _known()))
    _, _, _, best = indexer.search(search_rows)
    phase, g, score, entry, distinct = indexer.search_candidates(search_rows)
    assert torch.equal(entry[:, 0], best)
    assert (score[:, :-1] >= score[:, 1:]).all()
    assert distinct.all()
    operators = _operators(point_group)
    separation = live_indexer.CANDIDATE_SEPARATION * indexer.resolution
    for pattern in range(len(g)):
        for i in range(4):
            for j in range(i):
                angle = _pair_angles(
                    g[pattern, i].double().numpy(), g[pattern, j].double().numpy(), operators
                )
                assert angle > separation


def _wrong_basin(indexer, patterns):
    """The search led into the wrong basin for every pattern: a far-off dictionary
    entry is given a copy of the entry each pattern matches best, a little
    stronger, so the far one wins the search and the right one is the runner-up."""
    search_rows, _ = indexer.prepare(patterns)
    _, _, _, best = indexer.search(search_rows)
    far = (best + indexer.dictionary.shape[1] // 2) % indexer.dictionary.shape[1]
    indexer.dictionary[:, far] = 1.01 * indexer.dictionary[:, best]
    assert (
        _errors(indexer.rotations[far], _known()[: len(far)], indexer.point_groups[0]) > 10
    ).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=fused)])
def test_a_better_matching_candidate_replaces_a_wrong_basin(master, point_group, detector, device):
    """With one candidate a seed in the wrong basin is refined there; with two, the
    runner-ups from the right basin match its pattern better and win, with its
    own entry. On CUDA the workspace does it, with the fit."""
    master = master.to(device)
    patterns = master.project(detector, _known())
    steps = dict(refine_steps_deg=(5, 2.5, 1.2, 0.6, 0.3, 0.15))
    one = _small_indexer(master, point_group, detector, device=device, **steps)
    _wrong_basin(one, patterns)
    _, g, _, _ = one.index(patterns)
    assert (_errors(g, _known(), point_group) > 10).all()
    two = _everything_rechecked(
        _small_indexer(master, point_group, detector, device=device, candidates=2, **steps)
    )
    _wrong_basin(two, patterns)
    phase, g, score, entry = two.index(patterns, fit=device == "cuda")
    assert np.max(_errors(g, _known(), point_group)) < 0.5
    assert two.rechecked == len(patterns)
    assert (_errors(two.rotations[entry], _known(), point_group) < 10).all()


def test_index_with_strain_chooses_the_candidate_as_index_does(master, point_group, detector):
    patterns = master.project(detector, _known()[:2])
    indexer = _everything_rechecked(
        _small_indexer(master, point_group, detector, device="cpu", candidates=2)
    )
    _wrong_basin(indexer, patterns)
    _, g, _, _, _ = indexer.index_with_strain(patterns)
    # the strain fit of this band master's patterns is good to a couple of degrees;
    # the wrong basin's seed is 30 or more off
    assert np.max(_errors(g, _known()[:2], point_group)) < 3


def test_the_gate_leaves_a_clear_winner_alone(master, point_group, detector):
    """Clean patterns, whose best entry wins the search by a wide margin and refines
    to a high score, are not rechecked, and index as with one candidate."""
    patterns = master.project(detector, _known())
    one = _small_indexer(master, point_group, detector, device="cpu").index(patterns)
    indexer = _small_indexer(
        master, point_group, detector, device="cpu", candidates=4, recheck_margin=0.0
    )
    found = indexer.index(patterns)
    assert indexer.rechecked == 0
    for a, b in zip(found, one, strict=True):
        assert torch.equal(a, b)


@fused
def test_the_workspace_chooses_candidates_as_the_allocating_steps(master, point_group, detector):
    indexer = _everything_rechecked(
        _small_indexer(master, point_group, detector, device="cuda", candidates=4)
    )
    frames = _camera_frames(master, detector)
    indexer.restart_static(frames)
    indexer.index(frames, fit=True)
    in_workspace, allocating = _index_twice(indexer, frames)
    assert torch.equal(in_workspace[0], allocating[0])
    assert torch.equal(in_workspace[3], allocating[3])
    assert torch.allclose(in_workspace[1], allocating[1], atol=1e-4)
    assert torch.allclose(in_workspace[2], allocating[2], atol=1e-4)


@fused
def test_index_with_candidates_allocates_nothing_once_it_has_run(master, point_group, detector):
    """The candidates' search and recheck write into the indexer's
    :class:`~de_ebsd.index.live.workspace.CandidateBuffers`, made with it."""
    indexer = _everything_rechecked(
        _small_indexer(master, point_group, detector, device="cuda", candidates=4)
    )
    frames = _camera_frames(master, detector)
    indexer.restart_static(frames)
    first = indexer.index(frames, fit=True)
    out = tuple(torch.empty_like(column) for column in first)
    indexer.index(frames, fit=True, out=out)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    indexer.index(frames, fit=True, out=out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] - before <= 2


def test_retry_keeps_the_seed_that_matches_best(master, point_group, detector):
    """Seeds far off and near the truth: the near one is refined to it and chosen;
    passed over where it is not valid."""
    rotations = _known()
    patterns = master.project(detector, rotations)
    indexer = _small_indexer(master, point_group, detector, device="cpu")
    truth = quaternion_to_matrix(torch.as_tensor(rotations.data, dtype=torch.float32))
    near = truth @ rotation_vector_to_matrix(torch.tensor([[0.02, -0.01, 0.015]]))
    far = truth @ rotation_vector_to_matrix(torch.tensor([[0.0, 0.6, 0.0]]))
    seeds = torch.stack([far, near], 1)
    phases = torch.zeros(len(truth), 2, dtype=torch.int64)
    phase, g, score, chosen = indexer.retry(patterns, seeds, phases)
    assert (chosen == 1).all()
    assert np.max(_errors(g, rotations, point_group)) < 0.3
    valid = torch.tensor([[True, False]]).repeat(len(truth), 1)
    _, g, _, chosen = indexer.retry(patterns, seeds, phases, valid=valid)
    assert (chosen == 0).all()


def test_retry_speckle_puts_a_finished_maps_speckle_right(master, point_group, detector):
    """A 4 x 4 single crystal whose map has one point in a wrong orientation: the
    post-scan retry replaces that point, from its neighbors, and only that one."""
    rotation = _known()[1]
    patterns = master.project(detector, rotation).repeat(16, 1, 1)
    truth = quaternion_to_matrix(torch.as_tensor(rotation.data, dtype=torch.float32))
    g = truth.repeat(16, 1, 1)
    g[6] = g[6] @ rotation_vector_to_matrix(torch.tensor([[0.5, 0.2, 0.0]]))[0]
    indexer = _small_indexer(master, point_group, detector, device="cpu")
    phase = torch.zeros(16, dtype=torch.int64)
    retried = retry_speckle(
        indexer, patterns, phase, g, torch.zeros(16), torch.arange(16), shape=(4, 4)
    )
    assert retried.changed.tolist() == [i == 6 for i in range(16)]
    assert retried.tried == 1
    error = _errors(torch.as_tensor(retried.g), rotation, point_group)
    assert np.max(error) < 0.5
    assert retried.entry[6] in (2, 5, 7, 10)


def test_integer_frames_are_pooled_a_chunk_at_a_time_as_a_whole(
    master, point_group, detector, monkeypatch
):
    """A camera's integer frames, converted to float `POOL_CHUNK` at a time as
    they are pooled, average as the whole batch converted at once."""
    indexer = _small_indexer(master, point_group, detector, device="cpu")
    frames = _camera_frames(master, detector, device="cpu").to(torch.int32)
    monkeypatch.setattr(live_indexer, "POOL_CHUNK", 3)
    whole = indexer.pooled(frames.to(torch.float32))
    assert torch.equal(indexer.pooled(frames), whole)


def test_the_polynomial_basis_is_orthonormal_over_the_degree():
    """The log-polynomial basis is orthonormal and spans every polynomial of the
    degree: a surface of degree 2 is all projection, nothing left over."""
    basis = polynomial_basis((24, 32), 2)
    assert basis.shape == (6, 24 * 32)
    assert torch.allclose(basis @ basis.T, torch.eye(6), atol=1e-5)
    y, x = torch.meshgrid(torch.linspace(-1, 1, 24), torch.linspace(-1, 1, 32), indexing="ij")
    surface = (1.0 + 0.4 * x - 0.3 * y + 0.2 * x * y + 0.5 * x**2 - 0.1 * y**2).reshape(-1)
    left = surface - (surface @ basis.T) @ basis
    assert left.abs().max() < 1e-5


def test_the_log_polynomial_background_takes_off_a_smooth_envelope(master, detector, monkeypatch):
    """A pattern multiplied by a smooth envelope (a degree-2 surface in the log, as
    the screen's backscattered intensity is near enough) comes out of the
    log-polynomial background as the pattern alone: exactly with no offset in the
    log, and within a few percent with the offset, which trades that for a
    counted pattern's zeros."""
    patterns = 2.0 + master.project(detector, _known()).float()  # an intensity: positive
    y, x = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    envelope = torch.exp(0.8 * x - 0.5 * y - 0.6 * x**2 + 0.3 * x * y - 0.4 * y**2)
    background = ScanBackground((64, 64), torch.device("cpu"))
    assert background.kind == LOG_POLYNOMIAL
    plain = normalize_rows(background.remove(patterns).flatten(1))
    lit = normalize_rows(background.remove(patterns * envelope).flatten(1))
    assert ((plain * lit).sum(1) > 0.95).all()
    monkeypatch.setattr(background_module, "LOG_OFFSET", 0.0)
    plain = background.remove(patterns)
    lit = background.remove(patterns * envelope)
    assert torch.allclose(plain, lit, atol=1e-4), (plain - lit).abs().max()


def test_the_log_polynomial_background_needs_no_other_frame(master, point_group, detector):
    """With the log-polynomial background a pattern is prepared from itself alone:
    the frames seen before it change nothing, and there is no ghost."""
    indexer = LiveIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cpu"
    )
    assert indexer.background.kind == LOG_POLYNOMIAL
    frames = _camera_frames(master, detector, device="cpu")
    before = indexer.prepare(frames[:4])
    indexer.observe(frames[4:])
    indexer.index(frames, fit=True)
    after = indexer.prepare(frames[:4])
    for a, b in zip(before, after, strict=True):
        assert torch.equal(a, b)
    assert indexer.ghost is None


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=fused)])
def test_camera_frames_round_trip_through_the_log_polynomial_background(
    master, point_group, detector, device
):
    """A camera's frames (the known orientations under a backscatter falloff, with
    counting noise) indexed with the log-polynomial background, the default, come
    back within half a degree of the truth, on the torch path and the fused one."""
    indexer = LiveIndexer(
        [master],
        [point_group],
        detector,
        resolution=10.0,
        refine_shape=(64, 64),
        refine_steps_deg=(5, 2.5, 1.2, 0.6, 0.3, 0.15),
        device=device,
    )
    assert indexer.background.kind == LOG_POLYNOMIAL
    frames = _camera_frames(master, detector, device=device)
    phase, g, score, entry = indexer.index(frames)
    known = _known().data
    truth = Rotation(np.tile(known, (len(frames) // len(known), 1)))
    error = Orientation(Rotation.from_matrix(g.cpu().numpy()), point_group).angle_with(
        Orientation(truth, point_group), degrees=True
    )
    assert (phase == 0).all()
    assert np.max(error) < 0.5, error


def test_an_unknown_background_is_refused():
    with pytest.raises(ValueError, match="log-polynomial"):
        ScanBackground((8, 8), torch.device("cpu"), kind="mean")
