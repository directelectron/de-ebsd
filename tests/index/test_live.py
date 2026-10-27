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
from de_ebsd.index.live import LiveIndexer
from de_ebsd.index.live import indexer as live_indexer
from de_ebsd.index.live.background import MIN_FRAMES_FOR_STATIC
from de_ebsd.index.live.common import pick
from de_ebsd.index.live.deformation import DEFORMATION_LINE_SEARCH, DEFORMATION_MAX_STEP
from de_ebsd.index.live.indexer import texture_of
from de_ebsd.index.preprocess import normalize_rows

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


def _small_indexer(master, point_group, detector, **kwargs):
    return LiveIndexer(
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
    for start in (detector.pattern_center, tuple(v + 0.006 for v in detector.pattern_center)):
        indexer = LiveIndexer(
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
    indexer = LiveIndexer([master], [point_group], detector, resolution=10.0, device="cpu")
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
    indexer = LiveIndexer(
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
@pytest.mark.parametrize("blur", [None, 0.05])
def test_the_fused_prepare_is_the_torch_one(master, point_group, detector, blur):
    """The frames' preparation in one launch (static background, dynamic
    background, both shapes, normalized) against the torch path: the same rows to
    float32 rounding, with a plain and a blurred static background."""
    indexer = LiveIndexer(
        [master], [point_group], detector, resolution=10.0, refine_shape=(64, 64), device="cuda"
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
    indexer = LiveIndexer(
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
    indexer = LiveIndexer(
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
    indexer = LiveIndexer(
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
    indexer = LiveIndexer(
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
    indexer = LiveIndexer(
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
    indexer = LiveIndexer(
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

    indexer = LiveIndexer(
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


def test_uint16_frames_are_picked_as_they_are():
    frames = torch.randint(0, 2**16, (6, 4, 4), dtype=torch.int32).to(torch.uint16)
    rows = torch.tensor([4, 0, 4])
    assert torch.equal(pick(frames, rows).int(), frames.int()[rows])


def test_the_static_background_divides_out(master, point_group, detector):
    """Bands multiplied by the backscattered intensity, as on a real screen: divided
    by a static background of that intensity, they are as if it were even."""
    indexer = LiveIndexer(
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
    indexer = LiveIndexer(
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
    indexer = LiveIndexer(
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
    # 16-pixel regions of a 64-pixel pattern resolve a few 1e-4
    assert torch.allclose(found, strain, atol=8e-4), (found - strain).abs().max()
    assert 0.9 < float((found * strain).sum() / (strain * strain).sum()) < 1.1
    assert rotation.abs().max() < 6e-4
