"""The in-plane-factored search of :class:`~de_ebsd.index.live.LiveIndexer`.

The maths first, on the CPU: a turn about the screen normal is a shift of the polar
pattern, the two-angle grid covers the sphere for every point group, and the
masked normalized cross-correlation the search scores by frequency is the direct
one over the visible samples. Then the search against the flat one on patterns
drawn at known orientations, with two phases, and with each pattern at its own
center; then its memory; then, on CUDA, the fused kernels against the torch path.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from orix.quaternion import Orientation, Rotation, symmetry
from orix.sampling import sample_S2
from scipy.spatial.transform import Rotation as ScipyRotation
from test_dictionary import _KNOWN_EULER_DEG, _symmetric_band_master

from de_ebsd import EBSDDetector, KinematicMaster
from de_ebsd.core.lambert import sample_texture
from de_ebsd.core.rotation import misorientation_deg
from de_ebsd.crystal import Crystal
from de_ebsd.index import _kernels
from de_ebsd.index.io import phase_from_crystal
from de_ebsd.index.live import LiveIndexer
from de_ebsd.index.live import indexer as live_indexer
from de_ebsd.index.live.indexer import texture_of
from de_ebsd.index.live.inplane import (
    MIN_CHUNK,
    PolarGrid,
    in_plane_rotations,
    proper_rotations,
    rotation_taking,
    screen_frame,
    sphere_grid,
)
from de_ebsd.index.live.workspace import InplaneWorkspace
from de_ebsd.sim.kinematic_bands import reflectors_from_phase

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


def _inplane(master, point_group, detector, device="cpu", **kwargs):
    options = dict(resolution=5.0, refine_shape=(64, 64), device=device, search="inplane")
    return LiveIndexer([master], [point_group], detector, **(options | kwargs))


@pytest.fixture(scope="module")
def indexer(master, point_group, detector):
    return _inplane(master, point_group, detector)


def _known():
    return Rotation.from_euler(np.radians(_KNOWN_EULER_DEG))


def _random(count, seed):
    return Rotation.from_matrix(ScipyRotation.random(count, random_state=seed).as_matrix())


def _error_deg(g, rotations, point_group):
    found = Orientation(Rotation.from_matrix(g.cpu().numpy()), point_group)
    return found.angle_with(Orientation(rotations, point_group), degrees=True)


# --- the maths --------------------------------------------------------------------


def test_a_turn_about_the_screen_normal_turns_the_polar_pattern(master, detector):
    """The pattern of ``g0 @ R_n(theta)`` on the polar grid is ``g0``'s shifted by
    theta along the angle: ``g = g0(u) R_n(theta)`` factors exactly."""
    frame = torch.as_tensor(screen_frame(detector))
    grid = PolarGrid(frame, step=0.05, radius=1.2, angles=72, device="cpu")
    k, j = grid.shape
    texture = texture_of([master], "cpu")
    g0 = torch.as_tensor(ScipyRotation.random(3, random_state=4).as_matrix())
    shift = 7
    theta = torch.tensor(2 * math.pi * shift / j, dtype=torch.float64)
    turned = g0 @ in_plane_rotations(frame, theta)

    def draw(g):
        v = torch.einsum("mij,pj->mpi", g.float(), grid.directions)
        return sample_texture(texture, v).view(len(g), k, j)

    expected = draw(g0).roll(-shift, dims=-1)
    assert torch.allclose(draw(turned), expected, atol=1e-4 * float(expected.abs().max()))


def test_a_template_looks_along_its_crystal_direction():
    """``g0(u)`` takes the screen normal to the template's crystal direction."""
    normal = torch.tensor([0.1, -0.3, 0.9], dtype=torch.float64)
    normal = normal / normal.norm()
    directions = F.normalize(torch.randn(50, 3, dtype=torch.float64), dim=1)
    directions[0] = -normal
    g0 = rotation_taking(normal, directions)
    assert torch.allclose(g0 @ normal, directions, atol=1e-12)
    eye = torch.eye(3, dtype=torch.float64).expand(50, 3, 3)
    assert torch.allclose(g0 @ g0.transpose(1, 2), eye, atol=1e-12)
    assert torch.allclose(torch.linalg.det(g0), torch.ones(50, dtype=torch.float64))


@pytest.mark.parametrize("name", ["Oh", "Th", "D6h", "D3d", "D4h", "C4h", "D2h", "C2h", "Ci"])
def test_the_sphere_grid_covers_the_sphere_for_every_point_group(name):
    """Every direction has a symmetric equivalent as near a template as the whole
    icosahedral grid has a point to it, from a grid a little over 1/|S| of it."""
    point_group = getattr(symmetry, name)
    step = 8.0
    v = F.normalize(torch.as_tensor(np.random.default_rng(0).normal(size=(3000, 3))), dim=1)
    full = F.normalize(torch.as_tensor(sample_S2(step, method="icosahedral").data), dim=1)

    def covering(points):
        return float(torch.rad2deg(torch.arccos((v @ points.T).max(1).values.clamp(-1, 1))).max())

    grid = sphere_grid(point_group, step, 0.5 * step)
    operators = proper_rotations(point_group)
    images = torch.einsum("sij,uj->sui", operators, grid).reshape(-1, 3)
    assert covering(images) <= covering(full) + 1e-9
    assert len(grid) * len(operators) <= 1.6 * len(full)


def _all_scores(search, images):
    """``(T, U, N)``: every template's masked NCC at every in-plane angle, as the
    search scores them, without taking the best."""
    operands = search.operands(images)
    f = search.frequencies
    products = torch.stack([search.spectra @ operands[0], search.spectra @ operands[1]])
    scores = (search.angle_weights @ products.view(2 * f, -1)).view(search.thetas, -1, len(images))
    return scores * search.inverse_energy[:, :, None]


def test_the_masked_ncc_is_the_direct_one_over_the_visible_samples(master, point_group, detector):
    """With every frequency kept, the scores by frequency are the normalized
    cross-correlation, weighted by radius, of each pattern with each template
    turned, over the samples on the screen."""
    indexer = _inplane(
        master,
        point_group,
        detector,
        resolution=10.0,
        search_options=dict(theta_step=10.0, angles=216, frequencies=109),
    )
    search = indexer.inplane
    search.prepare_mask()
    rotations = _random(3, 1)
    patterns = master.project(detector, rotations).float()
    images = patterns + 0.05 * torch.randn(
        patterns.shape, generator=torch.Generator().manual_seed(0)
    )
    got = _all_scores(search, images)

    q = F.grid_sample(
        images[None],
        search.grid.sampling(detector.pattern_center, search.aspect),
        padding_mode="border",
        align_corners=False,
    )[0].double()
    visible = search.weights > 0
    w = search.grid.radii[:, None].expand_as(visible)[visible]
    templates = search._templates(0, search.template_count).double()
    step = search.angles // search.thetas
    for m in (0, 5, 17):
        t = templates.roll(-m * step, dims=-1)[:, visible]  # (U, visible)
        p = q[:, visible]  # (N, visible)
        p = p - (p * w).sum(1, keepdim=True) / w.sum()
        t = t - (t * w).sum(1, keepdim=True) / w.sum()
        direct = (t * w) @ p.T
        direct /= torch.outer((t * t * w).sum(1).sqrt(), (p * p * w).sum(1).sqrt())
        assert torch.allclose(got[m].double(), direct, atol=2e-4), (
            m,
            (got[m] - direct).abs().max(),
        )


# --- the search -------------------------------------------------------------------


def test_the_inplane_search_finds_what_the_flat_one_finds(master, point_group, detector, indexer):
    """On patterns at random orientations both searches seed within a few degrees
    of the truth, and the refine takes either seed to the same orientation."""
    rotations = _random(24, 3)
    patterns = master.project(detector, rotations)
    flat = LiveIndexer(
        [master],
        [point_group],
        detector,
        resolution=5.0,
        search_blur_deg=2.5,
        refine_shape=(64, 64),
        device="cpu",
    )
    seeds, found = [], []
    for each in (flat, indexer):
        search_rows, refine_rows = each.prepare(patterns)
        seeds.append(_error_deg(each.search(search_rows, refine_rows)[1], rotations, point_group))
        found.append(each.index(patterns)[1])
    assert max(seeds[0]) < 5.0 and max(seeds[1]) < 5.0, seeds
    assert np.median(seeds[1]) <= np.median(seeds[0]) + 0.5, seeds
    between = _error_deg(found[1], Rotation.from_matrix(found[0].numpy()), point_group)
    assert max(between) < 0.3, between


def test_the_round_trip_recovers_known_orientations(master, point_group, detector):
    """As the flat search's round trip: within 0.3 degrees from a 10 degree step."""
    rotations = _known()
    patterns = master.project(detector, rotations)
    indexer = _inplane(
        master,
        point_group,
        detector,
        resolution=10.0,
        refine_steps_deg=(5, 2.5, 1.2, 0.6, 0.3, 0.15),
    )
    phase, g, score, entry = indexer.index(patterns)
    assert (phase == 0).all()
    assert max(_error_deg(g, rotations, point_group)) < 0.3
    assert (score > 0.95).all(), score


def test_an_entry_stands_for_its_orientation(master, point_group, detector, indexer):
    """An entry's orientation is the search's within half an in-plane step (the
    sub-step moves it no further), and a phase's own entries number it there."""
    rotations = _random(10, 5)
    search_rows, refine_rows = indexer.prepare(master.project(detector, rotations))
    phase, g, _, entry = indexer.search(search_rows, refine_rows)
    seeds = indexer.entry_rotations(entry)
    between = _error_deg(seeds, Rotation.from_matrix(g.numpy()), point_group)
    assert max(between) <= 0.5 * indexer.inplane.theta_step + 1e-3
    local = indexer.phase_entries(phase, entry)
    assert torch.allclose(indexer.phase_rotations(0)[local], seeds)
    assert sum(indexer.sizes) == indexer.inplane.template_count * indexer.inplane.thetas


def test_a_search_in_chunks_finds_what_one_chunk_finds(master, point_group, detector):
    """Templates scored a few at a time, with a ragged last chunk, give the same
    seeds as all of them at once."""
    indexer = _inplane(master, point_group, detector)
    search_rows, refine_rows = indexer.prepare(master.project(detector, _random(8, 2)))
    whole = indexer.search(search_rows, refine_rows)
    indexer.inplane.chunk = 7
    assert indexer.inplane.template_count % 7
    chunked = indexer.search(search_rows, refine_rows)
    assert torch.equal(whole[3], chunked[3])
    assert torch.allclose(whole[2], chunked[2])


def test_the_pattern_center_is_refined_with_the_inplane_search(master, point_group, detector):
    """:meth:`LiveIndexer.refine_pattern_center` from a start 0.006 off comes back as
    the flat search's test does, and the search's mask follows the new center."""
    patterns = master.project(detector, _known())
    start = tuple(v + 0.006 for v in detector.pattern_center)
    indexer = _inplane(
        master,
        point_group,
        replace(detector, pattern_center=start),
        resolution=10.0,
        refine_steps_deg=(5, 2.5, 1.2, 0.6, 0.3, 0.15),
    )
    found = indexer.refine_pattern_center(patterns)
    assert np.allclose(found, detector.pattern_center, atol=0.005), found
    search_rows, refine_rows = indexer.prepare(patterns)
    indexer.search(search_rows, refine_rows)
    search = indexer.inplane
    expected = search.grid.visible(np.asarray(found)[None], search.aspect)
    assert torch.equal(search.weights > 0, expected)


# --- several phases ---------------------------------------------------------------


def _two_phases():
    nickel = Crystal.from_parameters(a=0.352, space_group=225, atoms=[("Ni", (0, 0, 0))])
    titanium = Crystal.from_parameters(
        a=0.295, c=0.468, gamma=120, space_group=194, atoms=[("Ti", (1 / 3, 2 / 3, 0.25))]
    )
    return [nickel, titanium]


def test_two_phases_are_told_apart_and_indexed():
    """Each phase's templates over its own symmetry, stacked: a cubic and a
    hexagonal phase's patterns, on a detector of another aspect, each get their
    own phase and orientation back."""
    crystals = _two_phases()
    point_groups = [phase_from_crystal(c).point_group for c in crystals]
    masters = [KinematicMaster(reflectors_from_phase(c, max_bands=30)) for c in crystals]
    detector = EBSDDetector(shape=(60, 80), pattern_center=(0.5, 0.45, 0.6), sample_tilt=70.0)
    truth = [_random(5, 11 + number) for number in range(2)]
    frames = (
        torch.cat(
            [m.project(detector, r, rescale=False) for m, r in zip(masters, truth, strict=True)]
        )
        + 0.2
    )
    indexer = LiveIndexer(
        masters,
        point_groups,
        detector,
        resolution=5.0,
        search_shape=32,
        refine_shape=64,
        device="cpu",
        search="inplane",
    )
    search = indexer.inplane
    assert search.templates_per_phase[1] > search.templates_per_phase[0]
    phase, g, score, entry = indexer.index(frames)
    assert (phase.numpy() == np.repeat([0, 1], 5)).all(), phase
    for number in range(2):
        rows = slice(5 * number, 5 * number + 5)
        error = [
            misorientation_deg(found[None], true, point_groups[number])[0]
            for found, true in zip(g[rows].numpy(), truth[number].to_matrix(), strict=True)
        ]
        assert max(error) < 0.15, (number, error)
    local = indexer.phase_entries(phase, entry)
    assert (local >= 0).all() and (local[5:] < indexer.sizes[1]).all()


# --- each pattern at its own center -----------------------------------------------


def _plane_scan(master, detector, side=4, amplitude=0.03):
    """Patterns of a ``side x side`` scan whose center is a plane over the beam
    position, ``amplitude`` off at the edges: ``(patterns, rotations, centers,
    corners)``."""
    y, x = np.mgrid[:side, :side]
    position = np.stack([x.ravel(), y.ravel()], 1) / (side - 1) * 2 - 1
    drift = amplitude * np.array([[-1.0, 0.0, 0.0], [0.0, 0.75, -0.5]])
    centers = np.asarray(detector.pattern_center) + position @ drift
    corners = (
        np.asarray(detector.pattern_center) + np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) @ drift
    )
    rotations = _random(side * side, 9)
    patterns = torch.cat(
        [
            master.project(replace(detector, pattern_center=tuple(c)), r)
            for c, r in zip(centers, rotations, strict=True)
        ]
    )
    return patterns, rotations, centers, corners


def test_the_mask_of_a_center_plane_is_its_four_corners(indexer, master, detector):
    """Visibility is linear in the center, so the samples on the screen at a
    plane's four corners are those on it at every point of the plane."""
    _, _, _, corners = _plane_scan(master, detector, side=2)
    y, x = np.mgrid[:9, :9]
    position = np.stack([x.ravel(), y.ravel()], 1) / 8 * 2 - 1
    drift = 0.03 * np.array([[-1.0, 0.0, 0.0], [0.0, 0.75, -0.5]])
    plane = np.asarray(detector.pattern_center) + position @ drift
    grid, aspect = indexer.inplane.grid, indexer.inplane.aspect
    assert torch.equal(grid.visible(corners, aspect), grid.visible(plane, aspect))
    assert grid.visible(corners, aspect).sum() < grid.visible(plane[:1], aspect).sum()


def test_each_pattern_is_searched_about_its_own_center(master, point_group, detector, indexer):
    """Patterns 0.03 off the indexer's center at a scan's edges: resampled about
    their own centers the seeds are as good as at one center; at the indexer's,
    they are worse. :meth:`LiveIndexer.index_with_strain` takes the centers."""
    patterns, rotations, centers, corners = _plane_scan(master, detector)
    search_rows, refine_rows = indexer.prepare(patterns)
    indexer.set_scan_centers(corners)
    own = _error_deg(indexer.search(search_rows, refine_rows, centers)[1], rotations, point_group)
    indexer.set_scan_centers(None)
    one = _error_deg(indexer.search(search_rows, refine_rows)[1], rotations, point_group)
    assert max(own) < 4.0, own
    assert np.mean(own) < np.mean(one), (own, one)

    # the strain fit on this band master sets the error, the flat search's alike
    # (measured: 1.07 degrees at worst from the flat search's seeds, 1.10 from these)
    _, g, _, _, _ = indexer.index_with_strain(patterns, pattern_centers=centers)
    assert max(_error_deg(g, rotations, point_group)) < 1.25
    assert indexer.inplane.scan_centers is None


# --- memory -----------------------------------------------------------------------


@pytest.mark.parametrize("megabytes", [128, 256])
def test_the_working_buffers_fit_the_memory_they_are_given(
    master, point_group, detector, megabytes
):
    """The chunk of templates is the most whose buffers, for a whole batch, fit the
    budget; nothing scales with patterns x templates x angles."""
    memory = megabytes * 2**20
    indexer = _inplane(
        master, point_group, detector, resolution=2.0, search_options=dict(memory=memory)
    )
    search = indexer.inplane
    k, j = search.grid.shape
    sizes = InplaneWorkspace.sizes(
        search.capacity,
        source_pixels=search.source_shape[0] * search.source_shape[1],
        polar_shape=(k, j),
        frequencies=search.frequencies,
        thetas=search.thetas,
        chunk=search.chunk,
        dtype=search.dtype,
    )
    assert MIN_CHUNK <= search.chunk < search.template_count
    assert sizes["total"] <= memory
    assert sizes["total"] + sizes["per_template"] > memory


def test_too_little_memory_is_said_and_the_search_still_runs(master, point_group, detector):
    with pytest.warns(RuntimeWarning, match="needs"):
        indexer = _inplane(master, point_group, detector, search_options=dict(memory=2**20))
    assert indexer.inplane.chunk == MIN_CHUNK


def test_the_flat_search_takes_no_inplane_options(master, point_group, detector):
    with pytest.raises(ValueError, match="inplane"):
        LiveIndexer([master], [point_group], detector, search_options=dict(memory=1))
    with pytest.raises(ValueError, match="flat"):
        LiveIndexer([master], [point_group], detector, search="spherical")


# --- the fused kernels ------------------------------------------------------------


def _camera_frames(master, detector, count=300, seed=0):
    rotations = _random(count, seed)
    patterns = master.to("cuda").project(detector, rotations).float().cuda()
    low, high = patterns.amin(), patterns.amax()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    expected = 20 * (0.5 + (patterns - low) / (high - low))
    return torch.poisson(expected, generator=generator), rotations


@fused
@pytest.mark.parametrize("own_centers", [False, True])
def test_the_fused_search_is_the_torch_one(master, point_group, detector, own_centers):
    """The polar resampling, the products, the scaled maximum and the sub-step on
    the fused kernels give the torch path's entries, scores and orientations."""
    indexer = _inplane(master.to("cuda"), point_group, detector, device="cuda", dtype=torch.float32)
    frames, _ = _camera_frames(master, detector)
    indexer.restart_static(frames)
    _, refine_rows = indexer.prepare(frames)
    centers = None
    if own_centers:
        rng = np.random.default_rng(0)
        centers = np.asarray(detector.pattern_center) + rng.uniform(-0.01, 0.01, (len(frames), 3))
        indexer.set_scan_centers(centers)
    search = indexer.inplane
    search.prepare_mask()
    fused_out = search.search(refine_rows, centers)
    torch_out = search._search_torch(refine_rows, centers)
    same = fused_out[3] == torch_out[3]
    assert same.float().mean() > 0.97
    assert torch.equal(fused_out[0], torch_out[0])
    assert torch.allclose(fused_out[2], torch_out[2], atol=1e-4)
    assert torch.allclose(fused_out[1][same], torch_out[1][same], atol=1e-4)


@fused
def test_half_precision_operands_search_as_single_precision_ones(master, point_group, detector):
    """float16 templates and patterns (tensor cores, float32 accumulation) seed
    the refine at the float32 search's orientations."""
    frames, rotations = _camera_frames(master, detector)
    seeds = []
    for dtype in (torch.float32, torch.float16):
        indexer = _inplane(master.to("cuda"), point_group, detector, device="cuda", dtype=dtype)
        indexer.restart_static(frames)
        search_rows, refine_rows = indexer.prepare(frames)
        seeds.append(indexer.search(search_rows, refine_rows))
    assert (seeds[0][3] == seeds[1][3]).float().mean() > 0.9
    between = _error_deg(seeds[1][1], Rotation.from_matrix(seeds[0][1].cpu().numpy()), point_group)
    assert np.percentile(between, 99) < 1.0


@fused
def test_the_inplane_index_allocates_nothing_once_it_has_run(master, point_group, detector):
    """As the flat search's fused path: with buffers given for the results, a
    second :meth:`LiveIndexer.index` allocates nothing on the device, and the
    search's buffers stay within its budget."""
    memory = 96 * 2**20
    indexer = _inplane(
        master.to("cuda"), point_group, detector, device="cuda", search_options=dict(memory=memory)
    )
    assert indexer.inplane.buffers.nbytes <= memory
    frames, _ = _camera_frames(master, detector)
    indexer.restart_static(frames)
    first = indexer.index(frames, fit=True)
    out = tuple(torch.empty_like(column) for column in first)
    indexer.index(frames, fit=True, out=out)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    indexer.index(frames, fit=True, out=out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] - before <= 2


@fused
def test_a_larger_batch_than_the_buffers_is_searched_in_parts(
    master, point_group, detector, monkeypatch
):
    monkeypatch.setattr(live_indexer, "FUSED_BATCH", 128)
    indexer = _inplane(master.to("cuda"), point_group, detector, device="cuda")
    assert indexer.inplane.capacity == 128
    frames, rotations = _camera_frames(master, detector)
    indexer.restart_static(frames)
    _, g, _, _ = indexer.index(frames, fit=True)
    assert np.median(_error_deg(g, rotations, point_group)) < 0.3
