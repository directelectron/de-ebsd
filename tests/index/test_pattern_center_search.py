"""Finding a pattern center from wherever it is (:mod:`de_ebsd.index.refine.pattern_center_search`).

Ported from SpyDE's ``test_phase_viewer`` (the search's inputs and its grid) and
Autopilot's ``test_ebsd`` (the two backgrounds, the Kikuchi check), plus a search
end to end on synthetic patterns at a known center.
"""

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation
from orix.quaternion.symmetry import Oh

from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.lambert import master_pattern_grid
from de_ebsd.index.live.background import ScanBackground
from de_ebsd.index.preprocess import bin_pattern
from de_ebsd.index.refine import pattern_center_search as search
from de_ebsd.index.refine.pattern_center_search import (
    MAX_SIDE,
    MIN_PATTERNS_FOR_STATIC,
    SEARCH_REFINE_STEPS,
    Calibration,
    calibration_positions,
    clean_patterns,
    grid_search,
    looks_like_kikuchi,
    search_pattern_center,
)
from de_ebsd.sim.kinematic_bands import KinematicMaster, reflectors_from_phase


@pytest.fixture(scope="module")
def cubic_master() -> MasterPattern:
    """The generic cubic bands as a Lambert master, so the m-3m fundamental zone
    bounds the dictionaries."""
    north = master_pattern_grid(100, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    bands = KinematicMaster(dtype=torch.float64)
    data = torch.stack([bands.sample(north), bands.sample(south)])[None].float()
    return MasterPattern(data, np.array([20.0]))


# --------------------------------------------------------------------------- #
# the inputs
# --------------------------------------------------------------------------- #
def test_positions_spread_over_the_scan_middle_first():
    positions = calibration_positions((40, 30))
    assert len(positions) == 25 and positions[0] == (20, 14)  # the grid point nearest (20, 15)
    assert {p[0] for p in positions} == {0, 10, 20, 29, 39}
    assert calibration_positions((3,)) == [(1,), (0,), (2,)]
    assert len(calibration_positions((100,))) == 25  # a line scan reads as many as a map
    assert calibration_positions(()) == [()]


def test_large_patterns_are_binned_to_the_same_screen():
    binned = bin_pattern(np.ones((2, 480, 640), np.float32), MAX_SIDE)
    assert binned.shape == (2, 120, 160)
    small = np.arange(12.0).reshape(3, 4)
    assert bin_pattern(small, MAX_SIDE) is small
    np.testing.assert_allclose(
        bin_pattern(np.arange(16.0).reshape(4, 4), 2), [[2.5, 4.5], [10.5, 12.5]]
    )


def test_the_static_background_is_what_the_patterns_share():
    """From :data:`MIN_PATTERNS_FOR_STATIC` patterns spread over a scan, not the 32 a
    live scan waits for; fewer have none."""
    profile = torch.linspace(1, 2, 16)[None, :] * torch.ones(16, 1)
    bands = torch.as_tensor(
        np.random.default_rng(0).normal(0, 0.1, (64, 16, 16)), dtype=torch.float32
    )
    background = ScanBackground((16, 16), torch.device("cpu"), min_frames=MIN_PATTERNS_FOR_STATIC)
    background.observe(profile + bands)
    assert (background.static - profile).abs().max() < 0.1
    few = ScanBackground((16, 16), torch.device("cpu"), min_frames=MIN_PATTERNS_FOR_STATIC)
    few.observe((profile + bands)[: MIN_PATTERNS_FOR_STATIC - 1])
    assert few.static is None


def test_a_single_crystals_static_background_is_blurred_and_keeps_its_bands():
    """The mean of one crystal's patterns carries its bands; divided by the plain
    mean they vanish, by the blurred mean they stay."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[:128, :128]
    lobe = np.exp(-((x - 64) ** 2 + (y - 48) ** 2) / (2 * 56.0**2))
    band = (np.abs(x - y) < 3).astype(float)
    frames = (lobe * (1 + 0.2 * band) * (1 + 0.01 * rng.normal(size=(12, 128, 128)))).astype(
        np.float32
    )
    plain = clean_patterns(frames)
    blurred = clean_patterns(frames, single_crystal=True)
    on, off = band.astype(bool), ~band.astype(bool)
    contrast = lambda t: float(t[:, on].mean() - t[:, off].mean())  # noqa: E731
    assert abs(contrast(plain)) < 1e-3  # measured 3e-9
    assert contrast(blurred) > 0.1  # measured 0.15


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
def _peak(center, width=0.06):
    def score(pc):
        return float(np.exp(-np.sum((np.asarray(pc) - center) ** 2) / (2 * width**2)))

    return score


def test_a_center_far_from_the_start_is_found_to_a_quarter_step():
    for center in ((0.42, 0.213, 0.499), (0.5, 0.3, 0.6), (0.47, 0.71, 0.83)):
        best = grid_search(_peak(np.array(center)), start_x=0.5)[0]
        assert np.abs(np.subtract(best, center)).max() <= SEARCH_REFINE_STEPS[-1], center


def test_each_center_is_scored_once():
    seen = []
    score = _peak(np.array((0.5, 0.3, 0.6)))
    grid_search(lambda pc: seen.append(pc) or score(pc), start_x=0.5)
    assert len(seen) == len(set(seen))


def test_the_best_fit_of_the_candidates_is_kept(monkeypatch, cubic_master):
    fits = {}

    def solve(patterns, master, detector, point_group, **kw):
        pc = detector.pattern_center
        fits[pc] = fit = 0.4 if pc[1] < 0.35 else 0.1
        return Calibration(pc, None, fit, 0.0, list(kw["positions"]))

    monkeypatch.setattr(search, "solve_pattern_center", solve)
    monkeypatch.setattr(
        search, "grid_search", lambda score, start_x, **kw: [(0.5, 0.5, 0.55), (0.5, 0.3, 0.6)]
    )
    monkeypatch.setattr(search, "_lead_scorer", lambda *args: lambda pc: 0.0)
    monkeypatch.setattr(search, "sample_orientations", lambda *args: None)
    found = search_pattern_center(
        np.ones((9, 8, 8), np.float32),
        cubic_master,
        EBSDDetector(shape=(8, 8)),
        Oh,
        single_crystal=False,
    )
    assert found.pattern_center == (0.5, 0.3, 0.6) and found.fit == 0.4 and len(fits) == 2


# --------------------------------------------------------------------------- #
# the two backgrounds
# --------------------------------------------------------------------------- #
def _fake_searches(monkeypatch, fits):
    """Each background's search finds *fits* [blurred]. What ran is returned."""
    ran = []

    def fake(patterns, master, detector, point_group, device, positions, progress, blurred, **_):
        ran.append(blurred)
        return Calibration((0.5, 0.3, 0.6), None, fits[blurred], 0.0, [], blurred)

    monkeypatch.setattr(search, "_search", fake)
    return ran


@pytest.mark.parametrize(
    "single_crystal, ran, kept",
    [(None, [False, True], True), (False, [False], False), (True, [True], True)],
)
def test_the_search_tries_the_blurred_background_unless_told_which(
    monkeypatch, single_crystal, ran, kept
):
    told = []
    done = _fake_searches(monkeypatch, {False: 0.3, True: 0.69})
    found = search_pattern_center(
        None, None, EBSDDetector(), None, searching=told.append, single_crystal=single_crystal
    )
    assert done == ran and told == ran
    assert found.single_crystal is kept


@pytest.mark.parametrize("plain, blurred, kept", [(0.37, 0.38, False), (0.31, 0.69, True)])
def test_the_blurred_background_has_to_fit_clearly_better(monkeypatch, plain, blurred, kept):
    """A polycrystal fits both backgrounds alike (de-twin's steel), a single crystal
    fits the blurred one far better (the indented silicon)."""
    _fake_searches(monkeypatch, {False: plain, True: blurred})
    assert search_pattern_center(None, None, EBSDDetector(), None).single_crystal is kept


@pytest.mark.parametrize(
    "found, near_start, searches",
    [((0.5, 0.42, 0.55), True, 1), ((0.5, 0.35, 0.55), False, 2)],
)
def test_a_start_is_searched_around_first(monkeypatch, found, near_start, searches):
    """A trusted center inside the window around *start* is kept; one at its edge
    means the center lay outside, and the full search runs."""
    windows = []

    def fake(patterns, master, detector, point_group, device, positions, progress, blurred, **kw):
        windows.append(kw.get("ys"))
        return Calibration(found, None, 0.4, 0.0, [], blurred)

    monkeypatch.setattr(search, "_search", fake)
    result = search_pattern_center(
        None, None, EBSDDetector(), None, single_crystal=False, start=(0.5, 0.5, 0.55)
    )
    assert result.near_start is near_start and len(windows) == searches
    np.testing.assert_allclose(windows[0], 0.5 + search.LOCAL_OFFSETS)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_a_search_recovers_a_known_center(cubic_master):
    """25 noisy patterns on a backscatter lobe at a known center, the search
    started 0.05-0.1 off -- beyond what the fit alone reaches."""
    truth, size = (0.45, 0.3, 0.6), 40
    detector = EBSDDetector(shape=(size, size), pattern_center=truth, sample_tilt=70.0)
    rng = np.random.default_rng(0)
    rotations = Rotation.from_euler(rng.uniform(0, 1, (25, 3)) * [2 * np.pi, np.pi, 2 * np.pi])
    bands = cubic_master.project(detector, rotations, rescale=False).numpy()
    y, x = np.mgrid[:size, :size]
    lobe = np.exp(-((x - 18) ** 2 + (y - 12) ** 2) / (2 * 16.0**2))
    patterns = rng.poisson(2000 * lobe * (1 + 0.3 * bands / bands.max())).astype(np.float32)
    assert looks_like_kikuchi(patterns)

    calls = []
    found = search_pattern_center(
        patterns,
        cubic_master,
        EBSDDetector(shape=(size, size), sample_tilt=70.0),
        Oh,
        single_crystal=False,
        start=(0.5, 0.4, 0.5),
        progress=lambda done, total: calls.append((done, total)),
    )
    assert found.trusted and found.near_start, found
    # measured 0.0104 off (pcy), at a fit of 0.34: the generic cubic bands are wide
    assert np.abs(np.subtract(found.pattern_center, truth)).max() < 0.015, found.pattern_center
    assert isinstance(found.rotations, Rotation) and found.rotations.size == 25
    assert calls[-1][0] == calls[-1][1]


# --------------------------------------------------------------------------- #
# the Kikuchi check
# --------------------------------------------------------------------------- #
def test_frames_that_are_not_kikuchi_patterns_are_told_apart(ni_crystal):
    """A Kikuchi pattern's bands on its backscatter lobe, against deapi's simulated
    disks and plain noise (de-twin's patterns, TEM images and diffraction are
    measured in ``looks_like_kikuchi``'s docstring)."""
    rng = np.random.default_rng(0)
    master = KinematicMaster(reflectors_from_phase(ni_crystal))
    detector = EBSDDetector(shape=(120, 120), pattern_center=(0.5, 0.3, 0.6))
    euler = rng.uniform(0, 1, (9, 3)) * [2 * np.pi, np.pi, 2 * np.pi]
    bands = master.project(detector, Rotation.from_euler(euler), rescale=False).numpy()
    bands = (bands - bands.min((1, 2), keepdims=True)) / np.ptp(bands, (1, 2), keepdims=True)
    y, x = np.mgrid[:120, :120]
    lobe = np.exp(-((x - 60) ** 2 + (y - 30) ** 2) / (2 * 45.0**2))
    assert looks_like_kikuchi(rng.poisson(300 * lobe * (1 + 0.1 * (bands - 0.5))))
    disks = np.zeros((120, 120))
    for angle in np.arange(6) * np.pi / 3:
        row, column = 60 + 33 * np.sin(angle), 60 + 33 * np.cos(angle)
        disks[(y - row) ** 2 + (x - column) ** 2 < 64] = 1
    disks[(y - 60) ** 2 + (x - 60) ** 2 < 64] = 1
    assert not looks_like_kikuchi(rng.poisson(20 + 2000 * np.repeat(disks[None], 9, 0)))
    assert not looks_like_kikuchi(rng.poisson(np.full((9, 120, 120), 300.0)))
