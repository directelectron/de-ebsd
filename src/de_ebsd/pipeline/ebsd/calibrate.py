# Moved verbatim from Autopilot (pyxem_base, a copy of directelectron/spyde spyde/ebsd/calibrate.py at b089ce47).
"""calibrate.py — solve an EBSD scan's pattern centre from its own patterns.

The pattern centre (where the beam spot sits relative to the screen, and how
far away the screen is) is rarely known well enough to index with: a few
percent off and every orientation is a few degrees wrong. The patterns
themselves pin it down. de_ebsd's ``determine_pattern_center`` fits ONE
centre shared by several patterns together with each pattern's orientation —
a single pattern cannot separate a centre shift from a small rotation, but
several patterns at different orientations can.

The steps, all on de_ebsd:

1. remove the detector's fixed intensity profile (the mean of the patterns,
   blurred for a single crystal) and each pattern's own background (the
   smooth backscatter gradient);
2. seed an orientation per pattern by dictionary indexing against the
   phase's dynamical master pattern (a 3° grid over its fundamental zone);
3. fit the shared centre, refining the orientations alongside.

The detector geometry is the viewer's (:mod:`spyde.ebsd.microscope`), which
agrees with de_ebsd's ``EBSDDetector`` exactly (both follow kikuchipy).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: Patterns read from the scan: a grid this many positions on a side, spread
#: across it, so the orientations differ (what separates centre from rotation).
CALIBRATION_GRID = 5
#: Orientation step of the seeding dictionary, in degrees. The centre fit
#: refines the orientations itself, so a seed only has to be close. A
#: spherical index at a cheap bandwidth was tried first and is NOT close
#: enough on small noisy patterns: its seeds matched no better than chance,
#: and the fit, with nothing to start from, did not move.
SEED_STEP_DEG = 3.0
#: Patterns are binned to at most this many pixels on a side. The centre is
#: measured in fractions of the screen, so binning does not change it, and
#: the seeding dictionary's cost grows with every pixel.
MAX_SIDE = 160
#: The mean per-tile correlation below which a solved centre is not trusted.
#: kikuchipy's real Ni calibration patterns (nine, 480×480) solve to within
#: 0.002 of kikuchipy's own refined centre from starts up to 0.05 off, at a
#: fit of 0.35–0.46. A failed solve "improves" the fit by wandering off along
#: the noise and stops near 0.08: those patterns with bad seeds, or
#: kikuchipy's 60×60 nickel scan, which moves 0.05 from its recorded centre.
MIN_FIT = 0.2
#: Chord directions the straightness is measured along.
KIKUCHI_ANGLES = 36
KIKUCHI_MIN_STRAIGHTNESS = 0.055
KIKUCHI_MAX_CONTRAST = 0.3
KIKUCHI_MIN_LOBE = 1.5
#: Below this contrast no lobe is asked for: a gain-corrected camera divides the
#: backscatter lobe out, and its bands are then a few percent of the signal.
KIKUCHI_CORRECTED_MAX_CONTRAST = 0.1
#: Fewer patterns than this do not average their bands away, so no static
#: background is estimated from them.
MIN_PATTERNS_FOR_STATIC = 9
#: A single crystal's patterns all carry its bands, so their mean does too and
#: would take them off with the background. Its static background is the mean
#: blurred by this (a Gaussian sigma, as a share of the pattern's width): the
#: backscatter's shape without the bands. On an indented silicon crystal's
#: 64 x 64 scan the centre then solves at a fit of 0.26, and not at all (0.14)
#: against the plain mean. The blur takes the detector's fine fixed pattern
#: off with the bands, so it suits gain-corrected frames: on an uncorrected
#: camera's polycrystal it lost the fit entirely.
SINGLE_CRYSTAL_BLUR = 0.05
#: How many times better the blurred background must fit before a scan is
#: taken for a single crystal. On a polycrystal the two fit alike, and a coin
#: flip to the blurred one biased its strain: de-twin's duplex steel fits 0.37
#: plain and 0.38 blurred, and blurred, its per-grain strain offset was 7.2e-4
#: against 2.6e-4. The indented silicon crystal fits 0.31 plain and 0.69 blurred.
SINGLE_CRYSTAL_MARGIN = 1.25

# ── The search, for a centre not known to within the fit's reach ──
#: The fit above only polishes: it moves the centre a few hundredths at most,
#: so from the usual defaults (0.5, 0.5, 0.55) it cannot reach a real centre
#: like kikuchipy's Ni scan's (0.42, 0.21, 0.50) or de-twin's (0.5, 0.3, 0.6),
#: and stops at a fit of about 0.1. The search first scores candidate centres
#: on a grid, by how far each pattern's best dictionary match stands above
#: that pattern's other matches (in standard deviations). The raw best
#: correlation is NOT that score: it rises steadily with the screen distance
#: (a far screen sees a smoother, easier-to-match pattern) and peaks at the
#: edge of any grid. Measured on both scans, the grid's best cell is within
#: 0.05 of the centre, and the fit from it lands within 0.0001 (Ni) and 0.004
#: (de-twin) of it.
SEARCH_PC_Y = np.round(np.arange(-0.2, 1.01, 0.1), 3)
SEARCH_PC_Z = np.round(np.arange(0.4, 1.41, 0.1), 3)
#: How far from a given starting centre it is searched for (pcy, pcz offsets), and
#: how near its window's edge a centre found there means it lay outside. 0.15: a
#: Zenith's own centre was 0.11 off in pcy from its Cu scan's; from 0.1 off in all
#: three this finds it in 36 s, where the full search takes 88.
LOCAL_OFFSETS = np.round(np.arange(-0.15, 0.151, 0.05), 3)
LOCAL_EDGE = 0.14
#: Finer grids around the best candidates, each half the last step.
SEARCH_REFINE_STEPS = (0.05, 0.025)
#: Candidates carried from one grid to the next, and finally fitted.
SEARCH_KEPT = 2
#: The scoring dictionary: coarse and matched binned, since it only has to
#: rank centres a grid step apart. Its orientations are sampled once for the
#: whole search: sampling the fundamental zone is 0.8 s on the CPU, eight
#: times the projection of a centre's dictionary, and it was being redone for
#: each of ~160 centres.
SEARCH_STEP_DEG = 5.0
SEARCH_BINNING = 2
#: Orientations projected at a time, which bounds the device memory a
#: dictionary takes while it is simulated.
SEARCH_CHUNK = 4096


@dataclass
class Calibration:
    pc: tuple[float, float, float]
    rotations: np.ndarray        # (N, 3, 3) crystal → sample, one per pattern
    fit: float                   # mean per-tile NCC at the solved geometry
    fit_before: float            # … and at the starting one
    positions: list              # the navigation indices the patterns came from
    single_crystal: bool = False  # solved against the blurred static background
    near_start: bool = False      # found within `LOCAL_OFFSETS` of the centre it started from

    @property
    def trusted(self) -> bool:
        return self.fit >= MIN_FIT


def calibration_positions(nav_shape, grid: int = CALIBRATION_GRID) -> list[tuple]:
    """Navigation indices spread over a scan — the centre one first.

    *nav_shape* is hyperspy's navigation shape (x first). A scan with no
    navigation gives the single position ``()``.
    """
    if not nav_shape:
        return [()]
    # a line scan gets as many patterns as a map, enough for a static background
    per_axis = grid if len(nav_shape) > 1 else grid * grid
    axes = [np.unique(np.linspace(0, n - 1, min(per_axis, n)).round().astype(int))
            for n in nav_shape]
    positions = [tuple(int(v) for v in combo)
                 for combo in np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T]
    centre = tuple(n // 2 for n in nav_shape)
    positions.sort(key=lambda p: sum((a - c) ** 2 for a, c in zip(p, centre)))
    return positions


def read_patterns(signal, positions) -> np.ndarray:
    """The frames at *positions*, as ``(N, rows, cols)`` float32, binned to
    at most :data:`MAX_SIDE` on a side.

    Reads one frame at a time — never the whole scan (CLAUDE.md, memory
    safety): ``inav`` of a lazy signal is a lazy one-frame slice.
    """
    frames = []
    for position in positions:
        frame = signal.inav[position] if position else signal
        frames.append(_bin(np.asarray(frame.data, dtype=np.float32)))
    return np.stack(frames)


def _bin(frame: np.ndarray) -> np.ndarray:
    factor = int(np.ceil(max(frame.shape) / MAX_SIDE))
    if factor <= 1:
        return frame
    rows, cols = (frame.shape[0] // factor) * factor, (frame.shape[1] // factor) * factor
    return frame[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor).mean((1, 3))


def static_background(patterns, *, single_crystal: bool = False) -> np.ndarray | None:
    """The detector's fixed intensity profile, as the mean of *patterns*;
    for a *single_crystal*, blurred by `SINGLE_CRYSTAL_BLUR`.

    Patterns at different orientations average their bands away, leaving the
    profile every frame shares. Without it the seeds go wrong: on kikuchipy's
    real Ni calibration patterns, dictionary seeds were right for 5 of 9
    patterns at the true centre (0 of 9 from 0.03 off) with only the
    per-pattern background removed, and 9 of 9 (8 of 9) with this subtracted
    too — as good as the background recorded with the scan. Too few patterns
    still carry their own bands, so then there is none.
    """
    if len(patterns) < MIN_PATTERNS_FOR_STATIC:
        return None
    mean = np.asarray(patterns, np.float32).mean(0)
    if single_crystal:
        from scipy.ndimage import gaussian_filter

        mean = gaussian_filter(mean, SINGLE_CRYSTAL_BLUR * mean.shape[-1])
    return mean


def solve_pattern_centre(patterns, master, setup, point_group, *, device=None,
                         positions=(), orientations=None,
                         single_crystal: bool = False) -> Calibration:
    """Fit the pattern centre shared by *patterns* ``(N, rows, cols)``.

    *master* is a de_ebsd ``MasterPattern`` of the phase at the scan's beam
    energy; *setup* the SEM :class:`~spyde.ebsd.microscope.Setup`, whose
    angles are held and whose centre is the starting guess. *point_group* is
    the phase's (orix), which bounds the seeding dictionary; *orientations*
    are that dictionary's (:func:`seed_orientations`), passed in when several
    starts are fitted, since sampling them is most of a seed's cost; and
    *single_crystal* blurs the static background (:func:`static_background`).
    """
    import torch
    import de_ebsd
    from de_ebsd.index import determine_pattern_center, remove_background

    patterns = np.asarray(patterns, np.float32)
    detector = _detector(setup, (setup["pc_x"], setup["pc_y"], setup["pc_z"]),
                         patterns.shape[-2:])
    device = device or master.data.device
    master = master.to(device)
    clean = remove_background(
        patterns, static_bg=static_background(patterns, single_crystal=single_crystal),
        dynamic=True, device=device)
    if orientations is None:
        orientations = seed_orientations(point_group)
    seeds = _best_matches(clean, master, detector, orientations, device)
    pc, rotations, info = determine_pattern_center(
        clean, master, detector, seeds, return_orientations=True, device=device)
    pc = tuple(float(v) for v in torch.as_tensor(pc).detach().cpu().numpy().reshape(3))
    # orix rotations here are Bunge matrices (sample → crystal); the viewer
    # keeps crystal → sample.
    bunge = np.asarray(rotations.to_matrix()).reshape(-1, 3, 3)
    return Calibration(pc=pc, rotations=np.transpose(bunge, (0, 2, 1)),
                       fit=float(info["ncc"]), fit_before=float(info["ncc_seed"]),
                       positions=list(positions), single_crystal=single_crystal)


def _detector(setup, pc, shape):
    import de_ebsd

    return de_ebsd.EBSDDetector(
        shape=tuple(shape), pattern_center=tuple(float(v) for v in pc),
        sample_tilt=setup["sample_tilt"], tilt=setup["detector_tilt"],
        azimuthal=setup["azimuthal"], twist=setup["twist"])


def grid_search(score, start_x: float, *, kept: int = SEARCH_KEPT, ys=SEARCH_PC_Y,
                zs=SEARCH_PC_Z) -> list[tuple]:
    """The *kept* best centres by *score* ``(pc) -> float``: the coarse
    (pcy, pcz) grid at *start_x*, then finer grids in all three around the
    best so far. Each centre is scored once."""
    scores: dict[tuple, float] = {}

    def ranked(candidates):
        for pc in candidates:
            if pc not in scores:
                scores[pc] = float(score(pc))
        return sorted(set(candidates), key=scores.__getitem__, reverse=True)[:kept]

    best = ranked([(round(float(start_x), 4), float(y), float(z))
                   for y in ys for z in zs])
    for step in SEARCH_REFINE_STEPS:
        offsets = (-step, 0.0, step)
        best = ranked([tuple(round(c + d, 4) for c, d in zip(centre, (dx, dy, dz)))
                       for centre in best
                       for dx in offsets for dy in offsets for dz in offsets])
    return best


def kikuchi_measures(frame) -> tuple[float, float, float]:
    """What tells a Kikuchi pattern from other frames, cheaply (~10 ms):

    * *straightness*: the share of the pattern's contrast that lies along
      straight lines across it — the mean along each long chord of the
      background-divided pattern, its variance over the chords against the
      pattern's own, the counting noise taken out of both (estimated from
      neighbouring pixels, so it holds from a few electrons a pixel up);
    * *contrast*: the pattern's relative modulation about its smooth
      background, the noise taken out — bands are a few percent of the
      backscatter, a TEM image's or a diffraction pattern's features are not;
    * *lobe*: the brightest of it (99.9th percentile, lightly smoothed) over
      its median — the backscatter lobe a Kikuchi pattern sits on spans a
      factor of a few; a TEM image is flat and a diffraction pattern's spots
      are far brighter.
    """
    from scipy import ndimage

    frame = _bin(np.asarray(frame, np.float64))
    rows, cols = frame.shape
    smooth = ndimage.gaussian_filter(frame, 0.1 * min(rows, cols))
    corrected = frame / np.maximum(smooth, 1e-9 + 1e-6 * smooth.max()) - 1
    neighbours = ndimage.convolve(
        corrected, np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]), mode="nearest")
    y, x = np.mgrid[:rows, :cols]
    y, x = y - (rows - 1) / 2, x - (cols - 1) / 2
    inside = x**2 + y**2 < (0.45 * min(rows, cols)) ** 2
    values, xs, ys = corrected[inside], x[inside], y[inside]
    # a pixel's difference from its four neighbours' mean is 1.25 times its noise
    noise = np.var((corrected - neighbours)[inside]) / 1.25
    signal = max(values.var() - noise, 1e-12)
    along = []
    for angle in np.linspace(0, np.pi, KIKUCHI_ANGLES, endpoint=False):
        distance = np.round(xs * np.cos(angle) + ys * np.sin(angle)).astype(int)
        distance -= distance.min()
        count = np.bincount(distance)
        total = np.bincount(distance, values)
        long = count >= 0.7 * count.max()
        means = total[long] / count[long]
        along.append(means.var() - noise / count[long].mean())
    straightness = max(float(np.mean(along)), 0.0) / signal
    lobe = np.percentile(ndimage.uniform_filter(frame, 3), 99.9) / max(np.median(frame), 1e-9)
    return straightness, float(np.sqrt(signal)), float(lobe)


def looks_like_kikuchi(patterns) -> bool:
    """Whether *patterns* ``(N, rows, cols)`` look like Kikuchi patterns, by
    the median of each of :func:`kikuchi_measures` over them; checked before a
    search, which would otherwise spend its minute on frames it cannot fit.

    Calibrated on the twin's EBSD patterns (Ni, Al and duplex-steel
    polycrystals, a Si wafer; 10 to 400 electrons a pixel) against its TEM
    images and diffraction patterns, deapi's simulated disks, flat noise and
    smooth random texture. The patterns' medians: straightness 0.13-0.17,
    contrast 0.05-0.08, lobe 2.6-2.9. The others fail at least one of: a
    diffraction pattern's straightness is 0.04-0.05; a TEM image's lobe is
    1.2-1.3 (one with straight hole edges, 0.24 straight, has a contrast of
    0.66); deapi's disks have a contrast of 1.7 and a lobe of 100; texture is
    0.05 straight and 0.28 in contrast; noise is flat. A smooth backscatter
    lobe without bands still passes, and is refused by the search's first fit
    instead, in seconds.

    A real camera's gain-corrected patterns have no lobe to speak of: on a
    Zenith's Cu scan (64 frames a point, 2 x 2 binned) straightness 0.065-0.112,
    contrast 0.016-0.030 and lobe 1.24-1.32, a TEM image's lobe. Their contrast
    tells them from one, so below `KIKUCHI_CORRECTED_MAX_CONTRAST` the lobe is
    not asked for."""
    measures = np.median([kikuchi_measures(frame) for frame in patterns], axis=0)
    straightness, contrast, lobe = (float(value) for value in measures)
    log.info("kikuchi check: straightness %.3f, contrast %.3f, lobe %.2f",
             straightness, contrast, lobe)
    return (straightness >= KIKUCHI_MIN_STRAIGHTNESS and contrast <= KIKUCHI_MAX_CONTRAST
            and (lobe >= KIKUCHI_MIN_LOBE or contrast <= KIKUCHI_CORRECTED_MAX_CONTRAST))



def search_pattern_centre(patterns, master, setup, point_group, *, device=None,
                          positions=(), progress=None, searching=None,
                          single_crystal: bool | None = None, start=None) -> Calibration:
    """:func:`solve_pattern_centre` from wherever the centre is: a
    :func:`grid_search` for the candidates, then the fit from each, the best
    fit kept. *setup*'s centre gives only the grid's pcx. About 25 s a search:
    some 160 centres are scored; *progress* ``(done, total)`` is called after
    each, and after each fit, counting each search from zero.

    Whether the patterns are one crystal's is *single_crystal*, or, left
    None, found: the search runs against the plain static background and
    again against the blurred one, and the blurred one is kept only when it
    fits `SINGLE_CRYSTAL_MARGIN` times better than the plain one
    (``Calibration.single_crystal`` says which). The patterns alone cannot
    tell: a camera's fixed pattern makes a polycrystal's as alike as a single
    crystal's. Nor can the plain search's fit alone: a single crystal's can
    pass as trusted while the blurred background fits it much better.
    *searching* ``(single_crystal)`` is told which background each search is
    against, before it starts.

    *start* ``(pcx, pcy, pcz)``, a centre known to about 0.1 (the camera's own),
    is searched around first, within `LOCAL_OFFSETS` of it: a third of the
    full grid. The full search runs only when that finds no trusted fit, or
    finds it at its window's edge (`LOCAL_EDGE`), the centre then lying
    outside it; ``Calibration.near_start`` says which."""
    choices = (False, True) if single_crystal is None else (single_crystal,)
    if start is not None:
        start = tuple(float(v) for v in start)
        values = dict(setup) if isinstance(setup, dict) else dict(setup.values)
        values.update(zip(("pc_x", "pc_y", "pc_z"), start))
        near = []
        for blurred in choices:
            if searching is not None:
                searching(blurred)
            near.append(_search(patterns, master, values, point_group, device, positions,
                                progress, blurred, ys=start[1] + LOCAL_OFFSETS,
                                zs=start[2] + LOCAL_OFFSETS))
            log.info("pattern centre near %s (single crystal %s): %s, fit %.2f", start, blurred,
                     near[-1].pc, near[-1].fit)
        best = _better(near)
        inside = all(abs(a - b) < LOCAL_EDGE for a, b in zip(best.pc, start))
        if best.trusted and inside:
            best.near_start = True
            return best
        log.info("pattern centre: none trusted within 0.1 of %s; searching everywhere", start)
    found = []
    for blurred in choices:
        if searching is not None:
            searching(blurred)
        found.append(_search(patterns, master, setup, point_group, device, positions,
                             progress, blurred))
        log.info("pattern centre search (single crystal %s): fit %.2f", blurred,
                 found[-1].fit)
    return _better(found)


def _better(found: list) -> Calibration:
    """The one of *found* (against the plain static background, then the
    blurred) to keep: the blurred only when it fits `SINGLE_CRYSTAL_MARGIN`
    times better."""
    if len(found) == 1:
        return found[0]
    plain, blurred = found
    return blurred if blurred.fit > SINGLE_CRYSTAL_MARGIN * plain.fit else plain


def _search(patterns, master, setup, point_group, device, positions, progress,
            single_crystal: bool, *, ys=SEARCH_PC_Y, zs=SEARCH_PC_Z) -> Calibration:
    import de_ebsd  # noqa: F401  (fail early, before the slow part)
    from de_ebsd.index import remove_background

    patterns = np.asarray(patterns, np.float32)
    device = device or master.data.device
    master = master.to(device)
    clean = remove_background(
        patterns, static_bg=static_background(patterns, single_crystal=single_crystal),
        dynamic=True, device=device)
    score = _lead_scorer(clean, master, setup, point_group, device)
    expected = (len(ys) * len(zs)
                + len(SEARCH_REFINE_STEPS) * SEARCH_KEPT * 27 + SEARCH_KEPT)
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(min(done, expected), expected)

    def contrast(pc):
        lead = score(pc)
        tick()
        return lead

    candidates = grid_search(contrast, setup["pc_x"], ys=ys, zs=zs)
    log.info("pattern centre search: candidates %s", candidates)
    # Every candidate is fitted: the best-scored is not always the best fit.
    orientations = seed_orientations(point_group)
    fits = []
    for pc in candidates:
        start = dict(setup) if isinstance(setup, dict) else dict(setup.values)
        start.update(zip(("pc_x", "pc_y", "pc_z"), pc))
        fits.append(solve_pattern_centre(patterns, master, start, point_group, device=device,
                                         positions=positions, orientations=orientations,
                                         single_crystal=single_crystal))
        tick()
    return max(fits, key=lambda calibration: calibration.fit)


def _normalized_rows(x):
    x = x - x.mean(-1, keepdim=True)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _lead_scorer(patterns, master, setup, point_group, device):
    """The search's score of a centre, ``(pc) -> float``: how far each of
    *patterns*' best match in the centre's dictionary stands above its other
    matches, in standard deviations, averaged over the patterns. The
    dictionary is de_ebsd's ``DictionaryIndexer``'s, binned by
    `SEARCH_BINNING`, with the orientations and the binned patterns made once
    for every centre scored."""
    import torch
    import torch.nn.functional as F
    from orix.sampling import get_sample_fundamental

    rotations = get_sample_fundamental(resolution=SEARCH_STEP_DEG, point_group=point_group)
    quaternions = torch.as_tensor(np.asarray(rotations.data), dtype=torch.float32)
    binned = torch.as_tensor(patterns).to(device=device, dtype=torch.float32)
    binned = F.avg_pool2d(binned[:, None], SEARCH_BINNING)[:, 0]
    shape = tuple(binned.shape[-2:])
    binned = _normalized_rows(binned.reshape(len(binned), -1))

    def score(pc) -> float:
        detector = _detector(setup, pc, shape)
        with torch.no_grad():
            parts = []
            for start in range(0, len(quaternions), SEARCH_CHUNK):
                simulated = master.project(detector, quaternions[start:start + SEARCH_CHUNK],
                                           rescale=False)
                simulated = _normalized_rows(simulated.reshape(len(simulated), -1).float())
                parts.append(binned @ simulated.T)
            scores = torch.cat(parts, dim=1)
            lead = (scores.max(1).values - scores.mean(1)) / scores.std(1)
        return float(lead.mean())

    return score


def seed_orientations(point_group):
    """The seeding dictionary's orientations, `SEED_STEP_DEG` apart over
    *point_group*'s fundamental zone, as ``(N, 4)`` quaternions."""
    from orix.sampling import get_sample_fundamental

    rotations = get_sample_fundamental(resolution=SEED_STEP_DEG, point_group=point_group)
    return np.asarray(rotations.data, np.float32)


def _best_matches(patterns, master, detector, orientations, device):
    """Each of *patterns*' best match among *orientations* projected from
    *master* onto *detector*: de_ebsd's ``DictionaryIndexer.index``, with the
    orientations given rather than sampled again."""
    import torch
    from orix.quaternion import Rotation

    quaternions = torch.as_tensor(orientations)
    flat = torch.as_tensor(patterns).to(device=device, dtype=torch.float32)
    flat = _normalized_rows(flat.reshape(len(flat), -1))
    best = torch.full((len(flat),), -torch.inf, device=flat.device)
    at = torch.zeros(len(flat), dtype=torch.long, device=flat.device)
    with torch.no_grad():
        for start in range(0, len(quaternions), SEARCH_CHUNK):
            simulated = master.project(detector, quaternions[start:start + SEARCH_CHUNK],
                                       rescale=False)
            simulated = _normalized_rows(simulated.reshape(len(simulated), -1).float())
            score, index = (flat @ simulated.T).max(1)
            better = score > best
            best = torch.where(better, score, best)
            at = torch.where(better, index + start, at)
    return Rotation(orientations[at.cpu().numpy()])
