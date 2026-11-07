"""Find a scan's pattern center from its own patterns, from wherever it is.

:func:`~de_ebsd.index.refine.pattern_center.determine_pattern_center` fits ONE
center shared by several patterns together with each pattern's orientation, but
it only polishes: it moves the center a few hundredths at most, and needs a seed
orientation per pattern. This module supplies both, so a center that is not known
to within the fit's reach can still be found:

1. the patterns' backgrounds are removed the way a live scan's are
   (:class:`~de_ebsd.index.live.background.ScanBackground`: divided by their
   static background, the mean of the patterns -- blurred, for a single crystal --
   then less each one's dynamic background);
2. :func:`grid_search` scores candidate centers by how far each pattern's best
   dictionary match stands above its other matches, coarse to fine;
3. :func:`solve_pattern_center` seeds an orientation per pattern by dictionary
   matching at each candidate and fits the shared center from it; the best fit
   is kept.

:func:`looks_like_kikuchi` tells, cheaply, whether frames are Kikuchi patterns at
all, before a search spends its minute on frames it cannot fit.

The measured numbers quoted in this module (kikuchipy's Ni scan, de-twin's
scans, a Zenith's Cu scan, an indented silicon crystal) were taken with SpyDE's
port of this search, whose static background was SUBTRACTED; here it is divided,
as everywhere else in de-ebsd. :data:`MIN_FIT` and :data:`SINGLE_CRYSTAL_MARGIN`
have not been re-measured on those scans since.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F

from de_ebsd.core.device import holds_lock, resolve_device
from de_ebsd.index.dictionary import DictionaryIndexer, dictionary_scores, sample_orientations
from de_ebsd.index.live.background import ScanBackground
from de_ebsd.index.preprocess import bin_pattern
from de_ebsd.index.refine.pattern_center import determine_pattern_center

__all__ = [
    "Calibration",
    "calibration_positions",
    "clean_patterns",
    "grid_search",
    "kikuchi_measures",
    "looks_like_kikuchi",
    "search_pattern_center",
    "solve_pattern_center",
]

log = logging.getLogger(__name__)

#: Patterns read from a scan for a search: a grid this many positions on a side,
#: spread across it, so the orientations differ (what separates a center shift
#: from a rotation). See :func:`calibration_positions`.
CALIBRATION_GRID = 5
#: Orientation step of the seeding dictionary, in degrees. The center fit refines
#: the orientations itself, so a seed only has to be close. A spherical index at a
#: cheap bandwidth was tried first and is NOT close enough on small noisy patterns:
#: its seeds matched no better than chance, and the fit did not move.
SEED_STEP_DEG = 3.0
#: Patterns are binned to at most this many pixels on a side (:func:`bin_pattern`).
#: The center is a fraction of the screen, so binning does not change it, and the
#: seeding dictionary's cost grows with every pixel.
MAX_SIDE = 160
#: The mean per-tile correlation below which a solved center is not trusted.
#: kikuchipy's real Ni calibration patterns (nine, 480 x 480) solved to within
#: 0.002 of kikuchipy's own refined center from starts up to 0.05 off, at a fit of
#: 0.35-0.46. A failed solve "improves" the fit by wandering off along the noise
#: and stops near 0.08.
MIN_FIT = 0.2
#: Fewer patterns than this do not average their bands away, so no static
#: background is estimated from them. Lower than a live scan's
#: :data:`~de_ebsd.index.live.background.MIN_FRAMES_FOR_STATIC` (32) because a
#: search's patterns come from all over the scan (:func:`calibration_positions`
#: gives 25), not from the few grains a scan's first frames cross: on kikuchipy's
#: nine Ni calibration patterns, dictionary seeds were right for 5 of 9 at the true
#: center with only the dynamic background removed, and 9 of 9 with a static one
#: from those nine too.
MIN_PATTERNS_FOR_STATIC = 9
#: A single crystal's patterns all carry its bands, so their mean does too and
#: would take them off with the background. Its static background is the mean
#: blurred by this (a Gaussian sigma, as a share of the pattern's width): the
#: backscatter's shape without the bands. On an indented silicon crystal's 64 x 64
#: scan the center then solved at a fit of 0.26, and not at all (0.14) against the
#: plain mean.
SINGLE_CRYSTAL_BLUR = 0.05
#: How many times better the blurred background must fit before a scan is taken
#: for a single crystal. On a polycrystal the two fit alike (de-twin's duplex
#: steel: 0.37 plain, 0.38 blurred), and a coin flip to the blurred one biased its
#: strain; the indented silicon crystal fit 0.31 plain and 0.69 blurred.
SINGLE_CRYSTAL_MARGIN = 1.25

#: Chord directions :func:`kikuchi_measures` measures straightness along, and the
#: thresholds :func:`looks_like_kikuchi` applies to the measures' medians.
KIKUCHI_ANGLES = 36
KIKUCHI_MIN_STRAIGHTNESS = 0.055
KIKUCHI_MAX_CONTRAST = 0.3
KIKUCHI_MIN_LOBE = 1.5
#: Below this contrast no lobe is asked for: a gain-corrected camera divides the
#: backscatter lobe out, and its bands are then a few percent of the signal.
KIKUCHI_CORRECTED_MAX_CONTRAST = 0.1

#: The coarse (pcy, pcz) grid of a full search, at the start's pcx. From the usual
#: defaults (0.5, 0.5, 0.55) the fit alone cannot reach a real center such as
#: kikuchipy's Ni scan's (0.42, 0.21, 0.50) or de-twin's (0.5, 0.3, 0.6). The score
#: is how far each pattern's best match stands above its other matches, in standard
#: deviations -- NOT the raw best correlation, which rises steadily with the screen
#: distance and peaks at the edge of any grid. Measured on both scans, the grid's
#: best cell was within 0.05 of the center and the fit from it within 0.004.
SEARCH_PC_Y = np.round(np.arange(-0.2, 1.01, 0.1), 3)
SEARCH_PC_Z = np.round(np.arange(0.4, 1.41, 0.1), 3)
#: How far from a given start the center is searched for first (pcy, pcz offsets),
#: and how near that window's edge a center found there means it lay outside. A
#: Zenith's own center was 0.11 off in pcy from its Cu scan's; from 0.1 off in all
#: three, the local search found it in 36 s where the full search took 88.
LOCAL_OFFSETS = np.round(np.arange(-0.15, 0.151, 0.05), 3)
LOCAL_EDGE = 0.14
#: Finer grids around the best candidates, each half the last step.
SEARCH_REFINE_STEPS = (0.05, 0.025)
#: Candidates carried from one grid to the next, and finally fitted.
SEARCH_KEPT = 2
#: The scoring dictionary: coarse and matched binned, since it only has to rank
#: centers a grid step apart. Its orientations are sampled once for the search.
SEARCH_STEP_DEG = 5.0
SEARCH_BINNING = 2


@dataclass
class Calibration:
    """A solved pattern center, and how far to trust it.

    Attributes
    ----------
    pattern_center
        ``(pcx, pcy, pcz)``, Bruker convention.
    rotations
        One orientation per pattern at that center, an orix ``Rotation`` (sample to
        crystal, as :func:`determine_pattern_center` returns them).
    fit
        Mean per-tile NCC at the solved geometry.
    fit_before
        The same at the seeds.
    positions
        Where in the scan the patterns came from.
    single_crystal
        Solved against the blurred static background.
    near_start
        Found within :data:`LOCAL_OFFSETS` of the center the search started from.
    """

    pattern_center: tuple[float, float, float]
    rotations: object
    fit: float
    fit_before: float
    positions: list = field(default_factory=list)
    single_crystal: bool = False
    near_start: bool = False

    @property
    def trusted(self) -> bool:
        """Whether :attr:`fit` reaches :data:`MIN_FIT`."""
        return self.fit >= MIN_FIT


def calibration_positions(nav_shape, grid: int = CALIBRATION_GRID) -> list[tuple]:
    """Scan positions spread over a scan of *nav_shape*, the middle one first.

    A *grid* x *grid* lattice over a 2-D scan, and as many (``grid ** 2``) along a
    line scan, enough for a static background. Each axis of *nav_shape* is indexed
    in its own order, so the positions index the scan the way *nav_shape* does. An
    empty *nav_shape* (a single pattern) gives the one position ``()``.
    """
    if not nav_shape:
        return [()]
    per_axis = grid if len(nav_shape) > 1 else grid * grid
    axes = [
        np.unique(np.linspace(0, n - 1, min(per_axis, n)).round().astype(int)) for n in nav_shape
    ]
    combos = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T
    positions = [tuple(int(v) for v in combo) for combo in combos]
    middle = tuple(n // 2 for n in nav_shape)
    positions.sort(key=lambda p: sum((a - c) ** 2 for a, c in zip(p, middle, strict=True)))
    return positions


def clean_patterns(patterns, *, single_crystal: bool = False, device=None) -> torch.Tensor:
    """*patterns* ``(N, H, W)`` with their backgrounds removed, as a live scan's are.

    A :class:`~de_ebsd.index.live.background.ScanBackground` over the patterns
    themselves: divided by their mean (blurred by :data:`SINGLE_CRYSTAL_BLUR` for a
    *single_crystal*) once there are :data:`MIN_PATTERNS_FOR_STATIC`, then less
    each pattern's own dynamic background. Without the static background the seeds
    go wrong (see :data:`MIN_PATTERNS_FOR_STATIC`).
    """
    device = resolve_device(device)
    t = torch.as_tensor(np.asarray(patterns, dtype=np.float32), device=device)
    if t.ndim == 2:
        t = t[None]
    background = ScanBackground(tuple(t.shape[-2:]), device, min_frames=MIN_PATTERNS_FOR_STATIC)
    background.restart(blur=SINGLE_CRYSTAL_BLUR if single_crystal else None)
    background.observe(t)
    return background.remove(t)


@holds_lock
def solve_pattern_center(
    patterns,
    master,
    detector,
    point_group=None,
    *,
    device=None,
    positions=(),
    orientations=None,
    single_crystal: bool = False,
    lock=None,
) -> Calibration:
    """Fit the pattern center shared by *patterns* ``(N, H, W)``, from seeds found
    by dictionary matching.

    Parameters
    ----------
    patterns
        Raw patterns, numpy or torch; their backgrounds are removed here
        (:func:`clean_patterns`).
    master
        The phase's :class:`~de_ebsd.sim.MasterPattern` at the scan's energy.
    detector
        :class:`~de_ebsd.sim.detector.EBSDDetector`: its angles are held and its
        pattern center is the start. Its shape is replaced by the patterns'.
    point_group
        orix ``Symmetry`` bounding the seeding dictionary; ``None`` takes it from
        ``master.metadata["crystal"]``.
    device
        Defaults to the master's.
    positions
        Recorded on the result.
    orientations
        The seeding dictionary's orientations, an orix ``Rotation``; by default
        :func:`~de_ebsd.index.sample_orientations` at :data:`SEED_STEP_DEG`. Passed
        in when several starts are fitted, since sampling them is most of a seed's
        cost.
    single_crystal
        Blur the static background (see :func:`clean_patterns`).
    lock
        Held around the device work; see :func:`~de_ebsd.core.device.resolve_lock`.
    """
    device = resolve_device(device, master.device)
    master = master.to(device)
    clean = clean_patterns(patterns, single_crystal=single_crystal, device=device)
    detector = replace(detector, shape=tuple(clean.shape[-2:]))
    if orientations is None:
        point_group = DictionaryIndexer._resolve_point_group(master, point_group)
        orientations = sample_orientations(SEED_STEP_DEG, point_group)
    seeds = _best_matches(clean, master, detector, orientations, device)
    center, rotations, info = determine_pattern_center(
        clean, master, detector, seeds, return_orientations=True, device=device, lock=lock
    )
    return Calibration(
        pattern_center=tuple(float(v) for v in center.detach().cpu().numpy().reshape(3)),
        rotations=rotations,
        fit=float(info["ncc"]),
        fit_before=float(info["ncc_seed"]),
        positions=list(positions),
        single_crystal=single_crystal,
    )


def _best_matches(patterns, master, detector, orientations, device):
    """Each of *patterns*' best match among *orientations*, an orix ``Rotation``."""
    from orix.quaternion import Rotation

    scores = dictionary_scores(patterns, master, detector, orientations, device=device)
    best = scores.argmax(1).cpu().numpy()
    return Rotation(np.asarray(orientations.data).reshape(-1, 4)[best])


def grid_search(
    score, start_x: float, *, kept: int = SEARCH_KEPT, ys=SEARCH_PC_Y, zs=SEARCH_PC_Z
) -> list[tuple]:
    """The *kept* best centers by *score* ``(pattern_center) -> float``.

    The coarse (pcy, pcz) grid *ys* x *zs* at *start_x*, then finer grids in all
    three around the best so far (:data:`SEARCH_REFINE_STEPS`). Each center is
    scored once.
    """
    scores: dict[tuple, float] = {}

    def ranked(candidates):
        for pc in candidates:
            if pc not in scores:
                scores[pc] = float(score(pc))
        return sorted(set(candidates), key=scores.__getitem__, reverse=True)[:kept]

    best = ranked([(round(float(start_x), 4), float(y), float(z)) for y in ys for z in zs])
    for step in SEARCH_REFINE_STEPS:
        offsets = (-step, 0.0, step)
        best = ranked(
            [
                tuple(round(c + d, 4) for c, d in zip(center, (dx, dy, dz), strict=True))
                for center in best
                for dx in offsets
                for dy in offsets
                for dz in offsets
            ]
        )
    return best


def _lead_scorer(patterns, master, detector, point_group, device):
    """The search's score of a center, ``(pattern_center) -> float``: how far each of
    the cleaned *patterns*' best match in that center's dictionary stands above its
    other matches, in standard deviations, averaged over the patterns. Binned by
    :data:`SEARCH_BINNING`, with the orientations sampled once for every center."""
    point_group = DictionaryIndexer._resolve_point_group(master, point_group)
    rotations = sample_orientations(SEARCH_STEP_DEG, point_group)
    binned = torch.as_tensor(patterns).to(device=device, dtype=torch.float32)
    binned = F.avg_pool2d(binned[:, None], SEARCH_BINNING)[:, 0]
    shape = tuple(binned.shape[-2:])

    def score(pc) -> float:
        at = replace(detector, shape=shape, pattern_center=tuple(float(v) for v in pc))
        with torch.no_grad():
            scores = dictionary_scores(binned, master, at, rotations, device=device)
        lead = (scores.max(1).values - scores.mean(1)) / scores.std(1)
        return float(lead.mean())

    return score


@holds_lock
def search_pattern_center(
    patterns,
    master,
    detector,
    point_group=None,
    *,
    device=None,
    positions=(),
    progress=None,
    searching=None,
    single_crystal: bool | None = None,
    start=None,
    lock=None,
) -> Calibration:
    """:func:`solve_pattern_center` from wherever the center is.

    A :func:`grid_search` for candidate centers, then the fit from each, the best
    fit kept. *detector*'s pattern center gives only the grid's pcx. Some 160
    centers are scored; *progress* ``(done, total)`` is called after each, and after
    each fit, counting each search from zero.

    Parameters
    ----------
    patterns, master, detector, point_group, device, positions, lock
        As in :func:`solve_pattern_center`.
    progress
        ``(done, total)`` callback.
    searching
        ``(single_crystal)`` callback, told which background each search is against
        before it starts.
    single_crystal
        Whether the patterns are one crystal's; ``None`` finds out: the search runs
        against the plain static background and again against the blurred one, and
        the blurred one is kept only when it fits :data:`SINGLE_CRYSTAL_MARGIN`
        times better (``Calibration.single_crystal`` says which). The patterns
        alone cannot tell: a camera's fixed pattern makes a polycrystal's as alike
        as a single crystal's.
    start
        ``(pcx, pcy, pcz)``, a center known to about 0.1 (the camera's own),
        searched around first, within :data:`LOCAL_OFFSETS` of it: a third of the
        full grid. The full search runs only when that finds no trusted fit, or
        finds it within :data:`LOCAL_EDGE` of its window's edge;
        ``Calibration.near_start`` says which.
    """
    choices = (False, True) if single_crystal is None else (single_crystal,)
    if start is not None:
        start = tuple(float(v) for v in start)
        near = []
        for blurred in choices:
            if searching is not None:
                searching(blurred)
            near.append(
                _search(
                    patterns,
                    master,
                    replace(detector, pattern_center=start),
                    point_group,
                    device,
                    positions,
                    progress,
                    blurred,
                    ys=start[1] + LOCAL_OFFSETS,
                    zs=start[2] + LOCAL_OFFSETS,
                    lock=lock,
                )
            )
            log.info(
                "pattern center near %s (single crystal %s): %s, fit %.2f",
                start,
                blurred,
                near[-1].pattern_center,
                near[-1].fit,
            )
        best = _better(near)
        inside = all(
            abs(a - b) < LOCAL_EDGE for a, b in zip(best.pattern_center, start, strict=True)
        )
        if best.trusted and inside:
            best.near_start = True
            return best
        log.info("pattern center: none trusted near %s; searching everywhere", start)
    found = []
    for blurred in choices:
        if searching is not None:
            searching(blurred)
        found.append(
            _search(
                patterns,
                master,
                detector,
                point_group,
                device,
                positions,
                progress,
                blurred,
                lock=lock,
            )
        )
        log.info("pattern center search (single crystal %s): fit %.2f", blurred, found[-1].fit)
    return _better(found)


def _better(found: list) -> Calibration:
    """The one of *found* (against the plain static background, then the blurred) to
    keep: the blurred only when it fits :data:`SINGLE_CRYSTAL_MARGIN` times better."""
    if len(found) == 1:
        return found[0]
    plain, blurred = found
    return blurred if blurred.fit > SINGLE_CRYSTAL_MARGIN * plain.fit else plain


def _search(
    patterns,
    master,
    detector,
    point_group,
    device,
    positions,
    progress,
    single_crystal: bool,
    *,
    ys=SEARCH_PC_Y,
    zs=SEARCH_PC_Z,
    lock=None,
) -> Calibration:
    device = resolve_device(device, master.device)
    master = master.to(device)
    point_group = DictionaryIndexer._resolve_point_group(master, point_group)
    clean = clean_patterns(patterns, single_crystal=single_crystal, device=device)
    score = _lead_scorer(clean, master, detector, point_group, device)
    expected = len(ys) * len(zs) + len(SEARCH_REFINE_STEPS) * SEARCH_KEPT * 27 + SEARCH_KEPT
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(min(done, expected), expected)

    def lead(pc):
        value = score(pc)
        tick()
        return value

    candidates = grid_search(lead, detector.pattern_center[0], ys=ys, zs=zs)
    log.info("pattern center search: candidates %s", candidates)
    # every candidate is fitted: the best-scored is not always the best fit
    orientations = sample_orientations(SEED_STEP_DEG, point_group)
    fits = []
    for pc in candidates:
        fits.append(
            solve_pattern_center(
                patterns,
                master,
                replace(detector, pattern_center=pc),
                point_group,
                device=device,
                positions=positions,
                orientations=orientations,
                single_crystal=single_crystal,
                lock=lock,
            )
        )
        tick()
    # centers shared by two refined grids are scored once, so the count can fall
    # short of the total it promised; a finished search says it is finished
    if progress is not None:
        progress(expected, expected)
    return max(fits, key=lambda calibration: calibration.fit)


def kikuchi_measures(frame) -> tuple[float, float, float]:
    """What tells a Kikuchi pattern from other frames, cheaply (~10 ms).

    Returns ``(straightness, contrast, lobe)``:

    * *straightness*: the share of the pattern's contrast that lies along straight
      lines across it -- the mean along each long chord of the background-divided
      pattern, its variance over the chords against the pattern's own, the
      counting noise taken out of both (estimated from neighboring pixels, so it
      holds from a few electrons a pixel up);
    * *contrast*: the pattern's relative modulation about its smooth background,
      the noise taken out -- bands are a few percent of the backscatter, a TEM
      image's or a diffraction pattern's features are not;
    * *lobe*: the brightest of it (99.9th percentile, lightly smoothed) over its
      median -- the backscatter lobe a Kikuchi pattern sits on spans a factor of a
      few; a TEM image is flat and a diffraction pattern's spots are far brighter.
    """
    from scipy import ndimage

    frame = bin_pattern(np.asarray(frame, np.float64), MAX_SIDE)
    rows, cols = frame.shape
    smooth = ndimage.gaussian_filter(frame, 0.1 * min(rows, cols))
    corrected = frame / np.maximum(smooth, 1e-9 + 1e-6 * smooth.max()) - 1
    neighbours = ndimage.convolve(
        corrected, np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]), mode="nearest"
    )
    y, x = np.mgrid[:rows, :cols]
    y, x = y - (rows - 1) / 2, x - (cols - 1) / 2
    inside = x**2 + y**2 < (0.45 * min(rows, cols)) ** 2
    values, xs, ys = corrected[inside], x[inside], y[inside]
    # a pixel's difference from its four neighbors' mean is 1.25 times its noise
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
    """Whether *patterns* ``(N, H, W)`` look like Kikuchi patterns, by the median of
    each of :func:`kikuchi_measures` over them.

    Calibrated on de-twin's EBSD patterns (Ni, Al and duplex-steel polycrystals, a
    Si wafer; 10 to 400 electrons a pixel) against its TEM images and diffraction
    patterns, deapi's simulated disks, flat noise and smooth random texture. The
    patterns' medians: straightness 0.13-0.17, contrast 0.05-0.08, lobe 2.6-2.9.
    The others fail at least one test: a diffraction pattern's straightness is
    0.04-0.05; a TEM image's lobe is 1.2-1.3 (one with straight hole edges, 0.24
    straight, has a contrast of 0.66); deapi's disks have a contrast of 1.7 and a
    lobe of 100; texture is 0.05 straight and 0.28 in contrast; noise is flat. A
    smooth backscatter lobe without bands still passes, and is refused by a
    search's first fit instead.

    A gain-corrected camera's patterns have no lobe to speak of: a Zenith's Cu scan
    (64 frames a point, 2 x 2 binned) measured straightness 0.065-0.112, contrast
    0.016-0.030 and lobe 1.24-1.32, a TEM image's lobe. Their contrast tells them
    from one, so below :data:`KIKUCHI_CORRECTED_MAX_CONTRAST` no lobe is asked for.
    """
    measures = np.median([kikuchi_measures(frame) for frame in patterns], axis=0)
    straightness, contrast, lobe = (float(value) for value in measures)
    log.info(
        "kikuchi check: straightness %.3f, contrast %.3f, lobe %.2f", straightness, contrast, lobe
    )
    return (
        straightness >= KIKUCHI_MIN_STRAIGHTNESS
        and contrast <= KIKUCHI_MAX_CONTRAST
        and (lobe >= KIKUCHI_MIN_LOBE or contrast <= KIKUCHI_CORRECTED_MAX_CONTRAST)
    )
