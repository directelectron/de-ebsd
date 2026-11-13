"""Calibrating the pattern center from a buffer of live frames."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from de_ebsd.index.live.common import parabola_offset, pick

#: Pattern-center steps (fractions of the detector, as the center is), each half the
# : last. On a twin scan (1024 patterns, 33 grains), starts 0.004-0.01 off in every coordinate came
# back
#: within 0.001 of the truth (256 patterns, 0.45 s); 0.002 off costs 0.07° of median
#: orientation error.
PATTERN_CENTER_STEPS = (0.008, 0.004, 0.002, 0.001)

#: The orientation refine run at each candidate center. It must resolve far finer
#: than the indexing's: a center 0.004 off in x moves every pattern's best fit by a
#: few tenths of a degree, and steps ending at 0.12° follow that in jumps, which
#: leaves the mean score flat in x and the center stuck 0.003 short.
PATTERN_CENTER_ORIENTATION_STEPS_DEG = (0.5, 0.25, 0.12, 0.06, 0.03)

#: Patterns a center is fitted with: the best-scored of the buffer.
PATTERN_CENTER_PATTERNS = 256


def refine_pattern_center(
    indexer,
    frames,
    *,
    patterns: int = PATTERN_CENTER_PATTERNS,
    steps=PATTERN_CENTER_STEPS,
    orientation_steps_deg=PATTERN_CENTER_ORIENTATION_STEPS_DEG,
) -> tuple[float, float, float]:
    """The pattern center that best fits the best-scored *patterns* of
    *frames* ``(N, H, W)``, which the indexer then uses
    (:meth:`LiveIndexer.set_pattern_center`).

    A center moved along x or y looks to every pattern much like the sample
    turned slightly, so the center is fitted with each orientation free: at
    each candidate center every orientation is refined again
    (*orientation_steps_deg*) and the candidate scored by the patterns' mean
    score. A level scores the center and one step either side of it on each
    axis and moves to each axis's parabola vertex, as :meth:`LiveIndexer.refine` does for
    an orientation. The patterns should come from several grains: one
    orientation alone cannot tell the center from a turn.
    """
    phase, g, score, _ = indexer.index(frames)
    best = torch.argsort(score, descending=True)[:patterns]
    _, rows = indexer.prepare(pick(torch.as_tensor(frames, device=indexer.device), best))
    phase, g = phase[best], g[best]
    orientation_steps = tuple(math.radians(s) for s in orientation_steps_deg)
    pattern_center = torch.tensor(indexer.detector.pattern_center, dtype=torch.float64)
    for step in steps:
        scores, refined = [], []
        for offset in indexer.stencil.cpu().double():
            directions = indexer.directions_at(pattern_center + step * offset, indexer.refine_shape)
            found, found_score = indexer.refine(
                rows, phase, g, steps=orientation_steps, directions=directions
            )
            scores.append(found_score.mean())
            refined.append(found)
        g = refined[0]
        pattern_center = (
            pattern_center + step * parabola_offset(torch.stack(scores).double().cpu()[None])[0]
        )
    center = tuple(float(v) for v in pattern_center)
    indexer.set_pattern_center(center)
    return center


# --- the pattern center read back from a scan's strain ------------------------

#: Tukey's biweight cut-off, in robust standard deviations of the points' distance
#: from the plane: a point further off (real strain, a poor fit) gets no weight.
ROBUST_CUTOFF = 4.685

#: Reweighting passes of the robust plane fit; it settles in a handful.
ROBUST_ITERATIONS = 20

#: Exact planes through random minimal sets of points, the one with the least
#: median distance starting the reweighting. A least-squares start does not do:
#: a strained corner tilts it toward itself, and the reweighting then keeps the
#: corner (on a 12 x 12 scan with a quarter of one corner strained, 0.0004 off
#: against 0.0001).
ROBUST_STARTS = 500

#: The six strain components the fit reads, the off-diagonal ones counted twice
#: (weight sqrt 2), so that every strain's size is its Frobenius norm.
_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_COMPONENT_WEIGHTS = np.array([1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)])


def strain_per_center(indexer, *, step: float = 1e-4) -> np.ndarray:
    """``(3, 3, 3)``: ``[k]`` the strain a pattern fitted at a center one unit
    too far along x, y or z (``k``, the center's own units) shows, at the
    indexer's center, in the sample frame the strain is returned in.

    Moving the center is a linear map of every pixel's direction
    (:func:`~de_ebsd.index.live.deformation.center_maps`), so a pattern fitted
    at the wrong center is fitted with that map folded into its deformation:
    its strain is the map's symmetric, trace-free part (its rotation is
    taken up by the orientation, its trace is invisible). That is worked out
    here from the geometry alone, by central differences of the maps -- no
    pattern is fitted. A center error ``d`` then shows as the strain
    ``tensordot(d, strain_per_center(...), 1)``."""
    center = np.asarray(indexer.detector.pattern_center, float)
    centers = [center + sign * step * axis for axis in np.eye(3) for sign in (1.0, -1.0)]
    maps = indexer.center_maps(centers).double().cpu().numpy()
    out = []
    for k in range(3):
        slope = (maps[2 * k] - maps[2 * k + 1]) / (2 * step)
        symmetric = 0.5 * (slope + slope.T)
        out.append(symmetric - np.trace(symmetric) / 3 * np.eye(3))
    return np.stack(out)


@dataclass(frozen=True)
class CenterPlane:
    """A scan's pattern center as a plane in the beam position:
    ``center + positions @ drift``.

    *center* is at position ``(0, 0)``; *drift* ``(2, 3)`` is the center's change
    per unit of position along x and along y. *points* ``(N, 3)`` is each
    point's own center as its strain gave it, and *weights* ``(N,)`` its weight
    in the fit: 0 for a point left out (real strain, a refused fit, outside
    *use*)."""

    center: tuple[float, float, float]
    drift: np.ndarray
    points: np.ndarray
    weights: np.ndarray

    def at(self, positions) -> np.ndarray:
        """The center ``(N, 3)`` at each beam position ``(N, 2)``."""
        return np.asarray(self.center) + np.asarray(positions, float) @ self.drift


def centers_from_strain(
    strain,
    positions,
    fitted_at,
    response,
    *,
    use=None,
    drift=None,
    robust: bool = True,
) -> CenterPlane:
    """The pattern center across a scan, read back from the strain its points
    were fitted with (:meth:`LiveIndexer.index_with_strain`).

    *strain* ``(N, 3, 3)`` was fitted with each point at *fitted_at* (one
    center ``(3,)`` or one per point ``(N, 3)``); *positions* ``(N, 2)`` are
    the beam's, in any unit (the drift is per that unit); *response* is
    :func:`strain_per_center`'s. Each point's own center is where its strain
    says it is, ``fitted_at - pinv(J) strain``, and a plane through those is
    the scan's: the beam moving over the specimen moves the center linearly.

    Within one pattern a center error and a real strain are the same thing; the
    plane tells them apart only by how they vary over the scan. A center error
    is a plane, and of the five deviatoric strains it can show only three
    combinations; a real strain that is local -- an indent, a grain's -- is
    neither. With *robust* (Tukey's biweight) the points the plane does not
    explain are given no weight, so it needs no telling where the specimen is
    strained; *use* ``(N,)`` restricts the fit to points known to be
    unstrained instead or as well. What it cannot separate is a strain the same
    over the whole field (three of its five components read as a center
    offset) and one varying linearly across it (as a drift): with *drift*
    ``(2, 3)``, from the geometry, only the center is fitted, and a linear
    strain survives. Anything that makes the master disagree with the camera
    (the energy, a lens distortion) shows as strain, and so moves the
    center found.

    Index the scan again at :meth:`CenterPlane.at` its positions for its
    strain: subtracting the strain this predicts is good only while the
    center was close, and on strained points not even then."""
    strain = np.asarray(strain, float)
    positions = np.asarray(positions, float)
    count = len(strain)
    jacobian = np.stack([response[:, i, j] for i, j in _COMPONENTS]) * _COMPONENT_WEIGHTS[:, None]
    vectors = np.stack([strain[:, i, j] for i, j in _COMPONENTS], axis=1) * _COMPONENT_WEIGHTS
    points = np.asarray(fitted_at, float) - vectors @ np.linalg.pinv(jacobian).T
    usable = np.isfinite(points).all(axis=1)
    if use is not None:
        usable &= np.asarray(use, bool)
    if usable.sum() < 3:
        raise ValueError("fewer than three usable points: no plane through them")
    if drift is None:
        design = np.column_stack([np.ones(count), positions])
        target = points
    else:
        design = np.ones((count, 1))
        target = points - positions @ np.asarray(drift, float)
    target = np.where(usable[:, None], target, 0.0)
    weights = usable.astype(float)
    if robust:
        weights = _tukey(_least_median_plane(design, target, usable), design, target, usable)
    for _ in range(ROBUST_ITERATIONS if robust else 1):
        root = np.sqrt(weights)[:, None]
        coefficients, *_ = np.linalg.lstsq(design * root, target * root, rcond=None)
        if robust:
            weights = _tukey(coefficients, design, target, usable)
    center = tuple(float(v) for v in coefficients[0])
    found_drift = coefficients[1:] if drift is None else np.asarray(drift, float)
    return CenterPlane(center, found_drift, points, weights)


def _least_median_plane(design, target, usable) -> np.ndarray:
    """Of exact fits through :data:`ROBUST_STARTS` random minimal sets of the
    *usable* points, the one whose median distance is least: a start no cluster
    of strained points, short of half of them, can pull."""
    rng = np.random.default_rng(0)
    indices = np.flatnonzero(usable)
    size = design.shape[1]
    best, best_median = None, np.inf
    for _ in range(ROBUST_STARTS):
        chosen = rng.choice(indices, size, replace=False)
        coefficients, *_ = np.linalg.lstsq(design[chosen], target[chosen], rcond=None)
        distance = np.linalg.norm(target[indices] - design[indices] @ coefficients, axis=1)
        median = float(np.median(distance))
        if median < best_median:
            best, best_median = coefficients, median
    return best


def _tukey(coefficients, design, target, usable) -> np.ndarray:
    """Tukey's biweight of each point's distance from the plane: 0 beyond
    :data:`ROBUST_CUTOFF` robust standard deviations, and for unusable points."""
    distance = np.linalg.norm(target - design @ coefficients, axis=1)
    scale = 1.4826 * np.median(distance[usable])
    if scale <= 0.0:
        return usable.astype(float)
    u = distance / (ROBUST_CUTOFF * scale)
    return np.where(usable & (u < 1.0), (1.0 - u**2) ** 2, 0.0)
