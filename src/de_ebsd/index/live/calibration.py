"""Calibrating the pattern center from a buffer of live frames."""

from __future__ import annotations

import math

import torch

from de_ebsd.index.live.common import parabola_offset, pick

#: Pattern-center steps (fractions of the detector, as the center is), each half the
#: last. On a twin scan (1024 patterns, 33 grains), starts 0.004-0.01 off in every coordinate came back
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
