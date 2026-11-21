"""A finished scan's speckle re-tried from its neighbors' orientations.

Once a scan is complete, its orientation map shows where the search sent a
pattern into the wrong basin: a single point that agrees with none of its
neighbors (:func:`~de_ebsd.index.neighbors.speckled`). Its neighbors'
orientations are the seeds most likely to be right for it, so each such point is
refined again from its own orientation and from each neighbor's, and the one that
matches its pattern best is kept (:meth:`LiveIndexer.retry
<de_ebsd.index.live.LiveIndexer.retry>`). A point a neighbor's orientation does
not match better keeps its own, so a real one-point grain stays.

A grain boundary leaves a different mistake: a rim of points whose patterns blend
both grains', matched by an orientation that is neither. The rim agrees with
itself along the boundary, so it is not speckle; it is found by its low score
(*score_fraction*) and seeded from the grains a few steps out on either side
(*radius*).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from de_ebsd.index.neighbors import SPECKLE_DEG, neighbor_seeds

__all__ = ["RetriedMap", "retry_speckle"]


@dataclass
class RetriedMap:
    """A map after :func:`retry_speckle`, as NumPy arrays over the scan's points
    (row-major).

    Attributes
    ----------
    phase, g, score, entry
        ``(N,)``, ``(N, 3, 3)``, ``(N,)`` and ``(N,)``: the map with the retried
        points that a neighbor's orientation matched better replaced.
    changed
        ``(N,)`` booleans: the points replaced.
    tried
        How many points were retried, over every round.
    """

    phase: np.ndarray
    g: np.ndarray
    score: np.ndarray
    entry: np.ndarray
    changed: np.ndarray
    tried: int


def _numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def retry_speckle(
    indexer,
    frames,
    phase,
    g,
    score,
    entry=None,
    *,
    shape,
    threshold_deg: float = SPECKLE_DEG,
    rounds: int = 2,
    fit: bool = False,
    score_fraction: float | None = None,
    radius: int = 1,
) -> RetriedMap:
    """Retry a finished scan's speckle, and its low-scoring points, from their
    neighbors' orientations.

    Parameters
    ----------
    indexer
        The :class:`~de_ebsd.index.live.LiveIndexer` the scan was indexed with,
        its static background as the scan left it.
    frames
        The scan's patterns ``(N, H, W)``, in the map's row-major order: anything
        :meth:`LiveIndexer.index <de_ebsd.index.live.LiveIndexer.index>` takes, or
        a NumPy memory map of a saved movie, of which only the speckled points'
        frames are read.
    phase, g, score, entry
        The finished map, ``(N,)``, ``(N, 3, 3)``, ``(N,)`` and ``(N,)`` (*entry*
        may be None), as :meth:`~de_ebsd.index.live.LiveIndexer.index` gave it.
    shape
        The scan's ``(rows, cols)``.
    threshold_deg
        How far a point must be from every neighbor to be retried; see
        :func:`~de_ebsd.index.neighbors.speckled`.
    rounds
        How many times to look for speckle again after replacing some: two
        neighboring wrong points can only be seeded right once one of them is.
    fit
        Fit the retried orientations, as ``index(..., fit=True)`` does; pass what
        the scan was indexed with, so a retried point's score compares with its
        neighbors'.
    score_fraction
        Retry too the points scoring below this share of the map's median score
        (a grain boundary's rim); None for speckle alone. See
        :func:`~de_ebsd.index.neighbors.neighbor_seeds`.
    radius
        How many steps out the seeds are taken from: 1, the four neighbors, or
        more, the eight directions at each step
        (:func:`~de_ebsd.index.neighbors.seed_offsets`).
    """
    rows, cols = int(shape[0]), int(shape[1])
    phase = _numpy(phase).astype(np.int64).copy()
    g = _numpy(g).astype(np.float32).reshape(-1, 3, 3).copy()
    score = _numpy(score).astype(np.float32).copy()
    entry = (
        np.full(len(phase), -1, dtype=np.int64)
        if entry is None
        else _numpy(entry).astype(np.int64).copy()
    )
    if len(phase) != rows * cols:
        raise ValueError(f"{len(phase)} points for a {rows} x {cols} scan")
    changed = np.zeros(len(phase), dtype=bool)
    tried = 0
    for _ in range(max(1, int(rounds))):
        seeds = neighbor_seeds(
            g.reshape(rows, cols, 3, 3),
            indexer.point_groups,
            phase.reshape(rows, cols),
            entry.reshape(rows, cols),
            threshold_deg=threshold_deg,
            score=score.reshape(rows, cols),
            score_fraction=score_fraction,
            radius=radius,
        )
        if not len(seeds):
            break
        tried += len(seeds)
        points = seeds.points
        if isinstance(frames, torch.Tensor):
            picked = frames[torch.as_tensor(points, device=frames.device)]
        else:
            picked = np.ascontiguousarray(frames[points])
        new_phase, new_g, new_score, chosen = (
            _numpy(column)
            for column in indexer.retry(
                picked, seeds.rotations, seeds.phases, valid=seeds.valid, fit=fit
            )
        )
        moved = chosen != 0
        if not moved.any():
            break
        where = points[moved]
        phase[where] = new_phase[moved]
        g[where] = new_g[moved]
        score[where] = new_score[moved]
        entry[where] = seeds.entries[moved, chosen[moved]]
        changed[where] = True
    return RetriedMap(phase=phase, g=g, score=score, entry=entry, changed=changed, tried=tried)
