"""Orientation maps against their own neighborhoods: speckle, and seeds to retry it with.

A point indexed into the wrong basin of the score -- a look-alike orientation, a
pseudo-symmetric partner -- shows on an orientation map as a single point that
agrees with none of its neighbors: speckle. Its neighbors' orientations are the
seeds most likely to be right for it, which :meth:`LiveIndexer.retry
<de_ebsd.index.live.LiveIndexer.retry>` refines and keeps where they score better.

A map is a grid ``(rows, cols)`` of orientations, as rotation matrices ``(rows, cols,
3, 3)`` (sample to crystal, as :class:`~de_ebsd.index.live.LiveIndexer` gives them)
or orix-ordered quaternions ``(rows, cols, 4)``, NumPy or torch; the neighbors of a
point are the four sharing an edge with it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SPECKLE_DEG",
    "NeighborSeeds",
    "neighbor_misorientation",
    "neighbor_seeds",
    "speckle_fraction",
    "speckled",
]

#: How far, in degrees, a point must be from every neighbor to be speckle: well
#: above a grain's own spread and a refine's error, well below any grain boundary
#: worth the name.
SPECKLE_DEG = 5.0

#: The four neighbors, as (row, column) offsets, in the order the last axis of
#: :func:`neighbor_misorientation` and :class:`NeighborSeeds` keeps them.
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def neighbor_slices(shape) -> Iterator[tuple[int, tuple[slice, slice], tuple[slice, slice]]]:
    """For each of :data:`NEIGHBORS`: ``(number, here, there)``, *here* the points
    of a ``(rows, cols)`` grid that have that neighbor and *there* the neighbors,
    as slices of the grid, so ``grid[here]`` and ``grid[there]`` line up."""
    rows, cols = int(shape[0]), int(shape[1])
    for number, (dr, dc) in enumerate(NEIGHBORS):
        here = (slice(max(0, -dr), rows - max(0, dr)), slice(max(0, -dc), cols - max(0, dc)))
        there = (slice(max(0, dr), rows + min(0, dr)), slice(max(0, dc), cols + min(0, dc)))
        yield number, here, there


def _numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _matrices(orientations) -> np.ndarray:
    """*orientations* ``(rows, cols, 3, 3)`` or quaternions ``(rows, cols, 4)`` as
    float64 matrices ``(rows, cols, 3, 3)``."""
    o = _numpy(orientations).astype(np.float64)
    if o.ndim == 4 and o.shape[-2:] == (3, 3):
        return o
    if o.ndim == 3 and o.shape[-1] == 4:
        from orix.quaternion import Rotation

        return Rotation(o.reshape(-1, 4)).to_matrix().reshape(*o.shape[:2], 3, 3)
    raise ValueError(
        f"orientations must be (rows, cols, 3, 3) matrices or (rows, cols, 4) "
        f"quaternions, got {o.shape}"
    )


def _operators(symmetry) -> np.ndarray:
    """A point group's proper rotations as matrices ``(K, 3, 3)``: an orix
    ``Symmetry``, or the matrices themselves. An improper operator never relates two
    orientations of one crystal, and its trace would read as a smaller angle."""
    if hasattr(symmetry, "to_matrix"):
        proper = getattr(symmetry, "proper_subgroup", symmetry)
        return np.asarray(proper.to_matrix(), dtype=np.float64)
    operators = np.asarray(symmetry, dtype=np.float64).reshape(-1, 3, 3)
    return operators[np.linalg.det(operators) > 0]


def _per_phase(symmetries) -> list[np.ndarray]:
    if hasattr(symmetries, "to_matrix") or (
        isinstance(symmetries, np.ndarray) and symmetries.ndim in (2, 3)
    ):
        return [_operators(symmetries)]
    return [_operators(s) for s in symmetries]


def _pair_angles(a: np.ndarray, b: np.ndarray, operators: np.ndarray) -> np.ndarray:
    """Degrees between orientations *a* and *b* ``(..., 3, 3)``, the smallest over
    *operators*: the trace of ``S a b^T`` for every operator at once, a group
    holding each operator's transpose too."""
    turn = a @ np.swapaxes(b, -1, -2)
    traces = np.einsum("kij,...ij->...k", operators, turn).max(-1)
    return np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))


def neighbor_misorientation(orientations, symmetries, phase=None) -> np.ndarray:
    """``(rows, cols, 4)``: each point's misorientation in degrees to each of its
    four neighbors (:data:`NEIGHBORS`), the smallest over its phase's point group.

    Parameters
    ----------
    orientations
        The map, ``(rows, cols, 3, 3)`` matrices or ``(rows, cols, 4)`` quaternions.
    symmetries
        The point group (an orix ``Symmetry`` or its operators ``(K, 3, 3)``), or
        one per phase.
    phase
        ``(rows, cols)`` phase numbers into *symmetries*; all zero when None.

    Returns
    -------
    numpy.ndarray
        NaN where there is no neighbor (the map's edges), infinity where the
        neighbor is of another phase.
    """
    g = _matrices(orientations)
    rows, cols = g.shape[:2]
    groups = _per_phase(symmetries)
    phase = np.zeros((rows, cols), dtype=np.int64) if phase is None else _numpy(phase)
    if phase.shape != (rows, cols):
        raise ValueError(f"phase is {phase.shape}, the map is {(rows, cols)}")
    angles = np.full((rows, cols, len(NEIGHBORS)), np.nan)
    for number, here, there in neighbor_slices((rows, cols)):
        a, b = g[here], g[there]
        same = phase[here] == phase[there]
        out = np.full(same.shape, np.inf)
        for p, operators in enumerate(groups):
            where = same & (phase[here] == p)
            out[where] = _pair_angles(a[where], b[where], operators)
        angles[here + (number,)] = out
    return angles


def speckled(
    orientations, symmetries, phase=None, *, threshold_deg: float = SPECKLE_DEG
) -> np.ndarray:
    """``(rows, cols)`` booleans: the points more than *threshold_deg* from every
    one of their neighbors (of another phase counting as further), each judged on
    the neighbors it has. See :func:`neighbor_misorientation`."""
    angles = neighbor_misorientation(orientations, symmetries, phase)
    has = ~np.isnan(angles)
    return has.any(-1) & ((angles > threshold_deg) | ~has).all(-1)


def speckle_fraction(
    orientations, symmetries, phase=None, *, threshold_deg: float = SPECKLE_DEG
) -> float:
    """The share of a map's points that are speckle: more than *threshold_deg*
    (default :data:`SPECKLE_DEG`) from every one of their neighbors, symmetry
    reduced within a phase. See :func:`speckled`."""
    return float(speckled(orientations, symmetries, phase, threshold_deg=threshold_deg).mean())


@dataclass
class NeighborSeeds:
    """Seeds to retry a map's speckle with: each speckled point's own orientation and
    its four neighbors'.

    Attributes
    ----------
    points
        ``(M,)`` the speckled points' flat (row-major) indices into the map.
    rotations
        ``(M, 5, 3, 3)`` float32: each point's own orientation first, so that a retry
        keeps it unless a neighbor's scores better, then its four neighbors'
        (:data:`NEIGHBORS`), the point's own again where it has no such neighbor.
    phases
        ``(M, 5)`` int64: the seeds' phases.
    entries
        ``(M, 5)`` int64: the seeds' dictionary entries, where the map's were
        given, else -1.
    valid
        ``(M, 5)`` booleans: whether the seed exists (the point's own always does).
    """

    points: np.ndarray
    rotations: np.ndarray
    phases: np.ndarray
    entries: np.ndarray
    valid: np.ndarray

    def __len__(self) -> int:
        return len(self.points)


def neighbor_seeds(
    orientations,
    symmetries,
    phase=None,
    entry=None,
    *,
    threshold_deg: float = SPECKLE_DEG,
) -> NeighborSeeds:
    """The map's speckled points (:func:`speckled`) and their neighbors' orientations,
    phases and dictionary entries, to give :meth:`LiveIndexer.retry
    <de_ebsd.index.live.LiveIndexer.retry>` as seeds.

    Parameters
    ----------
    orientations, symmetries, phase, threshold_deg
        As :func:`speckled`.
    entry
        ``(rows, cols)`` the map's dictionary entries (as
        :meth:`LiveIndexer.index <de_ebsd.index.live.LiveIndexer.index>` returns
        them), carried along so a retried point keeps an entry; None for none.
    """
    g = _matrices(orientations)
    rows, cols = g.shape[:2]
    phase = np.zeros((rows, cols), dtype=np.int64) if phase is None else _numpy(phase)
    entry = np.full((rows, cols), -1, dtype=np.int64) if entry is None else _numpy(entry)
    mask = speckled(g, symmetries, phase, threshold_deg=threshold_deg)
    # each point itself, then its neighbors, as a (rows, cols, 5) stack: the point
    # itself again where it has no such neighbor
    seeds = 1 + len(NEIGHBORS)
    rotations = np.repeat(g[:, :, None], seeds, 2)
    phases = np.repeat(phase[:, :, None], seeds, 2).astype(np.int64)
    entries = np.repeat(entry[:, :, None], seeds, 2).astype(np.int64)
    valid = np.zeros((rows, cols, seeds), dtype=bool)
    valid[:, :, 0] = True
    for number, here, there in neighbor_slices((rows, cols)):
        rotations[here + (1 + number,)] = g[there]
        phases[here + (1 + number,)] = phase[there]
        entries[here + (1 + number,)] = entry[there]
        valid[here + (1 + number,)] = True
    points = np.flatnonzero(mask)
    flat = rows * cols
    return NeighborSeeds(
        points=points,
        rotations=rotations.reshape(flat, seeds, 3, 3)[points].astype(np.float32),
        phases=phases.reshape(flat, -1)[points],
        entries=entries.reshape(flat, -1)[points],
        valid=valid.reshape(flat, -1)[points],
    )
