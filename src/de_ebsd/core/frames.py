"""The scan's sample frame, and the turn that takes de-ebsd's orientations into it.

de-ebsd reports every orientation in **kikuchipy's sample frame**, the one
:meth:`~de_ebsd.sim.MasterPattern.project` simulates patterns in. A scan's
frame -- x along a row of the scan, y down its columns, z into the sample -- is
turned 90 degrees about the surface normal from it. An IPF-Z color cannot show
the difference (a turn about z leaves z where it was), so a map drawn without the
turn looks right while every orientation is 90 degrees off: measured against a
duplex-steel scan's ground truth, no pattern was within 5 degrees of the truth
until it was turned.

The turn is applied by the caller, at the edge, rather than inside an indexer:
an orientation in kikuchipy's frame is what kikuchipy, orix and EMsoft read, and
a conversion buried in one path is one the other paths silently lack. See
:doc:`/user/conventions`.
"""

from __future__ import annotations

import numpy as np

from de_ebsd.core.rotation import euler_to_matrix, matrix_to_euler

__all__ = ["SCAN_TURN", "scan_euler", "scan_tensor", "to_scan_frame"]

#: The turn acting on vectors: a direction ``v`` in kikuchipy's sample frame is
#: ``SCAN_TURN @ v`` in the scan's. An orientation ``g`` (sample to crystal)
#: becomes ``g @ SCAN_TURN.T``, since ``g v = (g SCAN_TURN^T)(SCAN_TURN v)``.
SCAN_TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def scan_euler(rotations) -> np.ndarray:
    """Orientation matrices ``(..., 3, 3)`` in kikuchipy's sample frame, such as an
    indexer's, as Bunge Euler angles ``(..., 3)`` in radians in the scan's."""
    return matrix_to_euler(np.asarray(rotations, dtype=np.float64) @ SCAN_TURN.T)


def to_scan_frame(euler) -> np.ndarray:
    """Bunge Euler angles ``(..., 3)`` in radians, from kikuchipy's sample frame to
    the scan's: :func:`scan_euler` for orientations held as angles."""
    return scan_euler(euler_to_matrix(euler))


def scan_tensor(tensors) -> np.ndarray:
    """Tensors ``(..., 3, 3)`` in kikuchipy's sample frame, such as a strain, in the
    scan's: turned as the orientations are, so a strain map and the orientation map
    beside it share axes."""
    return np.einsum("ij,...jk,lk->...il", SCAN_TURN, np.asarray(tensors), SCAN_TURN)
