"""de-ebsd's sample frame, which is OIM's (EDAX TSL).

Every orientation de-ebsd reports -- an indexer's, a refine's, a dictionary
entry's -- and every tensor beside it, such as a strain, is in **OIM's (EDAX
TSL) sample frame**: the frame :meth:`~de_ebsd.sim.MasterPattern.project`
simulates patterns in, which is also kikuchipy's. OIM is the reference: a real
nickel scan indexed by de-ebsd and re-indexed by OIM at the same pattern center
agrees point by point to a fraction of a degree, with no conversion between the
two. So an orientation goes into an ``.ang``, an orix ``CrystalMap``, OIM, MTEX
or kikuchipy exactly as de-ebsd returns it. See :doc:`/user/conventions`.

The functions here used to turn orientations 90 degrees about the surface
normal, into what was called "the scan's frame". That turn was fitted to a
simulator's ground truth, not to an instrument, and put every orientation 90
degrees from OIM's. It is gone; the functions are kept only so that code calling
them keeps running, and they return their input unturned, with a warning.
"""

from __future__ import annotations

import warnings

import numpy as np

from de_ebsd.core.rotation import matrix_to_euler

__all__ = ["scan_euler", "scan_tensor", "to_scan_frame"]

_UNTURNED = (
    "{name} is deprecated: de-ebsd's orientations and tensors are already in OIM's "
    "(EDAX TSL) sample frame, so it no longer turns anything{instead}"
)


def _deprecated(name: str, instead: str = ".") -> None:
    warnings.warn(_UNTURNED.format(name=name, instead=instead), DeprecationWarning, stacklevel=3)


def scan_euler(rotations) -> np.ndarray:
    """Deprecated: orientation matrices ``(..., 3, 3)`` as Bunge Euler angles
    ``(..., 3)`` in radians, unturned. Use
    :func:`~de_ebsd.core.rotation.matrix_to_euler`."""
    _deprecated("scan_euler", "; use de_ebsd.core.rotation.matrix_to_euler.")
    return matrix_to_euler(np.asarray(rotations, dtype=np.float64))


def to_scan_frame(euler) -> np.ndarray:
    """Deprecated: Bunge Euler angles ``(..., 3)``, returned as they are."""
    _deprecated("to_scan_frame")
    return np.array(euler, dtype=np.float64)


def scan_tensor(tensors) -> np.ndarray:
    """Deprecated: tensors ``(..., 3, 3)``, returned as they are."""
    _deprecated("scan_tensor")
    return np.array(tensors)
