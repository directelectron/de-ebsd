"""Sample <-> detector frame geometry.

Shared by the forward projection (:mod:`torchebsd.sim.detector`) and the
back-projection (:mod:`torchebsd.index.backproject`) so both use one frame
convention. Port of kikuchipy's ``_sample_to_detector_matrix``.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["sample_to_detector_matrix", "sample_to_detector_tensor"]


def _rodrigues_step(basis: np.ndarray, axis_row: int, angle: float) -> np.ndarray:
    u = basis[axis_row]
    u = u / np.linalg.norm(u)
    c, s = np.cos(angle), np.sin(angle)
    new = np.empty_like(basis)
    for j in range(3):
        v = basis[j]
        new[j] = v * c + np.cross(u, v) * s + u * np.dot(u, v) * (1.0 - c)
    return new


def sample_to_detector_matrix(
    sample_tilt: float, tilt: float, azimuthal: float, twist: float
) -> np.ndarray:
    """Rows are the detector basis (X_d, Y_d, Z_d) in sample coordinates.

    Angles in degrees. Depends on the tilt geometry only, not the pattern center.
    """
    basis = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    sigma, theta, omega, gamma = np.deg2rad([sample_tilt, tilt, azimuthal, twist])
    for axis_row, angle in zip([0, 0, 1, 2], [-sigma, theta, -omega, -gamma], strict=True):
        basis = _rodrigues_step(basis, axis_row, angle)
    return basis


def sample_to_detector_tensor(detector, device, dtype: torch.dtype) -> torch.Tensor:
    """:func:`sample_to_detector_matrix` for a detector, as a torch tensor."""
    om = sample_to_detector_matrix(
        detector.sample_tilt, detector.tilt, detector.azimuthal, detector.twist
    )
    return torch.as_tensor(om, device=device, dtype=dtype)
