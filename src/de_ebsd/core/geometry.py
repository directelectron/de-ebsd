"""Sample <-> detector frame geometry.

Shared by the forward projection (:mod:`de_ebsd.sim.detector`) and the
back-projection (:mod:`de_ebsd.index.backproject`) so both use one frame
convention. Port of kikuchipy's ``_sample_to_detector_matrix``.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["detector_rays", "sample_to_detector_matrix", "sample_to_detector_tensor"]


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


def detector_rays(detector, pc: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor):
    """Detector-frame rays ``(..., K, 3)`` through fractional pixels, for a pattern center.

    The un-normalized ray is

        ``x_d = aspect * ((col + 0.5) / ncols - pcx)``,
        ``y_d = pcy - (row + 0.5) / nrows``,
        ``z_d = pcz``,

    in pure torch arithmetic, so it differentiates in ``pc`` (unlike the
    ``grid_sample`` back-projection). ``pc`` is ``(..., 3)`` and broadcasts against
    the ``(..., K)`` ``rows`` and ``cols``.
    """
    aspect = detector.ncols / detector.nrows
    pcx, pcy, pcz = pc[..., 0:1], pc[..., 1:2], pc[..., 2:3]
    x_d = aspect * ((cols + 0.5) / detector.ncols - pcx)
    y_d = pcy - (rows + 0.5) / detector.nrows
    z_d = pcz + torch.zeros_like(x_d)
    return torch.stack([x_d, y_d, z_d], dim=-1)
