"""EBSD detector geometry and direction cosines.

The :class:`EBSDDetector` describes the pixel grid, the projection center
(Bruker convention), and the tilt geometry, and produces the unit direction
vector of every pixel in the sample reference frame. The math is a torch port
of kikuchipy's ``EBSDDetector`` (gnomonic bounds from the Britton et al. 2016
tutorial conventions, the sample-to-detector rotation, and the direction
cosines of ``_get_direction_cosines_from_detector``), so patterns projected
from a master pattern here match kikuchipy pixel for pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["EBSDDetector"]


def _rodrigues_step(basis: np.ndarray, axis_row: int, angle: float) -> np.ndarray:
    """Rotate all basis rows about one current basis row by ``angle`` (radians)."""
    u = basis[axis_row]
    u = u / np.linalg.norm(u)
    c, s = np.cos(angle), np.sin(angle)
    new = np.empty_like(basis)
    for j in range(3):
        v = basis[j]
        new[j] = v * c + np.cross(u, v) * s + u * np.dot(u, v) * (1.0 - c)
    return new


def _sample_to_detector_matrix(
    sample_tilt: float, tilt: float, azimuthal: float, twist: float
) -> np.ndarray:
    """Rows are the detector basis (X_d, Y_d, Z_d) in sample coordinates.

    Port of kikuchipy ``_sample_to_detector_matrix``; angles in degrees.
    """
    basis = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    sigma, theta, omega, gamma = np.deg2rad([sample_tilt, tilt, azimuthal, twist])
    for axis_row, angle in zip([0, 0, 1, 2], [-sigma, theta, -omega, -gamma], strict=True):
        basis = _rodrigues_step(basis, axis_row, angle)
    return basis


@dataclass(frozen=True)
class EBSDDetector:
    """An EBSD detector with a fixed projection center.

    Parameters
    ----------
    shape
        (n_rows, n_cols) in pixels.
    pc
        Projection center (PCx, PCy, PCz) in the Bruker convention.
    sample_tilt
        Sample tilt in degrees (usually set from ``Microscope.sample_tilt``).
    tilt
        Detector tilt in degrees.
    azimuthal
        Detector azimuthal angle in degrees.
    twist
        Detector twist about its normal, in degrees.
    px_size
        Pixel size in micrometers (metadata; the projection itself only
        depends on the PC).
    """

    shape: tuple[int, int] = (480, 480)
    pc: tuple[float, float, float] = (0.5, 0.5, 0.5)
    sample_tilt: float = 70.0
    tilt: float = 0.0
    azimuthal: float = 0.0
    twist: float = 0.0
    px_size: float = 1.0

    @property
    def nrows(self) -> int:
        return int(self.shape[0])

    @property
    def ncols(self) -> int:
        return int(self.shape[1])

    @property
    def size(self) -> int:
        return self.nrows * self.ncols

    @property
    def aspect_ratio(self) -> float:
        return self.ncols / self.nrows

    @property
    def gnomonic_bounds(self) -> np.ndarray:
        """Detector bounds [x_min, x_max, y_min, y_max] in gnomonic coordinates."""
        pcx, pcy, pcz = self.pc
        x_min = -self.aspect_ratio * pcx / pcz
        x_max = self.aspect_ratio * (1.0 - pcx) / pcz
        y_min = -(1.0 - pcy) / pcz
        y_max = pcy / pcz
        return np.array([x_min, x_max, y_min, y_max])

    def direction_cosines(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Unit direction vectors (n_rows * n_cols, 3) in the sample frame.

        Row-major over the detector (row 0 = top). Port of kikuchipy
        ``_get_direction_cosines_for_fixed_pc``.
        """
        bounds = self.gnomonic_bounds
        pcz = self.pc[2]
        x_scale = (bounds[1] - bounds[0]) / self.ncols
        y_scale = (bounds[3] - bounds[2]) / self.nrows

        gn_x = bounds[0] + x_scale * np.arange(self.ncols) + 0.5 * x_scale
        gn_y = bounds[3] - y_scale * np.arange(self.nrows) - 0.5 * y_scale

        det = np.empty((self.nrows, self.ncols, 3))
        det[..., 0] = gn_x[None, :] * pcz
        det[..., 1] = gn_y[:, None] * pcz
        det[..., 2] = pcz

        om = _sample_to_detector_matrix(self.sample_tilt, self.tilt, self.azimuthal, self.twist)
        dc = det.reshape(-1, 3) @ om  # (det_to_sample = om.T; row vectors: v @ om)
        dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
        return torch.as_tensor(dc, device=device, dtype=dtype)

    def show(self, microscope=None, crystal=None):
        """Interactive geometry widget (see :class:`torchebsd.widgets.SetupWidget`)."""
        from torchebsd.widgets import SetupWidget

        return SetupWidget.from_setup(microscope=microscope, detector=self, crystal=crystal)
