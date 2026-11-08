"""Detector geometry and per-pixel direction cosines.

:class:`EBSDDetector` is a torch port of kikuchipy's ``EBSDDetector`` (Bruker
pattern-center convention), so patterns projected here match kikuchipy pixel for
pixel. :class:`TEMDetector` is a transmission screen normal to the beam, for
diffraction spots. Both turn their own frame (x right, y up the screen, z from the
sample towards it) into the sample frame with :meth:`detector_to_sample`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from de_ebsd.core.geometry import sample_to_detector_matrix

__all__ = ["EBSDDetector", "TEMDetector"]


@dataclass(frozen=True)
class EBSDDetector:
    """An EBSD detector with a fixed pattern center.

    Parameters
    ----------
    shape
        (n_rows, n_cols) in pixels.
    pattern_center
        Pattern center (PCx, PCy, PCz) in the Bruker convention.
    sample_tilt
        Sample tilt in degrees (usually set from ``Microscope.sample_tilt``).
    tilt
        Detector tilt in degrees.
    azimuthal
        Detector azimuthal angle in degrees.
    twist
        Detector twist about its normal, in degrees.
    px_size
        Effective screen pixel pitch in micrometers, including binning (not
        camera sensor pitch). Used for physical dimensions in the inspector;
        pattern projection itself only depends on the normalized PC.
    """

    shape: tuple[int, int] = (480, 480)
    pattern_center: tuple[float, float, float] = (0.5, 0.5, 0.5)
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
        pcx, pcy, pcz = self.pattern_center
        x_min = -self.aspect_ratio * pcx / pcz
        x_max = self.aspect_ratio * (1.0 - pcx) / pcz
        y_min = -(1.0 - pcy) / pcz
        y_max = pcy / pcz
        return np.array([x_min, x_max, y_min, y_max])

    @property
    def corners_sample_mm(self) -> np.ndarray:
        """Top-left clockwise corners relative to the beam impact, in mm.

        Uses the same Bruker PC and frame transform as direction_cosines.
        Physical distances assume px_size is the effective screen pixel pitch
        (including binning), not the camera sensor's pixel pitch.
        """
        height = self.nrows * self.px_size / 1000.0
        width = self.ncols * self.px_size / 1000.0
        x, y, z = self.pattern_center
        corners = np.array(
            [
                [-x * width, y * height, z * height],
                [(1 - x) * width, y * height, z * height],
                [(1 - x) * width, -(1 - y) * height, z * height],
                [-x * width, -(1 - y) * height, z * height],
            ]
        )
        return corners @ sample_to_detector_matrix(
            self.sample_tilt, self.tilt, self.azimuthal, self.twist
        )

    @property
    def pattern_center_sample_mm(self) -> np.ndarray:
        """Perpendicular foot from impact to screen, in sample coordinates."""
        distance = self.pattern_center[2] * self.nrows * self.px_size / 1000.0
        return (
            distance
            * sample_to_detector_matrix(self.sample_tilt, self.tilt, self.azimuthal, self.twist)[2]
        )

    def detector_to_sample(self) -> np.ndarray:
        """The rotation taking detector-frame directions into the sample frame: the
        transpose of :func:`~de_ebsd.core.geometry.sample_to_detector_matrix`."""
        return sample_to_detector_matrix(self.sample_tilt, self.tilt, self.azimuthal, self.twist).T

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
        pcz = self.pattern_center[2]
        x_scale = (bounds[1] - bounds[0]) / self.ncols
        y_scale = (bounds[3] - bounds[2]) / self.nrows

        gn_x = bounds[0] + x_scale * np.arange(self.ncols) + 0.5 * x_scale
        gn_y = bounds[3] - y_scale * np.arange(self.nrows) - 0.5 * y_scale

        det = np.empty((self.nrows, self.ncols, 3))
        det[..., 0] = gn_x[None, :] * pcz
        det[..., 1] = gn_y[:, None] * pcz
        det[..., 2] = pcz

        om = sample_to_detector_matrix(self.sample_tilt, self.tilt, self.azimuthal, self.twist)
        dc = det.reshape(-1, 3) @ om  # (det_to_sample = om.T; row vectors: v @ om)
        dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
        return torch.as_tensor(dc, device=device, dtype=dtype)

    def show(self, microscope=None, crystal=None):
        """Interactive screen placement and PC editor, also exportable to HTML.

        If a microscope is supplied, its sample tilt and azimuth must match.
        Edits are a separate snapshot; read them with widget.to_detector().
        """
        from de_ebsd.viz.widgets import MicroscopeWidget

        return MicroscopeWidget.from_setup(microscope=microscope, detector=self, crystal=crystal)


def _rotation_about(axis, degrees: float) -> np.ndarray:
    """Right-handed rotation by *degrees* about unit *axis* (Rodrigues)."""
    u = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    a = np.deg2rad(degrees)
    return np.eye(3) + np.sin(a) * k + (1.0 - np.cos(a)) * (k @ k)


@dataclass(frozen=True)
class TEMDetector:
    """A transmission diffraction screen, normal to the beam and centered on it.

    The detector frame is :class:`EBSDDetector`'s -- x right, y up the screen, z
    along the beam from the sample to the screen -- so a direction ``(x, y, 1)`` has
    gnomonic coordinates ``(x, y)``. Only the stage tilts turn the sample relative
    to it.

    Parameters
    ----------
    shape
        (n_rows, n_cols) in pixels.
    camera_length
        Camera length, in mm.
    size
        Width of the screen (its columns), in mm; the rows span the same pixel
        pitch. Only its ratio to *camera_length* matters.
    sample_tilt
        Stage tilt alpha, in degrees, about the detector x axis.
    tilt_beta
        Stage tilt beta, in degrees, about the detector y axis, applied first.
    """

    shape: tuple[int, int] = (256, 256)
    camera_length: float = 300.0
    size: float = 20.0
    sample_tilt: float = 0.0
    tilt_beta: float = 0.0

    def __post_init__(self) -> None:
        if not (self.camera_length > 0 and self.size > 0):
            raise ValueError("camera_length and size must be positive (mm)")

    @property
    def nrows(self) -> int:
        return int(self.shape[0])

    @property
    def ncols(self) -> int:
        return int(self.shape[1])

    @property
    def gnomonic_bounds(self) -> np.ndarray:
        """Screen edges [x_min, x_max, y_min, y_max] in gnomonic coordinates."""
        half = self.size / 2.0 / self.camera_length
        tall = half * self.nrows / self.ncols
        return np.array([-half, half, -tall, tall])

    def detector_to_sample(self) -> np.ndarray:
        """The rotation taking detector-frame directions into the sample frame:
        alpha about x, then beta about y."""
        return _rotation_about((1.0, 0.0, 0.0), self.sample_tilt) @ _rotation_about(
            (0.0, 1.0, 0.0), self.tilt_beta
        )

    def pixels(self, gnomonic) -> np.ndarray:
        """Fractional ``(row, col)`` of gnomonic points ``(..., 2)``, pixel centers on
        integers, as :meth:`direction_cosines` samples them."""
        gnomonic = np.asarray(gnomonic, dtype=np.float64)
        x_min, x_max, y_min, y_max = self.gnomonic_bounds
        col = (gnomonic[..., 0] - x_min) / (x_max - x_min) * self.ncols - 0.5
        row = (y_max - gnomonic[..., 1]) / (y_max - y_min) * self.nrows - 0.5
        return np.stack([row, col], axis=-1)

    def direction_cosines(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Unit direction vectors (n_rows * n_cols, 3) to every pixel center, in the
        sample frame, row-major (row 0 = top)."""
        x_min, x_max, y_min, y_max = self.gnomonic_bounds
        x = x_min + (np.arange(self.ncols) + 0.5) * (x_max - x_min) / self.ncols
        y = y_max - (np.arange(self.nrows) + 0.5) * (y_max - y_min) / self.nrows
        det = np.empty((self.nrows, self.ncols, 3))
        det[..., 0] = x[None, :]
        det[..., 1] = y[:, None]
        det[..., 2] = 1.0
        rays = det.reshape(-1, 3) @ self.detector_to_sample().T
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return torch.as_tensor(rays, device=device, dtype=dtype)
