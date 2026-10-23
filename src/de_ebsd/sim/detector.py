"""EBSD detector geometry and per-pixel direction cosines.

Torch port of kikuchipy's ``EBSDDetector`` (Bruker pattern-center convention),
so patterns projected here match kikuchipy pixel for pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from de_ebsd.core.geometry import sample_to_detector_matrix

__all__ = ["EBSDDetector"]


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
        corners = np.array([
            [-x * width, y * height, z * height],
            [(1 - x) * width, y * height, z * height],
            [(1 - x) * width, -(1 - y) * height, z * height],
            [-x * width, -(1 - y) * height, z * height],
        ])
        return corners @ sample_to_detector_matrix(
            self.sample_tilt, self.tilt, self.azimuthal, self.twist
        )

    @property
    def pattern_center_sample_mm(self) -> np.ndarray:
        """Perpendicular foot from impact to screen, in sample coordinates."""
        distance = self.pattern_center[2] * self.nrows * self.px_size / 1000.0
        return distance * sample_to_detector_matrix(
            self.sample_tilt, self.tilt, self.azimuthal, self.twist
        )[2]

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
