"""EBSD detector geometry and per-pixel direction cosines.

Torch port of kikuchipy's ``EBSDDetector`` (Bruker pattern-center convention),
so patterns projected here match kikuchipy pixel for pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from torchebsd.core.geometry import sample_to_detector_matrix

__all__ = ["EBSDDetector"]


@dataclass(frozen=True)
class EBSDDetector:
    """An EBSD detector with a fixed pattern center.

    Parameters
    ----------
    shape
        (n_rows, n_cols) in pixels.
    pc
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

        om = sample_to_detector_matrix(self.sample_tilt, self.tilt, self.azimuthal, self.twist)
        dc = det.reshape(-1, 3) @ om  # (det_to_sample = om.T; row vectors: v @ om)
        dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
        return torch.as_tensor(dc, device=device, dtype=dtype)

    def show(self, microscope=None, crystal=None):
        """Interactive geometry widget (see :class:`torchebsd.viz.widgets.SetupWidget`)."""
        from torchebsd.viz.widgets import SetupWidget

        return SetupWidget.from_setup(microscope=microscope, detector=self, crystal=crystal)
