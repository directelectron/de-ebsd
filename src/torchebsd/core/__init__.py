"""Frame conventions, physical constants and coordinate maps shared package-wide."""

from torchebsd.core.constants import (
    electron_wavelength_nm,
    interaction_constant,
    relativistic_correction,
)
from torchebsd.core.geometry import sample_to_detector_matrix, sample_to_detector_tensor
from torchebsd.core.lambert import master_pattern_grid, sphere_to_square, square_to_sphere
from torchebsd.core.rotation import quaternion_to_matrix, rotate_vectors

__all__ = [
    "electron_wavelength_nm",
    "interaction_constant",
    "master_pattern_grid",
    "quaternion_to_matrix",
    "relativistic_correction",
    "rotate_vectors",
    "sample_to_detector_matrix",
    "sample_to_detector_tensor",
    "sphere_to_square",
    "square_to_sphere",
]
