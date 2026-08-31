"""Microscope: beam energy and sample geometry.

The :class:`Microscope` collects the experimental parameters that belong to
the instrument rather than to the crystal or the detector: accelerating
voltage and the sample tilt. Detector geometry lives on
:class:`torchebsd.sim.detector.EBSDDetector`.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchebsd.core.constants import (
    electron_wavelength_nm,
    interaction_constant,
    relativistic_correction,
)

__all__ = ["Microscope"]


@dataclass(frozen=True)
class Microscope:
    """An SEM configured for EBSD.

    Parameters
    ----------
    energy
        Beam energy in eV (e.g. ``20e3``).
    sample_tilt
        Sample tilt from the horizontal in degrees (70 is standard for EBSD).
    azimuthal
        Detector azimuthal angle with respect to the tilt axis, in degrees.
    """

    energy: float = 20e3
    sample_tilt: float = 70.0
    azimuthal: float = 0.0

    def __post_init__(self) -> None:
        if self.energy <= 0:
            raise ValueError("energy must be positive (in eV)")

    @property
    def energy_kev(self) -> float:
        """Beam energy in keV."""
        return self.energy / 1e3

    @property
    def gamma(self) -> float:
        """Relativistic correction factor 1 + e V / (m0 c^2)."""
        return relativistic_correction(self.energy_kev)

    @property
    def vacuum_wavelength(self) -> float:
        """Relativistic electron wavelength in nm, without refraction correction.

        The in-crystal wavelength (corrected for the mean inner potential)
        is computed by the simulator once structure factors are available;
        see :attr:`torchebsd.crystal.structure_factors.StructureFactors.wavelength`.
        """
        return electron_wavelength_nm(self.energy_kev, v0mod=0.0)

    @property
    def sigma(self) -> float:
        """Interaction constant in (V nm)^-1."""
        return interaction_constant(self.energy_kev, v0mod=0.0)

    def show(self, detector=None, crystal=None):
        """Display the microscope and detector setup as an editable widget."""
        from torchebsd.viz.widgets import MicroscopeWidget

        return MicroscopeWidget.from_setup(
            microscope=self, detector=detector, crystal=crystal
        )

    def __repr__(self) -> str:
        return (
            f"Microscope(energy={self.energy:g} eV, sample_tilt={self.sample_tilt:g} deg, "
            f"azimuthal={self.azimuthal:g} deg)"
        )
