"""Microscope: beam energy and sample geometry.

The :class:`Microscope` collects the experimental parameters that belong to
the instrument rather than to the crystal or the detector: accelerating
voltage and the sample tilt. Detector geometry lives on
:class:`de_ebsd.sim.detector.EBSDDetector`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from de_ebsd.core.constants import (
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
    probe_diameter_nm
        Nominal circular beam diameter perpendicular to the beam, in nm.
        Optional geometry metadata, not a finite-probe scattering model.
    """

    energy: float = 20e3
    sample_tilt: float = 70.0
    azimuthal: float = 0.0
    probe_diameter_nm: float | None = None

    def __post_init__(self) -> None:
        if self.energy <= 0:
            raise ValueError("energy must be positive (in eV)")
        if self.probe_diameter_nm is not None and (
            not math.isfinite(self.probe_diameter_nm) or self.probe_diameter_nm <= 0
        ):
            raise ValueError("probe_diameter_nm must be finite and positive")

    @property
    def probe_footprint_nm(self) -> tuple[float, float] | None:
        """Major/minor diameters on a flat surface; None at grazing incidence.

        Geometric projection of a circular parallel probe, not an interaction
        volume, beam broadening prediction, or a measured spatial resolution.
        """
        cosine = math.cos(math.radians(self.sample_tilt))
        if self.probe_diameter_nm is None or cosine <= 1e-12:
            return None
        return self.probe_diameter_nm / cosine, self.probe_diameter_nm

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
        see :attr:`de_ebsd.crystal.structure_factors.StructureFactors.wavelength`.
        """
        return electron_wavelength_nm(self.energy_kev, v0mod=0.0)

    @property
    def sigma(self) -> float:
        """Interaction constant in (V nm)^-1."""
        return interaction_constant(self.energy_kev, v0mod=0.0)

    def show(self, detector=None, crystal=None):
        """Editable beam, footprint and optional detector geometry inspector.

        When supplied, detector sample tilt/azimuth must agree with this object.
        Edits are a separate snapshot; read them with widget.to_microscope().
        """
        from de_ebsd.viz.widgets import MicroscopeWidget

        return MicroscopeWidget.from_setup(
            microscope=self, detector=detector, crystal=crystal
        )

    def __repr__(self) -> str:
        return (
            f"Microscope(energy={self.energy:g} eV, sample_tilt={self.sample_tilt:g} deg, "
            f"azimuthal={self.azimuthal:g} deg)"
        )
