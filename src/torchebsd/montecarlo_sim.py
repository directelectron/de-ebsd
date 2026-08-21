"""Native electron Monte Carlo for exit depth/energy statistics (M-MC).

Scoped in ``MC_DESIGN.md``: an EMsoft-independent Monte Carlo that produces the
depth statistics (:class:`torchebsd.montecarlo.DepthDistribution`) the Bloch
depth integration consumes, removing the last EMsoft runtime dependency.

This is M-MC.1, the material foundation. EMsoft's Monte Carlo treats the sample
as a homogeneous bulk with mean properties rather than resolving individual
atoms, so the elastic (screened Rutherford) and inelastic (CSDA stopping power)
models need only the composition-averaged atomic number and weight, the mass and
number densities, and the mean ionization potential. :class:`MaterialProperties`
reduces a :class:`~torchebsd.crystal.Crystal` to exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchebsd.constants import C_AVOGADRO
from torchebsd.crystal import Crystal

__all__ = ["mean_ionization_potential_kev", "MaterialProperties"]


def mean_ionization_potential_kev(z: float) -> float:
    """Berger-Seltzer mean ionization potential J in keV (Joy-Luo form).

    ``J[eV] = 9.76 Z + 58.5 Z^(-0.19)`` (Joy & Luo, 1989) -- the standard mean
    ionization energy used by the modified-Bethe CSDA stopping power. ``z`` may
    be a fractional composition-mean atomic number.
    """
    if z <= 0:
        raise ValueError("atomic number must be positive")
    return (9.76 * z + 58.5 * z**-0.19) * 1e-3


@dataclass(frozen=True)
class MaterialProperties:
    """Bulk-target properties the Monte Carlo scatters in (a mean material).

    Attributes
    ----------
    mean_z, mean_a
        Composition- (number-) averaged atomic number and weight (g/mol).
    density
        Mass density in g/cm^3.
    number_density
        Atomic number density in atoms/nm^3.
    mean_ionization_kev
        Berger-Seltzer mean ionization potential J in keV.
    """

    mean_z: float
    mean_a: float
    density: float
    number_density: float
    mean_ionization_kev: float

    @classmethod
    def from_crystal(cls, crystal: Crystal) -> MaterialProperties:
        """Reduce a crystal to the mean homogeneous material the MC uses."""
        mean_z = crystal.mean_atomic_number
        mean_a = crystal.mean_atomic_weight
        density = crystal.density
        # atoms/nm^3 = (g/cm^3)(atoms/mol)/(g/mol) [=atoms/cm^3] * 1e-21 cm^3/nm^3
        number_density = density * C_AVOGADRO / mean_a * 1e-21
        return cls(
            mean_z=mean_z,
            mean_a=mean_a,
            density=density,
            number_density=number_density,
            mean_ionization_kev=mean_ionization_potential_kev(mean_z),
        )

    def __repr__(self) -> str:
        return (
            f"MaterialProperties(Z={self.mean_z:.3g}, A={self.mean_a:.4g} g/mol, "
            f"rho={self.density:.3g} g/cm^3, n={self.number_density:.4g} /nm^3, "
            f"J={self.mean_ionization_kev * 1e3:.1f} eV)"
        )
