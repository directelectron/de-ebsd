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

import numpy as np

from torchebsd.constants import C_AVOGADRO
from torchebsd.crystal import Crystal

__all__ = [
    "mean_ionization_potential_kev",
    "MaterialProperties",
    "screening_parameter",
    "elastic_cross_section_na",
    "mean_free_path_nm",
    "stopping_power",
    "sample_scattering_cos",
    "rotate_direction",
    "TrajectoryResult",
    "simulate_trajectory",
]

# --- EMsoft Monte Carlo model constants (opencl/EMMC.cl, screened Rutherford +
# Joy-Luo CSDA). All energies keV, density g/cm^3, atomic weight g/mol, lengths
# nm. Numeric literals are EMsoft's, kept verbatim for parity. -----------------
_SCREEN_C = 3.4e-3  # alpha = _SCREEN_C * Z^(2/3) / E
_SIGMA_C = 5.21 * 602.2  # N_A * 5.21e-21 cm^2 relativistic Rutherford prefactor
_REST_KEV = 511.0  # relativistic factor ((E+511)/(E+1024))^2
_REST2_KEV = 1024.0
_STOP_C = 0.00785  # Joy-Luo mass stopping-power prefactor
_STOP_A = 1.166  # stopping-power log scale:  ln(1.166 E/J + 0.9911)
_STOP_B = 0.9911  # stopping-power log offset
_CM_TO_NM = 1.0e7


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


# --------------------------------------------------------------------------- #
# M-MC.2: single-scattering physics (EMsoft EMMC.cl) and a reference trajectory
# --------------------------------------------------------------------------- #
def screening_parameter(z, energy_kev):
    """Rutherford screening parameter ``alpha = 3.4e-3 Z^(2/3) / E`` (E in keV)."""
    return _SCREEN_C * np.asarray(z, dtype=np.float64) ** (2.0 / 3.0) / energy_kev


def elastic_cross_section_na(z, energy_kev):
    """``N_A * sigma`` for relativistic screened Rutherford (EMsoft ``sig_eNA``).

    ``5.21*602.2 (Z^2/E^2) 4pi/(alpha(1+alpha)) ((E+511)/(E+1024))^2`` -- the
    Avogadro-scaled total elastic cross-section that feeds the mean free path.
    """
    z = np.asarray(z, dtype=np.float64)
    alpha = screening_parameter(z, energy_kev)
    rel = ((energy_kev + _REST_KEV) / (energy_kev + _REST2_KEV)) ** 2
    return _SIGMA_C * z * z / energy_kev**2 * 4.0 * np.pi / (alpha * (1.0 + alpha)) * rel


def mean_free_path_nm(z, energy_kev, density, a_weight):
    """Elastic mean free path in nm: ``A * 1e7 / (rho * N_A sigma)`` (EMsoft)."""
    return a_weight * _CM_TO_NM / (density * elastic_cross_section_na(z, energy_kev))


def stopping_power(z, energy_kev, a_weight, j_kev):
    """Joy-Luo modified-Bethe stopping power ``de_ds`` (EMsoft form).

    ``-0.00785 Z/(A E) ln(1.166 E/J + 0.9911)``. This is EMsoft's mass-scaled
    ``de_ds``; the energy lost over a step is ``step_nm * density * de_ds`` (keV),
    so the value is negative (energy decreases). ``J`` is the mean ionization
    potential in keV (see :func:`mean_ionization_potential_kev`).
    """
    z = np.asarray(z, dtype=np.float64)
    return -_STOP_C * z / (a_weight * energy_kev) * np.log(_STOP_A * energy_kev / j_kev + _STOP_B)


def sample_scattering_cos(alpha, u):
    """Cosine of the polar scattering angle, inverse-CDF of screened Rutherford.

    ``cos(theta) = 1 - 2 alpha u / (1 + alpha - u)`` for ``u`` uniform on [0, 1)
    (EMsoft). ``u -> 0`` is forward (cos=1); ``u -> 1`` is backscatter (cos=-1).
    """
    return 1.0 - 2.0 * alpha * u / (1.0 + alpha - u)


def rotate_direction(c: np.ndarray, cos_theta: float, psi: float) -> np.ndarray:
    """New unit direction after scattering by polar ``theta`` / azimuth ``psi``.

    EMsoft's direction-cosine rotation (``EMMC.cl``), with the ``|c_z| ~ 1``
    special case handled separately to avoid the ``1/sin`` singularity.
    """
    ct = cos_theta
    st = np.sqrt(max(0.0, 1.0 - ct * ct))
    cp, sp = np.cos(psi), np.sin(psi)
    cx, cy, cz = c
    if abs(cz) >= 0.99999:
        return np.array([st * cp, st * sp, (1.0 if cz >= 0 else -1.0) * ct])
    dsqi = 1.0 / np.sqrt(1.0 - cz * cz)
    return np.array([
        st * (cx * cz * cp - cy * sp) * dsqi + cx * ct,
        st * (cy * cz * cp + cx * sp) * dsqi + cy * ct,
        -st * cp / dsqi + cz * ct,
    ])


@dataclass
class TrajectoryResult:
    """Outcome of one electron trajectory.

    Attributes
    ----------
    backscattered
        True if the electron re-crossed the entry surface (z <= 0).
    exit_energy_kev
        Kinetic energy when the trajectory ended.
    escape_depth_nm
        EMsoft ``escape_depth = z / c_z`` at the final step (the depth measure
        binned into ``accum_z``; its exact role is finalized in M-MC.3).
    max_depth_nm
        Deepest z reached along the trajectory.
    n_steps
        Number of scattering steps taken.
    """

    backscattered: bool
    exit_energy_kev: float
    escape_depth_nm: float
    max_depth_nm: float
    n_steps: int


def simulate_trajectory(
    material: MaterialProperties,
    energy_kev: float,
    tilt_deg: float = 70.0,
    azimuth_deg: float = 0.0,
    e_min_kev: float = 0.05,
    max_steps: int = 1000,
    rng: np.random.Generator | None = None,
) -> TrajectoryResult:
    """Track one electron through the bulk (EMsoft single-scattering model).

    A faithful scalar port of ``EMMC.cl``: enter a ``tilt_deg`` sample at the
    origin, and per step draw an exponential free path, lose energy by the
    Joy-Luo CSDA, and scatter by screened Rutherford until the electron exits the
    surface (backscattered), drops below ``e_min_kev`` (absorbed), or reaches
    ``max_steps``. The vectorized torch port (M-MC.3) mirrors this arithmetic.
    """
    if rng is None:
        rng = np.random.default_rng()
    z, a_weight = material.mean_z, material.mean_a
    rho, j_kev = material.density, material.mean_ionization_kev

    sig, omega = np.deg2rad(tilt_deg), np.deg2rad(azimuth_deg)
    r = np.zeros(3)
    c = np.array([np.cos(omega) * np.sin(sig), np.sin(omega) * np.sin(sig), np.cos(sig)])
    energy = float(energy_kev)

    escape_depth = 0.0
    max_depth = 0.0
    backscattered = False
    n_steps = 0
    for _ in range(max_steps):
        n_steps += 1
        alpha = _SCREEN_C * z ** (2.0 / 3.0) / energy
        mfp = mean_free_path_nm(z, energy, rho, a_weight)
        step = -mfp * np.log(1.0 - rng.random())  # exponential free path (nm)

        energy = energy + step * rho * stopping_power(z, energy, a_weight, j_kev)

        cos_theta = sample_scattering_cos(alpha, rng.random())
        psi = 2.0 * np.pi * rng.random()
        c = rotate_direction(c, cos_theta, psi)
        r = r + step * c

        if abs(c[2]) > 1e-5:
            escape_depth = r[2] / c[2]
        max_depth = max(max_depth, r[2])
        if r[2] <= 0.0:
            backscattered = True
            break
        if energy <= e_min_kev:
            break

    return TrajectoryResult(backscattered, energy, escape_depth, max_depth, n_steps)
