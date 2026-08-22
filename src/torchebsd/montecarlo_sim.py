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
from typing import TYPE_CHECKING

import numpy as np
import torch

from torchebsd.constants import C_AVOGADRO
from torchebsd.crystal import Crystal

if TYPE_CHECKING:
    from torchebsd.montecarlo import DepthDistribution

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
    "simulate_depth_distribution",
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
        EMsoft ``depth`` value at exit: the pre-step depth over the exit
        direction's z-cosine (a slant path to the surface). Signed like EMsoft's
        ``depth[]``; ``accum_z`` bins its magnitude.
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
    penetrated = False
    n_steps = 0
    for _ in range(max_steps):
        n_steps += 1
        alpha = _SCREEN_C * z ** (2.0 / 3.0) / energy
        mfp = mean_free_path_nm(z, energy, rho, a_weight)
        step = -mfp * np.log(1.0 - rng.random())  # exponential free path (nm)

        energy = energy + step * rho * stopping_power(z, energy, a_weight, j_kev)

        cos_theta = sample_scattering_cos(alpha, rng.random())
        psi = 2.0 * np.pi * rng.random()
        c_new = rotate_direction(c, cos_theta, psi)

        # EMsoft escape_depth: pre-move depth over the new direction's z-cosine
        # (a slant path to the surface); binning takes its magnitude.
        if abs(c_new[2]) > 1e-5:
            escape_depth = r[2] / c_new[2]
        r = r + step * c_new
        c = c_new

        max_depth = max(max_depth, r[2])
        if r[2] <= 0.0:
            # a surface reflection that never entered the bulk is not a
            # backscatter event (it sampled no depth); EMsoft drops it too
            backscattered = penetrated
            break
        penetrated = True
        if energy <= e_min_kev:
            break

    return TrajectoryResult(backscattered, energy, escape_depth, max_depth, n_steps)


# --------------------------------------------------------------------------- #
# M-MC.3: vectorized (batched) Monte Carlo -> DepthDistribution
# --------------------------------------------------------------------------- #
def _rotate_batch(c: torch.Tensor, cos_theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Batched EMsoft direction-cosine rotation for a [N, 3] stack of directions."""
    ct = cos_theta
    st = torch.sqrt(torch.clamp(1.0 - ct * ct, min=0.0))
    cp, sp = torch.cos(psi), torch.sin(psi)
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]
    dsqi = 1.0 / torch.sqrt(torch.clamp(1.0 - cz * cz, min=1e-30))
    gx = st * (cx * cz * cp - cy * sp) * dsqi + cx * ct
    gy = st * (cy * cz * cp + cx * sp) * dsqi + cy * ct
    gz = -st * cp / dsqi + cz * ct
    pole = cz.abs() >= 0.99999
    nx = torch.where(pole, st * cp, gx)
    ny = torch.where(pole, st * sp, gy)
    nz = torch.where(pole, torch.sign(cz) * ct, gz)
    return torch.stack([nx, ny, nz], dim=-1)


def simulate_depth_distribution(
    material: MaterialProperties | Crystal,
    beam_energy_kev: float,
    *,
    e_min_kev: float | None = None,
    e_bin_kev: float = 1.0,
    depth_step_nm: float = 1.0,
    depth_max_nm: float = 100.0,
    n_electrons: int = 100_000,
    tilt_deg: float = 70.0,
    azimuth_deg: float = 0.0,
    max_steps: int = 1000,
    device: str | torch.device | None = None,
    seed: int | None = None,
) -> DepthDistribution:
    """Native (EMsoft-free) exit depth/energy statistics via a batched MC.

    Vectorizes :func:`simulate_trajectory` over ``n_electrons`` advanced in
    lockstep (an alive mask retires each one as it backscatters or is absorbed),
    then bins the backscattered electrons into EMsoft's ``accum_z`` layout --
    energy ``iE = round((E - Ehistmin)/Ebinsize)``, depth
    ``iz = round(|escape_depth|/depthstep)`` -- and returns a
    :class:`~torchebsd.montecarlo.DepthDistribution`, a drop-in for
    :meth:`~torchebsd.montecarlo.DepthDistribution.from_emsoft`.

    This is the M-MC.3 hard-mode forward. It counts every backscattered electron
    (no lateral exit-direction window yet; EMsoft's central +/- numsx/10 accum_z
    window is the reconciliation item for M-MC.4). ``e_min_kev`` doubles as the
    energy-bin floor (``Ehistmin``) and the absorption cutoff -- an electron that
    slows below it can only lose more energy, so it could never land in a bin.
    """
    from torchebsd.montecarlo import DepthDistribution

    if isinstance(material, Crystal):
        material = MaterialProperties.from_crystal(material)
    if e_min_kev is None:
        e_min_kev = beam_energy_kev - 5.0

    dev = torch.device(device) if device is not None else torch.device("cpu")
    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(int(seed))

    z, a_weight = float(material.mean_z), float(material.mean_a)
    rho, j_kev = float(material.density), float(material.mean_ionization_kev)
    z23 = z ** (2.0 / 3.0)
    n = int(n_electrons)
    f64 = torch.float64

    sig, omega = np.deg2rad(tilt_deg), np.deg2rad(azimuth_deg)
    c0 = torch.tensor(
        [np.cos(omega) * np.sin(sig), np.sin(omega) * np.sin(sig), np.cos(sig)],
        device=dev, dtype=f64,
    )
    pos = torch.zeros(n, 3, device=dev, dtype=f64)
    direction = c0.expand(n, 3).clone()
    energy = torch.full((n,), float(beam_energy_kev), device=dev, dtype=f64)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    penetrated = torch.zeros(n, dtype=torch.bool, device=dev)
    escape_depth = torch.zeros(n, device=dev, dtype=f64)
    exit_energy = torch.zeros(n, device=dev, dtype=f64)
    backscattered = torch.zeros(n, dtype=torch.bool, device=dev)

    def _rand():
        return torch.rand(n, generator=gen, device=dev, dtype=f64)

    for _ in range(max_steps):
        if not bool(alive.any()):
            break
        e = energy
        alpha = _SCREEN_C * z23 / e
        rel = ((e + _REST_KEV) / (e + _REST2_KEV)) ** 2
        sig_ena = _SIGMA_C * z * z / (e * e) * 4.0 * np.pi / (alpha * (1.0 + alpha)) * rel
        mfp = a_weight * _CM_TO_NM / (rho * sig_ena)
        step = -mfp * torch.log(1.0 - _rand())  # exponential free path (nm)
        de_ds = -_STOP_C * z / (a_weight * e) * torch.log(_STOP_A * e / j_kev + _STOP_B)

        cos_theta = 1.0 - 2.0 * alpha * (u := _rand()) / (1.0 + alpha - u)
        psi = 2.0 * np.pi * _rand()
        c_new = _rotate_batch(direction, cos_theta, psi)

        cz = c_new[:, 2]
        upd = alive & (cz.abs() > 1e-5)
        escape_depth = torch.where(upd, pos[:, 2] / cz, escape_depth)

        new_pos = pos + step[:, None] * c_new
        new_e = e + step * rho * de_ds
        exited = alive & (new_pos[:, 2] <= 0.0)
        absorbed = alive & ~exited & (new_e <= e_min_kev)

        # only electrons that entered the bulk count as backscattered -- a step-1
        # surface reflection sampled no depth (escape_depth 0); EMsoft drops it
        valid = exited & penetrated
        backscattered = backscattered | valid
        exit_energy = torch.where(valid, new_e, exit_energy)
        penetrated = penetrated | (alive & (new_pos[:, 2] > 0.0))
        pos = torch.where(alive[:, None], new_pos, pos)
        direction = torch.where(alive[:, None], c_new, direction)
        energy = torch.where(alive, new_e, energy)
        alive = alive & ~exited & ~absorbed

    # bin backscattered electrons into EMsoft's accum_z (E, z) layout
    num_e_bins = int(round((beam_energy_kev - e_min_kev) / e_bin_kev)) + 1
    num_z_bins = int(depth_max_nm / depth_step_nm) + 1
    edis = escape_depth.abs()
    i_e = torch.round((exit_energy - e_min_kev) / e_bin_kev).long()
    i_z = torch.round(edis / depth_step_nm).long()
    valid = (
        backscattered
        & (exit_energy > e_min_kev) & (exit_energy <= beam_energy_kev)
        & (i_e >= 0) & (i_e < num_e_bins)
        & (i_z >= 0) & (i_z < num_z_bins)
    )
    accum = torch.zeros(num_e_bins * num_z_bins, device=dev, dtype=f64)
    flat = i_e[valid] * num_z_bins + i_z[valid]
    accum.index_add_(0, flat, torch.ones(int(valid.sum()), device=dev, dtype=f64))
    accum_z = accum.reshape(num_e_bins, num_z_bins).cpu().numpy()[:, :, None, None]
    energies_kev = e_min_kev + e_bin_kev * np.arange(num_e_bins)

    return DepthDistribution(
        energies_kev=energies_kev,
        depth_step=float(depth_step_nm),
        accum_z=accum_z,
        n_total=float(n),
    )
