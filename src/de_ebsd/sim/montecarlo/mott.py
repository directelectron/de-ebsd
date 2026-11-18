"""Mott elastic scattering for the Monte Carlo, from ELSEPA cross sections.

The default elastic model, screened Rutherford (EMsoft's ``EMMC.cl``), is the
first Born approximation for a point nucleus inside an exponentially screened
cloud. At SEM energies it is wrong in the way backscatter is most sensitive to:
it scatters too much between about 5 and 60 degrees and too little past 90 (and
below 3). For nickel at 15 keV its differential cross section is 1.3-1.7 times
Mott's from 5 to 20 degrees and 0.8 times past 120; the total cross sections
agree, but its transport cross section is 17 % high, and the backscatter
coefficient at normal incidence comes out 0.328 against 0.304 for Mott and about
0.30 measured. For gold the excess reaches 4 times at 10 degrees. Mott cross
sections come from the Dirac equation solved by partial waves in the atom's
self-consistent potential, so they keep the spin and screening the Born
approximation drops; they are what NIST SRD 64 and PENELOPE tabulate.

The cross sections here were computed with ELSEPA (Salvat, Jablonski and
Powell, Comput. Phys. Commun. 165 (2005) 157-190) for a neutral free atom with
its default model: Dirac-Fock electron density, Fermi nuclear charge
distribution, Furness-McCarthy exchange, no correlation-polarization or
absorption potential. ELSEPA is not called at run time; the package table
``data/mott_elsepa.npz`` holds its differential cross section at a reduced set of
its angles for a grid of energies, and ``scripts/generate_mott_tables.py`` in
the de-ebsd repository rebuilds it (the metadata records the exact settings).

Sampling is by inverse cumulative distribution: on loading, each tabulated
differential cross section is integrated over ``mu = (1 - cos theta) / 2`` as a
power law between its angles, and inverted onto an even grid of probabilities;
a step then reads the angle off that grid at a uniform random number, with the
grid interpolated between the two tabulated energies that bracket the
electron's (linearly in ``ln E``). The total cross section, interpolated the
same way in ``ln sigma``, sets the mean free path. In a compound each collision
picks its atom with probability proportional to its number fraction times its
total cross section, the independent-atom model.
"""

from __future__ import annotations

import functools
import json
from importlib import resources
from pathlib import Path

import numpy as np
import torch

from de_ebsd.crystal.data.scattering_tables import ELEMENT_SYMBOLS

__all__ = [
    "MottCrossSections",
    "MottSampler",
    "mott_cross_sections",
    "inverse_cdf",
]

# Points in the inverse-CDF grid each tabulated energy is resampled onto when
# loaded: about 1 MB a tabulated element in memory, and with the package
# table's 30 energies and 128 angles nickel's backscatter coefficient is within
# 0.001 of the one from ELSEPA's full output (119 energies, 606 angles).
N_PROBABILITY = 4097
_CM2_TO_NM2 = 1.0e14
_GENERATOR = "scripts/generate_mott_tables.py (de-ebsd repository)"


def _symbol(z: int) -> str:
    return ELEMENT_SYMBOLS[z - 1] if 1 <= z <= len(ELEMENT_SYMBOLS) else f"Z={z}"


def inverse_cdf(theta_deg, dcs, n_probability: int = N_PROBABILITY):
    """``mu = (1 - cos theta) / 2`` at evenly spaced cumulative probabilities.

    ``dcs`` is a differential cross section per solid angle at ``theta_deg``
    (ascending, from 0 to 180 degrees), shape ``(..., n_angles)``. Between two
    angles it is taken as a power law in ``mu`` (it spans many decades over the
    first degrees, where a linear interpolation would overstate it), the first
    interval as flat. ``dOmega = 4 pi dmu``, so the cumulative distribution is
    exact for that piecewise model and is inverted in closed form. Returns
    ``(mu, total)``: ``mu`` of shape ``(..., n_probability)`` at probabilities
    ``0, 1/(n-1), ..., 1``, and the integral of the model, the total cross
    section in the units of ``dcs`` times steradian.
    """
    theta = np.radians(np.asarray(theta_deg, dtype=np.float64))
    mu = 0.5 * (1.0 - np.cos(theta))
    mu[0], mu[-1] = 0.0, 1.0
    dcs = np.asarray(dcs, dtype=np.float64)
    lead = dcs.shape[:-1]
    dcs = dcs.reshape(-1, dcs.shape[-1])

    mu0, mu1 = mu[:-1], mu[1:]
    d0, d1 = dcs[:, :-1], dcs[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = np.log(d1 / d0) / np.log(mu1 / mu0)  # nan on the first interval
        power = exponent + 1.0
        ratio = mu1 / mu0
        segment = np.where(
            np.abs(power) > 1e-9,
            d0 * mu0 * (ratio**power - 1.0) / power,
            d0 * mu0 * np.log(ratio),
        )
    segment[:, 0] = dcs[:, 0] * mu1[0]  # flat from 0 to the first angle
    segment *= 4.0 * np.pi
    cumulative = np.concatenate([np.zeros((len(dcs), 1)), np.cumsum(segment, axis=1)], axis=1)
    total = cumulative[:, -1]

    probability = np.linspace(0.0, 1.0, n_probability)
    out = np.empty((len(dcs), n_probability))
    for row in range(len(dcs)):
        target = probability * total[row]
        k = np.clip(np.searchsorted(cumulative[row], target, side="right") - 1, 0, len(mu) - 2)
        excess = (target - cumulative[row, k]) / (4.0 * np.pi * d0[row, k])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = power[row, k]
            curved = mu0[k] * (1.0 + p * excess / mu0[k]) ** (1.0 / p)
            flat_in_log = mu0[k] * np.exp(excess / mu0[k])
            value = np.where(np.abs(p) > 1e-9, curved, flat_in_log)
        value = np.where(k == 0, excess, value)
        out[row] = np.clip(np.minimum(value, mu1[k]), mu0[k], 1.0)
    out[:, 0], out[:, -1] = 0.0, 1.0
    return out.reshape(*lead, n_probability), total.reshape(lead)


class MottCrossSections:
    """ELSEPA Mott elastic cross sections for a set of elements.

    Attributes
    ----------
    z
        Atomic numbers tabulated, ascending, shape ``(n_elements,)``.
    energy_ev
        Kinetic energies in eV, ascending, shape ``(n_energies,)``, common to
        every element.
    theta_deg
        Scattering angles of the differential cross section, from 0 to 180.
    log_dcs
        ``ln`` of the differential cross section per atom in cm^2/sr, shape
        ``(n_elements, n_energies, n_angles)``.
    total_cm2, transport_cm2
        ELSEPA's total and first transport cross sections per atom in cm^2,
        shape ``(n_elements, n_energies)``; the total is the one the mean free
        path uses, as ELSEPA integrated it on its full angular grid.
    metadata
        The generator's record: program, citation, ELSEPA settings, units.
    """

    def __init__(self, z, energy_ev, theta_deg, log_dcs, total_cm2, transport_cm2, metadata):
        self.z = np.asarray(z, dtype=np.int64)
        self.energy_ev = np.asarray(energy_ev, dtype=np.float64)
        self.theta_deg = np.asarray(theta_deg, dtype=np.float64)
        self.log_dcs = np.asarray(log_dcs, dtype=np.float64)
        self.total_cm2 = np.asarray(total_cm2, dtype=np.float64)
        self.transport_cm2 = np.asarray(transport_cm2, dtype=np.float64)
        self.metadata = dict(metadata)
        self._inverse: dict[int, np.ndarray] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> MottCrossSections:
        """Read a table written by ``scripts/generate_mott_tables.py``."""
        with np.load(path, allow_pickle=False) as data:
            low, high = data["log_dcs_low"][..., None], data["log_dcs_high"][..., None]
            log_dcs = low + data["log_dcs_u16"] / 65535.0 * (high - low)
            return cls(
                z=data["z"],
                energy_ev=data["energy_ev"],
                theta_deg=data["theta_deg"],
                log_dcs=log_dcs,
                total_cm2=data["total_cm2"],
                transport_cm2=data["transport_cm2"],
                metadata=json.loads(str(data["metadata"])),
            )

    @property
    def elements(self) -> tuple[str, ...]:
        """Symbols of the tabulated elements."""
        return tuple(_symbol(int(z)) for z in self.z)

    def index(self, z: int) -> int:
        """Row of element ``z``; a ``ValueError`` naming the generator if absent."""
        hits = np.flatnonzero(self.z == int(z))
        if len(hits) == 0:
            raise ValueError(
                f"No Mott cross-section table for {_symbol(int(z))} (Z={int(z)}). The table "
                f"has {', '.join(self.elements)}. Generate one that includes it with ELSEPA "
                f"and {_GENERATOR}, e.g. `--elements {_symbol(int(z))} ... --out my_table.npz`, "
                "and pass elastic=MottCrossSections.from_file('my_table.npz')."
            )
        return int(hits[0])

    def dcs_cm2_sr(self, z: int) -> np.ndarray:
        """Differential cross section per atom, ``(n_energies, n_angles)`` cm^2/sr."""
        return np.exp(self.log_dcs[self.index(z)])

    def inverse_cdf(self, z: int) -> np.ndarray:
        """``mu`` on :data:`N_PROBABILITY` even probabilities per energy (cached)."""
        z = int(z)
        if z not in self._inverse:
            self._inverse[z], _ = inverse_cdf(self.theta_deg, self.dcs_cm2_sr(z))
        return self._inverse[z]

    def __repr__(self) -> str:
        return (
            f"MottCrossSections({len(self.z)} elements, {len(self.energy_ev)} energies "
            f"{self.energy_ev[0]:g}-{self.energy_ev[-1]:g} eV, {len(self.theta_deg)} angles)"
        )


@functools.cache
def mott_cross_sections() -> MottCrossSections:
    """The package's ELSEPA table (``de_ebsd/sim/montecarlo/data/mott_elsepa.npz``)."""
    path = resources.files("de_ebsd.sim.montecarlo").joinpath("data", "mott_elsepa.npz")
    with resources.as_file(path) as local:
        return MottCrossSections.from_file(local)


class MottSampler:
    """Mean free path and polar angle by Mott scattering, batched in torch.

    Parameters
    ----------
    composition
        ``(Z, number fraction)`` pairs of the material; fractions need not sum
        to one.
    number_density
        Atoms per nm^3.
    tables
        The cross sections; default the package table.
    device
        Torch device the per-step lookups run on.

    Energies outside the table are clamped to its ends, which only matters
    below the table's lowest energy (50 eV in the package table), long after the
    electron has left the energy window the depth statistics keep.
    """

    def __init__(
        self,
        composition,
        number_density: float,
        tables: MottCrossSections | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        tables = mott_cross_sections() if tables is None else tables
        f64 = torch.float64
        numbers = [int(z) for z, _ in composition]
        fractions = np.array([float(x) for _, x in composition])
        rows = [tables.index(z) for z in numbers]
        self.device = torch.device("cpu" if device is None else device)
        self.number_density = float(number_density)
        self.log_energy = torch.as_tensor(
            np.log(tables.energy_ev * 1e-3), device=self.device, dtype=f64
        )
        # ln(x_i sigma_i): the per-atom-of-material cross section of each element
        self.log_weighted = torch.as_tensor(
            np.log(fractions[:, None] / fractions.sum()) + np.log(tables.total_cm2[rows]),
            device=self.device,
            dtype=f64,
        )
        self.inverse = torch.as_tensor(
            np.stack([tables.inverse_cdf(z) for z in numbers]), device=self.device, dtype=f64
        )
        self.n_probability = self.inverse.shape[-1]

    @property
    def n_elements(self) -> int:
        return len(self.inverse)

    def _position(self, energy_kev: torch.Tensor):
        x = torch.log(energy_kev).clamp(self.log_energy[0], self.log_energy[-1])
        n = len(self.log_energy)
        i = (torch.searchsorted(self.log_energy, x, right=True) - 1).clamp(0, n - 2)
        w = (x - self.log_energy[i]) / (self.log_energy[i + 1] - self.log_energy[i])
        return i, w

    def _cross_sections(self, i, w) -> torch.Tensor:
        """``x_k sigma_k(E)`` in nm^2, shape ``(n_elements, n)``."""
        log = self.log_weighted[:, i] * (1.0 - w) + self.log_weighted[:, i + 1] * w
        return torch.exp(log) * _CM2_TO_NM2

    def mean_free_path_nm(self, energy_kev: torch.Tensor) -> torch.Tensor:
        """``1 / (n sum_k x_k sigma_k(E))`` in nm."""
        i, w = self._position(energy_kev)
        return 1.0 / (self.number_density * self._cross_sections(i, w).sum(dim=0))

    def sample_cos(
        self, energy_kev: torch.Tensor, u: torch.Tensor, v: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``cos theta`` at uniform ``u``; ``v`` picks the atom in a compound."""
        i, w = self._position(energy_kev)
        if self.n_elements > 1:
            weights = self._cross_sections(i, w)
            cumulative = torch.cumsum(weights, dim=0)
            pick = v * cumulative[-1]
            element = (cumulative < pick[None, :]).sum(dim=0).clamp(max=self.n_elements - 1)
        else:
            element = torch.zeros_like(i)
        p = u * (self.n_probability - 1)
        j = p.floor().long().clamp(0, self.n_probability - 2)
        f = p - j
        lo = self._lookup(element, i, j, f)
        hi = self._lookup(element, i + 1, j, f)
        mu = lo * (1.0 - w) + hi * w
        return 1.0 - 2.0 * mu

    def _lookup(self, element, i, j, f):
        # a flat gather: indexing the (element, energy) rows first would copy a
        # whole probability grid per electron
        flat = self.inverse.reshape(-1)
        n_e, n_p = self.inverse.shape[1], self.inverse.shape[2]
        base = (element * n_e + i) * n_p + j
        return flat[base] * (1.0 - f) + flat[base + 1] * f
