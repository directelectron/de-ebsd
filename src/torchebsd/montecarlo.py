"""Monte Carlo energy/depth statistics for the depth integration.

The Bloch-wave master pattern weights each depth slice by the number of
backscattered electrons that exit from that depth at each energy, taken from
an EMsoft Monte Carlo simulation (``EMMCOpenCL``). This module loads those
statistics from the EMsoft HDF5 file and reproduces the two derived
quantities the master-pattern program needs (``mod_EBSDmaster.f90``):

- the 99 % integration depth per energy bin (``izzmax`` / ``thick``),
- the depth weighting ``lambda_E(z)``, corrected for normal absorption with
  ``exp(+2 pi z / x_gp)`` so that the constant absorption already present in
  the Bloch eigenvalues is not double counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

__all__ = ["DepthDistribution"]


@dataclass
class DepthDistribution:
    """Backscatter yield versus energy and exit depth.

    Attributes
    ----------
    energies_kev
        Energy bin centers in keV, ascending, shape (nE,).
    depth_step
        Depth bin size in nm.
    accum_z
        Electron counts per (energy, depth, y, x), shape (nE, nz, ny, nx);
        the lateral window is EMsoft's central +/- numsx/10 region.
    n_total
        Total number of incident electrons (totnum_el * multiplier).
    """

    energies_kev: np.ndarray
    depth_step: float
    accum_z: np.ndarray
    n_total: float

    # ------------------------------------------------------------------ #
    @classmethod
    def from_emsoft(cls, path: str) -> DepthDistribution:
        """Load from an EMsoft Monte Carlo (or master pattern) HDF5 file."""
        with h5py.File(path, "r") as f:
            mc = f["EMData/MCOpenCL"]
            nml = f["NMLparameters/MCCLNameList"]
            # EMsoft writes Fortran order (nE, nz, nx, ny); h5py presents the
            # reversed C order (ny, nx, nz, nE).
            accum_z = np.asarray(mc["accum_z"]).transpose(3, 2, 1, 0).astype(np.float64)
            e_min = float(nml["Ehistmin"][0])
            e_bin = float(nml["Ebinsize"][0])
            n_e = accum_z.shape[0]
            energies = e_min + e_bin * np.arange(n_e)
            depth_step = float(nml["depthstep"][0])
            multiplier = float(np.asarray(mc.get("multiplier", [1]))[0])
            n_total = float(np.asarray(mc["totnum_el"])[0]) * multiplier
        return cls(
            energies_kev=energies,
            depth_step=depth_step,
            accum_z=accum_z,
            n_total=n_total,
        )

    @classmethod
    def single_energy(
        cls,
        energy_kev: float,
        thickness: float = 100.0,
        depth_step: float = 1.0,
        decay_length: float = 30.0,
    ) -> DepthDistribution:
        """A simple exponential exit-depth model for one energy bin.

        A stand-in when no Monte Carlo file is available; the master pattern
        then integrates ``exp(-z / decay_length)`` down to ``thickness`` nm.
        """
        n_z = int(round(thickness / depth_step))
        z = depth_step * np.arange(n_z)
        profile = np.exp(-z / decay_length)
        return cls(
            energies_kev=np.array([energy_kev]),
            depth_step=depth_step,
            accum_z=profile[None, :, None, None],
            n_total=float(profile.sum()),
        )

    # ------------------------------------------------------------------ #
    @property
    def n_energies(self) -> int:
        return len(self.energies_kev)

    @property
    def n_depths(self) -> int:
        return self.accum_z.shape[1]

    def energy_index(self, energy_kev: float) -> int:
        """Index of the energy bin closest to ``energy_kev``."""
        return int(np.argmin(np.abs(self.energies_kev - energy_kev)))

    def integration_depths(self, fraction: float = 0.99) -> np.ndarray:
        """Number of depth bins to integrate per energy (EMsoft ``izzmax``).

        For each lateral pixel the depth reaching ``fraction`` of that
        pixel's yield is found; the maximum over pixels (cumulative over
        energies, as in EMsoft) is returned per energy bin.
        """
        n_e, n_z = self.n_energies, self.n_depths
        izz = np.zeros(n_e, dtype=np.int64)
        running_max = 0
        for i_e in range(n_e):
            counts = self.accum_z[i_e]  # (nz, ny, nx)
            total = counts.sum(axis=0)
            cum = np.cumsum(counts, axis=0)
            need = fraction * total
            # first depth index where the cumulative count reaches the target
            reached = cum >= need[None, :, :]
            first = np.where(reached.any(axis=0), reached.argmax(axis=0), n_z - 1)
            running_max = max(running_max, int(first[total > 0].max(initial=0)) + 1)
            izz[i_e] = running_max
        return izz

    def depth_weights(self, energy_kev: float, normal_absorption_length: float) -> np.ndarray:
        """The weighting lambda_E(z) for one energy bin (EMsoft ``lambdaE``).

        The lateral-summed exit fraction per depth bin, multiplied by
        ``exp(+2 pi z / x_gp)`` to cancel the normal absorption that the
        Bloch eigenvalues already contain.
        """
        i_e = self.energy_index(energy_kev)
        fraction = self.accum_z[i_e].sum(axis=(1, 2)) / self.n_total
        z = self.depth_step * np.arange(self.n_depths)
        return fraction * np.exp(2.0 * np.pi * z / normal_absorption_length)

    def __repr__(self) -> str:
        e = self.energies_kev
        return (
            f"DepthDistribution({self.n_energies} energy bins "
            f"[{e.min():g}..{e.max():g} keV], {self.n_depths} depth bins x "
            f"{self.depth_step:g} nm, {self.n_total:.3g} electrons)"
        )
