"""The master-pattern simulation chain.

:class:`MasterPatternSimulator` composes a crystal, a microscope and a backend,
exposing each intermediate (structure factors, reflections, k-vectors) as an
inspectable, showable object before ``compute()`` runs the full simulation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from de_ebsd.crystal.lattice import Crystal
from de_ebsd.crystal.structure_factors import StructureFactors, compute_structure_factors
from de_ebsd.diffraction.backends import (
    DynamicalMatrix,
    MasterPatternBackend,
    SimulationInputs,
    build_dynamical_matrix,
    classify_reflections,
    resolve_backend,
)
from de_ebsd.diffraction.bethe import BetheParameters
from de_ebsd.diffraction.kvectors import reduce_to_fundamental_sector, sample_incident_directions
from de_ebsd.diffraction.reflections import ReflectionList, master_reflection_list
from de_ebsd.sim.masterpattern import MasterPattern
from de_ebsd.sim.microscope import Microscope
from de_ebsd.sim.montecarlo.depth import DepthDistribution

__all__ = ["MasterPatternSimulator"]


class MasterPatternSimulator:
    """Simulate an EBSD master pattern for one crystal and beam energy.

    Parameters
    ----------
    crystal
        The crystal structure.
    microscope
        Beam energy and sample geometry.
    npx
        Half-width of the square Lambert grid; the master pattern maps have
        shape ``(2 npx + 1, 2 npx + 1)`` per hemisphere.
    dmin
        Smallest interplanar spacing (nm) still included as a reflection.
        Lowering it adds beams and raises accuracy and cost.
    bethe
        Bethe-potential truncation thresholds.
    monte_carlo
        Energy/depth statistics for the depth integration (required by the
        Bloch-wave backend; ignored by the kinematical backend).
    backend
        ``"kinematical"`` (band geometry only: a fast placeholder), ``"bloch"``
        (the physical Bloch-wave model), or any
        :class:`~de_ebsd.diffraction.backends.MasterPatternBackend` instance.
    """

    def __init__(
        self,
        crystal: Crystal,
        microscope: Microscope,
        npx: int = 150,
        dmin: float = 0.1,
        bethe: BetheParameters | None = None,
        monte_carlo: DepthDistribution | None = None,
        backend: str | MasterPatternBackend = "kinematical",
    ) -> None:
        if npx < 1:
            raise ValueError("npx must be >= 1")
        if dmin <= 0:
            raise ValueError("dmin must be positive (nm)")
        self.crystal = crystal
        self.microscope = microscope
        self.npx = int(npx)
        self.dmin = float(dmin)
        self.bethe = bethe if bethe is not None else BetheParameters()
        self.monte_carlo = monte_carlo
        self.backend = backend
        self._sf: StructureFactors | None = None

    @property
    def energy_kev(self) -> float:
        return self.microscope.energy_kev

    def structure_factors(self) -> StructureFactors:
        """U_g with absorption for all reflections down to dmin (cached)."""
        if self._sf is None:
            self._sf = compute_structure_factors(self.crystal, self.energy_kev, self.dmin)
        return self._sf

    def reflections(self, k=None) -> ReflectionList:
        """The reflection list; with ``k``, classified strong/weak for that beam.

        ``k`` is a beam direction in the Cartesian crystal frame (any length;
        it is normalized). For cubic crystals Cartesian components coincide
        with lattice directions.
        """
        sf = self.structure_factors()
        if k is None:
            return master_reflection_list(sf)
        return classify_reflections(sf, _unit(k), self.bethe)

    def dynamical_matrix(self, k) -> DynamicalMatrix:
        """The Bloch dynamical matrix (with Bethe corrections) for one beam."""
        sf = self.structure_factors()
        refl = classify_reflections(sf, _unit(k), self.bethe)
        matrix = build_dynamical_matrix(sf, refl)
        return DynamicalMatrix(matrix=matrix, hkl=refl.strong(), k=refl.k, reflections=refl)

    def compute(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        chunk_size: int = 262_144,
        mc_electrons: int = 200_000,
        symmetry: bool = True,
        energy_resolved: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> MasterPattern:
        """Compute the master pattern on the Lambert grid.

        Parameters
        ----------
        device
            Torch device (e.g. ``"cuda"``); default: torch's current default.
        dtype
            Floating point type for the direction batch (FP32 recommended).
        chunk_size
            Number of beam directions per batch, bounding peak memory.
        mc_electrons
            Electron count for the native Monte Carlo depth profile built when
            the backend needs one and no ``monte_carlo`` was supplied. Ignored if
            a depth profile is provided or the backend does not use one.
        symmetry
            Fold the beam directions onto the point-group fundamental sector so
            each symmetry-distinct direction is computed once (exact; ~15x fewer
            evaluations for cubic). Disable to evaluate every grid point.
        energy_resolved
            Compute one Lambert map per Monte Carlo energy bin (structure factors
            and depth profile per energy), giving an ``(nE, 2, n, n)`` master.
            Requires a depth-integrating backend; ignored otherwise. Default is a
            single map at the beam energy.
        progress
            Called as ``progress(done, total)`` after each batch of beam
            directions, counting batches over all energies, so a caller can show
            how far a long simulation has got. At the default *chunk_size* a
            master of any usual size is one batch per energy, so a single-energy
            compute reports only its end. The Monte Carlo, the symmetry fold and
            the first energy all run before the first call. It is called on the
            calling thread, inside the loop, so it should be quick and never
            block; an exception raised in it stops the simulation.
        """
        device = torch.device(device) if device is not None else torch.device("cpu")
        backend = resolve_backend(self.backend)

        # depth-integrating backends need a Monte Carlo exit-depth profile; when
        # the user supplied none, run the native (EMsoft-free) MC at the
        # experimental tilt so a plain compute() stays standalone and accurate.
        monte_carlo = self.monte_carlo
        if monte_carlo is None and backend.uses_depth_profile:
            monte_carlo = DepthDistribution.from_simulation(
                self.crystal, self.energy_kev,
                tilt_deg=self.microscope.sample_tilt,
                n_electrons=mc_electrons, device=device, seed=0,
            )

        # beam directions and the (energy-independent) symmetry fold
        grid = sample_incident_directions(self.npx, device=device, dtype=dtype)
        directions = grid.flat()
        if symmetry:
            reps, inverse = reduce_to_fundamental_sector(
                directions, self.crystal.space_group, cache_key=self.npx)
        else:
            reps, inverse = directions, None
        n = 2 * self.npx + 1

        if energy_resolved and backend.uses_depth_profile:
            energies = np.asarray(monte_carlo.energies_kev, dtype=float)
        else:
            energies = np.array([self.energy_kev])

        starts = range(0, reps.shape[0], chunk_size)
        total, done = len(energies) * len(starts), 0
        maps = []
        for energy in energies:
            sf = (self.structure_factors() if abs(energy - self.energy_kev) < 1e-9
                  else compute_structure_factors(self.crystal, float(energy), self.dmin))
            backend.prepare(SimulationInputs(
                sf=sf, bethe=self.bethe, monte_carlo=monte_carlo, device=device, dtype=dtype,
            ))
            rep_values = torch.empty(reps.shape[0], device=device, dtype=dtype)
            for start in starts:
                stop = min(start + chunk_size, reps.shape[0])
                rep_values[start:stop] = backend.intensities(reps[start:stop])
                done += 1
                if progress is not None:
                    progress(done, total)
            values = rep_values if inverse is None else rep_values[inverse]
            maps.append(values.reshape(2, n, n))

        return MasterPattern(
            data=torch.stack(maps, dim=0),
            energies_kev=energies,
            metadata={
                # the object, not its repr: the indexers derive the phase and
                # point group from it (save() stringifies metadata on write)
                "crystal": self.crystal,
                "backend": backend.name,
                "npx": self.npx,
                "dmin": self.dmin,
                "energy_kev": self.energy_kev,
                "bethe": repr(self.bethe),
                "energy_resolved": len(energies) > 1,
            },
        )

    def refine(self, **changes) -> MasterPatternSimulator:
        """A new simulator with updated parameters, reusing valid caches.

        Accepted keys: ``npx``, ``dmin``, ``bethe``, ``backend``,
        ``monte_carlo``, ``microscope``, ``crystal``. Example::

            better = sim.refine(npx=300, dmin=0.05)

        Structure factors are carried over when the crystal, energy, and dmin
        are unchanged (raising ``npx`` or adjusting Bethe thresholds is free).
        """
        allowed = {"npx", "dmin", "bethe", "backend", "monte_carlo", "microscope", "crystal"}
        unknown = set(changes) - allowed
        if unknown:
            raise TypeError(f"unknown refine() parameters: {sorted(unknown)}")
        new = MasterPatternSimulator(
            crystal=changes.get("crystal", self.crystal),
            microscope=changes.get("microscope", self.microscope),
            npx=changes.get("npx", self.npx),
            dmin=changes.get("dmin", self.dmin),
            bethe=changes.get("bethe", self.bethe),
            monte_carlo=changes.get("monte_carlo", self.monte_carlo),
            backend=changes.get("backend", self.backend),
        )
        same_sf = (
            new.crystal is self.crystal
            and new.dmin == self.dmin
            and new.energy_kev == self.energy_kev
        )
        if same_sf:
            new._sf = self._sf
        return new

    def show_setup(self, detector=None):
        """Interactive geometry widget for this simulation's setup."""
        from de_ebsd.viz.widgets import SetupWidget

        return SetupWidget.from_setup(
            microscope=self.microscope, detector=detector, crystal=self.crystal
        )

    def __repr__(self) -> str:
        backend = self.backend if isinstance(self.backend, str) else self.backend.name
        return (
            f"MasterPatternSimulator({self.crystal!r}, {self.microscope!r}, "
            f"npx={self.npx}, dmin={self.dmin} nm, backend={backend!r})"
        )


def _unit(k) -> np.ndarray:
    v = np.asarray(k, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("beam direction must be nonzero")
    return v / norm
