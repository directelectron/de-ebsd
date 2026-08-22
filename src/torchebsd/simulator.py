"""The master-pattern simulator: composition root and stage access.

``MasterPatternSimulator`` wires crystal + microscope + numerical parameters
into the simulation chain and exposes every intermediate stage as an
inspectable object with ``.show()``:

    sim = MasterPatternSimulator(crystal, microscope, npx=150, dmin=0.1)
    sim.structure_factors().show()     # |U_g|, absences, V_0
    sim.reflections().show()           # master reflection list
    sim.reflections(k=(0, 0, 1)).show()  # Bethe strong/weak split for one beam
    sim.dynamical_matrix(k=(0, 0, 1)).show()
    master = sim.compute(device="cuda")
    master.show(projection="sphere")

Refinement returns a new simulator (``sim.refine(npx=300, dmin=0.05)``) that
reuses every cached stage still consistent with the new parameters.
"""

from __future__ import annotations

import numpy as np
import torch

from torchebsd.bethe import BetheParameters
from torchebsd.crystal import Crystal
from torchebsd.dynamical import (
    DynamicalMatrix,
    MasterPatternBackend,
    SimulationInputs,
    build_dynamical_matrix,
    classify_reflections,
    resolve_backend,
)
from torchebsd.kvectors import reduce_to_fundamental_sector, sample_incident_directions
from torchebsd.masterpattern import MasterPattern
from torchebsd.microscope import Microscope
from torchebsd.montecarlo import DepthDistribution
from torchebsd.reflections import ReflectionList, master_reflection_list
from torchebsd.structure_factors import StructureFactors, compute_structure_factors

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
        ``"kinematical"`` (runs today; band geometry only), ``"bloch"``
        (the physical model; scheduled -- see DESIGN.md), or any
        :class:`~torchebsd.dynamical.MasterPatternBackend` instance.
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

    # ------------------------------------------------------------------ #
    # stages
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # computation
    # ------------------------------------------------------------------ #
    def compute(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        chunk_size: int = 262_144,
        mc_electrons: int = 200_000,
        symmetry: bool = True,
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
        """
        device = torch.device(device) if device is not None else torch.device("cpu")
        sf = self.structure_factors()
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
        backend.prepare(
            SimulationInputs(
                sf=sf,
                bethe=self.bethe,
                monte_carlo=monte_carlo,
                device=device,
                dtype=dtype,
            )
        )

        grid = sample_incident_directions(self.npx, device=device, dtype=dtype)
        directions = grid.flat()
        if symmetry:
            reps, inverse = reduce_to_fundamental_sector(directions, self.crystal.space_group)
        else:
            reps, inverse = directions, None

        rep_values = torch.empty(reps.shape[0], device=device, dtype=dtype)
        for start in range(0, reps.shape[0], chunk_size):
            stop = min(start + chunk_size, reps.shape[0])
            rep_values[start:stop] = backend.intensities(reps[start:stop])
        values = rep_values if inverse is None else rep_values[inverse]

        n = 2 * self.npx + 1
        data = values.reshape(1, 2, n, n)
        return MasterPattern(
            data=data,
            energies_kev=np.array([self.energy_kev]),
            metadata={
                "crystal": repr(self.crystal),
                "backend": backend.name,
                "npx": self.npx,
                "dmin": self.dmin,
                "energy_kev": self.energy_kev,
                "bethe": repr(self.bethe),
            },
        )

    # ------------------------------------------------------------------ #
    # refinement
    # ------------------------------------------------------------------ #
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
        from torchebsd.widgets import SetupWidget

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
