"""de-ebsd: differentiable, GPU-accelerated dynamical EBSD master patterns.

Layered as :mod:`~de_ebsd.core` (frames, constants) -> :mod:`~de_ebsd.crystal`
(unit cells, structure factors) -> :mod:`~de_ebsd.diffraction` (Bloch waves) ->
:mod:`~de_ebsd.sim` (master patterns), with :mod:`~de_ebsd.index` matching
patterns back to orientations and :mod:`~de_ebsd.viz` drawing everything::

    import de_ebsd as de

    crystal = de.Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
    )
    microscope = de.Microscope(energy=20e3, sample_tilt=70.0)
    sim = de.MasterPatternSimulator(crystal, microscope, npx=200, dmin=0.1)
    master = sim.compute(device="cuda")

    detector = de.EBSDDetector(shape=(480, 480), pc=(0.5, 0.5, 0.6))
    patterns = master.project(detector, orientation=rotation)  # orix Rotation

See DESIGN.md for the architecture and INDEXING_DESIGN.md for the indexer.
"""

from __future__ import annotations

import importlib

from de_ebsd.crystal import AtomSite, Crystal
from de_ebsd.diffraction import (
    BetheParameters,
    BlochWaveBackend,
    KinematicalBackend,
    MasterPatternBackend,
)
from de_ebsd.sim import (
    DepthDistribution,
    EBSDDetector,
    MasterPattern,
    MasterPatternSimulator,
    Microscope,
)

__version__ = "0.1.0"

__all__ = [
    "AtomSite",
    "BetheParameters",
    "BlochWaveBackend",
    "Crystal",
    "DepthDistribution",
    "EBSDDetector",
    "KinematicalBackend",
    "MasterPattern",
    "MasterPatternBackend",
    "MasterPatternSimulator",
    "Microscope",
    "PhaseOverview",
    "SetupWidget",
    "__version__",
]

# the display stack is lazy so `import de_ebsd` stays functional on headless nodes
_LAZY = {"PhaseOverview": "de_ebsd.viz.views", "SetupWidget": "de_ebsd.viz.widgets"}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'de_ebsd' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
