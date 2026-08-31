"""torchebsd: differentiable, GPU-accelerated dynamical EBSD master patterns.

Layered as :mod:`~torchebsd.core` (frames, constants) -> :mod:`~torchebsd.crystal`
(unit cells, structure factors) -> :mod:`~torchebsd.diffraction` (Bloch waves) ->
:mod:`~torchebsd.sim` (master patterns), with :mod:`~torchebsd.index` matching
patterns back to orientations and :mod:`~torchebsd.viz` drawing everything::

    import torchebsd as te

    crystal = te.Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
    )
    microscope = te.Microscope(energy=20e3, sample_tilt=70.0)
    sim = te.MasterPatternSimulator(crystal, microscope, npx=200, dmin=0.1)
    master = sim.compute(device="cuda")

    detector = te.EBSDDetector(shape=(480, 480), pc=(0.5, 0.5, 0.6))
    patterns = master.project(detector, orientation=rotation)  # orix Rotation

See DESIGN.md for the architecture and INDEXING_DESIGN.md for the indexer.
"""

from __future__ import annotations

import importlib

from torchebsd.crystal import AtomSite, Crystal
from torchebsd.diffraction import (
    BetheParameters,
    BlochWaveBackend,
    KinematicalBackend,
    MasterPatternBackend,
)
from torchebsd.sim import (
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

# the display stack is lazy so `import torchebsd` stays functional on headless nodes
_LAZY = {"PhaseOverview": "torchebsd.viz.views", "SetupWidget": "torchebsd.viz.widgets"}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'torchebsd' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
