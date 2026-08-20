"""torchebsd: differentiable, GPU-accelerated dynamical EBSD master patterns.

A composable, object-oriented simulation chain (inspired by abTEM)::

    import torchebsd as te

    crystal = te.Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
    )
    microscope = te.Microscope(energy=20e3, sample_tilt=70.0)
    sim = te.MasterPatternSimulator(crystal, microscope, npx=200, dmin=0.1)

    sim.structure_factors().show()
    sim.reflections(k=(0, 0, 1)).show()
    master = sim.compute(device="cuda")
    master.show(projection="sphere")

    detector = te.EBSDDetector(shape=(480, 480), pc=(0.5, 0.5, 0.6))
    patterns = master.project(detector, orientation=rotation)  # orix Rotation

See DESIGN.md for the architecture and the roadmap toward full EMsoft parity.
"""

from torchebsd.bethe import BetheParameters
from torchebsd.crystal import AtomSite, Crystal
from torchebsd.detector import EBSDDetector
from torchebsd.dynamical import (
    BlochWaveBackend,
    KinematicalBackend,
    MasterPatternBackend,
)
from torchebsd.masterpattern import MasterPattern
from torchebsd.microscope import Microscope
from torchebsd.montecarlo import DepthDistribution
from torchebsd.simulator import MasterPatternSimulator

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
    "SetupWidget",
    "__version__",
]


def __getattr__(name: str):
    # SetupWidget is imported lazily so that `import torchebsd` stays light
    # and functional on headless nodes without the widget stack.
    if name == "SetupWidget":
        from torchebsd.widgets import SetupWidget

        return SetupWidget
    raise AttributeError(f"module 'torchebsd' has no attribute {name!r}")
