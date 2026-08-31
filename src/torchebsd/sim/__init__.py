"""Master-pattern generation: the simulator, its output, and the detector geometry.

The crystallography lives in :mod:`torchebsd.crystal`, the diffraction physics in
:mod:`torchebsd.diffraction`, and the electron transport in
:mod:`torchebsd.sim.montecarlo`. See DESIGN.md.
"""

from torchebsd.sim.detector import EBSDDetector
from torchebsd.sim.masterpattern import MasterPattern
from torchebsd.sim.microscope import Microscope
from torchebsd.sim.montecarlo import DepthDistribution
from torchebsd.sim.simulator import MasterPatternSimulator

__all__ = [
    "DepthDistribution",
    "EBSDDetector",
    "MasterPattern",
    "MasterPatternSimulator",
    "Microscope",
]
