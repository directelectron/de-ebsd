"""Master-pattern generation: the simulator, its output, and the detector geometry.

The crystallography lives in :mod:`de_ebsd.crystal`, the diffraction physics in
:mod:`de_ebsd.diffraction`, and the electron transport in
:mod:`de_ebsd.sim.montecarlo`.
"""

from de_ebsd.sim.detector import EBSDDetector
from de_ebsd.sim.masterpattern import MasterPattern
from de_ebsd.sim.microscope import Microscope
from de_ebsd.sim.montecarlo import DepthDistribution
from de_ebsd.sim.simulator import MasterPatternSimulator

__all__ = [
    "DepthDistribution",
    "EBSDDetector",
    "MasterPattern",
    "MasterPatternSimulator",
    "Microscope",
]
