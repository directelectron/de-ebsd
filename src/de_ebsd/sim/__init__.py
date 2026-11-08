"""Master-pattern generation: the simulator, its output, the kinematic band model
beside it, and the detector geometry.

The crystallography lives in :mod:`de_ebsd.crystal`, the diffraction physics in
:mod:`de_ebsd.diffraction`, and the electron transport in
:mod:`de_ebsd.sim.montecarlo`.
"""

from de_ebsd.sim.detector import EBSDDetector
from de_ebsd.sim.kinematic_bands import KinematicMaster
from de_ebsd.sim.masterpattern import MasterPattern
from de_ebsd.sim.microscope import Microscope
from de_ebsd.sim.montecarlo import DepthDistribution
from de_ebsd.sim.simulator import MasterPatternSimulator

__all__ = [
    "DepthDistribution",
    "EBSDDetector",
    "KinematicMaster",
    "MasterPattern",
    "MasterPatternSimulator",
    "Microscope",
]
