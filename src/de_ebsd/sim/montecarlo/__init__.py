"""Monte Carlo electron transport and the exit-depth statistics it produces."""

from de_ebsd.sim.montecarlo.depth import DepthDistribution
from de_ebsd.sim.montecarlo.transport import (
    MaterialProperties,
    TrajectoryResult,
    simulate_depth_distribution,
    simulate_trajectory,
    soft_depth_histogram,
)

__all__ = [
    "DepthDistribution",
    "MaterialProperties",
    "TrajectoryResult",
    "simulate_depth_distribution",
    "simulate_trajectory",
    "soft_depth_histogram",
]
