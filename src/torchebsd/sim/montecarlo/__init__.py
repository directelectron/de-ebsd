"""Monte Carlo electron transport and the exit-depth statistics it produces."""

from torchebsd.sim.montecarlo.depth import DepthDistribution
from torchebsd.sim.montecarlo.transport import (
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
