"""torchebsd: differentiable, GPU-accelerated dynamical EBSD master patterns.

The simulation code lives in :mod:`torchebsd.sim`; the full public API is
re-exported here so that ``import torchebsd as te; te.Crystal`` keeps working::

    import torchebsd as te

    crystal = te.Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
    )
    microscope = te.Microscope(energy=20e3, sample_tilt=70.0)
    sim = te.MasterPatternSimulator(crystal, microscope, npx=200, dmin=0.1)

See DESIGN.md for the architecture and the roadmap toward full EMsoft parity.
"""

from torchebsd.sim import (
    AtomSite,
    BetheParameters,
    BlochWaveBackend,
    Crystal,
    DepthDistribution,
    EBSDDetector,
    KinematicalBackend,
    MasterPattern,
    MasterPatternBackend,
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
    "SetupWidget",
    "__version__",
]


def __getattr__(name: str):
    # SetupWidget is imported lazily so that `import torchebsd` stays light
    # and functional on headless nodes without the widget stack.
    if name == "SetupWidget":
        from torchebsd.sim.widgets import SetupWidget

        return SetupWidget
    raise AttributeError(f"module 'torchebsd' has no attribute {name!r}")
