"""Dynamical electron diffraction: reflection lists, Bethe potentials, Bloch waves."""

from de_ebsd.diffraction.backends import (
    BlochWaveBackend,
    DynamicalMatrix,
    MasterPatternBackend,
    SimulationInputs,
    build_dynamical_matrix,
    classify_reflections,
    resolve_backend,
)
from de_ebsd.diffraction.bethe import BetheParameters
from de_ebsd.diffraction.kvectors import (
    KVectorGrid,
    reduce_to_fundamental_sector,
    sample_incident_directions,
)
from de_ebsd.diffraction.reflections import (
    ReflectionList,
    candidate_reflections,
    excitation_errors,
    master_reflection_list,
    partition_strong_weak,
)
from de_ebsd.diffraction.smatrix import BlochScatteringMatrixBackend

__all__ = [
    "BetheParameters",
    "BlochScatteringMatrixBackend",
    "BlochWaveBackend",
    "DynamicalMatrix",
    "KVectorGrid",
    "MasterPatternBackend",
    "ReflectionList",
    "SimulationInputs",
    "build_dynamical_matrix",
    "candidate_reflections",
    "classify_reflections",
    "excitation_errors",
    "master_reflection_list",
    "partition_strong_weak",
    "reduce_to_fundamental_sector",
    "resolve_backend",
    "sample_incident_directions",
]


def __getattr__(name: str):
    # The scattering-matrix backend's old name, kept a release with a warning.
    if name == "BlochPrismBackend":
        import warnings

        warnings.warn(
            "BlochPrismBackend is now BlochScatteringMatrixBackend; "
            "the old name goes in a later release",
            DeprecationWarning,
            stacklevel=2,
        )
        return BlochScatteringMatrixBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
