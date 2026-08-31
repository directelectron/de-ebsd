"""Dynamical electron diffraction: reflection lists, Bethe potentials, Bloch waves."""

from torchebsd.diffraction.backends import (
    BlochWaveBackend,
    DynamicalMatrix,
    KinematicalBackend,
    MasterPatternBackend,
    SimulationInputs,
    build_dynamical_matrix,
    classify_reflections,
    resolve_backend,
)
from torchebsd.diffraction.bethe import BetheParameters
from torchebsd.diffraction.kvectors import (
    KVectorGrid,
    reduce_to_fundamental_sector,
    sample_incident_directions,
)
from torchebsd.diffraction.prism import BlochPrismBackend
from torchebsd.diffraction.reflections import (
    ReflectionList,
    candidate_reflections,
    excitation_errors,
    master_reflection_list,
    partition_strong_weak,
)

__all__ = [
    "BetheParameters",
    "BlochPrismBackend",
    "BlochWaveBackend",
    "DynamicalMatrix",
    "KVectorGrid",
    "KinematicalBackend",
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
