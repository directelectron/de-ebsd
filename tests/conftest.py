"""Shared fixtures: small, fast crystals and structure-factor tables."""

import pytest

from torchebsd import Crystal, Microscope
from torchebsd.crystal.structure_factors import compute_structure_factors

SI_MC_PATH = r"D:\emsoft\EMdata\Si_MC.h5"


@pytest.fixture(scope="session")
def si_crystal():
    return Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))], debye_waller=0.004668
    )


@pytest.fixture(scope="session")
def ni_crystal():
    return Crystal.from_parameters(
        a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))], debye_waller=0.0035
    )


@pytest.fixture(scope="session")
def microscope():
    return Microscope(energy=20e3, sample_tilt=70.0)


@pytest.fixture(scope="session")
def si_sf(si_crystal):
    """Structure factors for Si at 20 keV with a coarse dmin (fast)."""
    return compute_structure_factors(si_crystal, 20.0, dmin=0.1)


@pytest.fixture(scope="session")
def ni_sf(ni_crystal):
    return compute_structure_factors(ni_crystal, 20.0, dmin=0.1)
