"""M-MC.1: material foundation for the native Monte Carlo.

Atomic weights, the Berger-Seltzer mean ionization potential, and the reduction
of a crystal to the mean bulk material the MC scatters in (mean Z / A, mass and
number densities). Densities are checked against handbook values as an
end-to-end test of the unit conversions.
"""

import numpy as np
import pytest

from torchebsd import Crystal
from torchebsd.montecarlo_sim import MaterialProperties, mean_ionization_potential_kev
from torchebsd.scattering import atomic_weight


def test_atomic_weight_values():
    assert atomic_weight("H") == pytest.approx(1.008, abs=1e-3)
    assert atomic_weight("Si") == pytest.approx(28.085, abs=1e-3)
    assert atomic_weight("Ni") == pytest.approx(58.693, abs=1e-3)
    assert atomic_weight("Au") == pytest.approx(196.97, abs=1e-2)


def test_atomic_weight_unknown_element():
    with pytest.raises(ValueError, match="Unknown element"):
        atomic_weight("Zz")


def test_mean_ionization_potential():
    # Joy-Luo: J[eV] = 9.76 Z + 58.5 Z^-0.19; returned in keV.
    assert mean_ionization_potential_kev(14) == pytest.approx(0.17210, abs=1e-4)  # Si
    assert mean_ionization_potential_kev(28) == pytest.approx(0.30434, abs=1e-4)  # Ni
    # monotonic increasing in Z
    z = np.arange(1, 99)
    j = np.array([mean_ionization_potential_kev(float(zz)) for zz in z])
    assert np.all(np.diff(j) > 0)


def test_mean_ionization_potential_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        mean_ionization_potential_kev(0.0)


def test_crystal_composition_single_element(si_crystal, ni_crystal):
    assert si_crystal.mean_atomic_number == pytest.approx(14.0)
    assert si_crystal.mean_atomic_weight == pytest.approx(28.085, abs=1e-3)
    assert ni_crystal.mean_atomic_number == pytest.approx(28.0)
    assert ni_crystal.mean_atomic_weight == pytest.approx(58.693, abs=1e-3)


def test_density_matches_handbook(si_crystal, ni_crystal):
    # Si 2.329, Ni 8.908 g/cm^3 (CRC handbook)
    assert si_crystal.density == pytest.approx(2.329, abs=5e-3)
    assert ni_crystal.density == pytest.approx(8.908, abs=1e-2)


def test_material_properties_si(si_crystal):
    mp = MaterialProperties.from_crystal(si_crystal)
    assert mp.mean_z == pytest.approx(14.0)
    assert mp.mean_a == pytest.approx(28.085, abs=1e-3)
    assert mp.density == pytest.approx(2.329, abs=5e-3)
    assert mp.mean_ionization_kev == pytest.approx(mean_ionization_potential_kev(14))
    # number density cross-checks the direct cell count: 8 Si atoms / V(nm^3)
    assert mp.number_density == pytest.approx(8.0 / si_crystal.volume, rel=1e-6)


def test_material_properties_ni(ni_crystal):
    mp = MaterialProperties.from_crystal(ni_crystal)
    assert mp.mean_z == pytest.approx(28.0)
    assert mp.number_density == pytest.approx(4.0 / ni_crystal.volume, rel=1e-6)  # 4 fcc atoms


def test_material_properties_compound_averaging():
    """A binary (GaAs) exercises composition (number) averaging and density."""
    gaas = Crystal.from_parameters(
        a=0.56533, space_group=216,
        atoms=[("Ga", (0.0, 0.0, 0.0)), ("As", (0.25, 0.25, 0.25))],
    )
    mp = MaterialProperties.from_crystal(gaas)
    assert mp.mean_z == pytest.approx(32.0)  # (31 + 33) / 2, equal 4a/4c occupancy
    assert mp.mean_a == pytest.approx(0.5 * (69.723 + 74.922), abs=1e-2)
    assert mp.density == pytest.approx(5.32, abs=0.1)  # handbook GaAs 5.317 g/cm^3
