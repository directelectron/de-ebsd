"""WK scattering factors: tables, special functions, amplitude properties."""

import numpy as np
import pytest

from torchebsd.sim.data.scattering_tables import ELEMENT_SYMBOLS, WK_A, WK_B
from torchebsd.sim.scattering import _ei, atomic_number, wk_scattering_factor


def test_tables_verbatim_from_emsoft():
    """Spot-check coefficients against the Fortran source (mod_others.f90)."""
    assert WK_A[0] == [0.00427, 0.00957, 0.00802, 0.00209]  # H
    assert WK_B[0] == [4.17218, 16.05892, 26.78365, 69.45643]
    assert WK_A[13] == [0.03812, 0.17833, 0.0628, 0.05605]  # Si
    assert WK_B[13] == [0.05396, 1.86461, 22.54263, 72.43144]
    assert WK_A[27] == [0.06049, 0.166, 0.37302, 0.07109]  # Ni
    assert WK_A[97] == [0.485, 0.95602, 0.77234, 0.13374]  # Cf
    assert ELEMENT_SYMBOLS[:3] == ["H", "He", "Li"]


def test_atomic_number():
    assert atomic_number("H") == 1
    assert atomic_number("Si") == 14
    assert atomic_number("Ni") == 28
    with pytest.raises(ValueError):
        atomic_number("Xx")


def test_exponential_integral_known_values():
    """Ei(x) against Abramowitz & Stegun tabulated values."""
    x = np.array([1.0, 0.5, -0.5, -1.0, -2.0])
    expected = np.array([1.8951178, 0.4542199, -0.5597736, -0.2193839, -0.0489005])
    assert np.allclose(_ei(x), expected, atol=2e-5)


def test_scattering_amplitude_basic_properties():
    g = np.linspace(0.0, 8.0, 30)
    f = wk_scattering_factor(14, g, ul=0.077, voltage_kv=20.0)
    # elastic part positive at g = 0 and monotonically decreasing overall
    assert f.real[0] > 0
    assert f.real[0] > f.real[-1]
    # absorption is positive and much smaller than the elastic part
    assert np.all(f.imag >= 0)
    assert f.imag[0] < f.real[0]


def test_absorption_toggle():
    g = np.array([0.0, 2.0])
    with_abs = wk_scattering_factor(28, g, ul=0.06, voltage_kv=20.0, absorption=True)
    without = wk_scattering_factor(28, g, ul=0.06, voltage_kv=20.0, absorption=False)
    assert np.allclose(without.imag, 0.0)
    assert np.allclose(with_abs.real, without.real)
    assert np.all(with_abs.imag > 0)


def test_relativistic_scaling():
    """The elastic part carries gamma = (E + 511) / 511."""
    g = np.array([1.0])
    f20 = wk_scattering_factor(14, g, ul=0.0, voltage_kv=20.0, apply_debye_waller=False)
    f200 = wk_scattering_factor(14, g, ul=0.0, voltage_kv=200.0, apply_debye_waller=False)
    gamma20 = (20.0 + 511.0) / 511.0
    gamma200 = (200.0 + 511.0) / 511.0
    assert f200.real[0] / f20.real[0] == pytest.approx(gamma200 / gamma20, rel=1e-6)


def test_input_validation():
    with pytest.raises(ValueError):
        wk_scattering_factor(0, np.array([1.0]), 0.05, 20.0)
    with pytest.raises(ValueError):
        wk_scattering_factor(14, np.array([-1.0]), 0.05, 20.0)
    with pytest.raises(ValueError):
        wk_scattering_factor(14, np.array([1.0]), -0.05, 20.0)
