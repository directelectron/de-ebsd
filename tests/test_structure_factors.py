"""Structure factors: absences, magnitudes, wavelength, lookup table."""

import numpy as np
import pytest

from torchebsd import Crystal
from torchebsd.constants import electron_wavelength_nm
from torchebsd.structure_factors import compute_structure_factors

# Complex U_g in nm^-2 dumped from EMsoft's ``EMqg`` utility (mod_diffraction
# ``CalcUcg``, WK branch) for Si.xtal at 20 kV with B = 0.0045 nm^2. Real part is
# the elastic amplitude, imaginary part the WK absorptive (anomalous) term.
_EMSOFT_SI_UCG = {
    (1, 1, 1): -3.763433 - 0.192065j,
    (2, 2, 0): -2.980579 - 0.192851j,
    (3, 1, 1): -1.678949 - 0.121393j,
    (4, 0, 0): -1.787894 - 0.149561j,
    (3, 3, 1): 1.104824 + 0.099184j,
    (4, 2, 2): 1.296174 + 0.128256j,
    (5, 1, 1): 0.832973 + 0.086546j,
    (4, 4, 0): 1.024494 + 0.114072j,
}


def test_ug_matches_emsoft():
    """Weickenmeier-Kohl U_g agree with EMsoft (``EMqg``) for both the real
    amplitude and the anomalous absorptive part, reflection by reflection."""
    crystal = Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))], debye_waller=0.0045
    )
    sf = compute_structure_factors(crystal, 20.0, dmin=0.1)
    for hkl, ucg in _EMSOFT_SI_UCG.items():
        got = complex(sf.lookup(np.array(hkl)))
        assert got.real == pytest.approx(ucg.real, abs=1e-3)
        assert got.imag == pytest.approx(ucg.imag, abs=1e-3)


def test_diamond_absences(si_sf):
    """(200) and (222) are glide/basis forbidden in diamond; (111), (220) are not."""
    assert abs(si_sf.lookup(np.array([2, 0, 0]))) < 1e-8
    assert abs(si_sf.lookup(np.array([2, 2, 2]))) < 1e-8
    assert abs(si_sf.lookup(np.array([1, 1, 1]))) > 1.0
    assert abs(si_sf.lookup(np.array([2, 2, 0]))) > 1.0


def test_fcc_absences(ni_sf):
    assert abs(ni_sf.lookup(np.array([1, 1, 0]))) < 1e-8
    assert abs(ni_sf.lookup(np.array([2, 1, 0]))) < 1e-8
    assert abs(ni_sf.lookup(np.array([1, 1, 1]))) > 1.0
    assert abs(ni_sf.lookup(np.array([2, 0, 0]))) > 1.0


def test_mean_inner_potential_plausible(si_sf, ni_sf):
    """V0 from WK factors: Si ~ 14 V, Ni ~ 30 V (matches EMsoft's output)."""
    assert 12.0 < si_sf.v0 < 17.0
    assert 25.0 < ni_sf.v0 < 34.0


def test_wavelength_refraction_corrected(si_sf):
    vacuum = electron_wavelength_nm(20.0)
    assert vacuum == pytest.approx(8.5885e-3, abs=2e-6)  # 20 kV literature value
    assert si_sf.wavelength < vacuum  # refraction shortens the wavelength
    assert si_sf.wavelength == pytest.approx(vacuum, rel=1e-3)


def test_absorption_is_weak_perturbation(si_sf):
    u111 = si_sf.lookup(np.array([1, 1, 1]))
    assert 0 < abs(u111.imag) < 0.15 * abs(u111.real)
    assert si_sf.upz0 > 0
    assert 50.0 < si_sf.normal_absorption_length < 1000.0  # nm, plausible range


def test_centrosymmetric_friedel(si_sf):
    """For centrosymmetric Si, U_g = U_-g exactly."""
    for hkl in ([1, 1, 1], [2, 2, 0], [3, 1, 1]):
        hkl = np.array(hkl)
        assert si_sf.lookup(hkl) == pytest.approx(si_sf.lookup(-hkl), rel=1e-10)


def test_lut_matches_listing(si_sf):
    """The flat reflection listing and the LUT must agree entry by entry."""
    idx = np.arange(0, len(si_sf.hkl), 7)
    assert np.allclose(si_sf.lookup(si_sf.hkl[idx]), si_sf.ucg[idx], atol=1e-12)


def test_listing_sorted_and_filtered(si_sf):
    assert np.all(np.diff(si_sf.g) >= 0)
    assert np.all(si_sf.d >= si_sf.dmin)
    assert si_sf.crystal.allowed_by_centering(si_sf.hkl).all()
    # (0 0 0) is not in the listing
    assert not np.any(np.all(si_sf.hkl == 0, axis=1))


def test_families_grouping(si_sf):
    hkl, g, ucg = si_sf.families(5)
    assert len(hkl) == 5
    assert np.all(np.diff(g) > 0)
