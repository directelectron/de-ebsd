"""Mott elastic scattering: the ELSEPA table, its sampling, and what it changes.

The table and its inverse cumulative distribution are checked against the
cross sections they were built from (ELSEPA's own total, and a closed-form
screened-Rutherford case); the sampler against the tabulated angular
distribution by a chi-square test; the Monte Carlo against nickel's measured
backscatter coefficient, which screened Rutherford overestimates.
"""

import numpy as np
import pytest
import torch

from de_ebsd import Crystal
from de_ebsd.core.constants import C_AVOGADRO
from de_ebsd.sim.montecarlo import mott_cross_sections
from de_ebsd.sim.montecarlo.mott import MottSampler, inverse_cdf
from de_ebsd.sim.montecarlo.transport import (
    ELASTIC_MODELS,
    MaterialProperties,
    elastic_cross_section_na,
    screening_parameter,
    simulate_depth_distribution,
)

EBSD_ELEMENTS = "Ni Fe Cu Al Si Ti Co Cr Mo W Au Zr Mg Zn".split()


@pytest.fixture(scope="module")
def tables():
    return mott_cross_sections()


def _gaas():
    return Crystal.from_parameters(
        a=0.56533, space_group=216, atoms=[("Ga", (0.0, 0.0, 0.0)), ("As", (0.25, 0.25, 0.25))]
    )


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def test_table_covers_the_common_ebsd_materials(tables):
    assert set(EBSD_ELEMENTS) <= set(tables.elements)
    assert np.all(np.diff(tables.energy_ev) > 0)
    assert tables.energy_ev[0] <= 100.0 and tables.energy_ev[-1] >= 30_000.0
    assert tables.theta_deg[0] == 0.0 and tables.theta_deg[-1] == 180.0
    n_el, n_e, n_a = len(tables.z), len(tables.energy_ev), len(tables.theta_deg)
    assert tables.log_dcs.shape == (n_el, n_e, n_a)
    assert tables.total_cm2.shape == tables.transport_cm2.shape == (n_el, n_e)


def test_table_records_how_it_was_made(tables):
    meta = tables.metadata
    assert "ELSEPA" in meta["program"] and "Comput. Phys. Commun. 165" in meta["citation"]
    # ELSEPA's free-atom defaults: Fermi nucleus, Dirac-Fock density,
    # Furness-McCarthy exchange, no polarization or absorption
    assert meta["settings"] == {
        "MNUCL": 3,
        "MELEC": 4,
        "MUFFIN": 0,
        "IELEC": -1,
        "MEXCH": 1,
        "MCPOL": 0,
        "MABS": 0,
        "IHEF": 1,
    }
    assert meta["generator"] == "scripts/generate_mott_tables.py"


def test_reduced_dcs_integrates_to_elsepas_total(tables):
    """The kept angles carry the cross section ELSEPA integrated on all of its."""
    keep = tables.energy_ev >= 1000.0
    for z in tables.z:
        _, total = inverse_cdf(tables.theta_deg, tables.dcs_cm2_sr(z)[keep])
        np.testing.assert_allclose(total, tables.total_cm2[tables.index(z), keep], rtol=0.01)


def test_rutherford_overstates_nickels_transport_cross_section(tables):
    """Ni near 15 keV: equal total cross sections, transport ~17 % high in Rutherford.

    The transport cross section ``2 pi int (1 - cos) dsigma`` sets how fast a
    direction is randomized, which is what backscatter follows.
    """
    row = tables.index(28)
    k = int(np.argmin(np.abs(tables.energy_ev - 15e3)))
    energy_kev = tables.energy_ev[k] * 1e-3
    alpha = float(screening_parameter(28, energy_kev))
    total = float(elastic_cross_section_na(28, energy_kev)) / C_AVOGADRO  # cm^2
    mean_one_minus_cos = 2 * alpha * (1 + alpha) * (np.log((1 + alpha) / alpha) - 1 / (1 + alpha))
    assert total / tables.total_cm2[row, k] == pytest.approx(1.0, abs=0.05)
    ratio = total * mean_one_minus_cos / tables.transport_cm2[row, k]
    assert ratio == pytest.approx(1.17, abs=0.05)


def test_inverse_cdf_reproduces_screened_rutherford_in_closed_form():
    """``mu = alpha u / (1 + alpha - u)`` for ``dcs ~ 1 / (mu + alpha)^2``."""
    alpha = 2e-3
    theta = np.concatenate([[0.0], np.geomspace(1e-3, 180.0, 400)])
    mu = 0.5 * (1.0 - np.cos(np.radians(theta)))
    mu_of_u, total = inverse_cdf(theta, 1.0 / (mu + alpha) ** 2)
    u = np.linspace(0.0, 1.0, mu_of_u.shape[-1])
    exact = alpha * u / (1.0 + alpha - u)
    np.testing.assert_allclose(mu_of_u, exact, rtol=2e-3, atol=1e-9)
    assert total == pytest.approx(4.0 * np.pi / (alpha * (1.0 + alpha)), rel=1e-3)


def test_inverse_cdf_is_monotone_from_forward_to_backward(tables):
    mu = tables.inverse_cdf(28)
    assert mu.shape == (len(tables.energy_ev), 4097)
    assert np.all(mu[:, 0] == 0.0) and np.all(mu[:, -1] == 1.0)
    assert np.all(np.diff(mu, axis=1) >= 0.0)


def test_unsupported_element_names_the_generator(tables):
    with pytest.raises(ValueError, match=r"Tc \(Z=43\).*generate_mott_tables\.py"):
        tables.index(43)
    technetium = MaterialProperties.from_crystal(
        Crystal.from_parameters(a=0.2735, space_group=229, atoms=[("Tc", (0.0, 0.0, 0.0))])
    )
    with pytest.raises(ValueError, match="generate_mott_tables"):
        simulate_depth_distribution(technetium, 15.0, n_electrons=10, elastic="mott")


def test_unknown_elastic_model_is_refused(ni_crystal):
    assert ELASTIC_MODELS == ("screened_rutherford", "mott")
    with pytest.raises(ValueError, match="elastic must be one of"):
        simulate_depth_distribution(ni_crystal, 15.0, n_electrons=10, elastic="born")


def test_mean_material_without_composition_needs_one():
    blend = MaterialProperties(
        mean_z=28.5, mean_a=60.0, density=8.9, number_density=90.0, mean_ionization_kev=0.3
    )
    with pytest.raises(ValueError, match="composition"):
        blend.element_fractions()
    pure = MaterialProperties(
        mean_z=28.0, mean_a=58.69, density=8.9, number_density=91.4, mean_ionization_kev=0.3
    )
    assert pure.element_fractions() == ((28, 1.0),)


def test_crystal_composition_is_by_number():
    composition = MaterialProperties.from_crystal(_gaas()).composition
    assert composition == ((31, pytest.approx(0.5)), (33, pytest.approx(0.5)))


# --------------------------------------------------------------------------- #
# the sampler
# --------------------------------------------------------------------------- #
def test_mean_free_path_interpolates_in_log_energy(tables):
    sampler = MottSampler([(28, 1.0)], number_density=91.4, tables=tables)
    row = tables.index(28)
    e = torch.tensor(tables.energy_ev[20:22] * 1e-3, dtype=torch.float64)
    expected = 1.0 / (91.4 * tables.total_cm2[row, 20:22] * 1e14)
    np.testing.assert_allclose(sampler.mean_free_path_nm(e).numpy(), expected, rtol=1e-12)
    middle = torch.tensor(
        [np.sqrt(tables.energy_ev[20] * tables.energy_ev[21]) * 1e-3], dtype=torch.float64
    )
    geometric = np.sqrt(expected[0] * expected[1])
    assert float(sampler.mean_free_path_nm(middle)[0]) == pytest.approx(geometric, rel=1e-12)


@pytest.mark.parametrize("symbol, energy_index", [("Ni", 25), ("Au", 18), ("Al", 27)])
def test_sampled_angles_follow_the_table(tables, symbol, energy_index):
    """Chi-square of sampled polar angles against the tabulated DCS."""
    z = int(tables.z[list(tables.elements).index(symbol)])
    energy_kev = tables.energy_ev[energy_index] * 1e-3
    sampler = MottSampler([(z, 1.0)], number_density=1.0, tables=tables)
    n = 400_000
    gen = torch.Generator().manual_seed(11)
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    cos = sampler.sample_cos(torch.full((n,), energy_kev, dtype=torch.float64), u).numpy()
    theta = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    # expected bin probabilities: the tabulated DCS integrated over solid
    # angle on a fine grid, interpolated log-log, independently of the sampler
    edges = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    fine = np.unique(np.concatenate([np.geomspace(1e-4, 180.0, 200_000), edges[1:]]))
    log_dcs = tables.log_dcs[tables.index(z), energy_index]
    dcs = np.exp(np.interp(np.log(fine), np.log(tables.theta_deg[1:]), log_dcs[1:]))
    weight = dcs * np.sin(np.radians(fine))
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(fine))]
    )
    probability = np.diff(np.interp(edges, fine, cumulative)) / cumulative[-1]

    observed, _ = np.histogram(theta, bins=edges)
    expected = probability * n
    chi2 = float((((observed - expected) ** 2) / expected).sum())
    # 11 degrees of freedom: chi2 > 40 has p < 1e-4; a systematic error of a
    # percent in any well-populated bin would exceed it by far
    assert chi2 < 40.0, (chi2, observed, expected.round())


def test_compound_picks_atoms_by_cross_section(tables):
    """In GaAs, the heavier As takes slightly more than half of the collisions."""
    sampler = MottSampler([(31, 0.5), (33, 0.5)], number_density=44.0, tables=tables)
    e = torch.full((1,), 15.0, dtype=torch.float64)
    i, w = sampler._position(e)
    weights = sampler._cross_sections(i, w)[:, 0].numpy()
    share = weights[1] / weights.sum()
    assert 0.5 < share < 0.56
    # sigma_total of the mixture is the number-weighted mean of the two
    mfp = float(sampler.mean_free_path_nm(e)[0])
    assert mfp == pytest.approx(1.0 / (44.0 * weights.sum()), rel=1e-12)


# --------------------------------------------------------------------------- #
# the Monte Carlo
# --------------------------------------------------------------------------- #
def test_default_is_screened_rutherford_unchanged(ni_crystal):
    kw = dict(e_min_kev=15.0, n_electrons=20_000, seed=42)
    default = simulate_depth_distribution(ni_crystal, 20.0, **kw)
    explicit = simulate_depth_distribution(ni_crystal, 20.0, elastic="screened_rutherford", **kw)
    mott = simulate_depth_distribution(ni_crystal, 20.0, elastic="mott", **kw)
    assert np.array_equal(default.accum_z, explicit.accum_z)
    assert not np.array_equal(default.accum_z, mott.accum_z)


def test_mott_runs_on_a_compound():
    dd = simulate_depth_distribution(_gaas(), 20.0, n_electrons=5_000, seed=0, elastic="mott")
    assert 0.0 < dd.accum_z.sum() / dd.n_total < 1.0


def _backscatter_coefficient(crystal, elastic):
    # every backscattered electron above 1 keV in one energy bin and wide depth
    # bins, normal incidence: the total yield a Faraday-cup measurement sees
    dd = simulate_depth_distribution(
        crystal,
        15.0,
        e_min_kev=1.0,
        e_bin_kev=15.0,
        depth_step_nm=100.0,
        depth_max_nm=1e6,
        n_electrons=20_000,
        tilt_deg=0.0,
        max_steps=5_000,
        device="cpu",
        seed=1,
        elastic=elastic,
    )
    return dd.accum_z.sum() / dd.n_total


def test_mott_brings_nickel_backscatter_to_the_measured_value(ni_crystal):
    """Ni, 15 keV, normal incidence: measured about 0.30.

    Screened Rutherford gives about 0.33 and Mott about 0.30 (1e6 electrons:
    0.328 and 0.304). With 20 000 electrons one standard deviation is 0.003,
    so the two are about six apart.
    """
    measured = 0.30
    mott = _backscatter_coefficient(ni_crystal, "mott")
    rutherford = _backscatter_coefficient(ni_crystal, "screened_rutherford")
    assert abs(mott - measured) < 0.012
    assert rutherford - mott > 0.015
    assert abs(mott - measured) < abs(rutherford - measured)
