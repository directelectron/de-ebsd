"""Monte Carlo depth statistics: EMsoft file reader and fallback model."""

import os

import numpy as np
import pytest

from conftest import SI_MC_PATH
from de_ebsd import DepthDistribution


def test_single_energy_model():
    mc = DepthDistribution.single_energy(20.0, thickness=100.0, depth_step=1.0)
    assert mc.n_energies == 1
    assert mc.n_depths == 100
    assert mc.energy_index(20.0) == 0
    weights = mc.depth_weights(20.0, normal_absorption_length=1e8)
    assert weights.shape == (100,)
    assert np.all(weights >= 0)
    assert weights[0] > weights[-1]  # decays with depth


def test_integration_depths_monotonic_in_energy():
    mc = DepthDistribution.single_energy(20.0, thickness=50.0, depth_step=1.0)
    izz = mc.integration_depths()
    assert izz.shape == (1,)
    assert 1 <= izz[0] <= 50


@pytest.mark.skipif(not os.path.exists(SI_MC_PATH), reason="EMsoft Si Monte Carlo file not found")
def test_from_emsoft_si():
    mc = DepthDistribution.from_emsoft(SI_MC_PATH)
    assert mc.n_energies == 11
    assert mc.energies_kev[0] == pytest.approx(10.0)
    assert mc.energies_kev[-1] == pytest.approx(20.0)
    assert mc.accum_z.shape[0] == 11
    assert mc.n_total > 0

    izz = mc.integration_depths()
    assert izz.shape == (11,)
    assert np.all(izz >= 1)
    assert np.all(np.diff(izz) >= 0)  # running maximum, as in EMsoft

    weights = mc.depth_weights(20.0, normal_absorption_length=150.0)
    assert weights.shape == (mc.n_depths,)
    assert np.all(np.isfinite(weights))
    assert weights.max() > 0
