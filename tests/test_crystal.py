"""Crystal: metric tensors, structure matrices, orbits, centering rules."""

import numpy as np
import pytest

from de_ebsd import Crystal


def test_si_d_spacings(si_crystal):
    assert si_crystal.d_spacing([1, 1, 1]) == pytest.approx(0.31356, abs=1e-4)
    assert si_crystal.d_spacing([2, 2, 0]) == pytest.approx(0.19201, abs=1e-4)
    assert si_crystal.d_spacing([4, 0, 0]) == pytest.approx(0.5431 / 4, abs=1e-5)


def test_volume_cubic(si_crystal):
    assert si_crystal.volume == pytest.approx(0.5431**3, rel=1e-10)


def test_metric_tensors_are_inverse(si_crystal):
    product = si_crystal.direct_metric @ si_crystal.reciprocal_metric
    assert np.allclose(product, np.eye(3), atol=1e-12)


def test_structure_matrix_consistent_with_metric():
    """dsm^T dsm must equal the direct metric tensor, also for triclinic cells."""
    crystal = Crystal.from_parameters(
        a=0.4,
        b=0.5,
        c=0.6,
        alpha=80.0,
        beta=95.0,
        gamma=105.0,
        space_group=1,
        atoms=[("C", (0.0, 0.0, 0.0))],
    )
    dsm = crystal.direct_structure_matrix
    assert np.allclose(dsm.T @ dsm, crystal.direct_metric, atol=1e-12)
    # reciprocal structure matrix is inv(dsm).T (EMsoft rsm convention)
    rsm = crystal.reciprocal_structure_matrix
    assert np.allclose(rsm.T @ rsm, crystal.reciprocal_metric, atol=1e-10)


def test_g_cartesian_length_matches_metric(si_crystal):
    hkl = np.array([[1, 1, 1], [2, 2, 0], [3, 1, 1], [1, -2, 3]])
    g_cart = si_crystal.g_cartesian(hkl)
    assert np.allclose(np.linalg.norm(g_cart, axis=1), si_crystal.g_length(hkl), atol=1e-10)


def test_diamond_orbit_origin_choice_2(si_crystal):
    """Si 8a at (1/8, 1/8, 1/8) must expand to 8 atoms (origin choice 2)."""
    assert si_crystal.multiplicities == (8,)
    positions = {tuple(p) for p in si_crystal.positions[0]}
    assert (0.125, 0.125, 0.125) in positions
    assert (0.875, 0.875, 0.875) in positions  # inversion partner


def test_fcc_and_bcc_orbits(ni_crystal):
    assert ni_crystal.multiplicities == (4,)
    fe = Crystal.from_parameters(a=0.2866, space_group=229, atoms=[("Fe", (0, 0, 0))])
    assert fe.multiplicities == (2,)


def test_centering_rules(si_crystal, ni_crystal):
    hkl = np.array([[1, 1, 1], [2, 0, 0], [1, 1, 0], [2, 2, 0], [3, 2, 1]])
    # F centering: h, k, l all same parity
    assert list(ni_crystal.allowed_by_centering(hkl)) == [True, True, False, True, False]
    fe = Crystal.from_parameters(a=0.2866, space_group=229, atoms=[("Fe", (0, 0, 0))])
    # I centering: h + k + l even
    assert list(fe.allowed_by_centering(hkl)) == [False, True, True, True, True]


def test_from_parameters_defaults_cubic():
    c = Crystal.from_parameters(a=0.3, space_group=225, atoms=[("Al", (0, 0, 0))])
    assert (c.b, c.c) == (0.3, 0.3)
    assert (c.alpha, c.beta, c.gamma) == (90.0, 90.0, 90.0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Crystal.from_parameters(a=0.3, space_group=231, atoms=[("Al", (0, 0, 0))])
    with pytest.raises(ValueError):
        Crystal.from_parameters(a=0.3, space_group=225, atoms=[])
