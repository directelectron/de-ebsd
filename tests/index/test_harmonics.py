"""Spherical harmonic transform (I1): reference, round-trip, autograd, batching.

Validates :mod:`torchebsd.index.harmonics` -- the band-limited SHT on the sphere.
The transform is exact for band-limited functions, so round trips hit floating
point (~1e-12 in float64), the basis matches ``scipy.special.sph_harm_y``, and
gradients through the transform match finite differences.
"""

import math

import pytest
import torch

from torchebsd.index.harmonics import (
    SphericalHarmonicTransform,
    forward_sht,
    gauss_legendre_grid,
    inverse_sht,
    lm_indices,
    n_coefficients,
    spherical_harmonics,
)


def test_grid_shapes_and_quadrature_weights():
    """Gauss-Legendre x equiangular grid: shapes, node ranges, weights sum to 4 pi."""
    L = 7
    theta, phi, weights = gauss_legendre_grid(L, dtype=torch.float64)
    assert theta.shape == (L + 1,)
    assert phi.shape == (2 * L + 2,)
    assert weights.shape == (L + 1, 2 * L + 2)
    assert theta.min() >= 0.0 and theta.max() <= math.pi
    assert phi.min() >= 0.0 and phi.max() < 2 * math.pi
    # the quadrature integrates the constant 1 over the sphere -> 4 pi
    assert weights.sum().item() == pytest.approx(4.0 * math.pi, abs=1e-12)


def test_reference_against_scipy():
    """Every Y_lm up to L = 8 matches scipy.special.sph_harm_y on the grid."""
    sp = pytest.importorskip("scipy.special")
    import numpy as np

    L = 8
    theta, phi, _ = gauss_legendre_grid(L, dtype=torch.float64)
    Y = spherical_harmonics(L, theta, phi)  # (n_coeff, N_theta, N_phi)
    degrees, orders = lm_indices(L)
    th, ph = np.meshgrid(theta.numpy(), phi.numpy(), indexing="ij")
    max_err = 0.0
    for c in range(n_coefficients(L)):
        ref = sp.sph_harm_y(int(degrees[c]), int(orders[c]), th, ph)
        max_err = max(max_err, float(np.abs(Y[c].numpy() - ref).max()))
    assert max_err < 1e-12


def test_y00_is_constant():
    """The l = m = 0 harmonic is the constant 1 / sqrt(4 pi)."""
    theta, phi, _ = gauss_legendre_grid(4, dtype=torch.float64)
    Y = spherical_harmonics(4, theta, phi)
    y00 = Y[0]
    assert torch.allclose(y00.imag, torch.zeros_like(y00.imag), atol=1e-14)
    assert torch.allclose(
        y00.real, torch.full_like(y00.real, 1.0 / math.sqrt(4.0 * math.pi)), atol=1e-14
    )


def test_roundtrip_recovers_coefficients_and_samples():
    """Band-limited a_lm -> samples -> a_lm recovers to floating point (<< 1e-6)."""
    L = 12
    plan = SphericalHarmonicTransform(L, dtype=torch.float64)
    torch.manual_seed(0)
    a = torch.randn(plan.n_coeff, dtype=torch.complex128)
    values = plan.inverse(a)
    a_rec = plan.forward(values)
    assert (a_rec - a).abs().max().item() < 1e-9
    # the samples come back too (idempotent synthesis)
    values_rec = plan.inverse(a_rec)
    assert (values_rec - values).abs().max().item() < 1e-9


def test_forward_of_sampled_harmonic_is_a_delta():
    """Forward of a single sampled Y_lm returns the unit coefficient at (l, m)."""
    L = 6
    plan = SphericalHarmonicTransform(L, dtype=torch.float64)
    Y = spherical_harmonics(L, plan.theta, plan.phi)  # (n_coeff, N_theta, N_phi)
    idx = n_coefficients(L) - 3  # some (l, m) with l = L
    coeffs = plan.forward(Y[idx])
    expected = torch.zeros(plan.n_coeff, dtype=torch.complex128)
    expected[idx] = 1.0
    assert (coeffs - expected).abs().max().item() < 1e-10


def test_real_function_has_hermitian_spectrum():
    """A real field on the sphere gives a Hermitian spectrum a_{l,-m} = (-1)^m conj(a_lm)."""
    L = 8
    plan = SphericalHarmonicTransform(L, dtype=torch.float64)
    grid = plan.grid_directions()
    field = grid[..., 2] ** 3 + 0.5 * grid[..., 0] * grid[..., 1]  # real, smooth
    a = plan.forward(field)
    degrees, orders = lm_indices(L)
    for c in range(plan.n_coeff):
        ell, m = int(degrees[c]), int(orders[c])
        c_neg = ell * ell + ell - m
        sign = (-1.0) ** abs(m)
        assert a[c].conj().item() == pytest.approx(
            (sign * a[c_neg]).item(), abs=1e-9, rel=1e-9
        )


def test_free_functions_match_plan_on_canonical_grid():
    """The stateless forward_sht/inverse_sht agree with the cached plan."""
    L = 10
    plan = SphericalHarmonicTransform(L, dtype=torch.float64)
    torch.manual_seed(1)
    a = torch.randn(plan.n_coeff, dtype=torch.complex128)
    values_plan = plan.inverse(a)
    values_free = inverse_sht(a, plan.theta, plan.phi)
    assert (values_plan - values_free).abs().max().item() < 1e-10
    a_plan = plan.forward(values_plan)
    a_free = forward_sht(values_free, plan.theta, plan.phi, L)
    assert (a_plan - a_free).abs().max().item() < 1e-10
    assert (a_free - a).abs().max().item() < 1e-9


def test_batched_transform_equals_elementwise():
    """A batched transform equals transforming each element of the batch."""
    L = 6
    plan = SphericalHarmonicTransform(L, dtype=torch.float64)
    torch.manual_seed(2)
    a = torch.randn(4, 3, plan.n_coeff, dtype=torch.complex128)
    values = plan.inverse(a)
    coeffs = plan.forward(values)
    for i in range(4):
        for j in range(3):
            v_ij = plan.inverse(a[i, j])
            assert torch.equal(values[i, j], v_ij)
            assert torch.equal(coeffs[i, j], plan.forward(v_ij))
    # and the batch round-trips
    assert (coeffs - a).abs().max().item() < 1e-9


def test_autograd_forward_matches_finite_differences():
    """Gradients through forward (samples -> coeffs) match central differences."""
    L = 5
    plan = SphericalHarmonicTransform(L, dtype=torch.float64)
    grid = plan.grid_directions()
    base = (grid[..., 2] ** 2 - 0.3 * grid[..., 0]).clone()  # a real field

    def functional(field):
        return (plan.forward(field).abs() ** 2).sum()

    f = base.clone().requires_grad_(True)
    functional(f).backward()
    grad = f.grad

    eps = 1e-6
    probes = [(0, 0), (1, 3), (2, 7), (3, 5), (5, 10)]
    for i, j in probes:
        fp, fm = base.clone(), base.clone()
        fp[i, j] += eps
        fm[i, j] -= eps
        fd = (functional(fp) - functional(fm)).item() / (2 * eps)
        assert grad[i, j].item() == pytest.approx(fd, abs=1e-6, rel=1e-6)


def test_autograd_inverse_matches_finite_differences():
    """Gradients through inverse (coeffs -> samples) match central differences."""
    L = 4
    plan = SphericalHarmonicTransform(L, dtype=torch.float64)
    torch.manual_seed(3)
    base = torch.randn(plan.n_coeff, dtype=torch.float64)

    def functional(params):
        vals = plan.inverse(params.to(torch.complex128))
        return (vals.real ** 2).sum()

    p = base.clone().requires_grad_(True)
    functional(p).backward()
    grad = p.grad

    eps = 1e-6
    for c in (0, 3, 7, 12, 20):
        pp, pm = base.clone(), base.clone()
        pp[c] += eps
        pm[c] -= eps
        fd = (functional(pp) - functional(pm)).item() / (2 * eps)
        assert grad[c].item() == pytest.approx(fd, abs=1e-6, rel=1e-6)


def test_grid_directions_conventions():
    """grid_directions are unit vectors following n = (st cp, st sp, ct)."""
    L = 5
    plan = SphericalHarmonicTransform(L, dtype=torch.float64)
    n = plan.grid_directions()
    assert n.shape == (plan.n_theta, plan.n_phi, 3)
    ones = torch.ones(plan.n_theta, plan.n_phi, dtype=torch.float64)
    assert torch.allclose(n.norm(dim=-1), ones, atol=1e-12)
    # z component is cos(theta), independent of phi
    ct = torch.cos(plan.theta)
    assert torch.allclose(n[..., 2], ct[:, None].expand_as(n[..., 2]), atol=1e-12)


def test_float32_roundtrip():
    """The float32 plan round-trips within single-precision tolerance."""
    L = 8
    plan = SphericalHarmonicTransform(L, dtype=torch.float32)
    assert plan.cdtype == torch.complex64
    torch.manual_seed(4)
    a = torch.randn(plan.n_coeff, dtype=torch.complex64)
    a_rec = plan.forward(plan.inverse(a))
    assert (a_rec - a).abs().max().item() < 1e-4


def test_shape_validation():
    """Wrong trailing shapes raise clear errors."""
    plan = SphericalHarmonicTransform(4, dtype=torch.float64)
    with pytest.raises(ValueError):
        plan.forward(torch.zeros(3, 3))
    with pytest.raises(ValueError):
        plan.inverse(torch.zeros(5, dtype=torch.complex128))
