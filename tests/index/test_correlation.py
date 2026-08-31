"""SO(3) cross-correlation (I2): brute force, peak recovery, autograd, batching.

Validates :mod:`torchebsd.index.spherical.correlation`. The SOFT-grid values are pinned
against a direct ``D^l(g)``-rotate-and-inner-product brute force, the rotate-and-
recover money test confirms the end-to-end orix convention, and gradients match
finite differences. fp64 throughout.
"""

import math

import pytest
import torch

from torchebsd.index.spherical.correlation import so3_cross_correlation
from torchebsd.index.spherical.harmonics import lm_indices, n_coefficients
from torchebsd.index.spherical.wigner import euler_zyz_to_orix, grid_to_rotation, so3_grid, wigner_D


def _pack(flat, L):
    """Flat packed coefficients -> padded (L+1, 2L+1) grid (independent of src)."""
    degrees, orders = lm_indices(L)
    g = torch.zeros(L + 1, 2 * L + 1, dtype=torch.complex128)
    for c in range(n_coefficients(L)):
        g[int(degrees[c]), int(orders[c]) + L] = flat[c]
    return g


def _unpack(grid, L):
    """Padded (L+1, 2L+1) grid -> flat packed coefficients."""
    degrees, orders = lm_indices(L)
    flat = torch.zeros(n_coefficients(L), dtype=torch.complex128)
    for c in range(n_coefficients(L)):
        flat[c] = grid[int(degrees[c]), int(orders[c]) + L]
    return flat


def _rotate_spectrum(flat, L, alpha, beta, gamma):
    """Coefficients of R(g) M for a master M with coefficients ``flat`` (brute force)."""
    fd = torch.float64
    av = torch.tensor(alpha, dtype=fd)
    bv = torch.tensor(beta, dtype=fd)
    gv = torch.tensor(gamma, dtype=fd)
    D = wigner_D(L, av, bv, gv)
    arot = torch.einsum("lum,lm->lu", D, _pack(flat, L))
    return _unpack(arot, L)


def _brute_force_C(master, pattern, L, alpha, beta, gamma):
    """Direct C(g) = <pattern, R(g) master> at one orientation g."""
    arot = _pack(_rotate_spectrum(master, L, alpha, beta, gamma), L)
    return (_pack(pattern, L) * arot.conj()).sum().real.item()


def test_output_shape_and_dtype():
    """C has shape (N, N, N) with N = 2L+2 and is real-valued."""
    L = 5
    n = 2 * L + 2
    torch.manual_seed(0)
    a = torch.randn(n_coefficients(L), dtype=torch.complex128)
    b = torch.randn(n_coefficients(L), dtype=torch.complex128)
    C = so3_cross_correlation(a, b, L)
    assert C.shape == (n, n, n)
    assert not torch.is_complex(C)
    assert C.dtype == torch.float64


def test_brute_force_matches_soft():
    """SOFT-grid values equal the direct D^l-rotate brute force at those grid points."""
    L = 6
    n = 2 * L + 2
    torch.manual_seed(1)
    a = torch.randn(n_coefficients(L), dtype=torch.complex128)
    b = torch.randn(n_coefficients(L), dtype=torch.complex128)
    C = so3_cross_correlation(a, b, L)
    alpha, beta, gamma = so3_grid(L)
    probes = [(0, 0, 0), (1, 3, 5), (7, 2, 11), (13, 13, 4), (n - 1, n - 2, n - 3)]
    for j, k, p in probes:
        direct = _brute_force_C(a, b, L, alpha[j].item(), beta[k].item(), gamma[p].item())
        assert C[j, k, p].item() == pytest.approx(direct, abs=1e-9, rel=1e-9)


def test_real_field_correlation_is_real():
    """For real fields (Hermitian spectra) the brute-force C is real; SOFT returns it."""
    L = 6
    plan_L = L
    # build Hermitian spectra from real fields on the I1 grid
    from torchebsd.index.spherical.harmonics import SphericalHarmonicTransform

    plan = SphericalHarmonicTransform(plan_L, dtype=torch.float64)
    grid = plan.grid_directions()
    fa = grid[..., 2] ** 2 - 0.4 * grid[..., 0] * grid[..., 1]
    fb = 0.3 * grid[..., 0] ** 2 + grid[..., 1] * grid[..., 2]
    a = plan.forward(fa)
    b = plan.forward(fb)
    C = so3_cross_correlation(a, b, L)
    alpha, beta, gamma = so3_grid(L)
    # the direct value at a grid point is real (imag part ~ 0), and equals C
    j, k, p = 3, 4, 9
    arot = _pack(_rotate_spectrum(a, L, alpha[j].item(), beta[k].item(), gamma[p].item()), L)
    direct = (_pack(b, L) * arot.conj()).sum()
    assert abs(direct.imag.item()) < 1e-10
    assert C[j, k, p].item() == pytest.approx(direct.real.item(), abs=1e-9, rel=1e-9)


def test_peak_at_g0_on_grid():
    """Rotate a by a grid orientation g0; the correlation peaks exactly at g0."""
    L = 8
    torch.manual_seed(2)
    a0 = torch.randn(n_coefficients(L), dtype=torch.complex128)
    alpha, beta, gamma = so3_grid(L)
    j0, k0, p0 = 5, 6, 11
    brot = _rotate_spectrum(a0, L, alpha[j0].item(), beta[k0].item(), gamma[p0].item())
    C = so3_cross_correlation(a0, brot, L)  # master = original, pattern = rotated
    idx = torch.argmax(C)
    j, k, p = (int(i) for i in torch.unravel_index(idx, C.shape))
    assert (j, k, p) == (j0, k0, p0)
    # the peak value is the Cauchy-Schwarz maximum ||pattern||^2
    assert C[j, k, p].item() == pytest.approx((brot.abs() ** 2).sum().item(), rel=1e-9)


def test_peak_recovers_offgrid_within_resolution():
    """A generic off-grid g0 is recovered to within the SOFT grid resolution (orix)."""
    pytest.importorskip("orix")
    import numpy as np

    L = 12
    n = 2 * L + 2
    torch.manual_seed(3)
    a0 = torch.randn(n_coefficients(L), dtype=torch.complex128)
    a_off, b_off, g_off = 1.234, 1.001, 2.710  # generic angles, not on the grid
    brot = _rotate_spectrum(a0, L, a_off, b_off, g_off)
    C = so3_cross_correlation(a0, brot, L)
    idx = torch.argmax(C)
    j, k, p = (int(i) for i in torch.unravel_index(idx, C.shape))
    r_peak = grid_to_rotation(L, j, k, p)
    r_true = euler_zyz_to_orix(a_off, b_off, g_off)
    misorientation = float(np.asarray((r_true * (~r_peak)).angle).ravel()[0])
    step = 2 * math.pi / n
    assert misorientation < 1.5 * step


def test_batched_equals_loop():
    """A batch of patterns equals the per-pattern loop."""
    L = 6
    torch.manual_seed(4)
    a = torch.randn(n_coefficients(L), dtype=torch.complex128)
    batch = torch.randn(4, n_coefficients(L), dtype=torch.complex128)
    C = so3_cross_correlation(a, batch, L)
    assert C.shape == (4, 2 * L + 2, 2 * L + 2, 2 * L + 2)
    for i in range(4):
        Ci = so3_cross_correlation(a, batch[i], L)
        assert torch.allclose(C[i], Ci, atol=1e-12)


def test_batched_master_broadcasts():
    """Leading dims of master and pattern broadcast together."""
    L = 4
    torch.manual_seed(5)
    a = torch.randn(3, 1, n_coefficients(L), dtype=torch.complex128)
    b = torch.randn(1, 2, n_coefficients(L), dtype=torch.complex128)
    C = so3_cross_correlation(a, b, L)
    assert C.shape == (3, 2, 2 * L + 2, 2 * L + 2, 2 * L + 2)
    assert torch.allclose(C[1, 0], so3_cross_correlation(a[1, 0], b[0, 0], L), atol=1e-12)


def test_autograd_matches_finite_differences():
    """Gradients into the pattern coefficients match central finite differences."""
    L = 4
    torch.manual_seed(1)
    a = torch.randn(n_coefficients(L), dtype=torch.complex128)
    b_re = torch.randn(n_coefficients(L), dtype=torch.float64)
    b_im = torch.randn(n_coefficients(L), dtype=torch.float64)

    def functional(re):
        b = re + 1j * b_im
        return (so3_cross_correlation(a, b, L) ** 2).mean()

    p = b_re.clone().requires_grad_(True)
    functional(p).backward()
    grad = p.grad
    eps = 1e-6
    for c in (0, 3, 7, 12, 20):
        pp, pm = b_re.clone(), b_re.clone()
        pp[c] += eps
        pm[c] -= eps
        fd = (functional(pp) - functional(pm)).item() / (2 * eps)
        assert grad[c].item() == pytest.approx(fd, abs=1e-6, rel=1e-6)


def test_shape_validation():
    """A coefficient count inconsistent with the bandwidth raises."""
    L = 5
    good = torch.zeros(n_coefficients(L), dtype=torch.complex128)
    with pytest.raises(ValueError):
        so3_cross_correlation(good, torch.zeros(n_coefficients(L) - 1), L)
    with pytest.raises(ValueError):
        so3_cross_correlation(torch.zeros(3, dtype=torch.complex128), good, L)


def test_wigner_d_cache_is_byte_identical():
    """Passing a precomputed ``wigner_d`` block is byte-identical to recomputing it.

    The cache lets :class:`SphericalIndexer` skip rebuilding the O(L^4) Wigner-d block
    per mini-batch; the correlation must be exactly the same either way.
    """
    from torchebsd.index.spherical.wigner import so3_grid, wigner_d

    L = 6
    n = 2 * L + 2
    torch.manual_seed(7)
    a = torch.randn(n_coefficients(L), dtype=torch.complex128)
    b = torch.randn(3, n_coefficients(L), dtype=torch.complex128)

    # build the cache exactly as the function would internally (same beta grid + dtype)
    _, beta, _ = so3_grid(L, dtype=torch.float64)
    d = wigner_d(L, beta)
    assert tuple(d.shape) == (n, L + 1, 2 * L + 1, 2 * L + 1)

    c_plain = so3_cross_correlation(a, b, L)
    c_cached = so3_cross_correlation(a, b, L, wigner_d_cached=d)
    assert torch.equal(c_plain, c_cached)  # byte-identical, not merely close
