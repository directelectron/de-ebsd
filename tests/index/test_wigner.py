"""Wigner-d / D matrices and the SO(3) grid (I2): closed forms, unitarity, orix.

Validates :mod:`torchebsd.index.wigner`. The small-d matrices are checked against
the analytic ``d^1`` / ``d^2`` formulas and the ``exp(-i beta J_y)`` group / unitary
properties; the SOFT grid and the orix bridge are checked against the documented
active ``ZYZ`` convention. fp64 throughout.
"""

import math

import pytest
import torch

from torchebsd.index.wigner import (
    euler_zyz_to_orix,
    grid_to_rotation,
    n_orientations,
    so3_grid,
    wigner_D,
    wigner_d,
    wigner_small_d,
)


def _rz(a):
    import numpy as np

    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(b):
    import numpy as np

    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_small_d_closed_form_d1():
    """d^1(beta) matches the textbook 3x3 form (rows/cols m' , m = -1, 0, 1)."""
    for beta in (0.0, 0.3, 1.1, 2.4, math.pi):
        b = torch.tensor(beta, dtype=torch.float64)
        d = wigner_small_d(1, b)
        cb, sb = math.cos(beta), math.sin(beta)
        expected = torch.tensor(
            [
                [(1 + cb) / 2, sb / math.sqrt(2), (1 - cb) / 2],
                [-sb / math.sqrt(2), cb, sb / math.sqrt(2)],
                [(1 - cb) / 2, -sb / math.sqrt(2), (1 + cb) / 2],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(d, expected, atol=1e-13)


def test_small_d_closed_form_d2():
    """A few analytic d^2 entries: cos^4, sin^4 half-angle and (3cos^2 - 1)/2."""
    for beta in (0.2, 0.9, 1.7):
        b = torch.tensor(beta, dtype=torch.float64)
        d = wigner_small_d(2, b)  # order m' , m = -2..2 -> idx 0..4
        hc = math.cos(beta / 2)
        hs = math.sin(beta / 2)
        assert d[4, 4].item() == pytest.approx(hc**4, abs=1e-13)  # d^2_{2,2}
        assert d[0, 4].item() == pytest.approx(hs**4, abs=1e-13)  # d^2_{-2,2}
        assert d[2, 2].item() == pytest.approx((3 * math.cos(beta) ** 2 - 1) / 2, abs=1e-13)
        assert d[4, 2].item() == pytest.approx(
            math.sqrt(3 / 8) * math.sin(beta) ** 2, abs=1e-13
        )  # d^2_{2,0}


def test_small_d_identity_and_orthogonal():
    """d(0) = I, d(beta) is real orthogonal, and d(beta)^T = d(-beta)."""
    for ell in (0, 1, 3, 5):
        n = 2 * ell + 1
        d0 = wigner_small_d(ell, torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(d0, torch.eye(n, dtype=torch.float64), atol=1e-13)
        b = torch.tensor(0.83, dtype=torch.float64)
        d = wigner_small_d(ell, b)
        assert torch.allclose(d @ d.T, torch.eye(n, dtype=torch.float64), atol=1e-12)
        dm = wigner_small_d(ell, -b)
        assert torch.allclose(d.T, dm, atol=1e-12)


def test_small_d_group_property():
    """Rotations about a fixed axis compose: d(b1) d(b2) = d(b1 + b2)."""
    b1 = torch.tensor(0.4, dtype=torch.float64)
    b2 = torch.tensor(1.15, dtype=torch.float64)
    for ell in (1, 2, 4):
        lhs = wigner_small_d(ell, b1) @ wigner_small_d(ell, b2)
        rhs = wigner_small_d(ell, b1 + b2)
        assert torch.allclose(lhs, rhs, atol=1e-12)


def test_small_d_batched_beta():
    """A batched beta gives per-beta matrices equal to the scalar calls."""
    betas = torch.tensor([0.1, 0.7, 2.0], dtype=torch.float64)
    d = wigner_small_d(3, betas)
    assert d.shape == (3, 7, 7)
    for i, b in enumerate(betas):
        assert torch.allclose(d[i], wigner_small_d(3, b), atol=1e-14)


def test_D_unitarity():
    """Each degree block of D^l(alpha, beta, gamma) is unitary."""
    L = 5
    a = torch.tensor(0.37, dtype=torch.float64)
    b = torch.tensor(1.02, dtype=torch.float64)
    g = torch.tensor(2.71, dtype=torch.float64)
    D = wigner_D(L, a, b, g)  # (L+1, 2L+1, 2L+1)
    for ell in range(L + 1):
        lo, hi = L - ell, L + ell + 1
        block = D[ell, lo:hi, lo:hi]
        eye = torch.eye(2 * ell + 1, dtype=torch.complex128)
        assert torch.allclose(block.conj().T @ block, eye, atol=1e-12)


def test_D_identity_at_zero():
    """D(0, 0, 0) is the identity on every degree block."""
    L = 4
    z = torch.tensor(0.0, dtype=torch.float64)
    D = wigner_D(L, z, z, z)
    for ell in range(L + 1):
        lo, hi = L - ell, L + ell + 1
        block = D[ell, lo:hi, lo:hi]
        assert torch.allclose(block, torch.eye(2 * ell + 1, dtype=torch.complex128), atol=1e-13)


def test_D_phase_factorization():
    """D^l_{m'm} = exp(-i m' alpha) d^l_{m'm}(beta) exp(-i m gamma)."""
    L = 3
    a, b, g = 0.5, 1.3, 2.2
    fd = torch.float64
    D = wigner_D(L, torch.tensor(a, dtype=fd), torch.tensor(b, dtype=fd), torch.tensor(g, dtype=fd))
    d = wigner_d(L, torch.tensor(b, dtype=fd))
    orders = torch.arange(-L, L + 1, dtype=torch.float64)
    row = torch.exp(-1j * orders * a)
    col = torch.exp(-1j * orders * g)
    expected = d.to(torch.complex128) * row[None, :, None] * col[None, None, :]
    assert torch.allclose(D, expected, atol=1e-13)


def test_wigner_d_padded_layout():
    """The padded d tensor has the right shape, zeros outside |m| <= l, and blocks."""
    L = 4
    beta = torch.tensor([0.3, 1.4], dtype=torch.float64)
    d = wigner_d(L, beta)
    assert d.shape == (2, L + 1, 2 * L + 1, 2 * L + 1)
    for ell in range(L + 1):
        lo, hi = L - ell, L + ell + 1
        block = d[:, ell, lo:hi, lo:hi]
        for i in range(2):
            assert torch.allclose(block[i], wigner_small_d(ell, beta[i]), atol=1e-14)
        # everything outside the centred (2l+1) block is zero
        mask = torch.ones(2 * L + 1, 2 * L + 1, dtype=torch.bool)
        mask[lo:hi, lo:hi] = False
        assert torch.count_nonzero(d[:, ell][:, mask]) == 0


def test_so3_grid_shapes_and_ranges():
    """The SOFT grid has 2L+2 nodes per angle in the documented ranges."""
    L = 6
    n = 2 * L + 2
    alpha, beta, gamma = so3_grid(L, dtype=torch.float64)
    assert alpha.shape == (n,) and beta.shape == (n,) and gamma.shape == (n,)
    assert torch.allclose(alpha, gamma)
    assert alpha.min() >= 0.0 and alpha.max() < 2 * math.pi
    assert beta.min() > 0.0 and beta.max() < math.pi
    # alpha_j = 2 pi j / N, beta_k = pi (2k+1) / (2N)
    j = torch.arange(n, dtype=torch.float64)
    assert torch.allclose(alpha, 2 * math.pi * j / n)
    assert torch.allclose(beta, math.pi * (2 * j + 1) / (2 * n))
    assert n_orientations(L) == n**3


def test_grid_to_rotation_convention():
    """grid_to_rotation(index).to_matrix() == (R_z(a) R_y(b) R_z(g))^T (orix passive)."""
    pytest.importorskip("orix")
    import numpy as np

    L = 8
    alpha, beta, gamma = so3_grid(L)
    for ia, ib, ig in [(0, 0, 0), (3, 4, 7), (17, 9, 2)]:
        R = grid_to_rotation(L, ia, ib, ig)
        m_zyz = _rz(alpha[ia].item()) @ _ry(beta[ib].item()) @ _rz(gamma[ig].item())
        assert np.allclose(R.to_matrix()[0], m_zyz.T, atol=1e-10)


def test_euler_zyz_to_orix_matches_active_matrix():
    """euler_zyz_to_orix reproduces the active ZYZ matrix (as orix's transpose)."""
    pytest.importorskip("orix")
    import numpy as np

    a, b, g = 1.2, 0.6, 2.9
    R = euler_zyz_to_orix(a, b, g)
    m_zyz = _rz(a) @ _ry(b) @ _rz(g)
    assert np.allclose(R.to_matrix()[0], m_zyz.T, atol=1e-12)
    # array broadcasting: a vector of angles gives a vector of rotations
    Rs = euler_zyz_to_orix([a, 0.1], [b, 0.2], [g, 0.3])
    assert Rs.shape == (2,)


def test_grid_to_rotation_full_grid_shape():
    """The full-grid rotation has shape (N, N, N) matching the correlation layout."""
    pytest.importorskip("orix")
    L = 5
    n = 2 * L + 2
    R = grid_to_rotation(L)
    assert R.shape == (n, n, n)
    # a single index matches the corresponding full-grid entry
    single = grid_to_rotation(L, 2, 3, 4)
    assert torch.allclose(
        torch.as_tensor(R[2, 3, 4].data), torch.as_tensor(single.data), atol=1e-12
    )
