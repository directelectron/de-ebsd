"""Wigner d/D matrices and the SOFT grid on SO(3).

Real Wigner-d at beta = pi/2 by stable recursion, the (alpha, beta, gamma) SOFT
sampling grid, and the bridge from ZYZ Euler angles to orix ``Rotation``.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "euler_zyz_to_orix",
    "grid_to_rotation",
    "n_orientations",
    "so3_grid",
    "wigner_D",
    "wigner_d",
    "wigner_small_d",
]


def _small_d_generator(
    ell: int, device: torch.device | str | None, dtype: torch.dtype
) -> torch.Tensor:
    n = 2 * ell + 1
    k = torch.zeros((n, n), device=device, dtype=dtype)
    if ell > 0:
        i = torch.arange(2 * ell, device=device, dtype=dtype)
        c = 0.5 * torch.sqrt((2 * ell - i) * (i + 1))  # coupling on the i-th sub-diagonal
        idx = torch.arange(2 * ell, device=device)
        k[idx + 1, idx] = -c  # (m+1, m)
        k[idx, idx + 1] = c  # (m, m+1) = -(m+1, m) -> antisymmetric
    return k


def wigner_small_d(
    ell: int,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Real small Wigner-d matrix ``d^l_{m'm}(beta)`` for a single degree ``ell``.

    ``beta`` is a real tensor of any shape. Returns ``(*beta.shape, 2l+1, 2l+1)``
    real, with row index ``m'`` and column index ``m`` (both ``-l .. l``), computed
    as ``matrix_exp(beta * K_l)`` (the ``exp(-i beta J_y)`` convention).
    """
    k = _small_d_generator(ell, beta.device, beta.dtype)
    mats = beta[..., None, None] * k  # (*beta.shape, n, n)
    return torch.matrix_exp(mats)


def wigner_d(
    bandwidth: int,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Small Wigner-d matrices ``d^l_{m'm}(beta)`` for all ``l <= bandwidth``.

    Returns a padded real tensor of shape ``(*beta.shape, L+1, 2L+1, 2L+1)``: axis
    ``-3`` is the degree ``l = 0 .. L``, axes ``-2`` / ``-1`` are the orders
    ``m' = idx - L`` (row) and ``m = idx - L`` (column). Entries with ``|m'| > l`` or
    ``|m| > l`` are zero, so degree ``l`` occupies the centred ``(2l+1)``-block.
    """
    L = int(bandwidth)
    out = beta.new_zeros((*beta.shape, L + 1, 2 * L + 1, 2 * L + 1))
    for ell in range(L + 1):
        d_l = wigner_small_d(ell, beta)  # (*beta.shape, 2l+1, 2l+1)
        lo, hi = L - ell, L + ell + 1
        out[..., ell, lo:hi, lo:hi] = d_l
    return out


def wigner_D(
    bandwidth: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Full complex Wigner-D matrices ``D^l_{m'm}(alpha, beta, gamma)`` for ``l <= L``.

    ``alpha``, ``beta``, ``gamma`` are real tensors, broadcast to a common shape
    ``S``. Returns a padded complex tensor ``(*S, L+1, 2L+1, 2L+1)`` with

        D^l_{m'm} = exp(-i m' alpha) d^l_{m'm}(beta) exp(-i m gamma),

    row index ``m'`` (axis ``-2``), column index ``m`` (axis ``-1``), both
    ``m = idx - L``. Entries outside ``|m'|, |m| <= l`` are zero.
    """
    L = int(bandwidth)
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    d = wigner_d(L, beta)  # (*S, L+1, 2L+1, 2L+1), real
    cdtype = torch.complex128 if d.dtype == torch.float64 else torch.complex64
    orders = torch.arange(-L, L + 1, device=d.device, dtype=d.dtype)  # (2L+1,)
    # row phase exp(-i m' alpha) and column phase exp(-i m gamma)
    row_phase = torch.exp(-1j * orders * alpha[..., None].to(d.dtype))  # (*S, 2L+1)
    col_phase = torch.exp(-1j * orders * gamma[..., None].to(d.dtype))  # (*S, 2L+1)
    D = d.to(cdtype)
    D = D * row_phase[..., None, :, None]  # broadcast over degree and column
    D = D * col_phase[..., None, None, :]  # broadcast over degree and row
    return D


def so3_grid(
    bandwidth: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The equiangular SOFT SO(3) sampling grid for bandwidth ``L``.

    Returns the three 1-D angle vectors ``(alpha, beta, gamma)`` of the separable
    grid, each of length ``N = 2L + 2`` (SOFT bandwidth ``B = L + 1``):

    - ``alpha[j] = 2 pi j / N`` and ``gamma[p] = 2 pi p / N`` -- equiangular in
      ``[0, 2 pi)``, the Fourier grids conjugate to the orders ``m', m`` (so the
      correlation is an inverse FFT over ``alpha`` and ``gamma``);
    - ``beta[k] = pi (2k + 1) / (2 N)`` -- the ``2N``-point equiangular colatitude
      grid in ``(0, pi)`` (no samples at the poles).

    The cross-correlation returned by :func:`.so3_cross_correlation` is indexed
    ``C[..., j, k, p]`` on this ``(alpha, beta, gamma)`` grid; :func:`grid_to_rotation`
    converts an ``(j, k, p)`` index to an orix Rotation.
    """
    L = int(bandwidth)
    n = 2 * L + 2
    j = torch.arange(n, device=device, dtype=dtype)
    alpha = 2.0 * math.pi * j / n
    gamma = 2.0 * math.pi * j / n
    beta = math.pi * (2.0 * j + 1.0) / (2.0 * n)
    return alpha, beta, gamma


def n_orientations(bandwidth: int) -> int:
    """Number of SOFT grid points ``(2L + 2) ** 3`` for bandwidth ``L``."""
    return (2 * bandwidth + 2) ** 3


def euler_zyz_to_orix(
    alpha,
    beta,
    gamma,
):
    """Convert active ``ZYZ`` Euler angles to an :class:`orix.quaternion.Rotation`.

    ``alpha``, ``beta``, ``gamma`` are array-likes (numpy arrays, torch tensors, or
    scalars, in radians) broadcast to a common shape. The returned Rotation has that
    shape and satisfies ``R.to_matrix() == (R_z(alpha) R_y(beta) R_z(gamma))^T``:
    the SOFT triple is passed to orix's Bunge (``ZXZ``) ``from_euler`` as
    ``(alpha + pi/2, beta, gamma - pi/2)``.
    """
    import numpy as np
    from orix.quaternion import Rotation

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=float)

    a = _to_np(alpha)
    b = _to_np(beta)
    g = _to_np(gamma)
    a, b, g = np.broadcast_arrays(a, b, g)
    phi1 = a + np.pi / 2.0
    Phi = b
    phi2 = g - np.pi / 2.0
    euler = np.stack([phi1, Phi, phi2], axis=-1)
    return Rotation.from_euler(euler)


def grid_to_rotation(
    bandwidth: int,
    alpha_index=None,
    beta_index=None,
    gamma_index=None,
):
    """Map SOFT grid indices to orix Rotations.

    With no indices, returns the full grid as a Rotation of shape
    ``(N, N, N)`` (``N = 2L + 2``) matching the ``C[..., j, k, p]`` correlation
    layout -- index it directly with the peak location, e.g.
    ``grid_to_rotation(L)[j, k, p]``. Passing ``alpha_index`` / ``beta_index`` /
    ``gamma_index`` (ints or index arrays, broadcast together) returns the Rotation
    at just those grid points. The conversion is :func:`euler_zyz_to_orix`.
    """
    alpha, beta, gamma = so3_grid(bandwidth)
    if alpha_index is None and beta_index is None and gamma_index is None:
        a, b, g = torch.meshgrid(alpha, beta, gamma, indexing="ij")
        return euler_zyz_to_orix(a, b, g)
    if alpha_index is None or beta_index is None or gamma_index is None:
        raise ValueError("pass all three of alpha_index, beta_index, gamma_index, or none")
    return euler_zyz_to_orix(alpha[alpha_index], beta[beta_index], gamma[gamma_index])
