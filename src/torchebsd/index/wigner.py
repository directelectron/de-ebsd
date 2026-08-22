"""Wigner-d / D matrices and the SO(3) (SOFT) grid (I2).

The building blocks of the SO(3) Fourier transform (SOFT; Kostelec & Rockmore,
*J. Fourier Anal. Appl.* 2008) used by spherical indexing: the real small Wigner-d
matrices ``d^l_{m'm}(beta)``, the full complex ``D^l_{m'm}(alpha, beta, gamma)``
rotation matrices, and the equiangular SO(3) grid on which the cross-correlation
is evaluated. Torch-native (CPU/CUDA), batched over the rotation angles.
See ``INDEXING_DESIGN.md`` (milestone I2).

Conventions (settled here and shared with :mod:`torchebsd.index.correlation`):

- **Euler angles: proper ``ZYZ``, active, intrinsic.** A rotation ``g`` is
  parameterised by ``(alpha, beta, gamma)`` as the active rotation

      R(alpha, beta, gamma) = R_z(alpha) R_y(beta) R_z(gamma),

  each factor an active (counter-clockwise, right-handed) rotation about the fixed
  lab axis. This is the standard SOFT / quantum-mechanics convention, and the one
  in which the Wigner-D factorisation below is diagonal in ``alpha`` and ``gamma``.

- **Wigner-D matrix.** Following Varshalovich (*Quantum Theory of Angular
  Momentum*, 1988), the rotation operator acting on functions on the sphere,
  ``[R(g) f](n) = f(g^{-1} n)``, transforms the harmonics of :mod:`.harmonics` as

      Y_l^m(g^{-1} n) = sum_{m'} D^l_{m'm}(g) Y_l^{m'}(n),
      D^l_{m'm}(alpha, beta, gamma) = exp(-i m' alpha) d^l_{m'm}(beta) exp(-i m gamma),

  with the first index ``m'`` the *row* and the second index ``m`` the *column*
  (both running ``-l .. l``). ``d^l`` is the real small-d matrix.

- **Small-d matrix.** ``d^l(beta) = exp(-i beta J_y)`` in the ``|l, m>`` basis. We
  evaluate it as ``d^l(beta) = matrix_exp(beta * K_l)`` with the real, antisymmetric
  generator ``K_l = -i J_y``,

      (K_l)_{m+1, m} = -1/2 sqrt((l - m)(l + m + 1)),   (K_l)_{m, m+1} = +...,

  which is numerically stable (bounded generator, scaling-and-squaring) and exact
  to floating point for the moderate bandwidths used in EBSD indexing. It matches
  the closed forms for ``d^1`` and ``d^2`` and is unitary/orthogonal by
  construction; the tests pin both. (For very large ``L`` the Risbo/ALF recurrences
  avoid the per-degree matrix exponential -- a future optimisation.)

- **orix bridge.** :func:`euler_zyz_to_orix` / :func:`grid_to_rotation` convert a
  ``ZYZ`` triple (or a SOFT grid index) to an :class:`orix.quaternion.Rotation`.
  orix's ``Rotation.from_euler`` uses the Bunge (proper ``ZXZ``) convention, and
  ``R_z(a) R_y(b) R_z(c) = R_z(a + pi/2) R_x(b) R_z(c - pi/2)``, so the SOFT triple
  ``(alpha, beta, gamma)`` maps to the orix Bunge triple
  ``(alpha + pi/2, beta, gamma - pi/2)``. With that mapping the returned orix
  Rotation satisfies ``R.to_matrix() == (R_z(alpha) R_y(beta) R_z(gamma))^T`` (orix
  stores the transform/passive matrix, so ``R * v == R.to_matrix() @ v``), and
  ``R.to_euler()`` round-trips the Bunge triple. Going through ``from_euler`` keeps
  us bit-for-bit consistent with how the later milestones read orientations out.

Coefficient / index packing follows :mod:`torchebsd.index.harmonics`: harmonic
coefficients are a flat vector ordered ``c = l*l + l + m``. The padded ``(l, m)``
tensors returned here index degree on one axis and order ``m = idx - L`` on the
others, with entries outside ``|m| <= l`` set to zero.
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
    """Real antisymmetric generator ``K_l = -i J_y`` for degree ``ell``.

    ``d^l(beta) = matrix_exp(beta * K_l)``. ``K_l`` is ``(2l+1, 2l+1)`` with sub- and
    super-diagonal ``+-1/2 sqrt((2l - i)(i + 1))`` (rows/cols ordered ``m = -l .. l``).
    """
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
    """Number of SOFT grid orientations ``(2L + 2) ** 3`` for bandwidth ``L``."""
    return (2 * bandwidth + 2) ** 3


def euler_zyz_to_orix(
    alpha,
    beta,
    gamma,
):
    """Convert active ``ZYZ`` Euler angles to an :class:`orix.quaternion.Rotation`.

    ``alpha``, ``beta``, ``gamma`` are array-likes (numpy arrays, torch tensors, or
    scalars, in radians) broadcast to a common shape. The returned Rotation has that
    shape and satisfies ``R.to_matrix() == (R_z(alpha) R_y(beta) R_z(gamma))^T`` (see
    the module docstring): the SOFT triple is passed to orix's Bunge (``ZXZ``)
    ``from_euler`` as ``(alpha + pi/2, beta, gamma - pi/2)``.
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
