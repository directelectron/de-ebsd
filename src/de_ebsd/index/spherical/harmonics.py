"""Spherical harmonic transform on a Gauss-Legendre x equiangular grid.

Forward and inverse SHT for band-limited functions on the sphere, with the
associated Legendre functions built by stable upward recursion.
:class:`SphericalHarmonicTransform` caches the grid and basis so the master and
every pattern are expanded identically.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from de_ebsd.core.device import resolve_device

__all__ = [
    "SphericalHarmonicTransform",
    "forward_sht",
    "gauss_legendre_grid",
    "inverse_sht",
    "lm_indices",
    "n_coefficients",
    "spherical_harmonics",
]

_INV_SQRT_4PI = 1.0 / math.sqrt(4.0 * math.pi)


def n_coefficients(bandwidth: int) -> int:
    """Number of packed coefficients for a transform of the given bandwidth ``L``."""
    return (bandwidth + 1) ** 2


def lm_indices(
    bandwidth: int, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Degree and order of each packed coefficient slot.

    Returns ``(degrees, orders)``, each a 1-D ``long`` tensor of length
    ``(L + 1) ** 2``, so that slot ``c`` holds ``a_{l m}`` with ``l = degrees[c]``
    and ``m = orders[c]``, ordered ``c = l * l + l + m``.
    """
    degrees = []
    orders = []
    for ell in range(bandwidth + 1):
        for m in range(-ell, ell + 1):
            degrees.append(ell)
            orders.append(m)
    dl = torch.tensor(degrees, dtype=torch.long, device=device)
    dm = torch.tensor(orders, dtype=torch.long, device=device)
    return dl, dm


def _gauss_legendre_nodes(
    n: int, device: torch.device | str | None, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    x, w = np.polynomial.legendre.leggauss(n)
    xt = torch.as_tensor(x, device=device, dtype=dtype)
    wt = torch.as_tensor(w, device=device, dtype=dtype)
    return xt, wt


def gauss_legendre_grid(
    bandwidth: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the exact quadrature grid for bandwidth ``L``.

    Returns ``(theta, phi, weights)``:

    - ``theta`` -- ``(L + 1,)`` colatitude nodes, ``arccos`` of the Gauss-Legendre
      nodes (in ``[0, pi]``).
    - ``phi`` -- ``(2 L + 2,)`` equiangular azimuth nodes ``2 pi k / N_phi``.
    - ``weights`` -- ``(L + 1, 2 L + 2)`` quadrature weights ``w_theta[j] * (2 pi /
      N_phi)`` such that ``sum weights[j, k] f(theta_j, phi_k)`` integrates any
      band-limited ``f`` over the sphere exactly.
    """
    n_theta = bandwidth + 1
    n_phi = 2 * bandwidth + 2
    x, w_theta = _gauss_legendre_nodes(n_theta, device, dtype)
    theta = torch.arccos(x)
    phi = 2.0 * math.pi * torch.arange(n_phi, device=device, dtype=dtype) / n_phi
    weights = w_theta[:, None] * (2.0 * math.pi / n_phi)
    weights = weights.expand(n_theta, n_phi).contiguous()
    return theta, phi, weights


def _normalized_alf(bandwidth: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    L = bandwidth
    p = x.new_zeros((L + 1, L + 1, *x.shape))
    p[0, 0] = _INV_SQRT_4PI
    # sectoral terms P_m^m (Condon-Shortley phase accumulates through the -1)
    for m in range(1, L + 1):
        p[m, m] = -math.sqrt((2 * m + 1) / (2 * m)) * u * p[m - 1, m - 1]
    # ascend in degree l at fixed order m
    for m in range(L + 1):
        if m + 1 <= L:
            # l = m + 1: the P_{l-2} term vanishes (P_{m-1}^m = 0)
            a = math.sqrt((2 * (m + 1) - 1) * (2 * (m + 1) + 1) / ((m + 1 - m) * (m + 1 + m)))
            p[m + 1, m] = a * x * p[m, m]
        for ell in range(m + 2, L + 1):
            a = math.sqrt((2 * ell - 1) * (2 * ell + 1) / ((ell - m) * (ell + m)))
            b = math.sqrt(
                (2 * ell + 1)
                * (ell + m - 1)
                * (ell - m - 1)
                / ((2 * ell - 3) * (ell - m) * (ell + m))
            )
            p[ell, m] = a * x * p[ell - 1, m] - b * p[ell - 2, m]
    return p


def spherical_harmonics(bandwidth: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Dense complex spherical-harmonic basis ``Y_lm`` on a grid.

    ``theta`` (``(N_theta,)`` colatitudes) and ``phi`` (``(N_phi,)`` azimuths) are
    the 1-D node vectors of a separable grid. Returns the complex design matrix of
    shape ``((L + 1) ** 2, N_theta, N_phi)`` with slot ``c`` (ordering of
    :func:`lm_indices`) holding ``Y_{l m}`` evaluated at every ``(theta_j, phi_k)``.
    Matches ``scipy.special.sph_harm_y(l, m, theta, phi)``.
    """
    x = torch.cos(theta)
    u = torch.sin(theta).clamp_min(0.0)
    p = _normalized_alf(bandwidth, x, u)  # (L+1, L+1, N_theta)
    degrees, orders = lm_indices(bandwidth, device=theta.device)
    # P_l^{|m|} for each coefficient slot -> (n_coeff, N_theta)
    p_lm = p[degrees, orders.abs()]
    # Condon-Shortley: Y_l^{-m} = (-1)^m conj(Y_l^m); fold the sign of negative m
    parity = 1.0 - 2.0 * (orders.abs() % 2).to(theta.dtype)  # (-1)^|m|
    sign = torch.where(orders < 0, parity, torch.ones_like(parity))
    m_real = orders.to(theta.dtype)
    phase = torch.exp(1j * (m_real[:, None] * phi[None, :]))  # (n_coeff, N_phi)
    y = (sign[:, None].to(phase.dtype) * p_lm.to(phase.dtype))[:, :, None] * phase[:, None, :]
    return y


def forward_sht(
    values: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    bandwidth: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stateless forward transform on a separable grid (dense design matrix).

    ``values`` has shape ``(..., N_theta, N_phi)`` (real or complex), sampled at the
    ``theta``/``phi`` nodes. Returns complex coefficients ``(..., (L + 1) ** 2)``.
    ``weights`` are the quadrature weights ``(N_theta, N_phi)``; if omitted the
    Gauss-Legendre weights for this bandwidth are used, which assumes ``theta`` /
    ``phi`` are the canonical grid from :func:`gauss_legendre_grid`. For repeated
    transforms prefer :class:`SphericalHarmonicTransform`.
    """
    y = spherical_harmonics(bandwidth, theta, phi)  # (n_coeff, N_theta, N_phi)
    if weights is None:
        _, _, weights = gauss_legendre_grid(bandwidth, theta.device, theta.dtype)
    weighted = (values * weights).to(y.dtype)
    return torch.einsum("...tp,ctp->...c", weighted, y.conj())


def inverse_sht(coeffs: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Stateless inverse transform: coefficients ``a_lm`` -> samples on a grid.

    ``coeffs`` has shape ``(..., (L + 1) ** 2)``; the bandwidth is inferred from its
    length. Returns complex samples ``(..., N_theta, N_phi)`` at the ``theta``/``phi``
    nodes (the dense synthesis ``sum_lm a_lm Y_lm``).
    """
    n_coeff = coeffs.shape[-1]
    bandwidth = int(round(math.sqrt(n_coeff))) - 1
    y = spherical_harmonics(bandwidth, theta, phi)
    return torch.einsum("...c,ctp->...tp", coeffs.to(y.dtype), y)


class SphericalHarmonicTransform:
    """A cached, separable spherical harmonic transform for a fixed bandwidth.

    Builds the exact Gauss-Legendre x equiangular quadrature grid and the
    associated-Legendre design matrices once, then transforms via an FFT over phi
    and a per-order matmul over theta. Both directions are batched over arbitrary
    leading dimensions and are autograd-differentiable in the samples/coefficients.

    Parameters
    ----------
    bandwidth : int
        The band limit ``L``. Sets the quadrature grid (``N_theta = L + 1``,
        ``N_phi = 2 L + 2``) and the packed coefficient count ``(L + 1) ** 2``.
    device : torch.device or str, optional
        Device the grid and cached design matrices are built on (default CPU).
    dtype : torch.dtype, optional
        Real floating dtype of the transform; must be ``torch.float32`` or
        ``torch.float64`` (default ``torch.float64``). The complex coefficient dtype
        is the matching ``complex64`` / ``complex128``.

    Attributes
    ----------
    bandwidth : int
        The band limit ``L``.
    theta, phi : torch.Tensor
        1-D quadrature nodes (``(L + 1,)`` and ``(2 L + 2,)``).
    weights : torch.Tensor
        ``(L + 1, 2 L + 2)`` quadrature weights.
    n_coeff : int
        Number of packed coefficients ``(L + 1) ** 2``.
    """

    def __init__(
        self,
        bandwidth: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be a real floating type (float32 or float64)")
        self.bandwidth = int(bandwidth)
        self.device = resolve_device(device)
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        L = self.bandwidth
        self.n_theta = L + 1
        self.n_phi = 2 * L + 2
        self.n_coeff = n_coefficients(L)

        theta, phi, weights = gauss_legendre_grid(L, self.device, dtype)
        self.theta = theta
        self.phi = phi
        self.weights = weights

        degrees, orders = lm_indices(L, device=self.device)
        self._degrees = degrees
        self._orders = orders
        # FFT bin holding Fourier order m (negative m wraps to N_phi + m)
        self._freq_bins = torch.remainder(orders, self.n_phi)

        with torch.no_grad():
            x = torch.cos(theta)
            u = torch.sin(theta).clamp_min(0.0)
            _, w_theta = _gauss_legendre_nodes(self.n_theta, self.device, dtype)
            p = _normalized_alf(L, x, u)  # (L+1, L+1, N_theta)
            p_lm = p[degrees, orders.abs()]  # (n_coeff, N_theta)
            # fold the Condon-Shortley sign of negative orders into the design matrix
            parity = 1.0 - 2.0 * (orders.abs() % 2).to(dtype)  # (-1)^|m|
            sign = torch.where(orders < 0, parity, torch.ones_like(parity))
            design = sign[:, None] * p_lm  # A[c, j] = s(m) P_l^{|m|}(cos theta_j)
            self._design = design.contiguous()
            # forward weights: (2 pi / N_phi) w_theta[j] A[c, j]
            self._fwd = ((2.0 * math.pi / self.n_phi) * w_theta[None, :] * design).contiguous()

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SphericalHarmonicTransform:
        """Return a plan on a new device / real dtype (cheap to rebuild)."""
        return SphericalHarmonicTransform(
            self.bandwidth,
            device=device if device is not None else self.device,
            dtype=dtype if dtype is not None else self.dtype,
        )

    def grid_directions(self) -> torch.Tensor:
        """Cartesian unit vectors ``(N_theta, N_phi, 3)`` of every quadrature node.

        ``n = (sin th cos ph, sin th sin ph, cos th)`` -- sample the sphere here
        (e.g. resample a master pattern onto these directions), then ``forward``.
        """
        st = torch.sin(self.theta)
        ct = torch.cos(self.theta)
        cp = torch.cos(self.phi)
        sp = torch.sin(self.phi)
        nx = st[:, None] * cp[None, :]
        ny = st[:, None] * sp[None, :]
        nz = ct[:, None].expand(self.n_theta, self.n_phi)
        return torch.stack([nx, ny, nz], dim=-1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Samples ``(..., N_theta, N_phi)`` -> coefficients ``(..., n_coeff)``.

        ``values`` may be real or complex; the result is complex. Differentiable
        in ``values``.
        """
        if values.shape[-2:] != (self.n_theta, self.n_phi):
            raise ValueError(
                f"values trailing shape {tuple(values.shape[-2:])} != grid "
                f"{(self.n_theta, self.n_phi)}"
            )
        # FFT over phi: fhat[..., j, m] = sum_k values[..., j, k] exp(-i m phi_k)
        fhat = torch.fft.fft(values.to(self.cdtype), dim=-1)
        # pick the bin of each coefficient's order -> (..., N_theta, n_coeff)
        fhat_c = fhat.index_select(-1, self._freq_bins)
        # a_c = sum_j fwd[c, j] fhat_c[..., j, c]
        return torch.einsum("cj,...jc->...c", self._fwd.to(self.cdtype), fhat_c)

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Coefficients ``(..., n_coeff)`` -> samples ``(..., N_theta, N_phi)``.

        Returns complex samples (take ``.real`` for a real-valued function).
        Differentiable in ``coeffs``.
        """
        if coeffs.shape[-1] != self.n_coeff:
            raise ValueError(f"coeffs last dim {coeffs.shape[-1]} != n_coeff {self.n_coeff}")
        coeffs = coeffs.to(self.cdtype)
        # per-order theta synthesis, scattered onto FFT bins: g[..., j, bin]
        contrib = torch.einsum("...c,cj->...jc", coeffs, self._design.to(self.cdtype))
        g = coeffs.new_zeros((*coeffs.shape[:-1], self.n_theta, self.n_phi))
        g.index_add_(-1, self._freq_bins, contrib)
        # inverse FFT (unnormalized): f[..., j, k] = sum_bin g[..., j, bin] exp(+i bin phi_k)
        return self.n_phi * torch.fft.ifft(g, dim=-1)
