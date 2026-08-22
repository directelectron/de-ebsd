"""Spherical harmonic transform (I1): a band-limited SHT on the sphere.

The dynamical master pattern and a back-projected experimental pattern are both
functions on the unit sphere. Spherical indexing (EMSphInx) matches them by
expanding each in spherical harmonics up to a bandwidth ``L`` and cross-correlating
the two spectra over SO(3). This module is the first primitive: the forward
transform (samples on the sphere -> complex coefficients ``a_lm``) and its inverse
(``a_lm`` -> samples), torch-native so the whole path is GPU-batched (a scan at
once) and autograd-differentiable. See ``INDEXING_DESIGN.md`` (milestone I1).

Design choices (settled here, per ``INDEXING_DESIGN.md`` section 8):

- **Basis: complex, fully normalized, Condon-Shortley phase.** Exactly SciPy's
  ``scipy.special.sph_harm_y(l, m, theta, phi)`` convention,

      Y_l^m(theta, phi) = N_l^m P_l^m(cos theta) exp(i m phi),
      N_l^m = sqrt((2l+1)/(4 pi) (l-m)! / (l+m)!),

  with ``theta`` the colatitude in ``[0, pi]`` (measured from +z) and ``phi`` the
  azimuth in ``[0, 2 pi)`` (measured from +x). Complex harmonics are the natural
  input to the Wigner-D / SO(3) step (I2), so we use them rather than the real
  basis. Cartesian direction ``n = (sin th cos ph, sin th sin ph, cos th)``.

- **Grid / quadrature: Gauss-Legendre in theta x equiangular in phi.** This is the
  exact quadrature for band-limited functions. A degree-``L`` function times a
  conjugate harmonic is a polynomial of degree <= 2L in ``cos theta``, so
  ``N_theta = L + 1`` Gauss-Legendre nodes (exact to degree ``2L + 1``) integrate
  the theta part exactly; the phi part is a Fourier mode of order <= 2L, so
  ``N_phi = 2L + 2`` equiangular nodes (the rectangle rule, exact to order
  ``N_phi - 1``) integrate it exactly. The forward transform is therefore a
  quadrature-weighted projection that inverts the synthesis exactly for any
  band-limited signal -- the round trip is exact to floating point, not merely
  approximate.

- **Kernel: separable design matrices (associated-Legendre matmul + phi FFT).**
  Because ``Y_l^m`` factors into a real associated-Legendre function of
  ``cos theta`` and the Fourier factor ``exp(i m phi)``, the transform separates:
  an FFT over phi followed by a per-order matrix multiply over theta (and the
  reverse for the inverse). This is the "design-matrix" approach specialised to
  the natural grid -- the associated-Legendre design matrix has size
  ``O(L^3)`` (a few MB at ``L ~ 120``) instead of the ``O(L^4)`` of a full dense
  ``Y`` matrix, which matters at the ``L ~ 60-120`` used for EBSD indexing. The
  normalized associated Legendre functions use the stable Holmes-Featherstone
  recurrence (2002), so there are no factorial overflows.

API:

- :class:`SphericalHarmonicTransform` -- a plan for a fixed bandwidth. It owns the
  quadrature grid (``theta``, ``phi``, ``weights``, and ``grid_directions()``) and
  the cached design matrices, and exposes ``.forward(values)`` / ``.inverse(coeffs)``
  (both batched over leading dims). This is the primary interface: build once,
  sample the sphere on ``grid_directions()``, transform many times.
- :func:`gauss_legendre_grid` -- build the quadrature grid for a bandwidth.
- :func:`spherical_harmonics` -- the dense complex basis ``Y_lm`` at arbitrary
  angles (the reference design matrix; used by the free functions below).
- :func:`forward_sht` / :func:`inverse_sht` -- stateless one-shot transforms via
  the dense basis, for arbitrary sample grids.

Coefficients are packed in a flat vector of length ``(L + 1) ** 2`` ordered by
``c = l * l + l + m`` (``l = 0..L``, ``m = -l..l``); :func:`lm_indices` returns the
``(degree, order)`` of each slot.
"""

from __future__ import annotations

import math

import numpy as np
import torch

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


def lm_indices(bandwidth: int, device: torch.device | str | None = None) -> tuple[
    torch.Tensor, torch.Tensor
]:
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
    """Gauss-Legendre nodes and weights on ``[-1, 1]`` (setup-time numpy -> torch)."""
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
    """Normalized associated Legendre functions with the Condon-Shortley phase.

    ``x = cos theta``, ``u = sin theta >= 0``. Returns a real tensor ``P`` of shape
    ``(L + 1, L + 1, *x.shape)`` with ``P[l, m] = N_l^m P_l^m(x)`` for ``0 <= m <=
    l`` and zero elsewhere, where ``N_l^m`` is the ``sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!)``
    normalization. Uses the stable Holmes-Featherstone (2002) recurrence; the phase
    is carried by the sectoral seed ``P_m^m = -sqrt((2m+1)/(2m)) sin theta P_{m-1}^{m-1}``.

    The grid angles are constants, so this is evaluated once (no autograd needed
    through ``theta``); differentiability of the transform is through the sample
    values, not the sampling directions.
    """
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
                (2 * ell + 1) * (ell + m - 1) * (ell - m - 1)
                / ((2 * ell - 3) * (ell - m) * (ell + m))
            )
            p[ell, m] = a * x * p[ell - 1, m] - b * p[ell - 2, m]
    return p


def spherical_harmonics(
    bandwidth: int, theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
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


def inverse_sht(
    coeffs: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
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

    Attributes:
        bandwidth: the band limit ``L``.
        theta, phi: 1-D quadrature nodes (``(L + 1,)`` and ``(2 L + 2,)``).
        weights: ``(L + 1, 2 L + 2)`` quadrature weights.
        n_coeff: number of packed coefficients ``(L + 1) ** 2``.
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
        self.device = torch.device(device) if device is not None else torch.device("cpu")
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
