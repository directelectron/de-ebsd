"""SO(3) cross-correlation of two spherical-harmonic spectra.

Combines the two spectra with the Wigner-d matrices into a coefficient cube,
then one 3-D inverse FFT gives the correlation over all of SO(3) on the SOFT
grid. See INDEXING_DESIGN.md (I2).
"""

from __future__ import annotations

import torch

from de_ebsd.index.spherical.harmonics import lm_indices, n_coefficients
from de_ebsd.index.spherical.wigner import so3_grid, wigner_d

__all__ = ["so3_cross_correlation"]


def _pack_to_grid(coeffs: torch.Tensor, bandwidth: int) -> torch.Tensor:
    L = int(bandwidth)
    degrees, orders = lm_indices(L, device=coeffs.device)
    pos = degrees * (2 * L + 1) + (orders + L)  # flat index into (L+1)*(2L+1)
    base = coeffs.new_zeros((*coeffs.shape[:-1], (L + 1) * (2 * L + 1)))
    packed = base.index_add(-1, pos, coeffs)
    return packed.reshape(*coeffs.shape[:-1], L + 1, 2 * L + 1)


def so3_cross_correlation(
    master_coeffs: torch.Tensor,
    pattern_coeffs: torch.Tensor,
    bandwidth: int,
    wigner_d_cached: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-correlation of two harmonic spectra over all of SO(3), via SOFT.

    Parameters
    ----------
    master_coeffs : torch.Tensor
        Master spectrum ``a_lm``, shape ``(..., (L+1)**2)`` (packed
        ``c = l*l + l + m``, the :mod:`.harmonics` convention). May carry leading
        batch dims that broadcast against ``pattern_coeffs``.
    pattern_coeffs : torch.Tensor
        Pattern spectrum ``b_lm``, shape ``(..., (L+1)**2)``. The leading dims are
        the batch of patterns (many indexed at once).
    bandwidth : int
        The band limit ``L``; both spectra must have ``(L+1)**2`` slots.
    wigner_d_cached : torch.Tensor, optional
        A precomputed Wigner-d block ``wigner_d(L, beta)`` for this bandwidth's SOFT
        ``beta`` grid, shape ``(N, L+1, 2L+1, 2L+1)`` with ``N = 2L + 2`` (see
        :func:`de_ebsd.index.spherical.wigner.wigner_d`). This ``O(L^4)`` block is otherwise
        rebuilt on every call; pass it to reuse one block across a mini-batched scan
        (:class:`~de_ebsd.index.spherical.indexer.SphericalIndexer` caches it in
        ``__init__``). The result is byte-identical to omitting it, provided the
        cached block matches this bandwidth's ``beta`` grid and the real dtype used
        here (``float64`` for complex128 spectra, else ``float32``).

    Returns
    -------
    torch.Tensor
        Real tensor ``C`` of shape ``(*batch, N, N, N)`` with ``N = 2L + 2``, sampled
        on the :func:`.so3_grid` ``(alpha, beta, gamma)`` grid (axes in that order),
        where ``batch`` is the broadcast of the two inputs' leading dims. ``argmax``
        gives the best-matching orientation index; feed it to
        :func:`.grid_to_rotation`. Differentiable in both spectra.
    """
    L = int(bandwidth)
    n_coeff = n_coefficients(L)
    if master_coeffs.shape[-1] != n_coeff or pattern_coeffs.shape[-1] != n_coeff:
        raise ValueError(
            f"expected {n_coeff} coefficients for bandwidth {L}, got "
            f"{master_coeffs.shape[-1]} (master) and {pattern_coeffs.shape[-1]} (pattern)"
        )
    a = master_coeffs
    b = pattern_coeffs
    cdtype = torch.result_type(a, b)
    if not cdtype.is_complex:
        cdtype = torch.complex128 if cdtype == torch.float64 else torch.complex64
    a = a.to(cdtype)
    b = b.to(cdtype)

    a_grid = _pack_to_grid(a, L)  # (..., L+1, 2L+1) over m'
    b_grid = _pack_to_grid(b, L)  # (..., L+1, 2L+1) over m
    # cross-spectrum f_{l, m, m'} = b_lm conj(a_lm')
    f = b_grid[..., :, :, None] * a_grid.conj()[..., :, None, :]  # (..., L+1, 2L+1, 2L+1)

    n = 2 * L + 2
    real_dtype = torch.float64 if cdtype == torch.complex128 else torch.float32
    alpha, beta, gamma = so3_grid(L, device=a.device, dtype=real_dtype)
    if wigner_d_cached is not None:
        d = wigner_d_cached.to(cdtype)  # reuse the precomputed O(L^4) block
    else:
        d = wigner_d(L, beta).to(cdtype)  # (N, L+1, 2L+1, 2L+1)

    # Wigner-d contraction over beta: S_{k, m, m'} = sum_l f_{l, m, m'} d^l_{m m'}(beta_k)
    s = torch.einsum("...lmn,klmn->...kmn", f, d)  # (..., N, 2L+1, 2L+1)

    # inverse FFT over alpha (m) and gamma (m'): pad the (2L+1) order axes to N,
    # then C = N^2 ifft2(s) with the linear phase that shifts m,m' back to -L..L
    s = torch.nn.functional.pad(s, (0, 1, 0, 1))  # (..., N, N, N)
    c = (n * n) * torch.fft.ifft2(s, dim=(-2, -1))  # axis -2 -> alpha, -1 -> gamma
    phase_a = torch.exp(-1j * L * alpha).to(cdtype).reshape(n, 1)
    phase_g = torch.exp(-1j * L * gamma).to(cdtype).reshape(1, n)
    c = c * phase_a * phase_g  # (..., N=beta, N=alpha, N=gamma)

    # reorder (beta, alpha, gamma) -> (alpha, beta, gamma) and take the real part
    c = c.transpose(-3, -2)
    return c.real.contiguous()
