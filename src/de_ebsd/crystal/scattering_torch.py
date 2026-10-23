"""Differentiable torch port of :mod:`de_ebsd.crystal.scattering`.

Same Weickenmeier-Kohl form factors, with the Debye-Waller factor ``u`` as a
tensor so structure factors carry gradients back to the crystal parameters.
"""

from __future__ import annotations

import numpy as np
import torch

from de_ebsd.crystal.data.scattering_tables import WK_A, WK_B
from de_ebsd.crystal.scattering import (
    _C_EULER,
    _FOURPI,
    _FOURPI_PHON,
    _PI,
    _RIH2_TABLE,
)

__all__ = ["wk_scattering_factor_torch"]

_EI_TERMS = 400  # fixed series length (numpy stops adaptively; this over-covers)


def _ei(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)

    rational = (x >= -60.0) & (x < -1.0)
    xp = torch.where(rational, x.abs(), torch.ones_like(x))
    a1, a2, a3, a4 = 8.57332, 18.05901, 8.63476, 0.26777
    b1, b2, b3, b4 = 9.57332, 25.63295, 21.09965, 3.95849
    num = a4 + xp * (a3 + xp * (a2 + xp * (a1 + xp)))
    den = b4 + xp * (b3 + xp * (b2 + xp * (b1 + xp)))
    out = torch.where(rational, -(num / den) * torch.exp(-xp) / xp, out)

    series = x >= -1.0
    xs = torch.where(series, x, torch.full_like(x, -0.5))
    xs = torch.where(xs.abs() < 1e-300, torch.full_like(xs, 1e-300), xs)
    total = xs.clone()
    term = xs.clone()
    for i in range(1, _EI_TERMS):
        ri = float(i)
        term = term * xs * ri / ((ri + 1.0) ** 2)
        total = total + term
    out = torch.where(series, _C_EULER + torch.log(xs.abs()) + total, out)
    return out


def _rih2(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    x1 = 1.0 / x
    idx = (200.0 * x1).long().clamp(0, table.shape[0] - 2)
    f0 = table[idx]
    f1 = table[idx + 1]
    return f0 + 200.0 * (f1 - f0) * (x1 - 5.0e-4 * idx.to(x.dtype))


def _rih1(x1, x2, x3, table: torch.Tensor) -> torch.Tensor:
    x1, x2, x3 = torch.broadcast_tensors(x1, x2, x3)
    # The _ei series overflows for large arguments; the large-x branch uses _rih2
    # instead, so feed _ei a safe (small) argument wherever its result is unused
    # -- otherwise 0 * inf in the discarded branch would poison the gradient.
    big2 = x2 > 20.0
    x2s = torch.where(big2, x2, torch.ones_like(x2))
    x2e = torch.where(big2, torch.zeros_like(x2), x2)
    out = torch.where(big2, torch.exp(x2 - x1) * _rih2(x2s, table) / x2s, torch.exp(-x1) * _ei(x2e))
    big3 = x3 > 20.0
    x3s = torch.where(big3, x3, torch.ones_like(x3))
    x3e = torch.where(big3, torch.zeros_like(x3), x3)
    out = out - torch.where(
        big3, torch.exp(x3 - x1) * _rih2(x3s, table) / x3s, torch.exp(-x1) * _ei(x3e)
    )
    return out


def _ri1(bi: float, bj: float, g: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    g2 = g * g
    small = (max(bi, bj) * g2) <= 0.1

    base = _PI * (bi * np.log((bi + bj) / bi) + bj * np.log((bi + bj) / bj))
    bi2, bj2 = bi * bi, bj * bj
    temp = 0.5 * bi2 * np.log(bi / (bi + bj)) + 0.5 * bj2 * np.log(bj / (bi + bj))
    temp += 0.75 * (bi2 + bj2) - 0.25 * (bi + bj) ** 2 - 0.5 * (bi - bj) ** 2
    small_val = base + _PI * g2 * temp

    g2s = torch.where(small, torch.ones_like(g2), g2)
    big2, bjg2 = bi * g2s, bj * g2s
    val = 2.0 * _C_EULER + torch.log(big2) + torch.log(bjg2)
    val = val - 2.0 * _ei(-bi * bj * g2s / (bi + bj))
    val = val + _rih1(big2, big2 * bi / (bi + bj), big2, table)
    val = val + _rih1(bjg2, bjg2 * bj / (bi + bj), bjg2, table)
    return torch.where(small, small_val, val * _PI / g2s)


def _ri2(bi: float, bj: float, g: torch.Tensor, u: torch.Tensor, table) -> torch.Tensor:
    g2 = g * g
    u2 = u * u
    u22 = 0.5 * u2
    biuh, bjuh = bi + 0.5 * u2, bj + 0.5 * u2
    biu, bju = bi + u2, bj + u2

    eps = torch.maximum(u2, torch.full_like(u2, max(bi, bj))) * g2
    small = eps <= 0.1

    base = (bi + u2) * torch.log((bi + bj + u2) / (bi + u2))
    base = base + bj * torch.log((bi + bj + u2) / (bj + u2))
    base = base + u2 * torch.log(u2 / (bj + u2))
    base = base * _PI
    temp = 0.5 * u22 * u22 * torch.log(biu * bju / (u2 * u2))
    temp = temp + 0.5 * biuh * biuh * torch.log(biu / (biuh + bjuh))
    temp = temp + 0.5 * bjuh * bjuh * torch.log(bju / (biuh + bjuh))
    temp = temp + 0.25 * biu * biu + 0.5 * bi * bi
    temp = temp + 0.25 * bju * bju + 0.5 * bj * bj
    temp = temp - 0.25 * (biuh + bjuh) ** 2
    temp = temp - 0.5 * ((bi * biu - bj * bju) / (biuh + bjuh)) ** 2
    temp = temp - u22 * u22
    small_val = base + _PI * g2 * temp

    g2s = torch.where(small, torch.ones_like(g2), g2)
    val = _ei(-0.5 * u2 * g2s * biuh / biu) + _ei(-0.5 * u2 * g2s * bjuh / bju)
    val = val - _ei(-biuh * bjuh * g2s / (biuh + bjuh)) - _ei(-0.25 * u2 * g2s)
    val = val * 2.0
    val = val + _rih1(0.5 * u2 * g2s, 0.25 * u2 * g2s, 0.25 * u2 * u2 * g2s / biu, table)
    val = val + _rih1(0.5 * u2 * g2s, 0.25 * u2 * g2s, 0.25 * u2 * u2 * g2s / bju, table)
    val = val + _rih1(biuh * g2s, biuh * biuh * g2s / (biuh + bjuh), biuh * biuh * g2s / biu, table)
    val = val + _rih1(bjuh * g2s, bjuh * bjuh * g2s / (biuh + bjuh), bjuh * bjuh * g2s / bju, table)
    return torch.where(small, small_val, val * _PI / g2s)


def _weko(a: np.ndarray, b: np.ndarray, s: torch.Tensor) -> torch.Tensor:
    s2 = s * s
    inv_s2 = torch.where(s2 > 0.0, 1.0 / s2.clamp_min(1e-300), torch.zeros_like(s2))
    total = torch.zeros_like(s)
    for ai, bi in zip(a, b, strict=True):
        argu = bi * s2
        small = argu < 0.1
        large = argu > 20.0
        mid = ~small & ~large
        argu_safe = torch.where(mid, argu, torch.ones_like(argu))
        total = total + torch.where(small, ai * bi * (1.0 - 0.5 * argu), torch.zeros_like(s))
        total = total + torch.where(large, ai * inv_s2, torch.zeros_like(s))
        total = total + torch.where(
            mid, ai * (1.0 - torch.exp(-argu_safe)) * inv_s2, torch.zeros_like(s)
        )
    return total


def _fphon(
    g: torch.Tensor, ul: torch.Tensor, a: np.ndarray, b: np.ndarray, table: torch.Tensor
) -> torch.Tensor:
    fp2 = _FOURPI_PHON * _FOURPI_PHON
    a1 = np.asarray(a) * fp2
    b1 = np.asarray(b) / fp2
    dewa = torch.exp(-0.5 * ul * ul * g * g)
    out = torch.zeros_like(g)
    for j in range(4):
        out = out + a1[j] * a1[j] * (
            dewa * _ri1(b1[j], b1[j], g, table) - _ri2(b1[j], b1[j], g, ul, table)
        )
        for i in range(j):
            out = out + 2.0 * a1[j] * a1[i] * (
                dewa * _ri1(b1[i], b1[j], g, table) - _ri2(b1[i], b1[j], g, ul, table)
            )
    return out


def _fcore(g: torch.Tensor, z: int, voltage_kv: float) -> torch.Tensor:
    a0 = 0.5289
    k0 = 0.5068 * np.sqrt(1022.0 * voltage_kv + voltage_kv * voltage_kv)
    de = (6 * z) * 1.0e-3
    thetae = de / (2.0 * voltage_kv) * (2.0 * voltage_kv + 1022.0) / (voltage_kv + 1022.0)
    r = 0.885 * a0 / float(z) ** 0.333333
    ta = 1.0 / (k0 * r)
    tb = g / (2.0 * k0)
    omega = 2.0 * tb / ta
    kappa = thetae / ta
    k2, o2 = kappa * kappa, omega * omega

    x1 = (
        omega
        / ((1.0 + o2) * torch.sqrt(o2 + 4.0 * k2))
        * torch.log((omega + torch.sqrt(o2 + 4.0 * k2)) / (2.0 * kappa))
    )
    x2 = (
        1.0
        / torch.sqrt((1.0 + o2) ** 2 + 4.0 * k2 * o2)
        * torch.log(
            (1.0 + 2.0 * k2 + o2 + torch.sqrt((1.0 + o2) ** 2 + 4.0 * k2 * o2))
            / (2.0 * kappa * np.sqrt(1.0 + k2))
        )
    )
    x3_small = 1.0 / (4.0 * (1.0 + k2))
    big = omega > 1.0e-2
    omega_safe = torch.where(big, omega, torch.ones_like(omega))
    x3_big = (
        1.0
        / (omega_safe * torch.sqrt(o2 + 4.0 * (1.0 + k2)))
        * torch.log((omega_safe + torch.sqrt(o2 + 4.0 * (1.0 + k2))) / (2.0 * np.sqrt(1.0 + k2)))
    )
    x3 = torch.where(big, x3_big, torch.full_like(omega, x3_small))

    hi = (2 * z) / (ta * ta) * (-x1 + x2 - x3)
    return 4.0 / (a0 * a0) * 2.0 * _PI / (k0 * k0) * hi


def wk_scattering_factor_torch(
    z: int,
    g_wk: torch.Tensor,
    ul: torch.Tensor,
    voltage_kv: float,
    absorption: bool = True,
) -> torch.Tensor:
    """Complex WK scattering amplitude in torch, differentiable in ``ul``/``g_wk``.

    Torch port of :func:`de_ebsd.crystal.scattering.wk_scattering_factor`; see that for
    the parameters and units. ``ul`` and ``g_wk`` are tensors (pass ``ul`` as a
    ``requires_grad`` leaf for Debye-Waller gradients). Returns a complex tensor.
    """
    a = np.asarray(WK_A[z - 1], dtype=np.float64)
    b = np.asarray(WK_B[z - 1], dtype=np.float64)
    table = torch.as_tensor(_RIH2_TABLE, device=g_wk.device, dtype=g_wk.dtype)

    dewa = torch.exp(-0.5 * ul * ul * g_wk * g_wk)
    f_real = _FOURPI * dewa * _weko(a, b, g_wk / _FOURPI)

    if absorption:
        f_imag = _fcore(g_wk, z, voltage_kv) * dewa + _fphon(g_wk, ul, a, b, table)
    else:
        f_imag = torch.zeros_like(g_wk)

    k0 = 0.5068 * np.sqrt(1022.0 * voltage_kv + voltage_kv * voltage_kv)
    gamma = (voltage_kv + 511.0) / 511.0
    return torch.complex(f_real * gamma, f_imag * gamma * gamma / k0)
