"""Weickenmeier-Kohl (WK) electron scattering factors with absorption.

This is a faithful port of the WK routines shipped with EMsoft
(``EMsoftOO/Source/EMsoftOOLib/mod_others.f90``: ``FSCATT``, ``WEKO``,
``FPHON``, ``RI1``, ``RI2``, ``RIH1``, ``RIH2``, ``EI``, ``FCORE``,
``GETWK``), vectorized over the scattering vector with numpy (float64).

Reference: A. Weickenmeier and H. Kohl, "Computation of absorptive form
factors for high-energy electron diffraction", Acta Cryst. A47 (1991) 590.

Conventions (identical to EMsoft):

- The scattering vector ``g_wk`` is in WK units: ``g_wk = |g| [nm^-1] * 2*pi/10``
  (Angstrom units with a 2*pi-scaled reciprocal space).
- ``ul`` is the RMS thermal displacement in Angstrom:
  ``ul = sqrt(B [nm^2] * 100 / (8 pi^2))``.
- The returned complex amplitude is in Angstrom and carries an extra factor
  of 4*pi; the relativistic factor gamma is applied to the real part and
  gamma^2 / k0 to the imaginary part (``accflg = .TRUE.`` in EMsoft).

The exact single-precision constants from the Fortran source are kept so
results track EMsoft closely.

This module is intentionally numpy-only: scattering factors are computed once
per (element, |g|) at simulation setup. Porting it to torch to make gradients
flow into lattice and Debye-Waller parameters is a planned extension
(see DESIGN.md, roadmap M4).
"""

from __future__ import annotations

import numpy as np

from torchebsd.data.scattering_tables import ELEMENT_SYMBOLS, WK_A, WK_B

# Exact constants from the Fortran source (single-precision literals).
_FOURPI = 12.56637062  # FSCATT
_FOURPI_PHON = 12.56636  # FPHON uses a slightly different 4*pi
_PI = 3.1415927
_C_EULER = 0.5772157

_ATOMIC_NUMBER = {symbol: i + 1 for i, symbol in enumerate(ELEMENT_SYMBOLS)}

# Abramowitz & Stegun table used by RIH2 for x * exp(-x) * Ei(x) at large x.
_RIH2_TABLE = np.array(
    [1.000000, 1.005051, 1.010206, 1.015472, 1.020852,
     1.026355, 1.031985, 1.037751, 1.043662, 1.049726,
     1.055956, 1.062364, 1.068965, 1.075780, 1.082830,
     1.090140, 1.097737, 1.105647, 1.113894, 1.122497,
     1.131470]
)


def atomic_number(element: str) -> int:
    """Atomic number Z for an element symbol (case sensitive, e.g. ``"Ni"``)."""
    try:
        return _ATOMIC_NUMBER[element]
    except KeyError:
        raise ValueError(f"Unknown element symbol {element!r}") from None


def _ei(x: np.ndarray) -> np.ndarray:
    """Exponential integral Ei(x), EMsoft ``EI`` (tested for -60 < x < 60)."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(x > 60.0):
        raise ValueError("Ei(x) untested for x > 60 (matches EMsoft behavior)")
    out = np.zeros_like(x)

    rational = (x >= -60.0) & (x < -1.0)
    if np.any(rational):
        # Abramowitz & Stegun (5.1.56)
        a1, a2, a3, a4 = 8.57332, 18.05901, 8.63476, 0.26777
        b1, b2, b3, b4 = 9.57332, 25.63295, 21.09965, 3.95849
        xp = np.abs(x[rational])
        num = a4 + xp * (a3 + xp * (a2 + xp * (a1 + xp)))
        den = b4 + xp * (b3 + xp * (b2 + xp * (b1 + xp)))
        out[rational] = -(num / den) * np.exp(-xp) / xp

    series = x >= -1.0
    if np.any(series):
        xs = x[series]
        result = _C_EULER + np.log(np.abs(xs))
        term = xs.copy()
        total = term.copy()
        for i in range(1, 500):
            ri = float(i)
            term = term * xs * ri / ((ri + 1.0) ** 2)
            total += term
            with np.errstate(invalid="ignore"):
                if np.all(np.abs(term) <= 1.0e-6 * np.abs(xs)):
                    break
        out[series] = result + total
    return out


def _rih2(x: np.ndarray) -> np.ndarray:
    """x * exp(-x) * Ei(x) for large x, by table interpolation (EMsoft ``RIH2``)."""
    x = np.asarray(x, dtype=np.float64)
    x1 = 1.0 / x
    idx = (200.0 * x1).astype(np.int64)
    idx = np.clip(idx, 0, len(_RIH2_TABLE) - 2)
    f0 = _RIH2_TABLE[idx]
    f1 = _RIH2_TABLE[idx + 1]
    return f0 + 200.0 * (f1 - f0) * (x1 - 5.0e-4 * idx)


def _rih1(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """exp(-x1) * (Ei(x2) - Ei(x3)) with large-argument handling (EMsoft ``RIH1``)."""
    x1, x2, x3 = np.broadcast_arrays(
        np.asarray(x1, dtype=np.float64),
        np.asarray(x2, dtype=np.float64),
        np.asarray(x3, dtype=np.float64),
    )
    out = np.zeros(x1.shape, dtype=np.float64)

    big2 = x2 > 20.0
    small2 = ~big2
    if np.any(big2):
        out[big2] += np.exp(x2[big2] - x1[big2]) * _rih2(x2[big2]) / x2[big2]
    if np.any(small2):
        out[small2] += np.exp(-x1[small2]) * _ei(x2[small2])

    big3 = x3 > 20.0
    small3 = ~big3
    if np.any(big3):
        out[big3] -= np.exp(x3[big3] - x1[big3]) * _rih2(x3[big3]) / x3[big3]
    if np.any(small3):
        out[small3] -= np.exp(-x1[small3]) * _ei(x3[small3])
    return out


def _ri1(bi: float, bj: float, g: np.ndarray) -> np.ndarray:
    """First absorptive-potential integral (EMsoft ``RI1``). bi, bj scalars."""
    g = np.asarray(g, dtype=np.float64)
    g2 = g * g
    out = np.zeros_like(g)
    eps = max(bi, bj) * g2

    small = eps <= 0.1
    if np.any(small):
        base = _PI * (bi * np.log((bi + bj) / bi) + bj * np.log((bi + bj) / bj))
        bi2 = bi * bi
        bj2 = bj * bj
        temp = 0.5 * bi2 * np.log(bi / (bi + bj)) + 0.5 * bj2 * np.log(bj / (bi + bj))
        temp += 0.75 * (bi2 + bj2) - 0.25 * (bi + bj) ** 2
        temp -= 0.5 * (bi - bj) ** 2
        out[small] = base + _PI * g2[small] * temp

    asym = ~small
    if np.any(asym):
        g2a = g2[asym]
        big2 = bi * g2a
        bjg2 = bj * g2a
        val = 2.0 * _C_EULER + np.log(big2) + np.log(bjg2)
        val -= 2.0 * _ei(-bi * bj * g2a / (bi + bj))
        val += _rih1(big2, big2 * bi / (bi + bj), big2)
        val += _rih1(bjg2, bjg2 * bj / (bi + bj), bjg2)
        out[asym] = val * _PI / g2a
    return out


def _ri2(bi: float, bj: float, g: np.ndarray, u: float) -> np.ndarray:
    """Second absorptive-potential integral (EMsoft ``RI2``). bi, bj, u scalars."""
    g = np.asarray(g, dtype=np.float64)
    g2 = g * g
    out = np.zeros_like(g)

    u2 = u * u
    u22 = 0.5 * u2
    biuh = bi + 0.5 * u2
    bjuh = bj + 0.5 * u2
    biu = bi + u2
    bju = bj + u2

    eps = max(bi, bj, u2) * g2
    small = eps <= 0.1
    if np.any(small):
        base = (bi + u2) * np.log((bi + bj + u2) / (bi + u2))
        base += bj * np.log((bi + bj + u2) / (bj + u2))
        base += u2 * np.log(u2 / (bj + u2))
        base *= _PI
        temp = 0.5 * u22 * u22 * np.log(biu * bju / (u2 * u2))
        temp += 0.5 * biuh * biuh * np.log(biu / (biuh + bjuh))
        temp += 0.5 * bjuh * bjuh * np.log(bju / (biuh + bjuh))
        temp += 0.25 * biu * biu + 0.5 * bi * bi
        temp += 0.25 * bju * bju + 0.5 * bj * bj
        temp -= 0.25 * (biuh + bjuh) ** 2
        temp -= 0.5 * ((bi * biu - bj * bju) / (biuh + bjuh)) ** 2
        temp -= u22 * u22
        out[small] = base + _PI * g2[small] * temp

    asym = ~small
    if np.any(asym):
        g2a = g2[asym]
        val = _ei(-0.5 * u2 * g2a * biuh / biu) + _ei(-0.5 * u2 * g2a * bjuh / bju)
        val -= _ei(-biuh * bjuh * g2a / (biuh + bjuh)) + _ei(-0.25 * u2 * g2a)
        val *= 2.0
        val += _rih1(0.5 * u2 * g2a, 0.25 * u2 * g2a, 0.25 * u2 * u2 * g2a / biu)
        val += _rih1(0.5 * u2 * g2a, 0.25 * u2 * g2a, 0.25 * u2 * u2 * g2a / bju)
        val += _rih1(biuh * g2a, biuh * biuh * g2a / (biuh + bjuh), biuh * biuh * g2a / biu)
        val += _rih1(bjuh * g2a, bjuh * bjuh * g2a / (biuh + bjuh), bjuh * bjuh * g2a / bju)
        out[asym] = val * _PI / g2a
    return out


def _weko(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Elastic electron scattering amplitude f(s) (EMsoft ``WEKO``)."""
    s = np.asarray(s, dtype=np.float64)
    s2 = s * s
    with np.errstate(divide="ignore"):
        inv_s2 = np.where(s2 > 0.0, 1.0 / np.maximum(s2, 1e-300), 0.0)
    total = np.zeros_like(s)
    for ai, bi in zip(a, b, strict=True):
        argu = bi * s2
        small = argu < 0.1
        large = argu > 20.0
        mid = ~small & ~large
        total += np.where(small, ai * bi * (1.0 - 0.5 * argu), 0.0)
        total += np.where(large, ai * inv_s2, 0.0)
        # only evaluate exp where needed to stay in range
        total += np.where(mid, ai * (1.0 - np.exp(-np.where(mid, argu, 1.0))) * inv_s2, 0.0)
    return total


def _fphon(g: np.ndarray, ul: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Phonon (thermal diffuse) absorptive form factor (EMsoft ``FPHON``)."""
    g = np.asarray(g, dtype=np.float64)
    fp2 = _FOURPI_PHON * _FOURPI_PHON
    a1 = np.asarray(a, dtype=np.float64) * fp2
    b1 = np.asarray(b, dtype=np.float64) / fp2
    u2 = ul * ul
    dewa = np.exp(-0.5 * u2 * g * g)

    out = np.zeros_like(g)
    for j in range(4):
        out += a1[j] * a1[j] * (dewa * _ri1(b1[j], b1[j], g) - _ri2(b1[j], b1[j], g, ul))
        for i in range(j):
            out += 2.0 * a1[j] * a1[i] * (
                dewa * _ri1(b1[i], b1[j], g) - _ri2(b1[i], b1[j], g, ul)
            )
    return out


def _fcore(g: np.ndarray, z: int, voltage_kv: float) -> np.ndarray:
    """Core-loss absorptive form factor (EMsoft ``FCORE``, after H. Rose)."""
    g = np.asarray(g, dtype=np.float64)
    a0 = 0.5289

    k0 = 0.5068 * np.sqrt(1022.0 * voltage_kv + voltage_kv * voltage_kv)
    de = (6 * z) * 1.0e-3
    thetae = de / (2.0 * voltage_kv) * (2.0 * voltage_kv + 1022.0) / (voltage_kv + 1022.0)
    r = 0.885 * a0 / float(z) ** 0.333333
    ta = 1.0 / (k0 * r)
    tb = g / (2.0 * k0)

    omega = 2.0 * tb / ta
    kappa = thetae / ta
    k2 = kappa * kappa
    o2 = omega * omega

    x1 = omega / ((1.0 + o2) * np.sqrt(o2 + 4.0 * k2)) * np.log(
        (omega + np.sqrt(o2 + 4.0 * k2)) / (2.0 * kappa)
    )
    x2 = 1.0 / np.sqrt((1.0 + o2) ** 2 + 4.0 * k2 * o2) * np.log(
        (1.0 + 2.0 * k2 + o2 + np.sqrt((1.0 + o2) ** 2 + 4.0 * k2 * o2))
        / (2.0 * kappa * np.sqrt(1.0 + k2))
    )
    x3_small = 1.0 / (4.0 * (1.0 + k2))
    big = omega > 1.0e-2
    omega_safe = np.where(big, omega, 1.0)
    x3_big = 1.0 / (omega_safe * np.sqrt(o2 + 4.0 * (1.0 + k2))) * np.log(
        (omega_safe + np.sqrt(o2 + 4.0 * (1.0 + k2))) / (2.0 * np.sqrt(1.0 + k2))
    )
    x3 = np.where(big, x3_big, x3_small)

    hi = (2 * z) / (ta * ta) * (-x1 + x2 - x3)
    return 4.0 / (a0 * a0) * 2.0 * _PI / (k0 * k0) * hi


def wk_scattering_factor(
    z: int,
    g_wk: np.ndarray,
    ul: float,
    voltage_kv: float,
    absorption: bool = True,
    apply_debye_waller: bool = True,
) -> np.ndarray:
    """Complex WK scattering amplitude, EMsoft ``FSCATT`` with ``accflg=.TRUE.``.

    Parameters
    ----------
    z
        Atomic number (1-98).
    g_wk
        Scattering vector lengths in WK units, ``|g| [nm^-1] * 2 pi / 10``.
    ul
        RMS thermal displacement in Angstrom, ``sqrt(B [nm^2] * 100/(8 pi^2))``.
    voltage_kv
        Accelerating voltage in kV.
    absorption
        Include the phonon and core absorptive parts (EMsoft ``absflg = 3``).
    apply_debye_waller
        Multiply the elastic part with the Debye-Waller factor.

    Returns
    -------
    numpy.ndarray
        Complex amplitude, same shape as ``g_wk``. Real part is the elastic
        amplitude times gamma; imaginary part is the absorptive amplitude
        times gamma^2 / k0. Units: Angstrom times 4 pi.
    """
    if not 1 <= z <= 98:
        raise ValueError(f"Z = {z} is out of range 1..98")
    if ul < 0:
        raise ValueError("ul must be non-negative")
    g_wk = np.asarray(g_wk, dtype=np.float64)
    if np.any(g_wk < 0):
        raise ValueError("g_wk must be non-negative")

    a = np.asarray(WK_A[z - 1], dtype=np.float64)
    b = np.asarray(WK_B[z - 1], dtype=np.float64)

    dewa = np.exp(-0.5 * ul * ul * g_wk * g_wk) if apply_debye_waller else np.ones_like(g_wk)
    s = g_wk / _FOURPI
    f_real = _FOURPI * dewa * _weko(a, b, s)

    if absorption:
        f_imag = _fcore(g_wk, z, voltage_kv) * dewa
        if ul > 0:  # no thermal displacement -> no phonon absorption
            f_imag = f_imag + _fphon(g_wk, ul, a, b)
    else:
        f_imag = np.zeros_like(g_wk)

    k0 = 0.5068 * np.sqrt(1022.0 * voltage_kv + voltage_kv * voltage_kv)
    gamma = (voltage_kv + 511.0) / 511.0
    f_real = f_real * gamma
    f_imag = f_imag * gamma * gamma / k0
    return f_real + 1j * f_imag
