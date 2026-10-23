"""Structure factors U_g for every reflection within ``dmin``.

Sums Weickenmeier-Kohl form factors over the symmetry-expanded basis. Provides
the numpy path (:func:`compute_structure_factors`) used by the Bloch-wave
backend and a differentiable lookup (:func:`differentiable_ucg_lut`). Follows
EMsoft ``CalcUcg`` (Weickenmeier-Kohl branch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from de_ebsd.core.constants import C_PI, DWWK, PREF_COEF, PREG, SWK, electron_wavelength_nm
from de_ebsd.crystal.lattice import Crystal
from de_ebsd.crystal.scattering import atomic_number, wk_scattering_factor
from de_ebsd.crystal.scattering_torch import wk_scattering_factor_torch

if TYPE_CHECKING:
    from anyplotlib import Figure

__all__ = ["StructureFactors", "compute_structure_factors", "differentiable_ucg_lut"]


@dataclass
class StructureFactors:
    """Structure factors of a crystal at one beam energy (inspectable stage).

    Attributes
    ----------
    crystal, energy_kev, dmin
        The inputs that produced this table.
    hmax
        Largest Miller index with d >= dmin along any axis; the lookup table
        spans ``-2 hmax .. 2 hmax`` per axis so any difference of two
        reflections in the sphere is available.
    ucg_lut
        Complex U_g in nm^-2, shape (4 hmax + 1,) * 3, indexed
        ``[h + 2 hmax, k + 2 hmax, l + 2 hmax]``.
    v0
        Mean inner potential in volts.
    upz0
        Normal absorption coefficient U'_0 in nm^-2.
    wavelength
        Relativistic electron wavelength in nm, refraction corrected with v0.
    normal_absorption_length
        Normal absorption length ``1 / (U'_0 lambda)`` in nm (EMsoft ``xgp``
        for g = 0), used to correct the Monte Carlo depth profile.
    hkl, g, d, ucg
        Flat listing of the reflections with 0 < g and d >= dmin that are
        allowed by the lattice centering, sorted by g: Miller indices (N, 3),
        lengths (N,) in nm^-1, spacings (N,) in nm and complex U_g (N,).
    """

    crystal: Crystal
    energy_kev: float
    dmin: float
    hmax: int
    ucg_lut: np.ndarray
    v0: float
    upz0: float
    wavelength: float
    hkl: np.ndarray = field(repr=False)
    g: np.ndarray = field(repr=False)
    d: np.ndarray = field(repr=False)
    ucg: np.ndarray = field(repr=False)

    @property
    def normal_absorption_length(self) -> float:
        """Normal absorption length 1 / (U'_0 lambda) in nm."""
        if self.upz0 <= 0:
            return 1.0e8
        return 1.0 / (self.upz0 * self.wavelength)

    def lookup(self, hkl: np.ndarray) -> np.ndarray:
        """U_g for Miller indices (..., 3); indices must lie within the table."""
        hkl = np.asarray(hkl, dtype=np.int64)
        if np.any(np.abs(hkl) > 2 * self.hmax):
            raise IndexError(f"|hkl| exceeds the lookup range 2*hmax = {2 * self.hmax}")
        idx = hkl + 2 * self.hmax
        return self.ucg_lut[idx[..., 0], idx[..., 1], idx[..., 2]]

    def as_torch_lut(
        self, device: torch.device | str | None = None, dtype: torch.dtype = torch.complex64
    ) -> torch.Tensor:
        """The lookup table as a torch tensor for the dynamical solver."""
        return torch.as_tensor(self.ucg_lut, dtype=dtype, device=device)

    def families(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Group the reflection list by ``|g|``; representative hkl, g, and U_g.

        Returns the first ``n`` (or all) families sorted by g length.
        """
        keys = np.round(self.g, 6)
        _, first = np.unique(keys, return_index=True)
        order = np.sort(first)
        if n is not None:
            order = order[:n]
        return self.hkl[order], self.g[order], self.ucg[order]

    def show(self, n: int = 25) -> Figure:
        """Bar chart of ``|U_g|`` for the first ``n`` reflection families.

        Systematic absences appear as (near-)zero bars.
        """
        from de_ebsd.viz.figures import bar_figure

        hkl, g, ucg = self.families(n)
        labels = [f"{h} {k} {l_}" for h, k, l_ in hkl]
        return bar_figure(
            np.abs(ucg),
            labels,
            title=f"|U_g| [nm^-2] at {self.energy_kev:g} keV (V0 = {self.v0:.2f} V)",
        )

    def __repr__(self) -> str:
        return (
            f"StructureFactors({self.crystal!r}, energy={self.energy_kev:g} keV, "
            f"dmin={self.dmin:g} nm, {len(self.hkl)} reflections, "
            f"V0={self.v0:.2f} V, lambda={self.wavelength * 1e3:.4f} pm)"
        )


def _hmax_for_dmin(crystal: Crystal, dmin: float) -> int:
    hmax = 0
    for axis in range(3):
        h = 1
        while True:
            hkl = np.zeros(3)
            hkl[axis] = h
            if crystal.d_spacing(hkl) < dmin:
                break
            h += 1
            if h > 200:
                raise ValueError("dmin is unreasonably small for this lattice")
        hmax = max(hmax, h)
    return hmax


def compute_structure_factors(
    crystal: Crystal,
    energy_kev: float,
    dmin: float,
    absorption: bool = True,
) -> StructureFactors:
    """Compute U_g over the full lookup box for one beam energy.

    Parameters
    ----------
    crystal
        The crystal.
    energy_kev
        Beam energy in keV (numerically equal to the accelerating voltage
        in kV).
    dmin
        Smallest interplanar spacing to include, in nm.
    absorption
        Include the WK absorptive parts (imaginary U''_g).
    """
    hmax = _hmax_for_dmin(crystal, dmin)
    m = 2 * hmax
    rng = np.arange(-m, m + 1)
    h, k, l = np.meshgrid(rng, rng, rng, indexing="ij")  # noqa: E741
    hkl_box = np.stack([h, k, l], axis=-1).reshape(-1, 3)

    g_len = crystal.g_length(hkl_box)  # nm^-1
    g_wk = g_len * SWK  # WK units

    # Scattering factors depend on the element and |g| only: evaluate on the
    # unique lengths and scatter back.
    g_unique, inverse = np.unique(np.round(g_wk, 9), return_inverse=True)

    volume = crystal.volume
    pref = PREF_COEF / volume / (4.0 * C_PI)
    pre = pref * PREG

    ff = np.zeros(len(hkl_box), dtype=np.complex128)
    gg = np.zeros(len(hkl_box), dtype=np.complex128)
    for site, positions in zip(crystal.sites, crystal.positions, strict=True):
        z = atomic_number(site.element)
        ul = float(np.sqrt(site.debye_waller * DWWK))
        sf = wk_scattering_factor(z, g_unique, ul, energy_kev, absorption=absorption)
        sf = sf[inverse] * site.occupancy
        phase = np.exp(-2j * np.pi * (hkl_box @ positions.T)).sum(axis=1)
        ff += phase * sf.real
        gg += phase * sf.imag

    if absorption:
        ucg = pre * (ff + 1j * gg)
    else:
        ucg = pre * ff

    center = len(hkl_box) // 2  # index of (0, 0, 0)
    v0 = float(pref * np.abs(ff[center]))
    upz0 = float(PREG * pref * np.abs(gg[center])) if absorption else 0.0
    wavelength = electron_wavelength_nm(energy_kev, v0mod=v0)

    # Flat reflection listing for inspection: allowed by centering, inside the
    # dmin sphere, g > 0, sorted by g.
    keep = (g_len > 0) & (1.0 / np.maximum(g_len, 1e-12) >= dmin)
    keep &= crystal.allowed_by_centering(hkl_box)
    keep &= np.all(np.abs(hkl_box) <= hmax, axis=1)
    order = np.argsort(g_len[keep], kind="stable")
    hkl_list = hkl_box[keep][order]
    g_list = g_len[keep][order]

    n = 4 * hmax + 1
    return StructureFactors(
        crystal=crystal,
        energy_kev=float(energy_kev),
        dmin=float(dmin),
        hmax=hmax,
        ucg_lut=ucg.reshape(n, n, n),
        v0=v0,
        upz0=upz0,
        wavelength=float(wavelength),
        hkl=hkl_list,
        g=g_list,
        d=1.0 / g_list,
        ucg=ucg[keep][order],
    )


def differentiable_ucg_lut(
    crystal: Crystal,
    energy_kev: float,
    dmin: float,
    *,
    positions: list[torch.Tensor] | None = None,
    occupancies: list[torch.Tensor] | None = None,
    debye_waller: list[torch.Tensor] | None = None,
    absorption: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Assemble the ``U_g`` lookup table in torch, differentiable in the crystal.

    ``U_g = pre * sum_site occ_site * f_site(g) * sum_r exp(-2 pi i g.r)``. All
    three parameter groups are differentiable when passed as ``requires_grad``
    tensors:

    - **positions** -- the structure-factor phase ``sum_r exp(-2 pi i g.r)``;
    - **occupancies** -- the linear per-site weight;
    - **debye_waller** (``B`` in nm^2, one per site) -- feeds the torch WK
      amplitude (:func:`de_ebsd.crystal.scattering_torch.wk_scattering_factor_torch`),
      so ``U_g`` differentiates w.r.t. the thermal factor.

    When ``debye_waller`` is None the amplitudes are evaluated in numpy and held
    constant (positions/occupancy gradients only). Feed the result to a torch
    backend (``bloch_prism``) as its ``u_lut`` for end-to-end
    ``d(intensity)/d(crystal parameter)``. Lattice gradients (through ``g``)
    remain a further step.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    hmax = _hmax_for_dmin(crystal, dmin)
    m = 2 * hmax
    rng = np.arange(-m, m + 1)
    h, k, ll = np.meshgrid(rng, rng, rng, indexing="ij")
    hkl_box = np.stack([h, k, ll], axis=-1).reshape(-1, 3)
    g_wk = crystal.g_length(hkl_box) * SWK
    g_unique, inverse = np.unique(np.round(g_wk, 9), return_inverse=True)
    pre = PREF_COEF / crystal.volume / (4.0 * C_PI) * PREG

    hkl_t = torch.as_tensor(hkl_box, device=dev, dtype=torch.float64)
    inverse_t = torch.as_tensor(inverse, device=dev)
    g_unique_t = torch.as_tensor(g_unique, device=dev, dtype=torch.float64)
    if positions is None:
        positions = [torch.as_tensor(p, device=dev, dtype=torch.float64) for p in crystal.positions]
    if occupancies is None:
        occupancies = [torch.as_tensor(s.occupancy, device=dev, dtype=torch.float64)
                       for s in crystal.sites]
    dw = debye_waller if debye_waller is not None else [None] * len(crystal.sites)

    ucg = torch.zeros(len(hkl_box), device=dev, dtype=dtype)
    for site, pos, occ, b in zip(crystal.sites, positions, occupancies, dw, strict=True):
        z = atomic_number(site.element)
        if b is None:  # detached WK amplitude (numpy)
            ul = float(np.sqrt(site.debye_waller * DWWK))
            f = wk_scattering_factor(z, g_unique, ul, energy_kev, absorption=absorption)
            f_t = torch.as_tensor(f, device=dev, dtype=dtype)[inverse_t]
        else:  # torch WK amplitude, differentiable in the Debye-Waller factor
            ul = torch.sqrt(b.to(dev, torch.float64) * DWWK)
            f_unique = wk_scattering_factor_torch(
                z, g_unique_t, ul, energy_kev, absorption=absorption)
            f_t = f_unique.to(dtype)[inverse_t]
        # structure-factor phase sum over the site's atoms (differentiable in pos)
        phase = torch.exp(-2j * np.pi * (hkl_t @ pos.to(dev, torch.float64).T)).sum(dim=1)
        ucg = ucg + pre * occ.to(dev, dtype) * f_t * phase

    n = 4 * hmax + 1
    return ucg.reshape(n, n, n)
