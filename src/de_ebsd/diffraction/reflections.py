"""Reflection lists, excitation errors, and the strong/weak Bethe partition.

Ports of EMsoft's ``Initialize_ReflectionList``, ``CalcsgDouble`` and
``Apply_BethePotentials``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from de_ebsd.crystal.structure_factors import StructureFactors
from de_ebsd.diffraction.bethe import BetheParameters

if TYPE_CHECKING:
    from anyplotlib import Figure

__all__ = [
    "STRONG",
    "WEAK",
    "IGNORED",
    "ReflectionList",
    "master_reflection_list",
    "excitation_errors",
    "candidate_reflections",
    "partition_strong_weak",
]

# status codes
STRONG = 0
WEAK = 1
IGNORED = 2

_FORBIDDEN_TOL = 1e-4  # |U_g| in nm^-2 below which a reflection counts as forbidden


@dataclass
class ReflectionList:
    """A list of reflections, optionally classified for one beam direction.

    Attributes
    ----------
    hkl, g, d, u_mod
        Miller indices (N, 3), lengths ``|g|`` (nm^-1), spacings d (nm), and
        structure-factor moduli ``|U_g|`` (nm^-2).
    forbidden
        True where ``|U_g|`` is (numerically) zero; such reflections only matter
        as double-diffraction channels.
    k
        The beam direction (unit vector, Cartesian crystal frame) used for
        the classification, or None for the master list.
    sg
        Excitation errors (nm^-1) for ``k``, or None.
    status
        Per-reflection STRONG / WEAK / IGNORED for ``k``, or None.
    """

    hkl: np.ndarray
    g: np.ndarray
    d: np.ndarray = field(repr=False)
    u_mod: np.ndarray = field(repr=False)
    forbidden: np.ndarray = field(repr=False)
    k: np.ndarray | None = None
    sg: np.ndarray | None = field(default=None, repr=False)
    status: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.hkl)

    @property
    def n_strong(self) -> int:
        if self.status is None:
            raise ValueError("no classification; call simulator.reflections(k=...)")
        return int(np.sum(self.status == STRONG))

    @property
    def n_weak(self) -> int:
        if self.status is None:
            raise ValueError("no classification; call simulator.reflections(k=...)")
        return int(np.sum(self.status == WEAK))

    def strong(self) -> np.ndarray:
        """Miller indices of the strong reflections."""
        if self.status is None:
            raise ValueError("no classification; call simulator.reflections(k=...)")
        return self.hkl[self.status == STRONG]

    def weak(self) -> np.ndarray:
        """Miller indices of the weak reflections."""
        if self.status is None:
            raise ValueError("no classification; call simulator.reflections(k=...)")
        return self.hkl[self.status == WEAK]

    def show(self, n: int = 40) -> Figure:
        """Bar chart of ``|U_g|`` for the first ``n`` reflections.

        With a classification, bars are colored by status (strong / weak /
        ignored); without one, all bars share one color.
        """
        from de_ebsd.viz.figures import bar_figure

        take = slice(0, n)
        labels = [f"{h} {k} {l_}" for h, k, l_ in self.hkl[take]]
        values = self.u_mod[take]
        colors = None
        title = f"Reflections (of {len(self)})"
        if self.status is not None:
            palette = {STRONG: "#4fc3f7", WEAK: "#ffb74d", IGNORED: "#9e9e9e"}
            colors = [palette[s] for s in self.status[take]]
            title = (
                f"Reflections for k = {np.round(self.k, 4)} "
                f"({self.n_strong} strong / {self.n_weak} weak of {len(self)})"
            )
        return bar_figure(values, labels, title=title, colors=colors)

    def __repr__(self) -> str:
        if self.status is None:
            return f"ReflectionList({len(self)} reflections, master list)"
        return (
            f"ReflectionList({len(self)} candidates for k={np.round(self.k, 4)}: "
            f"{self.n_strong} strong, {self.n_weak} weak)"
        )


def master_reflection_list(sf: StructureFactors) -> ReflectionList:
    """All centering-allowed reflections with d >= dmin, sorted by ``|g|``."""
    u_mod = np.abs(sf.ucg)
    return ReflectionList(
        hkl=sf.hkl,
        g=sf.g,
        d=sf.d,
        u_mod=u_mod,
        forbidden=u_mod < _FORBIDDEN_TOL,
    )


def excitation_errors(
    g_cart: torch.Tensor,
    k_hat: torch.Tensor,
    wavelength: float,
) -> torch.Tensor:
    """Excitation errors s_g in nm^-1 (EMsoft ``Calcsg``), batched over beams.

    Parameters
    ----------
    g_cart
        Reflections in the Cartesian crystal frame, shape (G, 3), nm^-1.
    k_hat
        Unit incident-beam directions, shape (..., 3). The foil normal is
        taken along the beam (master pattern geometry).
    wavelength
        Electron wavelength in nm (refraction corrected).

    Returns
    -------
    torch.Tensor
        s_g of shape (..., G).
    """
    k = k_hat / wavelength  # (..., 3), nm^-1
    kg = k.unsqueeze(-2) + g_cart  # (..., G, 3)
    numerator = -(g_cart * (2.0 * k.unsqueeze(-2) + g_cart)).sum(-1)  # -g.(2k+g)
    denominator = 2.0 * (kg * k_hat.unsqueeze(-2)).sum(-1)  # 2 (k+g).n
    return numerator / denominator


def candidate_reflections(
    refl: ReflectionList,
    sg: np.ndarray,
    wavelength: float,
    bethe: BetheParameters,
) -> np.ndarray:
    """Mask of reflections kept for one beam direction.

    EMsoft ``Initialize_ReflectionList``: keep when
    ``|s_g| / (lambda |U_g|) <= cutoff``; forbidden reflections are kept as
    double-diffraction channels when ``|s_g| <= sg_double``.
    """
    with np.errstate(divide="ignore"):
        r_g = np.abs(sg) / (wavelength * np.maximum(refl.u_mod, 1e-300))
    keep = r_g <= bethe.cutoff
    keep[refl.forbidden] = np.abs(sg[refl.forbidden]) <= bethe.sg_double
    return keep


def partition_strong_weak(
    hkl: np.ndarray,
    sg: np.ndarray,
    forbidden: np.ndarray,
    sf: StructureFactors,
    bethe: BetheParameters,
) -> np.ndarray:
    """Classify candidate reflections as STRONG / WEAK / IGNORED.

    Port of EMsoft ``Apply_BethePotentials``: for each reflection g the
    perturbation strength is ``min_h |s_g| / (lambda |U_{g-h}|)`` over all
    candidates h; thresholds ``bethe.strong`` and ``bethe.weak`` split the
    classes. The incident beam (first entry, g = 0) is always strong.

    Systematically absent (double-diffraction) reflections carry no structure
    factor (``|U_g| ~ 0``) and so cannot be Bloch beams: they are never strong
    or weak. They stay in the candidate list only as difference partners for
    genuine double-diffraction paths ``U_{g-h}`` of the *allowed* beams. Without
    this exclusion the degenerate self difference ``g - g = 0`` (whose
    coefficient ``U_0`` is the largest in the table) would drive ``m`` down to
    ``|s_g|/(lambda |U_0|)`` and spuriously promote any near-Bragg forbidden
    reflection to a strong beam -- adding beams EMsoft never includes and
    corrupting the dynamical matrix.

    Parameters are the candidate subset (index 0 must be the g = 0 beam).
    """
    n = len(hkl)
    status = np.full(n, IGNORED, dtype=np.int64)
    status[0] = STRONG

    diff = hkl[:, None, :] - hkl[None, :, :]  # (N, N, 3)
    u_diff = np.abs(sf.lookup(diff))  # (N, N)
    inv_lambda = 1.0 / sf.wavelength

    # rh(i, h) = |s_i| / lambda / |U_{g_i - g_h}|; forbidden difference
    # vectors are excluded from the minimum (EMsoft sets rh = 10000 there).
    with np.errstate(divide="ignore"):
        rh = (inv_lambda * np.abs(sg))[:, None] / u_diff
    rh = np.where(u_diff > _FORBIDDEN_TOL, rh, 1.0e4)
    m = rh.min(axis=1)

    status[1:][m[1:] <= bethe.strong] = STRONG
    between = (m[1:] > bethe.strong) & (m[1:] <= bethe.weak)
    status[1:][between] = WEAK
    # forbidden reflections are difference partners only, never beams themselves
    status[forbidden] = IGNORED
    return status
