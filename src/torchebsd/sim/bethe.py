"""Bethe-potential truncation parameters.

The Bloch-wave problem is truncated by classifying reflections as strong,
weak, or negligible using the perturbation strength ``|s_g| / (lambda |U_g|)``
(Callahan & De Graef 2013; EMsoft ``Apply_BethePotentials``):

- strong when the minimum over already-listed reflections is <= ``strong``,
- weak (folded in perturbatively) when between ``strong`` and ``weak``,
- ignored when above ``weak``.

``cutoff`` bounds the master reflection list itself, and ``sg_double`` is the
excitation-error window (nm^-1) within which forbidden reflections are kept
as potential double-diffraction channels.

Defaults are EMsoft's (``mod_diffraction.f90`` BetheParameterType).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BetheParameters"]


@dataclass(frozen=True)
class BetheParameters:
    """Strong/weak/ignore thresholds for the Bethe-potential truncation."""

    strong: float = 8.0
    weak: float = 50.0
    cutoff: float = 100.0
    sg_double: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.strong < self.weak <= self.cutoff:
            raise ValueError(
                "require 0 < strong < weak <= cutoff, got "
                f"strong={self.strong}, weak={self.weak}, cutoff={self.cutoff}"
            )
