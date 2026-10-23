"""Bethe potential cutoffs that split reflections into strong/weak/ignored.

The scheme is EMsoft's ``Apply_BethePotentials``; the defaults here differ from
EMsoft's own (4 / 8 / 50) and must be matched to compare against it.
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
