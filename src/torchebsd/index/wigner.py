"""Wigner-d / D matrices and the SO(3) grid (I2).

The building blocks of the SO(3) Fourier (SOFT) transform: the small Wigner-d
matrices ``d^l_{m'm}(beta)``, the full ``D^l`` rotation matrices, and the SO(3)
sampling grid on which the cross-correlation is evaluated. Torch, batched.
See ``INDEXING_DESIGN.md`` (milestone I2).
"""

from __future__ import annotations


def wigner_d(bandwidth, beta):
    """Small Wigner-d matrices ``d^l_{m'm}(beta)`` for all ``l <= bandwidth``."""
    raise NotImplementedError("I2: Wigner-d matrices")


def so3_grid(bandwidth):
    """The SOFT SO(3) sampling grid (Euler angles) for the given ``bandwidth``."""
    raise NotImplementedError("I2: SO(3) sampling grid")
