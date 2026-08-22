"""Spherical harmonic transform (I1): a band-limited SHT on the sphere.

Forward: real samples of a spherical function -> complex coefficients ``a_lm`` up
to bandwidth ``L``. Inverse: ``a_lm`` -> samples. Implemented in torch so it is
GPU-batched (a whole scan at once) and autograd-differentiable. See
``INDEXING_DESIGN.md`` (milestone I1).
"""

from __future__ import annotations


def forward_sht(values, theta, phi, bandwidth):
    """Forward transform: sphere samples -> ``a_lm``.

    ``values`` are real samples at colatitude/longitude (``theta``, ``phi``);
    returns complex coefficients up to ``bandwidth`` (L). Batched over leading dims.
    """
    raise NotImplementedError("I1: forward spherical harmonic transform")


def inverse_sht(coeffs, theta, phi):
    """Inverse transform: coefficients ``a_lm`` -> samples on the sphere."""
    raise NotImplementedError("I1: inverse spherical harmonic transform")
