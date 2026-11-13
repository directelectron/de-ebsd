"""Deprecated: the scattering-matrix backend is in
:mod:`de_ebsd.diffraction.smatrix`. It was called PRISM, but it is not Ophus's
PRISM algorithm (see that module); the old names go in a later release."""

from __future__ import annotations

import warnings

from de_ebsd.diffraction.smatrix import BlochScatteringMatrixBackend


def __getattr__(name: str):
    if name == "BlochPrismBackend":
        warnings.warn(
            "BlochPrismBackend is now BlochScatteringMatrixBackend "
            "(de_ebsd.diffraction.smatrix); the old name goes in a later release",
            DeprecationWarning,
            stacklevel=2,
        )
        return BlochScatteringMatrixBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
