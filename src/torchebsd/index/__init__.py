"""Orientation indexing: the matching counterpart to :mod:`torchebsd.sim`.

:class:`SphericalIndexer` back-projects an experimental pattern to the sphere and
cross-correlates it against the master over all of SO(3) via spherical harmonics
and the Wigner-D transform (after EMSphInx; Lenthe, Singh & De Graef, 2019).
:class:`DictionaryIndexer` is the brute-force template-matching alternative.
Both emit orix objects for kikuchipy. See INDEXING_DESIGN.md.
"""

from __future__ import annotations

import importlib

__all__ = [
    "DictionaryIndexer",
    "SphericalIndexer",
    "determine_pattern_center",
    "master_from_emsoft",
    "phase_from_crystal",
    "refine_hr",
    "refine_orientation",
    "remove_background",
    "remove_dynamic_background",
    "remove_static_background",
    "to_crystal_map",
]

_ORIGIN = {
    "DictionaryIndexer": "torchebsd.index.dictionary",
    "SphericalIndexer": "torchebsd.index.spherical",
    "determine_pattern_center": "torchebsd.index.refine",
    "refine_hr": "torchebsd.index.refine",
    "refine_orientation": "torchebsd.index.refine",
    "master_from_emsoft": "torchebsd.index.io",
    "phase_from_crystal": "torchebsd.index.io",
    "to_crystal_map": "torchebsd.index.io",
    "remove_background": "torchebsd.index.preprocess",
    "remove_dynamic_background": "torchebsd.index.preprocess",
    "remove_static_background": "torchebsd.index.preprocess",
}


def __getattr__(name: str):
    # lazy so ``import torchebsd.index`` stays light: orix, diffpy and
    # anyplotlib are only imported when actually used
    module = _ORIGIN.get(name)
    if module is None:
        raise AttributeError(f"module 'torchebsd.index' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
