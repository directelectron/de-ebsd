"""Orientation indexing: the matching counterpart to :mod:`de_ebsd.sim`.

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
    "HREBSDResult",
    "SphericalIndexer",
    "determine_pattern_center",
    "hrebsd",
    "master_from_emsoft",
    "phase_from_crystal",
    "refine_hr",
    "refine_orientation",
    "remap_reference",
    "remove_background",
    "remove_dynamic_background",
    "remove_static_background",
    "to_crystal_map",
]

_ORIGIN = {
    "DictionaryIndexer": "de_ebsd.index.dictionary",
    "SphericalIndexer": "de_ebsd.index.spherical",
    "determine_pattern_center": "de_ebsd.index.refine",
    "HREBSDResult": "de_ebsd.index.crosscorr",
    "hrebsd": "de_ebsd.index.crosscorr",
    "remap_reference": "de_ebsd.index.crosscorr",
    "refine_hr": "de_ebsd.index.refine",
    "refine_orientation": "de_ebsd.index.refine",
    "master_from_emsoft": "de_ebsd.index.io",
    "phase_from_crystal": "de_ebsd.index.io",
    "to_crystal_map": "de_ebsd.index.io",
    "remove_background": "de_ebsd.index.preprocess",
    "remove_dynamic_background": "de_ebsd.index.preprocess",
    "remove_static_background": "de_ebsd.index.preprocess",
}


def __getattr__(name: str):
    # lazy so ``import de_ebsd.index`` stays light: orix, diffpy and
    # anyplotlib are only imported when actually used
    module = _ORIGIN.get(name)
    if module is None:
        raise AttributeError(f"module 'de_ebsd.index' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
