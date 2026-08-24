"""torchebsd.index -- GPU-accelerated, differentiable spherical indexing for EBSD.

The matching counterpart to :mod:`torchebsd.sim`'s master-pattern *generation*. Like
dictionary / template indexing it recovers each pattern's orientation by matching it
against the master pattern -- but instead of brute-force scanning a precomputed
dictionary of simulated patterns, it back-projects the experimental pattern to the
sphere, cross-correlates it against the (differentiable) master pattern over all of
SO(3) via spherical harmonics and the Wigner-D transform, reads off the orientation,
and emits orix objects for kikuchipy.

Modelled on EMSphInx (Lenthe, Singh & De Graef, 2019); see ``INDEXING_DESIGN.md``
for the method, project structure, and milestones (I1-I8). The I1-I8 pipeline is
implemented; multi-candidate peak detection (:mod:`torchebsd.index.peaks`) is the one
piece still outstanding.
"""

from __future__ import annotations

__all__ = [
    "DictionaryIndexer",
    "SphericalIndexer",
    "master_from_emsoft",
    "phase_from_crystal",
    "refine_hr",
    "refine_orientation",
    "remove_background",
    "remove_dynamic_background",
    "remove_static_background",
    "show_correlation",
    "show_map",
    "show_match",
    "show_refinement",
    "to_crystal_map",
]

_DISPLAY_FUNCS = ("show_correlation", "show_map", "show_match", "show_refinement")
_PREPROCESS_FUNCS = (
    "remove_background",
    "remove_dynamic_background",
    "remove_static_background",
)


def __getattr__(name: str):
    # lazy so ``import torchebsd.index`` stays light until the engine is built
    # (and so orix/diffpy/anyplotlib are only imported when actually used)
    if name == "SphericalIndexer":
        from torchebsd.index.indexer import SphericalIndexer

        return SphericalIndexer
    if name == "DictionaryIndexer":
        from torchebsd.index.dictionary import DictionaryIndexer

        return DictionaryIndexer
    if name == "refine_orientation":
        from torchebsd.index.refine import refine_orientation

        return refine_orientation
    if name == "refine_hr":
        from torchebsd.index.refine_highres import refine_hr

        return refine_hr
    if name in ("phase_from_crystal", "to_crystal_map", "master_from_emsoft"):
        from torchebsd.index import io

        return getattr(io, name)
    if name in _DISPLAY_FUNCS:
        from torchebsd.index import display

        return getattr(display, name)
    if name in _PREPROCESS_FUNCS:
        from torchebsd.index import preprocess

        return getattr(preprocess, name)
    raise AttributeError(f"module 'torchebsd.index' has no attribute {name!r}")
