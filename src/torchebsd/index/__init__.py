"""torchebsd.index -- GPU-accelerated, differentiable spherical indexing for EBSD.

The matching counterpart to :mod:`torchebsd.sim`'s master-pattern *generation*:
back-project an experimental pattern to the sphere, cross-correlate it against the
(differentiable) master pattern over all of SO(3) via spherical harmonics and the
Wigner-D transform, read off the orientation, and emit orix objects for kikuchipy.

Modelled on EMSphInx (Lenthe, Singh & De Graef, 2019); see ``INDEXING_DESIGN.md``
for the method, project structure, and milestones (I1-I8). This package is
scaffolded -- implementations land per milestone.
"""

from __future__ import annotations

__all__ = [
    "SphericalIndexer",
    "phase_from_crystal",
    "refine_orientation",
    "show_correlation",
    "show_map",
    "show_match",
    "show_refinement",
    "to_crystal_map",
]

_DISPLAY_FUNCS = ("show_correlation", "show_map", "show_match", "show_refinement")


def __getattr__(name: str):
    # lazy so ``import torchebsd.index`` stays light until the engine is built
    # (and so orix/diffpy/anyplotlib are only imported when actually used)
    if name == "SphericalIndexer":
        from torchebsd.index.indexer import SphericalIndexer

        return SphericalIndexer
    if name == "refine_orientation":
        from torchebsd.index.refine import refine_orientation

        return refine_orientation
    if name in ("phase_from_crystal", "to_crystal_map"):
        from torchebsd.index import io

        return getattr(io, name)
    if name in _DISPLAY_FUNCS:
        from torchebsd.index import display

        return getattr(display, name)
    raise AttributeError(f"module 'torchebsd.index' has no attribute {name!r}")
