"""orix / kikuchipy adapters (I8) -- the ecosystem handoff.

Converts between the indexer's internal tensors and the community objects, so the
module is drop-in for the orix/kikuchipy world (and ultimately upstreamable into
kikuchipy): indexed orientations -> orix ``Rotation`` / ``CrystalMap``, kikuchipy
EBSD ``Signal`` in, and import of EMsoft/kikuchipy master patterns. See
``INDEXING_DESIGN.md`` (milestone I8).
"""

from __future__ import annotations


def to_crystal_map(rotations, quality, shape, phase):
    """Assemble indexed orientations + quality metrics into an orix ``CrystalMap``."""
    raise NotImplementedError("I8: orix CrystalMap output")


def master_from_kikuchipy(signal):
    """Adapt a kikuchipy/EMsoft master pattern to the indexer's representation."""
    raise NotImplementedError("I8: kikuchipy/EMsoft master import")
