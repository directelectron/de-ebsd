"""orix-native output + array ingest (I8) -- the ecosystem handoff.

torchebsd is the *foundation*: it emits orix objects and takes plain pattern
arrays, so it never imports kikuchipy. The intended direction is that kikuchipy
(and other tools) depend on torchebsd -- they pass their pattern data + detector
in and consume the orix ``Rotation`` / ``CrystalMap`` out. See
``INDEXING_DESIGN.md`` (milestone I8).
"""

from __future__ import annotations


def to_crystal_map(rotations, quality, shape, phase):
    """Assemble indexed orientations + quality metrics into an orix ``CrystalMap``."""
    raise NotImplementedError("I8: orix CrystalMap output")


def master_from_emsoft(path):
    """Load an EMsoft master pattern (via ``torchebsd.sim``) for indexing."""
    raise NotImplementedError("I8: EMsoft master import")
