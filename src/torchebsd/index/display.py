"""Rich visualization for indexing (I7), matching the simulation side.

Figures via anyplotlib / anywidget: the SO(3) cross-correlation surface, IPF and
correlation/quality maps of an indexed scan (orix IPF colouring), and an
interactive experimental-vs-simulated pattern overlay (like ``sim.show_setup``).
See ``INDEXING_DESIGN.md`` (milestone I7).
"""

from __future__ import annotations


def show_correlation(correlation):
    """Render an SO(3) cross-correlation grid (peak = best orientation)."""
    raise NotImplementedError("I7: SO(3) correlation visualization")


def show_map(crystal_map):
    """IPF + quality maps of an indexed scan (orix ``CrystalMap``)."""
    raise NotImplementedError("I7: indexed-map visualization")
