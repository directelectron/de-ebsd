"""Multi-candidate peak detection on the SO(3) correlation grid (future work).

Single-orientation indexing does **not** need this module: the best-orientation
argmax over ``C(g)`` is done directly in
:class:`torchebsd.index.indexer.SphericalIndexer` (``index`` / ``index_scan``), and
sub-grid-cell sharpening is done far better than grid upsampling by the
gradient refinement in :mod:`torchebsd.index.refine` (I6). This module is reserved
for the *multi*-candidate case that neither covers -- extracting the top ``n_peaks``
well-separated maxima of the SO(3) surface to disambiguate pseudo-symmetric
solutions or to score confidence from the peak ratio. It is not yet implemented.
"""

from __future__ import annotations


def peak_orientation(correlation, n_peaks=1):
    """Top ``n_peaks`` well-separated orientations from an SO(3) correlation grid.

    Would return orix ``Rotation``(s) with their correlation values as a quality
    metric. Not yet implemented -- for single-orientation indexing use
    :meth:`torchebsd.index.indexer.SphericalIndexer.index` (argmax) followed by
    :func:`torchebsd.index.refine.refine_orientation` (sub-grid-cell sharpening).
    """
    raise NotImplementedError(
        "multi-candidate SO(3) peak detection is future work; for a single "
        "orientation use SphericalIndexer.index + refine_orientation"
    )
