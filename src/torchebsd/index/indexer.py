"""End-to-end spherical indexer (I4/I5).

``SphericalIndexer`` ties the pipeline together: precompute the master's SH
spectrum once, then for a scan back-project each pattern to the sphere, transform,
SO(3)-cross-correlate, peak-find, optionally refine, and assemble the result into
an orix ``CrystalMap`` with quality metrics. Batched on the GPU. See
``INDEXING_DESIGN.md`` (milestones I4-I5).
"""

from __future__ import annotations


class SphericalIndexer:
    """Index EBSD patterns against a master pattern by spherical cross-correlation.

    Parameters
    ----------
    master
        A ``torchebsd.sim`` ``MasterPattern`` (or one imported from EMsoft) --
        the continuous "dictionary" ``M(n)``.
    detector
        ``torchebsd.sim`` ``EBSDDetector`` geometry.
    bandwidth
        Spherical-harmonic bandwidth ``L`` (the accuracy/speed knob).
    device
        Torch device for the transforms (``"cuda"`` recommended).
    """

    def __init__(self, master, detector, bandwidth=88, device=None):
        raise NotImplementedError("I4/I5: SphericalIndexer")

    def index(self, patterns, refine=False):
        """Index a scan of ``patterns`` (an array of shape (n, H, W)) -> orix ``CrystalMap``.

        Takes plain pattern arrays so no kikuchipy import is needed; a kikuchipy
        signal's ``.data`` and detector can be passed straight in.
        """
        raise NotImplementedError("I5: batched scan indexing -> CrystalMap")
