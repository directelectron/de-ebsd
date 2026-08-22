"""Detector <-> sphere back-projection (I3).

Maps an experimental EBSD pattern from the detector plane onto the unit sphere
(and back), using ``torchebsd.sim``'s own ``EBSDDetector`` geometry (kikuchipy-
parity pattern-center conventions) -- the same it uses to *project* a master
pattern -- so torchebsd is self-contained (no kikuchipy dependency). The sphere
samples feed the SHT. Torch, batched, differentiable in the pattern-center
(enables PC refinement). See ``INDEXING_DESIGN.md`` (milestone I3).
"""

from __future__ import annotations


def pattern_to_sphere(patterns, detector, theta, phi):
    """Back-project detector ``patterns`` onto sphere directions (``theta``, ``phi``).

    ``detector`` is a ``torchebsd.sim`` ``EBSDDetector``. Returns the spherical
    samples ready for :func:`torchebsd.index.harmonics.forward_sht`.
    """
    raise NotImplementedError("I3: detector -> sphere back-projection")
