"""Gradient-based refinement (I6) -- the differentiable payoff.

Sharpens an indexed orientation (and, optionally, the pattern center) by gradient
descent through the differentiable forward model: project the ``torchebsd.sim``
master pattern at the trial orientation/geometry, score against the experimental
pattern, and backprop. Because the master is also differentiable in the crystal
parameters (DESIGN.md M4), this same machinery extends to forward-model fitting of
the structure. See ``INDEXING_DESIGN.md`` (milestone I6).
"""

from __future__ import annotations


def refine_orientation(pattern, master, detector, orientation, refine_pc=False):
    """Refine ``orientation`` (and optionally the pattern center) by autograd.

    Minimises a pattern-match loss between the experimental ``pattern`` and the
    master projected at the trial orientation/geometry. Returns the refined orix
    ``Rotation`` (and pattern center when ``refine_pc``).
    """
    raise NotImplementedError("I6: gradient-based orientation / PC refinement")
