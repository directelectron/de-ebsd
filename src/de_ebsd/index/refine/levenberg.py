"""The Levenberg-Marquardt damping schedule shared by the refiners.

A step that improves the score is accepted and the damping eased toward
Gauss-Newton; a step that does not is rejected and the damping raised toward
steepest descent.
"""

from __future__ import annotations

import torch

__all__ = ["DAMPING_CEILING", "DAMPING_FLOOR", "adapt_damping", "adapt_shared_damping"]

DAMPING_FLOOR = 1e-6
DAMPING_CEILING = 1e2
_EASE = 0.5
_RAISE = 3.0


def adapt_damping(damping: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
    """Per-pattern damping ``(N,)`` after a step that was *accepted* ``(N,)`` or not."""
    return torch.where(
        accepted,
        (damping * _EASE).clamp_min(DAMPING_FLOOR),
        (damping * _RAISE).clamp_max(DAMPING_CEILING),
    )


def adapt_shared_damping(damping: float, accepted: bool) -> float:
    """The damping of one shared fit after a step that was *accepted* or not."""
    if accepted:
        return max(damping * _EASE, DAMPING_FLOOR)
    return min(damping * _RAISE, DAMPING_CEILING)
