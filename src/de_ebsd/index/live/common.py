"""Small pieces the live indexer's modules share."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from de_ebsd.core.constants import SINGLE_PRECISION_FLOOR

#: The local dictionary of one refine level, in units of its step: the seed, then
#: +x, -x, +y, -y, +z, -z.
STENCIL = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

#: Patterns per batched step, which bounds the largest intermediate: the refine's
#: (patterns, 7, pixels, 3) sample directions.
BATCH = 256


def average_to(stack: torch.Tensor, shape) -> torch.Tensor:
    """*stack* ``(N, H, W)`` averaged down to *shape*: a whole-number bin when the
    shapes divide, which runs several times faster than a fractional one."""
    rows, cols = stack.shape[-2] // shape[0], stack.shape[-1] // shape[1]
    if (rows * shape[0], cols * shape[1]) == tuple(stack.shape[-2:]):
        return F.avg_pool2d(stack[:, None], (rows, cols))[:, 0]
    return F.adaptive_avg_pool2d(stack[:, None], shape)[:, 0]


def pick(frames: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``frames[rows]``, for a camera's uint16 frames too: CUDA cannot index a
    uint16 tensor, but it can the same bits as int16."""
    if frames.dtype == torch.uint16:
        return frames.view(torch.int16)[rows].view(torch.uint16)
    return frames[rows]


def parabola_offset(scores: torch.Tensor) -> torch.Tensor:
    """The step ``(N, 3)``, in units of the stencil's, to each axis's best: the
    vertex of the parabola through the scores ``(N, 7)`` at -1, 0 and +1 (see
    :data:`STENCIL`), or the better neighbor where the three make no maximum."""
    center, plus, minus = scores[:, :1], scores[:, 1::2], scores[:, 2::2]
    curvature = plus - 2 * center + minus
    vertex = 0.5 * (minus - plus) / curvature.clamp_max(-SINGLE_PRECISION_FLOOR)
    fallback = torch.where(plus > minus, 1.0, -1.0) * (torch.maximum(plus, minus) > center)
    return torch.where(curvature < 0, vertex.clamp(-1, 1), fallback.to(scores.dtype))
