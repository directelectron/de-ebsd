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
#: Patterns per batched step when the fused kernels do the refine and the fit,
#: which store no per-pixel intermediate: larger batches pay the launches less
#: often. On a Titan X, 256 to 1024 took the refine and the orientation fit from
#: 37 to 15 ms per 256 patterns.
FUSED_BATCH = 1024


def average_to(stack: torch.Tensor, shape) -> torch.Tensor:
    """*stack* ``(N, H, W)`` averaged down to *shape*: a whole-number bin when the
    shapes divide, which runs several times faster than a fractional one. A stack
    already at *shape* is returned as it is, not copied."""
    if tuple(stack.shape[-2:]) == tuple(shape):
        return stack
    rows, cols = stack.shape[-2] // shape[0], stack.shape[-1] // shape[1]
    if (rows * shape[0], cols * shape[1]) == tuple(stack.shape[-2:]):
        return F.avg_pool2d(stack[:, None], (rows, cols))[:, 0]
    return F.adaptive_avg_pool2d(stack[:, None], shape)[:, 0]


def pool_into(frames: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """*frames* ``(N, H, W)`` float32 averaged down into *out* ``(N, h, w)``, a
    buffer the caller owns, such as a slot of a ring of frames waiting to be
    indexed; *out* is returned.

    The same operation as :func:`average_to`, so the same values to the bit, and
    a stack pooled here can be given to :meth:`LiveIndexer.observe`,
    :meth:`LiveIndexer.prepare` and :meth:`LiveIndexer.index` in place of the
    frames: each then reads the small stack instead of the frames at full
    resolution. At the refine shape a 512 x 512 frame is 16 times smaller."""
    shape = tuple(out.shape[-2:])
    if tuple(frames.shape[-2:]) == shape:
        return out.copy_(frames)
    rows, cols = frames.shape[-2] // shape[0], frames.shape[-1] // shape[1]
    source, target = frames[:, None], out[:, None]
    if (rows * shape[0], cols * shape[1]) == tuple(frames.shape[-2:]):
        # the out variant of F.avg_pool2d, which the functional form does not expose
        torch.ops.aten.avg_pool2d.out(
            source, [rows, cols], [rows, cols], [0, 0], False, True, None, out=target
        )
    else:
        torch.ops.aten.adaptive_avg_pool2d.out(source, list(shape), out=target)
    return out


def normalize_rows_into(rows: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    """:func:`~de_ebsd.index.preprocess.normalize_rows` of *rows* ``(N, pixels)`` in
    place, the same operations in the same order, so the same values to the bit.
    *scratch*, at least ``2 N`` floats, holds each row's mean and length."""
    mean = scratch[: len(rows)].view(-1, 1)
    length = scratch[len(rows) : 2 * len(rows)].view(-1, 1)
    torch.mean(rows, -1, keepdim=True, out=mean)
    rows.sub_(mean)
    torch.linalg.vector_norm(rows, dim=-1, keepdim=True, out=length)
    return rows.div_(length.clamp_min_(SINGLE_PRECISION_FLOOR))


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
