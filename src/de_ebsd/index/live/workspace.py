"""The buffers a :class:`~de_ebsd.index.live.LiveIndexer` indexes a batch in.

Every buffer is allocated once, when the indexer is built, sized for the most
patterns one batched step takes (:data:`~de_ebsd.index.live.common.FUSED_BATCH`).
Indexing a buffer of frames then allocates nothing on the device: each step writes
into the slice of these it needs, and the next batch overwrites them. A memory
pool would hide the cost of allocating; it would not stop the allocating.
"""

from __future__ import annotations

import math

import torch


class Workspace:
    """The working buffers of one indexer's fused path, for up to *capacity* patterns.

    Parameters
    ----------
    capacity
        The most patterns a batch holds.
    search_pixels, refine_pixels, coarse_pixels
        Pixels of a pattern at the search, refine and coarse shapes.
    entries_per_chunk
        The widest chunk of dictionary entries the search scores at once.
    dtype
        The search dictionary's.
    device
        Where the buffers live.
    """

    def __init__(
        self,
        capacity: int,
        *,
        search_pixels: int,
        refine_pixels: int,
        coarse_pixels: int,
        entries_per_chunk: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        def empty(*shape, dtype=torch.float32):
            return torch.empty(shape, device=device, dtype=dtype)

        self.capacity = capacity
        self.search_rows = empty(capacity, search_pixels)
        self.refine_rows = empty(capacity, refine_pixels)
        self.coarse_rows = empty(capacity, coarse_pixels)
        #: The search rows in the dictionary's precision, where it is not float32.
        self.typed_search_rows = (
            None if dtype == torch.float32 else empty(capacity, search_pixels, dtype=dtype)
        )
        # The search's scores of one chunk of entries, then the fits' simulations,
        # which are never needed at the same time: the two largest buffers share one.
        element = torch.finfo(dtype).bits // 8
        shared_bytes = capacity * max(entries_per_chunk * element, refine_pixels * 4)
        # a whole number of float32s, so that it can be viewed as either
        self._shared = empty(4 * math.ceil(shared_bytes / 4), dtype=torch.uint8)
        self.ghost = empty(refine_pixels)
        self.coarse_ghost = empty(coarse_pixels)
        #: Each row's mean and length while it is normalized, and a pattern's sums.
        self.scratch = empty(max(2 * capacity, refine_pixels))
        self.chunk_score = empty(capacity, dtype=dtype)
        self.chunk_entry = empty(capacity, dtype=torch.int64)
        self.better = empty(capacity, dtype=torch.bool)
        self.score = empty(capacity)
        self.entry = empty(capacity, dtype=torch.int64)
        self.phase = empty(capacity, dtype=torch.int64)
        #: Each pattern's first channel in the master texture: twice its phase.
        self.bases = empty(capacity, dtype=torch.int32)
        #: Orientations, passed from step to step: a kernel never writes the
        #: buffer it reads.
        self.seed = empty(capacity, 3, 3)
        self.coarse_refined = empty(capacity, 3, 3)
        self.refined = empty(capacity, 3, 3)
        self.fitted = empty(capacity, 3, 3)
        self.refused = empty(capacity, dtype=torch.int32)

    def scores(self, patterns: int, entries: int, dtype: torch.dtype) -> torch.Tensor | None:
        """A contiguous ``(patterns, entries)`` view for the search's scores, or None
        where they do not fit."""
        count = patterns * entries
        if count * torch.finfo(dtype).bits // 8 > self._shared.numel():
            return None
        return self._shared.view(dtype)[:count].view(patterns, entries)

    def simulations(self, patterns: int, pixels: int) -> torch.Tensor | None:
        """A contiguous float32 ``(patterns, pixels)`` view for projected patterns,
        or None where they do not fit."""
        return self.scores(patterns, pixels, torch.float32)

    @property
    def nbytes(self) -> int:
        """The device memory the workspace holds."""
        return sum(
            value.numel() * value.element_size()
            for value in vars(self).values()
            if isinstance(value, torch.Tensor)
        )
