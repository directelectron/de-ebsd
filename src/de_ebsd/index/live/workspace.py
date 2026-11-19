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


#: The most traces a candidate search's near-duplicate test holds at once: 16 MB.
TRACE_ELEMENTS = 4 * 1024 * 1024


class CandidateBuffers:
    """The buffers of a candidate search and its recheck
    (:meth:`~de_ebsd.index.live.LiveIndexer.search_candidates`), for up to
    *capacity* patterns, each keeping a *pool* of the search's best entries and
    *candidates* of them.

    The fused path holds one, built with the indexer, so indexing with candidates
    allocates nothing either; the allocating path builds one per batch, so both
    run the same arithmetic.

    Parameters
    ----------
    capacity
        The most patterns a batch holds.
    blocks
        The most blocks of entries one chunk of the search scores (see
        :data:`~de_ebsd.index.live.indexer.CANDIDATE_BLOCK`).
    pool, candidates
        Entries kept per pattern from the search, and candidates kept of those.
    operators, phases
        The rotations of every phase's point group, padded to the most any has,
        and the phases.
    refine_pixels, coarse_pixels
        Pixels of a pattern at the refine and coarse shapes.
    dtype
        The search dictionary's.
    device
        Where the buffers live.
    """

    def __init__(
        self,
        capacity: int,
        *,
        blocks: int,
        pool: int,
        candidates: int,
        operators: int,
        phases: int,
        refine_pixels: int,
        coarse_pixels: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        def empty(*shape, dtype=torch.float32):
            return torch.empty(shape, device=device, dtype=dtype)

        n, p, k = capacity, pool, candidates
        self.capacity, self.pool, self.candidates = capacity, pool, candidates
        # the search: each block's best entry, a slice of blocks' best, the pool
        self.block_score = empty(n, blocks, dtype=dtype)
        self.block_entry = empty(n, blocks, dtype=torch.int64)
        self.slice_score = empty(n, p, dtype=dtype)
        self.slice_position = empty(n, p, dtype=torch.int64)
        self.merged_score = empty(n, 2 * p)
        self.merged_entry = empty(n, 2 * p, dtype=torch.int64)
        self.order = empty(n, p, dtype=torch.int64)
        self.pool_score = empty(n, p)
        self.pool_entry = empty(n, p, dtype=torch.int64)
        # the pool's near-duplicates
        self.pool_rotations = empty(n * p, 3, 3)
        self.pool_phase = empty(n * p, dtype=torch.int64)
        self.pairs = empty(n, 3 * p, 3 * p)
        self.turns = empty(n, p, p, 9)
        # the traces of a slice of the pairs with every operator, a few MB at a time
        pairs = max(p * p, TRACE_ELEMENTS // max(1, operators))
        self.traces = empty(min(pairs, n * p * p), operators)
        self.per_phase = empty(min(pairs, n * p * p), phases)
        self.pair_phase = empty(n * p * p if phases > 1 else 1, dtype=torch.int64)
        self.closest = empty(n, p, p)
        self.near = empty(n, p, p, dtype=torch.bool)
        self.same = empty(n, p, p, dtype=torch.bool)
        self.suppressed = empty(n, p, dtype=torch.bool)
        self.invalid = empty(n, p, dtype=torch.bool)
        self.slice_entry = empty(n, p, dtype=torch.int64)
        self.key = empty(n, p)
        self.key_top = empty(n, k)
        self.pick = empty(n, k, dtype=torch.int64)
        # the candidates, best first
        self.entry = empty(n, k, dtype=torch.int64)
        self.score = empty(n, k)
        self.distinct = empty(n, k, dtype=torch.bool)
        self.phase = empty(n, k, dtype=torch.int64)
        self.rotations = empty(n, k, 3, 3)
        # the recheck of the patterns it is worth it for
        self.margin = empty(n)
        self.gate = empty(n, dtype=torch.bool)
        self.low = empty(n, dtype=torch.bool)
        self.index = empty(n, dtype=torch.int64)
        self.rows = empty(n, refine_pixels)
        self.coarse_rows = empty(n, coarse_pixels)
        self.best = empty(n, 3, 3)
        self.best_score = empty(n)
        self.best_match = empty(n)
        self.best_phase = empty(n, dtype=torch.int64)
        self.best_entry = empty(n, dtype=torch.int64)
        # one of each pattern's other candidates a pair, every pair refined at once
        pairs = n * max(1, k - 1)
        self.pair_candidate = empty(pairs, dtype=torch.int64)
        self.pair_pattern = empty(pairs, dtype=torch.int64)
        self.seed = empty(pairs, 3, 3)
        self.seed_phase = empty(pairs, dtype=torch.int64)
        self.seed_entry = empty(pairs, dtype=torch.int64)
        self.seed_bases = empty(pairs, dtype=torch.int32)
        self.seed_distinct = empty(pairs, dtype=torch.bool)
        self.coarse_refined = empty(pairs, 3, 3)
        self.refined = empty(pairs, 3, 3)
        self.refined_score = empty(pairs)
        self.match = empty(pairs)
        # each pattern's best pair
        self.top_match = empty(n)
        self.top = empty(n, dtype=torch.int64)
        self.pair_offset = empty(n, dtype=torch.int64)
        self.better = empty(n, dtype=torch.bool)
        self.winner = empty(n, 3, 3)
        self.winner_score = empty(n)
        self.winner_number = empty(n, dtype=torch.int64)

    def winner_value(self, values: torch.Tensor, patterns: int) -> torch.Tensor:
        """The buffer for each pattern's best pair's *values* (a score, a phase or
        an entry), for the first *patterns*."""
        if values.dtype == torch.int64:
            return self.winner_number[:patterns]
        return self.winner_score[:patterns]

    @property
    def nbytes(self) -> int:
        """The device memory the buffers hold."""
        return sum(
            value.numel() * value.element_size()
            for value in vars(self).values()
            if isinstance(value, torch.Tensor)
        )


class InplaneWorkspace:
    """The working buffers of the in-plane search's fused path
    (:class:`~de_ebsd.index.live.inplane.InplaneSearch`), for up to *capacity*
    patterns, allocated once; each call takes views of the size it needs.

    Two pairs of buffers share memory, each pair never needed at once: the patterns
    averaged to the source shape (read once, into the polar grid) with their
    spectra, and the patterns on the polar grid (read once, into their spectra)
    with one chunk's per-frequency products and scores.

    Parameters
    ----------
    capacity
        The most patterns a call holds.
    source_pixels
        Pixels of a pattern at the shape it is resampled from.
    polar_shape
        The polar grid's ``(K, J)``.
    frequencies, thetas
        Angular frequencies kept and in-plane angles searched.
    chunk
        The most templates scored at once.
    dtype
        The search's operands'.
    device
        Where the buffers live.
    """

    def __init__(
        self,
        capacity: int,
        *,
        source_pixels: int,
        polar_shape: tuple[int, int],
        frequencies: int,
        thetas: int,
        chunk: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        sizes = self.sizes(
            capacity,
            source_pixels=source_pixels,
            polar_shape=polar_shape,
            frequencies=frequencies,
            thetas=thetas,
            chunk=chunk,
            dtype=dtype,
        )
        self.capacity = capacity
        self.dtype = dtype
        self.polar_shape = tuple(polar_shape)
        self.frequencies = frequencies
        self.thetas = thetas
        self.chunk = chunk
        self._operands = torch.empty(sizes["operands"], dtype=torch.uint8, device=device)
        self._region = torch.empty(sizes["region"], dtype=torch.uint8, device=device)
        #: Each pattern's running best, a packed score and entry.
        self.best = torch.empty(capacity, dtype=torch.int64, device=device)
        #: Each pattern's center, where they are its own.
        self.centers = torch.empty(capacity, 3, dtype=torch.float32, device=device)

    @staticmethod
    def sizes(
        capacity: int,
        *,
        source_pixels: int,
        polar_shape: tuple[int, int],
        frequencies: int,
        thetas: int,
        chunk: int,
        dtype: torch.dtype,
    ) -> dict:
        """Bytes: ``operands`` (the spectra, or the source images), ``polar``,
        ``per_template`` (one template's products and scores), ``region`` (the polar
        patterns or a chunk's products and scores), ``fixed`` (all but the region)
        and ``total``."""
        element = torch.finfo(dtype).bits // 8
        k, j = polar_shape
        operands = max(2 * frequencies * 2 * k * capacity * element, capacity * source_pixels * 4)
        polar = k * j * capacity * element
        per_template = capacity * (2 * frequencies + thetas) * element
        region = max(polar, chunk * per_template)
        fixed = operands + capacity * (8 + 12)
        return dict(
            operands=operands,
            polar=polar,
            per_template=per_template,
            region=region,
            fixed=fixed,
            total=fixed + region,
        )

    def _view(self, storage, offset: int, dtype: torch.dtype, *shape) -> torch.Tensor:
        size = math.prod(shape) * (torch.finfo(dtype).bits // 8)
        return storage[offset : offset + size].view(dtype).view(*shape)

    def source(self, patterns: int, rows: int, cols: int) -> torch.Tensor:
        """``(patterns, rows, cols)`` float32 for the patterns at the source shape."""
        return self._view(self._operands, 0, torch.float32, patterns, rows, cols)

    def operands(self, patterns: int) -> torch.Tensor:
        """``(2, F, 2K, patterns)``: the patterns' spectra."""
        k = self.polar_shape[0]
        return self._view(self._operands, 0, self.dtype, 2, self.frequencies, 2 * k, patterns)

    def polar(self, patterns: int) -> torch.Tensor:
        """``(K, patterns, J)``: the patterns on the polar grid."""
        k, j = self.polar_shape
        return self._view(self._region, 0, self.dtype, k, patterns, j)

    def products(self, templates: int, patterns: int) -> torch.Tensor:
        """``(2, F, templates, patterns)``: one chunk's per-frequency products."""
        return self._view(self._region, 0, self.dtype, 2, self.frequencies, templates, patterns)

    def scores(self, templates: int, patterns: int) -> torch.Tensor:
        """``(T, templates * patterns)``: one chunk's scores at every in-plane angle."""
        element = torch.finfo(self.dtype).bits // 8
        offset = 2 * self.frequencies * templates * patterns * element
        return self._view(self._region, offset, self.dtype, self.thetas, templates * patterns)

    @property
    def nbytes(self) -> int:
        """The device memory the buffers hold."""
        return sum(
            value.numel() * value.element_size()
            for value in vars(self).values()
            if isinstance(value, torch.Tensor)
        )
