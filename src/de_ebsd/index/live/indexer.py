"""Live indexing: a buffer of camera frames indexed and refined where it already is.

Built for a camera streaming thousands of patterns a second into GPU memory. A
buffer of frames is indexed in a fixed number of batched steps -- nothing loops
over patterns or phases -- and only each pattern's phase, orientation and score
leave the device.

1. **Prepare** (:meth:`LiveIndexer.prepare`): each frame averaged over the whole
   sensor down to the refine shape, divided by the scan's static background (the
   running mean of its frames, :meth:`LiveIndexer.observe`, started from a search's
   and blurred for a single crystal, :meth:`LiveIndexer.restart_static`), the
   per-pattern dynamic background removed, then averaged again to the search shape.
2. **Search** (:meth:`LiveIndexer.search`): one matrix product against every
   phase's dictionary at once. The dictionaries are the phases' dynamical master
   patterns projected over their fundamental zones, stacked into one matrix, so the
   best entry gives the phase and a seed orientation together; a low symmetry's
   large dictionary is scored a chunk of entries at a time.
3. **Refine** (:meth:`LiveIndexer.refine`): a local dictionary around each seed --
   the seed and the six rotations of one step about each axis -- projected from the
   master on the fly at the refine shape. A parabola through each axis's three
   scores moves the orientation; the step shrinks each level. Every level is the
   same seven projections per pattern, so its cost is fixed and known.
   With ``search="inplane"`` the search is in-plane-factored instead
   (:mod:`de_ebsd.index.live.inplane`): one polar template per crystal direction
   along the screen normal, scored at every in-plane angle at once by a
   correlation about the pattern center, a masked normalized cross-correlation
   over the screen's visible part.
4. **Pattern center** (:meth:`LiveIndexer.refine_pattern_center`): the center
   shared by a buffer's best patterns, refined the same way -- seven centers a
   level, each scored with every pattern's orientation refined again there.

Orientations are rotation matrices ``g`` taking a sample-frame direction to the
crystal frame (``v_crystal = g @ v_sample``), as :meth:`MasterPattern.project`
uses them.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from de_ebsd.core.constants import SINGLE_PRECISION_FLOOR
from de_ebsd.core.device import holds_lock, resolve_device, resolve_lock
from de_ebsd.core.lambert import sample_texture
from de_ebsd.core.rotation import quaternion_to_matrix, rotation_vector_to_matrix
from de_ebsd.index import _kernels
from de_ebsd.index.live import calibration, deformation
from de_ebsd.index.live.background import ResidualBackground, ScanBackground
from de_ebsd.index.live.common import (
    BATCH,
    FUSED_BATCH,
    STENCIL,
    as_frames,
    average_to,
    normalize_rows_into,
    parabola_offset,
    pool_into,
    shape_in_aspect,
)
from de_ebsd.index.live.workspace import CandidateBuffers, Workspace
from de_ebsd.index.preprocess import _gaussian_kernel_1d, gaussian_blur, normalize_rows

__all__ = [
    "FITTED_REFINE_STEPS_DEG",
    "LiveIndexer",
    "dynamical_indexing_available",
    "fitted_refine_steps_deg",
    "fused_fit_available",
    "lambert_masters",
]


#: Refine steps, in degrees, each half the last. On a twin scan rendered from a
#: dynamical master (1024 patterns, 33 grains, 3° dictionary, 128² refine) these
#: gave a median of 0.14° from the truth, against 0.23° for (1.5, 0.5, 0.17) and
#: 0.19° with a fourth level of 0.06.
REFINE_STEPS_DEG = (2.0, 1.0, 0.5, 0.25, 0.12)

#: How much the search's copy of the masters is blurred (Gaussian sigma, degrees on
#: the sphere). A sharp dynamical pattern's score falls off so steeply away from
#: its orientation that at a 3° dictionary no entry is near enough for some
#: grains and a distant wrong one wins: 6-9% of a twin scan's patterns landed
#: 40-60° off unblurred, none at 2°.
SEARCH_BLUR_DEG = 2.0

#: Refine steps of this size and up are scored at half the refine's resolution on
#: the fused path: a step of a degree or two moves the bands by several of its
#: pixels, which a quarter of them resolve as well. On a customer Zenith's Cu
#: scan this took the refine from 3.9 to 2.5 ms per 256 patterns with the
#: orientations the fit then reached unchanged.
COARSE_REFINE_STEP_DEG = 1.0


#: Dictionary entries scored at a time by :meth:`LiveIndexer.search`, which bounds
#: its scores at (patterns, this): 64 MB for a batch. A cubic phase's 3° dictionary
#: (30,443 entries) is one chunk; a triclinic one's (729,000) would otherwise
#: score into 750 MB per batch.
SEARCH_CHUNK = 65_536

#: The refine's steps when :meth:`LiveIndexer.index` fits the orientations after it
#: (``fit=True``) on the fused kernels: the fit does what the finer levels did. On a
#: real Zenith Cu scan, stopping at 1 degree took indexing from 28,800 to 33,600
#: patterns a second and a single crystal's neighbours from 0.19 to 0.17 degrees
#: apart at the 90th percentile; no orientation moved more than 5 degrees. See
#: :func:`fitted_refine_steps_deg`.
FITTED_REFINE_STEPS_DEG = (2.0, 1.0)


#: How many of the search's best entries a candidate search pools per pattern, before
#: it drops the near-duplicates among them and keeps the best ``candidates``.
CANDIDATE_POOL = 16

#: A candidate search keeps only the best of each block of this many consecutive
#: dictionary entries before it ranks them: a block is a few neighboring entries,
#: so this drops near-duplicates early, and it keeps every ranking narrow enough
#: to sort in place (:data:`CANDIDATE_SLICE`).
CANDIDATE_BLOCK = 32

#: The widest row of scores a candidate search ranks at once. CUDA ranks a row of
#: up to several hundred in place, and a wider one through a temporary buffer,
#: which the workspace would then have to allocate on every batch.
CANDIDATE_SLICE = 512

#: How close two of a pattern's candidates of one phase may be, in dictionary
#: steps and symmetry reduced, before the worse is dropped as the same basin.
CANDIDATE_SEPARATION = 1.5

#: A pattern's candidates are refined beyond the best only where the search's best
#: and the best distinct runner-up score within this of each other ... On a
#: simulated Ni scan with 16 counts a pixel and four candidates, 0.02 rechecked 8 %
#: of the patterns and left 0.32 % more than 5 degrees off, against 0.29 % for
#: rechecking every pattern (1.39 % with one candidate); 0.01 rechecked 4 % and
#: left 0.47 %.
RECHECK_MARGIN = 0.02

#: ... or where the best's refined score is below this. A refined score depends on
#: the patterns' quality, so by default only a pattern that matches nothing is
#: rechecked for its score.
RECHECK_SCORE = 0.0


def fused_fit_available(device) -> bool:
    """Whether :meth:`LiveIndexer.index` fits orientations (``fit=True``) on the fused
    kernels on *device*: CUDA with CuPy, and a scheduled analytic fit."""
    return (
        _kernels.available(device)
        and bool(deformation.FIT_SCHEDULE)
        and bool(deformation.ANALYTIC_JACOBIAN)
    )


def fitted_refine_steps_deg(device) -> tuple[float, ...]:
    """The ``refine_steps_deg`` to build a :class:`LiveIndexer` with when it will index
    with ``fit=True`` on *device*: :data:`FITTED_REFINE_STEPS_DEG` where the fit is
    fused (:func:`fused_fit_available`), and a level of 0.5 degrees more where it is
    not -- the torch fit, with no pass schedule, reached 0.53 degrees from 1 degree
    on a real scan where it reaches 0.21 from 0.5."""
    if fused_fit_available(device):
        return FITTED_REFINE_STEPS_DEG
    return (*FITTED_REFINE_STEPS_DEG, 0.5)


def dynamical_indexing_available(device) -> bool:
    """Whether indexing against dynamical master patterns keeps up with a camera on
    *device*: a GPU. On a CPU a master takes minutes to simulate and the refine runs
    far slower than a camera, so a kinematic master
    (:class:`~de_ebsd.sim.KinematicMaster`, which :class:`LiveIndexer` takes too)
    is the one to index against there."""
    return resolve_device(device).type == "cuda"


def lambert_masters(masters) -> list:
    """*masters* as Lambert-stored :class:`~de_ebsd.sim.MasterPattern` s: each
    :class:`~de_ebsd.sim.KinematicMaster` sampled onto the grid the others are on
    (:meth:`~de_ebsd.sim.KinematicMaster.to_master_pattern`), or onto its default
    grid when every master is kinematic."""
    from de_ebsd.sim.kinematic_bands import LAMBERT_NPX

    stored = [m.npx for m in masters if hasattr(m, "npx")]
    npx = stored[0] if stored else LAMBERT_NPX
    return [m if hasattr(m, "npx") else m.to_master_pattern(npx) for m in masters]


def texture_of(masters, device, blur_deg: float = 0.0) -> torch.Tensor:
    """Every phase's hemispheres as channels of one texture ``(1, 2 * phases, n, n)``:
    one sampling pass serves all phases, and each pattern keeps its own phase's."""
    if len({m.npx for m in masters}) != 1:
        raise ValueError("every master must have the same npx")
    texture = torch.cat([m.data[m.energy_index(None)].to(device, torch.float32) for m in masters])
    # Near the center of the square Lambert grid a pixel spans 90° / npx.
    per_degree = masters[0].npx / 90.0
    if blur_deg:
        texture = gaussian_blur(texture, blur_deg * per_degree, 4.0)
    return texture[None].contiguous()


class LiveIndexer:
    """Index and refine EBSD patterns of several phases on the device.

    Parameters
    ----------
    masters
        One master per phase: a :class:`~de_ebsd.sim.MasterPattern` (all at the
        same ``npx``), used at its highest energy bin, or a
        :class:`~de_ebsd.sim.KinematicMaster`, sampled onto the same Lambert grid
        (:func:`lambert_masters`). Either physics, or a mix: every phase is
        searched in one product, and the best score gives the phase. Collapse an
        energy-resolved master first (:meth:`~de_ebsd.sim.MasterPattern.collapse`), or
        build it from a fitted appearance
        (:meth:`~de_ebsd.index.appearance.DetectorAppearance.master`).
    point_groups
        Each phase's orix point group, whose fundamental zone its dictionary samples.
    detector
        The camera's :class:`~de_ebsd.sim.detector.EBSDDetector` (its pattern center
        and angles); frames may come at any resolution, since they are averaged to
        ``search_shape`` and ``refine_shape``.
    resolution
        The dictionary's sampling, in degrees: for the in-plane search, the
        equivalent misorientation step its sphere and in-plane steps follow from.
    search_shape, refine_shape
        Detector shapes the search and the refine compare patterns at, as
        ``(rows, cols)`` or as one number: that many pixels on the long side of
        *detector*'s shape, in its aspect
        (:func:`~de_ebsd.index.live.common.shape_in_aspect`).
    refine_steps_deg
        The refine levels' steps; empty for no refine.
    search_blur_deg
        How much the search's copy of the masters is blurred: by default
        :data:`SEARCH_BLUR_DEG` for the flat search and half of *resolution* for
        the in-plane one, which it was validated at.
    candidates
        How many of the search's distinct best entries each pattern may be refined
        from, the refined orientation that matches its pattern best winning; see
        :meth:`set_candidates`. One, the default, refines the search's best alone.
    recheck_margin, recheck_score
        With several *candidates*, only the patterns whose search found a distinct
        runner-up within *recheck_margin* of its best, or whose best refined to a
        score below *recheck_score*, have the rest refined; ``math.inf`` for both
        refines every pattern's. See :data:`RECHECK_MARGIN`.
    candidate_separation_deg
        Candidates of one phase closer than this, symmetry reduced, are one
        basin, and only the better is kept; by default
        :data:`CANDIDATE_SEPARATION` dictionary steps.
    device
        Defaults to the first master's.
    dtype
        The search dictionary's. ``None`` picks float16 on GPUs with tensor cores
        (compute capability 7.0 and up) and float32 elsewhere: Pascal multiplies
        float16 at 1/64 of its float32 rate, and a CPU has no float16 product. Half
        precision halves the dictionary, which is what grows with low symmetry; the
        search only picks seeds, so the refine sets the accuracy either way.
    lock
        Held around every call that runs device work, building included, for an
        application that shares the device between threads; see
        :func:`~de_ebsd.core.device.resolve_lock`.
    residual_background
        A :class:`~de_ebsd.index.live.background.ResidualBackground` the corrected
        patterns still carry, from an appearance fit
        (:meth:`~de_ebsd.index.appearance.DetectorAppearance.residual_background`),
        removed from every prepared pattern; see :meth:`set_residual_background`.
    search
        ``"flat"``, a dictionary of every orientation at *resolution* over each
        phase's fundamental zone, or ``"inplane"``, the in-plane-factored search
        (:class:`~de_ebsd.index.live.inplane.InplaneSearch`): more accurate at the
        same step, faster at fine steps, and a fraction of the memory.
    search_options
        Keyword arguments for :class:`~de_ebsd.index.live.inplane.InplaneSearch`,
        such as its working-memory budget (``memory``, bytes).
    """

    @holds_lock
    def __init__(
        self,
        masters,
        point_groups,
        detector,
        *,
        resolution: float = 3.0,
        search_shape: tuple[int, int] = (32, 32),
        refine_shape: tuple[int, int] = (128, 128),
        refine_steps_deg=REFINE_STEPS_DEG,
        search_blur_deg: float | None = None,
        candidates: int = 1,
        recheck_margin: float = RECHECK_MARGIN,
        recheck_score: float = RECHECK_SCORE,
        candidate_separation_deg: float | None = None,
        device=None,
        dtype=None,
        lock=None,
        residual_background: ResidualBackground | None = None,
        search: str = "flat",
        search_options: dict | None = None,
    ) -> None:
        self.lock = resolve_lock(lock)
        from orix.sampling import get_sample_fundamental

        if search not in ("flat", "inplane"):
            raise ValueError(f'search is "flat" or "inplane", not {search!r}')
        if search_options and search != "inplane":
            raise ValueError('search_options are for search="inplane"')
        if search_blur_deg is None:
            search_blur_deg = SEARCH_BLUR_DEG if search == "flat" else 0.5 * resolution
        if len(masters) != len(point_groups):
            raise ValueError("one point group per master")
        if not len(masters):
            raise ValueError("indexing needs at least one phase")
        masters = lambert_masters(masters)
        self.device = resolve_device(device, masters[0].device)
        if dtype is None:
            tensor_cores = (
                self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 7
            )
            dtype = torch.float16 if tensor_cores else torch.float32
        self.dtype = dtype
        self.dictionary = None
        self.detector = detector
        self.search_shape = shape_in_aspect(search_shape, detector.shape)
        self.refine_shape = shape_in_aspect(refine_shape, detector.shape)
        #: Where the refine's coarse levels (:data:`COARSE_REFINE_STEP_DEG` and up)
        #: are scored on the fused path: half the refine's resolution.
        self.coarse_shape = tuple(max(8, s // 2) for s in self.refine_shape)
        self.refine_steps = tuple(math.radians(s) for s in refine_steps_deg)
        self.fused = _kernels.available(self.device)
        self.texture = texture_of(masters, self.device)
        self.search_texture = texture_of(masters, self.device, search_blur_deg)
        self.stencil = torch.tensor(STENCIL, device=self.device, dtype=torch.float32)

        #: The in-plane search (:class:`~de_ebsd.index.live.inplane.InplaneSearch`),
        #: or None for the flat one.
        self.inplane = None
        #: The flat dictionary's orientations and their phases; None in-plane.
        self.rotations = self.phase_of = None
        if search == "inplane":
            from de_ebsd.index.live.inplane import InplaneSearch

            self.inplane = InplaneSearch(
                self,
                point_groups,
                step=resolution,
                capacity=FUSED_BATCH,
                **(search_options or {}),
            )
            self.sizes = list(self.inplane.sizes)
        else:
            rotations, phases = [], []
            for number, point_group in enumerate(point_groups):
                q = np.asarray(
                    get_sample_fundamental(resolution=resolution, point_group=point_group).data
                )
                rotations.append(
                    quaternion_to_matrix(
                        torch.as_tensor(q, device=self.device, dtype=torch.float32)
                    )
                )
                phases.append(torch.full((len(q),), number, device=self.device, dtype=torch.int64))
            self.rotations = torch.cat(rotations)
            self.phase_of = torch.cat(phases)
            self.sizes = [len(r) for r in rotations]
        self.point_groups = list(point_groups)
        #: Each phase's proper rotations ``(phases, most, 3, 3)``, as the
        #: dictionary's matrices are made, padded with the identity: a candidate
        #: search's symmetry-reduced distances.
        self.operators = _operator_table(self.point_groups, self.device)
        #: :attr:`operators` as ``(9, phases * most)``: each column the nine numbers
        #: whose product with a turn's nine gives ``trace(S turn)``.
        self.trace_weights = self.operators.transpose(-1, -2).reshape(-1, 9).T.contiguous()
        #: Where each phase's entries start: in :attr:`rotations`, or in the in-plane
        #: entries ``u * thetas + m``.
        self.offsets = np.cumsum([0, *self.sizes[:-1]])
        self.set_pattern_center(detector.pattern_center)
        self.background = ScanBackground(self.refine_shape, self.device)
        self.set_residual_background(residual_background)
        #: The dynamic background's Gaussian for the fused :meth:`prepare`.
        self._prepare_weights, _ = _gaussian_kernel_1d(
            (self.refine_shape[1] // 4) / 8.0, 4.0, self.device, torch.float32
        )
        #: The fused path's buffers (:class:`~de_ebsd.index.live.workspace.Workspace`),
        #: for :data:`FUSED_BATCH` patterns; None off it.
        self.workspace = None
        if self.fused:
            self.workspace = Workspace(
                FUSED_BATCH,
                search_pixels=len(self.search_directions),
                refine_pixels=len(self.refine_directions),
                coarse_pixels=len(self.coarse_directions),
                entries_per_chunk=(0 if self.inplane else min(len(self.rotations), SEARCH_CHUNK)),
                dtype=self.dtype,
                device=self.device,
            )
        self.resolution = float(resolution)
        self.recheck_margin = float(recheck_margin)
        self.recheck_score = float(recheck_score)
        separation = (
            CANDIDATE_SEPARATION * self.resolution
            if candidate_separation_deg is None
            else candidate_separation_deg
        )
        #: Two candidates are one basin where ``trace(S g1 g2^T)`` is above this.
        self.near_trace = 1.0 + 2.0 * math.cos(math.radians(separation))
        #: Patterns whose candidates were refined beyond the best, over the
        #: indexer's life (see :meth:`set_candidates`).
        self.rechecked = 0
        self.candidate_buffers = None
        self.set_candidates(candidates)

    @holds_lock
    def set_candidates(self, count: int) -> None:
        """Refine each pattern from up to *count* of the search's distinct best
        entries from now on, and keep the refined orientation that matches its
        pattern best (the normalized cross-correlation at the refine shape against
        the unblurred master). One refines the search's best alone, as before.

        The search pools its :data:`CANDIDATE_POOL` best entries (the best of each
        block of :data:`CANDIDATE_BLOCK`), drops each that is within the
        candidate separation of a better one of its phase, and keeps the best
        *count*. A seed in the wrong basin -- a look-alike, a pseudo-symmetric
        partner -- is refined to the wrong orientation, which a better-matching
        distinct candidate then replaces; the cost is spent only on the patterns
        that :attr:`recheck_margin` and :attr:`recheck_score` pick. On the fused
        path the buffers are made here, so indexing still allocates nothing.

        The in-plane search (``search="inplane"``) pools each template's best
        in-plane angle the same way, its templates standing for the blocks: a
        pattern on a grain boundary, a blend of both grains', then has both grains
        among its candidates, and keeps the one whose refined orientation matches
        it best rather than a third that matched the blend's search best. On the
        in-plane path the candidate search allocates its scores per chunk of
        templates."""
        count = int(count)
        if count < 1:
            raise ValueError("candidates must be at least 1")
        self.candidates = count
        self.candidate_buffers = None
        if self.workspace is not None and count > 1:
            self.candidate_buffers = self._candidate_buffers(self.workspace.capacity)

    def _candidate_buffers(self, capacity: int) -> CandidateBuffers:
        if self.inplane is not None:
            blocks = 1
        else:
            blocks = max(1, min(len(self.rotations), SEARCH_CHUNK) // CANDIDATE_BLOCK)
        return CandidateBuffers(
            capacity,
            blocks=blocks,
            pool=max(CANDIDATE_POOL, 2 * self.candidates),
            candidates=self.candidates,
            operators=self.trace_weights.shape[1],
            phases=self.operators.shape[0],
            refine_pixels=len(self.refine_directions),
            coarse_pixels=len(self.coarse_directions),
            dtype=self.dtype,
            device=self.device,
        )

    @holds_lock
    def set_residual_background(self, residual: ResidualBackground | None) -> None:
        """Remove *residual* (or nothing, for None) from every pattern :meth:`prepare`
        corrects from now on.

        The refine rows lose ``share`` times the background's unit direction and are
        normalized again; the search rows are then those rows averaged to the search
        shape, as :meth:`prepare` makes them without the fused kernel. About one more
        pass over each pattern's refine rows, next to the refine's seven projections a
        level."""
        self.residual = residual
        self._residual_rows = None
        if residual is not None and residual.share:
            direction = residual.direction(self.refine_shape, self.device)
            self._residual_rows = direction.mul_(float(residual.share))

    def _remove_residual(self, refine_rows, out=None, scratch=None):
        """*refine_rows* ``(N, pixels)`` less the residual background, normalized again,
        and the search rows made from them: ``(search, refine)``. In place, into the
        search buffer *out*, where *scratch* is given (the workspace's)."""
        n = len(refine_rows)
        if scratch is None:
            refine_rows = normalize_rows(refine_rows - self._residual_rows)
            grid = refine_rows.view(n, *self.refine_shape)
            search = normalize_rows(average_to(grid, self.search_shape).flatten(1))
            return search, refine_rows
        refine_rows = normalize_rows_into(refine_rows.sub_(self._residual_rows), scratch)
        pool_into(refine_rows.view(n, *self.refine_shape), out.view(n, *self.search_shape))
        return normalize_rows_into(out, scratch), refine_rows

    @holds_lock
    def directions_at(self, pattern_center, shape) -> torch.Tensor:
        detector = replace(
            self.detector, pattern_center=tuple(float(v) for v in pattern_center), shape=shape
        )
        return detector.direction_cosines(device=self.device)

    @holds_lock
    def set_pattern_center(self, pattern_center) -> None:
        """Index at pattern center *pattern_center* from now on: the dictionary is projected
        again, which takes milliseconds with the fused kernels.

        Each chunk of entries is written straight into the dictionary, and a new
        center overwrites the old one's in place: the dictionary is the largest
        thing on the device, and building it beside a copy of itself (or a list of
        its chunks) took two to three times its size.

        The in-plane search's templates do not depend on the center; its mask and
        the templates' visible norms are worked out again at the next search, and
        any scan centers (:meth:`set_scan_centers`) are forgotten."""
        self.detector = replace(
            self.detector, pattern_center=tuple(float(v) for v in pattern_center)
        )
        self.search_directions = self.directions_at(pattern_center, self.search_shape)
        self.refine_directions = self.directions_at(pattern_center, self.refine_shape)
        self.coarse_directions = self.directions_at(pattern_center, self.coarse_shape)
        if self.inplane is not None:
            self.inplane.set_scan_centers(None)
            return
        shape = (len(self.search_directions), len(self.rotations))
        if self.dictionary is None or tuple(self.dictionary.shape) != shape:
            #: ``(pixels, entries)``, so the search's product reads it at full rate.
            self.dictionary = torch.empty(shape, device=self.device, dtype=self.dtype)
        for lo in range(0, len(self.rotations), BATCH * 8):
            entries = normalize_rows(
                self.project(
                    self.search_texture,
                    self.rotations[lo : lo + BATCH * 8],
                    self.phase_of[lo : lo + BATCH * 8],
                    self.search_directions,
                )
            )
            self.dictionary[:, lo : lo + len(entries)] = entries.T

    @holds_lock
    def project(
        self,
        texture: torch.Tensor,
        rotations: torch.Tensor,
        phases: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """The patterns ``(M, pixels)`` of *rotations* ``(M, 3, 3)``, each from its
        own phase's master in *texture*, over *directions* ``(pixels, 3)``."""
        if self.fused:
            return _kernels.project(texture, rotations, phases * 2, directions)
        v = torch.einsum("mij,pj->mpi", rotations, directions)
        return sample_texture(texture, v, phases[:, None])

    @holds_lock
    def score_candidates(
        self, candidates: torch.Tensor, phases: torch.Tensor, rows: torch.Tensor, directions
    ) -> torch.Tensor:
        """Scores ``(N, K)`` of candidates ``(N, K, 3, 3)`` against their patterns'
        normalized *rows* ``(N, pixels)``, sampled at *directions*."""
        n, k = candidates.shape[:2]
        flat = candidates.reshape(-1, 3, 3)
        bases = phases.repeat_interleave(k) * 2
        if self.fused:
            return _kernels.score(self.texture, flat, bases, directions, rows, k).view(n, k)
        simulated = normalize_rows(self.project(self.texture, flat, bases // 2, directions))
        return torch.einsum("nkp,np->nk", simulated.view(n, k, -1), rows)

    # --- the phases' dictionaries ----------------------------------------------

    @holds_lock
    def phase_entries(self, phase, entry):
        """Each dictionary *entry* (an index into :attr:`rotations`, as :meth:`index`
        returns it) as an index into its own *phase*'s entries
        (:meth:`phase_rotations`), so that anything worked out once per phase entry,
        such as an IPF color, stands for the orientation. NumPy or torch, as given."""
        if isinstance(entry, torch.Tensor):
            offsets = torch.as_tensor(self.offsets, device=entry.device)
            return entry - offsets[phase.to(entry.device)]
        return np.asarray(entry) - self.offsets[np.asarray(phase)]

    @holds_lock
    def phase_rotations(self, number: int) -> torch.Tensor:
        """Phase *number*'s dictionary orientations ``(entries, 3, 3)``, in
        OIM's sample frame, on the device: in-plane, worked out on each call."""
        first = int(self.offsets[number])
        if self.inplane is not None:
            entries = torch.arange(first, first + self.sizes[number], device=self.device)
            return self.inplane.entry_rotations(entries)
        return self.rotations[first : first + self.sizes[number]]

    @holds_lock
    def entry_rotations(self, entry) -> torch.Tensor:
        """The orientation ``(N, 3, 3)`` of each dictionary *entry*, as :meth:`index`
        returns them: what anything worked out once per entry, such as a color,
        stands for."""
        if self.inplane is not None:
            return self.inplane.entry_rotations(entry)
        return self.rotations[torch.as_tensor(entry, device=self.device)]

    @holds_lock
    def set_scan_centers(self, pattern_centers) -> None:
        """The pattern centers a scan's patterns span, ``(M, 3)``, for the in-plane
        search's visibility mask: the samples on the screen at every one of them.
        Pass the four corners of the scan's center plane
        (:meth:`~de_ebsd.index.live.calibration.CenterPlane.at` at the scan's corner
        positions), which give the mask the whole plane does, before indexing the
        scan with :meth:`index_with_strain` and each point's center; ``None`` goes
        back to the indexer's center, as :meth:`set_pattern_center` does. Without
        it, :meth:`index_with_strain` makes the mask of the centers each call is
        given. The flat search ignores it."""
        if self.inplane is not None:
            self.inplane.set_scan_centers(pattern_centers)

    # --- the scan's backgrounds ------------------------------------------------

    @property
    def static(self):
        """The scan's static background, ``None`` until enough frames have been seen."""
        return self.background.static

    @property
    def ghost(self):
        """The scan's grains as the static background shows them, or ``None``; see
        :attr:`ScanBackground.ghost`."""
        return self.background.ghost

    @holds_lock
    def pooled(self, frames) -> torch.Tensor:
        """*frames* on the device as float ``(N, H, W)`` averaged to the refine shape:
        *frames* themselves where they already are, such as a stack from
        :func:`~de_ebsd.index.live.common.pool_into`."""
        t = as_frames(frames).to(self.device)
        t = t[None] if t.ndim == 2 else t
        return average_to(t.to(torch.float32), self.refine_shape)

    @holds_lock
    def restart_static(self, frames=None, *, blur: float | None = None) -> None:
        """Forget the scan, for a new one: its static background and its ghost.

        *frames* from all over the same field, such as a search's, start the new
        static background, so the scan's first patterns are divided by one that
        holds no grain's bands rather than by the mean of its first few frames.

        With *blur* (a Gaussian sigma, as a share of the pattern's width) the
        static background is the frames' mean blurred, for a single crystal: its
        frames all carry its bands, so their mean would divide them out. It then
        has no ghost."""
        self.background.restart(blur)
        if frames is not None:
            self.observe(frames)

    @holds_lock
    def observe(self, frames) -> None:
        """Add *frames* to the scan's static background. Frames already at the refine
        shape (:func:`~de_ebsd.index.live.common.pool_into`) are summed as they are."""
        t = as_frames(frames).to(self.device)
        t = t[None] if t.ndim == 2 else t
        workspace = self.workspace
        if (
            workspace is not None
            and t.dtype == torch.float32
            and len(t) <= workspace.capacity
            and tuple(t.shape[-2:]) != self.refine_shape
        ):
            # pooled into the workspace, which no index is using between calls
            t = pool_into(t, workspace.refine_rows[: len(t)].view(len(t), *self.refine_shape))
        self.background.observe(self.pooled(t))

    @holds_lock
    def add_fits(self, phase, g, center_maps=None) -> None:
        """Add the simulations of fitted orientations *g* ``(N, 3, 3)`` to the
        scan's :attr:`ghost`."""
        if self.background.blur:
            return
        if center_maps is not None:
            g = g @ center_maps
        self.background.add_fits(self.project(self.texture, g, phase, self.refine_directions))

    @holds_lock
    def corrected(self, frames) -> torch.Tensor:
        """*frames* ``(N, H, W)`` with the backgrounds removed as :meth:`prepare` removes
        them, at their own resolution: what the indexer matches, to be looked at."""
        t = as_frames(frames).to(self.device)
        t = t[None] if t.ndim == 2 else t
        return self.background.remove(t.to(torch.float32))

    @holds_lock
    def prepare(self, frames) -> tuple[torch.Tensor, torch.Tensor]:
        """*frames* ``(N, H, W)`` corrected and normalized, as ``(search, refine)``
        rows ``(N, pixels)`` at the two shapes."""
        fused = self._fused_prepare(frames)
        if fused is None:
            t = self.background.remove(self.pooled(frames))
            search = average_to(t, self.search_shape)
            fused = normalize_rows(search.flatten(1)), normalize_rows(t.flatten(1))
        if self._residual_rows is not None:
            return self._remove_residual(fused[1])
        return fused

    def _fused_prepare(self, frames, out=None):
        """:meth:`prepare` in one launch (:func:`~de_ebsd.index._kernels.prepare`)
        where it can be: float32 frames on the device whose shape the refine shape
        divides. None where it cannot, for :meth:`prepare` to do it. *out* is the
        ``(search, refine)`` buffers to write, where given."""
        if not self.fused or not isinstance(frames, torch.Tensor):
            return None
        if frames.device.type != "cuda" or frames.dtype != torch.float32:
            return None
        frames = frames[None] if frames.ndim == 2 else frames
        coarse = (self.refine_shape[0] // 4, self.refine_shape[1] // 4)
        if not _kernels.prepare_fits(
            tuple(frames.shape[-2:]), self.refine_shape, coarse, self.search_shape
        ):
            return None
        return _kernels.prepare(
            frames.contiguous(),
            self.background.static,
            self.refine_shape,
            coarse,
            self.search_shape,
            self._prepare_weights,
            SINGLE_PRECISION_FLOOR,
            out=out,
        )

    @holds_lock
    def search(self, search_rows: torch.Tensor, refine_rows=None, pattern_centers=None):
        """The best dictionary entry for each pattern: ``(phase, g, score, entry)``,
        *entry* indexing :attr:`rotations` (every phase's entries, in phase order).

        The dictionary is scored :data:`SEARCH_CHUNK` entries at a time, keeping
        each pattern's best so far, so the scores never outgrow one chunk.

        The in-plane search reads the *refine_rows* (or, without them, the search
        rows drawn up to the refine shape), resampled about each pattern's own
        center where *pattern_centers* ``(N, 3)`` are given; *entry* is then ``u *
        thetas + m``, template ``u`` at in-plane step ``m`` (:meth:`entry_rotations`).
        The flat search reads the search rows at the indexer's center."""
        if self.inplane is not None:
            if refine_rows is None:
                n = len(search_rows)
                images = search_rows.float().view(n, 1, *self.search_shape)
                images = torch.nn.functional.interpolate(images, size=self.refine_shape)
                refine_rows = images.view(n, -1)
            return self.inplane.search(refine_rows, pattern_centers)
        rows = search_rows.to(self.dtype)
        score = torch.full((len(rows),), -math.inf, device=self.device)
        entry = torch.zeros(len(rows), dtype=torch.int64, device=self.device)
        for lo in range(0, self.dictionary.shape[1], SEARCH_CHUNK):
            chunk_score, chunk_entry = (rows @ self.dictionary[:, lo : lo + SEARCH_CHUNK]).max(1)
            better = chunk_score.float() > score
            score = torch.where(better, chunk_score.float(), score)
            entry = torch.where(better, chunk_entry + lo, entry)
        return self.phase_of[entry], self.rotations[entry], score, entry

    @holds_lock
    def search_candidates(self, search_rows: torch.Tensor):
        """Each pattern's :attr:`candidates` best distinct dictionary entries, best
        first: ``(phase, g, score, entry, distinct)``, each ``(N, candidates, ...)``.
        The first is :meth:`search`'s best; *distinct* is False where a pattern has
        fewer distinct entries than that, its places then holding repeats. See
        :meth:`set_candidates`. The in-plane search reads the search rows drawn up
        to the refine shape, as :meth:`search` does without refine rows."""
        n = len(search_rows)
        buffers = self._candidate_buffers(n)
        if self.inplane is not None:
            images = search_rows.float().view(n, 1, *self.search_shape)
            images = torch.nn.functional.interpolate(images, size=self.refine_shape)
            self._search_inplane_candidates(images.view(n, -1), buffers, None)
        else:
            self._search_candidates(search_rows.to(self.dtype), buffers, None)
        columns = (buffers.phase, buffers.rotations, buffers.score, buffers.entry)
        return (*(column[:n].clone() for column in columns), buffers.distinct[:n].clone())

    def _search_candidates(self, rows, buffers: CandidateBuffers, workspace) -> None:
        """The candidate search into *buffers*, scoring each chunk of the dictionary
        into the *workspace*'s scores where there is one."""
        n = len(rows)
        buffers.pool_score[:n].fill_(-math.inf)
        buffers.pool_entry[:n].zero_()
        for lo in range(0, self.dictionary.shape[1], SEARCH_CHUNK):
            chunk = self.dictionary[:, lo : lo + SEARCH_CHUNK]
            width = chunk.shape[1]
            scores = None if workspace is None else workspace.scores(n, width, self.dtype)
            scores = rows @ chunk if scores is None else torch.matmul(rows, chunk, out=scores)
            whole, tail = divmod(width, CANDIDATE_BLOCK)
            if whole:
                block_score = buffers.block_score.view(-1)[: n * whole].view(n, whole)
                block_entry = buffers.block_entry.view(-1)[: n * whole].view(n, whole)
                blocks = scores[:, : whole * CANDIDATE_BLOCK].view(n, whole, CANDIDATE_BLOCK)
                torch.max(blocks, 2, out=(block_score, block_entry))
                block_entry.add_(self._block_starts(whole)).add_(lo)
                for first in range(0, whole, CANDIDATE_SLICE):
                    last = min(whole, first + CANDIDATE_SLICE)
                    self._pool_slice(
                        block_score[:, first:last], block_entry[:, first:last], buffers
                    )
            if tail:
                tail_score = buffers.slice_score.view(-1)[:n].view(n, 1)
                tail_entry = buffers.slice_position.view(-1)[:n].view(n, 1)
                torch.max(
                    scores[:, whole * CANDIDATE_BLOCK :],
                    1,
                    keepdim=True,
                    out=(tail_score, tail_entry),
                )
                tail_entry.add_(lo + whole * CANDIDATE_BLOCK)
                self._merge_into_pool(tail_score, tail_entry, buffers)
        torch.isneginf(buffers.pool_score[:n], out=buffers.invalid[:n])
        self._drop_near_duplicates(n, buffers)

    def _search_inplane_candidates(self, refine_rows, buffers: CandidateBuffers, centers):
        """The in-plane candidate search into *buffers*: per chunk of templates,
        each template's best in-plane angle, ranked into each pattern's pool like a
        flat block's best, then the near-duplicates dropped. The scores of one
        chunk, ``(thetas, chunk, N)``, are the most it holds at once; *centers*
        are :meth:`search`'s *pattern_centers*."""
        search = self.inplane
        search.prepare_mask()
        n = len(refine_rows)
        if centers is not None:
            centers = np.asarray(centers, dtype=np.float64)
        buffers.pool_score[:n].fill_(-math.inf)
        buffers.pool_entry[:n].zero_()
        t, f = search.thetas, search.frequencies
        for first in range(0, n, search.capacity):
            last = min(n, first + search.capacity)
            here = None if centers is None else centers[first:last]
            operands = search.operands(search._source(refine_rows[first:last]), here)
            view = _PoolView(buffers, first, last)
            for lo in range(0, search.template_count, search.chunk):
                hi = min(search.template_count, lo + search.chunk)
                spectra = search.spectra[:, lo:hi]
                products = torch.stack([spectra @ operands[0], spectra @ operands[1]])
                scores = (search._angle_weights_typed @ products.view(2 * f, -1)).float()
                scores = scores.view(t, hi - lo, last - first)
                scores.mul_(search.inverse_energy[:, lo:hi, None])
                value, angle = scores.max(0)
                value = value.to(buffers.slice_score.dtype)
                templates = torch.arange(lo, hi, device=self.device, dtype=torch.int64)
                entry = angle.add_(templates[:, None] * t)
                for start in range(0, hi - lo, CANDIDATE_SLICE):
                    stop = min(hi - lo, start + CANDIDATE_SLICE)
                    self._pool_slice(
                        value[start:stop].T.contiguous(), entry[start:stop].T.contiguous(), view
                    )
        torch.isneginf(buffers.pool_score[:n], out=buffers.invalid[:n])
        self._drop_near_duplicates(n, buffers)

    def _entry_rotations_into(self, entries, out) -> torch.Tensor:
        """The orientations of dictionary *entries* (flat, as :meth:`search` numbers
        them) into *out*."""
        if self.inplane is not None:
            return out.copy_(self.inplane.entry_rotations(entries))
        return torch.index_select(self.rotations, 0, entries, out=out)

    def _entry_phases_into(self, entries, out) -> torch.Tensor:
        """The phases of dictionary *entries* into *out*."""
        if self.inplane is not None:
            phases = self.inplane.template_phase
            return torch.index_select(phases, 0, entries // self.inplane.thetas, out=out)
        return torch.index_select(self.phase_of, 0, entries, out=out)

    def _block_starts(self, whole: int) -> torch.Tensor:
        """The first entry of each of *whole* blocks, kept on the device."""
        starts = getattr(self, "_starts", None)
        if starts is None or len(starts) < whole:
            starts = torch.arange(whole, device=self.device, dtype=torch.int64) * CANDIDATE_BLOCK
            self._starts = starts
        return starts[:whole]

    def _pool_slice(self, score, entry, buffers: CandidateBuffers) -> None:
        """The best of one slice of blocks' bests (*score*, *entry* ``(N, width)``)
        merged into each pattern's pool."""
        n, width = score.shape
        keep = min(buffers.pool, width)
        best = buffers.slice_score.view(-1)[: n * keep].view(n, keep)
        position = buffers.slice_position.view(-1)[: n * keep].view(n, keep)
        torch.topk(score, keep, 1, out=(best, position))
        found = buffers.slice_entry.view(-1)[: n * keep].view(n, keep)
        torch.gather(entry, 1, position, out=found)
        self._merge_into_pool(best, found, buffers)

    @staticmethod
    def _merge_into_pool(score, entry, buffers: CandidateBuffers) -> None:
        """*score* and *entry* ``(N, width)`` merged into each pattern's pool, which
        keeps its best."""
        n, width = score.shape
        pool = buffers.pool
        merged_score, merged_entry = buffers.merged_score[:n], buffers.merged_entry[:n]
        merged_score[:, :pool].copy_(buffers.pool_score[:n])
        merged_entry[:, :pool].copy_(buffers.pool_entry[:n])
        merged_score[:, pool : pool + width].copy_(score)
        merged_entry[:, pool : pool + width].copy_(entry)
        merged_score[:, pool + width :].fill_(-math.inf)
        order = buffers.order[:n]
        torch.topk(merged_score, pool, 1, out=(buffers.pool_score[:n], order))
        torch.gather(merged_entry, 1, order, out=buffers.pool_entry[:n])

    def _drop_near_duplicates(self, n: int, buffers: CandidateBuffers) -> None:
        """Of each pattern's pool, the best :attr:`candidates` that are not within
        the candidate separation of a better one of their phase, into *buffers*."""
        pool, count = buffers.pool, buffers.candidates
        entries = buffers.pool_entry[:n].view(-1)
        g = self._entry_rotations_into(entries, buffers.pool_rotations[: n * pool])
        phase = self._entry_phases_into(entries, buffers.pool_phase[: n * pool])
        # every pair's turn g_i g_j^T in one product, (n, 3 pool, 3 pool), then laid
        # out as nine numbers a pair, whose product with the operators gives
        # trace(S g_i g_j^T) for every operator of every phase
        if self.operators.shape[0] > 1:
            # each pair (i, j) judged with i's phase's operators
            buffers.pair_phase[: n * pool * pool].view(n, pool, pool).copy_(
                phase.view(n, pool, 1).expand(n, pool, pool)
            )
        rows = g.view(n, pool * 3, 3)
        pairs = torch.bmm(rows, rows.transpose(1, 2), out=buffers.pairs[:n])
        turns = buffers.turns[:n]
        turns.view(n, pool, pool, 3, 3).copy_(pairs.view(n, pool, 3, pool, 3).transpose(2, 3))
        turns = turns.view(-1, 9)
        closest = buffers.closest[:n].view(-1)
        phases, most = self.operators.shape[:2]
        step = buffers.traces.shape[0]
        for lo in range(0, len(turns), step):
            hi = min(len(turns), lo + step)
            traces = torch.matmul(turns[lo:hi], self.trace_weights, out=buffers.traces[: hi - lo])
            if phases == 1:
                torch.amax(traces, 1, out=closest[lo:hi])
                continue
            per_phase = buffers.per_phase[: hi - lo]
            torch.amax(traces.view(hi - lo, phases, most), 2, out=per_phase)
            pair_phase = buffers.pair_phase[lo:hi]
            torch.gather(per_phase, 1, pair_phase.view(-1, 1), out=closest[lo:hi].view(-1, 1))
        closest = closest.view(n, pool, pool)
        near = torch.gt(closest, self.near_trace, out=buffers.near[:n])
        same = torch.eq(phase.view(n, pool, 1), phase.view(n, 1, pool), out=buffers.same[:n])
        near.logical_and_(same).logical_and_(self._earlier(pool))
        suppressed = torch.any(near, 2, out=buffers.suppressed[:n])
        suppressed.logical_or_(buffers.invalid[:n])
        # the kept first, best first, then the dropped: each ranked by its place in
        # the pool, the dropped a whole pool later
        key = buffers.key[:n].copy_(suppressed).mul_(pool).add_(self._places(pool))
        pick = buffers.pick[:n]
        torch.topk(key, count, 1, largest=False, out=(buffers.key_top[:n], pick))
        entry = torch.gather(buffers.pool_entry[:n], 1, pick, out=buffers.entry[:n])
        torch.gather(buffers.pool_score[:n], 1, pick, out=buffers.score[:n])
        torch.gather(suppressed, 1, pick, out=buffers.distinct[:n]).logical_not_()
        self._entry_phases_into(entry.view(-1), buffers.phase[:n].view(-1))
        self._entry_rotations_into(entry.view(-1), buffers.rotations[:n].view(-1, 3, 3))

    def _earlier(self, pool: int) -> torch.Tensor:
        """``(pool, pool)``: True where the column is a better place than the row."""
        earlier = getattr(self, "_earlier_mask", None)
        if earlier is None or len(earlier) != pool:
            earlier = torch.ones(pool, pool, device=self.device, dtype=torch.bool).tril(-1)
            self._earlier_mask = earlier
        return earlier

    def _places(self, pool: int) -> torch.Tensor:
        """``(pool,)`` float32: 0, 1, 2, ..., kept on the device."""
        places = getattr(self, "_place_numbers", None)
        if places is None or len(places) != pool:
            places = torch.arange(pool, device=self.device, dtype=torch.float32)
            self._place_numbers = places
        return places

    def _recheck(self, refine_rows, coarse_rows, phase, g, score, entry, buffers, in_place):
        """The patterns the recheck picks refined from their other candidates too,
        and their *phase*, *g*, *score* and *entry* replaced, in place, by the
        refined candidate that matches the pattern best (:meth:`_match`).

        Every (pattern, candidate) pair is refined in the same launch, a batch of
        pairs at a time, so the cost is the pairs', not one launch per rank.
        *in_place* runs the fused kernels into *buffers*, with the patterns'
        *coarse_rows* where a refine level is coarse; otherwise :meth:`refine`."""
        n = len(g)
        margin = torch.sub(buffers.score[:n, 0], buffers.score[:n, 1], out=buffers.margin[:n])
        gate = torch.lt(margin, self.recheck_margin, out=buffers.gate[:n])
        gate.logical_or_(torch.lt(score, self.recheck_score, out=buffers.low[:n]))
        gate.logical_and_(buffers.distinct[:n, 1])
        # one small copy to the host, which says how many pairs to launch for
        picked = np.flatnonzero(gate.cpu().numpy())
        m = len(picked)
        self.rechecked += m
        if not m:
            return
        count = buffers.candidates
        others = count - 1
        pairs = m * others
        index = buffers.index[:m].copy_(torch.from_numpy(picked))
        # the patterns' own refined orientations, matched first: the rows buffer is
        # then the pairs'
        rows = torch.index_select(refine_rows, 0, index, out=buffers.rows[:m])
        best = torch.index_select(g, 0, index, out=buffers.best[:m])
        best_score = torch.index_select(score, 0, index, out=buffers.best_score[:m])
        best_phase = torch.index_select(phase, 0, index, out=buffers.best_phase[:m])
        best_entry = torch.index_select(entry, 0, index, out=buffers.best_entry[:m])
        best_match = self._match(rows, best, best_phase, buffers.best_match[:m], buffers, in_place)
        # each pair: the pattern, and its candidate's place in the (N, count) columns
        candidate = (picked[:, None] * count + np.arange(1, count)).ravel()
        candidate = buffers.pair_candidate[:pairs].copy_(torch.from_numpy(candidate))
        pattern = buffers.pair_pattern[:pairs].copy_(torch.from_numpy(np.repeat(picked, others)))
        seed = torch.index_select(
            buffers.rotations[:n].view(-1, 3, 3), 0, candidate, out=buffers.seed[:pairs]
        )
        seed_phase = torch.index_select(
            buffers.phase[:n].view(-1), 0, candidate, out=buffers.seed_phase[:pairs]
        )
        seed_entry = torch.index_select(
            buffers.entry[:n].view(-1), 0, candidate, out=buffers.seed_entry[:pairs]
        )
        distinct = torch.index_select(
            buffers.distinct[:n].view(-1), 0, candidate, out=buffers.seed_distinct[:pairs]
        )
        refined, refined_score = buffers.refined[:pairs], buffers.refined_score[:pairs]
        match = buffers.match[:pairs]
        step = buffers.capacity
        for lo in range(0, pairs, step):
            hi = min(pairs, lo + step)
            rows = torch.index_select(refine_rows, 0, pattern[lo:hi], out=buffers.rows[: hi - lo])
            if in_place:
                coarse = None
                if coarse_rows is not None:
                    coarse = torch.index_select(
                        coarse_rows, 0, pattern[lo:hi], out=buffers.coarse_rows[: hi - lo]
                    )
                bases = torch.mul(seed_phase[lo:hi], 2, out=buffers.seed_bases[lo:hi])
                # the coarse levels write the refined buffer itself when no fine
                # level follows them
                fine = len(self._coarse_steps()) < len(self.refine_steps)
                first = buffers.coarse_refined[lo:hi] if fine else refined[lo:hi]
                outs = (first, refined[lo:hi], refined_score[lo:hi])
                self._refine_fused(rows, coarse, seed[lo:hi], bases, outs)
            else:
                found, found_score = self.refine(rows, seed_phase[lo:hi], seed[lo:hi])
                refined[lo:hi].copy_(found)
                refined_score[lo:hi].copy_(found_score)
            self._match(
                rows, refined[lo:hi], seed_phase[lo:hi], match[lo:hi], buffers, in_place, lo
            )
        # each pattern's best pair, a repeat of a better candidate never winning
        match.masked_fill_(distinct.logical_not_(), -math.inf)
        top_match, top = buffers.top_match[:m], buffers.top[:m]
        torch.max(match.view(m, others), 1, out=(top_match, top))
        top.add_(buffers.pair_offset[:m].copy_(torch.from_numpy(np.arange(m) * others)))
        better = torch.gt(top_match, best_match, out=buffers.better[:m])
        winner = torch.index_select(refined, 0, top, out=buffers.winner[:m])
        torch.where(better.view(m, 1, 1), winner, best, out=best)
        for chosen, values in (
            (best_score, refined_score),
            (best_phase, seed_phase),
            (best_entry, seed_entry),
        ):
            winner = torch.index_select(values, 0, top, out=buffers.winner_value(values, m))
            torch.where(better, winner, chosen, out=chosen)
        g.index_copy_(0, index, best)
        score.index_copy_(0, index, best_score)
        phase.index_copy_(0, index, best_phase)
        entry.index_copy_(0, index, best_entry)

    def _match(self, rows, g, phase, out, buffers, in_place, first=0):
        """How well each orientation *g* matches its pattern's *rows*, into *out*:
        the normalized cross-correlation at the refine shape against the unblurred
        master, which candidates are chosen by. In place, the phases' channels
        are written to the bases buffer from *first* on."""
        if in_place:
            bases = torch.mul(phase, 2, out=buffers.seed_bases[first : first + len(g)])
            return _kernels.score(self.texture, g, bases, self.refine_directions, rows, 1, out=out)
        scores = self.score_candidates(g[:, None], phase, rows, self.refine_directions)
        return out.copy_(scores[:, 0])

    @holds_lock
    def retry(self, frames, seeds, seed_phases, *, valid=None, fit: bool = False):
        """Each of *frames* ``(N, H, W)`` refined from each of its *seeds* ``(N, S,
        3, 3)`` of phases *seed_phases* ``(N, S)``, the refined seed that matches
        it best kept: ``(phase, g, score, chosen)``, *chosen* ``(N,)`` the seed's
        number. Seeds where *valid* ``(N, S)`` is False are passed over.

        For a finished scan's speckle, re-tried from its neighbors' orientations:
        put the point's own orientation first, so it is kept unless a neighbor's
        matches better (:func:`~de_ebsd.index.neighbors.neighbor_seeds`, and
        :func:`~de_ebsd.index.live.retry_speckle` for the whole pass). The frames
        are prepared with the scan's background as it stands; with *fit* the
        chosen orientations are fitted (:meth:`index`'s ``fit``), and the scan's
        ghost is left as it is."""
        frames = _stack(frames)
        seeds = torch.as_tensor(seeds, dtype=torch.float32, device=self.device)
        seed_phases = torch.as_tensor(seed_phases, dtype=torch.int64, device=self.device)
        if valid is None:
            valid = torch.ones(seed_phases.shape, dtype=torch.bool, device=self.device)
        valid = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        if seeds.shape[:2] != seed_phases.shape or len(seeds) != len(frames):
            raise ValueError("seeds (N, S, 3, 3), seed_phases (N, S) and frames (N, ...)")
        out = []
        for lo in range(0, len(frames), BATCH):
            hi = lo + BATCH
            _, rows = self.prepare(frames[lo:hi])
            best = None
            for number in range(seeds.shape[1]):
                phase = seed_phases[lo:hi, number]
                g, score = seeds[lo:hi, number], None
                if self.refine_steps:
                    g, score = self.refine(rows, phase, g)
                match = self.score_candidates(g[:, None], phase, rows, self.refine_directions)
                match = torch.where(valid[lo:hi, number], match[:, 0], -math.inf)
                score = match if score is None else score
                chosen = torch.full_like(phase, number)
                if best is None:
                    best = [phase, g, score, chosen, match]
                    continue
                better = match > best[4]
                best = [
                    torch.where(better.view(-1, *[1] * (new.ndim - 1)), new, old)
                    for new, old in zip((phase, g, score, chosen, match), best, strict=True)
                ]
            phase, g, score, chosen, _ = best
            if fit:
                g, _, score = self.refine_deformation(
                    rows, phase, g, ghost=self.ghost, strain=False
                )
            out.append((phase, g, score, chosen))
        return tuple(torch.cat(column) for column in zip(*out, strict=True))

    @holds_lock
    def refine(
        self,
        refine_rows: torch.Tensor,
        phase: torch.Tensor,
        g: torch.Tensor,
        *,
        steps=None,
        directions=None,
    ):
        """Each orientation *g* ``(N, 3, 3)`` refined against its pattern:
        ``(g, score)``, the score being the last level's at the seed. *steps*
        (radians) default to the indexer's, *directions* to its pattern center's."""
        steps = self.refine_steps if steps is None else steps
        if self.fused and directions is None and steps:
            # every level of every pattern in one launch, one block per pattern;
            # the coarse levels at half the resolution, a quarter of the pixels
            coarse = [s for s in steps if s >= math.radians(COARSE_REFINE_STEP_DEG)]
            fine = list(steps[len(coarse) :])
            score = None
            if coarse:
                rows = refine_rows.view(len(refine_rows), *self.refine_shape)
                rows = normalize_rows(average_to(rows, self.coarse_shape).flatten(1))
                g, score = _kernels.refine_stepped(
                    self.texture,
                    g,
                    phase * 2,
                    self.coarse_directions,
                    rows,
                    coarse,
                    SINGLE_PRECISION_FLOOR,
                )
            if fine:
                g, score = _kernels.refine_stepped(
                    self.texture,
                    g,
                    phase * 2,
                    self.refine_directions,
                    refine_rows,
                    fine,
                    SINGLE_PRECISION_FLOOR,
                )
            return g, score
        directions = self.refine_directions if directions is None else directions
        score = torch.zeros(len(g), device=self.device)
        for step in steps:
            candidates = g[:, None] @ rotation_vector_to_matrix(self.stencil * step)
            scores = self.score_candidates(candidates, phase, refine_rows, directions)
            g = g @ rotation_vector_to_matrix(parabola_offset(scores) * step)
            score = scores[:, 0]
        return g, score

    # --- the pattern center and the strain --------------------------------------

    @holds_lock
    def refine_pattern_center(self, frames, **kwargs) -> tuple[float, float, float]:
        """The pattern center that best fits the best-scored patterns of *frames*
        ``(N, H, W)``, which the indexer then uses; see
        :func:`~de_ebsd.index.live.calibration.refine_pattern_center`."""
        return calibration.refine_pattern_center(self, _stack(frames), **kwargs)

    @holds_lock
    def strain_per_center(self, **kwargs) -> np.ndarray:
        """``(3, 3, 3)``: the strain a pattern fitted at a center one unit off along
        x, y or z shows; see :func:`~de_ebsd.index.live.calibration.strain_per_center`
        and, to read a scan's centers back from its strain,
        :func:`~de_ebsd.index.live.calibration.centers_from_strain`."""
        return calibration.strain_per_center(self, **kwargs)

    @holds_lock
    def center_maps(self, pattern_centers) -> torch.Tensor:
        """One ``(3, 3)`` map per pattern center; see
        :func:`~de_ebsd.index.live.deformation.center_maps`."""
        detector = replace(self.detector, shape=self.refine_shape)
        return deformation.center_maps(detector, pattern_centers, self.device)

    @holds_lock
    def refine_deformation(self, refine_rows, phase, g, **kwargs):
        """Each orientation refined together with its pattern's strain; see
        :func:`~de_ebsd.index.live.deformation.refine_deformation`."""
        return deformation.refine_deformation(self, refine_rows, phase, g, **kwargs)

    @holds_lock
    def strain_from_reference(self, frames, reference, **kwargs):
        """The strain of *frames* relative to one real *reference* frame; see
        :func:`~de_ebsd.index.live.deformation.strain_from_reference`."""
        return deformation.strain_from_reference(
            self, _stack(frames), as_frames(reference), **kwargs
        )

    @holds_lock
    def index_with_strain(self, frames, pattern_centers=None):
        """:meth:`index`, then each orientation refined together with its pattern's
        strain (:meth:`refine_deformation`): ``(phase, g, strain, score, entry)``.
        *pattern_centers* ``(N, 3)``, one per frame, default to the indexer's.

        Each buffer's orientations join the scan's :attr:`ghost` before its
        strains are fitted, as its frames joined the static background. The strain
        is deviatoric, ``(N, 3, 3)``, in OIM's (EDAX TSL) sample frame as the
        orientations are (:mod:`de_ebsd.core.frames`)."""
        frames = _stack(frames)
        maps = None if pattern_centers is None else self.center_maps(pattern_centers)
        centers = None if pattern_centers is None else np.asarray(pattern_centers, float)
        scan_centers = None if self.inplane is None else self.inplane.scan_centers
        if centers is not None and self.inplane is not None and scan_centers is None:
            # the mask of this call's centers, for this call
            self.inplane.set_scan_centers(centers)
        try:
            return self._index_with_strain(frames, maps, centers)
        finally:
            if self.inplane is not None:
                self.inplane.set_scan_centers(scan_centers)

    def _index_with_strain(self, frames, maps, centers):
        out = []
        batch = FUSED_BATCH if self.fused and deformation.STRAIN_SCHEDULE else BATCH
        for lo in range(0, len(frames), batch):
            search_rows, refine_rows = self.prepare(frames[lo : lo + batch])
            batch_centers = None if centers is None else centers[lo : lo + batch]
            phase, g, _, entry = self._choose(search_rows, refine_rows, batch_centers)
            batch_maps = None if maps is None else maps[lo : lo + batch]
            self.add_fits(phase, g, batch_maps)
            g, strain, score = self.refine_deformation(
                refine_rows, phase, g, center_maps=batch_maps, ghost=self.ghost
            )
            out.append((phase, g, strain, score, entry))
        return tuple(torch.cat(column) for column in zip(*out, strict=True))

    @holds_lock
    def index(self, frames, *, fit: bool = False, out=None):
        """*frames* ``(N, H, W)`` on the device, indexed: ``(phase, g, score, entry)``.
        *entry* is the dictionary entry the search seeded the refine from (see
        :meth:`search`): anything worked out once per entry, such as a color, stands
        for the refined orientation to within the dictionary's resolution. With
        several :attr:`candidates` it is the chosen candidate's
        (:meth:`set_candidates`).

        With *fit*, the refine's orientations are fitted on, rotations only
        (:meth:`refine_deformation` without the strains): the refine's steps stop
        a little short on a noisy pattern, a few degrees off for one in ten on a
        real scan, and the fit takes them to a few tenths.

        *frames* may be at any resolution, or already pooled to the refine shape
        (:func:`~de_ebsd.index.live.common.pool_into`), which indexes the same.
        *out*, four tensors of at least N rows (int64, ``(N, 3, 3)`` float32,
        float32, int64), receives the results, and their first N rows are
        returned; without it the results are new tensors. On the fused path
        nothing else is allocated (:attr:`workspace`).

        *frames* may be one ``(H, W)`` frame, and a NumPy, torch or CuPy (any
        DLPack) array: a GPU array is read where it is, without a copy
        (:func:`~de_ebsd.index.live.common.as_frames`). The orientations are in
        OIM's (EDAX TSL) sample frame (:mod:`de_ebsd.core.frames`);
        :meth:`phase_entries` gives each entry within its own phase."""
        frames = _stack(frames)
        count = len(frames)
        if out is None:
            out = (
                torch.empty(count, dtype=torch.int64, device=self.device),
                torch.empty(count, 3, 3, device=self.device),
                torch.empty(count, device=self.device),
                torch.empty(count, dtype=torch.int64, device=self.device),
            )
        out = tuple(column[:count] for column in out)
        batch = FUSED_BATCH if fit and self.fused else BATCH
        for lo in range(0, count, batch):
            columns = self._index_batch(frames[lo : lo + batch], fit)
            for target, column in zip(out, columns, strict=True):
                target[lo : lo + len(column)].copy_(column)
        return out

    def _index_batch(self, frames, fit):
        """:meth:`index` of one batch, in the :attr:`workspace` where there is one:
        ``(phase, g, score, entry)``, which the next batch overwrites."""
        workspace = self.workspace
        if workspace is None or len(frames) > workspace.capacity or not self.refine_steps:
            return self._index_batch_allocating(frames, fit)
        n = len(frames)
        search_rows, refine_rows = self._prepare_in_workspace(frames)
        buffers = self.candidate_buffers
        if buffers is None:
            phase, g, _, entry = self._search_in_workspace(search_rows, refine_rows)
        else:
            if self.inplane is not None:
                self._search_inplane_candidates(refine_rows, buffers, None)
            else:
                rows = search_rows
                if workspace.typed_search_rows is not None:
                    rows = workspace.typed_search_rows[:n].copy_(search_rows)
                self._search_candidates(rows, buffers, workspace)
            phase = workspace.phase[:n].copy_(buffers.phase[:n, 0])
            g = workspace.seed[:n].copy_(buffers.rotations[:n, 0])
            entry = workspace.entry[:n].copy_(buffers.entry[:n, 0])
        bases = workspace.bases[:n].copy_(phase).mul_(2)
        g, score, coarse_rows = self._refine_in_workspace(refine_rows, g, bases)
        if buffers is not None:
            if coarse_rows is None and self._coarse_steps():
                coarse_rows = self._coarse_rows_in_workspace(refine_rows)
            self._recheck(refine_rows, coarse_rows, phase, g, score, entry, buffers, True)
            bases.copy_(phase).mul_(2)
        if fit:
            self._add_fits_in_workspace(phase, g, bases)
            g, score = self._fit_in_workspace(refine_rows, phase, g, bases, coarse_rows)
        return phase, g, score, entry

    def _index_batch_allocating(self, frames, fit):
        """:meth:`_index_batch` through the public steps, each returning new tensors."""
        search_rows, refine_rows = self.prepare(frames)
        phase, g, score, entry = self._choose(search_rows, refine_rows)
        if fit:
            self.add_fits(phase, g)
            g, _, score = self.refine_deformation(
                refine_rows, phase, g, ghost=self.ghost, strain=False
            )
        return phase, g, score, entry

    def _choose(self, search_rows, refine_rows, pattern_centers=None):
        """Each pattern searched and refined, from its :attr:`candidates` best
        distinct entries where there are several: ``(phase, g, score, entry)``.
        *pattern_centers* are :meth:`search`'s."""
        if self.candidates == 1 or not self.refine_steps:
            phase, g, score, entry = self.search(search_rows, refine_rows, pattern_centers)
            if self.refine_steps:
                g, score = self.refine(refine_rows, phase, g)
            return phase, g, score, entry
        n = len(search_rows)
        buffers = self._candidate_buffers(n)
        if self.inplane is not None:
            if refine_rows is None:
                images = search_rows.float().view(n, 1, *self.search_shape)
                images = torch.nn.functional.interpolate(images, size=self.refine_shape)
                refine_rows = images.view(n, -1)
            self._search_inplane_candidates(refine_rows, buffers, pattern_centers)
        else:
            self._search_candidates(search_rows.to(self.dtype), buffers, None)
        phase = buffers.phase[:n, 0].clone()
        entry = buffers.entry[:n, 0].clone()
        g, score = self.refine(refine_rows, phase, buffers.rotations[:n, 0].contiguous())
        self._recheck(refine_rows, None, phase, g, score, entry, buffers, False)
        return phase, g, score, entry

    def _coarse_steps(self):
        return [s for s in self.refine_steps if s >= math.radians(COARSE_REFINE_STEP_DEG)]

    # --- the same steps, in the workspace -----------------------------------------
    #
    # Each is its public step's arithmetic in the same order, writing into the
    # workspace rather than into new tensors, so the results are the same to the bit.

    def _prepare_in_workspace(self, frames):
        n = len(frames)
        workspace = self.workspace
        prepared = self._fused_prepare(
            frames, out=(workspace.search_rows[:n], workspace.refine_rows[:n])
        )
        if prepared is None:
            return self.prepare(frames)
        if self._residual_rows is not None:
            return self._remove_residual(prepared[1], prepared[0], workspace.scratch)
        return prepared

    def _search_in_workspace(self, search_rows, refine_rows):
        n = len(search_rows)
        workspace = self.workspace
        if self.inplane is not None:
            self.inplane.prepare_mask()
            out = (workspace.phase[:n], workspace.seed[:n], workspace.score[:n])
            return self.inplane.search_into(refine_rows, (*out, workspace.entry[:n]))
        rows = search_rows
        if workspace.typed_search_rows is not None:
            rows = workspace.typed_search_rows[:n].copy_(search_rows)
        score = workspace.score[:n].fill_(-math.inf)
        entry = workspace.entry[:n].zero_()
        chunk_score, chunk_entry = workspace.chunk_score[:n], workspace.chunk_entry[:n]
        better = workspace.better[:n]
        for lo in range(0, self.dictionary.shape[1], SEARCH_CHUNK):
            chunk = self.dictionary[:, lo : lo + SEARCH_CHUNK]
            scores = workspace.scores(n, chunk.shape[1], self.dtype)
            scores = rows @ chunk if scores is None else torch.matmul(rows, chunk, out=scores)
            torch.max(scores, 1, out=(chunk_score, chunk_entry))
            # compared and kept in float32, as the search does
            torch.gt(chunk_score, score, out=better)
            torch.where(better, chunk_score, score, out=score)
            torch.where(better, chunk_entry.add_(lo), entry, out=entry)
        phase = torch.index_select(self.phase_of, 0, entry, out=workspace.phase[:n])
        g = torch.index_select(self.rotations, 0, entry, out=workspace.seed[:n])
        return phase, g, score, entry

    def _coarse_rows_in_workspace(self, refine_rows):
        """The refine rows averaged to the coarse shape and normalized again."""
        n = len(refine_rows)
        workspace = self.workspace
        coarse = workspace.coarse_rows[:n]
        pool_into(refine_rows.view(n, *self.refine_shape), coarse.view(n, *self.coarse_shape))
        return normalize_rows_into(coarse, workspace.scratch)

    def _refine_in_workspace(self, refine_rows, g, bases):
        """:meth:`refine` on the fused path: ``(g, score, coarse_rows)``, the coarse
        rows None where no level was coarse."""
        n = len(g)
        workspace = self.workspace
        coarse_rows = None
        if self._coarse_steps():
            coarse_rows = self._coarse_rows_in_workspace(refine_rows)
        outs = (workspace.coarse_refined[:n], workspace.refined[:n], workspace.score[:n])
        g, score = self._refine_fused(refine_rows, coarse_rows, g, bases, outs)
        return g, score, coarse_rows

    def _refine_fused(self, refine_rows, coarse_rows, g, bases, out):
        """:meth:`refine`'s levels on the fused kernels, the coarse ones on
        *coarse_rows*: ``(g, score)`` written into *out*, ``(coarse_g, g, score)``."""
        coarse = self._coarse_steps()
        fine = list(self.refine_steps[len(coarse) :])
        score = None
        if coarse:
            g, score = _kernels.refine_stepped(
                self.texture,
                g,
                bases,
                self.coarse_directions,
                coarse_rows,
                coarse,
                SINGLE_PRECISION_FLOOR,
                out=(out[0], out[2]),
            )
        if fine:
            g, score = _kernels.refine_stepped(
                self.texture,
                g,
                bases,
                self.refine_directions,
                refine_rows,
                fine,
                SINGLE_PRECISION_FLOOR,
                out=(out[1], out[2]),
            )
        return g, score

    def _add_fits_in_workspace(self, phase, g, bases):
        """:meth:`add_fits`, the simulations projected into the workspace."""
        if self.background.blur:
            return
        workspace = self.workspace
        simulations = workspace.simulations(len(g), len(self.refine_directions))
        if simulations is None:
            self.add_fits(phase, g)
            return
        _kernels.project(self.texture, g, bases, self.refine_directions, out=simulations)
        self.background.add_fits_in_place(simulations, workspace.scratch)

    def _fit_in_workspace(self, refine_rows, phase, g, bases, coarse_rows):
        """:meth:`refine_deformation` without the strains: ``(g, score)``."""
        workspace = self.workspace
        ghost = self.background.ghost_into(workspace.ghost, workspace.scratch)
        schedule = deformation.FIT_SCHEDULE
        if not (deformation.ANALYTIC_JACOBIAN and schedule):
            g, _, score = self.refine_deformation(refine_rows, phase, g, ghost=ghost, strain=False)
            return g, score
        low = None
        if any(level for level, _ in schedule):
            if coarse_rows is None:
                coarse_rows = self._coarse_rows_in_workspace(refine_rows)
            coarse_ghost = None
            if ghost is not None:
                coarse_ghost = workspace.coarse_ghost[None]
                pool_into(
                    ghost.view(1, *self.refine_shape), coarse_ghost.view(1, *self.coarse_shape)
                )
                coarse_ghost = normalize_rows_into(coarse_ghost, workspace.scratch)[0]
            low = (coarse_rows, coarse_ghost)
        n = len(g)
        g, score, _ = deformation._fit_passes(
            self,
            refine_rows,
            phase,
            g,
            None,
            ghost,
            schedule,
            deformation.FIT_MAX_STEP,
            low=low,
            bases=bases,
            out=(workspace.fitted[:n], workspace.score[:n], workspace.refused[:n]),
        )
        return g, score


def _operator_table(point_groups, device) -> torch.Tensor:
    """Each point group's proper rotations as matrices ``(groups, most, 3, 3)``, made
    as the dictionary's are (:func:`~de_ebsd.core.rotation.quaternion_to_matrix`
    of orix's quaternions) and padded with the identity."""
    groups = []
    for point_group in point_groups:
        proper = getattr(point_group, "proper_subgroup", point_group)
        q = torch.as_tensor(np.asarray(proper.data), dtype=torch.float32, device=device)
        groups.append(quaternion_to_matrix(q.reshape(-1, 4)))
    most = max(len(m) for m in groups)
    table = torch.eye(3, device=device).repeat(len(groups), most, 1, 1)
    for number, matrices in enumerate(groups):
        table[number, : len(matrices)] = matrices
    return table


def _stack(frames) -> torch.Tensor:
    """*frames* where they are (:func:`~de_ebsd.index.live.common.as_frames`), one
    ``(H, W)`` frame as a stack of one."""
    t = as_frames(frames)
    return t[None] if t.ndim == 2 else t


class _PoolView:
    """Rows *first*:*last* of a :class:`CandidateBuffers`' pool and the scratch
    :meth:`LiveIndexer._pool_slice` ranks into, as the buffers themselves: the
    in-plane candidate search fills a pool a capacity of patterns at a time."""

    def __init__(self, buffers: CandidateBuffers, first: int, last: int) -> None:
        self._buffers = buffers
        self.pool = buffers.pool
        self.pool_score = buffers.pool_score[first:last]
        self.pool_entry = buffers.pool_entry[first:last]
        self.merged_score = buffers.merged_score[first:last]
        self.merged_entry = buffers.merged_entry[first:last]
        self.order = buffers.order[first:last]

    def __getattr__(self, name):
        return getattr(self._buffers, name)
