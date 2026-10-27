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
from de_ebsd.core.device import resolve_device
from de_ebsd.core.lambert import sample_texture
from de_ebsd.core.rotation import quaternion_to_matrix, rotation_vector_to_matrix
from de_ebsd.index import _kernels
from de_ebsd.index.live import calibration, deformation
from de_ebsd.index.live.background import ScanBackground
from de_ebsd.index.live.common import (
    BATCH,
    FUSED_BATCH,
    STENCIL,
    average_to,
    parabola_offset,
)
from de_ebsd.index.preprocess import gaussian_blur, normalize_rows

__all__ = ["LiveIndexer"]


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
        One :class:`~de_ebsd.sim.MasterPattern` per phase, all at the same ``npx``.
        Each is used at its highest energy bin.
    point_groups
        Each phase's orix point group, whose fundamental zone its dictionary samples.
    detector
        The camera's :class:`~de_ebsd.sim.detector.EBSDDetector` (its pattern center
        and angles); frames may come at any resolution, since they are averaged to
        ``search_shape`` and ``refine_shape``.
    resolution
        The dictionary's sampling, in degrees.
    search_shape, refine_shape
        Detector shapes the search and the refine compare patterns at.
    refine_steps_deg
        The refine levels' steps; empty for no refine.
    search_blur_deg
        How much the search's copy of the masters is blurred, see
        :data:`SEARCH_BLUR_DEG`.
    device
        Defaults to the first master's.
    dtype
        The search dictionary's. ``None`` picks float16 on GPUs with tensor cores
        (compute capability 7.0 and up) and float32 elsewhere: Pascal multiplies
        float16 at 1/64 of its float32 rate, and a CPU has no float16 product. Half
        precision halves the dictionary, which is what grows with low symmetry; the
        search only picks seeds, so the refine sets the accuracy either way.
    """

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
        search_blur_deg: float = SEARCH_BLUR_DEG,
        device=None,
        dtype=None,
    ) -> None:
        from orix.sampling import get_sample_fundamental

        if len(masters) != len(point_groups):
            raise ValueError("one point group per master")
        self.device = resolve_device(device, masters[0].device)
        if dtype is None:
            tensor_cores = (
                self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 7
            )
            dtype = torch.float16 if tensor_cores else torch.float32
        self.dtype = dtype
        self.dictionary = None
        self.detector = detector
        self.search_shape = tuple(search_shape)
        self.refine_shape = tuple(refine_shape)
        #: Where the refine's coarse levels (:data:`COARSE_REFINE_STEP_DEG` and up)
        #: are scored on the fused path: half the refine's resolution.
        self.coarse_shape = tuple(max(8, s // 2) for s in self.refine_shape)
        self.refine_steps = tuple(math.radians(s) for s in refine_steps_deg)
        self.fused = _kernels.available(self.device)
        self.texture = texture_of(masters, self.device)
        self.search_texture = texture_of(masters, self.device, search_blur_deg)
        self.stencil = torch.tensor(STENCIL, device=self.device, dtype=torch.float32)

        rotations, phases = [], []
        for number, point_group in enumerate(point_groups):
            q = np.asarray(
                get_sample_fundamental(resolution=resolution, point_group=point_group).data
            )
            rotations.append(
                quaternion_to_matrix(torch.as_tensor(q, device=self.device, dtype=torch.float32))
            )
            phases.append(torch.full((len(q),), number, device=self.device, dtype=torch.int64))
        self.rotations = torch.cat(rotations)
        self.phase_of = torch.cat(phases)
        self.sizes = [len(r) for r in rotations]
        self.set_pattern_center(detector.pattern_center)
        self.background = ScanBackground(self.refine_shape, self.device)

    def directions_at(self, pattern_center, shape) -> torch.Tensor:
        detector = replace(
            self.detector, pattern_center=tuple(float(v) for v in pattern_center), shape=shape
        )
        return detector.direction_cosines(device=self.device)

    def set_pattern_center(self, pattern_center) -> None:
        """Index at pattern center *pattern_center* from now on: the dictionary is projected
        again, which takes milliseconds with the fused kernels.

        Each chunk of entries is written straight into the dictionary, and a new
        center overwrites the old one's in place: the dictionary is the largest
        thing on the device, and building it beside a copy of itself (or a list of
        its chunks) took two to three times its size."""
        self.detector = replace(
            self.detector, pattern_center=tuple(float(v) for v in pattern_center)
        )
        self.search_directions = self.directions_at(pattern_center, self.search_shape)
        self.refine_directions = self.directions_at(pattern_center, self.refine_shape)
        self.coarse_directions = self.directions_at(pattern_center, self.coarse_shape)
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

    def pooled(self, frames) -> torch.Tensor:
        """*frames* on the device as float ``(N, H, W)`` averaged to the refine shape."""
        t = torch.as_tensor(frames, device=self.device)
        t = t[None] if t.ndim == 2 else t
        return average_to(t.to(torch.float32), self.refine_shape)

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

    def observe(self, frames) -> None:
        """Add *frames* to the scan's static background."""
        self.background.observe(self.pooled(frames))

    def add_fits(self, phase, g, center_maps=None) -> None:
        """Add the simulations of fitted orientations *g* ``(N, 3, 3)`` to the
        scan's :attr:`ghost`."""
        if self.background.blur:
            return
        if center_maps is not None:
            g = g @ center_maps
        self.background.add_fits(self.project(self.texture, g, phase, self.refine_directions))

    def corrected(self, frames) -> torch.Tensor:
        """*frames* ``(N, H, W)`` with the backgrounds removed as :meth:`prepare` removes
        them, at their own resolution: what the indexer matches, to be looked at."""
        t = torch.as_tensor(frames, device=self.device)
        t = t[None] if t.ndim == 2 else t
        return self.background.remove(t.to(torch.float32))

    def prepare(self, frames) -> tuple[torch.Tensor, torch.Tensor]:
        """*frames* ``(N, H, W)`` corrected and normalized, as ``(search, refine)``
        rows ``(N, pixels)`` at the two shapes."""
        t = self.background.remove(self.pooled(frames))
        search = average_to(t, self.search_shape)
        return normalize_rows(search.flatten(1)), normalize_rows(t.flatten(1))

    def search(self, search_rows: torch.Tensor):
        """The best dictionary entry for each pattern: ``(phase, g, score, entry)``,
        *entry* indexing :attr:`rotations` (every phase's entries, in phase order).

        The dictionary is scored :data:`SEARCH_CHUNK` entries at a time, keeping
        each pattern's best so far, so the scores never outgrow one chunk."""
        rows = search_rows.to(self.dtype)
        score = torch.full((len(rows),), -math.inf, device=self.device)
        entry = torch.zeros(len(rows), dtype=torch.int64, device=self.device)
        for lo in range(0, self.dictionary.shape[1], SEARCH_CHUNK):
            chunk_score, chunk_entry = (rows @ self.dictionary[:, lo : lo + SEARCH_CHUNK]).max(1)
            better = chunk_score.float() > score
            score = torch.where(better, chunk_score.float(), score)
            entry = torch.where(better, chunk_entry + lo, entry)
        return self.phase_of[entry], self.rotations[entry], score, entry

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

    def refine_pattern_center(self, frames, **kwargs) -> tuple[float, float, float]:
        """The pattern center that best fits the best-scored patterns of *frames*
        ``(N, H, W)``, which the indexer then uses; see
        :func:`~de_ebsd.index.live.calibration.refine_pattern_center`."""
        return calibration.refine_pattern_center(self, frames, **kwargs)

    def center_maps(self, pattern_centers) -> torch.Tensor:
        """One ``(3, 3)`` map per pattern center; see
        :func:`~de_ebsd.index.live.deformation.center_maps`."""
        detector = replace(self.detector, shape=self.refine_shape)
        return deformation.center_maps(detector, pattern_centers, self.device)

    def refine_deformation(self, refine_rows, phase, g, **kwargs):
        """Each orientation refined together with its pattern's strain; see
        :func:`~de_ebsd.index.live.deformation.refine_deformation`."""
        return deformation.refine_deformation(self, refine_rows, phase, g, **kwargs)

    def strain_from_reference(self, frames, reference, **kwargs):
        """The strain of *frames* relative to one real *reference* frame; see
        :func:`~de_ebsd.index.live.deformation.strain_from_reference`."""
        return deformation.strain_from_reference(self, frames, reference, **kwargs)

    def index_with_strain(self, frames, pattern_centers=None):
        """:meth:`index`, then each orientation refined together with its pattern's
        strain (:meth:`refine_deformation`): ``(phase, g, strain, score, entry)``.
        *pattern_centers* ``(N, 3)``, one per frame, default to the indexer's.

        Each buffer's orientations join the scan's :attr:`ghost` before its
        strains are fitted, as its frames joined the static background."""
        maps = None if pattern_centers is None else self.center_maps(pattern_centers)
        out = []
        for lo in range(0, len(frames), BATCH):
            search_rows, refine_rows = self.prepare(frames[lo : lo + BATCH])
            phase, g, _, entry = self.search(search_rows)
            g, _ = self.refine(refine_rows, phase, g)
            batch_maps = None if maps is None else maps[lo : lo + BATCH]
            self.add_fits(phase, g, batch_maps)
            g, strain, score = self.refine_deformation(
                refine_rows, phase, g, center_maps=batch_maps, ghost=self.ghost
            )
            out.append((phase, g, strain, score, entry))
        return tuple(torch.cat(column) for column in zip(*out, strict=True))

    def index(self, frames, *, fit: bool = False):
        """*frames* ``(N, H, W)`` on the device, indexed: ``(phase, g, score, entry)``.
        *entry* is the dictionary entry the search seeded the refine from (see
        :meth:`search`): anything worked out once per entry, such as a color, stands
        for the refined orientation to within the dictionary's resolution.

        With *fit*, the refine's orientations are fitted on, rotations only
        (:meth:`refine_deformation` without the strains): the refine's steps stop
        a little short on a noisy pattern, a few degrees off for one in ten on a
        real scan, and the fit takes them to a few tenths."""
        out = []
        batch = FUSED_BATCH if fit and self.fused else BATCH
        for lo in range(0, len(frames), batch):
            search_rows, refine_rows = self.prepare(frames[lo : lo + batch])
            phase, g, score, entry = self.search(search_rows)
            if self.refine_steps:
                g, score = self.refine(refine_rows, phase, g)
            if fit:
                self.add_fits(phase, g)
                g, _, score = self.refine_deformation(
                    refine_rows, phase, g, ghost=self.ghost, strain=False
                )
            out.append((phase, g, score, entry))
        return tuple(torch.cat(column) for column in zip(*out, strict=True))
