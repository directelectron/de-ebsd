"""Live indexing: a buffer of camera frames indexed and refined where it already is.

Built for a camera streaming thousands of patterns a second into GPU memory. A
buffer of frames is indexed in a fixed number of batched steps -- nothing loops
over patterns or phases -- and only each pattern's phase, orientation and score
leave the device.

1. **Prepare** (:meth:`LiveIndexer.prepare`): each frame averaged over the whole
   sensor down to the refine shape, the scan's static background (the running
   mean of its frames, :meth:`LiveIndexer.observe`) and the per-pattern dynamic
   background removed, then averaged again to the search shape.
2. **Search** (:meth:`LiveIndexer.search`): one matrix product against every
   phase's dictionary at once. The dictionaries are the phases' dynamical master
   patterns projected over their fundamental zones, stacked into one matrix, so the
   best entry gives the phase and a seed orientation together.
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
import torch.nn.functional as F

from de_ebsd.core.lambert import SQRT_PI_HALF, sphere_to_square
from de_ebsd.core.rotation import quaternion_to_matrix
from de_ebsd.index import _kernels
from de_ebsd.index.preprocess import _gaussian_blur

__all__ = ["LiveIndexer"]

_EPS = 1e-6

#: Frames the static background is averaged over before it is used: fewer still
#: carry the bands of the few grains they came from.
MIN_FRAMES_FOR_STATIC = 32

#: The local dictionary of one refine level, in units of its step: the seed, then
#: +x, -x, +y, -y, +z, -z.
STENCIL = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

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

#: Pattern-center steps (fractions of the detector, as the center is), each half the
#: last. On the twin scan above, starts 0.004-0.01 off in every coordinate came back
#: within 0.001 of the truth (256 patterns, 0.45 s); 0.002 off costs 0.07° of median
#: orientation error.
PATTERN_CENTER_STEPS = (0.008, 0.004, 0.002, 0.001)

#: The orientation refine run at each candidate center. It must resolve far finer
#: than the indexing's: a center 0.004 off in x moves every pattern's best fit by a
#: few tenths of a degree, and steps ending at 0.12° follow that in jumps, which
#: leaves the mean score flat in x and the center stuck 0.003 short.
PATTERN_CENTER_ORIENTATION_STEPS_DEG = (0.5, 0.25, 0.12, 0.06, 0.03)

#: Patterns a center is fitted with: the best-scored of the buffer.
PATTERN_CENTER_PATTERNS = 256

#: Patterns per batched step, which bounds the largest intermediate: the refine's
#: (patterns, 7, pixels, 3) sample directions.
BATCH = 256


def _normalized_rows(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(-1, keepdim=True)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def _texture(masters, device, blur_deg: float = 0.0) -> torch.Tensor:
    """Every phase's hemispheres as channels of one texture ``(1, 2 * phases, n, n)``:
    one sampling pass serves all phases, and each pattern keeps its own phase's."""
    if len({m.npx for m in masters}) != 1:
        raise ValueError("every master must have the same npx")
    texture = torch.cat([m.data[m.energy_index(None)].to(device, torch.float32) for m in masters])
    # Near the center of the square Lambert grid a pixel spans 90° / npx.
    per_degree = masters[0].npx / 90.0
    if blur_deg:
        texture = _gaussian_blur(texture, blur_deg * per_degree, 4.0)
    return texture[None].contiguous()


def _average_to(stack: torch.Tensor, shape) -> torch.Tensor:
    """*stack* ``(N, H, W)`` averaged down to *shape*: a whole-number bin when the
    shapes divide, which runs several times faster than a fractional one."""
    rows, cols = stack.shape[-2] // shape[0], stack.shape[-1] // shape[1]
    if (rows * shape[0], cols * shape[1]) == tuple(stack.shape[-2:]):
        return F.avg_pool2d(stack[:, None], (rows, cols))[:, 0]
    return F.adaptive_avg_pool2d(stack[:, None], shape)[:, 0]


def _parabola_offset(scores: torch.Tensor) -> torch.Tensor:
    """The step ``(N, 3)``, in units of the stencil's, to each axis's best: the
    vertex of the parabola through the scores ``(N, 7)`` at -1, 0 and +1 (see
    :data:`STENCIL`), or the better neighbor where the three make no maximum."""
    center, plus, minus = scores[:, :1], scores[:, 1::2], scores[:, 2::2]
    curvature = plus - 2 * center + minus
    vertex = 0.5 * (minus - plus) / curvature.clamp_max(-_EPS)
    fallback = torch.where(plus > minus, 1.0, -1.0) * (torch.maximum(plus, minus) > center)
    return torch.where(curvature < 0, vertex.clamp(-1, 1), fallback.to(scores.dtype))


def _rotation_matrices(w: torch.Tensor) -> torch.Tensor:
    """Rotation vectors ``(..., 3)`` in radians as matrices ``(..., 3, 3)``."""
    theta = w.norm(dim=-1, keepdim=True)
    k = w / theta.clamp_min(1e-12)
    zero = torch.zeros_like(k[..., 0])
    skew = torch.stack(
        [zero, -k[..., 2], k[..., 1], k[..., 2], zero, -k[..., 0], -k[..., 1], k[..., 0], zero],
        dim=-1,
    ).reshape(*k.shape[:-1], 3, 3)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(skew)
    return eye + s * skew + (1 - c) * (skew @ skew)


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
    ) -> None:
        from orix.sampling import get_sample_fundamental

        if len(masters) != len(point_groups):
            raise ValueError("one point group per master")
        self.device = torch.device(device) if device is not None else masters[0].device
        self.detector = detector
        self.search_shape = tuple(search_shape)
        self.refine_shape = tuple(refine_shape)
        self.refine_steps = tuple(math.radians(s) for s in refine_steps_deg)
        self.fused = _kernels.available(self.device)
        self.texture = _texture(masters, self.device)
        self.search_texture = _texture(masters, self.device, search_blur_deg)
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
        self.set_pattern_center(detector.pc)
        self.static = None
        self._static_sum = torch.zeros(self.refine_shape, device=self.device)
        self._static_count = 0

    def _directions(self, pc, shape) -> torch.Tensor:
        detector = replace(self.detector, pc=tuple(float(v) for v in pc), shape=shape)
        return detector.direction_cosines(device=self.device)

    def set_pattern_center(self, pc) -> None:
        """Index at pattern center *pc* from now on: the dictionary is projected
        again, which takes milliseconds with the fused kernels."""
        self.detector = replace(self.detector, pc=tuple(float(v) for v in pc))
        self.search_directions = self._directions(pc, self.search_shape)
        self.refine_directions = self._directions(pc, self.refine_shape)
        entries = [
            _normalized_rows(
                self._project(
                    self.search_texture,
                    self.rotations[lo : lo + BATCH * 8],
                    self.phase_of[lo : lo + BATCH * 8],
                    self.search_directions,
                )
            )
            for lo in range(0, len(self.rotations), BATCH * 8)
        ]
        #: ``(pixels, entries)``, contiguous so the search's product runs at full rate.
        self.dictionary = torch.cat(entries).T.contiguous()

    def _project(
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
        square = sphere_to_square(v.reshape(-1, 3)) / SQRT_PI_HALF
        values = F.grid_sample(
            texture,
            square.view(1, 1, -1, 2),
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        ).view(texture.shape[1], *v.shape[:2])
        channel = phases[:, None] * 2 + (v[..., 2] < 0)
        return values.gather(0, channel[None])[0]

    def _scores(
        self, candidates: torch.Tensor, phases: torch.Tensor, rows: torch.Tensor, directions
    ) -> torch.Tensor:
        """Scores ``(N, K)`` of candidates ``(N, K, 3, 3)`` against their patterns'
        normalized *rows* ``(N, pixels)``, sampled at *directions*."""
        n, k = candidates.shape[:2]
        flat = candidates.reshape(-1, 3, 3)
        bases = phases.repeat_interleave(k) * 2
        if self.fused:
            return _kernels.score(self.texture, flat, bases, directions, rows, k).view(n, k)
        simulated = _normalized_rows(self._project(self.texture, flat, bases // 2, directions))
        return torch.einsum("nkp,np->nk", simulated.view(n, k, -1), rows)

    def restart_static(self) -> None:
        """Forget the static background, for a new scan."""
        self.static = None
        self._static_sum.zero_()
        self._static_count = 0

    def _pooled(self, frames) -> torch.Tensor:
        t = torch.as_tensor(frames, device=self.device)
        t = t[None] if t.ndim == 2 else t
        return _average_to(t.to(torch.float32), self.refine_shape)

    def observe(self, frames) -> None:
        """Add *frames* to the scan's static background."""
        pooled = self._pooled(frames)
        self._static_sum += pooled.sum(0)
        self._static_count += len(pooled)
        if self._static_count >= MIN_FRAMES_FOR_STATIC:
            self.static = self._static_sum / self._static_count

    def prepare(self, frames) -> tuple[torch.Tensor, torch.Tensor]:
        """*frames* ``(N, H, W)`` corrected and normalized, as ``(search, refine)``
        rows ``(N, pixels)`` at the two shapes."""
        t = self._pooled(frames)
        if self.static is not None:
            t = t - self.static
        # The dynamic background is an eighth of the width wide, so it is found at a
        # quarter of the resolution and drawn back up: a full-size blur that wide
        # was most of this step's time.
        coarse = _average_to(t, (self.refine_shape[0] // 4, self.refine_shape[1] // 4))
        coarse = _gaussian_blur(coarse, coarse.shape[-1] / 8.0, 4.0)
        t = (
            t
            - F.interpolate(
                coarse[:, None], size=self.refine_shape, mode="bilinear", align_corners=False
            )[:, 0]
        )
        search = _average_to(t, self.search_shape)
        return _normalized_rows(search.flatten(1)), _normalized_rows(t.flatten(1))

    def search(self, search_rows: torch.Tensor):
        """The best dictionary entry for each pattern: ``(phase, g, score, entry)``,
        *entry* indexing :attr:`rotations` (every phase's entries, in phase order)."""
        score, entry = (search_rows @ self.dictionary).max(1)
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
        directions = self.refine_directions if directions is None else directions
        score = torch.zeros(len(g), device=self.device)
        for step in steps:
            candidates = g[:, None] @ _rotation_matrices(self.stencil * step)
            scores = self._scores(candidates, phase, refine_rows, directions)
            g = g @ _rotation_matrices(_parabola_offset(scores) * step)
            score = scores[:, 0]
        return g, score

    def refine_pattern_center(
        self,
        frames,
        *,
        patterns: int = PATTERN_CENTER_PATTERNS,
        steps=PATTERN_CENTER_STEPS,
        orientation_steps_deg=PATTERN_CENTER_ORIENTATION_STEPS_DEG,
    ) -> tuple[float, float, float]:
        """The pattern center that best fits the best-scored *patterns* of
        *frames* ``(N, H, W)``, which the indexer then uses
        (:meth:`set_pattern_center`).

        A center moved along x or y looks to every pattern much like the sample
        turned slightly, so the center is fitted with each orientation free: at
        each candidate center every orientation is refined again
        (*orientation_steps_deg*) and the candidate scored by the patterns' mean
        score. A level scores the center and one step either side of it on each
        axis and moves to each axis's parabola vertex, as :meth:`refine` does for
        an orientation. The patterns should come from several grains: one
        orientation alone cannot tell the center from a turn.
        """
        phase, g, score, _ = self.index(frames)
        best = torch.argsort(score, descending=True)[:patterns]
        _, rows = self.prepare(torch.as_tensor(frames, device=self.device)[best])
        phase, g = phase[best], g[best]
        orientation_steps = tuple(math.radians(s) for s in orientation_steps_deg)
        pc = torch.tensor(self.detector.pc, dtype=torch.float64)
        for step in steps:
            scores, refined = [], []
            for offset in self.stencil.cpu().double():
                directions = self._directions(pc + step * offset, self.refine_shape)
                found, found_score = self.refine(
                    rows, phase, g, steps=orientation_steps, directions=directions
                )
                scores.append(found_score.mean())
                refined.append(found)
            g = refined[0]
            pc = pc + step * _parabola_offset(torch.stack(scores).double().cpu()[None])[0]
        center = tuple(float(v) for v in pc)
        self.set_pattern_center(center)
        return center

    def index(self, frames):
        """*frames* ``(N, H, W)`` on the device, indexed: ``(phase, g, score, entry)``.
        *entry* is the dictionary entry the search seeded the refine from (see
        :meth:`search`): anything worked out once per entry, such as a color, stands
        for the refined orientation to within the dictionary's resolution."""
        out = []
        for lo in range(0, len(frames), BATCH):
            search_rows, refine_rows = self.prepare(frames[lo : lo + BATCH])
            phase, g, score, entry = self.search(search_rows)
            if self.refine_steps:
                g, score = self.refine(refine_rows, phase, g)
            out.append((phase, g, score, entry))
        return tuple(torch.cat(column) for column in zip(*out, strict=True))
