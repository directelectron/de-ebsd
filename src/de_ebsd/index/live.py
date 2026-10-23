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

#: Dictionary entries scored at a time by :meth:`LiveIndexer.search`, which bounds
#: its scores at (patterns, this): 64 MB for a batch. A cubic phase's 3° dictionary
#: (30,443 entries) is one chunk; a triclinic one's (729,000) would otherwise
#: score into 750 MB per batch.
SEARCH_CHUNK = 65_536

#: Gauss-Newton iterations of :meth:`LiveIndexer.refine_deformation`. On the twin
#: scans of :data:`REFINE_STEPS_DEG`, rendered with a known random strain of 1e-3 and
#: 3e-3, three took the orientation median from 0.14-0.20° (the stencil refine
#: alone) to 0.073-0.076°; a fourth and a fifth changed nothing.
DEFORMATION_ITERATIONS = 3

#: The central-difference step of the deformation's Jacobian, as a strain. Much
#: smaller steps measure the bilinear master texture's facets rather than its slope.
DEFORMATION_STEP = 2e-3

#: The largest change one Gauss-Newton iteration may make to any of the
#: deformation's eight numbers. A step is the solve of a local linear model, and on
#: a poor pattern (a grain boundary, two patterns overlapping) that model can
#: point far outside the elastic range; without the limit 1-3% of a real scan's
#: fits ran off to strains of 0.1 and more.
DEFORMATION_MAX_STEP = 5e-3

#: How far below its seed's score a deformation fit may end before it is refused:
#: a fit that only ties its seed loses that much to rounding.
DEFORMATION_SCORE_MARGIN = 1e-4


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


def _pick(frames: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``frames[rows]``, for a camera's uint16 frames too: CUDA cannot index a
    uint16 tensor, but it can the same bits as int16."""
    if frames.dtype == torch.uint16:
        return frames.view(torch.int16)[rows].view(torch.uint16)
    return frames[rows]


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


def _trace_free_basis(device) -> torch.Tensor:
    """The 8 trace-free 3x3 matrices ``(8, 3, 3)``, orthonormal in the Frobenius
    sense: rotations about x, y and z, then the five deviatoric strains. A
    projection cannot see the ninth, hydrostatic, component."""
    root2, root6 = math.sqrt(2.0), math.sqrt(6.0)
    basis = torch.zeros(8, 3, 3, dtype=torch.float64)
    for k, (i, j) in enumerate(((1, 2), (2, 0), (0, 1))):
        basis[k, i, j], basis[k, j, i] = -1 / root2, 1 / root2
        basis[3 + k, i, j] = basis[3 + k, j, i] = 1 / root2
    basis[6] = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)) / root2
    basis[7] = torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=torch.float64)) / root6
    return basis.to(device, torch.float32)


def polar_parts(beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The deformation gradient ``F = (I - beta)^-1`` of a displacement gradient
    *beta* ``(N, 3, 3)``, split as ``F = R U``: ``(R, U - I)``, the lattice rotation
    and the (Biot) strain, both in the sample frame."""
    eye = torch.eye(3, device=beta.device, dtype=torch.float64)
    f = torch.linalg.inv(eye - beta.double())
    u, s, vt = torch.linalg.svd(f)
    stretch = vt.transpose(-1, -2) @ torch.diag_embed(s) @ vt
    return (u @ vt).to(beta.dtype), (stretch - eye).to(beta.dtype)


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
        self.device = torch.device(device) if device is not None else masters[0].device
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
        self.static_blur = None
        self._static_sum = torch.zeros(self.refine_shape, device=self.device)
        self._static_count = 0
        self._fit_sum = torch.zeros(math.prod(self.refine_shape), device=self.device)
        self._fit_count = 0

    def _directions(self, pc, shape) -> torch.Tensor:
        detector = replace(self.detector, pc=tuple(float(v) for v in pc), shape=shape)
        return detector.direction_cosines(device=self.device)

    def set_pattern_center(self, pc) -> None:
        """Index at pattern center *pc* from now on: the dictionary is projected
        again, which takes milliseconds with the fused kernels.

        Each chunk of entries is written straight into the dictionary, and a new
        center overwrites the old one's in place: the dictionary is the largest
        thing on the device, and building it beside a copy of itself (or a list of
        its chunks) took two to three times its size."""
        self.detector = replace(self.detector, pc=tuple(float(v) for v in pc))
        self.search_directions = self._directions(pc, self.search_shape)
        self.refine_directions = self._directions(pc, self.refine_shape)
        shape = (len(self.search_directions), len(self.rotations))
        if self.dictionary is None or tuple(self.dictionary.shape) != shape:
            #: ``(pixels, entries)``, so the search's product reads it at full rate.
            self.dictionary = torch.empty(shape, device=self.device, dtype=self.dtype)
        for lo in range(0, len(self.rotations), BATCH * 8):
            entries = _normalized_rows(
                self._project(
                    self.search_texture,
                    self.rotations[lo : lo + BATCH * 8],
                    self.phase_of[lo : lo + BATCH * 8],
                    self.search_directions,
                )
            )
            self.dictionary[:, lo : lo + len(entries)] = entries.T

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

    def restart_static(self, frames=None, *, blur: float | None = None) -> None:
        """Forget the scan, for a new one: its static background and its ghost.

        *frames* from all over the same field, such as a search's, start the new
        static background, so the scan's first patterns are divided by one that
        holds no grain's bands rather than by the mean of its first few frames.

        With *blur* (a Gaussian sigma, as a share of the pattern's width) the
        static background is the frames' mean blurred, for a single crystal: its
        frames all carry its bands, so their mean would divide them out. It then
        has no ghost."""
        self.static = None
        self.static_blur = blur
        self._static_sum.zero_()
        self._static_count = 0
        self._fit_sum.zero_()
        self._fit_count = 0
        if frames is not None:
            self.observe(frames)

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
            static = self._static_sum / self._static_count
            if self.static_blur:
                static = _gaussian_blur(static[None], self.static_blur * static.shape[-1], 4.0)[0]
            self.static = static

    def _without_background(self, t: torch.Tensor) -> torch.Tensor:
        """*t* ``(N, H, W)`` divided by the static background, less each pattern's
        dynamic one, at whatever shape it is.

        Divided rather than less: the bands' contrast scales with the
        backscattered intensity, so a pattern less its static background still
        has its bands weighted by the bright middle of the screen, and that
        weighting moves a band's center differently in every grain."""
        shape = tuple(t.shape[-2:])
        if self.static is not None:
            static = self.static
            if shape != self.refine_shape:
                static = F.interpolate(
                    static[None, None], size=shape, mode="bilinear", align_corners=False
                )[0, 0]
            t = t / static.clamp_min(_EPS) - 1.0
        # The dynamic background is an eighth of the width wide, so it is found at a
        # quarter of the refine resolution and drawn back up: a full-size blur that
        # wide was most of the preparation's time.
        coarse = _average_to(t, (self.refine_shape[0] // 4, self.refine_shape[1] // 4))
        coarse = _gaussian_blur(coarse, coarse.shape[-1] / 8.0, 4.0)
        return (
            t
            - F.interpolate(coarse[:, None], size=shape, mode="bilinear", align_corners=False)[:, 0]
        )

    def corrected(self, frames) -> torch.Tensor:
        """*frames* ``(N, H, W)`` with the backgrounds removed as :meth:`prepare` removes
        them, at their own resolution: what the indexer matches, to be looked at."""
        t = torch.as_tensor(frames, device=self.device)
        t = t[None] if t.ndim == 2 else t
        return self._without_background(t.to(torch.float32))

    def prepare(self, frames) -> tuple[torch.Tensor, torch.Tensor]:
        """*frames* ``(N, H, W)`` corrected and normalized, as ``(search, refine)``
        rows ``(N, pixels)`` at the two shapes."""
        t = self._without_background(self._pooled(frames))
        search = _average_to(t, self.search_shape)
        return _normalized_rows(search.flatten(1)), _normalized_rows(t.flatten(1))

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
        _, rows = self.prepare(_pick(torch.as_tensor(frames, device=self.device), best))
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

    def center_maps(self, pattern_centers) -> torch.Tensor:
        """``(N, 3, 3)`` maps taking the refine's directions at the indexer's pattern
        center to those at each of *pattern_centers* ``(N, 3)``, up to length.

        On a flat screen a pixel's position is linear in its index at any center,
        so moving the center is exactly a linear map of the directions: in a
        candidate ``g @ m`` it is one more matrix, and the fused kernels, which
        normalize what they are given, need nothing new. It also means that, within
        one pattern, a center error is indistinguishable from a strain, which is
        why a scan's centers must come from its geometry rather than from each
        pattern.
        """
        from de_ebsd.core.geometry import sample_to_detector_matrix

        detector = replace(self.detector, shape=self.refine_shape)

        def screen(pc):
            # (col, row, 1) -> the unnormalized screen position, as direction_cosines
            b = replace(detector, pc=tuple(float(v) for v in pc)).gnomonic_bounds
            x_scale, y_scale = (b[1] - b[0]) / detector.ncols, (b[3] - b[2]) / detector.nrows
            z = float(pc[2])
            return np.array(
                [
                    [x_scale * z, 0.0, (b[0] + 0.5 * x_scale) * z],
                    [0.0, -y_scale * z, (b[3] - 0.5 * y_scale) * z],
                    [0.0, 0.0, z],
                ]
            )

        frame = sample_to_detector_matrix(
            detector.sample_tilt, detector.tilt, detector.azimuthal, detector.twist
        )
        to_index = np.linalg.inv(screen(detector.pc)) @ frame
        maps = [frame.T @ screen(pc) @ to_index for pc in np.asarray(pattern_centers, float)]
        return torch.as_tensor(np.stack(maps), device=self.device, dtype=torch.float32)

    def add_fits(self, phase, g, center_maps=None) -> None:
        """Add the simulations of fitted orientations *g* ``(N, 3, 3)`` to the
        scan's :attr:`ghost`."""
        if self.static_blur:
            return
        if center_maps is not None:
            g = g @ center_maps
        fits = self._project(self.texture, g, phase, self.refine_directions)
        self._fit_sum += _normalized_rows(fits).sum(0)
        self._fit_count += len(g)

    @property
    def ghost(self) -> torch.Tensor | None:
        """The scan's grains as the static background shows them: the normalized
        mean simulation of the orientations fitted so far (:meth:`add_fits`).

        The static background is the mean of the scan's frames, so it carries a
        faint copy of every grain's bands, and a frame divided by it carries
        their negative. A simulation has no such ghost; fitted without it, a
        strain bends to match the ghost of the other grains, differently in each
        grain. :meth:`refine_deformation` leaves each pattern its own amount of
        it instead. A blurred static background (:meth:`restart_static`) has no
        bands, so then there is no ghost.
        """
        if self.static is None or not self._fit_count:
            return None
        return _normalized_rows(self._fit_sum[None])[0]

    def refine_deformation(
        self,
        refine_rows: torch.Tensor,
        phase: torch.Tensor,
        g: torch.Tensor,
        *,
        iterations: int = DEFORMATION_ITERATIONS,
        step: float = DEFORMATION_STEP,
        center_maps: torch.Tensor | None = None,
        ghost: torch.Tensor | None = None,
    ):
        """Each orientation *g* ``(N, 3, 3)`` refined together with its pattern's
        deviatoric strain: ``(g, strain, score)``, *g* the lattice's orientation,
        *strain* ``(N, 3, 3)`` the symmetric, trace-free Biot strain in the sample
        frame (the hydrostatic part is invisible to a projection), *score* the fit's.

        A crystal deformed by ``F`` shows the master's pattern at ``g F^-1 r`` for a
        ray ``r``, so a candidate is the matrix ``g (I - beta)`` with *beta* the
        eight trace-free numbers -- three rotations and five strains -- that the
        fused kernels project like any orientation. Each Gauss-Newton iteration
        projects the current fit and its 16 central differences and solves the 8x8
        normal equations of the normalized pattern, for every pattern at once.
        Fitting the rotations alongside the strains is what sharpens the
        orientations: a strain left out of the model is partly fitted as a turn.

        *g* should already be within a few tenths of a degree (:meth:`refine`).
        *center_maps* (:meth:`center_maps`) give each pattern its own center.
        *ghost* (:attr:`ghost`) is taken out of the patterns and the simulations
        alike, so every pattern may have any amount of it.

        Each iteration moves at most :data:`DEFORMATION_MAX_STEP`, and a fit that
        ends scoring below its seed (:data:`DEFORMATION_SCORE_MARGIN`) is refused:
        its strain is NaN and its *g* and *score* are the seed's.
        """
        basis = _trace_free_basis(self.device)
        n, k = len(g), 1 + 2 * len(basis)
        eye = torch.eye(3, device=self.device)
        offsets = step * torch.cat(
            [torch.zeros(1, 8, device=self.device), torch.eye(8, device=self.device)]
        )
        offsets = torch.cat([offsets, -offsets[1:]])  # (17, 8): the fit, +each, -each
        damping = 1e-3 * torch.eye(8, device=self.device)
        p = torch.zeros(n, 8, device=self.device)
        if ghost is not None:
            refine_rows = _normalized_rows(refine_rows - (refine_rows @ ghost)[:, None] * ghost)

        def on_ghost(simulated):
            """Each simulation's component along the ghost. The simulations are
            never taken off it: every product below is corrected instead, which
            is one pass over them rather than several."""
            if ghost is None:
                return torch.zeros(simulated.shape[:-1], device=self.device)
            return simulated @ ghost

        seed_score = None
        for _ in range(iterations):
            beta = torch.einsum("nkc,cij->nkij", p[:, None] + offsets, basis)
            candidates = g[:, None] @ (eye - beta)
            if center_maps is not None:
                candidates = candidates @ center_maps[:, None]
            simulated = self._project(
                self.texture,
                candidates.reshape(-1, 3, 3),
                phase.repeat_interleave(k),
                self.refine_directions,
            )
            simulated = _normalized_rows(simulated).view(n, k, -1)
            along = on_ghost(simulated)
            fit, fit_along = simulated[:, 0], along[:, :1]
            jacobian = (simulated[:, 1:9] - simulated[:, 9:]) / (2 * step)
            jacobian_along = (along[:, 1:9] - along[:, 9:]) / (2 * step)
            # the pattern's contrast is free: the residual is against scale * fit,
            # both off the ghost (the patterns already are, so only the fit's
            # length and the products with the Jacobian need correcting)
            scale = (refine_rows * fit).sum(-1, keepdim=True) / (1 - fit_along**2)
            if seed_score is None:
                seed_score = scale[:, 0] * (1 - fit_along[:, 0] ** 2).sqrt()
            residual = refine_rows - scale * fit
            normal = torch.einsum("nkp,nlp->nkl", jacobian, jacobian)
            normal = normal - jacobian_along[:, :, None] * jacobian_along[:, None, :]
            normal = normal + damping * normal.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
            gradient = (
                torch.einsum("nkp,np->nk", jacobian, residual) + scale * jacobian_along * fit_along
            ) / scale.clamp_min(_EPS)
            change = torch.linalg.solve(normal, gradient)
            reach = change.abs().amax(-1, keepdim=True)
            p = p + change * (DEFORMATION_MAX_STEP / reach.clamp_min(DEFORMATION_MAX_STEP))
        beta = torch.einsum("nc,cij->nij", p, basis)
        final = g @ (eye - beta)
        if center_maps is not None:
            final = final @ center_maps
        simulated = _normalized_rows(
            self._project(self.texture, final, phase, self.refine_directions)
        )
        score = (simulated * refine_rows).sum(-1) / (1 - on_ghost(simulated) ** 2).sqrt()
        rotation, strain = polar_parts(beta)
        fitted = g @ rotation.transpose(-1, -2)
        if seed_score is not None:
            refused = ~(score >= seed_score - DEFORMATION_SCORE_MARGIN)  # NaN is refused too
            strain = torch.where(refused[:, None, None], torch.nan, strain)
            fitted = torch.where(refused[:, None, None], g, fitted)
            score = torch.where(refused, seed_score, score)
        return fitted, strain, score

    def strain_from_reference(self, frames, reference, *, grid: int = 6, n_passes: int = 2):
        """The strain of each of *frames* ``(N, H, W)`` relative to one real
        *reference* frame ``(H, W)``, by cross-correlating regions of the patterns
        (:func:`~de_ebsd.index.crosscorr.hrebsd`): ``(strain, rotation_vector,
        residual)``, the strain ``(N, 3, 3)`` and lattice rotation ``(N, 3)`` in the
        sample frame and the fit's residual ``(N,)``.

        No master enters, so unlike :meth:`refine_deformation` nothing about how
        well a simulation matches the camera limits it, but it measures only
        differences from the reference, which should be strain-free, and only
        within its grain: the remap holds for lattice rotations of a few degrees.
        On a twin scan with a random 3e-3 strain at a 128 refine, the error relative
        to a reference in each grain was 1.7e-3, against 7.3e-4 absolute from
        :meth:`refine_deformation`; correlating regions against simulations
        instead was 6e-3.
        """
        from de_ebsd.index.crosscorr import hrebsd

        test = torch.cat(
            [
                self._without_background(self._pooled(frames[lo : lo + BATCH]))
                for lo in range(0, len(frames), BATCH)
            ]
        )
        ref = self._without_background(self._pooled(reference))
        detector = replace(self.detector, shape=self.refine_shape)
        result = hrebsd(
            test,
            ref,
            detector,
            roi_size=self.refine_shape[0] // 4,
            grid=grid,
            n_passes=n_passes,
            batch=BATCH,
        )
        _, strain, rotation = result.to_sample(detector)
        vector = torch.stack([rotation[:, 2, 1], rotation[:, 0, 2], rotation[:, 1, 0]], dim=-1)
        return strain, vector, result.residual

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
