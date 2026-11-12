"""
indexing.py — EBSD indexing across several phases, live and through a scan.

SpyDE's EBSD wizard (``spyde/actions/ebsd_action.py`` at feat/phase-viewer
41ceec1b) builds ONE phase's dictionary — orientations sampled over the
phase's fundamental zone, simulated with its reflectors, high-passed exactly
as the experimental patterns are corrected — and matches patterns against it.
This is that, for several phases at once: every phase has its own dictionary,
a pattern is matched against each, and the best score wins (its phase is the
pattern's phase). The steps are SpyDE's functions (the copies in
``pyxem_base.ebsd``), in the wizard's order:

* :class:`PhaseDictionary` — the wizard's Build Dictionary, for one phase;
* :meth:`Indexer.prepared` — its background correction, on a frame averaged
  down to the dictionary's detector size;
* :meth:`Indexer.match` — its live match (one pattern), for every phase;
* :meth:`Indexer.index` — its whole-field index, for a stack of frames.

Two things differ from the wizard, both measured against de-twin's ground
truth (a duplex steel scan of its camera frames, 1024 patterns, 33 grains):

* **Orientations are reported in the scan's frame** — x along a row, y down
  the columns, z into the sample — not in the frame the patterns are
  simulated in, which is kikuchipy's (``bands.sample_to_detector``). The two
  differ by 90° about the surface normal: without the turn no pattern was
  within 5° of the truth. An IPF-Z color cannot show this (a turn about z
  leaves the z direction where it was), so the map looked right while every
  orientation was 90° off.
* **The detector's static background** is the mean of the scan's own frames
  (:meth:`Indexer.observe`), kept as the scan runs. The per-pattern blur
  alone leaves the static profile in, and the phases then confuse: 22% of
  those patterns were given the wrong phase, none once it is subtracted.

On a GPU with de_ebsd's live indexer, :class:`DynamicalIndexer` takes the
place of these kinematic dictionaries, with the same surface: the
dictionary is projected from each phase's dynamical master pattern, and each
orientation refined against the master to a tenth of a degree or so, where
a kinematic dictionary entry is only as close as its step.

A frame is processed where it arrives and never leaves the device: a CuPy
frame from the camera is handed to torch without a copy, and averaging,
correction, normalization and matching all run there. Only each pattern's
phase, orientation index and score come back. SpyDE's functions take and
return NumPy, so the correction here is theirs re-run on device tensors
(``preprocess._blur``, ``indexing._normalise``), not a call to them.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

from ..device_lock import accelerator_lock
from ..ebsd.bands import cubic_reflectors, reflectors_from_phase
from ..ebsd.indexing import (
    SinglePatternIndexer,
    _normalise,
    sample_orientations,
    simulate_dictionary,
)
from ..ebsd.preprocess import _blur
from .phases import Phase

#: The wizard's defaults (``ebsd_action.DEFAULTS``), apart from the step
#: and the background. At 4° and the per-pattern background only, 60% of
#: de-twin's patterns were within 5° of the truth; at 3° with the static
#: background as well, 99.8%. A 3° dictionary is 2.5 times the size: 20 s to
#: build for two phases, 0.1 ms a pattern to match.
STEP_DEG = 3.0
MIN_DSPACING = 0.7
BACKGROUND = "both"
BACKGROUND_SIGMA = 8.0
N_BANDS = 20
#: Frames prepared and matched at a time: bounds the device memory a batch
#: takes (256 frames of 1024² are 1 GiB in float32).
CHUNK = 256
#: Frames a static background is averaged over before it is used: fewer
#: still carry the bands of the few grains they came from.
MIN_FRAMES_FOR_STATIC = 32


def to_scan_frame(euler) -> np.ndarray:
    """Bunge Euler angles ``(N, 3)`` in radians, from the frame patterns are
    simulated in (kikuchipy's) to the scan's: turned 90° about the surface
    normal (see the module docstring)."""
    from orix.quaternion import Rotation

    turn = Rotation.from_axes_angles([0, 0, 1], -90, degrees=True)
    return (Rotation.from_euler(np.asarray(euler, float)) * turn).to_euler()


@dataclass(frozen=True)
class Geometry:
    """An EBSD detector: its shape (the dictionary's, after binning), its
    pattern center, and the beam energy the reflectors are simulated at.
    SpyDE's ``microscope.FIELDS["SEM"]`` names and defaults."""

    shape: tuple[int, int] = (60, 60)
    pc_x: float = 0.5
    pc_y: float = 0.5
    pc_z: float = 0.55
    beam_energy: float = 20.0
    #: How the detector sits relative to the sample, in degrees: the
    #: patterns and the bands drawn on them are simulated in this frame.
    sample_tilt: float = 70.0
    detector_tilt: float = 0.0
    azimuthal: float = 0.0
    twist: float = 0.0

    @property
    def pc(self) -> tuple[float, float, float]:
        return (self.pc_x, self.pc_y, self.pc_z)

    @property
    def angles(self) -> dict[str, float]:
        """The detector's angles, as SpyDE's band functions take them."""
        return {
            "sample_tilt": self.sample_tilt,
            "detector_tilt": self.detector_tilt,
            "azimuthal": self.azimuthal,
            "twist": self.twist,
        }


def _reflectors(phase: Phase, geometry: Geometry):
    """The bands a phase's fit is drawn with, which its kinematic dictionary is
    simulated with too."""
    try:
        return reflectors_from_phase(
            phase.structure, min_dspacing=MIN_DSPACING, voltage_kv=geometry.beam_energy
        )
    except Exception:  # noqa: BLE001 — SpyDE's fallback
        return cubic_reflectors()


class PhaseDictionary:
    """One phase's dictionary, normalised and resident for live matching."""

    def __init__(
        self,
        phase: Phase,
        geometry: Geometry,
        *,
        step_deg: float = STEP_DEG,
        background_sigma: float | None = BACKGROUND_SIGMA,
        progress=None,
    ) -> None:
        self.phase = phase
        self.geometry = geometry
        self.step_deg = float(step_deg)
        self.reflectors = _reflectors(phase, geometry)
        #: The orientations simulated, in the simulation's frame (the one
        #: bands are drawn in), and the same in the scan's frame.
        self.euler = sample_orientations(step_deg, point_group=phase.point_group)
        self.orientations = to_scan_frame(self.euler)
        patterns = simulate_dictionary(
            self.euler,
            geometry.shape,
            geometry.pc,
            reflectors=self.reflectors,
            background_sigma=background_sigma,
            progress=progress,
            **geometry.angles,
        )
        self.indexer = SinglePatternIndexer(patterns, self.euler)

    def __len__(self) -> int:
        return len(self.euler)


@dataclass(frozen=True)
class Match:
    """The best fit of one pattern: its phase, its orientation (in the
    simulation's frame, to draw its bands with, and in the scan's) and
    score."""

    phase: int
    euler: np.ndarray
    orientation: np.ndarray
    score: float


@dataclass(frozen=True)
class Indexed:
    """A stack of frames indexed, a row per frame: the phase, the dictionary
    entry (of that phase) the orientation came from — anything worked out
    once per entry, such as its IPF color, stands for the orientation — the
    score, and the orientation as a matrix ``(N, 3, 3)`` in the simulation's
    frame (:func:`scan_euler` turns it into the scan's)."""

    phase: np.ndarray
    entry: np.ndarray
    score: np.ndarray
    rotation: np.ndarray
    #: The deviatoric strain ``(N, 3, 3)`` in the simulation's frame
    #: (:func:`scan_tensor` turns it into the scan's), when it was measured.
    strain: np.ndarray | None = None


#: The turn :func:`to_scan_frame` makes, acting on vectors: a direction ``v`` in
#: the simulation's frame is ``SCAN_TURN @ v`` in the scan's.
SCAN_TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def scan_euler(rotations) -> np.ndarray:
    """Orientation matrices ``(N, 3, 3)`` (:attr:`Indexed.rotation`) as Bunge
    Euler angles in the scan's frame."""
    from orix.quaternion import Rotation

    return to_scan_frame(Rotation.from_matrix(np.asarray(rotations, float)).to_euler())


def scan_tensor(tensors) -> np.ndarray:
    """Tensors ``(..., 3, 3)`` in the simulation's frame, such as
    :attr:`Indexed.strain`, in the scan's."""
    return np.einsum("ij,...jk,lk->...il", SCAN_TURN, np.asarray(tensors), SCAN_TURN)


def misorientation_deg(rotations, reference, symmetry) -> np.ndarray:
    """The angle ``(N,)`` from each of *rotations* ``(N, 3, 3)`` to *reference*
    ``(3, 3)``, both in the simulation's frame, over the point group's
    *symmetry* matrices ``(K, 3, 3)``."""
    turn = np.asarray(rotations) @ np.asarray(reference).T
    # trace(S M) for every operator S at once; the group holds each S's transpose
    traces = np.einsum("kij,nij->nk", np.asarray(symmetry), turn).max(-1)
    return np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))


class Indexer:
    """Several phases' dictionaries, and the correction both sides share."""

    def __init__(
        self,
        dictionaries: list[PhaseDictionary],
        *,
        background: str = BACKGROUND,
        background_sigma: float = BACKGROUND_SIGMA,
    ) -> None:
        if not dictionaries:
            raise ValueError("indexing needs at least one phase")
        self.dictionaries = dictionaries
        self.background = background
        self.background_sigma = float(background_sigma)
        #: The static background in use (dictionary-sized, on the device),
        #: and the running sum the next one is being made from.
        self.static = None
        self._static_sum, self._static_count = None, 0
        self._static_blur = None
        #: Each dictionary's orientations as matrices, made the first time a
        #: stack is indexed.
        self._matrix_cache = None

    @property
    def geometry(self) -> Geometry:
        return self.dictionaries[0].geometry

    @property
    def device(self):
        return self.dictionaries[0].indexer.device

    def restart_static(self, frames=None, *, blur: float | None = None) -> None:
        """Start averaging a new static background (a new scan), from *frames*
        (a search's over the same field) if given, and blurred by *blur* (a
        share of the pattern's width) for a single crystal, as
        :meth:`DynamicalIndexer.restart_static`. The one in use stays until
        the new one has `MIN_FRAMES_FOR_STATIC` frames."""
        self._static_sum, self._static_count = None, 0
        self._static_blur = blur
        if frames is not None:
            self.observe(frames)

    def observe(self, frames) -> None:
        """Add *frames* to the static background being averaged."""
        import torch
        import torch.nn.functional as F

        with accelerator_lock(self.device), torch.no_grad():
            t = _on_device(frames, self.device)
            t = t[None] if t.ndim == 2 else t
            pooled = F.adaptive_avg_pool2d(t[:, None], self.geometry.shape)[:, 0].sum(0)
            total = pooled if self._static_sum is None else self._static_sum + pooled
            self._static_sum, self._static_count = total, self._static_count + len(t)
            if self._static_count >= MIN_FRAMES_FOR_STATIC:
                static = total / self._static_count
                if self._static_blur:
                    sigma = self._static_blur * static.shape[-1]
                    static = _blur(static[None], sigma, t.device, t.dtype)[0]
                self.static = static

    def prepared(self, frames):
        """Frames (``(H, W)`` or ``(N, H, W)``; CuPy, NumPy or torch) as the
        dictionary's patterns, ``(N, rows, cols)`` on its device: each
        averaged over the WHOLE frame down to the dictionary's shape — the
        pattern center refers to the whole sensor, so a crop to a multiple of
        it would move the pattern — then background-corrected as the
        dictionary was (``EbsdWizard.correct``), with the static background
        once there is one."""
        import torch.nn.functional as F

        t = _on_device(frames, self.device)
        t = t[None] if t.ndim == 2 else t
        t = F.adaptive_avg_pool2d(t[:, None], self.geometry.shape)[:, 0]
        static = self.static
        if self.background in ("static", "both") and static is not None:
            t = t - static
        if self.background in ("dynamic", "both"):
            t = t - _blur(t, self.background_sigma, t.device, t.dtype)
        return t

    def corrected(self, frame) -> np.ndarray:
        """One frame background-corrected at its own resolution, on the host
        — to be looked at; it is matched at the dictionary's. The dynamic
        background is the dictionary-sized one scaled back up: it is smooth
        over many dictionary pixels, so nothing is lost and the blur stays
        cheap at any frame size."""
        import torch
        import torch.nn.functional as F

        with accelerator_lock(self.device), torch.no_grad():
            t = _on_device(frame, self.device)
            static = self.static
            if self.background in ("static", "both") and static is not None:
                static = F.interpolate(
                    static[None, None], size=t.shape, mode="bilinear", align_corners=False
                )
                t = t - static[0, 0]
            if self.background in ("dynamic", "both"):
                small = F.adaptive_avg_pool2d(t[None, None], self.geometry.shape)[0]
                background = _blur(small, self.background_sigma, t.device, t.dtype)
                background = F.interpolate(
                    background[None], size=t.shape, mode="bilinear", align_corners=False
                )
                t = t - background[0, 0]
            return t.cpu().numpy()

    def match(self, frame) -> Match:
        """The best fit of one frame, over every phase."""
        phase, index, score = self.best(frame)
        number, at = int(phase[0]), int(index[0])
        dictionary = self.dictionaries[number]
        return Match(number, dictionary.euler[at], dictionary.orientations[at], float(score[0]))

    def index(self, frames) -> Indexed:
        """The best fit of every frame of a stack."""
        phase, index, score = self.best(frames)
        rotation = np.zeros((len(phase), 3, 3), np.float32)
        for number in range(len(self.dictionaries)):
            rows = phase == number
            rotation[rows] = self._matrices(number)[index[rows]]
        return Indexed(phase, index, score, rotation)

    def entry_orientations(self, number: int) -> np.ndarray:
        """Phase *number*'s dictionary orientations, in the scan's frame."""
        return self.dictionaries[number].orientations

    def _matrices(self, number: int) -> np.ndarray:
        if self._matrix_cache is None:
            from orix.quaternion import Rotation

            self._matrix_cache = [
                Rotation.from_euler(d.euler).to_matrix().astype(np.float32)
                for d in self.dictionaries
            ]
        return self._matrix_cache[number]

    def orientations(self, phase, index) -> np.ndarray:
        """The orientations, in the scan's frame, of dictionary entries
        *index* of phases *phase* (as :meth:`best` returns them)."""
        euler = np.zeros((len(phase), 3))
        for number, dictionary in enumerate(self.dictionaries):
            rows = phase == number
            euler[rows] = dictionary.orientations[index[rows]]
        return euler

    def best(self, frames):
        """Each frame's best phase, orientation index and score, matched on
        the device ``CHUNK`` frames at a time; the three come back."""
        import torch

        frames = frames[None] if frames.ndim == 2 else frames
        out = []
        with accelerator_lock(self.device), torch.no_grad():
            for lo in range(0, len(frames), CHUNK):
                patterns = self.prepared(frames[lo : lo + CHUNK])
                patterns = _normalise(patterns.reshape(len(patterns), -1))
                score = torch.full((len(patterns),), -torch.inf, device=patterns.device)
                phase = torch.zeros(len(patterns), dtype=torch.int16, device=patterns.device)
                index = torch.zeros(len(patterns), dtype=torch.int64, device=patterns.device)
                for number, dictionary in enumerate(self.dictionaries):
                    normalised = dictionary.indexer.normalised
                    best, at = (patterns.to(normalised.dtype) @ normalised.T).max(1)
                    better = best.float() > score
                    score = torch.where(better, best.float(), score)
                    phase = torch.where(better, number, phase)
                    index = torch.where(better, at, index)
                out.append((phase.cpu().numpy(), index.cpu().numpy(), score.cpu().numpy()))
        return tuple(np.concatenate(column) for column in zip(*out))


# ── Against dynamical master patterns, on a GPU ──
#: The live indexer's comparison sizes, on the frame's long side: the search
#: against the dictionary, and the refine against the master. de_ebsd's
#: defaults, which on de-twin's dynamical scans gave 0.08-0.14° median error
#: at 10-12 thousand patterns a second on a Titan X.
SEARCH_SIDE = 32
REFINE_SIDE = 128
#: The refine's steps (degrees) when the orientations are fitted after it
#: (de_ebsd's ``index(fit=True)``): the fit does what its finer levels did. On a
#: real Zenith Cu scan, with de_ebsd's five-pass fit, stopping at 1 degree took
#: indexing from 28,800 to 33,600 patterns a second and the single crystal's
#: neighbours from 0.19 to 0.17 degrees at the 90th percentile; no orientation
#: moved more than 5 degrees.
FITTED_REFINE_STEPS_DEG = (2.0, 1.0)
#: Frames the pattern center is refined from: an 8 x 8 grid over a search.
#: Fewer than `MIN_FRAMES_FOR_STATIC` (a 5 x 5 grid) leave no static
#: background, and half of de-twin's patterns were then the wrong phase.
CENTER_GRID = 8


def _fused_fit(device) -> bool:
    """Whether de_ebsd fits the orientations on its fused kernels on *device*:
    CUDA with CuPy, and a de_ebsd with the scheduled fit."""
    try:
        from de_ebsd.index import _kernels
        from de_ebsd.index.live import deformation
    except ImportError:
        return False
    return bool(getattr(deformation, "FIT_SCHEDULE", ())) and _kernels.available(device)


def dynamical_indexing_available(device) -> bool:
    """Whether patterns can be indexed against dynamical master patterns:
    de_ebsd has its live indexer and *device* is a GPU. On a CPU a master
    takes minutes to simulate and the refine is far slower than a camera, so
    the kinematic dictionaries are used there instead."""
    import importlib.util

    if str(device).split(":")[0] != "cuda":
        return False
    try:
        return importlib.util.find_spec("de_ebsd.index.live") is not None
    except ModuleNotFoundError:
        return False


def comparison_shape(side: int, frame_shape) -> tuple[int, int]:
    """*side* pixels on the frame's long side, in the frame's aspect."""
    rows, cols = int(frame_shape[0]), int(frame_shape[1])
    scale = side / max(rows, cols)
    return max(8, round(rows * scale)), max(8, round(cols * scale))


def pooled_for_center(frames, frame_shape, device) -> np.ndarray:
    """*frames* averaged down to the refine's size, on the host: what the
    center refine reads of them, kept small. Averaged over the whole frame
    (adaptively, never cropped), as the indexer does, so the center does not
    move."""
    import torch
    import torch.nn.functional as F

    with accelerator_lock(device), torch.no_grad():
        t = _on_device(frames, device)
        t = t[None] if t.ndim == 2 else t
        shape = comparison_shape(REFINE_SIDE, frame_shape)
        return F.adaptive_avg_pool2d(t[:, None], shape)[:, 0].cpu().numpy()


@dataclass(frozen=True)
class PhaseModel:
    """What drawing a phase's fit takes: the phase, the detector its bands are
    drawn on, and its reflectors."""

    phase: Phase
    geometry: Geometry
    reflectors: object


def _detector(geometry: Geometry, shape):
    import de_ebsd

    return de_ebsd.EBSDDetector(
        shape=tuple(int(v) for v in shape),
        pattern_center=geometry.pc,
        sample_tilt=geometry.sample_tilt,
        tilt=geometry.detector_tilt,
        azimuthal=geometry.azimuthal,
        twist=geometry.twist,
    )


class DynamicalIndexer:
    """Several phases indexed against their dynamical master patterns, by
    de_ebsd's ``LiveIndexer``: a search of a `STEP_DEG` dictionary projected
    from the masters, then each orientation refined against the master
    itself. :class:`Indexer`'s surface, as the workflow uses it; the frames
    never leave the device, and only each pattern's phase, entry, score and
    orientation come back.

    *masters* are de_ebsd ``MasterPattern`` s, one per phase, at one ``npx``;
    *geometry*'s shape is the frame's aspect the bands are drawn in, the
    comparison sizes are `SEARCH_SIDE` and `REFINE_SIDE`."""

    def __init__(
        self,
        phases: list[Phase],
        masters,
        geometry: Geometry,
        *,
        step_deg: float = STEP_DEG,
        device="cuda",
        lock=None,
    ) -> None:
        """*lock* is held by every call into the live indexer (see
        :func:`de_ebsd.core.device.resolve_lock`); by default the indexer has its
        own. Its calls share one device workspace, so two threads indexing at
        once without a lock overwrite each other's rows and score garbage."""
        import threading

        from de_ebsd.index.live import LiveIndexer

        if not phases:
            raise ValueError("indexing needs at least one phase")
        self.dictionaries = [PhaseModel(p, geometry, _reflectors(p, geometry)) for p in phases]
        self.device = device
        #: Whether de_ebsd can fit the orientations after its refine (0.3 on).
        self.fits = "fit" in inspect.signature(LiveIndexer.index).parameters
        # Only de_ebsd's fused fit (CUDA with CuPy) starts from 1 degree: its torch
        # fit, with no pass schedule, reached 0.53 degrees from there on a real scan
        # where it reaches 0.21 from 0.5.
        steps = {"refine_steps_deg": FITTED_REFINE_STEPS_DEG} if self.fits else {}
        if self.fits and not _fused_fit(device):
            steps = {"refine_steps_deg": (*FITTED_REFINE_STEPS_DEG, 0.5)}
        with accelerator_lock(device):
            self.live = LiveIndexer(
                [m.to(device) for m in masters],
                [p.point_group for p in phases],
                _detector(geometry, geometry.shape),
                resolution=step_deg,
                search_shape=comparison_shape(SEARCH_SIDE, geometry.shape),
                refine_shape=comparison_shape(REFINE_SIDE, geometry.shape),
                device=device,
                lock=threading.RLock() if lock is None else lock,
                **steps,
            )
        #: Where each phase's entries start in the live indexer's dictionary.
        self._first = np.cumsum([0, *self.live.sizes[:-1]])

    @property
    def geometry(self) -> Geometry:
        return self.dictionaries[0].geometry

    def set_pattern_center(self, pc) -> None:
        """Index at *pc* from now on: the dictionary is projected again, in
        milliseconds, rather than built again."""
        from dataclasses import replace

        geometry = replace(self.geometry, pc_x=pc[0], pc_y=pc[1], pc_z=pc[2])
        self.dictionaries = [replace(d, geometry=geometry) for d in self.dictionaries]
        with accelerator_lock(self.device):
            self.live.set_pattern_center(geometry.pc)

    def restart_static(self, frames=None, *, blur: float | None = None) -> None:
        """Forget the last scan's static background: the next one starts from
        *frames* (a search's over the same field) if given, and is blurred by
        *blur* (a share of the pattern's width) for a single crystal, whose
        frames' mean still holds its bands (de_ebsd's
        ``LiveIndexer.restart_static``)."""
        import torch

        with accelerator_lock(self.device), torch.no_grad():
            t = None if frames is None else _on_device(frames, self.device)
            self.live.restart_static(t, blur=blur)

    def observe(self, frames) -> None:
        import torch

        with accelerator_lock(self.device), torch.no_grad():
            self.live.observe(_on_device(frames, self.device))

    def corrected(self, frame) -> np.ndarray:
        """One frame background-corrected at its own resolution, on the host,
        to be looked at."""
        import torch

        with accelerator_lock(self.device), torch.no_grad():
            return self.live.corrected(_on_device(frame, self.device))[0].cpu().numpy()

    def index(self, frames, *, strain: bool = False) -> Indexed:
        """*frames* indexed; with *strain*, each orientation is refined together
        with its pattern's deviatoric strain (de_ebsd's ``refine_deformation``),
        several times slower than indexing alone. Without, each is fitted
        without the strains where de_ebsd can (`fits`): on a real Zenith Cu
        scan neighbouring orientations in a single crystal then differed by
        0.20° at the 90th percentile, against 1.70° from its refine alone and
        0.35° with the strains."""
        import torch

        frames = frames[None] if frames.ndim == 2 else frames
        with accelerator_lock(self.device), torch.no_grad():
            frames = _on_device(frames, self.device)
            measured = None
            if strain:
                phase, g, measured, score, entry = self.live.index_with_strain(frames)
                measured = measured.cpu().numpy()
            else:
                fit = {"fit": True} if self.fits else {}
                phase, g, score, entry = self.live.index(frames, **fit)
            phase = phase.cpu().numpy()
            return Indexed(
                phase,
                entry.cpu().numpy() - self._first[phase],
                score.cpu().numpy(),
                g.cpu().numpy(),
                measured,
            )

    def strain_from_reference(self, frames, reference) -> np.ndarray:
        """The deviatoric strain ``(N, 3, 3)`` of *frames*, in the simulation's
        frame, relative to one real *reference* frame: correlated against it
        rather than against the master, so it holds only within the
        reference's grain (de_ebsd's ``strain_from_reference``)."""
        import torch

        frames = frames[None] if frames.ndim == 2 else frames
        with accelerator_lock(self.device), torch.no_grad():
            strain, _, _ = self.live.strain_from_reference(
                _on_device(frames, self.device), _on_device(reference, self.device)
            )
            return strain.cpu().numpy()

    def symmetry(self, number: int) -> np.ndarray:
        """Phase *number*'s point-group operators as matrices ``(K, 3, 3)``."""
        return self.dictionaries[number].phase.point_group.to_matrix()

    def match(self, frame) -> Match:
        from orix.quaternion import Rotation

        found = self.index(frame)
        euler = Rotation.from_matrix(found.rotation[:1].astype(float)).to_euler()
        return Match(int(found.phase[0]), euler[0], to_scan_frame(euler)[0], float(found.score[0]))

    def entry_orientations(self, number: int) -> np.ndarray:
        first = int(self._first[number])
        rotations = self.live.rotations[first : first + self.live.sizes[number]]
        return scan_euler(rotations.cpu().numpy())


def refine_pattern_center(
    frames, masters, phases, geometry: Geometry, *, device, blur: float | None = None
) -> tuple:
    """The pattern center *frames* (``(N, H, W)``, several grains' patterns,
    at least `MIN_FRAMES_FOR_STATIC`) fit best, from *geometry*'s: de_ebsd's
    ``LiveIndexer.refine_pattern_center``, against every phase's master.

    It only polishes — it moves the center a hundredth at most — so it starts
    from a center already found, such as ``calibrate.search_pattern_centre``'s.
    On de-twin's dynamical scans that search lands within about 0.006, and
    0.006 off is 0.6° of median orientation error; this brought starts 0.004
    to 0.01 off back within 0.001, in half a second. *blur* is the static
    background's, for a single crystal (:meth:`DynamicalIndexer.restart_static`)."""
    import torch

    indexer = DynamicalIndexer(phases, masters, geometry, device=device)
    with accelerator_lock(device), torch.no_grad():
        t = _on_device(frames, device)
        indexer.live.restart_static(t, blur=blur)
        return indexer.live.refine_pattern_center(t)


def _on_device(frames, device):
    """*frames* as float32 on *device*. A CuPy array is handed over without a
    copy (DLPack) and converted there; a NumPy one is uploaded."""
    import torch

    if hasattr(frames, "__dlpack__") and not isinstance(frames, np.ndarray):
        t = torch.from_dlpack(frames)
    else:
        t = torch.as_tensor(np.asarray(frames))
    return t.to(device=device, dtype=torch.float32)
