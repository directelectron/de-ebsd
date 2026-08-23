"""Fundamental-zone dictionary indexing -- the symmetry-native, fp16 alternative.

:class:`DictionaryIndexer` is a fast, low-precision counterpart to the SOFT
:class:`~torchebsd.index.indexer.SphericalIndexer`. Where the spherical indexer
computes the *full* SO(3) cross-correlation spectrally, the dictionary indexer
samples only the **fundamental zone** of the crystal point group and matches in
the **detector domain** by a single real matmul. That buys three things the SOFT
path cannot:

* **symmetry.** :func:`orix.sampling.get_sample_fundamental` samples the
  fundamental zone of the point group, so a cubic dictionary carries ~24x fewer
  orientations than a full-SO(3) grid of the same angular resolution -- the
  crystal symmetry is used, not ignored.
* **low memory.** The dictionary is an ``(n_dict, H * W)`` matrix and matching is
  ``scores = patterns @ dictionary.T``; there are no ``O(N^3)`` transient SO(3)
  tensors to thrash the CUDA allocator.
* **low precision, high speed.** The dictionary is stored as ``float16`` and the
  match is an fp16 matmul, which on tensor cores accumulates in fp32 (accurate)
  while halving bandwidth and memory. PyTorch has no ``complex32`` matmul/FFT, so
  the spectral path is stuck at fp32+; this real matmul is not.

**Coarse over the zone, then refine -- the intended workflow.** The fundamental-zone
grid is deliberately coarse (a degree or two); the dictionary gets each pattern
into the right zone cell fast, and :func:`torchebsd.index.refine.refine_orientation`
(a *perfect* seed from the coarse cell) sharpens it to sub-degree accuracy through
the differentiable forward model. Use the dictionary indexer for a fast coarse pass
over the fundamental zone plus refinement; use the SOFT
:class:`~torchebsd.index.indexer.SphericalIndexer` when you want the full-SO(3)
spectral correlation (e.g. no reliable symmetry, or a masked-normalized score over
the whole rotation group).

**Dictionary construction is the one-time cost.** Projecting the master at every
fundamental-zone orientation happens **once** per ``(master, detector, resolution)``
in :meth:`~DictionaryIndexer.__init__`; after that, matching is a cheap matmul that
is reused across every scan and every pattern. The dictionary at fp16 costs
``n_dict * H * W * 2`` bytes (a 100k-orientation 60x60 dictionary is ~0.7 GB), so it
is built in chunks to bound the peak projection memory and the per-chunk score
matrix is bounded the same way when matching (see :meth:`~DictionaryIndexer.index`).

**Requires the master to carry the crystal point-group symmetry.** Sampling only the
fundamental zone is correct *because* a real EBSD master pattern is invariant under
the crystal point group: two orientations related by a symmetry operator project to
the same pattern, so one fundamental-zone representative stands in for the whole
orbit. A master without that symmetry (a synthetic asymmetric test field) would need
the full SO(3) grid, not the fundamental zone.
"""

from __future__ import annotations

import numpy as np
import torch
from orix.quaternion import Rotation, Symmetry
from orix.sampling import get_sample_fundamental

_EPS = 1e-6

# Project the master this many orientations at a time when building the dictionary,
# so a large fundamental-zone grid does not materialize its whole (n_dict, P, 3)
# sampling tensor at once.
_PROJ_CHUNK = 4096
# Bound the per-chunk (chunk, n_dict) score matrix to roughly this many elements when
# matching, so the pattern axis is chunked to a size that keeps the scores tensor small
# regardless of how large the dictionary is.
_MAX_SCORE_ELEMS = 64_000_000


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-norm each row so a dot product is a cross-correlation.

    ``x`` is ``(..., P)``; the last axis is centered and scaled to unit L2 norm
    (clamped off zero). ``patterns_n @ dict_n.T`` is then the normalized
    cross-correlation (NCC) of every pattern against every dictionary entry.
    """
    x = x - x.mean(-1, keepdim=True)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(_EPS)


class DictionaryIndexer:
    """Index EBSD patterns against a fundamental-zone dictionary by fp16 matmul.

    Build once -- the dictionary (the master projected at every fundamental-zone
    orientation, normalized, stored fp16) is precomputed in :meth:`__init__` -- then
    call :meth:`index` on a pattern or a stack. Matching is a real matmul, so a whole
    stack is scored in one shot; the fundamental-zone grid is coarse by design, so the
    intended workflow is a fast coarse pass here followed by
    :meth:`refine` (or ``index_scan(refine=True)``). See the module docstring for how
    this compares to the SOFT :class:`~torchebsd.index.indexer.SphericalIndexer`.

    Parameters
    ----------
    master
        A :class:`torchebsd.sim.MasterPattern` -- the continuous ``M(n)`` on the
        sphere, projected onto the detector at every dictionary orientation via
        :meth:`~torchebsd.sim.MasterPattern.project`. It must carry the crystal
        point-group symmetry for the fundamental-zone sampling to be valid (module
        docstring).
    detector
        :class:`torchebsd.sim.detector.EBSDDetector` geometry (fixed pattern center)
        the dictionary is projected onto; every indexed pattern must have this
        detector's ``(nrows, ncols)`` shape.
    point_group
        The crystal point group whose fundamental zone is sampled, as an orix
        :class:`~orix.quaternion.Symmetry`. Resolved as: an explicit ``point_group``
        wins; otherwise it is derived from ``master.metadata["crystal"]`` via
        :func:`torchebsd.index.io.phase_from_crystal` (``.point_group``). Dictionary
        indexing *needs* the symmetry to sample the fundamental zone, so if neither is
        available a clear error is raised.
    resolution
        Target average disorientation (degrees) of the fundamental-zone sampling
        (:func:`orix.sampling.get_sample_fundamental`). This is the coarse
        orientation resolution before refinement; smaller means a denser (larger)
        dictionary. Default ``2.0``.
    device
        Torch device the dictionary is built and matched on (``"cuda"`` recommended --
        the fp16 matmul path is a tensor-core operation). Resolved as: explicit
        ``device`` > ``master.device`` > CPU.
    dtype
        Storage/compute dtype of the dictionary. Default ``torch.float16`` (the fast,
        low-memory path); pass ``torch.float32`` for a CPU-friendly, fully-deterministic
        dictionary. **fp16 matmul is a CUDA (tensor-core) operation** -- PyTorch does
        not implement half-precision matmul on CPU, so use ``torch.float32`` on CPU.
    energy_kev
        Energy bin (keV) the master is projected at (nearest bin; see
        :meth:`~torchebsd.sim.MasterPattern.project`). ``None`` (default) selects the
        highest bin.

    Attributes
    ----------
    point_group : orix.quaternion.Symmetry
        The crystal point group whose fundamental zone was sampled.
    device : torch.device
        Device the dictionary lives on and matching runs on.
    dtype : torch.dtype
        Dictionary storage/compute dtype.
    dictionary_rotations : orix.quaternion.Rotation
        The fundamental-zone orientations, length ``n_dict`` -- the candidate
        orientations :meth:`index` reads its answer off of.
    dictionary : torch.Tensor
        The normalized dictionary, shape ``(n_dict, H * W)``, ``dtype`` on ``device``.
    n_dict : int
        Number of fundamental-zone orientations in the dictionary.

    Notes
    -----
    Dictionary construction (projecting the master at every fundamental-zone
    orientation) is the **one-time cost**, paid here in ``__init__``. It is chunked so
    a large dictionary does not OOM during projection. Matching afterwards is a cheap
    matmul reusable across every scan and pattern indexed with this instance.
    """

    def __init__(
        self,
        master,
        detector,
        point_group=None,
        resolution=2.0,
        device=None,
        dtype=torch.float16,
        energy_kev=None,
    ):
        self.master = master
        self.detector = detector
        self.resolution = float(resolution)
        self.dtype = dtype
        self.energy_kev = energy_kev

        self.point_group = self._resolve_point_group(master, point_group)

        # device: explicit arg > master.device > cpu (mirrors SphericalIndexer)
        if device is not None:
            self.device = torch.device(device)
        else:
            master_device = getattr(master, "device", None)
            self.device = (
                torch.device(master_device) if master_device is not None else torch.device("cpu")
            )

        # sample the fundamental zone -- this is the symmetry reduction
        self.dictionary_rotations = get_sample_fundamental(
            resolution=self.resolution, point_group=self.point_group
        )
        self.n_dict = int(self.dictionary_rotations.size)

        # project the master at every fundamental-zone orientation, normalize each
        # pattern, and store the dictionary as `dtype` on the device -- the one-time
        # cost, chunked so a large dictionary does not blow up projection memory.
        self.dictionary = self._build_dictionary()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_point_group(master, point_group) -> Symmetry:
        """Resolve the point group: explicit Symmetry wins, else from the crystal."""
        if point_group is not None:
            if not isinstance(point_group, Symmetry):
                raise TypeError(
                    "point_group must be an orix Symmetry (e.g. "
                    "orix.quaternion.symmetry.Oh), got "
                    f"{type(point_group).__name__}"
                )
            return point_group

        crystal = None
        meta = getattr(master, "metadata", None)
        if isinstance(meta, dict):
            crystal = meta.get("crystal")
        if crystal is None:
            raise ValueError(
                "no point_group given and master.metadata has no 'crystal' to derive "
                "one from; dictionary indexing needs the crystal symmetry to sample the "
                "fundamental zone -- pass point_group=<orix Symmetry> or a master whose "
                "metadata['crystal'] is a torchebsd Crystal"
            )
        from torchebsd.index.io import phase_from_crystal

        return phase_from_crystal(crystal).point_group

    def _build_dictionary(self) -> torch.Tensor:
        """Project + normalize + store the fp16 (or fp32) dictionary, chunked."""
        # project on the target device: master.project runs on master.data.device, so
        # move the master there once if needed (a device-local copy for projection).
        master = self.master
        if master.device != self.device:
            master = master.to(self.device)

        q = torch.as_tensor(np.asarray(self.dictionary_rotations.data), dtype=torch.float32)
        n_px = int(self.detector.nrows * self.detector.ncols)
        dictionary = torch.empty(self.n_dict, n_px, device=self.device, dtype=self.dtype)
        for i in range(0, self.n_dict, _PROJ_CHUNK):
            sim = master.project(
                self.detector,
                q[i : i + _PROJ_CHUNK],
                energy_kev=self.energy_kev,
                rescale=False,
            )
            sim = sim.reshape(sim.shape[0], -1).to(torch.float32)  # normalize in fp32
            dictionary[i : i + _PROJ_CHUNK] = _normalize_rows(sim).to(self.dtype)
        return dictionary

    def _match_chunk(self, n_patterns: int) -> int:
        """Pattern-axis chunk that keeps the ``(chunk, n_dict)`` score matrix bounded."""
        chunk = max(1, _MAX_SCORE_ELEMS // max(1, self.n_dict))
        return min(chunk, n_patterns)

    # ------------------------------------------------------------------ #
    def index(self, patterns, return_scores=False):
        """Index one pattern or a stack against the fundamental-zone dictionary.

        Normalizes each pattern the same way the dictionary was normalized, then
        matches by a chunked matmul (``scores = patterns @ dictionary.T``); the best
        fundamental-zone orientation is the per-pattern argmax and its NCC is the
        quality. The pattern axis is chunked so the ``(chunk, n_dict)`` score matrix
        stays bounded regardless of dictionary size.

        Parameters
        ----------
        patterns
            A single detector image ``(H, W)`` or a stack ``(Np, H, W)``, as a numpy
            array or a torch tensor (``H = detector.nrows``, ``W = detector.ncols``).
            Plain arrays are accepted directly (no kikuchipy needed).
        return_scores
            When ``True``, additionally return the full ``(Np, n_dict)`` NCC score
            matrix (float32) as a **third** tuple element. Default ``False``.

        Returns
        -------
        rotations : orix.quaternion.Rotation
            The best-matching fundamental-zone orientation of each pattern, length
            ``Np`` (read directly off :attr:`dictionary_rotations`).
        quality : torch.Tensor
            The peak NCC of each pattern, a length-``Np`` real tensor -- a
            correlation-coefficient-like number in roughly ``[-1, 1]``; higher is a
            better match.
        scores : torch.Tensor
            *Only when* ``return_scores``: the full NCC score matrix, shape
            ``(Np, n_dict)`` (float32), whose per-row argmax gives ``rotations``.
        """
        patt = torch.as_tensor(patterns)
        single = patt.ndim == 2
        if single:
            patt = patt[None]
        if patt.ndim != 3:
            raise ValueError(
                f"patterns must be (H, W) or (Np, H, W), got shape {tuple(patt.shape)}"
            )
        if tuple(patt.shape[-2:]) != (self.detector.nrows, self.detector.ncols):
            raise ValueError(
                f"pattern shape {tuple(patt.shape[-2:])} does not match the detector "
                f"{(self.detector.nrows, self.detector.ncols)}"
            )

        n_p = patt.shape[0]
        # normalize in fp32 (patterns are low-bit-depth; keep the precision until the
        # match), then cast to the dictionary dtype for the matmul.
        flat = patt.reshape(n_p, -1).to(device=self.device, dtype=torch.float32)
        exp_n = _normalize_rows(flat).to(self.dtype)

        best_idx = torch.empty(n_p, dtype=torch.long, device=self.device)
        best_score = torch.empty(n_p, dtype=torch.float32, device=self.device)
        scores_out = [] if return_scores else None
        dict_t = self.dictionary.T
        chunk = self._match_chunk(n_p)
        for i in range(0, n_p, chunk):
            # fp16 matmul accumulates in fp32 on tensor cores; read the peak in fp32
            scores = (exp_n[i : i + chunk] @ dict_t).float()  # (chunk, n_dict)
            bscore, bidx = scores.max(dim=1)
            best_score[i : i + chunk] = bscore
            best_idx[i : i + chunk] = bidx
            if return_scores:
                scores_out.append(scores)

        rot_data = np.asarray(self.dictionary_rotations.data)[best_idx.cpu().numpy()]
        rotations = Rotation(rot_data)
        if return_scores:
            return rotations, best_score, torch.cat(scores_out, dim=0)
        return rotations, best_score

    def refine(self, patterns, rotations=None, **kw):
        """Gradient-refine the coarse dictionary orientations to sub-grid accuracy.

        Thin wrapper over :func:`torchebsd.index.refine.refine_orientation` that seeds
        from this indexer's own master/detector. The fundamental-zone grid is coarse by
        design, so refining the coarse dictionary orientation is the intended way to
        reach sub-degree accuracy (the coarse cell is a *perfect* seed). If
        ``rotations`` is ``None`` the seeds are obtained by calling :meth:`index` first,
        so a single call goes from raw patterns straight to sharpened orientations.

        Parameters
        ----------
        patterns
            A single detector image ``(H, W)`` or a stack ``(Np, H, W)`` (numpy or
            torch).
        rotations
            Seed orientations, an :class:`orix.quaternion.Rotation` of length ``Np``. If
            ``None`` (default), :meth:`index` is run on ``patterns`` to obtain them.
        **kw
            Forwarded to :func:`~torchebsd.index.refine.refine_orientation` (``pc``,
            ``refine_pc``, ``n_iters``, ``lr``, ``keep_best``, ...). ``device`` defaults
            to this indexer's device.

        Returns
        -------
        tuple
            ``(rotation, score)`` -- the refined orix ``Rotation`` and final NCC of each
            pattern (plus ``pc`` / ``history`` when their flags are set), exactly as
            :func:`~torchebsd.index.refine.refine_orientation` returns them.
        """
        from torchebsd.index.refine import refine_orientation

        if rotations is None:
            rotations, _ = self.index(patterns)
        kw.setdefault("device", self.device)
        return refine_orientation(patterns, self.master, self.detector, rotations, **kw)

    def index_scan(
        self,
        patterns,
        shape=None,
        step=1.0,
        phase=None,
        batch_size=None,
        refine=False,
        refine_iters=60,
    ):
        """Index a whole scan and assemble the result into an orix ``CrystalMap``.

        Wraps :meth:`index` over an entire EBSD scan: it processes the patterns in
        mini-batches (to bound the per-batch device memory), concatenates the per-batch
        orientations and quality, and hands them to
        :func:`torchebsd.index.io.to_crystal_map`, which lays them out on the scan grid
        as a single-phase :class:`~orix.crystal_map.CrystalMap`. Mirrors
        :meth:`~torchebsd.index.indexer.SphericalIndexer.index_scan`.

        Parameters
        ----------
        patterns
            The scan patterns, numpy or torch, in either layout:

            * ``(rows, cols, H, W)`` -- the grid is read off directly and ``shape`` is
              inferred (any explicit ``shape`` must then match);
            * ``(N, H, W)`` -- a flat row-major stack, in which case ``shape`` **must**
              be given so the ``N = rows * cols`` points can be reshaped onto the grid.
        shape
            ``(rows, cols)`` of the scan. Required for ``(N, H, W)`` input; optional
            (and cross-checked) for ``(rows, cols, H, W)`` input.
        step
            Scan step size (same for both axes), written onto the map's ``x``/``y``.
        phase
            The map's crystal symmetry, as an :class:`orix.crystal_map.Phase` or a
            torchebsd :class:`~torchebsd.sim.Crystal` (converted via
            :func:`torchebsd.index.io.phase_from_crystal`). If ``None``, it is derived
            from ``master.metadata["crystal"]``; if neither is available a clear error
            tells the caller to pass ``phase``.
        batch_size
            Number of patterns indexed per mini-batch (default ``256``). Bounds the
            device memory of the moved pattern chunk (and of refinement when
            ``refine``); the result is identical to a single batch.
        refine
            If ``True``, each mini-batch's coarse dictionary orientations are
            gradient-refined with :meth:`refine` before assembly, and
            ``prop["quality"]`` becomes the refined NCC. The fundamental-zone grid is
            coarse, so refinement to sub-degree is the intended finish. Default
            ``False``.
        refine_iters
            Adam iterations for the per-batch refinement when ``refine`` is ``True``
            (see :func:`~torchebsd.index.refine.refine_orientation`).

        Returns
        -------
        orix.crystal_map.CrystalMap
            One orientation + ``prop["quality"]`` per scan point, single phase
            (``phase_id`` all zero), on the ``(rows, cols)`` grid. With ``refine=True``
            the orientations are gradient-sharpened and the quality is the refined NCC.
        """
        from torchebsd.index.io import phase_from_crystal, to_crystal_map

        # keep the whole stack on CPU (view-only reshape); only each mini-batch moves to
        # self.device inside the loop, so a large scan never lands wholesale on the GPU.
        patt = torch.as_tensor(patterns, dtype=torch.float32)
        if patt.ndim == 4:  # (rows, cols, H, W)
            rows, cols = int(patt.shape[0]), int(patt.shape[1])
            if shape is not None and (int(shape[0]), int(shape[1])) != (rows, cols):
                raise ValueError(
                    f"shape {tuple(shape)} does not match the (rows, cols, H, W) "
                    f"pattern grid {(rows, cols)}"
                )
            patt = patt.reshape(rows * cols, patt.shape[2], patt.shape[3])
        elif patt.ndim == 3:  # (N, H, W) -- shape required
            if shape is None:
                raise ValueError(
                    "shape=(rows, cols) is required for (N, H, W) patterns "
                    "(pass (rows, cols, H, W) to infer it instead)"
                )
            rows, cols = int(shape[0]), int(shape[1])
            if rows * cols != patt.shape[0]:
                raise ValueError(
                    f"shape {(rows, cols)} implies {rows * cols} patterns but got "
                    f"{patt.shape[0]}"
                )
        else:
            raise ValueError(
                "patterns must be (rows, cols, H, W) or (N, H, W), got shape "
                f"{tuple(patt.shape)}"
            )

        n = rows * cols
        bs = int(batch_size) if batch_size is not None else 256
        if bs < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        rot_chunks = []
        qual_chunks = []
        for start in range(0, n, bs):
            chunk = patt[start : start + bs].to(self.device)
            rot_c, qual_c = self.index(chunk)
            if refine:
                rot_c, qual_c = self.refine(chunk, rotations=rot_c, n_iters=refine_iters)
            rot_chunks.append(np.asarray(rot_c.data))
            qual_chunks.append(qual_c.detach().cpu())
        rotations = Rotation(np.concatenate(rot_chunks, axis=0))
        quality = torch.cat(qual_chunks)

        # resolve the phase: explicit arg (Phase or Crystal) wins; else derive from the
        # master's metadata; else a clear error.
        if phase is None:
            crystal = None
            meta = getattr(self.master, "metadata", None)
            if isinstance(meta, dict):
                crystal = meta.get("crystal")
            if crystal is None:
                raise ValueError(
                    "no phase given and master.metadata has no 'crystal' to derive one "
                    "from; pass phase=<orix Phase or torchebsd Crystal> to index_scan "
                    "(the Phase gives the CrystalMap its crystal symmetry)"
                )
            phase = phase_from_crystal(crystal)

        return to_crystal_map(rotations, quality, (rows, cols), phase, step=step)
