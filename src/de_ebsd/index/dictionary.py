"""Brute-force dictionary (template) indexing.

Projects a fundamental-zone orientation sample through a master pattern (the
dynamical :class:`~de_ebsd.sim.MasterPattern` or the kinematic
:class:`~de_ebsd.sim.KinematicMaster`) into simulated patterns, or takes a
dictionary already simulated, and ranks each experimental pattern's normalized
cross-correlation against every entry. Once both sides are zero-mean and unit-norm
that is one matrix product and a top-k. The conventional baseline that
:class:`~de_ebsd.index.spherical.indexer.SphericalIndexer` replaces.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F
from orix.quaternion import Rotation, Symmetry
from orix.sampling import get_sample_fundamental

from de_ebsd.core.device import holds_lock, resolve_device, resolve_lock
from de_ebsd.index.preprocess import normalize_rows, remove_dynamic_background

__all__ = ["DictionaryIndexer", "sample_orientations"]

# orientations projected per chunk when building the dictionary, so a large
# fundamental zone never materializes its whole (n_dict, P, 3) sampling tensor
_PROJ_CHUNK = 4096
# cap on the per-chunk (chunk, n_dict) score matrix, which bounds match memory
# regardless of dictionary size
_MAX_SCORE_ELEMS = 64_000_000


def sample_orientations(
    resolution=3.0, point_group=None, phi1=(0.0, 360.0), Phi=(0.0, 90.0), phi2=(0.0, 90.0)
) -> Rotation:
    """Orientations to build a dictionary from, as an orix ``Rotation``.

    With a *point_group* (an orix ``Symmetry``), orix's uniform sampling of its
    fundamental zone at *resolution* degrees: every distinct orientation once. That
    is what :class:`DictionaryIndexer` samples, and what to prefer.

    Without one, a plain Bunge Euler grid with a *resolution* degree step over the
    ranges *phi1*, *Phi* and *phi2* (degrees). It is neither equal-area (it bunches
    toward ``Phi = 0``) nor free of symmetric duplicates, and its default ranges
    cover the cubic fundamental region only: at 5 degrees it is ~26k orientations
    against m-3m's ~6.6k. It is for a crystal whose symmetry is not known, such as
    the generic :func:`~de_ebsd.sim.kinematic_bands.cubic_reflectors`.
    """
    if point_group is not None:
        return get_sample_fundamental(resolution=float(resolution), point_group=point_group)
    step = float(resolution)
    axes = [
        np.deg2rad(np.arange(phi1[0], phi1[1], step)),
        np.deg2rad(np.arange(Phi[0], Phi[1] + 1e-9, step)),
        np.deg2rad(np.arange(phi2[0], phi2[1] + 1e-9, step)),
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    return Rotation.from_euler(np.stack([g.ravel() for g in grid], -1))


class DictionaryIndexer:
    """Index EBSD patterns against a fundamental-zone dictionary by matmul.

    The dictionary -- the master projected at every fundamental-zone orientation,
    normalized -- is built once in :meth:`__init__` and kept on the device;
    :meth:`index` then scores one pattern or a whole stack with one matmul. The grid
    is coarse by design, so the intended workflow is this fast coarse pass followed
    by :meth:`refine`.

    Parameters
    ----------
    master
        A :class:`de_ebsd.sim.MasterPattern` or :class:`de_ebsd.sim.KinematicMaster`,
        projected onto the detector at every dictionary orientation. For a
        dictionary already simulated, use :meth:`from_patterns`.
    detector
        :class:`~de_ebsd.sim.detector.EBSDDetector` geometry; indexed patterns must
        match its ``(nrows, ncols)``.
    point_group
        orix :class:`~orix.quaternion.Symmetry` whose fundamental zone is sampled.
        Falls back to ``master.metadata["crystal"]``; one or the other is required
        unless *rotations* is given.
    resolution
        Target average disorientation (degrees) of the sampling. Default ``3.0``,
        which refines to the same sub-degree accuracy as ``2.0`` at a ~3x cheaper
        build, while ``4.0`` starts dropping patterns into a wrong basin that
        refinement cannot escape. Go finer only for strongly pseudo-symmetric phases.
    device
        Explicit argument > ``master.device`` > CPU.
    dtype
        ``None`` (default) auto-selects fp16 on tensor-core GPUs (compute capability
        >= 7.0) and fp32 elsewhere -- on Pascal (native fp16 at 1/64 the fp32 rate)
        and on CPU (no fp16 matmul) fp16 is a pessimization.
    energy_kev
        Energy bin the master is projected at; ``None`` (default) is the highest.
    match_binning
        Bin factor for the coarse match only (default ``1``). Kikuchi bands are
        low-frequency, so a binned pattern resolves the coarse cell just as well
        while the match shrinks by ``match_binning**2``. Must divide both detector
        dimensions. Refinement still runs at full resolution, so this trades nothing
        away.
    rotations
        The dictionary's orientations, an orix ``Rotation``, in place of sampling the
        fundamental zone (for instance :func:`sample_orientations` without a point
        group); *point_group* and *resolution* are then not used to sample.
    dynamic_background
        Remove each simulated pattern's dynamic background with
        :func:`~de_ebsd.index.preprocess.remove_dynamic_background` (its defaults: a
        Gaussian of ``sigma = width / 8``, truncated at 4 sigma) before it is
        normalized, so the dictionary goes through the filter the experimental
        patterns did. Default ``False``.
    lock
        Held around every call that runs device work, building included, for an
        application that shares the device between threads; see
        :func:`~de_ebsd.core.device.resolve_lock`.

    Attributes
    ----------
    dictionary_rotations : orix.quaternion.Rotation
        The dictionary orientations, length ``n_dict``.
    dictionary : torch.Tensor
        Normalized, stored transposed and contiguous as ``(n_px, n_dict)`` so the
        match runs at full matmul throughput (a transposed view runs at ~half).
    n_dict : int
        Number of orientations in the dictionary.
    """

    @holds_lock
    def __init__(
        self,
        master,
        detector,
        point_group=None,
        resolution=3.0,
        device=None,
        dtype=None,
        energy_kev=None,
        match_binning=1,
        rotations=None,
        dynamic_background=False,
        lock=None,
    ):
        self.lock = resolve_lock(lock)
        self.master = master
        self.detector = detector  # full resolution; refinement uses this one
        self.resolution = float(resolution)
        self.energy_kev = energy_kev
        self.match_binning = int(match_binning)
        self.dynamic_background = bool(dynamic_background)

        # the pattern center is a fraction, so the binned match detector shares the
        # full one's geometry
        b = self.match_binning
        if b > 1:
            h, w = self.detector.nrows, self.detector.ncols
            if h % b or w % b:
                raise ValueError(f"match_binning {b} must divide the detector shape {(h, w)}")
            self._match_detector = replace(self.detector, shape=(h // b, w // b))
        else:
            self._match_detector = self.detector
        self.pattern_shape = (self.detector.nrows, self.detector.ncols)

        if rotations is None:
            self.point_group = self._resolve_point_group(master, point_group)
            rotations = sample_orientations(self.resolution, self.point_group)
        else:
            self.point_group = point_group
        self.dictionary_rotations = Rotation(np.asarray(rotations.data).reshape(-1, 4))
        self.n_dict = int(self.dictionary_rotations.size)

        self.device = resolve_device(device, getattr(master, "device", "cpu"))
        self.dtype = self._default_dtype(self.device) if dtype is None else dtype
        self.dictionary = self._build_dictionary()

    @classmethod
    def from_patterns(cls, patterns, rotations, device=None, dtype=None, lock=None):
        """An indexer over a dictionary already simulated.

        Parameters
        ----------
        patterns
            The dictionary, ``(n_dict, H, W)`` or ``(n_dict, H * W)``, numpy or torch.
            Indexed patterns must have its shape.
        rotations
            The orientation of each entry, an orix ``Rotation`` of size ``n_dict``.
        device
            Torch device; the patterns' device when ``None``.
        dtype, lock
            As in :class:`DictionaryIndexer`.

        Returns
        -------
        DictionaryIndexer
            One without a master or a detector: it indexes and matches, but does not
            :meth:`refine`.
        """
        entries = torch.as_tensor(patterns)
        rotations = Rotation(np.asarray(rotations.data).reshape(-1, 4))
        if rotations.size != entries.shape[0]:
            raise ValueError(
                f"{entries.shape[0]} dictionary patterns but {rotations.size} rotations"
            )
        self = cls.__new__(cls)
        self.lock = resolve_lock(lock)
        self.master = self.detector = self.point_group = self.energy_kev = None
        self.resolution = None
        self.match_binning = 1
        self.dynamic_background = False
        self.pattern_shape = tuple(entries.shape[1:])
        self.dictionary_rotations = rotations
        self.n_dict = int(rotations.size)
        self.device = resolve_device(device, entries.device)
        self.dtype = self._default_dtype(self.device) if dtype is None else dtype
        with self.lock:
            flat = entries.reshape(self.n_dict, -1).to(device=self.device, dtype=torch.float32)
            self.dictionary = normalize_rows(flat).to(self.dtype).T.contiguous()
        return self

    @staticmethod
    def _default_dtype(device) -> torch.dtype:
        tensor_cores = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7
        return torch.float16 if tensor_cores else torch.float32

    @staticmethod
    def _resolve_point_group(master, point_group) -> Symmetry:
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
                "fundamental zone -- pass point_group=<orix Symmetry>, a master whose "
                "metadata['crystal'] is a de_ebsd Crystal, or the orientations "
                "themselves as rotations=sample_orientations(...)"
            )
        from de_ebsd.index.io import phase_from_crystal

        return phase_from_crystal(crystal).point_group

    def _build_dictionary(self) -> torch.Tensor:
        # master.project runs on the master's device, so move the master here once
        master = self.master
        if master.device != self.device:
            master = master.to(self.device)

        det = self._match_detector
        q = torch.as_tensor(np.asarray(self.dictionary_rotations.data), dtype=torch.float32)
        n_px = int(det.nrows * det.ncols)
        dictionary = torch.empty(self.n_dict, n_px, device=self.device, dtype=self.dtype)
        for i in range(0, self.n_dict, _PROJ_CHUNK):
            sim = master.project(
                det,
                q[i : i + _PROJ_CHUNK],
                energy_kev=self.energy_kev,
                rescale=False,
            )
            if self.dynamic_background:
                sim = remove_dynamic_background(sim, device=self.device)
            sim = sim.reshape(sim.shape[0], -1).to(torch.float32)  # normalize in fp32
            dictionary[i : i + _PROJ_CHUNK] = normalize_rows(sim).to(self.dtype)
        return dictionary.T.contiguous()

    def _match_chunk(self, n_patterns: int) -> int:
        chunk = max(1, _MAX_SCORE_ELEMS // max(1, self.n_dict))
        return min(chunk, n_patterns)

    def _prepare(self, patterns) -> torch.Tensor:
        """Patterns as ``(Np, n_px)``: binned to the match resolution and normalized.

        Takes ``(H, W)`` or ``(Np, H, W)``; an indexer from :meth:`from_patterns` of
        flat ``(n_dict, K)`` entries takes ``(K,)`` or ``(Np, K)``.
        """
        patt = torch.as_tensor(patterns)
        shape = tuple(self.pattern_shape)
        if patt.ndim == len(shape):
            patt = patt[None]
        if patt.ndim != len(shape) + 1:
            raise ValueError(
                f"patterns must be one pattern {shape} or a stack of them, "
                f"got shape {tuple(patt.shape)}"
            )
        if tuple(patt.shape[1:]) != shape:
            raise ValueError(
                f"pattern shape {tuple(patt.shape[1:])} does not match the detector {shape}"
            )
        n_p = patt.shape[0]
        patt = patt.to(device=self.device, dtype=torch.float32)
        if self.match_binning > 1:
            patt = F.avg_pool2d(patt[:, None], self.match_binning)[:, 0]
        # patterns are low-bit-depth: keep fp32 precision until the match itself
        return normalize_rows(patt.reshape(n_p, -1)).to(self.dtype)

    def _match(self, patterns, keep, progress=None, return_scores=False):
        """The one match: normalized patterns times the dictionary, chunked over the
        patterns so the score block stays bounded, each row reduced to its *keep* best.

        Returns ``(indices, scores, full)``: the ``(Np, keep)`` dictionary indices and
        float32 scores, best first, and the full ``(Np, n_dict)`` float32 score
        matrix when *return_scores* (``None`` otherwise).
        """
        exp_n = self._prepare(patterns)
        n_p = exp_n.shape[0]
        keep = max(1, min(int(keep), self.n_dict))
        best_idx = torch.empty(n_p, keep, dtype=torch.long, device=self.device)
        best_score = torch.empty(n_p, keep, dtype=torch.float32, device=self.device)
        scores_out = [] if return_scores else None
        chunk = self._match_chunk(n_p)
        for i in range(0, n_p, chunk):
            scores = exp_n[i : i + chunk] @ self.dictionary
            # reducing in the raw dtype is bit-identical to casting first, and skips a
            # large per-chunk fp32 allocation that otherwise dominates the fast matmul
            if keep == 1:
                bscore, bidx = scores.max(dim=1, keepdim=True)
            else:
                bscore, bidx = torch.topk(scores, keep, dim=1)
            best_score[i : i + chunk] = bscore.float()
            best_idx[i : i + chunk] = bidx
            if return_scores:
                scores_out.append(scores.float())
            if progress is not None:
                progress(min(i + chunk, n_p), n_p)
        full = torch.cat(scores_out, dim=0) if return_scores else None
        return best_idx, best_score, full

    @holds_lock
    def match(self, patterns, keep=1, progress=None):
        """The *keep* best dictionary entries for one pattern or a stack.

        Parameters
        ----------
        patterns
            A detector image ``(H, W)`` or stack ``(Np, H, W)``, numpy or torch, at
            full detector resolution.
        keep
            Matches kept per pattern, at most the dictionary size. More than one is
            what :func:`~de_ebsd.index.io.orientation_similarity_map` reads.
        progress
            Called as ``progress(done, total)`` after each chunk of patterns; an
            exception raised in it stops the match.

        Returns
        -------
        indices : torch.Tensor
            ``(Np, keep)`` indices into :attr:`dictionary_rotations`, best first.
        scores : torch.Tensor
            ``(Np, keep)`` float32 NCC in ``[-1, 1]``, best first.
        """
        indices, scores, _ = self._match(patterns, keep, progress)
        return indices, scores

    @holds_lock
    def index(self, patterns, return_scores=False, progress=None):
        """Index one pattern or a stack against the dictionary.

        Patterns are always given at full detector resolution; binning to the match
        resolution happens internally.

        Parameters
        ----------
        patterns
            A detector image ``(H, W)`` or stack ``(Np, H, W)``, numpy or torch.
        return_scores
            Also return the full ``(Np, n_dict)`` float32 score matrix.
        progress
            As in :meth:`match`.

        Returns
        -------
        rotations : orix.quaternion.Rotation
            Best-matching dictionary orientation per pattern.
        quality : torch.Tensor
            Peak NCC per pattern, roughly in ``[-1, 1]``; higher is better.
        scores : torch.Tensor
            Only when ``return_scores``: the ``(Np, n_dict)`` matrix whose per-row
            argmax gives ``rotations``.
        """
        best_idx, best_score, full = self._match(patterns, 1, progress, return_scores)
        rot_data = np.asarray(self.dictionary_rotations.data)[best_idx[:, 0].cpu().numpy()]
        rotations = Rotation(rot_data)
        if return_scores:
            return rotations, best_score[:, 0], full
        return rotations, best_score[:, 0]

    @holds_lock
    def refine(self, patterns, rotations=None, **kw):
        """Gradient-refine the coarse dictionary orientations to sub-grid accuracy.

        Wraps :func:`de_ebsd.index.refine.orientation.refine_orientation`, seeding
        from this indexer's master and detector (from :meth:`index` when
        ``rotations`` is ``None``).

        Defaults to ``method="lm"``: the second-order refiner reaches the same
        sub-degree accuracy in ~10 steps rather than Adam's ~60. LM is
        orientation-only, so ``refine_pattern_center=True`` falls back to Adam. ``**kw`` is
        forwarded; ``device`` defaults to this indexer's.
        """
        from de_ebsd.index.refine.orientation import refine_orientation

        if self.master is None:
            raise ValueError("an indexer made from_patterns has no master to refine against")
        if rotations is None:
            rotations, _ = self.index(patterns)
        kw.setdefault("method", "lm")
        kw.setdefault("device", self.device)
        return refine_orientation(patterns, self.master, self.detector, rotations, **kw)

    @holds_lock
    def index_scan(
        self,
        patterns,
        shape=None,
        step=1.0,
        phase=None,
        batch_size=None,
        refine=False,
        refine_iters=12,
    ):
        """Index a whole scan and assemble it into an orix ``CrystalMap``.

        Mirrors
        :meth:`~de_ebsd.index.spherical.indexer.SphericalIndexer.index_scan`.

        Parameters
        ----------
        patterns
            Either ``(rows, cols, H, W)``, from which the grid is inferred, or a flat
            row-major ``(N, H, W)``, which requires ``shape``.
        shape
            ``(rows, cols)`` of the scan. Cross-checked against 4-D input.
        step
            Scan step size, written onto the map's ``x``/``y``.
        phase
            An orix :class:`~orix.crystal_map.Phase` or a de_ebsd
            :class:`~de_ebsd.crystal.Crystal`. If ``None``, derived from
            ``master.metadata["crystal"]``.
        batch_size
            Patterns per mini-batch (default ``256``). Results are identical to a
            single batch.
        refine
            Gradient-refine each mini-batch before assembly, making
            ``prop["quality"]`` the refined NCC. The coarse grid makes this the
            intended finish.
        refine_iters
            Levenberg-Marquardt steps when ``refine`` (default ``12``).

        Returns
        -------
        orix.crystal_map.CrystalMap
            One orientation + ``prop["quality"]`` per scan point, single phase.
        """
        from de_ebsd.index.io import phase_from_crystal, to_crystal_map

        # the stack stays on CPU (view-only reshape); only each mini-batch moves to
        # self.device, so a large scan never lands wholesale on the GPU
        patt = torch.as_tensor(patterns, dtype=torch.float32)
        if patt.ndim == 4:
            rows, cols = int(patt.shape[0]), int(patt.shape[1])
            if shape is not None and (int(shape[0]), int(shape[1])) != (rows, cols):
                raise ValueError(
                    f"shape {tuple(shape)} does not match the (rows, cols, H, W) "
                    f"pattern grid {(rows, cols)}"
                )
            patt = patt.reshape(rows * cols, patt.shape[2], patt.shape[3])
        elif patt.ndim == 3:
            if shape is None:
                raise ValueError(
                    "shape=(rows, cols) is required for (N, H, W) patterns "
                    "(pass (rows, cols, H, W) to infer it instead)"
                )
            rows, cols = int(shape[0]), int(shape[1])
            if rows * cols != patt.shape[0]:
                raise ValueError(
                    f"shape {(rows, cols)} implies {rows * cols} patterns but got {patt.shape[0]}"
                )
        else:
            raise ValueError(
                f"patterns must be (rows, cols, H, W) or (N, H, W), got shape {tuple(patt.shape)}"
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
                rot_c, qual_c = self.refine(chunk, rotations=rot_c, n_iterations=refine_iters)
            rot_chunks.append(np.asarray(rot_c.data))
            qual_chunks.append(qual_c.detach().cpu())
        rotations = Rotation(np.concatenate(rot_chunks, axis=0))
        quality = torch.cat(qual_chunks)

        if phase is None:
            crystal = None
            meta = getattr(self.master, "metadata", None)
            if isinstance(meta, dict):
                crystal = meta.get("crystal")
            if crystal is None:
                raise ValueError(
                    "no phase given and master.metadata has no 'crystal' to derive one "
                    "from; pass phase=<orix Phase or de_ebsd Crystal> to index_scan "
                    "(the Phase gives the CrystalMap its crystal symmetry)"
                )
            phase = phase_from_crystal(crystal)

        return to_crystal_map(rotations, quality, (rows, cols), phase, step=step)
