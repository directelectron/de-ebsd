"""Brute-force dictionary (template) indexing.

Projects a fundamental-zone orientation sample through the master pattern into
simulated patterns and takes the best normalized cross-correlation against each
experimental pattern. The conventional baseline that
:class:`~de_ebsd.index.spherical.indexer.SphericalIndexer` replaces.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F
from orix.quaternion import Rotation, Symmetry
from orix.sampling import get_sample_fundamental

from de_ebsd.index.preprocess import normalize_rows

# orientations projected per chunk when building the dictionary, so a large
# fundamental zone never materializes its whole (n_dict, P, 3) sampling tensor
_PROJ_CHUNK = 4096
# cap on the per-chunk (chunk, n_dict) score matrix, which bounds match memory
# regardless of dictionary size
_MAX_SCORE_ELEMS = 64_000_000


class DictionaryIndexer:
    """Index EBSD patterns against a fundamental-zone dictionary by matmul.

    The dictionary -- the master projected at every fundamental-zone orientation,
    normalized -- is built once in :meth:`__init__`; :meth:`index` then scores a
    whole stack with one matmul. The grid is coarse by design, so the intended
    workflow is this fast coarse pass followed by :meth:`refine`.

    Parameters
    ----------
    master
        A :class:`de_ebsd.sim.MasterPattern`, projected onto the detector at every
        dictionary orientation.
    detector
        :class:`~de_ebsd.sim.detector.EBSDDetector` geometry; indexed patterns must
        match its ``(nrows, ncols)``.
    point_group
        orix :class:`~orix.quaternion.Symmetry` whose fundamental zone is sampled.
        Falls back to ``master.metadata["crystal"]``; one or the other is required.
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

    Attributes
    ----------
    dictionary_rotations : orix.quaternion.Rotation
        The fundamental-zone orientations, length ``n_dict``.
    dictionary : torch.Tensor
        Normalized, stored transposed and contiguous as ``(n_px, n_dict)`` so the
        match runs at full matmul throughput (a transposed view runs at ~half).
    n_dict : int
        Number of orientations in the dictionary.
    """

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
    ):
        self.master = master
        self.detector = detector  # full resolution; refinement uses this one
        self.resolution = float(resolution)
        self.energy_kev = energy_kev
        self.match_binning = int(match_binning)

        # the pattern center is a fraction, so the binned match detector shares the
        # full one's geometry
        b = self.match_binning
        if b > 1:
            h, w = self.detector.nrows, self.detector.ncols
            if h % b or w % b:
                raise ValueError(
                    f"match_binning {b} must divide the detector shape {(h, w)}"
                )
            self._match_detector = replace(self.detector, shape=(h // b, w // b))
        else:
            self._match_detector = self.detector

        self.point_group = self._resolve_point_group(master, point_group)

        if device is not None:
            self.device = torch.device(device)
        else:
            master_device = getattr(master, "device", None)
            self.device = (
                torch.device(master_device) if master_device is not None else torch.device("cpu")
            )

        if dtype is None:
            has_tensor_cores = (
                self.device.type == "cuda"
                and torch.cuda.get_device_capability(self.device)[0] >= 7
            )
            dtype = torch.float16 if has_tensor_cores else torch.float32
        self.dtype = dtype

        self.dictionary_rotations = get_sample_fundamental(
            resolution=self.resolution, point_group=self.point_group
        )
        self.n_dict = int(self.dictionary_rotations.size)
        self.dictionary = self._build_dictionary()

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
                "fundamental zone -- pass point_group=<orix Symmetry> or a master whose "
                "metadata['crystal'] is a de_ebsd Crystal"
            )
        from de_ebsd.index.io import phase_from_crystal

        return phase_from_crystal(crystal).point_group

    def _build_dictionary(self) -> torch.Tensor:
        # master.project runs on master.data.device, so move the master here once
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
            sim = sim.reshape(sim.shape[0], -1).to(torch.float32)  # normalize in fp32
            dictionary[i : i + _PROJ_CHUNK] = normalize_rows(sim).to(self.dtype)
        return dictionary.T.contiguous()

    def _match_chunk(self, n_patterns: int) -> int:
        chunk = max(1, _MAX_SCORE_ELEMS // max(1, self.n_dict))
        return min(chunk, n_patterns)

    def index(self, patterns, return_scores=False):
        """Index one pattern or a stack against the fundamental-zone dictionary.

        Patterns are always given at full detector resolution; binning to the match
        resolution happens internally.

        Parameters
        ----------
        patterns
            A detector image ``(H, W)`` or stack ``(Np, H, W)``, numpy or torch.
        return_scores
            Also return the full ``(Np, n_dict)`` float32 score matrix.

        Returns
        -------
        rotations : orix.quaternion.Rotation
            Best-matching fundamental-zone orientation per pattern.
        quality : torch.Tensor
            Peak NCC per pattern, roughly in ``[-1, 1]``; higher is better.
        scores : torch.Tensor
            Only when ``return_scores``: the ``(Np, n_dict)`` matrix whose per-row
            argmax gives ``rotations``.
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
        patt = patt.to(device=self.device, dtype=torch.float32)
        if self.match_binning > 1:
            patt = F.avg_pool2d(patt[:, None], self.match_binning)[:, 0]
        # patterns are low-bit-depth: keep fp32 precision until the match itself
        exp_n = normalize_rows(patt.reshape(n_p, -1)).to(self.dtype)

        best_idx = torch.empty(n_p, dtype=torch.long, device=self.device)
        best_score = torch.empty(n_p, dtype=torch.float32, device=self.device)
        scores_out = [] if return_scores else None
        chunk = self._match_chunk(n_p)
        for i in range(0, n_p, chunk):
            scores = exp_n[i : i + chunk] @ self.dictionary
            # argmax on the raw dtype is bit-identical to casting first, and skips a
            # large per-chunk fp32 allocation that otherwise dominates the fast matmul
            bscore, bidx = scores.max(dim=1)
            best_score[i : i + chunk] = bscore.float()
            best_idx[i : i + chunk] = bidx
            if return_scores:
                scores_out.append(scores.float())

        rot_data = np.asarray(self.dictionary_rotations.data)[best_idx.cpu().numpy()]
        rotations = Rotation(rot_data)
        if return_scores:
            return rotations, best_score, torch.cat(scores_out, dim=0)
        return rotations, best_score

    def refine(self, patterns, rotations=None, **kw):
        """Gradient-refine the coarse dictionary orientations to sub-grid accuracy.

        Wraps :func:`de_ebsd.index.refine.orientation.refine_orientation`, seeding
        from this indexer's master and detector (from :meth:`index` when
        ``rotations`` is ``None``).

        Defaults to ``method="lm"``: the second-order refiner reaches the same
        sub-degree accuracy in ~10 steps rather than Adam's ~60. LM is
        orientation-only, so ``refine_pc=True`` falls back to Adam. ``**kw`` is
        forwarded; ``device`` defaults to this indexer's.
        """
        from de_ebsd.index.refine.orientation import refine_orientation

        if rotations is None:
            rotations, _ = self.index(patterns)
        kw.setdefault("method", "lm")
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
