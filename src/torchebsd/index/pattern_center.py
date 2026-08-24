"""Standalone pattern-center calibration: fit ONE shared pattern center over many patterns.

Determining an EBSD pattern center (PC) from a *single* pattern is degenerate. A small PC
shift is nearly indistinguishable from a small crystal rotation -- shifting the projection
origin moves the simulated bands almost exactly as a tiny reorientation would -- so a
per-pattern joint orientation+PC fit has a near-flat valley along that R--PC direction and
wanders (the orientation silently absorbs the PC error). This is why the joint 6-DOF path
in :func:`torchebsd.index.refine_hr` is excellent for *orientation* at a known PC but must
**not** be trusted to pin the PC one pattern at a time.

The well-posed fix is to fit a **single shared PC across many patterns at once**. Each
pattern contributes its own three (weak, correlated) PC constraints, but all patterns are
tied to the *same* three PC numbers, so the ensemble over-determines them: a PC that is
consistent with the band positions of dozens of differently-oriented patterns cannot be
mimicked by any single rotation, and the R--PC degeneracy is broken. A few dozen to a few
hundred well-distributed calibration patterns are plenty to fit three parameters.

**Algorithm -- alternating block-coordinate descent.** Repeat ``n_rounds`` times:

1. **Orientation block** (``refine_orientations``): with the shared PC held fixed, refine
   *each* pattern's orientation independently via the well-behaved orientation-only HR path
   (the same batched Levenberg-Marquardt core as :func:`torchebsd.index.refine_hr` with
   ``refine_pc=False``). This step is not degenerate -- at a fixed PC the orientation is
   cleanly determined.
2. **Shared-PC block**: with all orientations held fixed, take ``n_steps_pc`` LM steps on
   the **3-parameter shared PC**, building each pattern's PC-block normal equations
   (``JtJ`` 3x3, ``Jtr`` 3, from the same forward-mode ``d dc/d pc`` Jacobian and per-tile
   standardization derivative as the HR refiner) and **summing them over all N patterns**
   before the damped solve. Accept/reject each step on the **total per-tile NCC summed over
   the whole calibration set** (grow/shrink the damping). This ensemble accumulation is
   exactly what makes the PC well-posed.

The two blocks share the fixed band-edge samples and per-tile normalization of the HR
refiner (computed once up front), so the whole routine reuses that machinery.

**Workflow.** Run this once on a representative subset of a scan to calibrate the PC, then
refine the *whole* scan's orientations at that fixed PC with
``refine_hr(..., refine_pc=False, pc=<calibrated pc>)``. Fitting orientations at a good
shared PC reaches sub-0.1-degree; fitting them at a wrong PC cannot.

**Scope.** This fits a *single* shared PC for the whole set. Spatially-varying PC (per-tile
or per-region drift across a large scan) is out of scope here and left as future work.
"""

from __future__ import annotations

import numpy as np
import torch

from torchebsd.index.refine_highres import (
    _ddc_dpc,
    _hr_lm_refine,
    _residual_jacobian,
    _standardize_tiled,
    direction_cosines_from_pc,
    edge_importance_samples,
)
from torchebsd.sim.rotation import quaternion_to_matrix

__all__ = ["determine_pattern_center"]

_LAM_MIN = 1e-6
_LAM_MAX = 1e2


def determine_pattern_center(
    patterns,
    master,
    detector,
    rotations,
    pc0=None,
    n_rounds: int = 5,
    n_tiles: int = 16,
    samples_per_tile: int = 8,
    refine_orientations: bool = True,
    n_steps_pc: int = 10,
    lam0: float = 1e-3,
    keep_best: bool = True,
    device=None,
    return_orientations: bool = False,
    n_steps_orient: int = 15,
):
    """Calibrate one shared pattern center from many EBSD patterns (the well-posed way).

    Fits a **single** pattern center (PC) shared by the whole calibration set by
    alternating (a) an orientation-only refinement of every pattern at the current shared
    PC with (b) a Levenberg-Marquardt step on the 3-parameter shared PC whose normal
    equations are **accumulated over all patterns**. The ensemble accumulation is what
    breaks the single-pattern R--PC degeneracy (see the module docstring). Run this once to
    calibrate PC, then refine the full scan's orientations at the fixed PC with
    ``refine_hr(refine_pc=False, pc=<calibrated pc>)``.

    Parameters
    ----------
    patterns
        The calibration patterns ``(N, H, W)`` (numpy or torch; ``H = detector.nrows``,
        ``W = detector.ncols``), or a single ``(H, W)`` image (``N = 1``). Typically a
        well-distributed **subset** of a scan -- a few dozen to a few hundred patterns are
        plenty to fit three PC parameters. The **raw** sampled values are used; the
        experimental signal is never back-projected.
    master
        The :class:`torchebsd.sim.MasterPattern`; the continuous ``M(n)`` sampled
        differentiably via ``master.sample``.
    detector
        The :class:`~torchebsd.sim.detector.EBSDDetector` geometry (its tilt angles and
        shape; the fixed ``detector.pc`` is only used when ``pc0`` is ``None``).
    rotations
        The per-pattern seed orientations (an :class:`orix.quaternion.Rotation` of length
        ``N``), e.g. from coarse dictionary / spherical indexing.
    pc0
        The starting shared pattern center ``(3,)`` (``pcx, pcy, pcz``, Bruker
        convention). ``None`` (default) uses ``detector.pc``.
    n_rounds
        Number of alternating (orientation, shared-PC) rounds (default 5).
    n_tiles
        Tiles per detector side; ``n_tiles**2`` tiles total (default 16 -> 256 tiles).
    samples_per_tile
        Band-edge samples kept per tile (default 8); with the default ``n_tiles`` this is
        ``256 * 8 = 2048`` samples per pattern.
    refine_orientations
        If ``True`` (default), re-refine each pattern's orientation at the current shared
        PC at the start of every round; if ``False``, the input ``rotations`` are held
        fixed and only the shared PC is fitted (use this when the orientations are already
        trustworthy).
    n_steps_pc
        Levenberg-Marquardt steps on the shared PC per round (default 10).
    lam0
        Initial LM damping for the shared-PC step (default ``1e-3``); adapted each step
        (x0.5 on an accepted step, x3 on a rejected one) and carried across rounds. Also
        the initial damping of the inner orientation refinement.
    keep_best
        If ``True`` (default), never return a shared PC / orientation set scoring below the
        best seen: the calibration is monotone in the total per-tile NCC.
    device
        Torch device to run on. ``None`` (default) uses ``master.data.device``; when given,
        the master, patterns and seeds are moved there.
    return_orientations
        If ``True``, also return the refined per-pattern orientations (the orientations at
        the calibrated PC). Default ``False``.
    n_steps_orient
        Levenberg-Marquardt steps for the inner orientation-only refinement each round
        (default 15). Ignored when ``refine_orientations`` is ``False``.

    Returns
    -------
    pc : torch.Tensor
        The calibrated shared pattern center ``(3,)`` (detached), on ``device``.
    rotations : orix.quaternion.Rotation
        *Only when* ``return_orientations``: the refined per-pattern orientations at the
        calibrated PC (length ``N``). Returned **before** ``info``.
    info : dict
        Fit-quality diagnostics: ``"ncc"`` (mean per-tile NCC over the calibration set at
        the calibrated PC and orientations, in ``[-1, 1]``), ``"ncc_seed"`` (the same at
        the seed PC and orientations), ``"pc_step"`` (the L2 magnitude of the shared-PC
        move in the final round -- small means converged), ``"pc_step_history"`` (that
        magnitude per round), and ``"n_rounds"``.
    """
    device = master.data.device if device is None else torch.device(device)
    dtype = master.data.dtype
    if master.data.device != device:
        master = master.to(device)

    patt = torch.as_tensor(patterns, device=device, dtype=dtype)
    if patt.ndim == 2:
        patt = patt[None]
    if patt.ndim != 3:
        raise ValueError(f"patterns must be (H, W) or (N, H, W), got shape {tuple(patt.shape)}")
    n_p = patt.shape[0]

    q0 = torch.as_tensor(np.asarray(rotations.data), device=device, dtype=dtype).reshape(-1, 4)
    if q0.shape[0] != n_p:
        raise ValueError(f"rotations length {q0.shape[0]} != number of patterns {n_p}")
    r_mat = quaternion_to_matrix(q0).detach()  # (N, 3, 3)

    if pc0 is None:
        pc = torch.as_tensor(detector.pc, device=device, dtype=dtype)
    else:
        pc = torch.as_tensor(pc0, device=device, dtype=dtype)
    if tuple(pc.shape) != (3,):
        raise ValueError(f"pc0 must be a shared 3-vector (3,), got shape {tuple(pc.shape)}")
    pc = pc.clone()

    # fixed band-edge samples + per-tile normalization, computed once (patterns never move)
    rows, cols, flat, tile_ids, counts = edge_importance_samples(
        patt, n_tiles, samples_per_tile
    )
    e_raw = patt.reshape(n_p, -1).gather(1, flat.long())  # (N, K) raw experimental
    n_total = int(counts.shape[0])
    n_active = int((counts > 0).sum())
    counts_ps = counts.index_select(0, tile_ids).to(dtype)  # (K,)
    _, _, e_hat = _standardize_tiled(e_raw, tile_ids, counts_ps, n_total)  # (N, K) fixed

    def _ncc_per_pattern(r_cur: torch.Tensor, pc_vec: torch.Tensor) -> torch.Tensor:
        """Mean per-tile NCC of each pattern at ``(r_cur, shared pc)`` -> ``(N,)``."""
        pc_bc = pc_vec[None].expand(n_p, 3)
        dc = direction_cosines_from_pc(detector, pc_bc, rows, cols)
        m = master.sample(dc @ r_cur.transpose(-1, -2)).to(device=device, dtype=dtype)
        _, _, m_hat = _standardize_tiled(m, tile_ids, counts_ps, n_total)
        return (e_hat * m_hat).sum(-1) / n_active

    def _total(r_cur: torch.Tensor, pc_vec: torch.Tensor) -> torch.Tensor:
        """Total per-tile NCC summed over the whole calibration set (the shared objective)."""
        return _ncc_per_pattern(r_cur, pc_vec).sum()

    ncc_seed = float(_ncc_per_pattern(r_mat, pc).mean())

    best_pc = pc.clone()
    best_r = r_mat.clone()
    best_total = _total(r_mat, pc)
    lam = float(lam0)
    pc_step_history: list[float] = []

    for _ in range(n_rounds):
        # 1. orientation block -- refine each pattern at the current shared PC (well-posed)
        if refine_orientations:
            pc_bc = pc[None].expand(n_p, 3).contiguous()
            r_mat, _, _ = _hr_lm_refine(
                master, detector, rows, cols, e_raw, tile_ids, counts, r_mat, pc_bc,
                False, n_steps_orient, lam0, keep_best, device, dtype,
            )

        # 2. shared-PC block -- LM on the 3-vector, normal equations summed over all patterns
        pc_before = pc.clone()
        pc, lam = _shared_pc_lm(
            master, detector, rows, cols, r_mat, pc, e_hat, tile_ids, counts_ps, n_total,
            n_active, lam, n_steps_pc, _total, device, dtype,
        )
        pc_step_history.append(float((pc - pc_before).norm()))

        total = _total(r_mat, pc)
        if bool(total > best_total):
            best_total, best_pc, best_r = total, pc.clone(), r_mat.clone()

    if keep_best:
        pc, r_mat = best_pc, best_r

    pc = pc.detach()
    info = {
        "ncc": float(_ncc_per_pattern(r_mat, pc).mean()),
        "ncc_seed": ncc_seed,
        "pc_step": pc_step_history[-1] if pc_step_history else 0.0,
        "pc_step_history": pc_step_history,
        "n_rounds": n_rounds,
    }
    if return_orientations:
        from orix.quaternion import Rotation

        rotation = Rotation.from_matrix(r_mat.cpu().numpy())
        return pc, rotation, info
    return pc, info


def _shared_pc_lm(
    master,
    detector,
    rows,
    cols,
    r_mat,
    pc,
    e_hat,
    tile_ids,
    counts_ps,
    n_total,
    n_active,
    lam,
    n_steps,
    total_fn,
    device,
    dtype,
):
    """Levenberg-Marquardt on the shared 3-vector PC with ensemble-summed normal equations.

    With the per-pattern orientations ``r_mat`` fixed, each step builds every pattern's
    PC-block Jacobian (the forward-mode ``d dc/d pc`` contraction ``q . d dc/d pc`` with
    ``q = gM @ R``, pushed through the per-tile standardization derivative -- identical to
    the ``pc`` columns of the HR refiner), forms that pattern's ``JtJ`` (3x3) and ``Jtr``
    (3), and **sums both over all patterns** before the damped solve
    ``(JtJ + lam diag(JtJ)) dpc = -Jtr``. The trial ``pc`` is accepted only if it raises
    the **total** per-tile NCC summed over the whole set (then ``lam`` shrinks), else
    rejected (``lam`` grows). Summing over the ensemble is what conditions the otherwise
    near-singular single-pattern 3x3 system. Returns ``(pc, lam)``.
    """
    n_p = r_mat.shape[0]
    cur_total = total_fn(r_mat, pc)

    for _ in range(n_steps):
        pc_bc = pc[None].expand(n_p, 3).contiguous()
        dc = direction_cosines_from_pc(detector, pc_bc, rows, cols).detach()  # (N, K, 3)
        # one reverse-mode grid_sample pass -> per-point master gradient dM/dv
        with torch.enable_grad():
            v = (dc @ r_mat.transpose(-1, -2)).detach().requires_grad_(True)
            m_raw = master.sample(v)
            (g_m,) = torch.autograd.grad(m_raw.sum(), v)
        m = m_raw.detach().to(device=device, dtype=dtype)
        g_m = g_m.to(device=device, dtype=dtype)

        q = torch.einsum("pka,pab->pkb", g_m, r_mat)  # gM @ R -> (N, K, 3)
        ddc_dpc = _ddc_dpc(detector, pc_bc, rows, cols)  # (N, K, 3, 3)
        dm_dpc = torch.einsum("pkb,pkbc->pkc", q, ddc_dpc)  # raw dM/dpc, (N, K, 3)

        res, j_r = _residual_jacobian(m, e_hat, dm_dpc, tile_ids, counts_ps, n_total)
        with torch.no_grad():
            jtr = torch.einsum("pk,pki->pi", res, j_r).sum(0)  # summed over patterns, (3,)
            jtj = torch.einsum("pki,pkj->pij", j_r, j_r).sum(0)  # summed over patterns, (3, 3)
            diag = torch.diagonal(jtj).clamp_min(1e-9)
            a = jtj + lam * torch.diag_embed(diag)  # Marquardt (scale-invariant) damping
            delta = -torch.linalg.solve(a, jtr)  # (3,)
            pc_new = pc + delta
            new_total = total_fn(r_mat, pc_new)
            if bool(new_total > cur_total):
                pc = pc_new
                cur_total = new_total
                lam = max(lam * 0.5, _LAM_MIN)
            else:
                lam = min(lam * 3.0, _LAM_MAX)
    return pc, lam
