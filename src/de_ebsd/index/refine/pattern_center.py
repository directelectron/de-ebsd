"""Calibrate a single pattern center shared across many patterns.

Alternates per-pattern orientation refinement with a global Levenberg-Marquardt
step on the shared PC, using the band-edge samples of
:mod:`de_ebsd.index.refine.highres`.
"""

from __future__ import annotations

import numpy as np
import torch

from de_ebsd.core.rotation import quaternion_to_matrix
from de_ebsd.index.refine.highres import (
    _ddc_dpc,
    _hr_lm_refine,
    _residual_jacobian,
    _standardize_tiled,
    direction_cosines_from_pc,
    edge_importance_samples,
)

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
    """Calibrate one shared pattern center from many EBSD patterns.

    Fits a single pattern center shared by the whole calibration set by alternating
    an orientation-only refinement of every pattern at the current PC with a
    Levenberg-Marquardt step on the 3-parameter PC whose normal equations are
    accumulated over all patterns. That ensemble accumulation is what breaks the
    single-pattern R--PC degeneracy. Calibrate once, then refine the full scan at the
    fixed PC with ``refine_hr(refine_pc=False, pc=<calibrated pc>)``.

    Parameters
    ----------
    patterns
        Calibration patterns ``(N, H, W)``, or a single ``(H, W)`` image. A
        well-distributed few dozen to few hundred are plenty to fit three
        parameters. The raw sampled values are used.
    master, detector
        The continuous :class:`de_ebsd.sim.MasterPattern` and the
        :class:`~de_ebsd.sim.detector.EBSDDetector` geometry. ``detector.pc`` is
        used only when ``pc0`` is ``None``.
    rotations
        Per-pattern seed orientations, e.g. from coarse indexing.
    pc0
        Starting shared pattern center ``(3,)``; ``None`` uses ``detector.pc``.
    n_rounds
        Alternating (orientation, shared-PC) rounds.
    n_tiles, samples_per_tile
        Tiles per detector side and band-edge samples kept per tile; the defaults
        give 2048 samples per pattern.
    refine_orientations
        Re-refine each orientation at the current PC every round (default), or hold
        the input ``rotations`` fixed and fit only the PC.
    n_steps_pc
        Levenberg-Marquardt steps on the shared PC per round.
    lam0
        Initial LM damping, adapted each step (x0.5 accepted, x3 rejected) and
        carried across rounds. Also seeds the inner orientation refinement.
    keep_best
        Never return a PC / orientation set scoring below the best seen, making the
        calibration monotone in the total per-tile NCC.
    device
        Defaults to ``master.data.device``; otherwise everything is moved there.
    return_orientations
        Also return the refined orientations, before ``info``.
    n_steps_orient
        LM steps for the inner orientation refinement each round.

    Returns
    -------
    pc : torch.Tensor
        The calibrated shared pattern center ``(3,)``.
    rotations : orix.quaternion.Rotation
        Only when ``return_orientations``: the orientations at the calibrated PC.
    info : dict
        ``"ncc"`` and ``"ncc_seed"`` (mean per-tile NCC at the calibrated and seed
        geometry), ``"pc_step"`` (L2 magnitude of the final round's PC move -- small
        means converged), ``"pc_step_history"``, and ``"n_rounds"``.
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
        pc_bc = pc_vec[None].expand(n_p, 3)
        dc = direction_cosines_from_pc(detector, pc_bc, rows, cols)
        m = master.sample(dc @ r_cur.transpose(-1, -2)).to(device=device, dtype=dtype)
        _, _, m_hat = _standardize_tiled(m, tile_ids, counts_ps, n_total)
        return (e_hat * m_hat).sum(-1) / n_active

    def _total(r_cur: torch.Tensor, pc_vec: torch.Tensor) -> torch.Tensor:
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
