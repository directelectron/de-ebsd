"""High-resolution refinement on band-edge samples.

Instead of a whole back-projected direction set, ``refine_hr`` importance-samples
a few thousand high-gradient band-edge pixels -- the ones that actually carry the
orientation and pattern-center signal -- and runs Levenberg-Marquardt on those.
Refines orientation and pattern center jointly. See INDEXING_DESIGN.md (I6).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from torchebsd.core.geometry import sample_to_detector_tensor
from torchebsd.core.rotation import quaternion_to_matrix
from torchebsd.index.refine.orientation import _skew

__all__ = ["direction_cosines_from_pc", "edge_importance_samples", "refine_hr"]

_EPS = 1e-12


def _rotvec_to_matrix(w: torch.Tensor) -> torch.Tensor:
    k = _skew(w)
    th2 = (w * w).sum(-1)  # (...,)
    th = th2.clamp_min(_EPS).sqrt()
    small = th2 < 1e-12
    a = torch.where(small, torch.ones_like(th), torch.sin(th) / th)  # sin th / th
    b = torch.where(small, torch.full_like(th, 0.5), (1.0 - torch.cos(th)) / th2.clamp_min(_EPS))
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(k.shape)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def direction_cosines_from_pc(
    detector, pc: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """Sample-frame direction cosines at pixels ``(row, col)``, differentiable in ``pc``.

    A ``grid_sample``-free rewrite of
    :meth:`~torchebsd.sim.detector.EBSDDetector.direction_cosines` over an arbitrary
    pixel set, as pure torch arithmetic -- so it works under forward-mode AD, unlike
    the ``grid_sample`` back-projection. The detector-frame ray is

        ``x_d = aspect * ((col + 0.5) / ncols - pcx)``,
        ``y_d = pcy - (row + 0.5) / nrows``,
        ``z_d = pcz``,

    rotated into the sample frame and normalized. Reproduces
    ``EBSDDetector.direction_cosines`` to ~1e-16.

    ``pc`` is ``(..., 3)``, overriding ``detector.pc``, and broadcasts against the
    leading dims of the ``(..., K)`` fractional ``rows``/``cols``. Returns unit
    direction cosines ``(..., K, 3)``.
    """
    device, dtype = rows.device, rows.dtype
    om = sample_to_detector_tensor(detector, device, dtype)
    aspect = detector.ncols / detector.nrows
    pcx = pc[..., 0].unsqueeze(-1)
    pcy = pc[..., 1].unsqueeze(-1)
    pcz = pc[..., 2].unsqueeze(-1)
    x_d = aspect * ((cols + 0.5) / detector.ncols - pcx)
    y_d = pcy - (rows + 0.5) / detector.nrows
    z_d = pcz + torch.zeros_like(x_d)
    ray = torch.stack([x_d, y_d, z_d], dim=-1)  # (..., K, 3)
    dc = ray @ om  # sample-frame ray (rows: v @ om, matching direction_cosines)
    return dc / dc.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def _ddc_dpc(detector, pc: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    p = pc.detach()

    def f(pc_arg: torch.Tensor) -> torch.Tensor:
        return direction_cosines_from_pc(detector, pc_arg, rows, cols)

    cols_ddc = []
    for c in range(3):
        tangent = torch.zeros_like(p)
        tangent[..., c] = 1.0
        _, dd = torch.func.jvp(f, (p,), (tangent,))
        cols_ddc.append(dd)
    return torch.stack(cols_ddc, dim=-1)  # (..., K, 3, 3)


def _sobel_gradient_magnitude(patterns: torch.Tensor) -> torch.Tensor:
    kx = patterns.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = kx.t().contiguous()
    p = F.pad(patterns[:, None], (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(p, kx[None, None])
    gy = F.conv2d(p, ky[None, None])
    return torch.sqrt(gx * gx + gy * gy)[:, 0]


def edge_importance_samples(
    patterns: torch.Tensor,
    n_tiles: int = 16,
    samples_per_tile: int = 8,
    all_pixels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the band-edge sample pixels for each pattern (done once, up front).

    Partitions the detector into ``n_tiles x n_tiles`` tiles and keeps the
    ``samples_per_tile`` highest-Sobel-gradient pixels in each. Spreading the pick
    across tiles keeps every region represented, and the tile grouping drives the
    per-tile normalization of the objective. ``all_pixels`` ignores the gradient
    pick and uses every pixel (the slow reference path).

    Returns ``(rows, cols, flat_idx, tile_ids, counts)``: fractional coordinates
    ``(Np, K)``, flat indices ``(Np, K)``, the tile index ``(K,)`` of each sample
    slot (shared across patterns), and the slot count per tile. A tile with fewer
    pixels than ``samples_per_tile`` is padded by repeating its best pixel, so ``K``
    stays uniform.
    """
    np_, height, width = patterns.shape
    device, dtype = patterns.device, patterns.dtype
    n_total = n_tiles * n_tiles

    row_tile = (torch.arange(height, device=device) * n_tiles // height).clamp_max(n_tiles - 1)
    col_tile = (torch.arange(width, device=device) * n_tiles // width).clamp_max(n_tiles - 1)
    tile_map = (row_tile[:, None] * n_tiles + col_tile[None, :]).reshape(-1)  # (H*W,)

    if all_pixels:
        flat = torch.arange(height * width, device=device)
        tile_ids = tile_map
        counts = torch.bincount(tile_ids, minlength=n_total)
        flat = flat[None].expand(np_, -1).contiguous()
    else:
        grad = _sobel_gradient_magnitude(patterns).reshape(np_, height * width)
        th, tw = height // n_tiles, width // n_tiles
        # regular blocks: one topk over (Np, n_total, tile_pixels) instead of a
        # per-tile Python loop, whose nonzero/topk dominate the whole refine
        if height % n_tiles == 0 and width % n_tiles == 0 and th * tw >= samples_per_tile:
            gt = (
                grad.view(np_, n_tiles, th, n_tiles, tw)
                .permute(0, 1, 3, 2, 4)
                .reshape(np_, n_total, th * tw)
            )
            top = gt.topk(samples_per_tile, dim=2).indices  # (Np, n_total, spt), within-tile
            # within-tile slot (r * tw + c) -> global flat pixel ((a*th+r)*W + b*tw+c)
            a = torch.arange(n_tiles, device=device)[:, None, None, None]
            b = torch.arange(n_tiles, device=device)[None, :, None, None]
            r = torch.arange(th, device=device)[None, None, :, None]
            c = torch.arange(tw, device=device)[None, None, None, :]
            tile_to_flat = ((a * th + r) * width + (b * tw + c)).reshape(n_total, th * tw)
            flat = torch.gather(tile_to_flat[None].expand(np_, -1, -1), 2, top)
            flat = flat.reshape(np_, n_total * samples_per_tile)  # (Np, K)
            tile_ids = torch.arange(n_total, device=device).repeat_interleave(samples_per_tile)
            counts = torch.bincount(tile_ids, minlength=n_total)
        else:  # irregular tiles: per-tile fallback (correct, rarely hit)
            flat_cols, tid_cols = [], []
            for t in range(n_total):
                pix = torch.nonzero(tile_map == t, as_tuple=False).reshape(-1)
                if pix.numel() == 0:
                    continue
                k = min(samples_per_tile, pix.numel())
                top = grad[:, pix].topk(k, dim=1).indices  # (Np, k) into pix
                sel = pix[top]  # (Np, k) flat pixel indices
                if k < samples_per_tile:  # pad short tiles by repeating their best pixel
                    sel = torch.cat([sel, sel[:, :1].expand(np_, samples_per_tile - k)], dim=1)
                flat_cols.append(sel)
                tid_cols.append(
                    torch.full((samples_per_tile,), t, device=device, dtype=torch.long)
                )
            flat = torch.cat(flat_cols, dim=1)  # (Np, K)
            tile_ids = torch.cat(tid_cols)  # (K,)
            counts = torch.bincount(tile_ids, minlength=n_total)

    rows = (flat // width).to(dtype)
    cols = (flat % width).to(dtype)
    return rows, cols, flat, tile_ids, counts


def _seg_broadcast_sum(x: torch.Tensor, tile_ids: torch.Tensor, n_total: int) -> torch.Tensor:
    sums = x.new_zeros((x.shape[0], n_total, *x.shape[2:]))
    sums.index_add_(1, tile_ids, x)
    return sums.index_select(1, tile_ids)


def _standardize_tiled(
    x: torch.Tensor, tile_ids: torch.Tensor, counts_ps: torch.Tensor, n_total: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = _seg_broadcast_sum(x, tile_ids, n_total) / counts_ps
    c = x - mean
    s = _seg_broadcast_sum(c * c, tile_ids, n_total).sqrt().clamp_min(_EPS)
    return c, s, c / s


def _residual_jacobian(
    m: torch.Tensor,
    e_hat: torch.Tensor,
    g: torch.Tensor,
    tile_ids: torch.Tensor,
    counts_ps: torch.Tensor,
    n_total: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    c, s, m_hat = _standardize_tiled(m, tile_ids, counts_ps, n_total)
    g_mean = _seg_broadcast_sum(g, tile_ids, n_total) / counts_ps.view(1, -1, 1)
    nu = _seg_broadcast_sum(c.unsqueeze(-1) * g, tile_ids, n_total) / s.unsqueeze(-1)
    dm_hat = ((g - g_mean) - m_hat.unsqueeze(-1) * nu) / s.unsqueeze(-1)
    return e_hat - m_hat, -dm_hat


def _hr_lm_refine(
    master,
    detector,
    rows,
    cols,
    e_raw,
    tile_ids,
    counts,
    r0_mat,
    pc0,
    refine_pc,
    n_steps,
    lam0,
    keep_best,
    device,
    dtype,
):
    n_total = int(counts.shape[0])
    n_active = int((counts > 0).sum())
    counts_ps = counts.index_select(0, tile_ids).to(dtype)  # (K,)
    _, _, e_hat = _standardize_tiled(e_raw, tile_ids, counts_ps, n_total)

    n_p = r0_mat.shape[0]
    r_mat = r0_mat.clone()
    pc = pc0.clone()
    lam = torch.full((n_p,), float(lam0), device=device, dtype=dtype)

    # orientation-only: the pattern center never moves, so build its direction
    # cosines once instead of rebuilding them every step
    dc_fixed = None if refine_pc else direction_cosines_from_pc(detector, pc0, rows, cols).detach()

    def _ncc(r_cur, pc_cur):
        dc = dc_fixed if dc_fixed is not None else direction_cosines_from_pc(
            detector, pc_cur, rows, cols
        )
        m = master.sample(dc @ r_cur.transpose(-1, -2)).to(device=device, dtype=dtype)
        _, _, m_hat = _standardize_tiled(m, tile_ids, counts_ps, n_total)
        return (e_hat * m_hat).sum(-1) / n_active

    seed_score = _ncc(r0_mat, pc0).detach()
    cur_score = seed_score.clone()

    for _ in range(n_steps):
        # (Np, K, 3); constant when the pattern center is fixed, so reuse the precompute
        dc = dc_fixed if dc_fixed is not None else direction_cosines_from_pc(
            detector, pc, rows, cols
        ).detach()
        # one reverse-mode grid_sample pass -> per-point master gradient dM/dv
        with torch.enable_grad():
            v = (dc @ r_mat.transpose(-1, -2)).detach().requires_grad_(True)
            m_raw = master.sample(v)
            (g_m,) = torch.autograd.grad(m_raw.sum(), v)
        m = m_raw.detach().to(device=device, dtype=dtype)
        g_m = g_m.to(device=device, dtype=dtype)

        q = torch.einsum("pka,pab->pkb", g_m, r_mat)  # gM @ R -> (Np, K, 3)
        dm_dw = torch.linalg.cross(dc, q, dim=-1)  # dc x (gM @ R), (Np, K, 3)
        if refine_pc:
            ddc_dpc = _ddc_dpc(detector, pc, rows, cols)  # (Np, K, 3, 3)
            dm_dpc = torch.einsum("pkb,pkbc->pkc", q, ddc_dpc)  # (Np, K, 3)
            g_raw = torch.cat([dm_dw, dm_dpc], dim=-1)  # (Np, K, 6)
        else:
            g_raw = dm_dw

        res, j_r = _residual_jacobian(m, e_hat, g_raw, tile_ids, counts_ps, n_total)
        with torch.no_grad():
            jtr = torch.einsum("pk,pki->pi", res, j_r)  # (Np, P)
            jtj = torch.einsum("pki,pkj->pij", j_r, j_r)  # (Np, P, P)
            diag = torch.diagonal(jtj, dim1=-2, dim2=-1).clamp_min(1e-9)
            a = jtj + lam[:, None, None] * torch.diag_embed(diag)
            delta = -torch.linalg.solve(a, jtr[..., None])[..., 0]  # (Np, P)
            r_new = r_mat @ _rotvec_to_matrix(delta[:, :3])
            pc_new = pc + delta[:, 3:6] if refine_pc else pc
            new_score = _ncc(r_new, pc_new)
            # per-pattern accept/reject: robust to the bilinear grid_sample kinks,
            # and monotone by construction
            better = new_score > cur_score
            r_mat = torch.where(better[:, None, None], r_new, r_mat)
            pc = torch.where(better[:, None], pc_new, pc)
            lam = torch.where(better, (lam * 0.5).clamp_min(1e-6), (lam * 3.0).clamp_max(1e2))
            cur_score = torch.where(better, new_score, cur_score)

    if keep_best:
        keep = (cur_score >= seed_score).reshape(-1)
        r_mat = torch.where(keep[:, None, None], r_mat, r0_mat)
        pc = torch.where(keep[:, None], pc, pc0)
        cur_score = torch.maximum(cur_score, seed_score)
    return r_mat, pc, cur_score


def refine_hr(
    patterns,
    master,
    detector,
    rotations,
    pc=None,
    refine_pc=True,
    n_tiles=16,
    samples_per_tile=8,
    n_steps=15,
    lm_lambda=1e-3,
    keep_best=True,
    device=None,
    all_pixels=False,
    batch_size=None,
    sample_by="master",
):
    """High-precision joint orientation + pattern-center refinement.

    The sub-0.1-degree-capable path. Rather than touching every detector pixel, it
    importance-samples a few thousand band-edge points spread across a tile grid and
    runs batched Levenberg-Marquardt on ``(R, pc)`` against the continuous master,
    over an objective normalized per tile (robust to intensity ramps). The sample
    budget (``n_tiles``, ``samples_per_tile``) is the accuracy/speed knob.

    Parameters
    ----------
    patterns
        A detector image ``(H, W)`` or stack ``(Np, H, W)``. The raw sampled values
        are used -- the experimental signal is never back-projected or interpolated.
    master, detector
        The continuous :class:`torchebsd.sim.MasterPattern` and the
        :class:`~torchebsd.sim.detector.EBSDDetector` geometry.
    rotations
        Seed :class:`orix.quaternion.Rotation`, typically a coarse index result.
    pc
        ``None`` uses ``detector.pc``. A ``(3,)`` value shares one center across the
        stack; ``(Np, 3)`` gives a per-pattern center. Returned in the same shape.
    refine_pc
        Refine the pattern center jointly (6 DOF, default) and return it, or leave
        it untouched and refine orientation only (3 DOF).
    n_tiles, samples_per_tile
        Tiles per detector side and samples kept per tile; the defaults give
        ``256 * 8 = 2048`` samples. Raising them improves accuracy proportionally.
    n_steps
        Levenberg-Marquardt steps (default 15); second-order, so a coarse seed
        sharpens well inside this budget.
    lm_lambda
        Initial damping, adapted per pattern each step (x0.5 accepted, x3 rejected).
    keep_best
        Never return a ``(R, pc)`` scoring below the seed, making refinement
        monotone.
    device
        Defaults to ``master.data.device``; otherwise everything is moved there.
    all_pixels
        Use every detector pixel instead of the band-edge subsample (slow reference
        path for validation).
    batch_size
        Patterns per mini-batch. ``None`` picks a memory-aware size; the result is
        identical for any batch size.
    sample_by
        Which gradient picks the sample pixels. ``"master"`` (default) renders the
        simulated pattern at the seed and samples its noise-free gradient, so
        samples land on true band edges. ``"experimental"`` samples the raw pattern,
        which on real data is noise-dominated and adds substantial refinement
        scatter; it is retained for noise-free/synthetic use.

    Returns
    -------
    rotation : orix.quaternion.Rotation
        The refined orientation(s), in the convention ``MasterPattern.project``
        consumes.
    score : torch.Tensor
        Mean per-tile NCC per pattern, in ``[-1, 1]``; ``>=`` the seed when
        ``keep_best``.
    pc : torch.Tensor
        Only when ``refine_pc``: the refined pattern center, shaped like the input.
    """
    device = master.data.device if device is None else torch.device(device)
    dtype = master.data.dtype
    if master.data.device != device:
        master = master.to(device)

    patt = torch.as_tensor(patterns, device=device, dtype=dtype)
    single = patt.ndim == 2
    if single:
        patt = patt[None]
    if patt.ndim != 3:
        raise ValueError(f"patterns must be (H, W) or (Np, H, W), got shape {tuple(patt.shape)}")
    n_p = patt.shape[0]

    q0 = torch.as_tensor(np.asarray(rotations.data), device=device, dtype=dtype).reshape(-1, 4)
    if q0.shape[0] != n_p:
        raise ValueError(f"rotations length {q0.shape[0]} != number of patterns {n_p}")
    r0_mat = quaternion_to_matrix(q0).detach()  # (Np, 3, 3)

    if pc is None:
        pc_init = torch.as_tensor(detector.pc, device=device, dtype=dtype)
    else:
        pc_init = torch.as_tensor(pc, device=device, dtype=dtype)
    if pc_init.ndim == 1 and pc_init.shape[0] == 3:
        pc_shared = True
        pc0 = pc_init[None].expand(n_p, 3).contiguous()
    elif pc_init.ndim == 2 and pc_init.shape == (n_p, 3):
        pc_shared = False
        pc0 = pc_init.contiguous()
    else:
        raise ValueError(
            f"pc must be (3,) shared or (Np, 3) per-pattern (Np={n_p}), got "
            f"{tuple(pc_init.shape)}"
        )

    # size the mini-batch from the dominant per-step working set -- a (bs, K, 3, 3)
    # d dc/d pc block with pattern-center refinement, (bs, K, 3) without -- to ~256 MB
    k = int(n_tiles * n_tiles * samples_per_tile) if not all_pixels else patt[0].numel()
    if batch_size is not None:
        bs = int(batch_size)
    else:
        per = k * (36 if refine_pc else 12)  # bytes/pattern of the dominant intermediate
        if sample_by == "master":
            # the seed render also holds a (bs, H*W, 3) direction-cosine block
            per = max(per, int(detector.nrows) * int(detector.ncols) * 12)
        bs = int(max(1, min(n_p, 256_000_000 // max(per, 1))))

    # full pixel grid, for rendering the seed pattern that picks the sample pixels
    if sample_by == "master":
        hh, ww = int(detector.nrows), int(detector.ncols)
        rr, cc = torch.meshgrid(
            torch.arange(hh, device=device, dtype=dtype),
            torch.arange(ww, device=device, dtype=dtype),
            indexing="ij",
        )
        full_rows, full_cols = rr.reshape(-1), cc.reshape(-1)
    elif sample_by != "experimental":
        raise ValueError(f"sample_by must be 'master' or 'experimental', got {sample_by!r}")

    r_parts, pc_parts, score_parts = [], [], []
    for i in range(0, n_p, bs):
        sl = slice(i, i + bs)
        p_chunk = patt[sl]
        if sample_by == "master":
            # the master is noise-free, so its gradient marks true band edges; the
            # experimental gradient adds ~0.5 deg of scatter on real patterns
            with torch.no_grad():
                dc_full = direction_cosines_from_pc(detector, pc0[sl], full_rows, full_cols)
                sim = master.sample(dc_full @ r0_mat[sl].transpose(-1, -2))
                sim = sim.to(device=device, dtype=dtype).reshape(-1, hh, ww)
            rows, cols, flat, tile_ids, counts = edge_importance_samples(
                sim, n_tiles, samples_per_tile, all_pixels
            )
        else:
            rows, cols, flat, tile_ids, counts = edge_importance_samples(
                p_chunk, n_tiles, samples_per_tile, all_pixels
            )
        e_raw = p_chunk.reshape(p_chunk.shape[0], -1).gather(1, flat.long())
        rf, pf, sc = _hr_lm_refine(
            master, detector, rows, cols, e_raw, tile_ids, counts, r0_mat[sl], pc0[sl],
            refine_pc, n_steps, lm_lambda, keep_best, device, dtype,
        )
        r_parts.append(rf)
        pc_parts.append(pf)
        score_parts.append(sc)
    r_final = torch.cat(r_parts)
    pc_final = torch.cat(pc_parts)
    score = torch.cat(score_parts)

    from orix.quaternion import Rotation

    rotation = Rotation.from_matrix(r_final.cpu().numpy())
    if not refine_pc:
        return rotation, score
    pc_out = pc_final.mean(0) if pc_shared else pc_final
    return rotation, score, pc_out.detach()
