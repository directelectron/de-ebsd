"""High-precision, edge-importance-sampled, tiled joint orientation + pattern-center
refinement (the sub-0.1 deg "HR" refine path).

This is the high-resolution sibling of :func:`torchebsd.index.refine.refine_orientation`.
Both maximize the same detector-vs-master normalized cross-correlation over the SO(3)
tangent (and, jointly, the pattern center); the difference is *where* the objective is
evaluated -- and that difference is the whole point.

**Why band-edge sampling.** A full-pattern refine that touches every detector pixel is
*sample-count-bound*: at 512x512 it runs at ~1 pattern/s no matter how the Jacobian is
written (this is the sample count, not an implementation artefact). But the orientation
and pattern-center information in a Kikuchi pattern lives in the **band edges** -- the
high-gradient regions -- not in the flat inter-band plateaus. So this refiner
**importance-samples a few thousand band-edge points** instead of all pixels. Measured on
a 512x512 pattern: 2048 samples -> ~4100 patt/s, 8192 samples -> ~430 patt/s. The sample
budget (``n_tiles``, ``samples_per_tile``) is therefore the accuracy/speed knob, and
precision scales with pattern resolution (a sharper, higher-resolution pattern offers
more and crisper edges to sample). Because we subsample to ~1-4k points, memory is a
non-issue (a ``2048 x 6`` Jacobian is tiny) -- there is **no** streaming or
tiling-for-memory here.

**What the tiling is for.** The detector is partitioned into ``n_tiles x n_tiles`` tiles
and the ``samples_per_tile`` highest-gradient pixels are taken *within each tile*. Tiling
serves two purposes, neither of them memory: (1) it spatially spreads the samples so
every region of the pattern contributes (a globally-thresholded edge pick would starve
whole quadrants), and (2) it enables **per-tile local normalization** -- each tile's
experimental and simulated values are standardized (zero-mean / unit-norm) independently,
which is robust to smooth intensity ramps across the detector (the HR-EBSD trick). The
objective is the sum over tiles of the per-tile NCC.

**Forward model (6 DOF, per pattern).** The parameters are ``theta = [w, dpc]`` with
``w`` a rotation vector (``R = R0 @ matrix_exp(skew(w))``, right-composed exactly as
:meth:`torchebsd.sim.MasterPattern.project` and the Adam/LM paths in ``refine.py``) and
``dpc`` an additive pattern-center correction (``pc = pc0 + dpc``, only when
``refine_pc``). For the fixed sampled pixels ``(row, col)`` the sample-frame directions
come from :func:`direction_cosines_from_pc` -- the detector's own gnomonic pixel ->
direction map rewritten as smooth (``grid_sample``-free) torch arithmetic, so it is
differentiable in the pattern center. Then ``v = dc @ R.T`` gives ``R n`` per sample and
``m = master.sample(v)`` reads the continuous master there. The experimental side ``E``
is the **raw** sampled pixel values (``patterns[sample_idx]``) -- the precious signal is
never back-projected or interpolated; only the smooth master is interpolated.

**Analytic Jacobian.** With ``K`` samples (a few thousand) the Jacobian is small, so no
streaming is needed. One reverse-mode pass through ``grid_sample`` gives the master's
per-point spatial gradient ``gM = dM/dv`` (its forward mode is unimplemented, and being
pointwise its Jacobian is diagonal in the points). Writing ``q = gM @ R``:

* **orientation block** ``dm/dw = dc x q`` -- the cross-product form (verified identical
  to the ``R @ (-skew(dc))`` tangent contraction to ~3e-8, and avoiding the ``(K, 3, 3)``
  tangent tensor);
* **pattern-center block** ``dm/dpc_c = q . (d dc/d pc_c)`` -- with ``d dc/d pc`` from
  **forward-mode AD** (``torch.func.jvp``) on :func:`direction_cosines_from_pc`, which is
  ``grid_sample``-free so forward mode works. (The two blocks share ``q``: since
  ``v = dc @ R.T``, ``dm/dpc_c = gM . (d dc/dpc_c @ R.T) = q . d dc/dpc_c``.)

The raw ``dM/dtheta`` is then pushed through the **per-tile** standardization derivative
(the rank-1 correction ``[(g - mean_tile g) - M_hat * nu_tile] / s_tile`` with
``nu_tile = sum_tile c g / s``, applied within each tile group) to get the residual
Jacobian ``J_r``. Each Levenberg-Marquardt step solves the damped normal equations
``(JtJ + lam diag(JtJ)) dtheta = -Jtr`` per pattern (a batched ``(Np, 3, 3)`` or
``(Np, 6, 6)`` solve) and, as in the ``refine.py`` LM path, **accepts/damps per pattern**
(recompute the summed-tile NCC at the trial ``theta``; take it and shrink ``lam`` where it
improved, reject and grow ``lam`` where it did not) -- which makes the step robust to the
bilinear ``grid_sample`` kinks and monotone by construction (``keep_best`` reasserts it).

**Pattern-center parameterization.** Every pattern refines its own 6-vector ``theta``
(the ``(Np, 6, 6)`` per-pattern normal equations), so ``pc`` is per-pattern internally. A
**shared** input ``pc`` (shape ``(3,)``) is broadcast to all patterns, refined
independently, and returned as the batch mean (a consensus over the stack, since every
pattern sees the same detector); a **per-pattern** input ``pc`` (shape ``(Np, 3)``) is
returned per pattern. The return shape always matches the input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from torchebsd.index.backproject import _sample_to_detector_tensor
from torchebsd.index.refine import _skew
from torchebsd.sim.rotation import quaternion_to_matrix

__all__ = ["direction_cosines_from_pc", "edge_importance_samples", "refine_hr"]

_EPS = 1e-12


def _rotvec_to_matrix(w: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(..., 3, 3)`` from rotation vectors ``w`` ``(..., 3)``.

    ``exp(skew(w))`` in closed form (Rodrigues), which is numerically identical to
    :func:`torch.matrix_exp` of the skew-symmetric ``skew(w)`` but far cheaper on the
    GPU -- ``matrix_exp`` runs an adaptive scaling-and-squaring routine whose device
    syncs (``.item()``, ``unique_consecutive``) dominate the LM step. ``R = I +
    (sin th / th) K + ((1 - cos th) / th^2) K^2`` with ``th = |w|``, ``K = skew(w)``;
    the ``th -> 0`` limits (``1`` and ``1/2``) are taken where ``th`` is tiny.
    """
    k = _skew(w)
    th2 = (w * w).sum(-1)  # (...,)
    th = th2.clamp_min(_EPS).sqrt()
    small = th2 < 1e-12
    a = torch.where(small, torch.ones_like(th), torch.sin(th) / th)  # sin th / th
    b = torch.where(small, torch.full_like(th, 0.5), (1.0 - torch.cos(th)) / th2.clamp_min(_EPS))
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(k.shape)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


# --------------------------------------------------------------------------- #
# differentiable pixel -> direction map (forward DC map, differentiable in pc)
# --------------------------------------------------------------------------- #
def direction_cosines_from_pc(
    detector, pc: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """Sample-frame direction cosines at pixels ``(row, col)``, differentiable in ``pc``.

    A ``grid_sample``-free rewrite of :meth:`torchebsd.sim.detector.EBSDDetector.
    direction_cosines` restricted to an arbitrary set of (possibly per-pattern) pixel
    coordinates, as pure torch arithmetic so it is autograd-differentiable in the pattern
    center (and works under forward-mode AD, unlike the ``grid_sample`` back-projection).
    The detector-frame ray of pixel ``(row, col)`` is

        ``x_d = aspect * ((col + 0.5) / ncols - pcx)``,
        ``y_d = pcy - (row + 0.5) / nrows``,
        ``z_d = pcz``,

    (the pattern-center factor ``pcz`` cancels out of ``x_d, y_d`` exactly as in
    ``gnomonic_bounds``), rotated into the sample frame by the orthonormal
    sample -> detector matrix and normalized. Verified to reproduce
    ``EBSDDetector.direction_cosines`` to ~1e-16 on the full grid.

    Parameters
    ----------
    detector
        The :class:`~torchebsd.sim.detector.EBSDDetector` geometry (its tilt angles and
        shape; the fixed ``detector.pc`` is *not* used -- ``pc`` overrides it).
    pc
        Pattern center ``(..., 3)`` (``pcx, pcy, pcz``), broadcasting against the leading
        dims of ``rows`` / ``cols``. Typically ``(Np, 3)`` for a per-pattern center; pass
        it with ``requires_grad`` (or through ``torch.func.jvp``) to differentiate the
        directions in the pattern center.
    rows, cols
        Fractional pixel coordinates ``(..., K)`` (row 0 = top, col 0 = left), matching
        ``pc``'s leading dims (e.g. ``(Np, K)``).

    Returns
    -------
    torch.Tensor
        Unit direction cosines ``(..., K, 3)`` in the sample frame.
    """
    device, dtype = rows.device, rows.dtype
    om = _sample_to_detector_tensor(detector, device, dtype)
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
    """Forward-mode ``d dc / d pc`` at the sampled pixels, shape ``(..., K, 3, 3)``.

    Three :func:`torch.func.jvp` passes (unit tangents along ``pcx, pcy, pcz``) through
    the smooth :func:`direction_cosines_from_pc` -- forward-mode AD works here because the
    map is ``grid_sample``-free. The last axis indexes the pattern-center component: entry
    ``[..., b, c] = d dc_b / d pc_c``.
    """
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


# --------------------------------------------------------------------------- #
# edge importance sampling
# --------------------------------------------------------------------------- #
def _sobel_gradient_magnitude(patterns: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude ``|grad E|`` of each pattern, shape ``(Np, H, W)``.

    A plain 3x3 Sobel with replicate padding -- band edges (the high-gradient regions
    that carry the orientation / pattern-center information) light up, flat inter-band
    plateaus stay dark.
    """
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

    Partitions the ``H x W`` detector into ``n_tiles x n_tiles`` tiles and, within each
    tile, selects the ``samples_per_tile`` pixels of largest Sobel gradient magnitude --
    the band edges crossing that tile. Spreading the pick across tiles keeps every region
    represented, and the fixed tile grouping drives the per-tile normalization of the
    objective.

    Parameters
    ----------
    patterns
        Experimental patterns ``(Np, H, W)``.
    n_tiles
        Number of tiles per detector side (``n_tiles**2`` tiles total).
    samples_per_tile
        Highest-gradient pixels kept per tile (``K = n_tiles**2 * samples_per_tile`` when
        every tile is non-empty; a tile with fewer pixels is padded by repeating its best
        pixel so ``K`` stays uniform).
    all_pixels
        If ``True``, ignore the gradient pick and use **every** detector pixel (the slow
        reference path); tiling still groups the pixels for the per-tile normalization.

    Returns
    -------
    rows, cols
        Fractional pixel coordinates ``(Np, K)`` (per pattern -- the gradient pick differs
        pattern to pattern) in ``patterns``' dtype.
    flat_idx
        The same samples as flat pixel indices ``(Np, K)`` (``row * W + col``), long --
        ready to gather the raw experimental values.
    tile_ids
        The tile index ``(K,)`` of each sample slot, shared across patterns (slot ``k``
        always belongs to the same tile). Values in ``[0, n_tiles**2)``.
    counts
        Number of sample slots per tile ``(n_tiles**2,)``, long (for the per-tile
        normalization); ``0`` for any empty tile.
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
        # Fast path when the tiles are regular blocks (H, W divisible by n_tiles) with
        # enough pixels: reshape into ``(Np, n_total, tile_pixels)`` and take every tile's
        # top-``samples_per_tile`` in ONE ``topk`` -- avoids the per-tile Python loop whose
        # ``nonzero``/``topk`` per tile dominate the whole refine (profiled).
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


# --------------------------------------------------------------------------- #
# per-tile standardization and residual Jacobian
# --------------------------------------------------------------------------- #
def _seg_broadcast_sum(x: torch.Tensor, tile_ids: torch.Tensor, n_total: int) -> torch.Tensor:
    """Sum ``x`` over the samples sharing a tile, broadcast back to per-sample.

    ``x`` is ``(Np, K, ...)``; returns the same shape where every sample carries its
    tile's sum. The engine behind the per-tile mean / norm / ``nu`` reductions (works for
    equal-size sampled tiles and the variable-size ``all_pixels`` tiles alike).
    """
    sums = x.new_zeros((x.shape[0], n_total, *x.shape[2:]))
    sums.index_add_(1, tile_ids, x)
    return sums.index_select(1, tile_ids)


def _standardize_tiled(
    x: torch.Tensor, tile_ids: torch.Tensor, counts_ps: torch.Tensor, n_total: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-tile zero-mean / unit-norm standardization of ``x`` ``(Np, K)``.

    Returns ``(c, s, x_hat)`` with ``c = x - mean_tile(x)``, ``s = ||c||_tile`` (clamped
    off zero, constant within a tile), and ``x_hat = c / s``. ``counts_ps`` is the tile
    size of each sample ``(K,)``.
    """
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
    """Residual ``r`` and its Jacobian ``J_r`` under the per-tile standardization.

    ``m`` ``(Np, K)`` is the raw master sample, ``e_hat`` ``(Np, K)`` the (fixed) per-tile
    standardized experimental field, and ``g`` ``(Np, K, P)`` the raw ``dM/dtheta``
    (``P = 3`` or ``6``). Applies, *within each tile group*, the standardization
    derivative

        ``d m_hat_i / d theta_k = [(g_ik - mean_tile g) - m_hat_i nu_k] / s``,
        ``nu_k = sum_tile c g_.k / s``,

    and returns ``(r = e_hat - m_hat, J_r = -d m_hat/d theta)``.
    """
    c, s, m_hat = _standardize_tiled(m, tile_ids, counts_ps, n_total)
    g_mean = _seg_broadcast_sum(g, tile_ids, n_total) / counts_ps.view(1, -1, 1)
    nu = _seg_broadcast_sum(c.unsqueeze(-1) * g, tile_ids, n_total) / s.unsqueeze(-1)
    dm_hat = ((g - g_mean) - m_hat.unsqueeze(-1) * nu) / s.unsqueeze(-1)
    return e_hat - m_hat, -dm_hat


# --------------------------------------------------------------------------- #
# batched Levenberg-Marquardt core
# --------------------------------------------------------------------------- #
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
    """Batched LM refinement of ``(R, pc)`` over the tiled band-edge objective.

    Returns ``(R (Np, 3, 3), pc (Np, 3), score (Np,))``; ``score`` is the mean per-tile
    NCC (in ``[-1, 1]``). See the module docstring for the method.
    """
    n_total = int(counts.shape[0])
    n_active = int((counts > 0).sum())
    counts_ps = counts.index_select(0, tile_ids).to(dtype)  # (K,)
    _, _, e_hat = _standardize_tiled(e_raw, tile_ids, counts_ps, n_total)

    n_p = r0_mat.shape[0]
    r_mat = r0_mat.clone()
    pc = pc0.clone()
    lam = torch.full((n_p,), float(lam0), device=device, dtype=dtype)

    # orientation-only: the pattern center never moves, so its direction cosines are
    # constant -- build them once instead of rebuilding every step (and inside _ncc).
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
            # per-pattern accept/reject: take the step where the summed-tile NCC improved
            # (shrink lam), keep the old (R, pc) where it did not (grow lam). Robust to the
            # bilinear grid_sample kinks; monotone by construction.
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


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
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
    """High-precision joint orientation + pattern-center refinement by band-edge sampling.

    The sub-0.1-degree-capable "HR" refine path. Rather than touching every detector pixel
    (which is sample-count-bound), it importance-samples a few thousand **band-edge**
    points -- the high-gradient regions that actually carry the orientation and
    pattern-center information -- spread across a tile grid, and runs a batched
    Levenberg-Marquardt refinement of ``(R, pc)`` against the continuous master over an
    objective that is **locally normalized per tile** (robust to intensity ramps). The
    sample budget (``n_tiles``, ``samples_per_tile``) is the accuracy/speed knob; precision
    scales with pattern resolution. See the module docstring for the full method (the
    cross-product orientation Jacobian, the forward-mode pattern-center Jacobian, and the
    per-tile standardization derivative).

    Parameters
    ----------
    patterns
        A single detector image ``(H, W)`` or a stack ``(Np, H, W)`` (numpy or torch;
        ``H = detector.nrows``, ``W = detector.ncols``). The **raw** sampled values are
        used -- the experimental signal is never back-projected or interpolated.
    master
        The :class:`torchebsd.sim.MasterPattern` -- the continuous ``M(n)`` sampled
        (differentiably in the sampling directions) via ``master.sample``.
    detector
        The :class:`~torchebsd.sim.detector.EBSDDetector` geometry; its pixel -> direction
        map is rebuilt as differentiable arithmetic (:func:`direction_cosines_from_pc`).
    rotations
        The seed :class:`orix.quaternion.Rotation` (length ``Np``, or length 1 / scalar
        for a single pattern) -- typically a coarse index result.
    pc
        Pattern center. ``None`` (default) uses ``detector.pc``. A length-3 array/tensor
        shares one center across the stack (returned as ``(3,)``); an ``(Np, 3)``
        array/tensor gives a per-pattern center (returned as ``(Np, 3)``).
    refine_pc
        If ``True`` (default), the pattern center is refined jointly with the orientation
        (6 DOF per pattern) and returned; if ``False`` the refinement is orientation-only
        (3 DOF), the pattern center is left untouched and **not** returned.
    n_tiles
        Tiles per detector side; ``n_tiles**2`` tiles total (default 16 -> 256 tiles).
    samples_per_tile
        Band-edge samples kept per tile (default 8); with the default ``n_tiles`` this is
        ``256 * 8 = 2048`` samples. Raising it improves accuracy at proportional cost.
    n_steps
        Levenberg-Marquardt steps (default 15). Second-order, so a coarse seed sharpens
        well inside this budget.
    lm_lambda
        Initial LM damping (default ``1e-3``); adapted per pattern each step (x0.5 on an
        accepted step, x3 on a rejected one).
    keep_best
        If ``True`` (default), never return a ``(R, pc)`` scoring below the seed: any
        pattern whose refined tiled NCC does not exceed its seed's reverts to the seed
        orientation and pattern center. Makes refinement monotone.
    device
        Torch device to run on. ``None`` (default) uses ``master.data.device``; when given,
        the master, patterns and seeds are moved there.
    all_pixels
        If ``True``, use **every** detector pixel instead of the band-edge subsample (a
        slow reference path for validation; the tiling still drives the per-tile
        normalization). Default ``False``.
    batch_size
        Number of patterns refined per mini-batch. ``None`` (default) picks a
        memory-aware size from the sample count and the 6-DOF working set, so a large
        scan does not re-allocate GB-scale per-step intermediates at once (which
        thrashes the CUDA allocator). The result is identical for any batch size (each
        pattern is refined independently); it only bounds peak memory. Pass an int to
        override.
    sample_by
        Which gradient chooses the band-edge sample pixels. ``"master"`` (default)
        renders the simulated pattern at the seed orientation and samples its
        **noise-free** gradient -- so the samples land on true Kikuchi band edges.
        ``"experimental"`` samples the raw pattern's own gradient, which on real (noisy)
        patterns is noise-dominated and adds substantial per-pattern refinement scatter
        (a within-grain KAM comparison on real data motivated the ``"master"`` default);
        it is retained mainly for noise-free/synthetic use.

    Returns
    -------
    rotation : orix.quaternion.Rotation
        The refined orientation(s), length ``Np`` (matrix -> quaternion via
        ``Rotation.from_matrix``), in the same convention the coarse index returns and
        ``MasterPattern.project`` consumes.
    score : torch.Tensor
        The mean per-tile NCC of each pattern, length ``Np`` (detached), in ``[-1, 1]``;
        ``>=`` the seed score when ``keep_best``. Always the second element.
    pc : torch.Tensor
        *Only when* ``refine_pc``: the refined pattern center (detached), same shape as
        the input ``pc`` (``(3,)`` shared -> batch mean, or ``(Np, 3)`` per pattern).
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

    # Mini-batch over patterns so a large scan never allocates GB-scale per-step
    # intermediates at once. Identical result for any batch (patterns are independent);
    # bounds peak memory. Size from the dominant per-step working set: the pattern-center
    # path holds a ``(bs, K, 3, 3)`` d dc/d pc block, the orientation-only path a
    # ``(bs, K, 3)`` one -- keep it within ~256 MB.
    k = int(n_tiles * n_tiles * samples_per_tile) if not all_pixels else patt[0].numel()
    if batch_size is not None:
        bs = int(batch_size)
    else:
        per = k * (36 if refine_pc else 12)  # bytes/pattern of the dominant intermediate
        if sample_by == "master":
            # the seed render also holds a (bs, H*W, 3) direction-cosine block
            per = max(per, int(detector.nrows) * int(detector.ncols) * 12)
        bs = int(max(1, min(n_p, 256_000_000 // max(per, 1))))

    # For master-gradient sampling: the full detector pixel grid (row, col) coordinates,
    # used to render the simulated pattern at the seed to pick band-edge sample pixels.
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
            # Pick sample pixels from the SIMULATED pattern at the seed -- the master is
            # noise-free, so its gradient marks the true Kikuchi band edges, whereas the
            # experimental gradient on real (noisy) patterns is noise-dominated and would
            # sample noise (measured to add ~0.5 deg of per-pattern refinement scatter).
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
