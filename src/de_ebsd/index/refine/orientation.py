"""Gradient refinement of orientation and (optionally) pattern center.

Maximizes a masked spherical NCC between the back-projected pattern and the
master pattern sampled at trial-rotated directions -- both differentiable, so a
coarse seed sharpens to sub-grid-cell accuracy. ``method="adam"`` is first-order
and supports pattern-center refinement; ``method="lm"`` is a batched
Gauss-Newton with an analytic Jacobian, ~2x faster but orientation-only.
"""

from __future__ import annotations

import numpy as np
import torch

from de_ebsd.core.rotation import quaternion_to_matrix, rotation_vector_to_matrix, skew_matrix
from de_ebsd.index.backproject import pattern_to_sphere
from de_ebsd.index.refine.levenberg import adapt_damping

__all__ = ["refine_orientation"]

_EPS = 1e-12


def _masked_ncc(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(-1, keepdim=True).clamp_min(_EPS)
    ma = (a * mask).sum(-1, keepdim=True) / denom
    mb = (b * mask).sum(-1, keepdim=True) / denom
    a0 = (a - ma) * mask
    b0 = (b - mb) * mask
    num = (a0 * b0).sum(-1)
    na = (a0 * a0).sum(-1).clamp_min(_EPS).sqrt()
    nb = (b0 * b0).sum(-1).clamp_min(_EPS).sqrt()
    return num / (na * nb)


def _fov_directions(detector, device, dtype, target: int = 1024) -> torch.Tensor:
    pix = detector.direction_cosines(device=device, dtype=dtype).reshape(-1, 3)
    step = max(1, pix.shape[0] // int(target))
    return pix[::step][: int(target)].contiguous()


def _backproject(patt, detector, dirs, pattern_center, per_pattern):
    if not per_pattern:
        e, mask = pattern_to_sphere(patt, detector, dirs, pattern_center=pattern_center)
        return e, mask
    es, masks = [], []
    for i in range(patt.shape[0]):
        e_i, m_i = pattern_to_sphere(patt[i : i + 1], detector, dirs, pattern_center=pattern_center[i])
        es.append(e_i)
        masks.append(m_i)
    return torch.cat(es, dim=0), torch.stack(masks, dim=0)


def _standardize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = x - x.mean(-1, keepdim=True)
    return c, c.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def _lm_residual_jacobian(master, dirs, r_mat, e_hat, device, dtype):
    n = dirs  # (M, 3), all in the FOV
    # grid_sample has no forward-mode AD, and being pointwise its Jacobian is
    # diagonal in the points -- so one reverse-mode pass gives every dM/dv
    with torch.enable_grad():
        v = (n @ r_mat.transpose(-1, -2)).detach().requires_grad_(True)  # (Np, M, 3)
        m_raw = master.sample(v)  # (Np, M)
        (g_m,) = torch.autograd.grad(m_raw.sum(), v)  # (Np, M, 3) = dM_i/dv_i
    m = m_raw.detach().to(device=device, dtype=dtype)
    g_m = g_m.to(device=device, dtype=dtype)
    # tangent of v = R n wrt delta_k: R @ (-skew(n)), columns k -> (Np, M, 3, 3)
    tang = r_mat[:, None] @ (-skew_matrix(n))[None]
    g = torch.einsum("pma,pmak->pmk", g_m, tang)  # raw dM/ddelta (Np, M, 3)
    c, s = _standardize(m)
    m_hat = c / s
    mu_g = g.mean(1)  # mean_k(g) over the directions (Np, 3)
    nu = torch.einsum("pm,pmk->pk", c, g) / s  # (Np, 3)
    dm_hat = ((g - mu_g[:, None]) - m_hat[..., None] * nu[:, None]) / s[..., None]
    return e_hat - m_hat, -dm_hat, m


def _lm_refine(
    master, dirs, r0_mat, e, mask, n_steps, damping, keep_best, return_history, device, dtype
):
    e_c, e_s = _standardize(e)
    e_hat = e_c / e_s  # (Np, M) standardized experimental field
    r_mat = r0_mat.clone()
    n_p = r_mat.shape[0]
    damping = torch.full((n_p,), float(damping), device=device, dtype=dtype)

    with torch.no_grad():
        m_seed = master.sample(dirs @ r0_mat.transpose(-1, -2)).to(device=device, dtype=dtype)
        seed_score = _masked_ncc(e, m_seed, mask)
    cur_score = seed_score
    history = [] if return_history else None

    for _ in range(n_steps):
        if return_history:
            history.append(cur_score)
        res, j_r, _ = _lm_residual_jacobian(master, dirs, r_mat, e_hat, device, dtype)
        with torch.no_grad():
            jtr = torch.einsum("pm,pmk->pk", res, j_r)  # (Np, 3)
            jtj = torch.einsum("pmk,pml->pkl", j_r, j_r)  # (Np, 3, 3)
            # LM damping: add damping * diag(JtJ) to the Gauss-Newton normal matrix.
            diag = torch.diagonal(jtj, dim1=-2, dim2=-1).clamp_min(1e-9)
            a = jtj + damping[:, None, None] * torch.diag_embed(diag)
            delta = -torch.linalg.solve(a, jtr[..., None])[..., 0]  # (Np, 3)
            r_new = r_mat @ rotation_vector_to_matrix(delta)
            m_new = master.sample(dirs @ r_new.transpose(-1, -2)).to(device=device, dtype=dtype)
            new_score = _masked_ncc(e, m_new, mask)
            # per-pattern accept/reject: robust to the bilinear grid_sample kinks,
            # and monotone by construction
            better = new_score > cur_score
            r_mat = torch.where(better[:, None, None], r_new, r_mat)
            damping = adapt_damping(damping, better)
            cur_score = torch.where(better, new_score, cur_score)

    refined_score = cur_score
    if keep_best:
        # already guaranteed by the accept/reject; kept symmetric with the Adam path
        keep = (refined_score >= seed_score).reshape(-1, 1, 1)
        r_final = torch.where(keep, r_mat, r0_mat)
        score = torch.maximum(refined_score, seed_score)
    else:
        r_final = r_mat
        score = refined_score
    if return_history:
        hist = torch.stack(history, dim=0) if history else e.new_zeros((0, n_p))
    else:
        hist = None
    return r_final, score, hist


def refine_orientation(
    patterns,
    master,
    detector,
    rotations,
    pattern_center=None,
    refine_pattern_center=False,
    method="adam",
    n_iters=None,
    learning_rate=0.02,
    pattern_center_learning_rate=None,
    damping=1e-3,
    directions=None,
    device=None,
    keep_best=True,
    return_history=False,
):
    """Refine ``rotations`` (and optionally the pattern center) by autograd.

    Sharpens a coarse seed to sub-grid-cell accuracy by maximizing a spherical
    masked NCC between the back-projected pattern ``E(n)`` and the master sampled at
    the trial-rotated directions ``M(R n)``, with ``R = R0 . dR(w)`` for a
    differentiable rotation-vector correction ``w``. The whole stack is refined in
    one parallel optimization on the master's device.

    Parameters
    ----------
    patterns
        A detector image ``(H, W)`` or stack ``(Np, H, W)``, numpy or torch.
    master, detector
        The continuous :class:`de_ebsd.sim.MasterPattern` and the
        :class:`~de_ebsd.sim.detector.EBSDDetector` whose pixel -> direction map is
        inverted (differentiably in the pattern center) to build ``E(n)``.
    rotations
        Seed :class:`orix.quaternion.Rotation`, typically the coarse index output.
    pattern_center
        ``None`` uses ``detector.pattern_center``. A ``(3,)`` value shares one center across the
        stack -- the common, fully-vectorized path; ``(Np, 3)`` gives a per-pattern
        center and loops the back-projection.
    refine_pattern_center
        Refine the pattern center jointly with the orientation. Forces
        ``method="adam"``, since the LM path is orientation-only.
    method
        ``"adam"`` (default) is first-order. ``"lm"`` is Levenberg-Marquardt with an
        analytic Jacobian, reaching the same sub-degree accuracy in ~10 steps
        instead of ~60 (~2.3x faster at equal accuracy on real data). LM is
        orientation-only and assumes the default in-FOV ``directions``.
    n_iters
        Step budget. ``None`` resolves per method: 60 for Adam, 12 for LM.
    learning_rate
        Adam step size for the rotation vector, in radians. The key stability knob:
        too large a step overshoots the shallow, noisy basin of real patterns and
        diverges. ``0.02`` is stable on real data; use ~0.01 for noisy scans.
    pattern_center_learning_rate
        Adam step size for the pattern center (default ``learning_rate / 5``, since its natural
        scale is finer). Ignored unless ``refine_pattern_center``.
    damping
        Initial LM damping, adapted per pattern each step (x0.5 accepted, x3
        rejected). LM only.
    directions
        Sphere directions ``(M, 3)`` the NCC is evaluated over. ``None`` uses the
        detector's in-FOV pixel directions, where every point contributes -- ~3x
        faster and more accurate than a whole-sphere set that is mostly masked away.
        ``method="lm"`` requires an all-in-FOV set; the default satisfies this, but a
        custom non-FOV set is unsafe for LM (use Adam, whose masked NCC handles it).
    device
        Defaults to ``master.data.device``; otherwise everything is moved there.
    keep_best
        Never return an orientation worse than the seed, making refinement monotone.
        This removes the stray outliers gradient refinement can introduce on
        low-quality real patterns.
    return_history
        Also return the per-step NCC trajectory.

    Returns
    -------
    rotation : orix.quaternion.Rotation
        The refined orientation(s), in the convention ``MasterPattern.project``
        consumes.
    score : torch.Tensor
        Final NCC per pattern, in ``[-1, 1]``; ``>=`` the coarse score. This is what
        ``index_scan(refine=True)`` stores as ``prop["quality"]``.
    pattern_center : torch.Tensor
        Only when ``refine_pattern_center``: the refined pattern center, shaped like the input.
    history : torch.Tensor
        Only when ``return_history``: the NCC at every step, ``(n_iters, Np)``, with
        ``history[0]`` the seed score.
    """
    # LM is orientation-only, so refine_pattern_center forces Adam
    method = str(method).lower()
    if method not in ("adam", "lm"):
        raise ValueError(f"method must be 'adam' or 'lm', got {method!r}")
    if method == "lm" and refine_pattern_center:
        method = "adam"  # joint pattern-center + orientation LM is future work
    if n_iters is None:
        n_iters = 12 if method == "lm" else 60

    device = master.data.device if device is None else torch.device(device)
    dtype = master.data.dtype
    # master.data.device drives sampling, so move the master to honor `device`
    if master.data.device != device:
        master = master.to(device)

    patt = torch.as_tensor(patterns, device=device, dtype=dtype)
    single = patt.ndim == 2
    if single:
        patt = patt[None]
    if patt.ndim != 3:
        raise ValueError(f"patterns must be (H, W) or (Np, H, W), got shape {tuple(patt.shape)}")
    n_p = patt.shape[0]

    # the detached anchor: exactly the matrix MasterPattern.project builds
    q0 = torch.as_tensor(np.asarray(rotations.data), device=device, dtype=dtype).reshape(-1, 4)
    if q0.shape[0] != n_p:
        raise ValueError(
            f"rotations length {q0.shape[0]} != number of patterns {n_p}"
        )
    r0_mat = quaternion_to_matrix(q0).detach()  # (Np, 3, 3)

    if directions is None:
        dirs = _fov_directions(detector, device, dtype)
    else:
        dirs = torch.as_tensor(directions, device=device, dtype=dtype).reshape(-1, 3)

    # pattern center: shared (3,) or per-pattern (Np, 3)
    if pattern_center is None:
        pc_init = torch.as_tensor(detector.pattern_center, device=device, dtype=dtype)
    else:
        pc_init = torch.as_tensor(pattern_center, device=device, dtype=dtype)
    if pc_init.ndim == 1 and pc_init.shape[0] == 3:
        per_pattern = False
    elif pc_init.ndim == 2 and pc_init.shape == (n_p, 3):
        per_pattern = True
    else:
        raise ValueError(
            f"pattern_center must be (3,) shared or (Np, 3) per-pattern (Np={n_p}), got "
            f"{tuple(pc_init.shape)}"
        )

    # orientation-only, so the pattern center is fixed and E is back-projected once
    if method == "lm":
        with torch.no_grad():
            e_lm, mask_lm = _backproject(patt, detector, dirs, pc_init, per_pattern)
        r_final, score, hist = _lm_refine(
            master, dirs, r0_mat, e_lm, mask_lm, n_iters, damping, keep_best,
            return_history, device, dtype,
        )
        from orix.quaternion import Rotation

        rotation = Rotation.from_matrix(r_final.cpu().numpy())
        out = [rotation, score]
        if return_history:
            out.append(hist)
        return tuple(out)

    # the FOV mask is held fixed at the seed geometry so the objective stays smooth
    # as the pattern center moves
    with torch.no_grad():
        _, mask = _backproject(patt, detector, dirs, pc_init, per_pattern)

    w = torch.zeros(n_p, 3, device=device, dtype=dtype, requires_grad=True)
    params = [{"params": [w], "lr": learning_rate}]
    if refine_pattern_center:
        pc_param = pc_init.clone().requires_grad_(True)
        params.append({"params": [pc_param], "lr": pattern_center_learning_rate if pattern_center_learning_rate is not None else learning_rate / 5.0})
    else:
        pc_param = pc_init
        # pattern center is fixed -> E does not change; back-project it once.
        e_fixed, _ = _backproject(patt, detector, dirs, pc_param, per_pattern)

    opt = torch.optim.Adam(params)
    history = [] if return_history else None

    def _score():
        if refine_pattern_center:
            e = _backproject(patt, detector, dirs, pc_param, per_pattern)[0]
        else:
            e = e_fixed
        d_r = rotation_vector_to_matrix(w)  # (Np, 3, 3)
        r = r0_mat @ d_r  # right-compose: R = R0 . dR(w)
        v = dirs @ r.transpose(-1, -2)  # (Np, M, 3) = R n, matching project
        m_r = master.sample(v).to(device=device, dtype=dtype)  # (Np, M)
        return _masked_ncc(e, m_r, mask)

    # seed NCC (w = 0), for the keep_best gate below
    with torch.no_grad():
        seed_score = _score().detach()

    for _ in range(n_iters):
        opt.zero_grad()
        s = _score()
        if return_history:
            history.append(s.detach())
        (-s.sum()).backward()
        opt.step()

    with torch.no_grad():
        refined_score = _score().detach()
        r_ref = r0_mat @ rotation_vector_to_matrix(w)  # (Np, 3, 3)
        if keep_best:
            # keep the coarse seed where refinement did not raise the NCC. Under
            # refine_pattern_center the gate reverts only the orientation, so seed_score (taken
            # at the seed pattern_center) is an approximation there -- a safe one.
            keep = (refined_score >= seed_score).reshape(-1, 1, 1)
            r_final = torch.where(keep, r_ref, r0_mat).cpu().numpy()
            score = torch.maximum(refined_score, seed_score)
        else:
            r_final = r_ref.cpu().numpy()
            score = refined_score

    from orix.quaternion import Rotation

    rotation = Rotation.from_matrix(r_final)

    out = [rotation, score]
    if refine_pattern_center:
        out.append(pc_param.detach())
    if return_history:
        out.append(torch.stack(history, dim=0))
    return tuple(out)
