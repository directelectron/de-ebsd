"""Gradient-based refinement (I6) -- the differentiable payoff.

The coarse indexer (:class:`torchebsd.index.indexer.SphericalIndexer`) is only as
accurate as its SOFT grid cell (``360 / (2L + 2)`` degrees -- a few degrees at
production bandwidth). This module sharpens a seed orientation (and, optionally, the
pattern center) to *sub-grid-cell* accuracy by continuous gradient ascent through the
**differentiable** forward model -- the motivation for torchebsd's whole PyTorch design.

**Spherical-domain NCC, reusing validated pieces.** The objective is a masked
normalized cross-correlation (NCC) between two functions on the unit sphere, evaluated
over a fixed set of field-of-view (FOV) directions ``n``:

* the **experimental** side ``E(n)`` is the pattern back-projected off the detector by
  :func:`torchebsd.index.backproject.pattern_to_sphere` -- which is *differentiable in
  the pattern center* (I3; validated autograd-vs-FD to ~1.7e-8). This is the pattern-
  center gradient path. **No** detector-domain forward projection (``master.project``)
  is used: that routes through ``detector.direction_cosines``, which is numpy and not
  differentiable in the pattern center.
* the **master** side ``M_R(n) = M(R n)`` is the continuous master pattern sampled at
  the *trial-rotated* directions via :meth:`torchebsd.sim.MasterPattern.sample` -- a
  ``grid_sample`` that is *differentiable in the sampling directions*. This is the
  orientation gradient path.

**Rotation parameterization and compose order (pinned).** The seed orientation ``R0``
(an orix ``Rotation`` from the coarse index) is turned into its rotation matrix
``R0_mat = quaternion_to_matrix(R0.data)`` -- the *same* matrix
:meth:`torchebsd.sim.MasterPattern.project` builds from a quaternion (validated: it
equals ``orix``'s ``Rotation.to_matrix()`` bit for bit). ``R0_mat`` is **detached**: it
is the fixed anchor. A small orientation correction is a rotation vector ``w`` (shape
``(Np, 3)``, ``requires_grad``, initialised to 0) mapped to ``SO(3)`` by the
differentiable exponential ``dR = matrix_exp(skew(w))``, and composed on the **right**:

    R = R0_mat @ dR(w),   so the refined orientation is  R0 . dR(w*).

Directions are rotated **exactly as the sim does**. ``project`` samples the master at
``v = R_mat @ dir`` per direction; for a stack of directions ``dirs`` this is
``v = dirs @ R.transpose(-1, -2)``, giving ``M_R(n) = M(R n)``. Back-projection makes
``E(n) = M(R_true n)`` (up to contrast, which the NCC removes), so ``E`` and ``M_R``
agree exactly when ``R = R_true`` -- i.e. at ``w = 0`` a *correct* seed already sits at
the objective's maximum and does **not** drift. Composing on the right means ``w*``
recovers ``dR* = R0_mat^T R_true_mat``, the residual the coarse cell left behind.

**Objective.** Both fields are windowed to the detector FOV (the mask from
``pattern_to_sphere``, held fixed at the seed geometry so the objective is smooth) and
scored by the masked, mean-subtracted, normalized dot product

    NCC = <E - <E>_W, M_R - <M_R>_W>_W / (||E - <E>_W||_W ||M_R - <M_R>_W||_W),

a contrast-invariant correlation coefficient in ``[-1, 1]``, differentiable in both
``w`` and (via ``E``) the pattern center. It is *maximized* (loss ``-NCC``) with Adam;
the per-pattern NCCs are summed into one scalar loss, so a whole stack refines in
parallel with fully independent per-pattern gradients (the pattern center is the one
optionally-shared parameter). The NCC is evaluated over the detector's **in-FOV** pixel
directions by default (:func:`_fov_directions`) rather than a whole-sphere set most of
whose points fall outside the field of view and are masked to zero -- ~3x cheaper per
iteration and more accurate, since every sample contributes. ~60 Adam steps at
``lr = 0.02`` take a coarse seed a few degrees off down to well under a degree; the
pattern center is refined with a smaller step (``pc_lr``, default ``lr / 5``) since its
natural scale is finer.

``lr`` is the key stability knob. The objective's basin around the true orientation is
sharp and deep for a clean, band-limited synthetic master but *shallow and noisy* for
real, low-resolution experimental patterns; too large a step then overshoots that basin
and the optimizer walks off into noise (an ``lr = 0.05`` that converges beautifully on a
synthetic master was observed to diverge by ~18 degrees on real Ni patterns). The
default ``0.02`` is stable on real data while still converging on synthetic; drop to
``~0.01`` for especially noisy scans.

**Second-order (Levenberg-Marquardt / Gauss-Newton) path -- ``method="lm"``.** The Adam
path above is first-order and takes ~60 iterations; the same sub-degree accuracy is
reached in ~10 steps by a batched Gauss-Newton refiner (``method="lm"``, orientation
only). It minimizes the *same* objective written as a least-squares residual,
``0.5 ||E_hat - M_hat(R)||^2`` over the SO(3) tangent ``delta`` (``R <- R . exp[skew
(delta)]``) -- which, since ``E_hat`` and ``M_hat`` are both zero-mean/unit-norm over
the directions, is exactly ``1 - NCC``, so minimizing the residual *maximizes* the NCC.
Because the LM path uses the detector's **in-FOV** pixel directions (the default
:func:`_fov_directions`, all inside the field of view), the mask is all-ones and the
standardization is a plain zero-mean/unit-norm -- which is what keeps its Jacobian clean.

The crux is an **analytic** Jacobian. Forward-mode AD does not work through
``grid_sample`` (it raises ``NotImplementedError``), so the master's per-point spatial
gradient is taken from one *reverse-mode* pass: with ``v = R n`` made a leaf,
``gM_i = dM_i/dv_i`` (the cross-terms vanish -- ``grid_sample`` is pointwise). The
tangent of ``v`` wrt ``delta_k`` is ``R (-skew(n))`` (columns ``k``); chaining gives the
raw ``g = dM/ddelta``, then the standardization derivative
``d m_hat_i/d delta_k = [(g_ik - mean_k(g)) - m_hat_i nu_k] / s`` with
``nu_k = sum_i c_i g_ik / s``, ``c = M - mean(M)``, ``s = ||c||``; the residual Jacobian
is ``J_r = -d m_hat/d delta`` (FD-checked to a tight tolerance on a smooth band master).
Each step solves the damped normal equations ``(JtJ + lam diag(JtJ)) delta = -Jtr`` (a
batched ``(Np, 3, 3)`` solve) and composes ``R <- R . exp[skew(delta)]``. To stay robust
to the bilinear ``grid_sample`` kinks, every step is **accepted/damped per pattern**: the
NCC is recomputed at the trial ``R``; where it improved the step is taken and ``lam``
shrinks (x0.5), where it did not the step is rejected (old ``R`` kept) and ``lam`` grows
(x3). That per-step accept/reject also makes the LM path monotone by construction (it
never leaves a pattern worse than its seed), the same guarantee ``keep_best`` gives Adam.
Measured ~2.3x faster than Adam at equal accuracy on a real Ni scan. It is
**orientation-only**: ``refine_pc=True`` falls back to the Adam path (joint pattern-center
+ orientation LM is future work), and a caller passing a non-FOV ``directions`` set with
``method="lm"`` breaks the all-ones-mask assumption the Jacobian relies on.

Because the master is *also* differentiable in the crystal parameters (DESIGN.md M4),
this same machinery extends to forward-model fitting of the structure. See
``INDEXING_DESIGN.md`` (milestone I6).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from torchebsd.index.backproject import pattern_to_sphere
from torchebsd.sim.rotation import quaternion_to_matrix

__all__ = ["refine_orientation"]

_EPS = 1e-12


def _skew(w: torch.Tensor) -> torch.Tensor:
    """Map rotation vectors ``w`` (..., 3) to skew-symmetric generators (..., 3, 3).

    ``matrix_exp(skew(w))`` is then the rotation by ``|w|`` radians about ``w / |w|``.
    Built with :func:`torch.stack` (no in-place writes) so it is autograd-friendly.
    """
    zero = torch.zeros_like(w[..., 0])
    row0 = torch.stack([zero, -w[..., 2], w[..., 1]], dim=-1)
    row1 = torch.stack([w[..., 2], zero, -w[..., 0]], dim=-1)
    row2 = torch.stack([-w[..., 1], w[..., 0], zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _masked_ncc(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked, mean-subtracted, normalized cross-correlation over the last axis.

    ``a``, ``b`` are ``(Np, M)``; ``mask`` is ``(M,)`` (shared FOV) or ``(Np, M)``
    (per-pattern FOV), non-negative, 1 inside the field of view and 0 outside. Returns
    a length-``Np`` correlation coefficient in ``[-1, 1]``, invariant to the (per-field)
    affine contrast of either input.
    """
    denom = mask.sum(-1, keepdim=True).clamp_min(_EPS)
    ma = (a * mask).sum(-1, keepdim=True) / denom
    mb = (b * mask).sum(-1, keepdim=True) / denom
    a0 = (a - ma) * mask
    b0 = (b - mb) * mask
    num = (a0 * b0).sum(-1)
    na = (a0 * a0).sum(-1).clamp_min(_EPS).sqrt()
    nb = (b0 * b0).sum(-1).clamp_min(_EPS).sqrt()
    return num / (na * nb)


def _fibonacci_directions(n: int, device, dtype) -> torch.Tensor:
    """A uniform Fibonacci-sphere set of ``n`` unit vectors ``(n, 3)``.

    The default direction set for the functional core when the caller does not pass its
    own (the :class:`~torchebsd.index.indexer.SphericalIndexer` passes its SHT quadrature
    grid). Uniform and dense, so a healthy fraction lands in any detector FOV -- ample
    support for the masked NCC.
    """
    i = torch.arange(n, device=device, dtype=dtype) + 0.5
    phi = math.pi * (1.0 + 5.0**0.5) * i  # golden-angle azimuth
    z = 1.0 - 2.0 * i / n  # even in cos(theta)
    r = (1.0 - z * z).clamp_min(0.0).sqrt()
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def _fov_directions(detector, device, dtype, target: int = 1024) -> torch.Tensor:
    """The detector's own pixel directions, subsampled to ~``target`` -- all in the FOV.

    The refinement objective is a masked NCC *over the field of view*, so directions
    outside it are pure waste: the master is sampled and gradient-tracked there every
    iteration only to be zeroed by the mask. The detector's ``direction_cosines`` are
    exactly the in-FOV sphere directions (one per pixel), so subsampling them gives a
    grid where **every** point contributes. Measured to refine as accurately as a
    whole-sphere Fibonacci set several times larger (of which only ~a third lands in the
    FOV) -- and with a markedly better worst case, since the hard patterns are no longer
    starved of support. This is the default direction set (the
    :class:`~torchebsd.index.indexer.SphericalIndexer` still passes its own SHT grid).
    """
    pix = detector.direction_cosines(device=device, dtype=dtype).reshape(-1, 3)
    step = max(1, pix.shape[0] // int(target))
    return pix[::step][: int(target)].contiguous()


def _backproject(patt, detector, dirs, pc, per_pattern):
    """Back-project ``patt`` (Np, H, W) onto ``dirs`` (M, 3), returning ``E`` and mask.

    Shared pattern center (``pc`` shape ``(3,)``) is one vectorized
    :func:`pattern_to_sphere` call with a shared ``(M,)`` mask; a per-pattern center
    (``pc`` shape ``(Np, 3)``) loops per pattern and stacks a ``(Np, M)`` mask. ``E`` is
    ``(Np, M)`` in both cases and carries the pattern-center autograd graph.
    """
    if not per_pattern:
        e, mask = pattern_to_sphere(patt, detector, dirs, pc=pc)
        return e, mask
    es, masks = [], []
    for i in range(patt.shape[0]):
        e_i, m_i = pattern_to_sphere(patt[i : i + 1], detector, dirs, pc=pc[i])
        es.append(e_i)
        masks.append(m_i)
    return torch.cat(es, dim=0), torch.stack(masks, dim=0)


def _standardize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-mean over the last axis; return ``(centered, norm)`` (norm clamped off 0).

    ``centered / norm`` is the zero-mean, unit-norm standardization of ``x``; its dot
    product with another standardized field is their normalized cross-correlation. The
    LM path standardizes over the *in-FOV* directions (the all-ones-mask assumption --
    module docstring), so this plain form is exactly the masked standardization there.
    """
    c = x - x.mean(-1, keepdim=True)
    return c, c.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def _lm_residual_jacobian(master, dirs, r_mat, e_hat, device, dtype):
    """Analytic residual ``r`` and its Jacobian ``J_r`` of the LM objective at ``r_mat``.

    Minimizing ``0.5 ||E_hat - M_hat(R)||^2`` over the SO(3) tangent ``delta``
    (``R <- R . exp[skew(delta)]``) with ``E_hat``, ``M_hat`` zero-mean/unit-norm over
    the in-FOV directions. The master's per-point spatial gradient
    ``gM_i = dM_i/dv_i`` comes from one **reverse-mode** pass through ``grid_sample`` (its
    forward-mode is unimplemented, and it is pointwise so the cross-terms vanish). The
    tangent of ``v = R n`` wrt ``delta_k`` is ``R (-skew(n))``; chaining gives the raw
    ``g = dM/ddelta``, then the standardization derivative

        d m_hat_i / d delta_k = [(g_ik - mean_k(g)) - m_hat_i nu_k] / s,
        nu_k = sum_i c_i g_ik / s,  c = M - mean(M),  s = ||c||.

    Parameters mirror the module docstring. Returns ``(r, J_r, M)`` -- the residual
    ``r = E_hat - M_hat`` ``(Np, M)``, its Jacobian ``J_r = -d m_hat/d delta``
    ``(Np, M, 3)``, and the raw master samples ``M`` ``(Np, M)`` at ``r_mat`` (reused to
    score the current NCC without re-sampling).
    """
    n = dirs  # (M, 3), all in the FOV
    # one reverse-mode pass gives the per-point spatial gradient dM/dv; grid_sample has
    # no forward-mode AD, and being pointwise its Jacobian is diagonal in the points.
    with torch.enable_grad():
        v = (n @ r_mat.transpose(-1, -2)).detach().requires_grad_(True)  # (Np, M, 3)
        m_raw = master.sample(v)  # (Np, M)
        (g_m,) = torch.autograd.grad(m_raw.sum(), v)  # (Np, M, 3) = dM_i/dv_i
    m = m_raw.detach().to(device=device, dtype=dtype)
    g_m = g_m.to(device=device, dtype=dtype)
    # tangent of v = R n wrt delta_k: R @ (-skew(n)), columns k -> (Np, M, 3, 3)
    tang = r_mat[:, None] @ (-_skew(n))[None]
    g = torch.einsum("pma,pmak->pmk", g_m, tang)  # raw dM/ddelta (Np, M, 3)
    c, s = _standardize(m)
    m_hat = c / s
    mu_g = g.mean(1)  # mean_k(g) over the directions (Np, 3)
    nu = torch.einsum("pm,pmk->pk", c, g) / s  # (Np, 3)
    dm_hat = ((g - mu_g[:, None]) - m_hat[..., None] * nu[:, None]) / s[..., None]
    return e_hat - m_hat, -dm_hat, m


def _lm_refine(
    master, dirs, r0_mat, e, mask, n_steps, lam0, keep_best, return_history, device, dtype
):
    """Batched Levenberg-Marquardt / Gauss-Newton orientation refinement.

    Sharpens the seed matrices ``r0_mat`` ``(Np, 3, 3)`` by the second-order method of
    the module docstring, fully vectorized over the ``Np`` patterns. ``e`` / ``mask`` are
    the back-projected experimental field and its (all-ones, in-FOV) mask; the NCC that
    drives per-step acceptance and the returned score is the house
    :func:`_masked_ncc`. Returns ``(r_final (Np, 3, 3), score (Np,), history)`` --
    ``history`` is the per-step NCC ``(n_steps, Np)`` when ``return_history`` else
    ``None`` (``history[0]`` is the seed's NCC).
    """
    e_c, e_s = _standardize(e)
    e_hat = e_c / e_s  # (Np, M) standardized experimental field
    r_mat = r0_mat.clone()
    n_p = r_mat.shape[0]
    lam = torch.full((n_p,), float(lam0), device=device, dtype=dtype)

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
            # LM damping: add lam * diag(JtJ) to the Gauss-Newton normal matrix.
            diag = torch.diagonal(jtj, dim1=-2, dim2=-1).clamp_min(1e-9)
            a = jtj + lam[:, None, None] * torch.diag_embed(diag)
            delta = -torch.linalg.solve(a, jtr[..., None])[..., 0]  # (Np, 3)
            r_new = r_mat @ torch.matrix_exp(_skew(delta))
            m_new = master.sample(dirs @ r_new.transpose(-1, -2)).to(device=device, dtype=dtype)
            new_score = _masked_ncc(e, m_new, mask)
            # per-pattern accept/reject: take the step where the NCC improved (shrink
            # lam), otherwise keep the old R (grow lam). This is what makes LM robust to
            # the bilinear grid_sample kinks -- and monotone by construction.
            better = new_score > cur_score
            r_mat = torch.where(better[:, None, None], r_new, r_mat)
            lam = torch.where(better, (lam * 0.5).clamp_min(1e-6), (lam * 3.0).clamp_max(1e2))
            cur_score = torch.where(better, new_score, cur_score)

    refined_score = cur_score
    if keep_best:
        # LM is already monotone (accept/reject never lowers cur_score below the seed),
        # so this only reasserts the guarantee -- symmetric with the Adam path.
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
    pc=None,
    refine_pc=False,
    method="adam",
    n_iters=None,
    lr=0.02,
    pc_lr=None,
    lm_lambda=1e-3,
    directions=None,
    device=None,
    keep_best=True,
    return_history=False,
):
    """Refine ``rotations`` (and optionally the pattern center) by autograd.

    Sharpens a coarse seed orientation to sub-grid-cell accuracy by maximizing a
    spherical masked normalized cross-correlation between the back-projected
    experimental pattern ``E(n)`` and the master sampled at the trial-rotated directions
    ``M(R n)``, with ``R = R0 . dR(w)`` and ``w`` a differentiable rotation-vector
    correction (module docstring for the full method and the pinned compose order).
    Torch-native and device-following (everything runs on the master's device/dtype);
    the whole stack is refined in one parallel optimization.

    Parameters
    ----------
    patterns
        A single detector image ``(H, W)`` or a stack ``(Np, H, W)`` (numpy array or
        torch tensor; ``H = detector.nrows``, ``W = detector.ncols``). No kikuchipy
        needed -- a kikuchipy signal's ``.data`` passes straight in.
    master
        The :class:`torchebsd.sim.MasterPattern` -- the continuous ``M(n)`` sampled
        (differentiably in the sampling directions) via ``master.sample``.
    detector
        The :class:`torchebsd.sim.detector.EBSDDetector` geometry; its pixel -> direction
        map is inverted (differentiably in the pattern center) to build ``E(n)``.
    rotations
        The seed :class:`orix.quaternion.Rotation` (length ``Np``, or length 1 / scalar
        for a single pattern) -- typically the coarse ``index`` output.
    pc
        Pattern center. ``None`` (default) uses ``detector.pc``. A length-3 tensor/array
        ``(pcx, pcy, pcz)`` shares one center across the stack (typical -- one detector
        geometry); an ``(Np, 3)`` tensor/array gives a per-pattern center. Shared is the
        common, fully-vectorized path; per-pattern loops the back-projection.
    refine_pc
        If ``True``, the pattern center is a free parameter refined jointly with the
        orientation (the I3 pattern-center gradient path). If ``False`` (default) it is
        held fixed and returned untouched. **``refine_pc=True`` forces ``method="adam"``**
        -- the ``"lm"`` path is orientation-only (joint pattern-center + orientation LM is
        future work), so it silently falls back to Adam here (still returning the refined
        ``pc``).
    method
        Which optimizer sharpens the orientation: ``"adam"`` (default) is the first-order
        Adam path described in the module docstring; ``"lm"`` is the second-order
        Levenberg-Marquardt / Gauss-Newton path (analytic Jacobian from the reverse-mode
        ``grid_sample`` gradient, per-step damping/acceptance for the ``grid_sample``
        kinks), which reaches the same sub-degree accuracy in ~10 steps instead of ~60
        (~2.3x faster at equal accuracy on real data). ``"lm"`` is **orientation-only**
        and assumes the default in-FOV ``directions`` (its all-ones-mask standardization);
        ``refine_pc=True`` or a non-FOV ``directions`` set means Adam. ``lr`` / ``pc_lr``
        are Adam-only; ``lm_lambda`` is LM-only.
    n_iters
        Iteration/step budget. ``None`` (default) resolves per method: **60** for
        ``"adam"``, **12** for ``"lm"`` (its second-order steps need far fewer). With the
        default in-FOV directions the seed (a coarse index cell, a degree or two off)
        reaches sub-degree accuracy well inside either budget; raise it for a tighter fit,
        lower it for more speed.
    lr
        Adam learning rate for the rotation-vector correction ``w`` (radians; default
        0.02). Adam-only (ignored for ``method="lm"``). The key stability knob (see the
        module docstring): too large a step overshoots the shallow, noisy objective basin
        of real experimental patterns and the optimizer diverges. ``0.02`` is stable on
        real data and converges on synthetic; use ``~0.01`` for especially noisy scans.
    pc_lr
        Adam learning rate for the pattern center when ``refine_pc`` (default
        ``lr / 5``): the pattern center's natural scale is finer than the orientation's,
        so it takes smaller steps. Ignored when ``refine_pc`` is ``False``.
    lm_lambda
        Initial Levenberg-Marquardt damping ``lam`` for ``method="lm"`` (default
        ``1e-3``); it is adapted per pattern each step (x0.5 on an accepted step, x3 on a
        rejected one). LM-only (ignored for ``method="adam"``).
    directions
        Sphere directions ``(M, 3)`` over which the NCC is evaluated (the FOV is masked
        out of these). ``None`` (default) uses the detector's own **in-FOV** pixel
        directions (:func:`_fov_directions`), where every point contributes -- ~3x faster
        and more accurate than a whole-sphere set of which most points fall outside the
        FOV. Pass an explicit set to override (the
        :meth:`~torchebsd.index.indexer.SphericalIndexer.refine` convenience passes its
        SHT quadrature grid so the SOFT refinement and coarse index share one grid).
        ``method="lm"`` **requires** an all-in-FOV set (its standardization assumes an
        all-ones mask); the default satisfies this, but a custom non-FOV set is unsafe for
        LM (use ``method="adam"``, whose masked NCC handles out-of-FOV directions).
    device
        Torch device to run the optimization on. ``None`` (default) uses
        ``master.data.device``; when given, the master, patterns and seeds are moved
        there (:meth:`~torchebsd.index.indexer.SphericalIndexer.refine` passes the
        indexer's own device). The result ``Rotation`` is device-independent; the
        returned ``score`` (and ``pc``) live on this device.
    keep_best
        If ``True`` (default), refinement never returns an orientation *worse* than the
        seed: for any pattern whose refined NCC does not exceed its seed NCC, the coarse
        seed orientation (and seed score) is kept. This makes refinement monotone -- it
        can only sharpen, never degrade -- which removes the stray outliers that gradient
        refinement can introduce on low-quality real patterns. Set ``False`` to always
        take the optimizer's result.
    return_history
        If ``True``, also return the per-iteration (Adam) / per-step (LM) NCC trajectory
        (see Returns).

    Returns
    -------
    rotation : orix.quaternion.Rotation
        The refined orientation(s), length ``Np``. ``rotation[i]`` is
        ``R0[i] . dR(w*[i])`` (Adam) or the accepted LM composition, converted back to
        orix (matrix -> quaternion via ``Rotation.from_matrix``), in the *same* convention
        the coarse index returns and ``MasterPattern.project`` consumes.
    score : torch.Tensor
        The final NCC of each pattern, length ``Np`` (detached), a
        correlation-coefficient-like number in ``[-1, 1]`` -- refinement quality, ``>=``
        the coarse score. Always the second element. This is the same family of
        confidence metric as :meth:`~torchebsd.index.indexer.SphericalIndexer.index`'s
        ``quality``, and is what
        :meth:`~torchebsd.index.indexer.SphericalIndexer.index_scan` with
        ``refine=True`` stores as ``prop["quality"]`` on the returned ``CrystalMap``.
    pc : torch.Tensor
        *Only when* ``refine_pc``: the refined pattern center (detached), same shape as
        the input ``pc``. Third element.
    history : torch.Tensor
        *Only when* ``return_history``: the NCC at every iteration (Adam) / step (LM),
        shape ``(n_iters, Np)`` (``history[0]`` is the seed's score). Last element.
    """
    # method resolution: LM is orientation-only, so refine_pc forces the Adam path; the
    # step budget then defaults per method (Adam 60, LM 12) when n_iters is unset.
    method = str(method).lower()
    if method not in ("adam", "lm"):
        raise ValueError(f"method must be 'adam' or 'lm', got {method!r}")
    if method == "lm" and refine_pc:
        method = "adam"  # joint pattern-center + orientation LM is future work
    if n_iters is None:
        n_iters = 12 if method == "lm" else 60

    device = master.data.device if device is None else torch.device(device)
    dtype = master.data.dtype
    # honor an explicit device: move the master there so master.sample and the
    # back-projection run on it (master.data.device drives sampling).
    if master.data.device != device:
        master = master.to(device)

    patt = torch.as_tensor(patterns, device=device, dtype=dtype)
    single = patt.ndim == 2
    if single:
        patt = patt[None]
    if patt.ndim != 3:
        raise ValueError(f"patterns must be (H, W) or (Np, H, W), got shape {tuple(patt.shape)}")
    n_p = patt.shape[0]

    # seed matrix R0 = quaternion_to_matrix(seed) -- the detached anchor, exactly the
    # matrix MasterPattern.project builds from a quaternion.
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
    if pc is None:
        pc_init = torch.as_tensor(detector.pc, device=device, dtype=dtype)
    else:
        pc_init = torch.as_tensor(pc, device=device, dtype=dtype)
    if pc_init.ndim == 1 and pc_init.shape[0] == 3:
        per_pattern = False
    elif pc_init.ndim == 2 and pc_init.shape == (n_p, 3):
        per_pattern = True
    else:
        raise ValueError(
            f"pc must be (3,) shared or (Np, 3) per-pattern (Np={n_p}), got "
            f"{tuple(pc_init.shape)}"
        )

    # ------------------------------------------------------------------ #
    # Second-order (Levenberg-Marquardt / Gauss-Newton) path, orientation only. The
    # pattern center is fixed (refine_pc already forced Adam above), so E is back-
    # projected once; the LM core does the batched analytic-Jacobian steps. See the
    # module docstring for the method.
    # ------------------------------------------------------------------ #
    if method == "lm":
        with torch.no_grad():
            e_lm, mask_lm = _backproject(patt, detector, dirs, pc_init, per_pattern)
        r_final, score, hist = _lm_refine(
            master, dirs, r0_mat, e_lm, mask_lm, n_iters, lm_lambda, keep_best,
            return_history, device, dtype,
        )
        from orix.quaternion import Rotation

        rotation = Rotation.from_matrix(r_final.cpu().numpy())
        out = [rotation, score]
        if return_history:
            out.append(hist)
        return tuple(out)

    # the FOV mask is held FIXED at the seed geometry so the objective stays smooth as
    # the pattern center moves (only the bilinear intensity read-off varies with pc).
    with torch.no_grad():
        _, mask = _backproject(patt, detector, dirs, pc_init, per_pattern)

    w = torch.zeros(n_p, 3, device=device, dtype=dtype, requires_grad=True)
    params = [{"params": [w], "lr": lr}]
    if refine_pc:
        pc_param = pc_init.clone().requires_grad_(True)
        params.append({"params": [pc_param], "lr": pc_lr if pc_lr is not None else lr / 5.0})
    else:
        pc_param = pc_init
        # pattern center is fixed -> E does not change; back-project it once.
        e_fixed, _ = _backproject(patt, detector, dirs, pc_param, per_pattern)

    opt = torch.optim.Adam(params)
    history = [] if return_history else None

    def _score():
        if refine_pc:
            e = _backproject(patt, detector, dirs, pc_param, per_pattern)[0]
        else:
            e = e_fixed
        d_r = torch.matrix_exp(_skew(w))  # (Np, 3, 3)
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
        r_ref = r0_mat @ torch.matrix_exp(_skew(w))  # (Np, 3, 3)
        if keep_best:
            # never return an orientation worse than the seed: where refinement did not
            # raise the NCC (e.g. it overshot the shallow, noisy basin of a low-quality
            # real pattern) keep the coarse seed. Guarantees monotonic non-degradation
            # and removes the stray outliers refinement can introduce. (For refine_pc the
            # shared refined pattern center is still returned; the gate reverts only the
            # per-pattern orientation, so seed_score -- taken at the seed pc -- is an
            # approximation there, but a safe one.)
            keep = (refined_score >= seed_score).reshape(-1, 1, 1)
            r_final = torch.where(keep, r_ref, r0_mat).cpu().numpy()
            score = torch.maximum(refined_score, seed_score)
        else:
            r_final = r_ref.cpu().numpy()
            score = refined_score

    from orix.quaternion import Rotation

    rotation = Rotation.from_matrix(r_final)

    out = [rotation, score]
    if refine_pc:
        out.append(pc_param.detach())
    if return_history:
        out.append(torch.stack(history, dim=0))
    return tuple(out)
