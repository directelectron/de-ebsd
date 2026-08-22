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
optionally-shared parameter). ~100 Adam steps at ``lr = 0.02`` take a coarse seed a few
degrees off down to well under half a degree; the pattern center is refined with a
smaller step (``pc_lr``, default ``lr / 5``) since its natural scale is finer.

``lr`` is the key stability knob. The objective's basin around the true orientation is
sharp and deep for a clean, band-limited synthetic master but *shallow and noisy* for
real, low-resolution experimental patterns; too large a step then overshoots that basin
and the optimizer walks off into noise (an ``lr = 0.05`` that converges beautifully on a
synthetic master was observed to diverge by ~18 degrees on real Ni patterns). The
default ``0.02`` is stable on real data while still converging on synthetic; drop to
``~0.01`` for especially noisy scans.

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


def refine_orientation(
    patterns,
    master,
    detector,
    rotations,
    pc=None,
    refine_pc=False,
    n_iters=100,
    lr=0.02,
    pc_lr=None,
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
        held fixed and returned untouched.
    n_iters
        Number of Adam iterations (default 100).
    lr
        Adam learning rate for the rotation-vector correction ``w`` (radians; default
        0.02). The key stability knob (see the module docstring): too large a step
        overshoots the shallow, noisy objective basin of real experimental patterns and
        the optimizer diverges. ``0.02`` is stable on real data and converges on
        synthetic; use ``~0.01`` for especially noisy scans.
    pc_lr
        Adam learning rate for the pattern center when ``refine_pc`` (default
        ``lr / 5``): the pattern center's natural scale is finer than the orientation's,
        so it takes smaller steps. Ignored when ``refine_pc`` is ``False``.
    directions
        Sphere directions ``(M, 3)`` over which the NCC is evaluated (the FOV is masked
        out of these). ``None`` (default) uses a uniform Fibonacci-sphere set; the
        :meth:`~torchebsd.index.indexer.SphericalIndexer.refine` convenience passes its
        SHT quadrature grid so the refinement and the coarse index share one grid.
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
        If ``True``, also return the per-iteration NCC trajectory (see Returns).

    Returns
    -------
    rotation : orix.quaternion.Rotation
        The refined orientation(s), length ``Np``. ``rotation[i]`` is
        ``R0[i] . dR(w*[i])`` converted back to orix (matrix -> quaternion via
        ``Rotation.from_matrix``), in the *same* convention the coarse index returns and
        ``MasterPattern.project`` consumes.
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
        *Only when* ``return_history``: the NCC at every iteration, shape
        ``(n_iters, Np)`` (``history[0]`` is the seed's score at ``w = 0``). Last element.
    """
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
        dirs = _fibonacci_directions(2500, device, dtype)
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
