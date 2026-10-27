"""Strain from a live-indexed pattern: orientation and deformation fitted together.

:func:`refine_deformation` fits each pattern against the master with eight free
numbers (three rotations, five strains), :func:`strain_from_reference`
cross-correlates real regions against a reference frame instead, and
:func:`center_maps` lets every pattern of a scan have its own pattern center.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from de_ebsd.core.constants import SINGLE_PRECISION_FLOOR
from de_ebsd.index.live.common import BATCH
from de_ebsd.index.preprocess import normalize_rows

#: Gauss-Newton iterations of :func:`refine_deformation`. A plain step
#: closes only about half of what is left, because the simulation is sharper than
#: the pattern; with :data:`DEFORMATION_LINE_SEARCH` three iterations do what seven
#: plain ones did, for half the projections. On a 15 kV silicon indent, against
#: CrossCourt's HR-EBSD, the four strongest strains reached 0.94-1.05 of their size
#: (0.64-0.87 at three plain iterations). Iterating on to convergence came no
#: closer but scattered more (rms difference 5.0-6.5e-4 against 3.4-5.4e-4).
DEFORMATION_ITERATIONS = 3

#: The multiples of each Gauss-Newton step that are tried, keeping whichever scores
#: best (or none, if all score worse than the current fit). Three more projections
#: per pattern on top of an iteration's seventeen.
DEFORMATION_LINE_SEARCH = (1.0, 2.0, 3.0)

#: The central-difference step of the deformation's Jacobian, as a strain. Much
#: smaller steps measure the bilinear master texture's facets rather than its slope.
DEFORMATION_STEP = 2e-3

#: The largest change one Gauss-Newton iteration may make to any of the
#: deformation's eight numbers. A step is the solve of a local linear model, and on
#: a poor pattern (a grain boundary, two patterns overlapping) that model can
#: point far outside the elastic range; without the limit 1-3% of a real scan's
#: fits ran off to strains of 0.1 and more.
DEFORMATION_MAX_STEP = 5e-3

#: How far below its seed's score a deformation fit may end before it is refused:
#: a fit that only ties its seed loses that much to rounding.
DEFORMATION_SCORE_MARGIN = 1e-4


def trace_free_basis(device) -> torch.Tensor:
    """The 8 trace-free 3x3 matrices ``(8, 3, 3)``, orthonormal in the Frobenius
    sense: rotations about x, y and z, then the five deviatoric strains. A
    projection cannot see the ninth, hydrostatic, component."""
    root2, root6 = math.sqrt(2.0), math.sqrt(6.0)
    basis = torch.zeros(8, 3, 3, dtype=torch.float64)
    for k, (i, j) in enumerate(((1, 2), (2, 0), (0, 1))):
        basis[k, i, j], basis[k, j, i] = -1 / root2, 1 / root2
        basis[3 + k, i, j] = basis[3 + k, j, i] = 1 / root2
    basis[6] = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)) / root2
    basis[7] = torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=torch.float64)) / root6
    return basis.to(device, torch.float32)


def polar_parts(beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The deformation gradient ``F = (I - beta)^-1`` of a displacement gradient
    *beta* ``(N, 3, 3)``, split as ``F = R U``: ``(R, U - I)``, the lattice rotation
    and the (Biot) strain, both in the sample frame."""
    eye = torch.eye(3, device=beta.device, dtype=torch.float64)
    f = torch.linalg.inv(eye - beta.double())
    u, s, vt = torch.linalg.svd(f)
    stretch = vt.transpose(-1, -2) @ torch.diag_embed(s) @ vt
    return (u @ vt).to(beta.dtype), (stretch - eye).to(beta.dtype)


def center_maps(detector, pattern_centers, device) -> torch.Tensor:
    """``(N, 3, 3)`` maps taking the refine's directions at the *detector*'s pattern
    center to those at each of *pattern_centers* ``(N, 3)``, up to length. *detector* is at the
    refine shape.

    On a flat screen a pixel's position is linear in its index at any center,
    so moving the center is exactly a linear map of the directions: in a
    candidate ``g @ m`` it is one more matrix, and the fused kernels, which
    normalize what they are given, need nothing new. It also means that, within
    one pattern, a center error is indistinguishable from a strain, which is
    why a scan's centers must come from its geometry rather than from each
    pattern.
    """
    from de_ebsd.core.geometry import sample_to_detector_matrix

    def screen(pattern_center):
        # (col, row, 1) -> the unnormalized screen position, as direction_cosines
        b = replace(
            detector, pattern_center=tuple(float(v) for v in pattern_center)
        ).gnomonic_bounds
        x_scale, y_scale = (b[1] - b[0]) / detector.ncols, (b[3] - b[2]) / detector.nrows
        z = float(pattern_center[2])
        return np.array(
            [
                [x_scale * z, 0.0, (b[0] + 0.5 * x_scale) * z],
                [0.0, -y_scale * z, (b[3] - 0.5 * y_scale) * z],
                [0.0, 0.0, z],
            ]
        )

    frame = sample_to_detector_matrix(
        detector.sample_tilt, detector.tilt, detector.azimuthal, detector.twist
    )
    to_index = np.linalg.inv(screen(detector.pattern_center)) @ frame
    maps = [
        frame.T @ screen(pattern_center) @ to_index
        for pattern_center in np.asarray(pattern_centers, float)
    ]
    return torch.as_tensor(np.stack(maps), device=device, dtype=torch.float32)


def refine_deformation(
    indexer,
    refine_rows: torch.Tensor,
    phase: torch.Tensor,
    g: torch.Tensor,
    *,
    iterations: int = DEFORMATION_ITERATIONS,
    line_search: tuple[float, ...] = DEFORMATION_LINE_SEARCH,
    step: float = DEFORMATION_STEP,
    center_maps: torch.Tensor | None = None,
    ghost: torch.Tensor | None = None,
    strain: bool = True,
):
    """Each orientation *g* ``(N, 3, 3)`` refined together with its pattern's
    deviatoric strain: ``(g, strain, score)``, *g* the lattice's orientation,
    *strain* ``(N, 3, 3)`` the symmetric, trace-free Biot strain in the sample
    frame (the hydrostatic part is invisible to a projection), *score* the fit's.

    A crystal deformed by ``F`` shows the master's pattern at ``g F^-1 r`` for a
    ray ``r``, so a candidate is the matrix ``g (I - beta)`` with *beta* the
    eight trace-free numbers -- three rotations and five strains -- that the
    fused kernels project like any orientation. Each Gauss-Newton iteration
    projects the current fit and its 16 central differences and solves the 8x8
    normal equations of the normalized pattern, for every pattern at once.
    Fitting the rotations alongside the strains is what sharpens the
    orientations: a strain left out of the model is partly fitted as a turn.

    *g* should already be within a few tenths of a degree (:meth:`LiveIndexer.refine`).
    *center_maps* (:meth:`center_maps`) give each pattern its own center.
    *ghost* (:attr:`ghost`) is taken out of the patterns and the simulations
    alike, so every pattern may have any amount of it.

    *line_search* tries multiples of each Gauss-Newton step, keeping the best
    scored (no move, if none scores better than the current fit); an empty
    one takes each step as it is. Each step is at most
    :data:`DEFORMATION_MAX_STEP`, so a move is at most that times the
    largest multiple. A fit that ends scoring below its seed
    (:data:`DEFORMATION_SCORE_MARGIN`) is refused: its strain is NaN and its
    *g* and *score* are the seed's.

    With *strain* False only the three rotations are fitted, and *strain* comes
    back zero: the orientation-only fit, 7 projections an iteration rather than
    17. On a real Zenith Cu scan its single crystal's neighbouring orientations
    then differed by 0.20° at the 90th percentile, against 0.35° fitted with the
    strains and 1.70° from :meth:`LiveIndexer.refine` alone.
    """
    if not strain and getattr(indexer, "fused", False):
        return _fit_orientations_fused(
            indexer,
            refine_rows,
            phase,
            g,
            iterations=iterations,
            line_search=line_search,
            step=step,
            center_maps=center_maps,
            ghost=ghost,
        )
    basis = trace_free_basis(indexer.device)
    if not strain:
        basis = basis[:3]
    m = len(basis)
    n, k = len(g), 1 + 2 * m
    eye = torch.eye(3, device=indexer.device)
    offsets = step * torch.cat(
        [torch.zeros(1, m, device=indexer.device), torch.eye(m, device=indexer.device)]
    )
    offsets = torch.cat([offsets, -offsets[1:]])  # (k, m): the fit, +each, -each
    damping = 1e-3 * torch.eye(m, device=indexer.device)
    p = torch.zeros(n, m, device=indexer.device)
    if ghost is not None:
        refine_rows = normalize_rows(refine_rows - (refine_rows @ ghost)[:, None] * ghost)

    def on_ghost(simulated):
        """Each simulation's component along the ghost. The simulations are
        never taken off it: every product below is corrected instead, which
        is one pass over them rather than several."""
        if ghost is None:
            return torch.zeros(simulated.shape[:-1], device=indexer.device)
        return simulated @ ghost

    seed_score = None
    for _ in range(iterations):
        beta = torch.einsum("nkc,cij->nkij", p[:, None] + offsets, basis)
        candidates = g[:, None] @ (eye - beta)
        if center_maps is not None:
            candidates = candidates @ center_maps[:, None]
        simulated = indexer.project(
            indexer.texture,
            candidates.reshape(-1, 3, 3),
            phase.repeat_interleave(k),
            indexer.refine_directions,
        )
        simulated = normalize_rows(simulated).view(n, k, -1)
        along = on_ghost(simulated)
        fit, fit_along = simulated[:, 0], along[:, :1]
        jacobian = (simulated[:, 1 : 1 + m] - simulated[:, 1 + m :]) / (2 * step)
        jacobian_along = (along[:, 1 : 1 + m] - along[:, 1 + m :]) / (2 * step)
        # the pattern's contrast is free: the residual is against scale * fit,
        # both off the ghost (the patterns already are, so only the fit's
        # length and the products with the Jacobian need correcting)
        scale = (refine_rows * fit).sum(-1, keepdim=True) / (1 - fit_along**2)
        if seed_score is None:
            seed_score = scale[:, 0] * (1 - fit_along[:, 0] ** 2).sqrt()
        residual = refine_rows - scale * fit
        normal = torch.einsum("nkp,nlp->nkl", jacobian, jacobian)
        normal = normal - jacobian_along[:, :, None] * jacobian_along[:, None, :]
        normal = normal + damping * normal.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
        gradient = (
            torch.einsum("nkp,np->nk", jacobian, residual) + scale * jacobian_along * fit_along
        ) / scale.clamp_min(SINGLE_PRECISION_FLOOR)
        change = torch.linalg.solve(normal, gradient)
        reach = change.abs().amax(-1, keepdim=True)
        change = change * (DEFORMATION_MAX_STEP / reach.clamp_min(DEFORMATION_MAX_STEP))
        if not line_search:
            p = p + change
            continue
        # The simulation is sharper than the pattern, so each step falls short
        # by about the same share: try it longer too, and keep the best scored.
        # Capping the longer moves as well cost a third of the strain near an
        # indent, where the real steps are largest.
        multiples = torch.tensor((0.0, *line_search), device=indexer.device)
        tried = p[:, None] + multiples[None, :, None] * change[:, None]
        trial = g[:, None] @ (eye - torch.einsum("nmc,cij->nmij", tried[:, 1:], basis))
        if center_maps is not None:
            trial = trial @ center_maps[:, None]
        simulated = normalize_rows(
            indexer.project(
                indexer.texture,
                trial.reshape(-1, 3, 3),
                phase.repeat_interleave(len(line_search)),
                indexer.refine_directions,
            )
        )
        scores = (simulated.view(n, len(line_search), -1) * refine_rows[:, None]).sum(-1)
        scores = scores / (1 - on_ghost(simulated).view(n, -1) ** 2).sqrt()
        current = scale[:, 0] * (1 - fit_along[:, 0] ** 2).sqrt()
        scores = torch.cat([current[:, None], scores], 1).nan_to_num(-torch.inf)
        p = tried[torch.arange(n, device=indexer.device), scores.argmax(1)]
    beta = torch.einsum("nc,cij->nij", p, basis)
    final = g @ (eye - beta)
    if center_maps is not None:
        final = final @ center_maps
    simulated = normalize_rows(
        indexer.project(indexer.texture, final, phase, indexer.refine_directions)
    )
    score = (simulated * refine_rows).sum(-1) / (1 - on_ghost(simulated) ** 2).sqrt()
    rotation, strain = polar_parts(beta)
    if m < len(trace_free_basis(indexer.device)):
        # a rotation as I - beta is a rotation only to first order: its second
        # order is a stretch of ~beta^2/2, not a strain the fit measured
        strain = torch.zeros_like(strain)
    fitted = g @ rotation.transpose(-1, -2)
    if seed_score is not None:
        refused = ~(score >= seed_score - DEFORMATION_SCORE_MARGIN)  # NaN is refused too
        strain = torch.where(refused[:, None, None], torch.nan, strain)
        fitted = torch.where(refused[:, None, None], g, fitted)
        score = torch.where(refused, seed_score, score)
    return fitted, strain, score


_TEXTURE_MEANS: dict[tuple[int, int], float] = {}


def _texture_mean(texture: torch.Tensor) -> float:
    """The mean of a master *texture*, once per texture: reading it waits for the
    device, which each iteration of the fit should not."""
    key = (texture.data_ptr(), texture.numel())
    if key not in _TEXTURE_MEANS:
        _TEXTURE_MEANS[key] = float(texture.mean())
    return _TEXTURE_MEANS[key]


#: Whether the fused orientation fit takes its Jacobian from the master's
#: gradient (one sample a pixel) rather than central differences (seven).
ANALYTIC_JACOBIAN = True
#: The fused orientation fit's passes, ``(coarse, alpha)`` each: scored at half
#: the refine's resolution or not, and moved by alpha times the Gauss-Newton
#: step (capped as :data:`DEFORMATION_MAX_STEP`). No line search and no final
#: pass: every pass's sums score where it stands, and the best-scored
#: full-resolution point is kept. Empty for the line-searched fit instead
#: (:data:`DEFORMATION_ITERATIONS`, :data:`DEFORMATION_LINE_SEARCH`).
#: On a customer Zenith's Cu scan, from a refine stopped at 1 degree, (full, half,
#: half, half, full) fitted its single crystal's neighbours to 0.17/0.25 degrees
#: at the 90th/99th percentile in 2.5 ms per 256 patterns, where the
#: line-searched fit took 6.1 ms (and a refine on to 0.5 degrees) for 0.22/0.34.
FIT_SCHEDULE: tuple[tuple[bool, float], ...] = (
    (False, 1.0),
    (True, 1.0),
    (True, 1.0),
    (True, 1.0),
    (False, 1.0),
)
#: The largest change one of :data:`FIT_SCHEDULE`'s passes may make, before its
#: alpha. Six times :data:`DEFORMATION_MAX_STEP`: a pass has no line search to
#: lengthen its step, and with that cap the schedule's fit fell short (0.62
#: degrees at the 90th percentile on that scan).
FIT_MAX_STEP = 3e-2

_TEXTURE_GRADIENTS: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _texture_gradients(texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A master *texture*'s derivatives along its columns and rows, per texel, by
    central differences (one-sided at its edges), once per texture: sampled
    bilinearly they are the slope of the master, where the bilinear sample's
    own derivative is the slope of its facets."""
    key = (texture.data_ptr(), texture.numel())
    if key not in _TEXTURE_GRADIENTS:
        t = texture.float().contiguous()
        x, y = torch.empty_like(t), torch.empty_like(t)
        x[..., 1:-1] = 0.5 * (t[..., 2:] - t[..., :-2])
        x[..., 0], x[..., -1] = t[..., 1] - t[..., 0], t[..., -1] - t[..., -2]
        y[..., 1:-1, :] = 0.5 * (t[..., 2:, :] - t[..., :-2, :])
        y[..., 0, :], y[..., -1, :] = t[..., 1, :] - t[..., 0, :], t[..., -1, :] - t[..., -2, :]
        _TEXTURE_GRADIENTS[key] = (x.contiguous(), y.contiguous())
    return _TEXTURE_GRADIENTS[key]


def _fit_orientations_fused(
    indexer, refine_rows, phase, g, *, iterations, line_search, step, center_maps, ghost
):
    """:func:`refine_deformation` without the strains on the fused path: every
    iteration of every pattern in one launch
    (:func:`~de_ebsd.index._kernels.fit_orientations`), one block per pattern,
    rather than some 400 small operations a batch between projections. The same
    arithmetic, so the same fit to float32 rounding (``tests/index/test_live.py``)."""
    from de_ebsd.index import _kernels

    if ANALYTIC_JACOBIAN and FIT_SCHEDULE:
        from de_ebsd.index.live.common import average_to

        shape, coarse_shape = indexer.refine_shape, indexer.coarse_shape

        def coarse(rows):
            return normalize_rows(average_to(rows.view(len(rows), *shape), coarse_shape).flatten(1))

        fitted, score, refused = _kernels.fit_passes(
            indexer.texture,
            g,
            phase * 2,
            center_maps,
            (indexer.refine_directions, refine_rows, ghost),
            (
                indexer.coarse_directions,
                coarse(refine_rows),
                coarse(ghost[None])[0] if ghost is not None else None,
            ),
            FIT_SCHEDULE,
            max_step=FIT_MAX_STEP,
            damping=1e-3,
            margin=DEFORMATION_SCORE_MARGIN,
            offset=_texture_mean(indexer.texture),
            floor=SINGLE_PRECISION_FLOOR,
            gradients=_texture_gradients(indexer.texture),
        )
        strain = torch.where(
            refused[:, None, None], torch.nan, torch.zeros(len(g), 3, 3, device=g.device)
        )
        return fitted, strain, score
    fitted, score, refused = _kernels.fit_orientations(
        indexer.texture,
        g,
        phase * 2,
        indexer.refine_directions,
        refine_rows,
        ghost,
        center_maps,
        iterations=iterations,
        step=step,
        max_step=DEFORMATION_MAX_STEP,
        multiples=tuple(line_search) or (1.0,),
        damping=1e-3,
        margin=DEFORMATION_SCORE_MARGIN,
        offset=_texture_mean(indexer.texture),
        floor=SINGLE_PRECISION_FLOOR,
        gradients=_texture_gradients(indexer.texture) if ANALYTIC_JACOBIAN else None,
    )
    strain = torch.zeros(len(g), 3, 3, device=g.device)
    strain = torch.where(refused[:, None, None], torch.nan, strain)
    return fitted, strain, score


def strain_from_reference(indexer, frames, reference, *, grid: int = 6, n_passes: int = 2):
    """The strain of each of *frames* ``(N, H, W)`` relative to one real
    *reference* frame ``(H, W)``, by cross-correlating regions of the patterns
    (:func:`~de_ebsd.index.crosscorr.hrebsd`): ``(strain, rotation_vector,
    residual)``, the strain ``(N, 3, 3)`` and lattice rotation ``(N, 3)`` in the
    sample frame and the fit's residual ``(N,)``.

    No master enters, so unlike :meth:`refine_deformation` nothing about how
    well a simulation matches the camera limits it, but it measures only
    differences from the reference, which should be strain-free, and only
    within its grain: the remap holds for lattice rotations of a few degrees.
    On a twin scan with a random 3e-3 strain at a 128 refine, the error relative
    to a reference in each grain was 1.7e-3, against 7.3e-4 absolute from
    :meth:`refine_deformation`; correlating regions against simulations
    instead was 6e-3.
    """
    from de_ebsd.index.crosscorr import hrebsd

    test = torch.cat(
        [
            indexer.background.remove(indexer.pooled(frames[lo : lo + BATCH]))
            for lo in range(0, len(frames), BATCH)
        ]
    )
    ref = indexer.background.remove(indexer.pooled(reference))
    detector = replace(indexer.detector, shape=indexer.refine_shape)
    result = hrebsd(
        test,
        ref,
        detector,
        roi_size=indexer.refine_shape[0] // 4,
        grid=grid,
        n_passes=n_passes,
        batch=BATCH,
    )
    _, strain, rotation = result.to_sample(detector)
    vector = torch.stack([rotation[:, 2, 1], rotation[:, 0, 2], rotation[:, 1, 0]], dim=-1)
    return strain, vector, result.residual
