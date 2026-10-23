"""High-precision HR refinement (:func:`de_ebsd.index.refine_hr`).

The sub-0.1-degree, edge-importance-sampled, tiled joint orientation + pattern-center
refiner. Where :func:`~de_ebsd.index.refine.orientation.refine_orientation` evaluates
its NCC over a back-projected direction set, ``refine_hr`` importance-samples a few
thousand band-edge pixels, spreads them across a tile grid, normalizes each tile
locally, and runs a batched Levenberg-Marquardt refinement of a 6-DOF
``theta = [w, dpc]``.

The money tests:

1. **6-DOF Jacobian FD-check** -- both the cross-product orientation block and the
   forward-mode (``jvp``) pattern-center block match central finite differences on the
   smooth band master to a tight tolerance (the per-tile standardization derivative
   included).
2. **Joint (R, pattern_center) recovery** -- project known ``(R, pattern_center)``, seed ~1 deg off with a
   perturbed pattern center, and recover disorientation ``< 0.1`` deg *and* pattern-center
   error ``< 0.2x`` the seed's, batched over several general orientations.
3. **Edge sampling picks edges** -- the selected samples have a higher mean ``|grad E|``
   than a uniform-random pick, and every tile is represented (the importance sampling
   works and is spatially spread).
4. **Tiled robustness** -- under a smooth intensity ramp, per-tile normalization recovers
   the orientation better than a single global tile.
5. **Speed knob monotonicity** -- more samples never hurt recovery, and the ``all_pixels``
   reference path agrees with a dense subsample.
6. **Plumbing** -- ``refine_pattern_center=False`` is orientation-only and returns no pattern center;
   numpy and torch, single and batched inputs; a CUDA-guarded path.

fp64 on CPU at small sizes throughout -- exact-ish, fast, small.
"""

import math

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation

from band_master import make_band_master
from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.rotation import quaternion_to_matrix, skew_matrix
from de_ebsd.index import refine_hr
from de_ebsd.index.refine.highres import (
    _ddc_dpc,
    _residual_jacobian,
    _sobel_gradient_magnitude,
    _standardize_tiled,
    direction_cosines_from_pc,
    edge_importance_samples,
)


@pytest.fixture(scope="module")
def master() -> MasterPattern:
    """A band master: 12 great-circle bands, distinctive with clear high-gradient edges."""
    return make_band_master()


@pytest.fixture(scope="module")
def detector() -> EBSDDetector:
    """A realistic tilted detector with a generous field of view."""
    return EBSDDetector(shape=(80, 80), pattern_center=(0.5, 0.5, 0.20), sample_tilt=70.0, tilt=5.0)


# general / asymmetric orientations (Bunge Euler, degrees)
_EULER_DEG = np.array(
    [
        [10.0, 20.0, 30.0],
        [45.0, 55.0, 15.0],
        [123.0, 47.0, 205.0],  # general / asymmetric
        [200.0, 100.0, 300.0],  # general / asymmetric
    ]
)


def _dis(r_a: Rotation, r_b: Rotation) -> float:
    """Disorientation (degrees) between two orix Rotations, no symmetry applied."""
    return float(np.asarray((r_a * (~r_b)).angle).ravel()[0]) * 180.0 / math.pi


def _seed(r_true: Rotation, deg: float, rng_seed: int) -> Rotation:
    """Rotate ``r_true`` by ``deg`` about a random axis per orientation (a realistic seed)."""
    rng = np.random.default_rng(rng_seed)
    axis = rng.normal(size=(r_true.size, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return Rotation.from_axes_angles(axis, np.deg2rad(deg)) * r_true


# --------------------------------------------------------------------------- #
# 0. the differentiable forward DC map reproduces the detector's own
# --------------------------------------------------------------------------- #
def test_direction_cosines_from_pc_matches_detector(detector):
    """:func:`direction_cosines_from_pc` reproduces ``EBSDDetector.direction_cosines``."""
    ref = detector.direction_cosines(dtype=torch.float64).reshape(
        detector.nrows, detector.ncols, 3
    )
    rr, cc = torch.meshgrid(
        torch.arange(detector.nrows, dtype=torch.float64),
        torch.arange(detector.ncols, dtype=torch.float64),
        indexing="ij",
    )
    pattern_center = torch.tensor(detector.pattern_center, dtype=torch.float64)
    mine = direction_cosines_from_pc(detector, pattern_center, rr, cc)
    assert torch.allclose(mine, ref, atol=1e-12)


# --------------------------------------------------------------------------- #
# 1. 6-DOF Jacobian FD-check (both w and dpc blocks)
# --------------------------------------------------------------------------- #
def test_6dof_jacobian_matches_finite_differences(master, detector):
    """FD-check the analytic residual Jacobian, orientation and pattern-center blocks.

    The orientation block is the cross-product form ``dc x (gM @ R)`` and the
    pattern-center block is the forward-mode (``jvp``) ``q . d dc/d pattern_center``, both pushed
    through the per-tile standardization derivative. On the smooth master the analytic
    Jacobian must match central finite differences of the tiled residual tightly.
    """
    dtype = torch.float64
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(detector, r_true, rescale=True)
    n_p = r_true.size

    rows, cols, flat, tile_ids, counts = edge_importance_samples(
        patterns, n_tiles=6, samples_per_tile=8
    )
    n_total = int(counts.shape[0])
    counts_ps = counts.index_select(0, tile_ids).to(dtype)
    e_raw = patterns.reshape(n_p, -1).gather(1, flat.long())
    _, _, e_hat = _standardize_tiled(e_raw, tile_ids, counts_ps, n_total)

    # evaluate at a seed ~2 deg off and pattern_center offset by +0.01 (a realistic coarse seed)
    q0 = quaternion_to_matrix(torch.as_tensor(np.asarray(r_true.data), dtype=dtype))
    rng = np.random.default_rng(3)
    axis = torch.as_tensor(rng.normal(size=(n_p, 3)), dtype=dtype)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    r_eval = q0 @ torch.matrix_exp(skew_matrix(math.radians(2.0) * axis))
    pc_eval = torch.tensor(detector.pattern_center, dtype=dtype)[None].expand(n_p, 3).contiguous() + 0.01

    def analytic(r_mat, pattern_center):
        dc = direction_cosines_from_pc(detector, pattern_center, rows, cols).detach()
        with torch.enable_grad():
            v = (dc @ r_mat.transpose(-1, -2)).detach().requires_grad_(True)
            m_raw = master.sample(v)
            (g_m,) = torch.autograd.grad(m_raw.sum(), v)
        m = m_raw.detach()
        q = torch.einsum("pka,pab->pkb", g_m, r_mat)
        dm_dw = torch.linalg.cross(dc, q, dim=-1)
        ddc = _ddc_dpc(detector, pattern_center, rows, cols)
        dm_dpc = torch.einsum("pkb,pkbc->pkc", q, ddc)
        g_raw = torch.cat([dm_dw, dm_dpc], dim=-1)
        return _residual_jacobian(m, e_hat, g_raw, tile_ids, counts_ps, n_total)

    def residual(r_mat, pattern_center):
        dc = direction_cosines_from_pc(detector, pattern_center, rows, cols)
        m = master.sample(dc @ r_mat.transpose(-1, -2))
        _, _, m_hat = _standardize_tiled(m, tile_ids, counts_ps, n_total)
        return e_hat - m_hat

    _, j_r = analytic(r_eval, pc_eval)  # (Np, K, 6)
    eps = 1e-6
    j_fd = torch.zeros_like(j_r)
    for k in range(3):  # orientation block via matrix_exp perturbation
        d = torch.zeros(n_p, 3, dtype=dtype)
        d[:, k] = eps
        rp = residual(r_eval @ torch.matrix_exp(skew_matrix(d)), pc_eval)
        rm = residual(r_eval @ torch.matrix_exp(skew_matrix(-d)), pc_eval)
        j_fd[..., k] = (rp - rm) / (2 * eps)
    for k in range(3):  # pattern-center block via additive perturbation
        dp = torch.zeros(n_p, 3, dtype=dtype)
        dp[:, k] = eps
        rp = residual(r_eval, pc_eval + dp)
        rm = residual(r_eval, pc_eval - dp)
        j_fd[..., 3 + k] = (rp - rm) / (2 * eps)

    rel = (j_r - j_fd).abs() / j_fd.abs().clamp_min(1e-9)
    w_rel_med = float(rel[..., :3].median())
    pc_rel_med = float(rel[..., 3:].median())
    # the median pins the math (essentially exact); a robust high percentile bounds the
    # bulk worst case. The extreme tail (~0.1% of points) sits at bilinear grid_sample
    # kinks where the central-FD *reference* itself is wrong (the analytic Jacobian is
    # fine there) -- the same near-kink artifact the refine.py LM FD-check tolerates, only
    # amplified here by the small per-tile groups -- so it is excluded from the bound.
    p99 = float(torch.quantile(rel.flatten(), 0.99))
    assert w_rel_med < 1e-4, f"orientation block median rel err {w_rel_med:.2e}"
    assert pc_rel_med < 1e-4, f"pattern-center block median rel err {pc_rel_med:.2e}"
    assert p99 < 1e-3, f"99th-percentile rel err {p99:.2e}"


# --------------------------------------------------------------------------- #
# 2. joint (R, pattern_center) recovery -- the payoff
# --------------------------------------------------------------------------- #
def test_joint_orientation_and_pc_recovery(master, detector):
    """Recover a ~1 deg orientation error and a perturbed pattern center jointly.

    Project at the true ``(R, pattern_center)``, seed each orientation ~1 deg off with a pattern
    center offset by +0.01 in every component, and assert the refined disorientation is
    ``< 0.1`` deg AND the refined pattern-center error is ``< 0.2x`` the seed's, for
    several general orientations refined in one batched call.
    """
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    pc_true = torch.tensor(detector.pattern_center, dtype=torch.float64)
    patterns = master.project(detector, r_true, rescale=True)

    seed = _seed(r_true, 1.0, rng_seed=1)
    pc_seed = pc_true + 0.01
    seed_pc_err = float((pc_seed - pc_true).norm())

    refined, score, refined_pattern_center = refine_hr(
        patterns, master, detector, seed, pattern_center=pc_seed, refine_pattern_center=True,
        n_tiles=8, samples_per_tile=8, n_steps=25,
    )
    refined_pc_err = float((refined_pattern_center - pc_true).norm())

    assert tuple(refined_pattern_center.shape) == (3,)
    assert torch.isfinite(score).all()
    for i in range(r_true.size):
        d_seed = _dis(r_true[i], seed[i])
        d_ref = _dis(r_true[i], refined[i])
        assert d_seed == pytest.approx(1.0, abs=1e-6)  # the seed really is 1 deg off
        assert d_ref < 0.1, f"orientation {i}: refined {d_ref:.4f} deg not < 0.1"
    assert refined_pc_err < 0.2 * seed_pc_err, (
        f"pattern center: seed err {seed_pc_err:.5f} -> refined {refined_pc_err:.5f}"
    )
    assert float(score.min()) > 0.99  # near-perfect tiled NCC


def test_joint_recovery_per_pattern_pc(master, detector):
    """A per-pattern ``(Np, 3)`` pattern center is accepted and refined per pattern."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:3]))
    pc_true = torch.tensor(detector.pattern_center, dtype=torch.float64)
    patterns = master.project(detector, r_true, rescale=True)

    seed = _seed(r_true, 1.0, rng_seed=5)
    pc_seed = pc_true[None].repeat(3, 1) + 0.01  # (3, 3) per-pattern seed

    refined, score, refined_pattern_center = refine_hr(
        patterns, master, detector, seed, pattern_center=pc_seed, refine_pattern_center=True,
        n_tiles=8, samples_per_tile=8, n_steps=25,
    )
    assert tuple(refined_pattern_center.shape) == (3, 3)
    for i in range(r_true.size):
        seed_err = float((pc_seed[i] - pc_true).norm())
        refined_err = float((refined_pattern_center[i] - pc_true).norm())
        assert refined_err < 0.2 * seed_err
        assert _dis(r_true[i], refined[i]) < 0.1


# --------------------------------------------------------------------------- #
# 3. edge sampling picks band edges and spreads across tiles
# --------------------------------------------------------------------------- #
def test_edge_sampling_prefers_edges_and_spreads(master, detector):
    """Selected samples have higher mean ``|grad E|`` than random, and cover every tile."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)
    n_tiles, spt = 8, 8

    rows, cols, flat, tile_ids, counts = edge_importance_samples(
        patterns, n_tiles=n_tiles, samples_per_tile=spt
    )
    grad = _sobel_gradient_magnitude(patterns).reshape(patterns.shape[0], -1)

    # importance sampling: the selected pixels are higher-gradient than a random pick
    sel_grad = grad.gather(1, flat.long()).mean()
    rng = np.random.default_rng(0)
    rand_idx = torch.as_tensor(
        rng.integers(0, grad.shape[1], size=flat.shape), dtype=torch.long
    )
    rand_grad = grad.gather(1, rand_idx).mean()
    assert float(sel_grad) > 1.5 * float(rand_grad), (
        f"selected mean |grad| {float(sel_grad):.4f} not >> random {float(rand_grad):.4f}"
    )

    # spatial spread: every tile is represented, each by exactly samples_per_tile slots
    assert bool((counts > 0).all())
    assert int(counts.shape[0]) == n_tiles * n_tiles
    assert torch.equal(counts.unique(), torch.tensor([spt]))
    assert set(tile_ids.tolist()) == set(range(n_tiles * n_tiles))


# --------------------------------------------------------------------------- #
# 4. tiled robustness: per-tile normalization beats a single global tile under a ramp
# --------------------------------------------------------------------------- #
def test_tiled_beats_global_under_intensity_ramp(master, detector):
    """A smooth intensity ramp corrupts a global-normalized fit; per-tile survives it.

    Per-tile local normalization removes a spatially-varying ramp (its local mean/scale
    per tile), which one global standardization cannot -- so the tiled refine recovers the
    orientation better than a single-tile (global) refine on ramped experimental patterns.
    """
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)

    # a smooth linear intensity ramp across the detector, added to the experimental signal
    h, w = detector.nrows, detector.ncols
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, dtype=torch.float64),
        torch.linspace(-1.0, 1.0, w, dtype=torch.float64),
        indexing="ij",
    )
    ramp = 0.4 * (1.5 * xx - 1.0 * yy)  # a tilt comparable to the pattern's own contrast
    ramped = patterns + ramp[None]

    seed = _seed(r_true, 1.0, rng_seed=2)
    # tiled: many small tiles, each locally normalized (the ramp is ~flat within a tile)
    tiled, _ = refine_hr(
        ramped, master, detector, seed, refine_pattern_center=False,
        n_tiles=16, samples_per_tile=8, n_steps=30,
    )
    # global: a single tile -> one mean/norm over the whole detector, which cannot remove
    # the spatially-varying ramp
    glob, _ = refine_hr(
        ramped, master, detector, seed, refine_pattern_center=False,
        n_tiles=1, samples_per_tile=16 * 16 * 8, n_steps=30,
    )
    tiled_dis = np.array([_dis(r_true[i], tiled[i]) for i in range(r_true.size)])
    glob_dis = np.array([_dis(r_true[i], glob[i]) for i in range(r_true.size)])
    # per-tile normalization is markedly more robust to the ramp (measured ~3x closer),
    # and recovers the orientation while the global fit is dragged well off by the tilt
    assert np.median(tiled_dis) < 0.6 * np.median(glob_dis), (
        f"tiled median {np.median(tiled_dis):.3f} not << global {np.median(glob_dis):.3f}"
    )
    assert np.median(tiled_dis) < 0.6  # tiled genuinely recovers under the ramp


# --------------------------------------------------------------------------- #
# 5. speed knob monotonicity and the all_pixels reference path
# --------------------------------------------------------------------------- #
def test_more_samples_do_not_hurt(master, detector):
    """More samples per tile give equal-or-better recovery (the accuracy/speed knob)."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)
    seed = _seed(r_true, 1.5, rng_seed=7)

    few, _ = refine_hr(
        patterns, master, detector, seed, refine_pattern_center=False,
        n_tiles=6, samples_per_tile=2, n_steps=8,
    )
    many, _ = refine_hr(
        patterns, master, detector, seed, refine_pattern_center=False,
        n_tiles=6, samples_per_tile=16, n_steps=8,
    )
    few_dis = np.array([_dis(r_true[i], few[i]) for i in range(r_true.size)])
    many_dis = np.array([_dis(r_true[i], many[i]) for i in range(r_true.size)])
    # more samples are at least as accurate (small slack for the shared clean-master floor)
    assert np.median(many_dis) <= np.median(few_dis) + 1e-3


def test_all_pixels_agrees_with_dense_sample(master, detector):
    """``all_pixels=True`` runs and lands at the same orientation as a dense subsample."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(detector, r_true, rescale=True)
    seed = _seed(r_true, 1.0, rng_seed=9)

    dense, _ = refine_hr(
        patterns, master, detector, seed, refine_pattern_center=False,
        n_tiles=8, samples_per_tile=32, n_steps=20,
    )
    allpix, score = refine_hr(
        patterns, master, detector, seed, refine_pattern_center=False,
        n_tiles=8, all_pixels=True, n_steps=20,
    )
    for i in range(r_true.size):
        assert _dis(r_true[i], allpix[i]) < 0.1  # the reference path recovers truth
        assert _dis(dense[i], allpix[i]) < 0.1  # and agrees with the dense subsample
    assert float(score.min()) > 0.99


# --------------------------------------------------------------------------- #
# 6. plumbing: refine_pattern_center, numpy/torch, single/batched, device
# --------------------------------------------------------------------------- #
def test_refine_pc_false_is_orientation_only(master, detector):
    """``refine_pattern_center=False`` returns just ``(rotation, score)`` and leaves ``pattern_center`` untouched."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(detector, r_true, rescale=True)
    seed = _seed(r_true, 1.0, rng_seed=3)

    out = refine_hr(patterns, master, detector, seed, refine_pattern_center=False, n_steps=15)
    assert len(out) == 2  # no pattern center in the result
    rotation, score = out
    assert isinstance(rotation, Rotation)
    assert tuple(score.shape) == (2,)
    for i in range(r_true.size):
        assert _dis(r_true[i], rotation[i]) < 0.1


def test_single_and_numpy_input(master, detector):
    """A bare ``(H, W)`` numpy pattern refines to length-1 results."""
    r_true = Rotation.from_euler(np.deg2rad([[30.0, 40.0, 50.0]]))
    pattern = master.project(detector, r_true, rescale=True)[0].cpu().numpy()  # (H, W)
    assert pattern.ndim == 2
    seed = _seed(r_true, 1.0, rng_seed=4)

    refined, score = refine_hr(pattern, master, detector, seed, refine_pattern_center=False, n_steps=15)
    assert refined.size == 1
    assert tuple(score.shape) == (1,)
    assert _dis(r_true[0], refined[0]) < 0.1


def test_batched_matches_single(master, detector):
    """A stack refines in one shot; each pattern matches its own single refinement."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)
    seed = _seed(r_true, 1.0, rng_seed=6)

    batched, score_b = refine_hr(patterns, master, detector, seed, refine_pattern_center=False, n_steps=15)
    for i in range(r_true.size):
        single, score_s = refine_hr(
            patterns[i], master, detector, seed[i], refine_pattern_center=False, n_steps=15
        )
        assert _dis(batched[i], single[0]) < 1e-4  # independent per-pattern refinement
        assert float(score_s[0]) == pytest.approx(float(score_b[i]), abs=1e-6)


def test_shape_validation(master, detector):
    """A rotations / patterns length mismatch raises."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(detector, r_true, rescale=True)
    wrong = Rotation.from_euler(np.deg2rad(_EULER_DEG[:1]))
    with pytest.raises(ValueError, match="rotations length"):
        refine_hr(patterns, master, detector, wrong, refine_pattern_center=False, n_steps=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_path(master, detector):
    """The refine runs on CUDA and recovers the orientation and pattern center there."""
    master_cuda = master.to("cuda")
    r_true = Rotation.from_euler(np.deg2rad([[123.0, 47.0, 205.0]]))
    pattern = master_cuda.project(detector, r_true, rescale=True)[0]
    pc_true = torch.tensor(detector.pattern_center, dtype=torch.float64)
    seed = _seed(r_true, 1.0, rng_seed=8)

    refined, score, refined_pattern_center = refine_hr(
        pattern, master_cuda, detector, seed, pattern_center=(pc_true + 0.01), refine_pattern_center=True,
        n_steps=25, device="cuda",
    )
    assert score.device.type == "cuda"
    assert refined_pattern_center.device.type == "cuda"
    assert _dis(r_true[0], refined[0]) < 0.1
    assert float((refined_pattern_center.cpu() - pc_true).norm()) < 0.2 * float(0.01 * math.sqrt(3))
