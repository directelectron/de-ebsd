"""Gradient-based refinement (I6): the differentiable payoff, end to end.

Validates :func:`de_ebsd.index.refine.orientation.refine_orientation` and
:meth:`de_ebsd.index.spherical.indexer.SphericalIndexer.refine`. The coarse spherical index is
only accurate to its SOFT grid cell (``360 / (2L + 2)`` degrees); refinement sharpens a
seed to sub-grid-cell accuracy by gradient ascent on a spherical masked NCC through the
differentiable forward model, and recovers a perturbed pattern center.

The money tests:

1. **Orientation sharpening** -- project known orientations, index to the coarse seed,
   refine, and assert the refined disorientation is a large factor smaller than the
   coarse one (and well under half a degree) for several general/asymmetric
   orientations, batched.
2. **Sub-grid-cell from a deliberately-off seed** -- seed ~1 grid cell off truth and
   show it converges back to ``< 0.5`` deg (independent of the coarse index).
3. **Pattern-center recovery** -- render at the true pattern center, seed a *perturbed*
   one, ``refine_pattern_center=True``, and assert the refined center is much closer to truth than
   the seed was (the I3 pattern-center gradient path, end to end).
4. **Objective sanity** -- the NCC rises over iterations and the refined score is ``>=``
   the coarse seed's.
5. **Plumbing** -- single ``(H, W)`` and batched inputs, dtype/device, and that
   ``refine_pattern_center=False`` leaves the pattern center untouched.

The synthetic band master + tilted detector are the I4 indexer fixtures (band fields
have near-uniform local norm, so the coarse index is reliable and the NCC basin is
smooth). fp64 on CPU at a small bandwidth throughout -- exact-ish, fast, small.
"""

import math

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation

from band_master import make_band_master
from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.rotation import quaternion_to_matrix, rotation_vector_to_matrix
from de_ebsd.index import refine_orientation
from de_ebsd.index.backproject import pattern_to_sphere
from de_ebsd.index.refine.orientation import (
    _fov_directions,
    _lm_residual_jacobian,
    _standardize,
)
from de_ebsd.index.spherical.indexer import SphericalIndexer

BANDWIDTH = 20
GRID_STEP_DEG = 360.0 / (2 * BANDWIDTH + 2)  # ~8.57 deg SOFT cell at L = 20


@pytest.fixture(scope="module")
def master() -> MasterPattern:
    """The I4 band master: 12 great-circle bands, distinctive yet near-uniform norm."""
    return make_band_master()


@pytest.fixture(scope="module")
def detector() -> EBSDDetector:
    """A realistic tilted detector with a generous field of view."""
    return EBSDDetector(shape=(90, 90), pattern_center=(0.5, 0.5, 0.20), sample_tilt=70.0, tilt=5.0)


@pytest.fixture(scope="module")
def indexer(master, detector) -> SphericalIndexer:
    return SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")


def _dis(r_a: Rotation, r_b: Rotation) -> float:
    """Disorientation (degrees) between two orix Rotations, no symmetry applied."""
    return float(np.asarray((r_a * (~r_b)).angle).ravel()[0]) * 180.0 / math.pi


# general / asymmetric orientations (Bunge Euler, degrees)
_EULER_DEG = np.array(
    [
        [10.0, 20.0, 30.0],
        [45.0, 55.0, 15.0],
        [123.0, 47.0, 205.0],  # general / asymmetric
        [200.0, 100.0, 300.0],  # general / asymmetric
    ]
)


# --------------------------------------------------------------------------- #
# 1. orientation sharpening -- THE payoff
# --------------------------------------------------------------------------- #
def test_refine_sharpens_coarse_orientation(master, detector, indexer):
    """Index to the coarse cell, refine, and land a large factor closer to truth.

    For each of several general orientations the refined disorientation must be ``<
    0.3x`` the coarse one AND ``< 0.5`` deg -- the sub-grid-cell sharpening that
    motivates the differentiable model. Batched: the whole stack refines in one call.
    """
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)  # (4, H, W)

    coarse, _ = indexer.index(patterns)
    refined, score = indexer.refine(
        patterns, rotations=coarse, n_iterations=100, learning_rate=0.05
    )

    assert refined.size == r_true.size
    assert tuple(score.shape) == (r_true.size,)
    for i in range(r_true.size):
        dis_coarse = _dis(r_true[i], coarse[i])
        dis_refined = _dis(r_true[i], refined[i])
        assert dis_coarse < 1.5 * GRID_STEP_DEG  # coarse really is only cell-accurate
        assert dis_refined < 0.3 * dis_coarse, (
            f"orientation {i}: refined {dis_refined:.3f} not < 0.3x coarse {dis_coarse:.3f} deg"
        )
        assert dis_refined < 0.5, f"orientation {i}: refined {dis_refined:.3f} deg > 0.5"
    assert torch.isfinite(score).all()
    assert float(score.min()) > 0.9  # a genuine, near-perfect match


def test_refine_index_then_refine_from_raw_patterns(master, detector, indexer):
    """``refine`` with ``rotations=None`` indexes first, then sharpens -- one call."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(detector, r_true, rescale=True)

    refined, score = indexer.refine(patterns, n_iterations=100, learning_rate=0.05)
    for i in range(r_true.size):
        assert _dis(r_true[i], refined[i]) < 0.5
    assert torch.isfinite(score).all()


# --------------------------------------------------------------------------- #
# 2. sub-grid-cell recovery from a deliberately-off seed
# --------------------------------------------------------------------------- #
def test_refine_converges_from_off_seed(master, detector, indexer):
    """A seed a full grid cell off truth converges back to ``< 0.5`` deg.

    Independent of the coarse index: the seed is built by rotating truth by exactly one
    SOFT cell about a random axis, so this pins the refiner's basin of attraction.
    """
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[2:]))
    patterns = master.project(detector, r_true, rescale=True)

    rng = np.random.default_rng(1)
    axis = rng.normal(size=(r_true.size, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    offset = Rotation.from_axes_angles(axis, np.deg2rad(GRID_STEP_DEG))
    seed = offset * r_true
    for i in range(r_true.size):
        assert _dis(r_true[i], seed[i]) == pytest.approx(GRID_STEP_DEG, abs=1e-6)

    refined, _ = indexer.refine(patterns, rotations=seed, n_iterations=120, learning_rate=0.05)
    for i in range(r_true.size):
        assert _dis(r_true[i], refined[i]) < 0.5


def test_refine_correct_seed_does_not_drift(master, detector, indexer):
    """Refining a *perfect* seed stays put -- the compose order matches ``project``.

    At ``w = 0`` a correct seed already sits at the NCC maximum, so it must not wander
    (a wrong compose order / inverse convention would send it ~90-180 deg away).
    """
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)

    refined, score = indexer.refine(
        patterns, rotations=r_true, n_iterations=100, learning_rate=0.05
    )
    for i in range(r_true.size):
        assert _dis(r_true[i], refined[i]) < 0.3  # interpolation-limited, tiny
    assert float(score.min()) > 0.99


# --------------------------------------------------------------------------- #
# 3. pattern-center recovery -- the I3 gradient path end to end
# --------------------------------------------------------------------------- #
def test_refine_recovers_perturbed_pattern_center(master):
    """Seed a perturbed pattern center; ``refine_pattern_center`` pulls it back toward truth.

    Render at the true center, index/refine from a center offset by +0.02 in each of
    ``pcx, pcy, pcz``, and assert the refined center is far closer to truth than the
    seed (proving the pattern-center gradient path), with the orientation still recovered.
    """
    pc_true = (0.5, 0.5, 0.20)
    det_true = EBSDDetector(shape=(90, 90), pattern_center=pc_true, sample_tilt=70.0, tilt=5.0)
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(det_true, r_true, rescale=True)

    pc_true_t = torch.tensor(pc_true, dtype=torch.float64)
    pc_seed = pc_true_t + 0.02
    seed_err = float((pc_seed - pc_true_t).norm())

    # seed the orientation by indexing with the (wrong) perturbed geometry
    det_seed = EBSDDetector(
        shape=(90, 90), pattern_center=tuple(pc_seed.tolist()), sample_tilt=70.0, tilt=5.0
    )
    idx_seed = SphericalIndexer(master, det_seed, bandwidth=BANDWIDTH, device="cpu")
    coarse, _ = idx_seed.index(patterns)

    refined, score, reference_pattern_center = idx_seed.refine(
        patterns,
        rotations=coarse,
        pattern_center=pc_seed,
        refine_pattern_center=True,
        n_iterations=200,
        learning_rate=0.05,
    )
    refined_err = float((reference_pattern_center - pc_true_t).norm())

    assert tuple(reference_pattern_center.shape) == (3,)
    assert refined_err < 0.25 * seed_err, (
        f"pattern center not recovered: seed err {seed_err:.4f} -> refined {refined_err:.4f}"
    )
    for i in range(r_true.size):
        assert _dis(r_true[i], refined[i]) < 0.5
    assert torch.isfinite(score).all()


def test_refine_pc_false_leaves_pc_untouched(master, detector, indexer):
    """``refine_pattern_center=False`` returns just ``(rotation, score)`` -- no pattern center."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:1]))
    patterns = master.project(detector, r_true, rescale=True)
    out = indexer.refine(patterns, refine_pattern_center=False, n_iterations=50)
    assert len(out) == 2  # rotation, score -- pattern_center not part of the result
    rotation, score = out
    assert isinstance(rotation, Rotation)
    assert tuple(score.shape) == (1,)


# --------------------------------------------------------------------------- #
# 4. objective sanity: the NCC rises and the refined score beats the seed's
# --------------------------------------------------------------------------- #
def test_ncc_increases_over_iterations(master, detector, indexer):
    """The NCC objective rises over iterations; the refined score ``>=`` the seed's.

    Not necessarily monotonic for Adam, but final > initial, and the final returned
    score is at least the coarse seed's NCC (refinement never makes the match worse).
    """
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)
    coarse, _ = indexer.index(patterns)

    refined, score, history = indexer.refine(
        patterns, rotations=coarse, n_iterations=100, learning_rate=0.05, return_history=True
    )
    assert tuple(history.shape) == (100, r_true.size)
    # history[0] is the coarse seed's NCC (w = 0); the objective rises overall
    assert float(history[-1].mean()) > float(history[0].mean())
    for i in range(r_true.size):
        assert float(score[i]) >= float(history[0, i]) - 1e-6  # refined >= coarse NCC
        assert float(score[i]) > 0.99  # near-perfect after refinement


# --------------------------------------------------------------------------- #
# 5. plumbing: shapes, dtype/device, single vs batched, default directions
# --------------------------------------------------------------------------- #
def test_single_pattern_input(master, detector, indexer):
    """A bare ``(H, W)`` pattern refines to a length-1 Rotation + length-1 score."""
    r_true = Rotation.from_euler(np.deg2rad([[30.0, 40.0, 50.0]]))
    pattern = master.project(detector, r_true, rescale=True)[0]  # (H, W)
    assert pattern.ndim == 2
    coarse, _ = indexer.index(pattern)

    refined, score = indexer.refine(pattern, rotations=coarse, n_iterations=100)
    assert refined.size == 1
    assert tuple(score.shape) == (1,)
    assert score.dtype == torch.float64
    assert score.device == torch.device("cpu")
    assert _dis(r_true[0], refined[0]) < 0.5


def test_functional_core_with_default_directions(master, detector):
    """``refine_orientation`` works standalone with its default Fibonacci directions."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(detector, r_true, rescale=True)
    idx = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    coarse, _ = idx.index(patterns)

    refined, score = refine_orientation(patterns, master, detector, coarse, n_iterations=100)
    assert refined.size == 2
    for i in range(2):
        assert _dis(r_true[i], refined[i]) < 0.5


def test_batched_matches_single(master, detector, indexer):
    """A stack refines in one shot; each pattern matches its own single refinement."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)
    coarse, _ = indexer.index(patterns)

    refined_b, score_b = indexer.refine(
        patterns, rotations=coarse, n_iterations=100, learning_rate=0.05
    )
    for i in range(r_true.size):
        refined_s, score_s = indexer.refine(
            patterns[i], rotations=coarse[i], n_iterations=100, learning_rate=0.05
        )
        # independent per-pattern gradients: batched == single, to tight tolerance
        assert _dis(refined_b[i], refined_s[0]) < 1e-4
        assert float(score_s[0]) == pytest.approx(float(score_b[i]), abs=1e-6)


def test_per_pattern_pattern_center(master):
    """A per-pattern ``(Np, 3)`` pattern center is accepted and refined per pattern."""
    pc_true = (0.5, 0.5, 0.20)
    det_true = EBSDDetector(shape=(90, 90), pattern_center=pc_true, sample_tilt=70.0, tilt=5.0)
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(det_true, r_true, rescale=True)

    idx = SphericalIndexer(master, det_true, bandwidth=BANDWIDTH, device="cpu")
    coarse, _ = idx.index(patterns)
    pc_true_t = torch.tensor(pc_true, dtype=torch.float64)
    pc_seed = pc_true_t[None].repeat(2, 1) + 0.015  # (2, 3) per-pattern seed

    refined, score, reference_pattern_center = idx.refine(
        patterns,
        rotations=coarse,
        pattern_center=pc_seed,
        refine_pattern_center=True,
        n_iterations=200,
        learning_rate=0.05,
    )
    assert tuple(reference_pattern_center.shape) == (2, 3)
    for i in range(2):
        seed_err = float((pc_seed[i] - pc_true_t).norm())
        refined_err = float((reference_pattern_center[i] - pc_true_t).norm())
        assert refined_err < seed_err
        assert _dis(r_true[i], refined[i]) < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_path_matches_cpu(master, detector):
    """When CUDA is present, refinement runs on it and recovers the orientation."""
    master_cuda = master.to("cuda")
    indexer = SphericalIndexer(master_cuda, detector, bandwidth=BANDWIDTH, device="cuda")
    r_true = Rotation.from_euler(np.deg2rad([[123.0, 47.0, 205.0]]))
    pattern = master_cuda.project(detector, r_true, rescale=True)[0]
    coarse, _ = indexer.index(pattern)
    refined, score = indexer.refine(pattern, rotations=coarse, n_iterations=100)
    assert score.device.type == "cuda"
    assert _dis(r_true[0], refined[0]) < 0.5


# --------------------------------------------------------------------------- #
# A4: refine_orientation honors an explicit device
# --------------------------------------------------------------------------- #
def test_refine_honors_device_cpu(master, detector, indexer):
    """``refine_orientation(device='cpu')`` runs on CPU and returns a CPU score."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:1]))
    patterns = master.project(detector, r_true, rescale=True)
    coarse, _ = indexer.index(patterns)

    refined, score = refine_orientation(
        patterns, master, detector, coarse, n_iterations=100, device="cpu"
    )
    assert score.device == torch.device("cpu")
    assert _dis(r_true[0], refined[0]) < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_refine_device_arg_is_honored_cuda(master, detector):
    """A CPU master refined with ``device='cuda'`` runs on CUDA (the score lands there).

    The master, patterns and seeds are moved to the requested device -- the arg wins
    over ``master.data.device`` (which is CPU here).
    """
    assert master.data.device.type == "cpu"
    idx = SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")
    r_true = Rotation.from_euler(np.deg2rad([[123.0, 47.0, 205.0]]))
    patterns = master.project(detector, r_true, rescale=True)
    coarse, _ = idx.index(patterns)

    refined, score = refine_orientation(
        patterns, master, detector, coarse, n_iterations=100, device="cuda"
    )
    assert score.device.type == "cuda"
    assert _dis(r_true[0], refined[0]) < 0.5


# --------------------------------------------------------------------------- #
# 6. stability -- the real-data (kikuchipy Ni) lesson
# --------------------------------------------------------------------------- #
def test_default_refine_lr_is_conservative():
    """The default rotation ``learning_rate`` stays small enough for real, shallow-basin data.

    The divergence a too-large ``learning_rate`` causes shows up only on *real* low-resolution
    patterns: the synthetic band master's NCC basin is too deep to reproduce it (even
    ``learning_rate = 0.05`` converges on synthetic data, noise included), so this guards the value
    directly. A regression to the old ``learning_rate = 0.05`` -- which walked the seed off by ~18
    deg on real kikuchipy Ni patterns while ``learning_rate <= 0.02`` refined cleanly -- fails here.
    """
    import inspect

    for fn in (refine_orientation, SphericalIndexer.refine):
        default_lr = inspect.signature(fn).parameters["learning_rate"].default
        assert default_lr <= 0.02, (
            f"{fn.__qualname__} default learning_rate {default_lr} is too aggressive for real data"
        )


def test_refine_stable_under_moderate_noise(master, detector, indexer):
    """With the default ``learning_rate``, refinement converges on a noise-perturbed pattern.

    A behavioral check that additive noise does not make refinement diverge: the refined
    orientation lands near the truth and moves the coarse seed only a little (a small,
    controlled step, not a walk-off).
    """
    rng = np.random.default_rng(0)
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[2:4]))  # two asymmetric
    clean = master.project(detector, r_true, rescale=True)  # (2, H, W) in [-1, 1]
    noise = torch.as_tensor(rng.normal(0.0, 0.25, size=tuple(clean.shape)), dtype=clean.dtype)
    patterns = clean + noise
    coarse = indexer.index(patterns)[0]

    refined, score = indexer.refine(
        patterns, rotations=coarse, n_iterations=100
    )  # default learning_rate
    assert torch.isfinite(score).all()
    for i in range(r_true.size):
        assert _dis(r_true[i], refined[i]) < GRID_STEP_DEG  # converges toward truth
        assert _dis(coarse[i], refined[i]) < GRID_STEP_DEG  # small, controlled step


def test_keep_best_never_degrades_below_seed(master, detector, indexer):
    """``keep_best`` guarantees refinement is monotone: it never returns a worse fit.

    Forced with an absurd ``learning_rate`` that makes the optimizer overshoot and *degrade* the
    NCC, ``keep_best=True`` reverts each such pattern to its coarse seed (so the score
    never drops below the seed's), while ``keep_best=False`` lets the degradation
    through. This is the guard that removes the stray refinement outliers seen on real
    data.
    """
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)
    coarse = indexer.index(patterns)[0]
    _, seed_score = indexer.refine(patterns, rotations=coarse, n_iterations=0)  # seed NCC

    gated, s_gated = indexer.refine(
        patterns, rotations=coarse, n_iterations=30, learning_rate=2.0, keep_best=True
    )
    ungated, s_ungated = indexer.refine(
        patterns, rotations=coarse, n_iterations=30, learning_rate=2.0, keep_best=False
    )
    # gated is never worse than the seed; the absurd learning_rate degrades the ungated result
    assert torch.all(s_gated >= seed_score - 1e-6)
    assert torch.any(s_ungated < seed_score - 1e-3)
    # where the ungated run degraded, the gated run kept the seed orientation exactly
    for i in range(r_true.size):
        if float(s_ungated[i]) < float(seed_score[i]) - 1e-3:
            assert _dis(coarse[i], gated[i]) < 1e-4


# --------------------------------------------------------------------------- #
# 7. second-order (Levenberg-Marquardt / Gauss-Newton) refinement, method="lm"
# --------------------------------------------------------------------------- #
def test_lm_analytic_jacobian_matches_finite_differences(master, detector):
    """FD-check the analytic residual Jacobian on the smooth, band-limited master.

    On a smooth master the bilinear ``grid_sample`` has no kinks, so the analytic
    Jacobian (the crux of the LM path -- assembled from one reverse-mode ``grid_sample``
    gradient) must match central finite differences tightly. The check is done in
    float64 at a small deterministic offset off truth (a realistic seed). The median
    elementwise relative error pins the math; a global scale-free error (max abs error
    over the peak Jacobian magnitude, the prototype's own metric) bounds the worst case.
    """
    device, dtype = torch.device("cpu"), torch.float64
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)
    n_p = r_true.size

    dirs = _fov_directions(detector, device, dtype)
    e, mask = pattern_to_sphere(patterns.to(dtype), detector, dirs)
    assert bool((mask == 1).all())  # the LM all-in-FOV (all-ones-mask) assumption
    e_c, e_s = _standardize(e)
    e_hat = e_c / e_s

    # evaluate the Jacobian at a seed ~2 deg off truth (a realistic coarse cell)
    q0 = quaternion_to_matrix(torch.as_tensor(np.asarray(r_true.data), dtype=dtype))
    rng = np.random.default_rng(3)
    axis = torch.as_tensor(rng.normal(size=(n_p, 3)), dtype=dtype)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    r_eval = q0 @ rotation_vector_to_matrix(math.radians(2.0) * axis)

    _, j_r, _ = _lm_residual_jacobian(master, dirs, r_eval, e_hat, device, dtype)

    def residual(r_mat):
        m = master.sample(dirs @ r_mat.transpose(-1, -2))
        c, s = _standardize(m)
        return e_hat - c / s

    eps = 1e-6
    j_fd = torch.zeros_like(j_r)
    for k in range(3):
        d = torch.zeros(n_p, 3, dtype=dtype)
        d[:, k] = eps
        r_plus = residual(r_eval @ rotation_vector_to_matrix(d))
        r_minus = residual(r_eval @ rotation_vector_to_matrix(-d))
        j_fd[..., k] = (r_plus - r_minus) / (2 * eps)

    rel = (j_r - j_fd).abs() / j_fd.abs().clamp_min(1e-9)
    median_rel = float(rel.median())
    max_rel = float(rel.max())
    global_rel = float((j_r - j_fd).abs().max() / j_fd.abs().max().clamp_min(1e-9))
    # median pins the math (essentially exact); the loose max/global bounds the worst
    # case (a few ill-conditioned near-kink points, tiny in absolute terms).
    assert median_rel < 1e-2, f"median rel err {median_rel:.2e}"
    assert global_rel < 1e-2, f"global rel err {global_rel:.2e}"
    assert max_rel < 0.1, f"max rel err {max_rel:.2e}"


def test_lm_matches_adam_accuracy_in_far_fewer_steps(master, detector, indexer):
    """LM (12 steps) reaches Adam's sub-degree accuracy in a fraction of the iterations.

    On the synthetic round trip (project known orientations, coarse-seed, refine), the
    second-order LM path in 12 steps lands at least as close to truth as Adam at the same
    step count -- and as close as Adam's full 100-iteration budget -- for several
    general/asymmetric orientations, batched.
    """
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)
    coarse, _ = indexer.index(patterns)

    lm12, lm_score = refine_orientation(
        patterns, master, detector, coarse, method="lm", n_iterations=12
    )
    adam12, _ = refine_orientation(
        patterns, master, detector, coarse, method="adam", n_iterations=12, learning_rate=0.05
    )
    adam100, _ = refine_orientation(
        patterns, master, detector, coarse, method="adam", n_iterations=100, learning_rate=0.05
    )

    lm_dis, adam100_dis = [], []
    for i in range(r_true.size):
        d_lm = _dis(r_true[i], lm12[i])
        lm_dis.append(d_lm)
        adam100_dis.append(_dis(r_true[i], adam100[i]))
        assert d_lm < 0.5, f"orientation {i}: LM {d_lm:.3f} deg > 0.5"
        # at an equal 12-step budget LM is at least as accurate as Adam (far fewer steps)
        assert d_lm <= _dis(r_true[i], adam12[i]) + 1e-6
    # LM in 12 steps is as accurate as Adam in 100 (same objective, second order)
    assert np.median(lm_dis) <= np.median(adam100_dis) + 0.05
    assert float(lm_score.min()) > 0.99  # a genuine, near-perfect match


def test_lm_is_monotone_and_keep_best(master, detector, indexer):
    """LM never returns an orientation worse than the seed (monotone by construction).

    Its per-step accept/reject can only raise the NCC, so the refined score is ``>=`` the
    seed's whether ``keep_best`` is on or off; and refining an already-correct seed does
    not drift it (the compose order matches ``project``).
    """
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG))
    patterns = master.project(detector, r_true, rescale=True)
    coarse, _ = indexer.index(patterns)

    # seed NCC (an LM run of zero steps returns the seed untouched)
    _, seed_score = refine_orientation(
        patterns, master, detector, coarse, method="lm", n_iterations=0
    )
    gated, s_gated = refine_orientation(
        patterns, master, detector, coarse, method="lm", n_iterations=12, keep_best=True
    )
    ungated, s_ungated = refine_orientation(
        patterns, master, detector, coarse, method="lm", n_iterations=12, keep_best=False
    )
    assert torch.all(s_gated >= seed_score - 1e-6)
    assert torch.all(s_ungated >= seed_score - 1e-6)  # LM is monotone even ungated

    # a perfect seed stays put
    refined, score = refine_orientation(
        patterns, master, detector, r_true, method="lm", n_iterations=12
    )
    for i in range(r_true.size):
        assert _dis(r_true[i], refined[i]) < 0.05
    assert float(score.min()) > 0.99


def test_lm_and_adam_both_run_and_history(master, detector, indexer):
    """Both methods run through ``refine_orientation`` and honor ``return_history``."""
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(detector, r_true, rescale=True)
    coarse, _ = indexer.index(patterns)

    for method, n_iterations in (("adam", 20), ("lm", 8)):
        out = refine_orientation(
            patterns,
            master,
            detector,
            coarse,
            method=method,
            n_iterations=n_iterations,
            return_history=True,
        )
        assert len(out) == 3
        rotation, score, history = out
        assert isinstance(rotation, Rotation)
        assert tuple(history.shape) == (n_iterations, r_true.size)
        # history[0] is the seed NCC; the objective rises to the final score
        assert float(score.mean()) >= float(history[0].mean()) - 1e-6

    with pytest.raises(ValueError, match="method must be"):
        refine_orientation(patterns, master, detector, coarse, method="newton")


def test_lm_refine_pc_falls_back_to_adam(master):
    """``method="lm"`` with ``refine_pattern_center=True`` falls back to Adam and still returns
    pattern_center."""
    pc_true = (0.5, 0.5, 0.20)
    det_true = EBSDDetector(shape=(90, 90), pattern_center=pc_true, sample_tilt=70.0, tilt=5.0)
    r_true = Rotation.from_euler(np.deg2rad(_EULER_DEG[:2]))
    patterns = master.project(det_true, r_true, rescale=True)

    pc_true_t = torch.tensor(pc_true, dtype=torch.float64)
    pc_seed = pc_true_t + 0.02
    seed_err = float((pc_seed - pc_true_t).norm())

    det_seed = EBSDDetector(
        shape=(90, 90), pattern_center=tuple(pc_seed.tolist()), sample_tilt=70.0, tilt=5.0
    )
    idx_seed = SphericalIndexer(master, det_seed, bandwidth=BANDWIDTH, device="cpu")
    coarse, _ = idx_seed.index(patterns)

    # LM is orientation-only, so refine_pattern_center=True must transparently use the Adam path --
    # which still recovers both the orientation and the pattern center.
    out = refine_orientation(
        patterns,
        master,
        det_seed,
        coarse,
        pattern_center=pc_seed,
        refine_pattern_center=True,
        method="lm",
        directions=idx_seed._dirs,
        n_iterations=200,
        learning_rate=0.05,
        device="cpu",
    )
    assert len(out) == 3  # (rotation, score, pattern_center) -- the pattern_center IS returned
    refined, score, reference_pattern_center = out
    assert tuple(reference_pattern_center.shape) == (3,)
    assert (
        float((reference_pattern_center - pc_true_t).norm()) < 0.25 * seed_err
    )  # pattern_center genuinely refined
    for i in range(r_true.size):
        assert _dis(r_true[i], refined[i]) < 0.5


def test_lm_single_and_numpy_input(master, detector, indexer):
    """LM accepts a bare ``(H, W)`` pattern and numpy input, returning length-1 results."""
    r_true = Rotation.from_euler(np.deg2rad([[30.0, 40.0, 50.0]]))
    pattern = master.project(detector, r_true, rescale=True)[0].cpu().numpy()  # (H, W)
    assert pattern.ndim == 2
    coarse, _ = indexer.index(pattern)

    refined, score = refine_orientation(
        pattern, master, detector, coarse, method="lm", n_iterations=12
    )
    assert refined.size == 1
    assert tuple(score.shape) == (1,)
    assert _dis(r_true[0], refined[0]) < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lm_cuda_path(master, detector):
    """The LM path runs on CUDA and recovers the orientation there."""
    master_cuda = master.to("cuda")
    idx = SphericalIndexer(master_cuda, detector, bandwidth=BANDWIDTH, device="cuda")
    r_true = Rotation.from_euler(np.deg2rad([[123.0, 47.0, 205.0]]))
    pattern = master_cuda.project(detector, r_true, rescale=True)[0]
    coarse, _ = idx.index(pattern)
    refined, score = refine_orientation(
        pattern, master_cuda, detector, coarse, method="lm", n_iterations=12, device="cuda"
    )
    assert score.device.type == "cuda"
    assert _dis(r_true[0], refined[0]) < 0.5
