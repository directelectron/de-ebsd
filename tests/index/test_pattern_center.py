"""Shared pattern-center calibration (:func:`de_ebsd.index.determine_pattern_center`).

``determine_pattern_center`` fits ONE pattern center (PC) shared across many patterns --
the well-posed way to determine a PC. A single pattern is R--PC degenerate (a small PC
shift mimics a small rotation), so a per-pattern joint fit wanders; tying one PC to a whole
ensemble of differently-oriented patterns over-determines the three PC numbers and breaks
that degeneracy. The routine alternates an orientation-only refine of every pattern (at the
current shared PC) with a Levenberg-Marquardt step on the shared PC whose 3x3 normal
equations are **summed over all patterns**.

The money tests:

1. **PC recovery** -- project ``N=24`` patterns at known orientations and a KNOWN true PC,
   start from a WRONG ``pc0`` (off by 0.03 per component), and recover the shared PC to
   ``< 0.005`` (>= 6x better than the seed), on both a smooth and a sharper band master and
   several wrong-PC seeds.
2. **Degeneracy contrast** -- fitting the PC from single patterns (``N=1``) is *erratic*
   (their errors scatter over 100x and the worst is several-fold worse than the ensemble),
   while the ``N=20`` ensemble is reliably accurate -- demonstrating the ensemble breaks the
   R--PC degeneracy.
3. **Robustness** -- several wrong ``pc0`` seeds converge to the same PC;
   ``refine_orientations=True`` (from perturbed seeds) and ``=False`` (from the true
   orientations) both recover it.
4. **Workflow integration** -- after calibration, orientation-only
   ``refine_hr(refine_pc=False, pc=<calibrated>)`` recovers the known orientations to
   ``< 0.1`` deg, whereas at the wrong ``pc0`` it lands degrees off.
5. **Plumbing** -- numpy and torch input agree; ``return_orientations``; the ``info`` diag
   fields; single ``(H, W)`` and batched ``(N, H, W)`` inputs; shape validation; a
   CUDA-guarded path.

fp64 on CPU at small sizes throughout -- exact-ish, fast, small.
"""

import math

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation

from de_ebsd import EBSDDetector, MasterPattern
from de_ebsd.core.lambert import master_pattern_grid
from de_ebsd.index import determine_pattern_center, refine_hr

# fixed band-edge sample budget for the whole file (crisp, fast, plenty for 3 PC params)
_NT, _SPT = 12, 8


def _band_field(v: torch.Tensor, poles: torch.Tensor, amps: torch.Tensor, width: float):
    """A smooth Kikuchi-band-like field: bright Gaussian bands about great circles."""
    d = v.to(torch.float64) @ poles.T
    return (amps * torch.exp(-(d / width) ** 2)).sum(-1)


def _make_band_master(npx: int, width: float, nbands: int, seed: int) -> MasterPattern:
    """A band master of ``nbands`` great-circle bands with high-gradient edges."""
    g = torch.Generator().manual_seed(seed)
    poles = torch.randn(nbands, 3, generator=g, dtype=torch.float64)
    poles = poles / poles.norm(dim=-1, keepdim=True)
    amps = 0.5 + torch.rand(nbands, generator=g, dtype=torch.float64)
    north = master_pattern_grid(npx, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    data = torch.stack(
        [_band_field(north, poles, amps, width), _band_field(south, poles, amps, width)],
        dim=0,
    )[None]  # (1, 2, n, n)
    return MasterPattern(data=data, energies_kev=np.array([20.0]))


@pytest.fixture(scope="module")
def master() -> MasterPattern:
    """A sharper band master (narrow bands -> crisp edges -> tight PC information)."""
    return _make_band_master(npx=160, width=0.06, nbands=14, seed=11)


@pytest.fixture(scope="module")
def smooth_master() -> MasterPattern:
    """A smooth band master (broad bands) -- the harder, lower-gradient case."""
    return _make_band_master(npx=140, width=0.11, nbands=12, seed=7)


@pytest.fixture(scope="module")
def detector() -> EBSDDetector:
    """A tilted detector with a wide field of view (small pcz -> curved bands)."""
    return EBSDDetector(shape=(72, 72), pc=(0.5, 0.5, 0.22), sample_tilt=70.0, tilt=8.0)


def _orientations(n: int, seed: int) -> Rotation:
    """``n`` well-distributed random orientations (a stand-in calibration subset)."""
    rng = np.random.default_rng(seed)
    eul = rng.uniform([0.0, 0.0, 0.0], [360.0, 180.0, 360.0], size=(n, 3))
    return Rotation.from_euler(np.deg2rad(eul))


def _perturb(r: Rotation, deg: float, seed: int) -> Rotation:
    """Rotate each orientation by ``deg`` about a random axis (a coarse-index seed)."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=(r.size, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return Rotation.from_axes_angles(axis, np.deg2rad(deg)) * r


def _dis(a: Rotation, b: Rotation) -> float:
    """Disorientation (degrees) between two orix Rotations, no symmetry applied."""
    return float(np.asarray((a * (~b)).angle).ravel()[0]) * 180.0 / math.pi


# --------------------------------------------------------------------------- #
# 1. PC recovery -- the money test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("which", ["sharp", "smooth"])
@pytest.mark.parametrize("off", [(0.03, 0.03, 0.03), (-0.03, 0.02, -0.03)])
def test_pc_recovery(which, off, master, smooth_master, detector):
    """Recover a shared PC from a wrong ``pc0`` given N=24 patterns at known orientations.

    Project 24 patterns at random orientations and the true PC, hand the routine the true
    orientations and a ``pc0`` offset by 0.03 in every component, and assert the calibrated
    shared PC is within ``0.005`` of truth AND at least 6x closer than the seed -- on both a
    smooth and a sharper band master, for several wrong-PC seeds.
    """
    mp = master if which == "sharp" else smooth_master
    pc_true = torch.tensor(detector.pc, dtype=torch.float64)
    r_true = _orientations(24, seed=0)
    patterns = mp.project(detector, r_true, rescale=True)

    pc0 = pc_true + torch.tensor(off, dtype=torch.float64)
    seed_err = float((pc0 - pc_true).norm())

    pc_ref, info = determine_pattern_center(
        patterns, mp, detector, r_true, pc0=pc0, n_rounds=8,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    ref_err = float((pc_ref - pc_true).norm())

    assert tuple(pc_ref.shape) == (3,)
    assert ref_err < 0.005, f"{which} {off}: refined PC err {ref_err:.5f} not < 0.005"
    assert ref_err < seed_err / 6.0, (
        f"{which} {off}: seed err {seed_err:.4f} -> refined {ref_err:.5f} (< 6x better)"
    )
    assert info["ncc"] > info["ncc_seed"]  # the calibration improved the fit


# --------------------------------------------------------------------------- #
# 2. degeneracy contrast -- single pattern is erratic, the ensemble is reliable
# --------------------------------------------------------------------------- #
def test_single_pattern_degenerate_ensemble_reliable(master, detector):
    """Single-pattern PC fits scatter wildly; the shared ensemble fit is reliable.

    Fitting the PC from one pattern at a time (``N=1``) is nearly R--PC degenerate: a small
    PC shift is almost a small rotation, so the fit is ill-conditioned and noise moves it a
    long way. Across a set of noisy patterns the recovered PC error scatters widely and the
    worst is several-fold worse than the ensemble -- you cannot trust any single one.
    Fitting one shared PC over the whole ``N=20`` set (the ensemble accumulation) recovers
    it reliably. Noise-free, even a single pattern is solved exactly, so the noise is what
    this test is about.
    """
    pc_true = torch.tensor(detector.pc, dtype=torch.float64)
    n = 20
    r_true = _orientations(n, seed=0)
    patterns = master.project(detector, r_true, rescale=True)
    g = torch.Generator().manual_seed(5)
    patterns = patterns + 0.3 * patterns.std() * torch.randn(
        patterns.shape, generator=g, dtype=patterns.dtype)
    pc0 = pc_true + 0.03

    single = np.array([
        float(
            (
                determine_pattern_center(
                    patterns[i:i + 1], master, detector, r_true[i:i + 1], pc0=pc0,
                    n_rounds=12, n_tiles=_NT, samples_per_tile=_SPT,
                )[0]
                - pc_true
            ).norm()
        )
        for i in range(n)
    ])
    pc_ens, _ = determine_pattern_center(
        patterns, master, detector, r_true, pc0=pc0, n_rounds=12,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    ens = float((pc_ens - pc_true).norm())

    assert ens < 0.003, f"ensemble PC err {ens:.5f} not reliably small"
    # the ensemble beats the typical single pattern...
    assert ens < float(single.mean()), (
        f"ensemble {ens:.5f} not below mean single-pattern err {single.mean():.5f}"
    )
    # ...and you cannot trust any single one: the worst is far worse than the ensemble
    assert float(single.max()) > 3.0 * ens, (
        f"worst single {single.max():.5f} not >> ensemble {ens:.5f}"
    )


# --------------------------------------------------------------------------- #
# 3. robustness -- many seeds agree; both orientation modes work
# --------------------------------------------------------------------------- #
def test_multiple_seeds_converge_to_same_pc(master, detector):
    """Several different wrong ``pc0`` seeds converge to the same calibrated PC."""
    pc_true = torch.tensor(detector.pc, dtype=torch.float64)
    r_true = _orientations(24, seed=1)
    patterns = master.project(detector, r_true, rescale=True)

    seeds = [(0.03, 0.03, 0.03), (-0.03, -0.02, 0.03), (0.02, -0.03, -0.03)]
    refined = []
    for off in seeds:
        pc_ref, _ = determine_pattern_center(
            patterns, master, detector, r_true,
            pc0=pc_true + torch.tensor(off, dtype=torch.float64),
            n_rounds=8, n_tiles=_NT, samples_per_tile=_SPT,
        )
        refined.append(pc_ref)
        assert float((pc_ref - pc_true).norm()) < 0.005
    stack = torch.stack(refined)
    spread = float((stack - stack.mean(0)).norm(dim=-1).max())
    assert spread < 0.003, f"seeds disagree by {spread:.5f} (not a common PC)"


def test_refine_orientations_true_and_false_both_recover(master, detector):
    """Both orientation modes recover the PC: perturbed+refine, and fixed-true-orientations.

    ``refine_orientations=True`` starts from coarse (~1 deg perturbed) orientations and
    co-refines them; ``refine_orientations=False`` trusts the given (here true) orientations
    and fits the PC alone (which, at correct orientations, pins it near-exactly).
    """
    pc_true = torch.tensor(detector.pc, dtype=torch.float64)
    r_true = _orientations(24, seed=2)
    patterns = master.project(detector, r_true, rescale=True)
    pc0 = pc_true + 0.03

    # (a) coarse-perturbed seeds, orientations co-refined
    seed_r = _perturb(r_true, 1.0, seed=10)
    pc_a, _ = determine_pattern_center(
        patterns, master, detector, seed_r, pc0=pc0, n_rounds=12, refine_orientations=True,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    assert float((pc_a - pc_true).norm()) < 0.003

    # (b) true orientations held fixed -> PC alone, near-exact
    pc_b, _ = determine_pattern_center(
        patterns, master, detector, r_true, pc0=pc0, n_rounds=6, refine_orientations=False,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    assert float((pc_b - pc_true).norm()) < 5e-4


# --------------------------------------------------------------------------- #
# 4. workflow integration -- calibrate PC, then refine orientations at fixed PC
# --------------------------------------------------------------------------- #
def test_workflow_orientation_refine_after_calibration(master, detector):
    """The downstream payoff: orientations refine to < 0.1 deg only at the calibrated PC.

    Calibrate the shared PC from coarse-seeded patterns, then run orientation-only
    ``refine_hr(refine_pc=False, pc=<calibrated>)`` on the whole set: the disorientations
    fall below 0.1 deg. Running the same refine at the original wrong ``pc0`` instead leaves
    them degrees off -- tying the calibration to its benefit.
    """
    pc_true = torch.tensor(detector.pc, dtype=torch.float64)
    n = 24
    r_true = _orientations(n, seed=0)
    patterns = master.project(detector, r_true, rescale=True)
    pc0 = pc_true + 0.03
    seed_r = _perturb(r_true, 1.0, seed=20)

    pc_cal, rot_cal, info = determine_pattern_center(
        patterns, master, detector, seed_r, pc0=pc0, n_rounds=18,
        n_tiles=_NT, samples_per_tile=_SPT, return_orientations=True,
    )
    assert isinstance(rot_cal, Rotation) and rot_cal.size == n
    assert float((pc_cal - pc_true).norm()) < 8e-4  # tight enough for sub-0.1-deg orient

    good, _ = refine_hr(
        patterns, master, detector, seed_r, pc=pc_cal, refine_pc=False,
        n_tiles=_NT, samples_per_tile=_SPT, n_steps=30,
    )
    bad, _ = refine_hr(
        patterns, master, detector, seed_r, pc=pc0, refine_pc=False,
        n_tiles=_NT, samples_per_tile=_SPT, n_steps=30,
    )
    good_dis = np.array([_dis(r_true[i], good[i]) for i in range(n)])
    bad_dis = np.array([_dis(r_true[i], bad[i]) for i in range(n)])

    assert float(good_dis.max()) < 0.1, (
        f"orient refine at calibrated PC: max {good_dis.max():.4f} deg not < 0.1"
    )
    assert float(np.median(bad_dis)) > 1.0, (
        f"orient refine at wrong pc0 unexpectedly good ({np.median(bad_dis):.3f} deg)"
    )


# --------------------------------------------------------------------------- #
# 5. plumbing
# --------------------------------------------------------------------------- #
def test_numpy_and_torch_input_agree(master, detector):
    """A numpy pattern stack gives the same calibrated PC as the torch stack."""
    pc_true = torch.tensor(detector.pc, dtype=torch.float64)
    r_true = _orientations(12, seed=3)
    patterns = master.project(detector, r_true, rescale=True)
    pc0 = pc_true + 0.02

    pc_t, _ = determine_pattern_center(
        patterns, master, detector, r_true, pc0=pc0, n_rounds=4,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    pc_n, _ = determine_pattern_center(
        patterns.cpu().numpy(), master, detector, r_true, pc0=pc0, n_rounds=4,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    assert torch.allclose(pc_t, pc_n, atol=1e-9)


def test_return_orientations_and_info_fields(master, detector):
    """``return_orientations`` yields ``(pc, Rotation, info)``; ``info`` reports the fit."""
    pc_true = torch.tensor(detector.pc, dtype=torch.float64)
    r_true = _orientations(16, seed=4)
    patterns = master.project(detector, r_true, rescale=True)
    pc0 = pc_true + 0.03

    out = determine_pattern_center(
        patterns, master, detector, r_true, pc0=pc0, n_rounds=6,
        n_tiles=_NT, samples_per_tile=_SPT, return_orientations=True,
    )
    assert len(out) == 3
    pc_ref, rotation, info = out
    assert tuple(pc_ref.shape) == (3,)
    assert isinstance(rotation, Rotation) and rotation.size == 16

    assert set(info) == {"ncc", "ncc_seed", "pc_step", "pc_step_history", "n_rounds"}
    assert info["n_rounds"] == 6
    assert len(info["pc_step_history"]) == 6
    assert -1.0 <= info["ncc"] <= 1.0
    assert info["ncc"] > info["ncc_seed"]  # improved over the seed
    # converged: the last round moved the PC less than the first
    assert info["pc_step"] < info["pc_step_history"][0]

    # without the flag, only (pc, info)
    two = determine_pattern_center(
        patterns, master, detector, r_true, pc0=pc0, n_rounds=2,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    assert len(two) == 2 and tuple(two[0].shape) == (3,) and isinstance(two[1], dict)


def test_single_pattern_and_batch_shapes(master, detector):
    """A bare ``(H, W)`` input is accepted (N=1), as is a batched ``(N, H, W)`` stack."""
    r_true = _orientations(3, seed=5)
    patterns = master.project(detector, r_true, rescale=True)
    pc0 = torch.tensor(detector.pc, dtype=torch.float64) + 0.02

    pc_single, _ = determine_pattern_center(
        patterns[0], master, detector, r_true[0], pc0=pc0, n_rounds=2,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    assert tuple(pc_single.shape) == (3,)

    pc_batch, _ = determine_pattern_center(
        patterns, master, detector, r_true, pc0=pc0, n_rounds=2,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    assert tuple(pc_batch.shape) == (3,)


def test_shape_validation(master, detector):
    """A rotations/patterns length mismatch and a bad ``pc0`` shape both raise."""
    r_true = _orientations(4, seed=6)
    patterns = master.project(detector, r_true, rescale=True)

    with pytest.raises(ValueError, match="rotations length"):
        determine_pattern_center(
            patterns, master, detector, _orientations(3, seed=6), n_rounds=1,
            n_tiles=_NT, samples_per_tile=_SPT,
        )
    with pytest.raises(ValueError, match="pc0 must be"):
        determine_pattern_center(
            patterns, master, detector, r_true, pc0=torch.zeros(2), n_rounds=1,
            n_tiles=_NT, samples_per_tile=_SPT,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_path(master, detector):
    """Calibration runs on CUDA and returns the PC on the CUDA device."""
    master_cuda = master.to("cuda")
    pc_true = torch.tensor(detector.pc, dtype=torch.float64)
    r_true = _orientations(12, seed=7)
    patterns = master_cuda.project(detector, r_true, rescale=True)
    pc0 = pc_true + 0.03

    pc_ref, info = determine_pattern_center(
        patterns, master_cuda, detector, r_true, pc0=pc0, n_rounds=8,
        n_tiles=_NT, samples_per_tile=_SPT, device="cuda",
    )
    assert pc_ref.device.type == "cuda"
    assert float((pc_ref.cpu() - pc_true).norm()) < 0.005


def test_a_sharp_master_is_reached_from_outside_its_basin(detector):
    """A real dynamical master's band edges are narrow: its fit falls to ~0
    within 0.02 of PC, so a start 0.03 off sees no slope. The coarse-to-fine
    blur schedule must still recover the PC -- and without it, it must not
    (otherwise this test is not testing the schedule)."""
    sharp = _make_band_master(npx=200, width=0.012, nbands=30, seed=3)
    pc_true = torch.tensor(detector.pc, dtype=torch.float64)
    r_true = _orientations(24, seed=0)
    patterns = sharp.project(detector, r_true, rescale=True)
    pc0 = pc_true + torch.tensor((0.03, -0.03, 0.03), dtype=torch.float64)

    pc_ref, info = determine_pattern_center(
        patterns, sharp, detector, _perturb(r_true, 1.0, seed=4), pc0=pc0, n_rounds=8,
        n_tiles=_NT, samples_per_tile=_SPT,
    )
    assert float((pc_ref - pc_true).norm()) < 0.005, info
    pc_flat, _ = determine_pattern_center(
        patterns, sharp, detector, _perturb(r_true, 1.0, seed=4), pc0=pc0, n_rounds=8,
        n_tiles=_NT, samples_per_tile=_SPT, blur=None,
    )
    assert float((pc_flat - pc_true).norm()) > 0.02
