"""The whole-pattern fit against a reference (:mod:`de_ebsd.index.reference_fit`).

Test patterns are rendered straight from a band master with the lattice
deformed (``test_hrebsd._render``), so the ground truth does not share the
fit's remap. 128-pixel screens keep the tests quick on a CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_hrebsd import _BETA, _render, _rotation

from band_master import make_band_master
from de_ebsd import EBSDDetector
from de_ebsd.core.geometry import sample_to_detector_tensor
from de_ebsd.index.reference_fit import (
    fit_reference,
    lattice_turns,
    prepare_patterns,
    seed_from_orientations,
    within_grain,
)

DT = torch.float64
SIDE = 128


@pytest.fixture(scope="module")
def master():
    return make_band_master(npx=300, width=0.05, n_bands=18, seed=11)


@pytest.fixture(scope="module")
def detector():
    return EBSDDetector(
        shape=(SIDE, SIDE), pattern_center=(0.5, 0.5, 0.6), sample_tilt=70.0, tilt=0.0
    )


def _deviatoric_stretch(f, detector):
    """The strain the fit reports for *f*: the left stretch less its trace, in
    the sample frame."""
    f = f / torch.linalg.det(f) ** (1 / 3)
    u, s, _ = torch.linalg.svd(f)
    v = u @ torch.diag(s) @ u.T - torch.eye(3, dtype=DT)
    v = v - torch.eye(3, dtype=DT) * torch.trace(v) / 3
    om = sample_to_detector_tensor(detector, "cpu", DT)
    return om.T @ v @ om


def _background(dx, dy, width):
    y, x = torch.meshgrid(torch.arange(SIDE, dtype=DT), torch.arange(SIDE, dtype=DT), indexing="ij")
    return torch.exp(-((x - 70 - dx) ** 2 + (y - 45 - dy) ** 2) / (2 * width**2))


@pytest.mark.parametrize("deg", [0.0, 1.5, 4.0])
def test_a_strain_comes_back_from_an_orientation_seed(master, detector, deg):
    """A 2e-3 strain on a lattice turned by *deg*, under a backscatter that
    moved with the beam and with noise, seeded with the turn alone. Without the
    noise the error is about 2e-6: what is left is the noise's."""
    rng = np.random.default_rng(3)
    turn = _rotation([0.2, 0.9, -0.4], np.deg2rad(deg))
    f = turn @ (torch.eye(3, dtype=DT) + 2e-3 * _BETA)
    noise = torch.as_tensor(rng.standard_normal((2, SIDE, SIDE)))
    ref = _background(0, 0, 60) * (4 + _render(master, detector)) + 0.003 * noise[0]
    test = _background(3, 2, 58) * (4 + _render(master, detector, f)) + 0.01 * noise[1]
    fit = fit_reference(
        prepare_patterns(test.float()), prepare_patterns(ref.float())[0], detector, seed=turn
    )
    want = _deviatoric_stretch(f, detector)
    assert want.abs().max() > 1e-3  # the test means something
    error = (fit.strain[0].double() - want).abs().max()
    assert error < 2e-4, error
    assert bool(fit.converged[0])
    assert fit.score[0] > 0.99


def test_unstrained_patterns_at_two_centers_have_no_strain(master, detector):
    """The reference's and the test's pattern centers differ: told so, the fit
    finds the geometry, not a strain."""
    reference_center = (0.5, 0.5, 0.6)
    center = (0.506, 0.495, 0.604)
    ref = _render(master, detector, pattern_center=reference_center) + 2
    test = _render(master, detector, pattern_center=center) + 2
    fit = fit_reference(
        prepare_patterns(test.float()),
        prepare_patterns(ref.float())[0],
        detector,
        center,
        reference_center,
    )
    assert fit.strain.abs().max() < 1e-4, fit.strain.abs().max()
    assert fit.rotation_vector.abs().max() < 1e-4


def test_a_stack_in_batches_is_the_stack_at_once(master, detector):
    ref = prepare_patterns((_render(master, detector) + 2).float())[0]
    turns = [_rotation([0.3, -0.5, 0.8], np.deg2rad(d)) for d in (0.2, 0.6, 1.0)]
    test = prepare_patterns(torch.stack([_render(master, detector, t) + 2 for t in turns]).float())
    seed = torch.stack(turns)
    whole = fit_reference(test, ref, detector, seed=seed)
    parts = fit_reference(test, ref, detector, seed=seed, batch_pixels=SIDE * SIDE)
    assert torch.allclose(whole.deformation, parts.deformation, atol=1e-6)
    assert torch.equal(whole.converged, parts.converged)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_batch_at_camera_counts_is_each_pattern_alone(master, detector, device):
    """Patterns of tens of thousands of counts, several to a call: each comes
    out as it does alone, and the reference against itself is the identity.
    Summed in single precision, the normal equations of a batch on CUDA once
    sent the reference's own fit to a 2e-2 strain."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(8)
    background = _background(0, 0, 70)
    ref = 2e4 * background * (4 + _render(master, detector))
    turns = [_rotation([0.3, -0.5, 0.8], np.deg2rad(d)) for d in (0.3, 0.8)]
    test = [ref] + [2e4 * background * (4 + _render(master, detector, t)) for t in turns]
    test = torch.stack(test) + 300 * torch.as_tensor(rng.standard_normal((3, SIDE, SIDE)))
    test = prepare_patterns(test.float().to(device))
    ref = test[0]
    together = fit_reference(test, ref, detector)
    for i in range(len(test)):
        alone = fit_reference(test[i], ref, detector)
        assert torch.allclose(together.strain[i], alone.strain[0], atol=1e-6)
    assert together.strain[0].abs().max() < 1e-6
    assert together.converged.all()


def test_a_reference_of_another_shape_is_refused(master, detector):
    ref = torch.zeros(SIDE // 2, SIDE // 2)
    with pytest.raises(ValueError, match="reference"):
        fit_reference(torch.zeros(1, SIDE, SIDE), ref, detector)


# ---------------------------------------------------------------------------
# the grain and the seed
# ---------------------------------------------------------------------------

_CUBIC = np.stack(
    [
        np.diag(d) @ p
        for p in (np.eye(3), np.eye(3)[[1, 2, 0]], np.eye(3)[[2, 0, 1]])
        for d in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    ]
)  # the twelve proper rotations of 23, enough symmetry to test with


def _orientation(seed):
    return _rotation(np.random.default_rng(seed).normal(size=3), 1.0).numpy()


def test_the_turn_between_orientations_is_the_lattice_rotation_whatever_the_variant():
    """``g_test = g_ref T^T`` turns the lattice by ``T`` in the sample frame; a
    test indexed as another symmetry variant ``S g_test`` turns by it too."""
    g_ref = _orientation(1)
    turn = _rotation([1.0, 2.0, -0.5], np.deg2rad(2.0)).numpy()
    g_test = g_ref @ turn.T
    variants = np.stack([g_test, _CUBIC[5] @ g_test, _CUBIC[10] @ g_test])
    assert np.allclose(lattice_turns(variants, g_ref, _CUBIC), turn, atol=1e-12)


def test_the_seed_is_the_turn_in_the_detector_frame(detector):
    g_ref = _orientation(2)
    turn = _rotation([0.0, 1.0, 1.0], np.deg2rad(3.0))
    seed = seed_from_orientations((g_ref @ turn.numpy().T)[None], g_ref, _CUBIC, detector)
    om = sample_to_detector_tensor(detector, "cpu", DT)
    assert torch.allclose(seed[0], om @ turn @ om.T, atol=1e-12)


def test_within_grain_is_the_misorientation_over_the_symmetry():
    g_ref = _orientation(4)
    near = g_ref @ _rotation([1.0, 0.0, 1.0], np.deg2rad(4.0)).numpy().T
    far = g_ref @ _rotation([1.0, 0.0, 1.0], np.deg2rad(6.0)).numpy().T
    rotations = np.stack([near, _CUBIC[7] @ near, far, np.full((3, 3), np.nan)])
    assert within_grain(rotations, g_ref, _CUBIC).tolist() == [True, True, False, False]
