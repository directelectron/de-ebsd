"""Reflection lists, excitation errors, and the Bethe partition."""

import numpy as np
import pytest
import torch

from torchebsd import BetheParameters
from torchebsd.sim.dynamical import classify_reflections
from torchebsd.sim.reflections import (
    IGNORED,
    STRONG,
    WEAK,
    excitation_errors,
    master_reflection_list,
)


def test_master_list_marks_forbidden(si_sf):
    refl = master_reflection_list(si_sf)
    hkl200 = np.all(refl.hkl == [2, 0, 0], axis=1)
    assert refl.forbidden[hkl200].all()
    hkl111 = np.all(refl.hkl == [1, 1, 1], axis=1)
    assert not refl.forbidden[hkl111].any()


def test_excitation_error_zero_on_ewald_sphere(si_sf):
    """s_g vanishes when k + g lies on the Ewald sphere."""
    lam = si_sf.wavelength
    k_hat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    k_len = 1.0 / lam
    # construct g on the sphere: g = (gx, 0, gz) with |k+g| = |k|,
    # i.e. gz^2 + 2 k gz + gx^2 = 0
    gx = 3.0
    gz = -k_len + np.sqrt(k_len**2 - gx**2)
    g = torch.tensor([[gx, 0.0, gz]], dtype=torch.float64)
    sg = excitation_errors(g, k_hat, lam)
    assert abs(sg[0].item()) < 1e-10


def test_excitation_error_sign_and_magnitude(si_sf):
    """Small g perpendicular to k: s_g ~ -g^2 lambda / 2 (inside the sphere)."""
    lam = si_sf.wavelength
    k_hat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    g = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    sg = excitation_errors(g, k_hat, lam).item()
    assert sg == pytest.approx(-(2.0**2) * lam / 2.0, rel=1e-3)


def test_excitation_errors_batched_shape(si_sf):
    k = torch.randn(5, 7, 3, dtype=torch.float64)
    k = k / k.norm(dim=-1, keepdim=True)
    g = torch.randn(11, 3, dtype=torch.float64)
    sg = excitation_errors(g, k, si_sf.wavelength)
    assert sg.shape == (5, 7, 11)


def test_classification_structure(si_sf):
    refl = classify_reflections(si_sf, np.array([0.4, 0.1, 1.0]) / np.linalg.norm([0.4, 0.1, 1.0]),
                                BetheParameters())
    # first entry is the incident beam, always strong, sg = 0
    assert np.all(refl.hkl[0] == 0)
    assert refl.status[0] == STRONG
    assert refl.sg[0] == 0.0
    assert set(np.unique(refl.status)) <= {STRONG, WEAK, IGNORED}
    assert refl.n_strong >= 1
    assert refl.n_strong + refl.n_weak <= len(refl)


def test_forbidden_reflections_are_never_beams(si_sf):
    """Systematically absent (double-diffraction) reflections carry no structure
    factor and must never be strong or weak beams.

    Along the [001] zone axis several ``(200)``/``(420)``-type reflections sit
    close to Bragg, so the degenerate self-difference term (``g - g = 0`` ->
    ``U_0``) would otherwise promote them to strong beams. EMsoft excludes them;
    including them corrupts the dynamical matrix (this was the Si master-pattern
    correlation bug).
    """
    refl = classify_reflections(si_sf, np.array([0.0, 0.0, 1.0]), BetheParameters())
    forbidden = refl.forbidden
    assert forbidden.any(), "expected forbidden reflections among the [001] candidates"
    assert np.all(refl.status[forbidden] == IGNORED)
    # allowed beams are unaffected: the incident beam and near-Bragg allowed
    # reflections are still strong
    assert refl.status[0] == STRONG
    assert refl.n_strong > 1


def test_tighter_bethe_gives_fewer_strong(si_sf):
    k = np.array([0.3, 0.2, 1.0]) / np.linalg.norm([0.3, 0.2, 1.0])
    loose = classify_reflections(si_sf, k, BetheParameters(strong=8.0, weak=50.0))
    tight = classify_reflections(si_sf, k, BetheParameters(strong=2.0, weak=50.0))
    assert tight.n_strong <= loose.n_strong


def test_bethe_parameter_validation():
    with pytest.raises(ValueError):
        BetheParameters(strong=50.0, weak=8.0)
