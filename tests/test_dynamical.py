"""Dynamical matrix construction and the backend seam."""

import numpy as np
import pytest
import torch

from torchebsd import BetheParameters, BlochWaveBackend, KinematicalBackend
from torchebsd.dynamical import (
    SimulationInputs,
    build_dynamical_matrix,
    classify_reflections,
    resolve_backend,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_matrix_shape_and_diagonal(si_sf):
    bethe = BetheParameters()
    refl = classify_reflections(si_sf, _unit([0.3, 0.1, 1.0]), bethe)
    a = build_dynamical_matrix(si_sf, refl)
    ns = refl.n_strong
    assert a.shape == (ns, ns)
    # the incident beam: sg = 0, no weak-beam diagonal shift of its real part
    assert a[0, 0].imag == pytest.approx(si_sf.upz0, rel=1e-12)


def test_off_diagonals_without_weak_beams(si_sf):
    """With the weak channel disabled, off-diagonals are exactly U_{g-h}."""
    bethe = BetheParameters(strong=8.0, weak=8.000001, cutoff=8.000001)  # no weak class
    refl = classify_reflections(si_sf, _unit([0.3, 0.1, 1.0]), bethe)
    assert refl.n_weak == 0
    a = build_dynamical_matrix(si_sf, refl)
    hkl_s = refl.strong()
    sg_s = refl.sg[refl.status == 0]
    lam = si_sf.wavelength
    for i in range(min(4, len(hkl_s))):
        for j in range(min(4, len(hkl_s))):
            if i == j:
                expected = 2.0 * sg_s[i] / lam + 1j * si_sf.upz0
            else:
                expected = si_sf.lookup(hkl_s[i] - hkl_s[j])
            assert a[i, j] == pytest.approx(expected, rel=1e-10)


def test_weak_beams_perturb_matrix(si_sf):
    """Folding weak beams in must change the matrix but keep it finite."""
    k = _unit([0.35, 0.15, 1.0])
    with_weak = build_dynamical_matrix(si_sf, classify_reflections(si_sf, k, BetheParameters()))
    no_weak = build_dynamical_matrix(
        si_sf, classify_reflections(si_sf, k, BetheParameters(strong=8.0, weak=8.000001,
                                                              cutoff=8.000001))
    )
    assert np.all(np.isfinite(with_weak))
    # the two matrices describe the same strong set differently sized or valued
    assert with_weak.shape != no_weak.shape or not np.allclose(with_weak, no_weak)


def test_bloch_backend_is_explicitly_unfinished(si_sf):
    backend = BlochWaveBackend()
    backend.prepare(
        SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                         device=torch.device("cpu"), dtype=torch.float32)
    )
    with pytest.raises(NotImplementedError, match="DESIGN.md"):
        backend.intensities(torch.eye(3))


def test_kinematical_backend_intensities(si_sf):
    backend = KinematicalBackend()
    backend.prepare(
        SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                         device=torch.device("cpu"), dtype=torch.float32)
    )
    v = torch.randn(100, 3)
    v = v / v.norm(dim=-1, keepdim=True)
    intensity = backend.intensities(v)
    assert intensity.shape == (100,)
    assert torch.all(intensity >= 0)
    assert torch.all(torch.isfinite(intensity))
    # inversion symmetry of the band model
    assert torch.allclose(backend.intensities(-v), intensity, atol=1e-6)


def test_resolve_backend():
    assert resolve_backend("kinematical").name == "kinematical"
    assert resolve_backend("bloch").name == "bloch"
    instance = KinematicalBackend()
    assert resolve_backend(instance) is instance
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("prism")
