"""Dynamical matrix construction and the backend seam."""

import numpy as np
import pytest
import torch

from de_ebsd import BetheParameters, BlochWaveBackend
from de_ebsd.diffraction.backends import (
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


def test_calc_lgh_single_beam():
    """The depth integral for a single (incident) beam matches a direct sum."""
    from de_ebsd.diffraction.backends import calc_lgh

    a = np.array([[0.0 + 0.3j]], dtype=np.complex128)  # one strong beam
    kn, thick, depthstep = 116.0, 50.0, 1.0
    lam = np.exp(-np.arange(20) / 5.0)
    lgh = calc_lgh(a, kn, thick, depthstep, lam, gzero=0)

    w = a[0, 0] / (2 * kn)
    qold = complex(2 * np.pi * depthstep * 2 * w.imag, 0.0)
    if qold.real < 0:
        qold = -qold
    q = sum(lam[iz - 1] * np.exp(-qold * iz) for iz in range(1, len(lam) + 1))
    expected = q * (depthstep / thick)
    assert lgh.shape == (1, 1)
    assert abs(lgh[0, 0] - expected) < 1e-9


def test_bloch_backend_intensities(si_sf):
    """The Bloch backend returns finite, non-uniform backscatter intensities."""
    backend = BlochWaveBackend()
    backend.prepare(
        SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                         device=torch.device("cpu"), dtype=torch.float32)
    )
    v = torch.randn(12, 3)
    v = v / v.norm(dim=-1, keepdim=True)
    intensity = backend.intensities(v)
    assert intensity.shape == (12,)
    assert torch.all(torch.isfinite(intensity))
    assert intensity.std() > 0  # real dynamical contrast, not a constant map


def test_resolve_backend():
    assert resolve_backend("bloch_prism").name == "bloch_prism"
    assert resolve_backend("bloch").name == "bloch"
    instance = BlochWaveBackend()
    assert resolve_backend(instance) is instance
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("prism")
