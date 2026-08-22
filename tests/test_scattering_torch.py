"""Torch Weickenmeier-Kohl amplitude: matches numpy and differentiates in DW."""

import numpy as np
import torch

from torchebsd.scattering import wk_scattering_factor
from torchebsd.scattering_torch import wk_scattering_factor_torch


def test_torch_amplitude_matches_numpy():
    """The torch amplitude reproduces the numpy WK amplitude across elements."""
    g = np.linspace(0.0, 60.0, 200)
    g_t = torch.as_tensor(g, dtype=torch.float64)
    for z in (14, 28, 79):  # Si, Ni, Au
        for ul in (0.05, 0.09):
            f_np = wk_scattering_factor(z, g, ul, 20.0)
            f_t = wk_scattering_factor_torch(z, g_t, torch.tensor(ul, dtype=torch.float64), 20.0)
            assert np.abs(f_t.detach().numpy() - f_np).max() < 1e-6


def test_torch_amplitude_debye_waller_gradient():
    """d|f|^2/d(ul) through the absorptive integrals matches finite differences."""
    z, v_kv = 14, 20.0
    g = torch.linspace(0.0, 60.0, 300, dtype=torch.float64)
    ul0 = 0.077
    ul = torch.tensor(ul0, dtype=torch.float64, requires_grad=True)
    (wk_scattering_factor_torch(z, g, ul, v_kv).abs() ** 2).sum().backward()
    assert torch.isfinite(ul.grad)  # the absorptive-branch NaN trap is disarmed

    g_np = g.numpy()

    def total(u):
        return float((np.abs(wk_scattering_factor(z, g_np, u, v_kv)) ** 2).sum())

    d = 1e-6
    fd = (total(ul0 + d) - total(ul0 - d)) / (2 * d)
    assert abs(float(ul.grad) - fd) / abs(fd) < 1e-6
