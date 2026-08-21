"""The PRISM scattering-matrix backend reproduces the eig reference exactly.

PRISM computes the same depth-integrated Lgh by matrix exponential instead of
eigendecomposition, so its intensities must equal BlochWaveBackend to machine
precision in float64. This is the gate that lets the fast backend inherit the
reference's validated correlation against EMsoft.
"""

import numpy as np
import torch

from torchebsd import BetheParameters
from torchebsd.dynamical import SimulationInputs, resolve_backend
from torchebsd.prism import BlochPrismBackend


def _dirs(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    return torch.as_tensor(v / np.linalg.norm(v, axis=1, keepdims=True))


def test_bloch_prism_registered():
    assert resolve_backend("bloch_prism").name == "bloch_prism"


def test_prism_matches_bloch_reference(si_sf):
    """PRISM (matrix_exp) equals the eig backend to machine precision (fp64)."""
    inp = SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                           device=torch.device("cpu"), dtype=torch.float64)
    oracle = resolve_backend("bloch")
    oracle.prepare(inp)
    prism = resolve_backend("bloch_prism")
    prism.prepare(inp)

    d = _dirs(24, seed=1)
    io = oracle.intensities(d)
    ip = prism.intensities(d)
    assert ip.shape == io.shape
    assert torch.all(torch.isfinite(ip))
    assert ip.std() > 0  # real dynamical contrast
    denom = io.abs().max().clamp_min(1e-30)
    assert ((ip - io).abs().max() / denom).item() < 1e-9


def test_prism_batches_across_strong_counts(si_sf):
    """Directions spanning many strong-beam counts (multiple buckets) all match."""
    inp = SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                           device=torch.device("cpu"), dtype=torch.float64)
    oracle = resolve_backend("bloch")
    oracle.prepare(inp)
    prism = resolve_backend("bloch_prism")
    prism.prepare(inp)

    d = _dirs(64, seed=2)
    io = oracle.intensities(d)
    ip = prism.intensities(d)
    denom = io.abs().max().clamp_min(1e-30)
    assert ((ip - io).abs().max() / denom).item() < 1e-9


def test_prism_complex64_matches_reference(si_sf):
    """The complex64 (production-fast) path tracks the fp64 reference to ~1e-4.

    complex128 is exact against the eig reference; complex64 trades a little
    accuracy for speed/memory. This guards that the fast dtype stays a faithful
    approximation (measured rel err ~2e-5) rather than drifting silently.
    """
    inp64 = SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                             device=torch.device("cpu"), dtype=torch.float64)
    oracle = resolve_backend("bloch")
    oracle.prepare(inp64)

    prism = BlochPrismBackend(cdtype=torch.complex64)
    inp32 = SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                             device=torch.device("cpu"), dtype=torch.float32)
    prism.prepare(inp32)

    d = _dirs(24, seed=1)
    io = oracle.intensities(d)
    ip = prism.intensities(d)
    assert ip.dtype == torch.float32
    denom = io.abs().max().clamp_min(1e-30)
    assert ((ip.double() - io).abs().max() / denom).item() < 1e-3
