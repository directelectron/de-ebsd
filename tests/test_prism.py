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


def test_bloch_solve_is_differentiable(si_sf):
    """Autograd flows through the prism Bloch solve to the structure factors.

    The production backend is torch-native (matrix_exp + a depth recurrence), so
    gradients propagate through the non-Hermitian complex Bloch solve, the
    weak-beam folding, and the Sgh reduction back to U_g -- the differentiability
    the project is built for. Validated against complex finite differences.
    """
    prism = resolve_backend("bloch_prism")
    prism.prepare(SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                                   device=torch.device("cpu"), dtype=torch.float64))
    u0 = prism.u_lut.detach().clone()
    prism.u_lut = u0.clone().requires_grad_(True)

    d = _dirs(4, seed=0)
    intensity = prism.intensities(d)
    assert intensity.requires_grad
    intensity.sum().backward()
    grad = prism.u_lut.grad
    assert grad is not None and torch.isfinite(grad).all() and bool((grad.abs() > 0).any())

    # complex finite-difference check on the most influential U_g entry
    idx = np.unravel_index(int(grad.abs().flatten().argmax()), grad.shape)

    def total(pert):
        prism.u_lut = u0.clone()
        prism.u_lut[idx] = prism.u_lut[idx] + pert
        return float(prism.intensities(d).sum())

    eps = 1e-4
    g_re_fd = (total(eps) - total(-eps)) / (2 * eps)
    g_im_fd = (total(1j * eps) - total(-1j * eps)) / (2 * eps)
    g_auto = complex(grad[idx])
    assert abs(g_auto.real - g_re_fd) < 1e-3
    assert abs(g_auto.imag - g_im_fd) < 1e-3


def test_intensity_differentiable_wrt_positions(si_sf):
    """End-to-end: atom positions -> torch structure factors -> prism Bloch solve
    -> intensity, with the gradient matching finite differences.

    This closes the differentiable chain from a crystal parameter to the master-
    pattern intensity: torch U_g (differentiable_ucg_lut) feeds the torch backend.
    """
    from torchebsd.structure_factors import differentiable_ucg_lut

    crystal = si_sf.crystal
    prism = resolve_backend("bloch_prism")
    prism.prepare(SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                                   device=torch.device("cpu"), dtype=torch.float64))
    base = [torch.as_tensor(np.asarray(p), dtype=torch.float64) for p in crystal.positions]
    d = _dirs(4, seed=1)

    def total(shift):
        pos = [b.clone() for b in base]
        pos[0] = pos[0].clone()
        pos[0][0, 0] = pos[0][0, 0] + shift
        prism.u_lut = differentiable_ucg_lut(
            crystal, si_sf.energy_kev, si_sf.dmin, positions=pos
        ).to(torch.complex128)
        return prism.intensities(d).sum()

    s = torch.zeros((), dtype=torch.float64, requires_grad=True)
    total(s).backward()
    eps = 1e-5
    with torch.no_grad():
        g_fd = (float(total(torch.tensor(eps))) - float(total(torch.tensor(-eps)))) / (2 * eps)
    assert abs(float(s.grad) - g_fd) / (abs(g_fd) + 1e-9) < 1e-2


def test_intensity_differentiable_wrt_debye_waller(si_sf):
    """End-to-end d(intensity)/d(Debye-Waller) through the torch WK amplitude,
    the structure factors, and the prism Bloch solve, matching finite diffs."""
    from torchebsd.structure_factors import differentiable_ucg_lut

    crystal = si_sf.crystal
    prism = resolve_backend("bloch_prism")
    prism.prepare(SimulationInputs(sf=si_sf, bethe=BetheParameters(), monte_carlo=None,
                                   device=torch.device("cpu"), dtype=torch.float64))
    d = _dirs(4, seed=2)
    b0 = float(crystal.sites[0].debye_waller)

    def total(bval):
        prism.u_lut = differentiable_ucg_lut(
            crystal, si_sf.energy_kev, si_sf.dmin, debye_waller=[bval]
        ).to(torch.complex128)
        return prism.intensities(d).sum()

    b = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
    total(b).backward()
    assert torch.isfinite(b.grad)
    eps = 1e-6
    with torch.no_grad():
        fp, fm = float(total(torch.tensor(b0 + eps))), float(total(torch.tensor(b0 - eps)))
    g_fd = (fp - fm) / (2 * eps)
    assert abs(float(b.grad) - g_fd) / (abs(g_fd) + 1e-9) < 1e-3
