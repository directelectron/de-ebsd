"""M-MC.1: material foundation for the native Monte Carlo.

Atomic weights, the Berger-Seltzer mean ionization potential, and the reduction
of a crystal to the mean bulk material the MC scatters in (mean Z / A, mass and
number densities). Densities are checked against handbook values as an
end-to-end test of the unit conversions.
"""

import os

import numpy as np
import pytest
import torch

from de_ebsd import Crystal
from de_ebsd.core.constants import C_AVOGADRO
from de_ebsd.crystal.scattering import atomic_weight
from de_ebsd.sim.montecarlo.depth import DepthDistribution
from de_ebsd.sim.montecarlo.transport import (
    MaterialProperties,
    elastic_cross_section_na,
    mean_free_path_nm,
    mean_ionization_potential_kev,
    rotate_direction,
    sample_scattering_cos,
    screening_parameter,
    simulate_depth_distribution,
    simulate_trajectory,
    soft_depth_histogram,
    stopping_power,
)


def _ni_material():
    return MaterialProperties.from_crystal(
        Crystal.from_parameters(a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))])
    )


def _ni_crystal():
    return Crystal.from_parameters(a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))])


def test_atomic_weight_values():
    assert atomic_weight("H") == pytest.approx(1.008, abs=1e-3)
    assert atomic_weight("Si") == pytest.approx(28.085, abs=1e-3)
    assert atomic_weight("Ni") == pytest.approx(58.693, abs=1e-3)
    assert atomic_weight("Au") == pytest.approx(196.97, abs=1e-2)


def test_atomic_weight_unknown_element():
    with pytest.raises(ValueError, match="Unknown element"):
        atomic_weight("Zz")


def test_mean_ionization_potential():
    # Joy-Luo: J[eV] = 9.76 Z + 58.5 Z^-0.19; returned in keV.
    assert mean_ionization_potential_kev(14) == pytest.approx(0.17210, abs=1e-4)  # Si
    assert mean_ionization_potential_kev(28) == pytest.approx(0.30434, abs=1e-4)  # Ni
    # monotonic increasing in Z
    z = np.arange(1, 99)
    j = np.array([mean_ionization_potential_kev(float(zz)) for zz in z])
    assert np.all(np.diff(j) > 0)


def test_mean_ionization_potential_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        mean_ionization_potential_kev(0.0)


def test_crystal_composition_single_element(si_crystal, ni_crystal):
    assert si_crystal.mean_atomic_number == pytest.approx(14.0)
    assert si_crystal.mean_atomic_weight == pytest.approx(28.085, abs=1e-3)
    assert ni_crystal.mean_atomic_number == pytest.approx(28.0)
    assert ni_crystal.mean_atomic_weight == pytest.approx(58.693, abs=1e-3)


def test_density_matches_handbook(si_crystal, ni_crystal):
    # Si 2.329, Ni 8.908 g/cm^3 (CRC handbook)
    assert si_crystal.density == pytest.approx(2.329, abs=5e-3)
    assert ni_crystal.density == pytest.approx(8.908, abs=1e-2)


def test_material_properties_si(si_crystal):
    mp = MaterialProperties.from_crystal(si_crystal)
    assert mp.mean_z == pytest.approx(14.0)
    assert mp.mean_a == pytest.approx(28.085, abs=1e-3)
    assert mp.density == pytest.approx(2.329, abs=5e-3)
    assert mp.mean_ionization_kev == pytest.approx(mean_ionization_potential_kev(14))
    # number density cross-checks the direct cell count: 8 Si atoms / V(nm^3)
    assert mp.number_density == pytest.approx(8.0 / si_crystal.volume, rel=1e-6)


def test_material_properties_ni(ni_crystal):
    mp = MaterialProperties.from_crystal(ni_crystal)
    assert mp.mean_z == pytest.approx(28.0)
    assert mp.number_density == pytest.approx(4.0 / ni_crystal.volume, rel=1e-6)  # 4 fcc atoms


def test_material_properties_compound_averaging():
    """A binary (GaAs) exercises composition (number) averaging and density."""
    gaas = Crystal.from_parameters(
        a=0.56533, space_group=216,
        atoms=[("Ga", (0.0, 0.0, 0.0)), ("As", (0.25, 0.25, 0.25))],
    )
    mp = MaterialProperties.from_crystal(gaas)
    assert mp.mean_z == pytest.approx(32.0)  # (31 + 33) / 2, equal 4a/4c occupancy
    assert mp.mean_a == pytest.approx(0.5 * (69.723 + 74.922), abs=1e-2)
    assert mp.density == pytest.approx(5.32, abs=0.1)  # handbook GaAs 5.317 g/cm^3


# --------------------------------------------------------------------------- #
# M-MC.2: single-scattering physics + reference trajectory
# --------------------------------------------------------------------------- #
def test_screening_parameter():
    # alpha = 3.4e-3 Z^(2/3) / E
    assert screening_parameter(28, 20.0) == pytest.approx(3.4e-3 * 28 ** (2 / 3) / 20.0)
    # screening weakens (alpha down) as energy rises
    assert screening_parameter(28, 30.0) < screening_parameter(28, 20.0)


def test_mean_free_path(ni_crystal):
    mp = MaterialProperties.from_crystal(ni_crystal)
    mfp20 = mean_free_path_nm(mp.mean_z, 20.0, mp.density, mp.mean_a)
    assert mfp20 == pytest.approx(5.17, abs=0.05)  # Ni, 20 keV, a few nm
    # longer free path at higher energy
    assert mean_free_path_nm(mp.mean_z, 30.0, mp.density, mp.mean_a) > mfp20


def test_mean_free_path_cross_check(ni_crystal):
    """MFP via A/(rho N_A sigma) equals 1/(n sigma) from the number density."""
    mp = MaterialProperties.from_crystal(ni_crystal)
    sigma_nm2 = elastic_cross_section_na(mp.mean_z, 20.0) / C_AVOGADRO * 1e14  # cm^2 -> nm^2
    mfp_from_n = 1.0 / (mp.number_density * sigma_nm2)
    assert mfp_from_n == pytest.approx(
        mean_free_path_nm(mp.mean_z, 20.0, mp.density, mp.mean_a), rel=1e-9
    )


def test_stopping_power(ni_crystal):
    mp = MaterialProperties.from_crystal(ni_crystal)
    de_ds = stopping_power(mp.mean_z, 20.0, mp.mean_a, mp.mean_ionization_kev)
    assert de_ds < 0.0  # energy decreases
    assert de_ds == pytest.approx(-8.15e-4, abs=5e-6)
    # energy lost over one mean free path is small (slow, continuous slowing down)
    mfp = mean_free_path_nm(mp.mean_z, 20.0, mp.density, mp.mean_a)
    loss = -de_ds * mp.density * mfp
    assert 0.0 < loss < 0.1  # keV


def test_sample_scattering_cos():
    alpha = screening_parameter(28, 20.0)
    assert sample_scattering_cos(alpha, 0.0) == pytest.approx(1.0)  # forward
    assert sample_scattering_cos(alpha, 1.0 - 1e-12) == pytest.approx(-1.0, abs=1e-3)  # backscatter
    u = np.linspace(0.0, 1.0 - 1e-9, 500)
    cos = sample_scattering_cos(alpha, u)
    assert np.all(cos <= 1.0 + 1e-12) and np.all(cos >= -1.0 - 1e-12)
    assert np.all(np.diff(cos) < 0)  # monotonically decreasing in u


def test_rotate_direction_preserves_norm_and_angle():
    rng = np.random.default_rng(0)
    for _ in range(100):
        c = rng.standard_normal(3)
        c /= np.linalg.norm(c)
        cos_theta = float(rng.uniform(-1, 1))
        psi = float(rng.uniform(0, 2 * np.pi))
        c_new = rotate_direction(c, cos_theta, psi)
        assert np.linalg.norm(c_new) == pytest.approx(1.0, abs=1e-12)
        assert float(c @ c_new) == pytest.approx(cos_theta, abs=1e-9)  # angle == theta


def test_rotate_direction_pole_special_case():
    # |c_z| ~ 1 branch: z-component becomes cos_theta, norm preserved
    c_new = rotate_direction(np.array([0.0, 0.0, 1.0]), 0.3, 1.2)
    assert np.linalg.norm(c_new) == pytest.approx(1.0, abs=1e-12)
    assert c_new[2] == pytest.approx(0.3)


def test_trajectory_deterministic_and_physical(ni_crystal):
    mp = MaterialProperties.from_crystal(ni_crystal)
    a = simulate_trajectory(mp, 20.0, rng=np.random.default_rng(42))
    b = simulate_trajectory(mp, 20.0, rng=np.random.default_rng(42))
    assert a == b  # same seed -> identical trajectory
    assert 0.0 < a.exit_energy_kev < 20.0  # electron lost energy
    assert a.max_depth_nm > 0.0
    assert a.n_steps >= 1


def test_backscatter_yield_sanity(ni_crystal, si_crystal):
    """Coarse yield check; the precise accum_z validation is M-MC.4."""
    ni = MaterialProperties.from_crystal(ni_crystal)
    si = MaterialProperties.from_crystal(si_crystal)

    def yield_at_tilt(mat, n=300):
        rng = np.random.default_rng(0)
        hits = sum(
            simulate_trajectory(mat, 20.0, tilt_deg=70.0, e_min_kev=2.0,
                                max_steps=400, rng=rng).backscattered
            for _ in range(n)
        )
        return hits / n

    eta_ni, eta_si = yield_at_tilt(ni), yield_at_tilt(si)
    assert 0.3 < eta_ni < 0.85  # high at 70 deg tilt
    assert 0.2 < eta_si < 0.75
    assert eta_ni > eta_si  # higher Z backscatters more


# --------------------------------------------------------------------------- #
# M-MC.3: vectorized MC -> DepthDistribution
# --------------------------------------------------------------------------- #
def test_from_simulation_returns_usable_distribution(ni_crystal):
    dd = DepthDistribution.from_simulation(
        ni_crystal, 20.0, e_min_kev=15.0, e_bin_kev=1.0,
        depth_step_nm=1.0, depth_max_nm=100.0, n_electrons=20_000, seed=0,
    )
    assert isinstance(dd, DepthDistribution)
    assert dd.n_energies == 6 and dd.n_depths == 101  # 15..20 keV, 0..100 nm
    assert np.all(np.diff(dd.energies_kev) > 0)  # ascending, like from_emsoft
    assert dd.energy_index(20.0) == dd.n_energies - 1  # beam energy is the top bin
    # the derived quantities the Bloch backend calls must accept it
    w = dd.depth_weights(20.0, normal_absorption_length=50.0)
    assert w.shape == (dd.n_depths,) and np.all(np.isfinite(w))
    assert dd.integration_depths().shape == (dd.n_energies,)


def test_depth_distribution_deterministic(ni_crystal):
    kw = dict(e_min_kev=15.0, n_electrons=20_000, seed=42)
    a = simulate_depth_distribution(ni_crystal, 20.0, **kw)
    b = simulate_depth_distribution(ni_crystal, 20.0, **kw)
    assert np.array_equal(a.accum_z, b.accum_z)


def test_depth_profile_is_shallow_weighted(ni_crystal):
    """High-energy backscatter comes from shallow depths: the profile decays."""
    dd = simulate_depth_distribution(ni_crystal, 20.0, e_min_kev=15.0,
                                     n_electrons=40_000, seed=1)
    top = dd.accum_z[-1, :, 0, 0]  # ~20 keV bin
    assert top.sum() > 0
    assert top[:5].sum() > top[20:].sum()  # concentrated near the surface


def test_vectorized_matches_scalar_yield(ni_crystal):
    """The batched MC reproduces the scalar reference's accepted yield."""
    mp = MaterialProperties.from_crystal(ni_crystal)
    rng = np.random.default_rng(7)
    m = 500
    scalar_hits = 0
    for _ in range(m):
        tr = simulate_trajectory(mp, 20.0, tilt_deg=70.0, e_min_kev=15.0, rng=rng)
        accepted = (
            tr.backscattered
            and 15.0 < tr.exit_energy_kev <= 20.0
            and abs(tr.escape_depth_nm) <= 100.0
        )
        scalar_hits += int(accepted)
    scalar_frac = scalar_hits / m

    dd = simulate_depth_distribution(ni_crystal, 20.0, e_min_kev=15.0,
                                     depth_max_nm=100.0, n_electrons=40_000, seed=3)
    vec_frac = dd.accum_z.sum() / dd.n_total
    assert abs(vec_frac - scalar_frac) < 0.05  # agree within Monte Carlo noise


def test_higher_energy_penetrates_deeper(ni_crystal):
    def mean_top_depth(energy_kev):
        dd = simulate_depth_distribution(ni_crystal, energy_kev, e_min_kev=energy_kev - 5.0,
                                         depth_max_nm=200.0, n_electrons=30_000, seed=5)
        top = dd.accum_z[-1, :, 0, 0]
        z = np.arange(len(top)) * dd.depth_step
        return (z * top).sum() / top.sum()

    assert mean_top_depth(30.0) > mean_top_depth(20.0)


def test_native_mc_feeds_bloch_backend(si_sf, si_crystal):
    """The native MC depth profile is a drop-in for the Bloch depth integration."""
    import torch

    from de_ebsd import BetheParameters
    from de_ebsd.diffraction.backends import BlochWaveBackend, SimulationInputs

    dd = simulate_depth_distribution(
        si_crystal, si_sf.energy_kev, e_min_kev=si_sf.energy_kev - 5.0,
        n_electrons=8_000, seed=0,
    )
    backend = BlochWaveBackend()
    backend.prepare(SimulationInputs(
        sf=si_sf, bethe=BetheParameters(), monte_carlo=dd,
        device=torch.device("cpu"), dtype=torch.float32,
    ))
    v = torch.randn(8, 3)
    v = v / v.norm(dim=-1, keepdim=True)
    inten = backend.intensities(v)
    assert inten.shape == (8,)
    assert torch.all(torch.isfinite(inten))
    assert inten.std() > 0  # real dynamical contrast with the native depth profile


# --------------------------------------------------------------------------- #
# M-MC.5: differentiable (soft) Monte Carlo
# --------------------------------------------------------------------------- #
def _soft_yield(params, td, te_, n=8000, max_steps=300, seed=1):
    z, a, r, j = params
    acc = soft_depth_histogram(z, a, r, j, 20.0, e_min_kev=15.0, n_electrons=n,
                               max_steps=max_steps, tau_depth=td, tau_energy=te_, seed=seed)
    return acc.sum() / n


def _tensors(mat, requires_grad=False):
    return tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=requires_grad)
        for v in (mat.mean_z, mat.mean_a, mat.density, mat.mean_ionization_kev)
    )


def test_soft_histogram_approaches_hard_as_tau_shrinks():
    mat = _ni_material()
    params = _tensors(mat)
    y_wide = float(_soft_yield(params, 2.0, 1.0, n=6000))
    y_narrow = float(_soft_yield(params, 0.3, 0.15, n=6000))
    hard = simulate_depth_distribution(_ni_crystal(), 20.0, e_min_kev=15.0,
                                       n_electrons=6000, seed=1)
    hard_yield = hard.accum_z.sum() / hard.n_total
    assert y_wide < hard_yield  # soft gates leak, underestimating the yield
    assert y_narrow > y_wide  # sharper gates -> closer to the hard limit
    assert abs(y_narrow - hard_yield) < abs(y_wide - hard_yield)  # converging


def test_soft_histogram_gradients_flow_to_all_params():
    params = _tensors(_ni_material(), requires_grad=True)
    (_soft_yield(params, 0.6, 0.3)).backward()
    grads = [p.grad for p in params]
    assert all(g is not None and torch.isfinite(g) for g in grads)
    assert sum(abs(float(g)) for g in grads) > 0  # yield depends on the material


def test_soft_histogram_gradient_matches_finite_difference():
    """Autograd d(yield)/d(density) matches common-random-number finite diff."""
    mat = _ni_material()
    z, a, j = (torch.tensor(v, dtype=torch.float64)
               for v in (mat.mean_z, mat.mean_a, mat.mean_ionization_kev))

    def yield_at(rho_val, want_grad):
        rho = torch.tensor(rho_val, dtype=torch.float64, requires_grad=want_grad)
        y = _soft_yield((z, a, rho, j), 0.6, 0.3, seed=2)
        if want_grad:
            y.backward()
            return float(rho.grad)
        return float(y)

    g_auto = yield_at(mat.density, want_grad=True)
    d = 0.05
    g_fd = (yield_at(mat.density + d, False) - yield_at(mat.density - d, False)) / (2 * d)
    assert g_auto < 0  # denser -> shorter free path -> fewer escape, yield falls
    assert abs(g_auto - g_fd) / (abs(g_fd) + 1e-12) < 0.1  # CRN finite-diff agreement


# --------------------------------------------------------------------------- #
# M-MC.7: native MC is the simulator's default depth source
# --------------------------------------------------------------------------- #
def test_backend_depth_profile_flags():
    from de_ebsd.diffraction.backends import resolve_backend

    assert resolve_backend("bloch").uses_depth_profile is True
    assert resolve_backend("bloch_prism").uses_depth_profile is True
    assert resolve_backend("kinematical").uses_depth_profile is False


def test_simulator_defaults_to_native_mc(si_crystal, microscope):
    """A plain bloch compute() with no Monte Carlo runs the native MC itself."""
    import torch

    from de_ebsd import MasterPatternSimulator

    sim = MasterPatternSimulator(si_crystal, microscope, npx=6, dmin=0.1, backend="bloch")
    assert sim.monte_carlo is None  # nothing supplied; standalone by default
    master = sim.compute(mc_electrons=20_000)
    assert master.metadata["backend"] == "bloch"
    assert torch.all(torch.isfinite(master.data))
    assert master.data.std() > 0  # real dynamical contrast, EMsoft-free


# --------------------------------------------------------------------------- #
# M-MC.4: validation against EMsoft accum_z (needs the reference file)
# --------------------------------------------------------------------------- #
_SI_MC = os.environ.get("DE_EBSD_SI_MC", r"D:\emsoft\EMdata\Si_MC.h5")


@pytest.mark.skipif(not os.path.exists(_SI_MC), reason="EMsoft Si_MC.h5 reference not available")
def test_native_mc_matches_emsoft_accum_z():
    """The native MC reproduces EMsoft's depth statistics (no lateral window).

    Runs with the reference run's own binning and correlates the exit-depth
    profiles. The full per-bin check (every bin > 0.995) is in
    ``scripts/validate_montecarlo.py`` at high electron count; this lighter
    regression guards the yield, the beam-energy bin (the one the single-energy
    Bloch backend uses), and the mean correlation. The two bins at the Ehistmin
    cutoff are boundary-truncated and noisier at this electron count.
    """
    from de_ebsd.sim.montecarlo.depth import DepthDistribution

    ref = DepthDistribution.from_emsoft(_SI_MC)
    crystal = Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))], debye_waller=0.004668
    )
    e_min = float(ref.energies_kev.min())
    e_bin = float(np.diff(ref.energies_kev)[0])
    native = simulate_depth_distribution(
        crystal, float(ref.energies_kev.max()), e_min_kev=e_min, e_bin_kev=e_bin,
        depth_step_nm=ref.depth_step, depth_max_nm=ref.depth_step * (ref.n_depths - 1),
        n_electrons=200_000, tilt_deg=70.0, seed=0,
    )

    def pearson(a, b):
        a = a.astype(np.float64) - a.mean()
        b = b.astype(np.float64) - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    ref_yield = ref.accum_z.sum() / ref.n_total
    nat_yield = native.accum_z.sum() / native.n_total
    assert abs(ref_yield - nat_yield) < 0.02  # total backscatter yield

    corrs = []
    for i_e in range(ref.n_energies):
        r = ref.accum_z[i_e].sum(axis=(1, 2))
        n = native.accum_z[i_e].sum(axis=(1, 2))
        if r.sum() and n.sum():
            corrs.append(pearson(r, n))
    assert corrs[-1] > 0.99  # beam-energy bin (used by the Bloch backend)
    assert float(np.mean(corrs)) > 0.99  # profiles match across the spectrum
