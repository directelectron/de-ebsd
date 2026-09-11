"""End-to-end API: the composition chain, refinement, persistence, projection."""

import numpy as np
import pytest
import torch

import de_ebsd as de


@pytest.fixture(scope="module")
def sim(si_crystal_module=None):
    crystal = de.Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))], debye_waller=0.004668
    )
    return de.MasterPatternSimulator(crystal, de.Microscope(energy=20e3), npx=32, dmin=0.12)


@pytest.fixture(scope="module")
def master(sim):
    return sim.compute()


def test_public_api_surface():
    for name in de.__all__:
        assert getattr(de, name) is not None


def test_symmetry_reduction_matches_full(sim):
    """Fundamental-sector folding must reproduce the fully-sampled master."""
    reduced = sim.compute(symmetry=True)
    full = sim.compute(symmetry=False)
    assert torch.allclose(reduced.data, full.data, atol=1e-5)


def test_energy_resolved_master():
    """A depth-integrating backend yields one Lambert map per MC energy bin."""
    crystal = de.Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))], debye_waller=0.004668
    )
    sim = de.MasterPatternSimulator(
        crystal, de.Microscope(energy=20e3, sample_tilt=70.0), npx=8, dmin=0.1, backend="bloch"
    )
    er = sim.compute(energy_resolved=True, mc_electrons=20_000, dtype=torch.float64)
    assert er.data.shape[0] == er.energies_kev.shape[0] > 1  # (nE, 2, n, n)
    assert er.energies_kev[-1] == pytest.approx(20.0)  # top bin is the beam energy
    assert torch.all(torch.isfinite(er.data))
    assert er.data[-1].std() > 0  # per-energy dynamical contrast
    # the top energy bin reproduces the single-energy compute exactly
    se = sim.compute(mc_electrons=20_000, dtype=torch.float64)
    assert torch.allclose(er.data[-1], se.data[0], atol=1e-9)


def test_stage_objects(sim):
    sf = sim.structure_factors()
    assert sf is sim.structure_factors()  # cached
    refl = sim.reflections()
    assert len(refl) > 20
    classified = sim.reflections(k=(0, 0, 1))
    assert classified.n_strong >= 1
    dm = sim.dynamical_matrix(k=(1, 1, 1))
    assert dm.matrix.shape == (dm.n_beams, dm.n_beams)


def test_master_metadata_carries_the_crystal_object(sim, master):
    """Regression: the indexers derive the phase from metadata['crystal'].

    A simulated master used to record ``repr(crystal)`` here while an
    EMsoft-imported one recorded the ``Crystal``, so ``DictionaryIndexer(master,
    detector)`` failed on exactly the master de-ebsd had just produced.
    """
    from de_ebsd.index import DictionaryIndexer, phase_from_crystal

    assert master.metadata["crystal"] is sim.crystal
    assert phase_from_crystal(master.metadata["crystal"]).space_group.number == 227

    detector = de.EBSDDetector(shape=(12, 16), pc=(0.5, 0.5, 0.6), sample_tilt=70.0)
    indexer = DictionaryIndexer(master, detector, resolution=20.0, device="cpu")
    assert indexer.point_group.name == "m-3m"


def test_phase_from_crystal_rejects_a_stringified_crystal():
    """MasterPattern.save() stringifies metadata; the loaded master must say so."""
    from de_ebsd.index import phase_from_crystal

    with pytest.raises(TypeError, match="loaded from HDF5"):
        phase_from_crystal("Crystal(a=0.5431 nm, space_group=227)")


def test_compute_master_pattern(sim, master):
    n = 2 * sim.npx + 1
    assert master.data.shape == (1, 2, n, n)
    assert master.npx == sim.npx
    assert torch.all(torch.isfinite(master.data))
    assert master.metadata["backend"] == "kinematical"
    # centrosymmetric intensity: south hemisphere is the inverted north
    north = master.data[0, 0]
    south = master.data[0, 1]
    assert torch.allclose(south, torch.flip(north, dims=(0, 1)), atol=1e-5)


def test_master_pattern_sampling_consistency(sim, master):
    """Sampling exactly at grid directions returns the grid values."""
    from de_ebsd.core.lambert import master_pattern_grid

    grid = master_pattern_grid(sim.npx)
    values = master.sample(grid)
    assert torch.allclose(values, master.data[0, 0], atol=1e-5)


def test_projection_shapes(master):
    detector = de.EBSDDetector(shape=(32, 48), pc=(0.5, 0.5, 0.5))
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.3, 0.2]])
    patterns = master.project(detector, q)
    assert patterns.shape == (2, 32, 48)
    assert torch.all(torch.isfinite(patterns))


def test_projection_accepts_orix_rotation(master):
    orix_quaternion = pytest.importorskip("orix.quaternion")
    rotation = orix_quaternion.Rotation.random(3)
    detector = de.EBSDDetector(shape=(16, 16), pc=(0.5, 0.5, 0.5))
    patterns = master.project(detector, rotation)
    assert patterns.shape == (3, 16, 16)


def test_save_load_round_trip(master, tmp_path):
    path = str(tmp_path / "master.h5")
    master.save(path)
    loaded = de.MasterPattern.load(path)
    assert torch.allclose(loaded.data, master.data.cpu())
    assert np.allclose(loaded.energies_kev, master.energies_kev)
    assert loaded.metadata["backend"] == "kinematical"


def test_refine_reuses_structure_factors(sim):
    sf = sim.structure_factors()
    finer_grid = sim.refine(npx=32)
    assert finer_grid.structure_factors() is sf  # npx does not touch U_g
    finer_physics = sim.refine(dmin=0.1)
    assert finer_physics._sf is None  # dmin invalidates the cache
    with pytest.raises(TypeError, match="unknown refine"):
        sim.refine(npix=64)


def test_bloch_backend_runs(sim):
    bloch = sim.refine(backend="bloch", npx=6)
    master = bloch.compute()
    assert master.metadata["backend"] == "bloch"
    assert torch.all(torch.isfinite(master.data))
    assert master.data.std() > 0  # real dynamical contrast
    # centrosymmetric crystal: south hemisphere is the inverted north
    north, south = master.data[0, 0], master.data[0, 1]
    assert torch.allclose(south, torch.flip(north, dims=(0, 1)), atol=1e-4)


def test_stereographic_and_equirectangular_views(master):
    stereo = master.to_stereographic(n=65)
    assert stereo.shape == (65, 65)
    equi = master.to_equirectangular(height=32)
    assert equi.shape == (32, 64)
    assert torch.all(torch.isfinite(equi))
