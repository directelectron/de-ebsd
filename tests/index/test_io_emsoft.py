"""EMsoft master-pattern import (I8): ``torchebsd.index.io.master_from_emsoft``.

Covers the real-data loop close: an EMsoft EBSD master-pattern HDF5 on disk loads
into a torchebsd :class:`~torchebsd.sim.MasterPattern` (with the crystal rebuilt
into ``metadata["crystal"]``) and flows straight into
:meth:`SphericalIndexer.index_scan` with the phase auto-derived.

The portable tests write a **synthetic** EMsoft-format HDF5 to ``tmp_path`` with
exactly the groups ``master_from_emsoft`` reads and tiny arrays (2 sites, 2
energies, ``n = 9``, a 1-atom Si-like cell, space group 227, origin setting 2).
Per-site values are planted distinct so the mandated **sum over the site axis**
is verifiable (not just an ``[0]`` index). A ``skipif``-guarded test loads the
maintainer's real ``Si_MC.h5`` when present, and is skipped cleanly otherwise.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest
import torch
from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation

from torchebsd import EBSDDetector
from torchebsd.index import master_from_emsoft
from torchebsd.sim import Crystal

# real-file test target (maintainer's local copy; absent on most machines)
_REAL_SI_MC = os.environ.get("TORCHEBSD_SI_MC", r"D:\emsoft\EMdata\Si_MC.h5")

N_SITES = 2
N_ENERGY = 2
N_SIDE = 9  # square-Lambert side length (data.shape[-1])
EKEVS = np.array([15.0, 20.0])
SPACE_GROUP = 227  # Fd-3m (Si)
SETTING = 2
LATTICE = np.array([0.5431, 0.5431, 0.5431, 90.0, 90.0, 90.0])  # nm, degrees
SI_Z = 14  # atomic number -> ELEMENT_SYMBOLS[13] == "Si"
SI_OCC = 1.0
SI_DW = 0.0045  # Debye-Waller B (nm^2)
DMIN = 0.05
BETHE = np.array([4.0, 8.0, 50.0, 1.0])  # c1, c2, c3, sgdbdiff


def _write_synthetic_emsoft(path: str, seed: int = 0):
    """Write a tiny EMsoft-format master-pattern HDF5; return the planted masters.

    Returns ``(mlpnh, mlpsh)`` each shaped ``(N_SITES, N_ENERGY, N_SIDE, N_SIDE)``
    with distinct per-site values, so ``sum(axis=0)`` is distinguishable from
    ``[0]`` in the assertions.
    """
    rng = np.random.default_rng(seed)
    mlpnh = rng.random((N_SITES, N_ENERGY, N_SIDE, N_SIDE)).astype(np.float32)
    mlpsh = rng.random((N_SITES, N_ENERGY, N_SIDE, N_SIDE)).astype(np.float32)
    # make the site components unmistakably different (so summing != indexing [0])
    mlpnh[1] += 10.0
    mlpsh[1] += 20.0

    atom_data = np.zeros((5, 1), dtype=np.float64)
    atom_data[:, 0] = [0.125, 0.125, 0.125, SI_OCC, SI_DW]

    with h5py.File(path, "w") as f:
        f.create_dataset("EMData/EBSDmaster/mLPNH", data=mlpnh)
        f.create_dataset("EMData/EBSDmaster/mLPSH", data=mlpsh)
        f.create_dataset("EMData/EBSDmaster/EkeVs", data=EKEVS)
        f.create_dataset("EMData/EBSDmaster/BetheParameters", data=BETHE)
        # dmin is stored as a length-1 array in real EMsoft files
        f.create_dataset("NMLparameters/EBSDMasterNameList/dmin", data=np.array([DMIN]))
        f.create_dataset("CrystalData/LatticeParameters", data=LATTICE)
        f.create_dataset("CrystalData/AtomData", data=atom_data)
        f.create_dataset("CrystalData/Atomtypes", data=np.array([SI_Z], dtype=np.int32))
        f.create_dataset("CrystalData/SpaceGroupNumber", data=SPACE_GROUP)
        f.create_dataset("CrystalData/SpaceGroupSetting", data=SETTING)
    return mlpnh, mlpsh


@pytest.fixture()
def synthetic_file(tmp_path):
    """A synthetic EMsoft HDF5 on disk plus the planted north/south masters."""
    path = str(tmp_path / "synthetic_master.h5")
    mlpnh, mlpsh = _write_synthetic_emsoft(path)
    return path, mlpnh, mlpsh


# --------------------------------------------------------------------------- #
# master read: shape, energies, and the site-sum
# --------------------------------------------------------------------------- #
def test_master_shape_energies_and_site_sum(synthetic_file):
    """Data is ``(n_energy, 2, n, n)``, energies match, and hemispheres are site-summed."""
    path, mlpnh, mlpsh = synthetic_file
    m = master_from_emsoft(path)

    assert m.data.shape == (N_ENERGY, 2, N_SIDE, N_SIDE)
    assert m.data.dtype == torch.float32
    assert np.allclose(m.energies_kev, EKEVS)

    north = m.data[:, 0].numpy()
    south = m.data[:, 1].numpy()
    # the mandated sum over the site axis (axis 0 of the raw masters)
    assert np.allclose(north, mlpnh.sum(axis=0))
    assert np.allclose(south, mlpsh.sum(axis=0))
    # prove the sum actually happened -- not just first-site indexing
    assert not np.allclose(north, mlpnh[0])
    assert not np.allclose(south, mlpsh[0])


def test_master_metadata_provenance(synthetic_file):
    """dmin, Bethe parameters, and the source path land in metadata."""
    path, _, _ = synthetic_file
    m = master_from_emsoft(path)

    assert m.metadata["source"] == path
    assert m.metadata["dmin"] == pytest.approx(DMIN)
    assert m.metadata["bethe"] == pytest.approx(tuple(BETHE))


def test_master_collapses_3d_master(tmp_path):
    """An already site-collapsed ``(n_energy, n, n)`` master is used as-is."""
    path = str(tmp_path / "collapsed.h5")
    north = np.random.default_rng(1).random((N_ENERGY, N_SIDE, N_SIDE)).astype(np.float32)
    south = np.random.default_rng(2).random((N_ENERGY, N_SIDE, N_SIDE)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("EMData/EBSDmaster/mLPNH", data=north)
        f.create_dataset("EMData/EBSDmaster/mLPSH", data=south)
        f.create_dataset("EMData/EBSDmaster/EkeVs", data=EKEVS)

    m = master_from_emsoft(path)
    assert m.data.shape == (N_ENERGY, 2, N_SIDE, N_SIDE)
    assert np.allclose(m.data[:, 0].numpy(), north)
    assert np.allclose(m.data[:, 1].numpy(), south)


# --------------------------------------------------------------------------- #
# crystal rebuild (metadata["crystal"])
# --------------------------------------------------------------------------- #
def test_master_rebuilds_crystal(synthetic_file):
    """metadata['crystal'] is a Crystal with the right sg, lattice, origin, and Si site."""
    path, _, _ = synthetic_file
    m = master_from_emsoft(path)

    crystal = m.metadata["crystal"]
    assert isinstance(crystal, Crystal)
    assert crystal.space_group == SPACE_GROUP
    assert crystal.origin == SETTING
    assert crystal.a == pytest.approx(LATTICE[0])
    assert (crystal.b, crystal.c) == pytest.approx((LATTICE[1], LATTICE[2]))
    assert (crystal.alpha, crystal.beta, crystal.gamma) == pytest.approx((90.0, 90.0, 90.0))

    assert len(crystal.sites) == 1
    site = crystal.sites[0]
    assert site.element == "Si"  # proves the Z=14 -> ELEMENT_SYMBOLS[13] mapping
    assert site.occupancy == pytest.approx(SI_OCC)
    assert site.debye_waller == pytest.approx(SI_DW)
    assert site.position == pytest.approx((0.125, 0.125, 0.125))


def test_master_survives_missing_crystaldata(tmp_path):
    """A file with no CrystalData still imports the master, with a clear note."""
    path = str(tmp_path / "no_crystal.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "EMData/EBSDmaster/mLPNH",
            data=np.ones((N_SITES, N_ENERGY, N_SIDE, N_SIDE), np.float32),
        )
        f.create_dataset(
            "EMData/EBSDmaster/mLPSH",
            data=np.ones((N_SITES, N_ENERGY, N_SIDE, N_SIDE), np.float32),
        )
        f.create_dataset("EMData/EBSDmaster/EkeVs", data=EKEVS)

    m = master_from_emsoft(path)
    assert m.data.shape == (N_ENERGY, 2, N_SIDE, N_SIDE)
    assert "crystal" not in m.metadata
    assert "crystal_error" in m.metadata  # a clear, actionable note


# --------------------------------------------------------------------------- #
# closed loop: loaded master -> index_scan with the phase auto-derived
# --------------------------------------------------------------------------- #
def test_loaded_master_feeds_index_scan(synthetic_file):
    """The EMsoft master flows into index_scan; the phase auto-derives from metadata."""
    from torchebsd.index.indexer import SphericalIndexer

    path, _, _ = synthetic_file
    m = master_from_emsoft(path, device="cpu")
    detector = EBSDDetector(shape=(12, 12), pc=(0.5, 0.5, 0.55), sample_tilt=70.0, tilt=0.0)

    # a tiny 2x2 scan: project a few orientations off the loaded master itself
    # (no singleton axis, so orix keeps the 2-D CrystalMap shape).
    r_true = Rotation.from_euler(
        np.deg2rad(
            [[10.0, 20.0, 30.0], [45.0, 55.0, 15.0], [123.0, 47.0, 205.0], [70.0, 25.0, 110.0]]
        )
    )
    patterns = m.project(detector, r_true, rescale=True)  # (4, H, W)

    indexer = SphericalIndexer(m, detector, bandwidth=16, device="cpu")
    # note: no phase= passed -- it must auto-derive from m.metadata["crystal"]
    xmap = indexer.index_scan(patterns, shape=(2, 2))

    assert isinstance(xmap, CrystalMap)
    assert xmap.shape == (2, 2)
    assert xmap.size == 4
    # the phase carries the crystal symmetry (Fd-3m -> point group m-3m)
    assert xmap.phases[0].space_group.number == SPACE_GROUP
    assert xmap.phases[0].point_group.name == "m-3m"
    assert np.isfinite(xmap.prop["quality"]).all()


# --------------------------------------------------------------------------- #
# real EMsoft file (skipped cleanly when absent -- most machines)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.exists(_REAL_SI_MC),
    reason=f"real EMsoft Si_MC.h5 not found at {_REAL_SI_MC} (set TORCHEBSD_SI_MC)",
)
def test_real_si_mc_file():
    """The real EMsoft Si_MC.h5 loads with sane shape, energies, and a Si crystal."""
    m = master_from_emsoft(_REAL_SI_MC)

    assert m.data.ndim == 4
    assert m.data.shape[1] == 2  # two hemispheres
    assert m.data.shape[2] == m.data.shape[3]  # square Lambert
    assert m.data.shape[0] == len(m.energies_kev)
    assert m.energies_kev.max() == pytest.approx(20.0, abs=1.0)

    crystal = m.metadata["crystal"]
    assert isinstance(crystal, Crystal)
    assert crystal.space_group == SPACE_GROUP
    assert any(site.element == "Si" for site in crystal.sites)
    assert m.metadata["dmin"] is not None
