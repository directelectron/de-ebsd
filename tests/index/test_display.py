"""Indexing visualization (I7) smoke tests -- anyplotlib figures render headless.

Each test builds small synthetic inputs (the same great-circle *band* master and
tilted detector as ``test_indexer``/``test_io``, plus a tiny ``(2, 3)`` scan),
calls one of the four ``show_*`` figures, and proves it is a valid figure by
serializing it to HTML (``fig.to_html()``) -- exactly how the simulation-side viz
tests (``tests/test_display_widgets.py``) validate a figure. No display backend or
browser is needed. ``anyplotlib`` is skipped gracefully when absent.
"""

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation

from torchebsd import EBSDDetector, MasterPattern
from torchebsd.index import (
    show_correlation,
    show_map,
    show_match,
    show_refinement,
    to_crystal_map,
)
from torchebsd.index.indexer import SphericalIndexer
from torchebsd.sim import Crystal
from torchebsd.sim.lambert import master_pattern_grid

BANDWIDTH = 16
_SCAN_SHAPE = (2, 3)
_SCAN_EULER_DEG = np.array(
    [
        [10.0, 20.0, 30.0],
        [45.0, 55.0, 15.0],
        [123.0, 47.0, 205.0],
        [200.0, 100.0, 300.0],
        [300.0, 80.0, 40.0],
        [15.0, 35.0, 95.0],
    ]
)


def _band_field(v, poles, amps, width):
    d = v.to(torch.float64) @ poles.T
    return (amps * torch.exp(-(d / width) ** 2)).sum(-1)


@pytest.fixture(scope="module")
def crystal():
    return Crystal.from_parameters(
        a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))]
    )


@pytest.fixture(scope="module")
def master(crystal):
    g = torch.Generator().manual_seed(7)
    poles = torch.randn(12, 3, generator=g, dtype=torch.float64)
    poles = poles / poles.norm(dim=-1, keepdim=True)
    amps = 0.5 + torch.rand(12, generator=g, dtype=torch.float64)
    north = master_pattern_grid(120, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    data = torch.stack(
        [_band_field(north, poles, amps, 0.11), _band_field(south, poles, amps, 0.11)], dim=0
    )[None]
    return MasterPattern(data=data, energies_kev=np.array([20.0]), metadata={"crystal": crystal})


@pytest.fixture(scope="module")
def detector():
    return EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.20), sample_tilt=70.0, tilt=5.0)


@pytest.fixture(scope="module")
def true_rotations():
    return Rotation.from_euler(np.deg2rad(_SCAN_EULER_DEG))


@pytest.fixture(scope="module")
def scan_patterns(master, detector, true_rotations):
    return master.project(detector, true_rotations, rescale=True)  # (6, H, W)


@pytest.fixture(scope="module")
def indexer(master, detector):
    return SphericalIndexer(master, detector, bandwidth=BANDWIDTH, device="cpu")


@pytest.fixture(scope="module")
def crystal_map(indexer, scan_patterns):
    return indexer.index_scan(scan_patterns, shape=_SCAN_SHAPE)


def test_show_match(master, detector, scan_patterns, true_rotations):
    pytest.importorskip("anyplotlib")
    fig = show_match(scan_patterns[0], master, detector, true_rotations[0:1])
    html = fig.to_html()
    assert isinstance(html, str) and len(html) > 100
    # exact orientation -> the simulated pattern matches the (rescaled) experimental well
    assert fig.match_ncc > 0.9


def test_show_match_accepts_numpy(master, detector, scan_patterns, true_rotations):
    pytest.importorskip("anyplotlib")
    exp = scan_patterns[0].cpu().numpy()
    assert show_match(exp, master, detector, true_rotations[0:1]).to_html()


def test_show_map_ipf(crystal_map):
    pytest.importorskip("anyplotlib")
    assert show_map(crystal_map, value="ipf").to_html()


def test_show_map_quality(crystal_map):
    pytest.importorskip("anyplotlib")
    assert show_map(crystal_map, value="quality").to_html()


def test_show_map_both(crystal_map):
    pytest.importorskip("anyplotlib")
    assert show_map(crystal_map, value="both").to_html()


def test_show_map_bad_value(crystal_map):
    pytest.importorskip("anyplotlib")
    with pytest.raises(ValueError, match="value must be"):
        show_map(crystal_map, value="nope")


def test_show_map_from_to_crystal_map(crystal):
    """A CrystalMap built directly by to_crystal_map renders too (row-major layout)."""
    pytest.importorskip("anyplotlib")
    rot = Rotation.from_euler(np.deg2rad(_SCAN_EULER_DEG))
    quality = np.arange(rot.size, dtype=np.float64)
    xmap = to_crystal_map(rot, quality, _SCAN_SHAPE, crystal)
    assert show_map(xmap, value="both").to_html()


def test_show_correlation_peak_and_render():
    pytest.importorskip("anyplotlib")
    rng = np.random.default_rng(0)
    corr = rng.random((8, 8, 8))
    corr[3, 5, 2] = 10.0  # a distinct planted peak
    fig = show_correlation(corr)
    assert fig.peak_index == np.unravel_index(np.argmax(corr), corr.shape)
    assert fig.peak_index == (3, 5, 2)
    assert fig.peak_value == pytest.approx(10.0)
    assert fig.to_html()


def test_show_correlation_accepts_stack_and_torch():
    pytest.importorskip("anyplotlib")
    corr = torch.zeros((2, 6, 6, 6), dtype=torch.float64)
    corr[0, 1, 2, 3] = 5.0  # peak of the FIRST pattern (the one shown)
    corr[1, 5, 5, 5] = 9.0
    fig = show_correlation(corr)
    assert fig.peak_index == (1, 2, 3)
    assert fig.to_html()


def test_show_refinement_real_history(indexer, scan_patterns, true_rotations):
    pytest.importorskip("anyplotlib")
    _, _, history = indexer.refine(
        scan_patterns[:3], rotations=true_rotations[:3], n_iters=5, return_history=True
    )
    assert history.shape == (5, 3)
    assert show_refinement(history).to_html()


def test_show_refinement_single_pattern():
    pytest.importorskip("anyplotlib")
    history = np.linspace(0.6, 0.99, 12)  # 1-D (n_iters,) trajectory
    assert show_refinement(history).to_html()
