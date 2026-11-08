"""The device-work entry points hold the lock their caller passes, and pick a device."""

import threading

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation
from orix.quaternion.symmetry import Oh

from band_master import make_band_master
from de_ebsd import EBSDDetector
from de_ebsd.core.device import default_device, resolve_dtype, resolve_lock
from de_ebsd.index.dictionary import DictionaryIndexer
from de_ebsd.index.live import LiveIndexer
from de_ebsd.index.refine.orientation import refine_orientation
from de_ebsd.index.refine.pattern_center import determine_pattern_center
from de_ebsd.index.spherical.indexer import SphericalIndexer


class CountingLock:
    """A reentrant lock that counts how often it is entered and checks it is held
    by the thread that entered it."""

    def __init__(self):
        self._lock = threading.RLock()
        self.entered = 0

    def __enter__(self):
        self._lock.__enter__()
        self.entered += 1
        return self

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)

    def held(self) -> bool:
        return self._lock._is_owned()


@pytest.fixture(scope="module")
def master():
    return make_band_master(npx=64, width=0.08, n_bands=8, seed=3)


@pytest.fixture(scope="module")
def detector():
    return EBSDDetector(shape=(24, 24), pattern_center=(0.5, 0.5, 0.5), sample_tilt=70.0)


@pytest.fixture(scope="module")
def patterns(master, detector):
    np.random.seed(2)
    return master.project(detector, Rotation.random(2), rescale=True)


def test_no_lock_is_a_context_that_does_nothing():
    lock = resolve_lock(None)
    with lock, lock:  # reusable and reentrant, as every call needs
        pass
    given = threading.RLock()
    assert resolve_lock(given) is given


def test_the_dictionary_indexer_holds_the_lock(master, detector, patterns):
    lock = CountingLock()
    indexer = DictionaryIndexer(master, detector, Oh, resolution=20.0, device="cpu", lock=lock)
    built = lock.entered
    assert built >= 1
    indexer.index(patterns)
    assert lock.entered > built
    assert not lock.held()


def test_the_live_indexer_holds_the_lock(master, detector, patterns):
    lock = CountingLock()
    indexer = LiveIndexer(
        [master],
        [Oh],
        detector,
        resolution=20.0,
        search_shape=(12, 12),
        refine_shape=(24, 24),
        refine_steps_deg=(2.0,),
        device="cpu",
        lock=lock,
    )
    built = lock.entered
    assert built >= 1
    indexer.index(patterns)
    assert lock.entered > built


def test_the_spherical_indexer_holds_the_lock(master, detector, patterns):
    lock = CountingLock()
    indexer = SphericalIndexer(master, detector, bandwidth=6, device="cpu", lock=lock)
    built = lock.entered
    assert built >= 1
    indexer.index(patterns)
    assert lock.entered > built


def test_the_refinements_hold_the_lock(master, detector, patterns):
    np.random.seed(2)
    seeds = Rotation.random(2)
    lock = CountingLock()
    refine_orientation(patterns, master, detector, seeds, n_iterations=1, lock=lock)
    assert lock.entered == 1
    determine_pattern_center(
        patterns, master, detector, seeds, n_rounds=1, n_tiles=4, blur=None, lock=lock
    )
    assert lock.entered == 2


def test_the_device_can_be_forced(monkeypatch):
    monkeypatch.setenv("DE_EBSD_DEVICE", "cpu")
    assert default_device() == torch.device("cpu")
    monkeypatch.delenv("DE_EBSD_DEVICE")
    assert default_device().type in ("cuda", "mps", "cpu")


def test_mps_runs_double_as_single():
    assert resolve_dtype("mps", torch.float64) is torch.float32
    assert resolve_dtype("mps", "float64") == "float32"
    assert resolve_dtype("mps", torch.float16) is torch.float16
    assert resolve_dtype("cpu", torch.float64) is torch.float64
