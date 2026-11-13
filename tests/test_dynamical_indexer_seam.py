"""The seam between Autopilot's dynamical indexer and de_ebsd's live indexer,
on the CPU: each call the workflow makes reaches de_ebsd under the name and
arguments de_ebsd has, and comes back in the shape the workflow reads.

A random master stands in for a simulated one: the seam is what is checked
here, not the indexing, which ``test_ebsd.py`` checks on Kikuchi patterns."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("de_ebsd.index.live")

DATA = Path(__file__).parent / "data"
FRAMES = 4
FRAME_SHAPE = (24, 30)


@pytest.fixture(scope="module")
def indexer():
    import torch

    import de_ebsd
    from de_ebsd.pipeline.orientation.indexing import (
        DynamicalIndexer,
        Geometry,
    )
    from de_ebsd.pipeline.orientation.phases import Phase

    phase = Phase.from_cif(0, str(DATA / "Si.cif"))
    generator = torch.Generator().manual_seed(0)
    master = de_ebsd.MasterPattern(
        torch.rand(1, 2, 31, 31, generator=generator), np.array([20.0]), {}
    )
    return DynamicalIndexer([phase], [master], Geometry(FRAME_SHAPE), step_deg=15.0, device="cpu")


@pytest.fixture(scope="module")
def frames():
    return np.random.default_rng(0).random((FRAMES, *FRAME_SHAPE), dtype=np.float32)


def test_indexing_returns_a_row_per_frame(indexer, frames):
    found = indexer.index(frames)
    assert found.phase.shape == found.entry.shape == found.score.shape == (FRAMES,)
    assert found.rotation.shape == (FRAMES, 3, 3)
    assert found.strain is None


def test_indexing_with_strain_returns_a_tensor_per_frame(indexer, frames):
    found = indexer.index(frames, strain=True)
    assert found.rotation.shape == (FRAMES, 3, 3)
    assert found.strain.shape == (FRAMES, 3, 3)


def test_the_static_background_restarts_from_frames_blurred(indexer, frames):
    indexer.restart_static(frames, blur=0.1)
    assert indexer.index(frames).score.shape == (FRAMES,)
    indexer.restart_static()


def test_strain_from_a_reference_returns_a_tensor_per_frame(indexer, frames):
    assert indexer.strain_from_reference(frames, frames[0]).shape == (FRAMES, 3, 3)


def test_the_indexer_holds_its_own_lock_by_default(indexer):
    """Its calls share one device workspace: without a lock, two threads
    indexing at once overwrite each other's rows."""
    assert hasattr(indexer.live.lock, "acquire")


def test_a_lock_passed_in_is_the_one_held(frames):
    import threading

    import torch

    import de_ebsd
    from de_ebsd.pipeline.orientation.indexing import DynamicalIndexer, Geometry
    from de_ebsd.pipeline.orientation.phases import Phase

    phase = Phase.from_cif(0, str(DATA / "Si.cif"))
    master = de_ebsd.MasterPattern(
        torch.rand(1, 2, 31, 31, generator=torch.Generator().manual_seed(0)), np.array([20.0]), {}
    )
    lock = threading.RLock()
    indexer = DynamicalIndexer(
        [phase], [master], Geometry(FRAME_SHAPE), step_deg=15.0, device="cpu", lock=lock
    )
    assert indexer.live.lock is lock


def test_two_threads_indexing_at_once_keep_every_score_a_correlation():
    """On CUDA the live indexer's calls share one device workspace: unlocked,
    an observe and an index from two threads scored up to 5e7 here."""
    import threading

    import torch

    import de_ebsd
    from de_ebsd.pipeline.orientation.indexing import DynamicalIndexer, Geometry
    from de_ebsd.pipeline.orientation.phases import Phase

    if not torch.cuda.is_available():
        pytest.skip("the shared workspace this guards is the CUDA one")
    shape = (60, 80)
    phase = Phase.from_cif(0, str(DATA / "Si.cif"))
    master = de_ebsd.MasterPattern(
        torch.rand(1, 2, 63, 63, generator=torch.Generator().manual_seed(0)), np.array([20.0]), {}
    )
    indexer = DynamicalIndexer([phase], [master], Geometry(shape), step_deg=8.0, device="cuda")
    frames = np.random.default_rng(0).random((64, *shape), dtype=np.float32)
    scores, errors = [], []

    def work(observe):
        try:
            for _ in range(40):
                if observe:
                    indexer.observe(frames)
                scores.append(indexer.index(frames).score)
        except Exception as error:  # surfaced below, not lost in the thread
            errors.append(error)

    threads = [threading.Thread(target=work, args=(observe,)) for observe in (True, False)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    found = np.concatenate(scores)
    assert np.isfinite(found).all() and (np.abs(found) <= 1.0 + 1e-5).all()


def test_the_refine_reads_the_patterns_at_the_side_asked_for():
    """refine_side sets the refine's and the fits' resolution, on the frame's
    long side and in its aspect; the search's is its own."""
    import torch

    import de_ebsd
    from de_ebsd.pipeline.orientation.indexing import DynamicalIndexer, Geometry
    from de_ebsd.pipeline.orientation.phases import Phase

    phase = Phase.from_cif(0, str(DATA / "Si.cif"))
    master = de_ebsd.MasterPattern(
        torch.rand(1, 2, 31, 31, generator=torch.Generator().manual_seed(0)), np.array([20.0]), {}
    )
    shapes = {}
    for side in (20, 40):
        built = DynamicalIndexer(
            [phase], [master], Geometry((240, 300)), step_deg=15.0, device="cpu", refine_side=side
        )
        shapes[side] = (built.live.refine_shape, built.live.search_shape)
    assert shapes[20][0] == (16, 20) and shapes[40][0] == (32, 40)
    assert shapes[20][1] == shapes[40][1]
