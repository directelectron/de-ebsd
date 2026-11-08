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
