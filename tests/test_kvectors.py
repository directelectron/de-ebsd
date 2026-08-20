"""Incident beam direction sampling."""

import torch

from torchebsd.kvectors import sample_incident_directions


def test_grid_shapes_and_counts():
    grid = sample_incident_directions(16)
    n = 2 * 16 + 1
    assert grid.directions.shape == (2, n, n, 3)
    assert grid.n_directions == 2 * n * n
    assert grid.flat().shape == (2 * n * n, 3)


def test_hemispheres_and_norms():
    grid = sample_incident_directions(10, dtype=torch.float64)
    norms = grid.directions.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)
    assert torch.all(grid.directions[0, ..., 2] >= -1e-12)  # north
    assert torch.all(grid.directions[1, ..., 2] <= 1e-12)  # south
    # north and south differ only in the z sign
    assert torch.allclose(grid.directions[0, ..., :2], grid.directions[1, ..., :2])
