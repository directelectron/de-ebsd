"""Incident beam direction sampling."""

import numpy as np
import pytest
import torch

from de_ebsd.diffraction.kvectors import reduce_to_fundamental_sector, sample_incident_directions


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


def test_reduce_to_fundamental_sector_cubic():
    d = sample_incident_directions(30, dtype=torch.float64).flat()
    reps, inverse = reduce_to_fundamental_sector(d, 227)  # m-3m
    assert inverse.shape == (d.shape[0],)
    assert reps.shape[0] < d.shape[0] / 10  # cubic folds ~15x
    ones = torch.ones(reps.shape[0], dtype=torch.float64)
    assert torch.allclose(reps.norm(dim=-1), ones, atol=1e-9)
    # reps are already in the fundamental sector: reducing again is idempotent
    reps2, _ = reduce_to_fundamental_sector(reps, 227)
    assert reps2.shape[0] == reps.shape[0]


def test_reduce_to_fundamental_sector_triclinic_barely_folds():
    """C1 (P1) has only the identity, so it folds only coincident grid points."""
    d = sample_incident_directions(12, dtype=torch.float64).flat()
    reps, _ = reduce_to_fundamental_sector(d, 1)
    assert reps.shape[0] > 0.85 * d.shape[0]  # ~no fold (only shared equator points)


def test_reduce_to_fundamental_sector_cache():
    """cache_key memoizes the fold; cached and freshly-computed results agree."""
    from de_ebsd.diffraction.kvectors import _FZ_CACHE

    _FZ_CACHE.pop((22, 227), None)
    d = sample_incident_directions(22, dtype=torch.float64).flat()
    reps1, inv1 = reduce_to_fundamental_sector(d, 227, cache_key=22)
    assert (22, 227) in _FZ_CACHE
    reps2, inv2 = reduce_to_fundamental_sector(d, 227, cache_key=22)  # cache hit
    assert torch.equal(reps1, reps2) and torch.equal(inv1, inv2)
    reps3, inv3 = reduce_to_fundamental_sector(d, 227)  # uncached recompute
    assert torch.equal(inv1, inv3) and torch.allclose(reps1, reps3)


@pytest.mark.parametrize("space_group", [225, 194, 62, 2])
def test_every_representative_is_a_symmetry_image_and_none_is_spare(space_group):
    """The fold keeps one image per direction under the point group (on the
    directions' device): each direction's representative is one of its own
    images, and there are no more representatives than orix's fundamental
    sector keeps."""
    from orix.quaternion.symmetry import get_point_group
    from orix.vector import Vector3d

    from de_ebsd.diffraction.kvectors import _dedup_on_device

    directions = sample_incident_directions(40).flat()
    reps, inverse = reduce_to_fundamental_sector(directions, space_group)
    pg = get_point_group(space_group)
    signs = np.where(np.asarray(pg.improper).reshape(-1), -1.0, 1.0)
    operations = np.asarray(pg.to_matrix()).reshape(-1, 3, 3) * signs[:, None, None]
    images = np.einsum("kij,nj->kni", operations, directions.double().numpy())
    own = reps[inverse].double().numpy()
    assert np.abs(images - own[None]).max(-1).min(0).max() < 1e-6
    sector = Vector3d(directions.numpy()).in_fundamental_sector(pg).data
    orix_reps, _ = _dedup_on_device(torch.as_tensor(sector, dtype=directions.dtype))
    assert len(reps) <= len(orix_reps)
