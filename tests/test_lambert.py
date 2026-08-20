"""Square Lambert projection: round trips, anchors, grid conventions."""

import math

import pytest
import torch

from torchebsd.lambert import (
    SQRT_PI_HALF,
    master_pattern_grid,
    sphere_to_square,
    square_to_sphere,
)


def test_analytic_anchor_points():
    v = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    v = v / v.norm(dim=-1, keepdim=True)
    xy = sphere_to_square(v)
    length = SQRT_PI_HALF
    assert torch.allclose(xy[0], torch.tensor([0.0, 0.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(xy[1], torch.tensor([length, 0.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(xy[2], torch.tensor([0.0, length], dtype=torch.float64), atol=1e-12)
    # the 111 direction lies on the square diagonal
    assert xy[3, 0] == pytest.approx(xy[3, 1].item(), abs=1e-12)
    expected = math.sqrt(2.0 * (1.0 - 1.0 / math.sqrt(3.0))) * math.sqrt(math.pi) / 2.0
    assert xy[3, 0].item() == pytest.approx(expected, abs=1e-12)


def test_round_trip_sphere_to_square_to_sphere():
    torch.manual_seed(0)
    v = torch.randn(4096, 3, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)
    xy = sphere_to_square(v)
    back = square_to_sphere(xy)  # northern-hemisphere reconstruction
    back[:, 2] = back[:, 2] * torch.sign(v[:, 2])
    assert torch.allclose(back, v, atol=1e-9)


def test_round_trip_square_to_sphere_to_square():
    torch.manual_seed(1)
    xy = (torch.rand(4096, 2, dtype=torch.float64) * 2.0 - 1.0) * SQRT_PI_HALF
    v = square_to_sphere(xy)
    assert torch.allclose(v.norm(dim=-1), torch.ones(len(v), dtype=torch.float64), atol=1e-12)
    assert torch.all(v[:, 2] >= -1e-12)  # northern hemisphere
    back = sphere_to_square(v)
    assert torch.allclose(back, xy, atol=1e-9)


def test_equal_area_property():
    """The Lambert map preserves area: z is uniform for uniform square points."""
    n = 401
    grid = master_pattern_grid((n - 1) // 2, dtype=torch.float64)
    z = grid[..., 2]
    # mean of z over the equal-area grid approximates the hemisphere mean 1/2
    assert z.mean().item() == pytest.approx(0.5, abs=5e-3)


def test_master_pattern_grid_conventions():
    npx = 8
    grid = master_pattern_grid(npx, dtype=torch.float64)
    n = 2 * npx + 1
    assert grid.shape == (n, n, 3)
    assert torch.allclose(grid.norm(dim=-1), torch.ones(n, n, dtype=torch.float64), atol=1e-12)
    # center pixel is the north pole
    assert torch.allclose(grid[npx, npx], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    # column axis is Lambert X: pixel (npx, 2 npx) is the +x equator point
    assert torch.allclose(grid[npx, -1], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                          atol=1e-9)
    # row axis is Lambert Y: pixel (2 npx, npx) is the +y equator point
    assert torch.allclose(grid[-1, npx], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
                          atol=1e-9)
    # equator on all edges
    assert torch.allclose(grid[0, :, 2], torch.zeros(n, dtype=torch.float64), atol=1e-9)


def test_matches_validated_reference_values():
    """Anchor against nn_hrebsd/kikuchipy's _vector2lambert conventions."""
    v = torch.tensor([[0.3, 0.4, 0.866025], [-0.5, 0.1, -0.860233]], dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)
    xy = sphere_to_square(v)
    # values computed with the validated nn_hrebsd implementation
    x, y, z = v[0]
    sq = math.sqrt(2 * (1 - abs(z.item())))
    assert xy[0, 1].item() == pytest.approx(sq * math.sqrt(math.pi) / 2, abs=1e-12)
    assert xy[0, 0].item() == pytest.approx(
        sq * 2 / math.sqrt(math.pi) * math.atan(x.item() / y.item()), abs=1e-12
    )


def test_grid_sample_compatibility():
    """Sampling a linear-gradient master with grid coords reproduces the ramp."""
    import torch.nn.functional as F

    npx = 50
    n = 2 * npx + 1
    ramp = torch.linspace(-1, 1, n).expand(n, n)  # varies along columns (X)
    grid_v = master_pattern_grid(npx, dtype=torch.float64)
    xy = sphere_to_square(grid_v.reshape(-1, 3)) / SQRT_PI_HALF
    sampled = F.grid_sample(
        ramp[None, None].double(),
        xy.view(1, 1, -1, 2),
        mode="bilinear",
        align_corners=True,
    ).view(n, n)
    assert torch.allclose(sampled, ramp.double(), atol=1e-6)
