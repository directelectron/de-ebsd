"""Equal-area Lambert projection between the unit sphere and the square grid.

The master pattern is stored on the square; ``sphere_to_square`` maps a
direction to its (x, y) in [-L, L]^2 with L = sqrt(pi/2), and
``square_to_sphere`` inverts it. Both are differentiable and batched, and follow
kikuchipy's ``_vector2lambert`` and EMsoft's Rosca-Lambert mapping.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SQRT_PI = math.sqrt(math.pi)
SQRT_PI_HALF = math.sqrt(math.pi / 2.0)  # the half-edge L of the square
_SQRT_PI_OVER_2 = SQRT_PI / 2.0
_TWO_OVER_SQRT_PI = 2.0 / SQRT_PI

__all__ = [
    "SQRT_PI_HALF",
    "master_pattern_grid",
    "sample_texture",
    "sphere_to_square",
    "square_to_sphere",
]


def sphere_to_square(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Unit vectors (..., 3) to square Lambert coordinates (..., 2), book units.

    Matches kikuchipy ``_vector2lambert`` (validated against EMsoft). Both
    hemispheres map onto the same square through ``|z|``.
    """
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(eps)
    x, y, z = v.unbind(-1)
    abs_z = z.abs()
    # clamped above zero: sqrt's slope at 0 is infinite, and at a pole (where |z|
    # rounds to 1, within ~0.02 deg in float32) it made the gradient NaN even
    # though the pole branch below discards the value
    sqrt_z = torch.sqrt((2.0 * (1.0 - abs_z)).clamp_min(torch.finfo(v.dtype).tiny))

    cond = y.abs() <= x.abs()

    x_safe = torch.where(x.abs() < eps, torch.full_like(x, eps), x)
    y_safe = torch.where(y.abs() < eps, torch.full_like(y, eps), y)
    sign_x = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    sign_y = torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))

    # branch |y| <= |x|
    x_a = sign_x * sqrt_z * _SQRT_PI_OVER_2
    y_a = sign_x * sqrt_z * _TWO_OVER_SQRT_PI * torch.atan(y / x_safe)
    # branch |y| > |x|
    x_b = sign_y * sqrt_z * _TWO_OVER_SQRT_PI * torch.atan(x / y_safe)
    y_b = sign_y * sqrt_z * _SQRT_PI_OVER_2

    lx = torch.where(cond, x_a, x_b)
    ly = torch.where(cond, y_a, y_b)

    at_pole = abs_z >= 1.0
    lx = torch.where(at_pole, torch.zeros_like(lx), lx)
    ly = torch.where(at_pole, torch.zeros_like(ly), ly)
    return torch.stack([lx, ly], dim=-1)


def square_to_sphere(xy: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Square Lambert coordinates (..., 2, book units) to unit vectors (..., 3).

    Inverse of :func:`sphere_to_square` for the northern hemisphere
    (z >= 0); negate z for the southern hemisphere. Port of EMsoft
    ``Lambert2DSquareForwardDouble`` (Rosca 2010, eqs. 8-9).
    """
    x, y = xy.unbind(-1)
    cond = x.abs() <= y.abs()

    # branch |x| <= |y|
    y_safe = torch.where(y.abs() < eps, torch.full_like(y, eps), y)
    q_a = 2.0 * y / math.pi * torch.sqrt((math.pi - y * y).clamp_min(0.0))
    qq_a = x * math.pi * 0.25 / y_safe
    vx_a = q_a * torch.sin(qq_a)
    vy_a = q_a * torch.cos(qq_a)
    vz_a = 1.0 - 2.0 * y * y / math.pi

    # branch |x| > |y|
    x_safe = torch.where(x.abs() < eps, torch.full_like(x, eps), x)
    q_b = 2.0 * x / math.pi * torch.sqrt((math.pi - x * x).clamp_min(0.0))
    qq_b = y * math.pi * 0.25 / x_safe
    vx_b = q_b * torch.cos(qq_b)
    vy_b = q_b * torch.sin(qq_b)
    vz_b = 1.0 - 2.0 * x * x / math.pi

    vx = torch.where(cond, vx_a, vx_b)
    vy = torch.where(cond, vy_a, vy_b)
    vz = torch.where(cond, vz_a, vz_b)

    at_origin = (x.abs() < eps) & (y.abs() < eps)
    vx = torch.where(at_origin, torch.zeros_like(vx), vx)
    vy = torch.where(at_origin, torch.zeros_like(vy), vy)
    vz = torch.where(at_origin, torch.ones_like(vz), vz)

    v = torch.stack([vx, vy, vz], dim=-1)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def master_pattern_grid(
    npx: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Northern-hemisphere unit vectors for every master pattern pixel.

    Returns a tensor of shape ``(2 npx + 1, 2 npx + 1, 3)`` where element
    ``[row, col]`` is the direction whose square Lambert projection is
    ``X = (col - npx)/npx * L`` and ``Y = (row - npx)/npx * L``. Negate the
    z component for the southern hemisphere.
    """
    coords = torch.linspace(-1.0, 1.0, 2 * npx + 1, device=device, dtype=dtype)
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    xy = torch.stack([xs, ys], dim=-1) * SQRT_PI_HALF
    return square_to_sphere(xy)


def sample_texture(
    texture: torch.Tensor, directions: torch.Tensor, phase: torch.Tensor | None = None
) -> torch.Tensor:
    """Bilinear lookup of master patterns at unit *directions* ``(..., 3)``.

    *texture* ``(1, 2 * phases, n, n)`` holds each phase's northern and southern
    hemisphere as consecutive channels; *phase* (integer, broadcast against the
    leading dimensions of *directions*) picks the phase, default the first. Edge
    handling is ``grid_sample(align_corners=True, padding_mode="border")``,
    identical to kikuchipy's interpolation. The hemisphere is selected by the
    sign of ``z``. Returns the values ``(...)``.
    """
    lead = directions.shape[:-1]
    square = sphere_to_square(directions.reshape(-1, 3)) / SQRT_PI_HALF
    values = F.grid_sample(
        texture,
        square.view(1, 1, -1, 2),
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    ).view(texture.shape[1], *lead)
    channel = (directions[..., 2] < 0).long()
    if phase is not None:
        channel = channel + 2 * phase
    return values.gather(0, channel[None])[0]
