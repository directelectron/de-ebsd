"""Incident beam directions on the master pattern grid.

For the master pattern every pixel of the square Lambert grid is one incident
beam direction (EMsoft ``Calckvectors`` with the Rosca-Lambert map mode). Both
hemispheres are sampled; :func:`reduce_to_fundamental_sector` then folds the
directions onto the crystal point group's fundamental sector so a backend
computes each symmetry-distinct direction only once (DESIGN.md, roadmap M3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from torchebsd.lambert import master_pattern_grid

__all__ = ["KVectorGrid", "sample_incident_directions", "reduce_to_fundamental_sector"]


def reduce_to_fundamental_sector(
    directions: torch.Tensor, space_group: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold directions onto the point-group fundamental sector.

    The master pattern is invariant under the crystal point group -- ``I(k) =
    I(g k)`` -- so directions related by symmetry share one intensity. Reduces
    every direction to its canonical fundamental-sector representative (orix
    ``in_fundamental_sector``) and deduplicates: symmetry-equivalent grid points
    map to a bit-identical representative, so the fold is exact (no
    interpolation). Cubic grids collapse ~15x; low-symmetry groups less.

    Returns ``(reps, inverse)`` -- the unique representative directions to
    evaluate, and an index mapping each input direction to its representative,
    so ``full_values = backend(reps)[inverse]``.
    """
    from orix.quaternion.symmetry import get_point_group
    from orix.vector import Vector3d

    dev, dt = directions.device, directions.dtype
    pg = get_point_group(int(space_group))
    reduced = Vector3d(directions.detach().cpu().numpy()).in_fundamental_sector(pg).data
    key = np.round(reduced, 6)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    reps = torch.as_tensor(reduced[first], device=dev, dtype=dt)
    inverse = torch.as_tensor(np.asarray(inverse).ravel(), device=dev, dtype=torch.long)
    return reps, inverse


@dataclass
class KVectorGrid:
    """Incident-beam directions covering the sphere.

    Attributes
    ----------
    npx
        Half-width of the square Lambert grid; each hemisphere map has shape
        ``(2 npx + 1, 2 npx + 1)``.
    directions
        Unit vectors, shape ``(2, 2 npx + 1, 2 npx + 1, 3)``; index 0 is the
        northern hemisphere (z >= 0), index 1 the southern.
    """

    npx: int
    directions: torch.Tensor

    @property
    def n_directions(self) -> int:
        return 2 * (2 * self.npx + 1) ** 2

    def flat(self) -> torch.Tensor:
        """All directions as a flat (K, 3) tensor (north first, row-major)."""
        return self.directions.reshape(-1, 3)


def sample_incident_directions(
    npx: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> KVectorGrid:
    """Both hemispheres of the square Lambert grid as unit direction vectors."""
    north = master_pattern_grid(npx, device=device, dtype=dtype)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    return KVectorGrid(npx=npx, directions=torch.stack([north, south], dim=0))
