"""Incident beam directions on the master pattern grid.

For the master pattern every pixel of the square Lambert grid is one incident
beam direction (EMsoft ``Calckvectors`` with the Rosca-Lambert map mode).
This first version samples both full hemispheres; reducing the sampling to
the point-group fundamental sector and filling the rest by symmetry is the
planned optimization (DESIGN.md, roadmap M3) and belongs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchebsd.lambert import master_pattern_grid

__all__ = ["KVectorGrid", "sample_incident_directions"]


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
