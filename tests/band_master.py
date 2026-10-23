"""A synthetic Kikuchi-band master pattern shared by the indexing tests.

Each pole contributes a bright Gaussian band about its great circle. The bands
cover the whole sphere, so the field has a near-uniform local norm (an unnormalized
correlation is then unbiased) yet is asymmetric enough to test the frame convention.
"""

import numpy as np
import torch

from de_ebsd.core.lambert import master_pattern_grid
from de_ebsd.sim.masterpattern import MasterPattern


def band_field(
    directions: torch.Tensor, poles: torch.Tensor, amplitudes: torch.Tensor, width: float
) -> torch.Tensor:
    """The band field at unit *directions* ``(..., 3)``; *width* is in direction cosine."""
    cosines = directions.to(torch.float64) @ poles.T
    return (amplitudes * torch.exp(-((cosines / width) ** 2))).sum(-1)


def make_band_master(
    npx: int = 160, width: float = 0.11, n_bands: int = 12, seed: int = 7, metadata=None
) -> MasterPattern:
    """A float64 master of *n_bands* random great-circle bands on an *npx* Lambert grid."""
    generator = torch.Generator().manual_seed(seed)
    poles = torch.randn(n_bands, 3, generator=generator, dtype=torch.float64)
    poles = poles / poles.norm(dim=-1, keepdim=True)
    amplitudes = 0.5 + torch.rand(n_bands, generator=generator, dtype=torch.float64)
    north = master_pattern_grid(npx, dtype=torch.float64)
    south = north.clone()
    south[..., 2] = -south[..., 2]
    data = torch.stack(
        [band_field(north, poles, amplitudes, width), band_field(south, poles, amplitudes, width)],
        dim=0,
    )[None]  # (1, 2, n, n)
    return MasterPattern(data=data, energies_kev=np.array([20.0]), metadata=metadata or {})
