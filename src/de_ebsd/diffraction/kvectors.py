"""Incident beam directions on the master pattern grid.

For the master pattern every pixel of the square Lambert grid is one incident
beam direction (EMsoft ``Calckvectors`` with the Rosca-Lambert map mode). Both
hemispheres are sampled; :func:`reduce_to_fundamental_sector` then folds the
directions onto the crystal point group's fundamental sector so a backend
computes each symmetry-distinct direction only once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from de_ebsd.core.lambert import master_pattern_grid

__all__ = ["KVectorGrid", "sample_incident_directions", "reduce_to_fundamental_sector"]

# (cache_key, space_group) -> (reps_cpu, inverse_cpu). The fold is deterministic
# per grid + point group, so it is computed once and reused across energy bins,
# refine() calls, and reruns.
_FZ_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _dedup_on_device(reduced: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.round(reduced.double() * 1e5).to(torch.int64) + 100_001  # >0, < 200_003
    base = 200_003
    packed = (q[:, 0] * base + q[:, 1]) * base + q[:, 2]
    _, inverse = torch.unique(packed, return_inverse=True)
    n = int(inverse.max()) + 1 if inverse.numel() else 0
    order = torch.arange(reduced.shape[0], device=reduced.device)
    first = torch.full((n,), reduced.shape[0], dtype=torch.long, device=reduced.device)
    first = first.scatter_reduce(0, inverse, order, reduce="amin", include_self=True)
    return reduced[first], inverse


#: Directions folded at a time by :func:`_canonical_images`, bounding its
#: (operations, directions) images: 48 x 262,144 x 3 doubles is 300 MB.
FOLD_CHUNK = 262_144


def _canonical_images(directions: torch.Tensor, space_group: int) -> torch.Tensor:
    """Each direction's image under the point group with the largest (z, y, x),
    rounded as :func:`_dedup_on_device` rounds, on the directions' device:
    symmetry-equivalent directions share their set of images, so they pick the
    same one. In place of orix's ``in_fundamental_sector`` on the CPU, 3.4 s for
    a 150 npx grid (6 s at 512); any one image per class is as good a
    representative, the intensity being the same at all of them."""
    from orix.quaternion.symmetry import get_point_group

    pg = get_point_group(int(space_group))
    matrices = np.asarray(pg.to_matrix(), dtype=np.float64).reshape(-1, 3, 3)
    signs = np.where(np.asarray(pg.improper).reshape(-1), -1.0, 1.0)
    operations = torch.as_tensor(matrices * signs[:, None, None], device=directions.device)
    out = torch.empty(directions.shape, device=directions.device, dtype=torch.float64)
    base = 200_003
    for lo in range(0, len(directions), FOLD_CHUNK):
        d = directions[lo : lo + FOLD_CHUNK].double()
        images = torch.einsum("kij,nj->kni", operations, d)  # (operations, n, 3)
        q = torch.round(images * 1e5).to(torch.int64) + 100_001
        packed = (q[..., 2] * base + q[..., 1]) * base + q[..., 0]  # z first
        best = packed.argmax(0)
        out[lo : lo + FOLD_CHUNK] = images[best, torch.arange(len(d), device=d.device)]
    return out


def reduce_to_fundamental_sector(
    directions: torch.Tensor, space_group: int, cache_key=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold directions onto the point-group fundamental sector.

    The master pattern is invariant under the crystal point group -- ``I(k) =
    I(g k)`` -- so directions related by symmetry share one intensity. Reduces
    every direction to one canonical image under the point group
    (:func:`_canonical_images`, on the directions' device) and deduplicates
    on-device: symmetry-equivalent grid points map to the same representative,
    so the fold is exact (no interpolation). Cubic grids collapse ~15x;
    low-symmetry groups less.

    ``cache_key`` (e.g. the grid ``npx``) memoizes the result -- the fold depends
    only on the grid and the point group, so passing it makes repeated calls
    (energy bins, ``refine``) free.

    Returns ``(reps, inverse)`` -- the unique representative directions to
    evaluate, and an index mapping each input direction to its representative,
    so ``full_values = backend(reps)[inverse]``.
    """
    dev, dt = directions.device, directions.dtype
    ck = (cache_key, int(space_group)) if cache_key is not None else None
    if ck is not None and ck in _FZ_CACHE:
        reps_cpu, inverse_cpu = _FZ_CACHE[ck]
        return (
            torch.as_tensor(reps_cpu, device=dev, dtype=dt),
            torch.as_tensor(inverse_cpu, device=dev, dtype=torch.long),
        )

    reduced = _canonical_images(directions, int(space_group))
    reps, inverse = _dedup_on_device(reduced.to(dt))
    if ck is not None:
        _FZ_CACHE[ck] = (reps.detach().cpu().numpy(), inverse.detach().cpu().numpy())
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

    def show(self, title: str = "Incident-direction sampling"):
        """Display the sampled incident directions on an interactive sphere."""
        from de_ebsd.viz.patterns import sampling_sphere_figure

        return sampling_sphere_figure(self.flat().detach().cpu().numpy(), title=title)


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
