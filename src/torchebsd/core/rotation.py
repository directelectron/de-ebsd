"""Quaternion rotations, matching orix / kikuchipy conventions.

Quaternions are (a, b, c, d) with the scalar part first (the order of
``orix.quaternion.Rotation.data``), so an orix ``Rotation`` feeds in directly
via ``torch.as_tensor(rotation.data)``. The matrix construction reproduces
kikuchipy's ``rotate_vector`` exactly (validated in ``nn_hrebsd.sim``), so
projections agree with kikuchipy to numerical precision.
"""

from __future__ import annotations

import torch

__all__ = ["quaternion_to_matrix", "rotate_vectors"]


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (..., 4), order (a, b, c, d), to matrices (..., 3, 3).

    The returned R rotates column vectors: ``v_rotated = R @ v``.
    """
    a, b, c, d = q.unbind(-1)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    ab, ac, ad = a * b, a * c, a * d
    bc, bd, cd = b * c, b * d, c * d
    row0 = torch.stack([aa + bb - cc - dd, 2 * (bc - ad), 2 * (ac + bd)], dim=-1)
    row1 = torch.stack([2 * (ad + bc), aa - bb + cc - dd, 2 * (cd - ab)], dim=-1)
    row2 = torch.stack([2 * (bd - ac), 2 * (ab + cd), aa - bb - cc + dd], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotate_vectors(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` (..., P, 3) by quaternions ``q`` (..., 4)."""
    r = quaternion_to_matrix(q)
    return torch.einsum("...ij,...pj->...pi", r, v)
