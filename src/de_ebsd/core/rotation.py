"""Quaternion rotations, matching orix / kikuchipy conventions.

Quaternions are (a, b, c, d) with the scalar part first (the order of
``orix.quaternion.Rotation.data``), so an orix ``Rotation`` feeds in directly
via ``torch.as_tensor(rotation.data)``. The matrix construction reproduces
kikuchipy's ``rotate_vector`` exactly , so
projections agree with kikuchipy to numerical precision.
"""

from __future__ import annotations

import torch

__all__ = ["quaternion_to_matrix", "rotate_vectors", "rotation_vector_to_matrix", "skew_matrix"]

_EPS = 1e-12


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


def skew_matrix(w: torch.Tensor) -> torch.Tensor:
    """Matrices ``(..., 3, 3)`` taking ``v`` to ``w x v``, for rotation vectors ``w``."""
    zero = torch.zeros_like(w[..., 0])
    row0 = torch.stack([zero, -w[..., 2], w[..., 1]], dim=-1)
    row1 = torch.stack([w[..., 2], zero, -w[..., 0]], dim=-1)
    row2 = torch.stack([-w[..., 1], w[..., 0], zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotation_vector_to_matrix(w: torch.Tensor) -> torch.Tensor:
    """Rotation vectors ``(..., 3)`` in radians as matrices ``(..., 3, 3)`` (Rodrigues).

    Written with the series limits of ``sin(t) / t`` and ``(1 - cos t) / t^2`` at small
    angles, so it and its gradient stay finite at ``w = 0``.
    """
    k = skew_matrix(w)
    th2 = (w * w).sum(-1)  # (...,)
    th = th2.clamp_min(_EPS).sqrt()
    small = th2 < 1e-12
    a = torch.where(small, torch.ones_like(th), torch.sin(th) / th)  # sin th / th
    b = torch.where(small, torch.full_like(th, 0.5), (1.0 - torch.cos(th)) / th2.clamp_min(_EPS))
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(k.shape)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)
