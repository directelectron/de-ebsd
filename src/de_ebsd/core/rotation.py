"""Quaternion rotations, matching orix / kikuchipy conventions.

Quaternions are (a, b, c, d) with the scalar part first (the order of
``orix.quaternion.Rotation.data``), so an orix ``Rotation`` feeds in directly
via ``torch.as_tensor(rotation.data)``. The matrix construction reproduces
kikuchipy's ``rotate_vector`` exactly , so
projections agree with kikuchipy to numerical precision.

Bunge Euler angles ``(phi1, Phi, phi2)`` are ZXZ, in radians, and their matrices are
orix's ``Rotation.from_euler(euler).to_matrix()``: they take a sample-frame
direction to the crystal frame, ``v_crystal = g @ v_sample``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from de_ebsd.core.constants import DIVISION_FLOOR

__all__ = [
    "axis_angle_matrix",
    "euler_to_matrix",
    "euler_to_matrix_torch",
    "matrix_to_euler",
    "matrix_to_quaternion",
    "misorientation_deg",
    "quaternion_to_matrix",
    "rotate_vectors",
    "rotation_vector_to_matrix",
    "skew_matrix",
]


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
    th = th2.clamp_min(DIVISION_FLOOR).sqrt()
    small = th2 < 1e-12
    a = torch.where(small, torch.ones_like(th), torch.sin(th) / th)  # sin th / th
    b = torch.where(
        small, torch.full_like(th, 0.5), (1.0 - torch.cos(th)) / th2.clamp_min(DIVISION_FLOOR)
    )
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(k.shape)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def axis_angle_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotation matrix about *axis* by *angle_deg* degrees (right-handed)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = math.radians(float(angle_deg))
    c, s = math.cos(t), math.sin(t)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to scalar-first quaternion."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation matrix must be finite and have shape (3, 3)")
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-8) or np.linalg.det(r) <= 0:
        raise ValueError("rotation matrix must be a proper rotation")
    trace = float(np.trace(r))
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = np.array(
            [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = 2.0 * math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0))
            q = np.array(
                [
                    (r[2, 1] - r[1, 2]) / s,
                    0.25 * s,
                    (r[0, 1] + r[1, 0]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                ]
            )
        elif i == 1:
            s = 2.0 * math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0))
            q = np.array(
                [
                    (r[0, 2] - r[2, 0]) / s,
                    (r[0, 1] + r[1, 0]) / s,
                    0.25 * s,
                    (r[1, 2] + r[2, 1]) / s,
                ]
            )
        else:
            s = 2.0 * math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0))
            q = np.array(
                [
                    (r[1, 0] - r[0, 1]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                    (r[1, 2] + r[2, 1]) / s,
                    0.25 * s,
                ]
            )
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def euler_to_matrix(euler) -> np.ndarray:
    """Bunge Euler angles ``(..., 3)`` in radians as matrices ``(..., 3, 3)``.

    Equal to orix's ``Rotation.from_euler(euler).to_matrix()`` without building the
    quaternions: sample to crystal, ``v_crystal = g @ v_sample``.
    """
    euler = np.asarray(euler, dtype=np.float64)
    phi1, Phi, phi2 = euler[..., 0], euler[..., 1], euler[..., 2]
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    rows = (
        (c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s),
        (-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s),
        (s1 * s, -c1 * s, c),
    )
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def euler_to_matrix_torch(euler: torch.Tensor) -> torch.Tensor:
    """:func:`euler_to_matrix` in torch, differentiable: ``(..., 3)`` to ``(..., 3, 3)``.

    Built by stacking rather than by writing into an empty tensor: an in-place write
    into a tensor that carries gradients is the classic way to lose them.
    """
    phi1, Phi, phi2 = euler.unbind(-1)
    c1, s1 = torch.cos(phi1), torch.sin(phi1)
    c, s = torch.cos(Phi), torch.sin(Phi)
    c2, s2 = torch.cos(phi2), torch.sin(phi2)
    row0 = torch.stack([c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s], dim=-1)
    row1 = torch.stack([-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s], dim=-1)
    row2 = torch.stack([s1 * s, -c1 * s, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_euler(matrix) -> np.ndarray:
    """Bunge Euler angles ``(..., 3)`` in radians of matrices ``(..., 3, 3)``.

    The inverse of :func:`euler_to_matrix`, with ``phi1`` and ``phi2`` in ``[0, 2 pi)``
    and ``Phi`` in ``[0, pi]``. At ``Phi = 0`` only ``phi1 + phi2`` is defined (at
    ``Phi = pi`` only ``phi1 - phi2``), and all of it is given to ``phi1``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    Phi = np.arccos(np.clip(m[..., 2, 2], -1.0, 1.0))
    general = np.abs(np.sin(Phi)) > 1e-8
    phi1 = np.where(
        general,
        np.arctan2(m[..., 2, 0], -m[..., 2, 1]),
        np.arctan2(m[..., 0, 1], m[..., 0, 0]),
    )
    phi2 = np.where(general, np.arctan2(m[..., 0, 2], m[..., 1, 2]), 0.0)
    return np.mod(np.stack([phi1, Phi, phi2], axis=-1), 2 * np.pi)


def misorientation_deg(rotations, reference, symmetry) -> np.ndarray:
    """The angle in degrees ``(N,)`` from each of *rotations* ``(N, 3, 3)`` to
    *reference* ``(3, 3)``, the smallest over the crystal's point group.

    Parameters
    ----------
    rotations, reference
        Orientation matrices, sample to crystal (:func:`euler_to_matrix`), both in
        the same sample frame. Which frame does not matter: a turn of the sample
        frame cancels in ``g @ reference.T``.
    symmetry
        The point group, as an orix ``Symmetry`` or its operators as matrices
        ``(K, 3, 3)``.
    """
    operators = symmetry.to_matrix() if hasattr(symmetry, "to_matrix") else symmetry
    turn = np.asarray(rotations, dtype=np.float64) @ np.asarray(reference, dtype=np.float64).T
    # trace(S^T M) for every operator S at once; a group holds each S's transpose too
    traces = np.einsum("kij,nij->nk", np.asarray(operators, dtype=np.float64), turn).max(-1)
    return np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))
