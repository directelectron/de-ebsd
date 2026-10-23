"""Camera and great-circle geometry for the orientation explorer's 3-D sphere."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["band_path", "camera_angles", "camera_direction", "camera_frame"]


def camera_direction(azimuth: float, elevation: float) -> np.ndarray:
    """Near-facing sphere direction for anyplotlib's Plot3D camera convention."""
    a, e = np.deg2rad([float(azimuth), float(elevation)])
    return np.array([math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)])


def camera_angles(direction: np.ndarray) -> tuple[float, float]:
    n = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    return float(np.rad2deg(np.arctan2(n[0], -n[1]))), float(
        np.rad2deg(np.arcsin(np.clip(n[2], -1, 1)))
    )


def camera_frame(azimuth: float, elevation: float) -> np.ndarray:
    """The exact right/up/facing basis used by anyplotlib's Plot3D renderer."""
    a, e = np.deg2rad([float(azimuth), float(elevation)])
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    facing = np.array([ce * sa, -ce * ca, se])
    right = np.array([ca, sa, 0.0])
    up = np.array([-se * sa, se * ca, ce])
    return np.column_stack([right, up, facing])


def band_path(normals: np.ndarray) -> np.ndarray:
    """Pack reciprocal-plane great circles into a dense native point path.

    Plot3D has one connected line primitive and no pen-lift/NaN contract.  A
    fine point cloud preserves each great circle without drawing false chords
    between unrelated planes.
    """
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    rings = []
    t = np.linspace(0.0, 2.0 * np.pi, 256)
    for normal in normals:
        n = normal / max(np.linalg.norm(normal), 1e-15)
        if not np.isfinite(n).all():
            continue
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = ref - np.dot(ref, n) * n
        u /= max(np.linalg.norm(u), 1e-15)
        v = np.cross(n, u)
        rings.append(np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    if not rings:
        return np.zeros((2, 3), dtype=float)
    return np.concatenate(rings, axis=0)
