"""A synthetic EBSD scan with known orientations, drawn from kinematic cubic bands.

Two grains and a gradient, placed asymmetrically: a lower-right wedge of one
orientation against a grain whose phi1 drifts along x. The patterns sit on a smooth
off-center detector background, as real ones do, which background removal has to
take away. Ported from SpyDE's ``spyde.data.synthetic.ebsd_patterns``.
"""

import numpy as np
import torch
from orix.quaternion import Rotation

from de_ebsd import EBSDDetector, KinematicMaster

PC = (0.5, 0.5, 0.55)


def scan_detector(shape) -> EBSDDetector:
    """The standard 70 degree set-up at the scan's pattern center."""
    return EBSDDetector(shape=tuple(shape), pattern_center=PC, sample_tilt=70.0)


def kinematic_patterns(rotations, shape, master=None) -> np.ndarray:
    """Raw kinematic patterns ``(N, H, W)`` float32 for an orix ``Rotation``."""
    master = KinematicMaster() if master is None else master
    return master.project(scan_detector(shape), rotations, rescale=False).numpy()


def detector_background(shape) -> np.ndarray:
    """The smooth, off-center gradient every experimental pattern sits on."""
    dy, dx = shape
    gy, gx = np.mgrid[0:dy, 0:dx].astype(np.float32)
    return 0.35 + 0.30 * ((gy / dy) * 0.6 + (gx / dx) * 0.4)


def kinematic_scan(nav=(8, 8), shape=(40, 40), noise=0.0, seed=0):
    """``(patterns, euler, grain2)``: uint8 ``(ny, nx, H, W)`` patterns on a detector
    background, their Bunge Euler angles ``(ny, nx, 3)`` in radians, and the mask of
    the uniform second grain."""
    ny, nx = nav
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    fy, fx = yy / max(ny - 1, 1), xx / max(nx - 1, 1)
    phi1 = np.deg2rad(10.0 + 35.0 * fx)
    Phi = np.full((ny, nx), np.deg2rad(30.0))
    phi2 = np.full((ny, nx), np.deg2rad(15.0))
    grain2 = (fx + fy) > 1.15
    phi1[grain2], Phi[grain2], phi2[grain2] = np.deg2rad([70.0, 55.0, 40.0])
    euler = np.stack([phi1, Phi, phi2], -1)

    data = kinematic_patterns(Rotation.from_euler(euler.reshape(-1, 3)), shape)
    data = data / max(float(data.max()), 1e-9) + detector_background(shape)
    if noise:
        data = data + np.random.default_rng(seed).normal(0.0, noise, data.shape)
    data = np.clip(data, 0, None)
    data = (data / data.max() * 255.0).astype(np.uint8)
    return data.reshape(ny, nx, *shape), euler, grain2


def pattern_similarity(rotations_a, rotations_b, shape=(32, 32)) -> np.ndarray:
    """Compare orientations by the patterns they make (NCC per pair), not by Euler
    distance: Euler angles are not a metric space, and cubic symmetry gives several
    triples for one crystal."""
    master = KinematicMaster(dtype=torch.float64)
    a = kinematic_patterns(rotations_a, shape, master).reshape(rotations_a.size, -1)
    b = kinematic_patterns(rotations_b, shape, master).reshape(rotations_b.size, -1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)
