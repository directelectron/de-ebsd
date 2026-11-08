"""The scan's frame is kikuchipy's sample frame turned 90 degrees about the normal."""

import numpy as np
from orix.quaternion import Rotation
from orix.vector import Vector3d

from de_ebsd.core.frames import SCAN_TURN, scan_euler, scan_tensor, to_scan_frame

EULER = np.radians([[35.0, 48.0, 112.0]])


def test_orientations_turn_ninety_degrees_about_the_surface_normal():
    """An IPF-Z color cannot show the turn, so it is checked on the x axis."""
    simulated = Rotation.from_euler(EULER)
    scan = Rotation.from_euler(to_scan_frame(EULER))
    assert np.isclose(np.degrees(simulated.angle_with(scan)[0]), 90.0)
    # the scan's x is kikuchipy's -y
    assert np.allclose((simulated * -Vector3d.yvector()).data, (scan * Vector3d.xvector()).data)
    # the surface normal stays where it was, and with it every IPF-Z color
    assert np.allclose((simulated * Vector3d.zvector()).data, (scan * Vector3d.zvector()).data)


def test_the_turn_is_orix_s():
    turn = Rotation.from_axes_angles([0, 0, 1], -90, degrees=True)
    by_orix = (Rotation.from_euler(EULER) * turn).to_euler()
    assert np.allclose(to_scan_frame(EULER), by_orix, atol=1e-12)


def test_strain_turns_into_the_scans_frame_as_the_orientations_do():
    np.random.seed(3)
    g = Rotation.random(4).to_matrix()
    scan = Rotation.from_euler(scan_euler(g)).to_matrix()
    # v_crystal = g v_simulation = g_scan v_scan, with v_scan = SCAN_TURN v_simulation
    assert np.allclose(scan, g @ SCAN_TURN.T, atol=1e-9)
    stretch = np.diag([1e-3, -1e-3, 0.0])  # along kikuchipy's x
    assert np.allclose(scan_tensor(stretch), np.diag([-1e-3, 1e-3, 0.0]))
    assert scan_tensor(np.stack([stretch] * 5)).shape == (5, 3, 3)
