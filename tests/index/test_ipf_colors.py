"""ipf_colors takes a CrystalMap, an orix Rotation or Euler angles, and agrees across them."""

import numpy as np
import pytest
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.plot import IPFColorKeyTSL
from orix.quaternion import Rotation
from orix.quaternion.symmetry import Oh
from orix.vector import Vector3d

from de_ebsd.viz.maps import ipf_colors


@pytest.fixture(scope="module")
def rotations() -> Rotation:
    np.random.seed(9)
    return Rotation.random(12)


def test_euler_angles_color_as_orix_does(rotations):
    expected = np.asarray(IPFColorKeyTSL(Oh).orientation2color(rotations))
    rgb = ipf_colors(rotations.to_euler().reshape(3, 4, 3), Oh)
    assert rgb.shape == (3, 4, 3)
    assert rgb.dtype == np.float64 and rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert np.allclose(rgb.reshape(-1, 3), expected)


def test_every_form_of_the_point_group_agrees(rotations):
    euler = rotations.to_euler()
    by_symmetry = ipf_colors(euler, Oh)
    assert np.allclose(ipf_colors(euler, 225), by_symmetry)  # Fm-3m's point group is m-3m
    assert np.allclose(ipf_colors(euler, Phase(space_group=225)), by_symmetry)
    assert np.allclose(ipf_colors(rotations, Oh), by_symmetry)


def test_the_direction_is_the_sample_direction_shown(rotations):
    expected = np.asarray(
        IPFColorKeyTSL(Oh, direction=Vector3d.xvector()).orientation2color(rotations)
    )
    assert np.allclose(ipf_colors(rotations, Oh, direction=[1, 0, 0]), expected)


def test_a_crystal_map_colors_each_phase_by_its_own_symmetry(rotations):
    phases = PhaseList(
        names=["cubic", "hexagonal"], space_groups=[225, 194], point_groups=[None, None]
    )
    phase_id = np.array([0, 1] * 6)
    phase_id[-1] = -1  # not indexed
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    crystal_map = CrystalMap(
        rotations, phase_id=phase_id, x=x.ravel(), y=y.ravel(), phase_list=phases
    )
    rgb = ipf_colors(crystal_map)
    assert rgb.shape == (3, 4, 3)
    flat = rgb.reshape(-1, 3)
    assert np.allclose(flat[0::2], ipf_colors(rotations[0::2], 225))
    assert np.allclose(flat[1:-1:2], ipf_colors(rotations[1:-1:2], 194))
    assert np.allclose(flat[-1], 0.0)


def test_anything_but_a_crystal_map_needs_a_point_group(rotations):
    with pytest.raises(ValueError, match="point group"):
        ipf_colors(rotations)
