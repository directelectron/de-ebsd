"""Focused tests for the native STEM explorer interface and callbacks."""

from types import SimpleNamespace

import numpy as np
import pytest

anyplotlib = pytest.importorskip("anyplotlib")

from de_ebsd.crystal.lattice import Crystal
from de_ebsd.diffraction.kinematic import orientation_from_zone_axis
from de_ebsd.viz.explorer import CrystalExplorer


class _Model:
    band_normals = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dmin = 0.1

    def __init__(self):
        self.energy_kev = 200.0
        self.calls = 0

    def project(self, orientation, *, semiangle_deg, excitation_width):
        self.calls += 1
        return SimpleNamespace(
            lines=np.asarray([[[0.0, -0.02], [0.02, 0.02]]]),
            line_hkl=np.asarray([[1, 0, 0]]),
            spots=np.asarray([[0.01, -0.01]]),
            spot_hkl=np.asarray([[1, 1, 0]]),
            intensities=np.asarray([1.0]),
        )


@pytest.fixture
def crystal():
    return Crystal.from_parameters(
        a=0.5,
        b=0.6,
        c=0.7,
        alpha=90,
        beta=90,
        gamma=90,
        space_group=1,
        atoms=[("Si", (0, 0, 0))],
    )


def test_zone_axis_uses_direct_lattice_and_maps_beam(crystal):
    q = orientation_from_zone_axis(crystal, [0, 0, 1])
    assert q.shape == (4,)
    # The c axis is the beam direction even though this cell is non-cubic.
    from de_ebsd.viz.explorer import _quaternion_to_matrix

    np.testing.assert_allclose(
        _quaternion_to_matrix(q)[:, 2],
        crystal.direct_structure_matrix @ np.asarray([0.0, 0.0, 1.0]) / crystal.c,
    )


def test_native_layers_toggle_and_sphere_camera_callback(crystal):
    model = _Model()
    explorer = CrystalExplorer(crystal, model=model)
    figure = explorer.figure
    assert len(figure.get_axes()) == 3
    assert model.calls == 1

    explorer.set_show_lines(False)
    explorer.set_show_spots(False)
    # Marker serialization is exercised by anyplotlib on every setter.
    assert explorer.show_lines is False and explorer.show_spots is False
    explorer.set_show_lines(True)
    explorer.set_show_spots(True)

    from anyplotlib.callbacks import Event

    before = explorer.orientation
    explorer._sphere_artist.callbacks.fire(
        Event(event_type="pointer_up", azimuth=12.0, elevation=35.0)
    )
    assert not np.allclose(before, explorer.orientation)
    assert explorer.result is not None


def test_empty_reflection_set_and_reset_preserve_constructor_values(crystal):
    explorer = CrystalExplorer(crystal, model=_Model(), energy_kev=120, dmin=10, show_spots=False)
    _ = explorer.figure
    assert explorer.result.spots.shape == (1, 2)  # fake model still has one spot
    explorer.set_show_spots(True)
    explorer.reset()
    assert explorer.energy_kev == pytest.approx(120)
    assert explorer.dmin == pytest.approx(10)
    assert explorer.show_spots is False


def test_real_empty_reflection_set_serializes_with_spots_hidden(crystal):
    from de_ebsd.diffraction.kinematic import KinematicPatternModel

    model = KinematicPatternModel(crystal, energy_kev=200, dmin=10)
    explorer = CrystalExplorer(crystal, model=model, show_spots=False)
    _ = explorer.figure
    assert explorer.result.lines.shape == (0, 2, 2)
    assert explorer.result.spots.shape == (0, 2)
    explorer.set_show_spots(True)
    assert explorer.result.spots.shape == (0, 2)
