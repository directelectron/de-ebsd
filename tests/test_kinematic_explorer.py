"""Focused checks for the lightweight STEM kinematic explorer."""

import numpy as np
import pytest

from de_ebsd import Crystal
from de_ebsd.diffraction.kinematic import (
    KinematicPatternModel,
    orientation_from_zone_axis,
)


@pytest.fixture
def silicon():
    return Crystal.from_parameters(a=0.5431, space_group=227, atoms=[("Si", (0.125, 0.125, 0.125))])


def test_silicon_absence_and_shape(silicon):
    model = KinematicPatternModel(silicon, dmin=0.1)
    pattern = model.project(orientation_from_zone_axis(silicon, [0, 0, 1]))
    assert pattern.lines.ndim == 3 and pattern.lines.shape[1:] == (2, 2)
    assert pattern.spots.ndim == 2 and pattern.spots.shape[1] == 2
    assert not any(np.all(h == [2, 0, 0]) for h in pattern.spot_hkl)
    # A shortest allowed 400 harmonic is available as a geometric guide.
    assert any(np.all(h == [4, 0, 0]) for h in model.forbidden_hkl) is False


def test_roll_rotates_lines_and_spots_together(silicon):
    model = KinematicPatternModel(silicon, dmin=0.1)
    p0 = model.project(orientation_from_zone_axis(silicon, [0, 0, 1]), semiangle_deg=10)
    p1 = model.project(
        orientation_from_zone_axis(silicon, [0, 0, 1], roll_deg=90), semiangle_deg=10
    )
    # Set equality by hkl so ordering is not part of the contract.
    common = set(map(tuple, p0.spot_hkl)) & set(map(tuple, p1.spot_hkl))
    assert common
    for hkl in list(common)[:5]:
        a = p0.spots[np.all(p0.spot_hkl == hkl, axis=1)][0]
        b = p1.spots[np.all(p1.spot_hkl == hkl, axis=1)][0]
        # A positive roll rotates the sample axes around +z, hence crystal
        # vectors appear clockwise in the fixed screen frame.
        assert np.allclose(b, [a[1], -a[0]], atol=1e-8)


def test_non_cubic_zone_axis_uses_direct_basis():
    crystal = Crystal.from_parameters(
        a=0.4,
        b=0.5,
        c=0.7,
        alpha=80,
        beta=75,
        gamma=70,
        atoms=[("Si", (0, 0, 0))],
    )
    q = orientation_from_zone_axis(crystal, [1, 2, 3])
    # R maps the beam +z into the direct-lattice Cartesian direction.
    from de_ebsd.diffraction.kinematic import _as_rotation_matrix

    got = _as_rotation_matrix(q)[:, 2]
    expected = np.asarray([1, 2, 3]) @ crystal.direct_structure_matrix.T
    expected /= np.linalg.norm(expected)
    assert np.allclose(got, expected)


def test_energy_changes_spot_radius_but_not_lines(silicon):
    model = KinematicPatternModel(silicon, energy_kev=80, dmin=0.1)
    q = orientation_from_zone_axis(silicon, [0, 0, 1])
    low = model.project(q, semiangle_deg=20, excitation_width=1.0)
    model.energy_kev = 300
    high = model.project(q, semiangle_deg=20, excitation_width=1.0)
    assert np.array_equal(low.line_hkl, high.line_hkl)
    assert np.allclose(low.lines, high.lines)
    common = set(map(tuple, low.spot_hkl)) & set(map(tuple, high.spot_hkl))
    assert common
    hkl = next(iter(common))
    r_low = np.linalg.norm(low.spots[np.all(low.spot_hkl == hkl, axis=1)][0])
    r_high = np.linalg.norm(high.spots[np.all(high.spot_hkl == hkl, axis=1)][0])
    assert r_high < r_low


@pytest.mark.parametrize("bad", [0, -1, np.nan])
def test_invalid_inputs_fail(silicon, bad):
    with pytest.raises(ValueError):
        KinematicPatternModel(silicon, energy_kev=bad)
    model = KinematicPatternModel(silicon)
    with pytest.raises(ValueError):
        model.project([1, 0, 0, 0], excitation_width=bad)


def test_filtered_reflections_keep_their_own_intensities(silicon):
    """Basis extinctions must not shift the cached intensity/reflection mapping."""
    from diffpy.structure import Atom, Lattice, Structure
    from diffsims.utils.sim_utils import get_kinematical_intensities

    model = KinematicPatternModel(silicon)
    p = model.project([1, 0, 0, 0], excitation_width=0.5)
    structure = Structure(
        atoms=[Atom("Si", xyz) for xyz in silicon.positions[0]],
        lattice=Lattice(5.431, 5.431, 5.431, 90, 90, 90),
    )
    expected = get_kinematical_intensities(
        structure,
        p.spot_hkl,
        silicon.g_length(p.spot_hkl) / 10,
        debye_waller_factors={"Si": 0.5},
    )
    expected /= 1 + (p.excitation_errors / 0.5) ** 2
    expected /= expected.max()
    np.testing.assert_allclose(p.intensities, expected, atol=1e-12)
    assert p.intensities.min() > 0


def test_bragg_condition_for_positive_incident_beam():
    """An exact off-axis elastic reflection selects +hkl, not its Friedel mate."""
    from de_ebsd.core.constants import electron_wavelength_nm

    c = Crystal.from_parameters(a=0.4, atoms=[("Si", (0, 0, 0))])
    lam = electron_wavelength_nm(200)
    theta = np.arcsin(lam / (2 * c.a))
    beam = [-np.sin(theta), 0, np.cos(theta)]
    model = KinematicPatternModel(c, dmin=0.3)
    p = model.project(orientation_from_zone_axis(c, beam), excitation_width=1e-7)
    np.testing.assert_array_equal(p.spot_hkl, [[1, 0, 0]])
    np.testing.assert_allclose(p.spots, [[np.tan(2 * theta), 0]], atol=1e-12)
    assert abs(p.excitation_errors[0]) < 1e-12


def test_skew_cell_keeps_mixed_index_reflections():
    c = Crystal.from_parameters(
        a=0.5,
        b=0.5,
        c=0.05,
        gamma=10,
        atoms=[("Si", (0, 0, 0))],
    )
    model = KinematicPatternModel(c, dmin=0.1)
    assert 1 / c.g_length(np.array([[4, 4, 0]]))[0] > 0.1
    assert (4, 4, 0) in set(map(tuple, model._hkl))
    p = model.project(orientation_from_zone_axis(c, [0, 0, 1]))
    assert (4, 4, 0) in set(map(tuple, p.spot_hkl))


def test_per_site_occupancy_and_debye_waller_are_preserved():
    from de_ebsd import AtomSite
    from de_ebsd.diffraction.kinematic import _DiffsimsElasticScatteringAdapter

    c = Crystal.from_parameters(
        a=0.4,
        atoms=[
            AtomSite("Si", (0, 0, 0), occupancy=0.5, debye_waller=0),
            AtomSite("Si", (0.5, 0, 0), occupancy=0.25, debye_waller=0.008),
        ],
    )
    single = Crystal.from_parameters(
        a=0.4,
        atoms=[("Si", (0, 0, 0))],
        debye_waller=0,
    )
    hkl = np.array([[1, 0, 0], [2, 0, 0]])
    g = c.g_length(hkl)
    base = _DiffsimsElasticScatteringAdapter(single).elastic_intensities(hkl, g)
    actual = _DiffsimsElasticScatteringAdapter(c).elastic_intensities(hkl, g)
    dw = np.exp(-0.008 * g**2 / 4)
    expected = base * (0.5 + np.array([-1, 1]) * 0.25 * dw) ** 2
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_clipped_lines_obey_rotated_plane_equation(silicon):
    from de_ebsd.diffraction.kinematic import _as_rotation_matrix

    model = KinematicPatternModel(silicon)
    q = orientation_from_zone_axis(silicon, [1, 2, 3], roll_deg=27)
    p = model.project(q, semiangle_deg=25)
    assert len(p.lines)
    normals = silicon.g_cartesian(p.line_hkl) @ _as_rotation_matrix(q)
    rays = np.concatenate([p.lines, np.ones((*p.lines.shape[:2], 1))], axis=-1)
    np.testing.assert_allclose(np.einsum("li,lpi->lp", normals, rays), 0, atol=1e-12)
    assert np.max(np.abs(p.lines)) <= np.tan(np.deg2rad(25)) + 1e-12


def test_rotation_reuses_elastic_cache(silicon, monkeypatch):
    model = KinematicPatternModel(silicon)

    def fail(*args, **kwargs):
        pytest.fail("rotation should not recompute structure factors")

    monkeypatch.setattr(model._adapter, "elastic_intensities", fail)
    model.project([1, 0, 0, 0])
    model.energy_kev = 300
    model.project(orientation_from_zone_axis(silicon, [1, 1, 1]))


def test_excessive_reflection_search_fails_before_allocation(silicon):
    with pytest.raises(ValueError, match="increase dmin"):
        KinematicPatternModel(silicon, dmin=1e-5)


@pytest.mark.parametrize("q", [[0, 0, 0, 0], [1, np.nan, 0, 0], [1, 2, 3], -np.eye(3)])
def test_invalid_rotations_are_rejected(silicon, q):
    with pytest.raises(ValueError):
        KinematicPatternModel(silicon).project(q)
