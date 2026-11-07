"""Kinematic bands (:mod:`de_ebsd.sim.kinematic_bands`): the overlay and its patterns.

Two different things are pinned:

* **the lines land on the bands**: sampling the projected pattern across each
  drawn line, the pattern peaks on it, and a WRONG orientation's lines do not;
* **the rotation convention is orix's**, which no score can see: a transposed
  rotation matches its own dictionary perfectly and then colors an IPF map from the
  inverse orientation. The tell is symmetry: a CRYSTAL symmetry operation leaves the
  pattern alone, a SAMPLE rotation does not.

Ported from SpyDE's ``test_ebsd_bands``.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation

from de_ebsd import EBSDDetector
from de_ebsd.core.geometry import sample_to_detector_matrix
from de_ebsd.core.rotation import euler_to_matrix
from de_ebsd.sim.kinematic_bands import (
    KinematicMaster,
    Reflectors,
    band_lines,
    cubic_reflectors,
    reflectors_from_phase,
    zone_axis_points,
)

SHAPE = (60, 60)
PC = (0.5, 0.5, 0.55)
EULER = np.deg2rad([33.3, 30.0, 15.0])
SILVER_CIF = Path(__file__).parent / "data" / "cif" / "Silver_AMCSD0011135.cif"


def detector(shape=SHAPE, pc=PC, sample_tilt=70.0, tilt=0.0, azimuthal=0.0, twist=0.0):
    return EBSDDetector(
        shape=shape,
        pattern_center=pc,
        sample_tilt=sample_tilt,
        tilt=tilt,
        azimuthal=azimuthal,
        twist=twist,
    )


def rotation(euler):
    return Rotation.from_euler(np.atleast_2d(euler))


def patterns(euler, reflectors=None, det=None):
    """Raw kinematic patterns ``(N, H, W)`` as numpy."""
    master = KinematicMaster(reflectors, dtype=torch.float64)
    det = detector() if det is None else det
    return master.project(det, rotation(euler), rescale=False).numpy()


def _sample_along(pattern, segments, n=40, offset=0.0):
    """Mean intensity along each segment, shifted *offset* pixels perpendicular to it."""
    out = []
    h, w = pattern.shape
    for (x0, y0), (x1, y1) in segments:
        t = np.linspace(0.05, 0.95, n)
        v = np.array([x1 - x0, y1 - y0], float)
        nrm = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-12) * offset
        xs = np.clip(np.round(x0 + t * (x1 - x0) + nrm[0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(y0 + t * (y1 - y0) + nrm[1]).astype(int), 0, h - 1)
        out.append(float(pattern[ys, xs].mean()))
    return np.asarray(out)


def _perpendicular_profile(pattern, segments, half=6):
    """Intensity across each band, ``(n_bands, 2 * half + 1)``, at offsets ``-half..half``.

    What the overlay claims is LOCAL: the pattern is brightest on the line and falls
    away to either side. A comparison with the whole-frame mean would pass a line
    through the bright middle drawn in the wrong place.
    """
    offs = np.arange(-half, half + 1, dtype=float)
    return np.column_stack([_sample_along(pattern, segments, offset=o) for o in offs]), offs


def _distances_to_lines(point, segments):
    d = []
    for (x0, y0), (x1, y1) in segments:
        v = np.array([x1 - x0, y1 - y0], float)
        n = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-12)
        d.append(abs(float((point - np.array([x0, y0])) @ n)))
    return sorted(d)


class TestConventionAgreesWithOrix:
    def test_euler_matrix_matches_orix(self):
        assert np.allclose(euler_to_matrix(EULER), rotation(EULER).to_matrix()[0], atol=1e-9)

    def test_crystal_symmetry_leaves_the_pattern_alone(self):
        """phi2 + 90 deg is a cubic 4-fold about the crystal z: the same pattern."""
        from orix.quaternion import Orientation, symmetry

        turned = EULER + np.deg2rad([0, 0, 90])
        a = Orientation.from_euler(EULER[None], symmetry.Oh)
        b = Orientation.from_euler(turned[None], symmetry.Oh)
        assert float(np.rad2deg(a.angle_with(b))[0]) < 1e-6
        assert np.allclose(patterns(EULER), patterns(turned), atol=1e-6), (
            "a crystal symmetry operation changed the pattern: the normals are being "
            "rotated by the transpose"
        )

    def test_a_sample_rotation_does_change_the_pattern(self):
        """Without this a simulator that ignores the rotation would pass the above."""
        p0 = patterns(EULER)[0]
        p1 = patterns(EULER + np.deg2rad([90, 0, 0]))[0]
        assert not np.allclose(p0, p1, atol=1e-3)


def _numpy_reference(euler, reflectors, shape, pc, angles):
    """The projection written out independently: Bruker rays, the sample-to-detector
    matrix, orix's matrix and the Gaussian band profile, in numpy."""
    dy, dx = shape
    pcx, pcy, pcz = pc
    gy, gx = np.mgrid[0:dy, 0:dx].astype(float)
    rays = np.stack(
        [dx / dy * ((gx + 0.5) / dx - pcx), pcy - (gy + 0.5) / dy, np.full_like(gx, pcz)], -1
    )
    rays = rays.reshape(-1, 3) @ sample_to_detector_matrix(*angles)  # sample frame
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    out = []
    for g in rotation(euler).to_matrix():
        d = rays @ (reflectors.normals @ g).T
        out.append((np.exp(-0.5 * (d / reflectors.widths) ** 2) * reflectors.weights).sum(1))
    return np.asarray(out).reshape(-1, dy, dx)


class TestProjectionMatchesANumpyReference:
    @pytest.mark.parametrize(
        "angles", [(70.0, 0.0, 0.0, 0.0), (62.0, 7.0, 4.0, -3.0)], ids=["default-70", "all-angles"]
    )
    def test_matches(self, angles):
        euler = np.array([[0.3, 0.7, 1.1], [1.5, 0.4, 0.2], [4.0, 2.0, 0.5]])
        refl = cubic_reflectors()
        det = detector((20, 28), (0.45, 0.55, 0.6), *angles)
        want = _numpy_reference(euler, refl, (20, 28), (0.45, 0.55, 0.6), angles)
        got = KinematicMaster(refl).project(det, rotation(euler), rescale=False).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_is_differentiable_in_the_orientation(self):
        q = torch.as_tensor(rotation(EULER).data, dtype=torch.float32).requires_grad_(True)
        KinematicMaster().project(detector((16, 16)), q).sum().backward()
        assert torch.isfinite(q.grad).all() and (q.grad.abs() > 0).any()

    def test_sampling_in_blocks_is_sampling_at_once(self, monkeypatch):
        import de_ebsd.sim.kinematic_bands as kb

        master = KinematicMaster(dtype=torch.float64)
        v = torch.randn(500, 3, dtype=torch.float64)
        v = v / v.norm(dim=-1, keepdim=True)
        whole = master.sample(v)
        monkeypatch.setattr(kb, "_SAMPLE_BLOCK_ELEMENTS", 13 * 7)
        np.testing.assert_allclose(master.sample(v).numpy(), whole.numpy(), atol=1e-12)

    def test_the_sphere_texture_is_the_same_intensity(self):
        """The phase viewer's band sphere and a detector pattern are one function."""
        master = KinematicMaster(dtype=torch.float64)
        texture = master.to_equirectangular(height=32)
        assert texture.shape == (32, 64)
        assert np.isclose(float(texture[0, 0]), float(master.sample(torch.tensor([0.0, 0, 1]))))


class TestBandLinesLandOnBands:
    def test_every_line_sits_on_a_local_intensity_peak(self):
        pat = patterns(EULER)[0]
        segs, weights = band_lines(rotation(EULER), detector=detector())
        assert len(segs) > 3 and len(weights) == len(segs)
        profile, offs = _perpendicular_profile(pat, segs)
        peak = offs[profile.argmax(axis=1)]
        assert np.abs(peak).max() <= 1.0, f"lines are up to {np.abs(peak).max()} px off"
        assert (profile.max(1) > profile[:, [0, -1]].max(1)).all()

    def test_a_wrong_orientation_does_not_land_on_them(self):
        pat = patterns(EULER)[0]
        segs, _ = band_lines(rotation(np.deg2rad([70.0, 55.0, 40.0])), detector=detector())
        profile, offs = _perpendicular_profile(pat, segs)
        assert np.abs(offs[profile.argmax(axis=1)]).max() > 1.0
        assert _sample_along(pat, segs).mean() < pat.mean()

    def test_segments_are_the_line_collection_shape(self):
        segs, _ = band_lines(rotation(EULER), detector=detector())
        assert segs.ndim == 3 and segs.shape[1:] == (2, 2)

    def test_segments_stay_inside_the_detector(self):
        segs, _ = band_lines(rotation(EULER), detector=detector())
        assert segs.min() >= -0.5 - 1e-4
        assert segs[..., 0].max() <= SHAPE[1] - 0.5 + 1e-4
        assert segs[..., 1].max() <= SHAPE[0] - 0.5 + 1e-4

    def test_pixel_convention_is_center_on_integer(self):
        """A band through the pattern center is drawn through the center pixel."""
        # the sample y axis is the detector's x axis at every tilt, so the (020) plane
        # of the identity orientation is the vertical line through the pattern center
        segs, _ = band_lines(Rotation.identity(), detector=detector())
        center = (SHAPE[1] - 1) / 2.0
        assert np.abs(segs.mean(axis=1) - center).min() < 1e-5

    def test_max_bands_keeps_the_strongest(self):
        segs_all, w_all = band_lines(rotation(EULER), detector=detector())
        segs_few, w_few = band_lines(rotation(EULER), detector=detector(), max_bands=3)
        assert len(segs_few) <= 3 < len(segs_all)
        assert w_few.min() >= np.sort(w_all)[::-1][: len(w_few)].min() - 1e-6

    def test_a_band_that_misses_the_detector_is_dropped(self):
        few, _ = band_lines(rotation(EULER), detector=detector(pc=(0.5, 0.5, 4.0)))
        many, _ = band_lines(rotation(EULER), detector=detector(pc=(0.5, 0.5, 0.4)))
        assert len(few) < len(many)

    def test_one_orientation_only(self):
        with pytest.raises(ValueError, match="one orientation"):
            band_lines(rotation(np.zeros((2, 3))), detector=detector())


class TestReflectors:
    def test_cubic_set_is_friedel_unique(self):
        """4 {111} + 3 {200} + 6 {220} = 13 bands, one per +-g pair."""
        r = cubic_reflectors()
        assert len(r) == 13
        dots = np.abs(r.normals @ r.normals.T)
        np.fill_diagonal(dots, 0.0)
        assert dots.max() < 1 - 1e-9

    def test_widths_and_weights_are_per_reflector(self):
        r = cubic_reflectors()
        assert r.widths.shape == r.weights.shape == (len(r),)
        assert (r.widths > 0).all()

    def test_brightest_is_a_subset(self):
        r = cubic_reflectors()
        sub = r.brightest(5)
        assert len(sub) == 5
        assert sub.weights.min() >= np.sort(r.weights)[::-1][:5].min() - 1e-12

    def test_brightest_none_is_a_no_op(self):
        r = cubic_reflectors()
        assert len(r.brightest(None)) == len(r) == len(r.brightest(999))

    def test_the_dataclass_broadcasts_scalars(self):
        r = Reflectors(np.eye(3), 1.0, 0.05)
        assert r.weights.shape == (3,) and r.widths.shape == (3,)

    def test_a_crystal_and_its_phase_give_the_same_bands(self, ni_crystal):
        """de-ebsd's own reflections for a Crystal and diffsims' for its orix phase are
        the same planes at the same Bragg widths; only the structure-factor tables
        behind the weights differ."""
        from de_ebsd.index.io import phase_from_crystal

        def bands(source):
            r = reflectors_from_phase(source, dmin=0.08, max_bands=None)
            canonical = [tuple(h if next(x for x in h if x) > 0 else -h) for h in r.hkl]
            return dict(zip(canonical, r.widths, strict=True)), dict(
                zip(canonical, r.weights, strict=True)
            )

        our_widths, our_weights = bands(ni_crystal)
        their_widths, their_weights = bands(phase_from_crystal(ni_crystal))
        assert our_widths.keys() == their_widths.keys()
        assert all(np.isclose(our_widths[k], their_widths[k], rtol=1e-6) for k in our_widths)
        keys = list(our_weights)
        ours = [our_weights[k] for k in keys]
        theirs = [their_weights[k] for k in keys]
        assert np.corrcoef(ours, theirs)[0, 1] > 0.99

    def test_from_a_real_phase(self, ni_crystal):
        """Bragg-angle widths, an order of magnitude narrower than the cubic set's."""
        from de_ebsd.index.io import phase_from_crystal

        for source in (ni_crystal, phase_from_crystal(ni_crystal)):
            r = reflectors_from_phase(source, dmin=0.08, energy=20e3)
            assert len(r) > 5
            assert np.isfinite(r.weights).all() and r.weights.max() > 0
            assert (r.widths > 0).all() and r.widths.max() < 0.1
            assert np.allclose(np.linalg.norm(r.normals, axis=1), 1.0)

    def test_widths_are_the_bragg_angle(self, ni_crystal):
        """lambda / 2d, both in nm: (111) of Ni at 20 keV."""
        from de_ebsd.core.constants import electron_wavelength_nm

        r = reflectors_from_phase(ni_crystal, dmin=0.08, energy=20e3)
        i = next(k for k, h in enumerate(r.hkl) if sorted(np.abs(h)) == [1, 1, 1])
        d111 = ni_crystal.a / np.sqrt(3.0)
        assert r.widths[i] == pytest.approx(electron_wavelength_nm(20.0) / (2 * d111), rel=1e-6)

    def test_fcc_keeps_its_200_and_220_bands(self):
        """(100), (200) and (400) share a line and fcc's extinct (100) comes first:
        reducing by line alone kept it and dropped the real {200} and {220}."""
        from de_ebsd.index import phase_from_cif

        r = reflectors_from_phase(phase_from_cif(SILVER_CIF), energy=20e3)
        families = {tuple(sorted(np.abs(h))) for h in r.hkl}
        assert {(1, 1, 1), (0, 0, 2), (0, 2, 2)} <= families
        assert (0, 0, 1) not in families and (0, 1, 1) not in families
        assert (r.weights > 0).all()
        assert np.sum(np.abs(r.normals @ [0.0, 0.0, 1.0]) < r.widths) >= 6, "[001] is on no band"

    def test_a_structureless_phase_falls_back(self):
        from orix.crystal_map import Phase

        r = reflectors_from_phase(Phase(name="p", space_group=225))
        assert len(r) == len(cubic_reflectors())


class TestZoneAxes:
    def test_axes_fall_where_bands_cross(self):
        segs, _ = band_lines(rotation(EULER), detector=detector())
        za = zone_axis_points(rotation(EULER), detector=detector())
        assert len(za) >= 1
        for p in za:
            assert _distances_to_lines(p, segs)[1] < 1.0, f"zone axis {p} is on < 2 bands"

    def test_points_stay_on_the_detector(self):
        za = zone_axis_points(rotation(EULER), detector=detector())
        assert ((za >= -0.5) & (za <= np.array([SHAPE[1], SHAPE[0]]) - 0.5)).all()


GEOMETRIES = [
    pytest.param({}, id="default-70"),
    pytest.param(
        {"sample_tilt": 62.0, "tilt": 7.0, "azimuthal": 4.0, "twist": -3.0}, id="all-angles"
    ),
]
ORIENTATIONS = np.deg2rad(
    [[33.3, 30.0, 15.0], [120.0, 50.0, 70.0], [250.0, 80.0, 10.0], [10.0, 140.0, 300.0]]
)


class TestLinesAndPatternsShareTheGeometry:
    """The drawn lines (a closed form) and the projected patterns (a per-pixel dot
    product) must place the detector identically at every orientation and angle."""

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    @pytest.mark.parametrize(
        "euler", ORIENTATIONS, ids=lambda e: str(np.round(np.rad2deg(e)).astype(int).tolist())
    )
    def test_every_line_is_on_its_band(self, euler, geometry):
        """One band at a time, so a neighbor cannot pull the peak: every pixel at the
        top of a band's profile lies on the line drawn for it."""
        det = detector(**geometry)
        reflectors = cubic_reflectors()
        worst, compared = 0.0, 0
        for k in range(len(reflectors)):
            band = Reflectors(reflectors.normals[k : k + 1], [1.0], [0.01])
            segs, _ = band_lines(rotation(euler), band, det)
            rows, cols = np.nonzero(patterns(euler, band, det)[0] > 0.9)
            assert bool(len(segs)) == bool(len(rows)), "a band drawn but not shown, or back"
            if not len(segs):
                continue
            compared += 1
            (x0, y0), (x1, y1) = segs[0]
            normal = np.array([y0 - y1, x1 - x0])
            normal /= np.linalg.norm(normal)
            offset = (np.column_stack([cols, rows]) - [x0, y0]) @ normal
            worst = max(worst, float(np.abs(offset).max()))
        assert compared > 3, "no bands crossed the detector"
        assert worst < 1.0, f"a band line is {worst:.2f} px off its band"

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_zone_axes_fall_where_the_lines_cross(self, geometry):
        det = detector(**geometry)
        segs, _ = band_lines(rotation(EULER), detector=det)
        za = zone_axis_points(rotation(EULER), detector=det)
        assert len(za) >= 1
        for p in za:
            assert _distances_to_lines(p, segs)[1] < 1.0

    def test_the_tilt_moves_the_bands(self):
        """Lines drawn without the sample tilt miss a tilted pattern's bands."""
        pattern = patterns(EULER)[0]
        segs, _ = band_lines(rotation(EULER), detector=detector(sample_tilt=90.0, twist=90.0))
        profile, offs = _perpendicular_profile(pattern, segs)
        assert np.abs(offs[profile.argmax(axis=1)]).max() > 1.0
