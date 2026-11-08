"""Kinematic diffraction spots on a TEM screen
(:meth:`~de_ebsd.diffraction.kinematic.KinematicPatternModel.spots_on`).

Ported from SpyDE's ``test_fcc_001_zone_spot_rings``. SpyDE's spot model and
de-ebsd's differed; de-ebsd's won (a Lorentzian relrod of half-width 0.1 nm^-1, cut
off there), so the width that reaches {400} and {420} at 200 keV is asked for.
"""

import dataclasses

import numpy as np
import pytest

from de_ebsd import Crystal, TEMDetector
from de_ebsd.diffraction.kinematic import KinematicPatternModel


@pytest.fixture(scope="module")
def silver():
    crystal = Crystal.from_parameters(a=0.40862, space_group=225, atoms=[("Ag", (0, 0, 0))])
    return KinematicPatternModel(crystal, energy_kev=200.0, dmin=0.05)


@pytest.fixture(scope="module")
def screen():
    return TEMDetector(shape=(160, 160), camera_length=300.0, size=20.0)


def _radii(spots):
    return np.unique(np.round(np.hypot(*(spots.pixels - spots.direct_beam).T), 1))


def test_fcc_001_zone_spot_rings(silver, screen):
    """At [001] the spots are {200}, {220}, {400}, {420}: radii in the ratio
    1 : sqrt(2) : 2 : sqrt(5)."""
    spots = silver.spots_on(np.eye(3), screen, excitation_width=0.2)
    np.testing.assert_allclose(spots.direct_beam, [79.5, 79.5])
    radii = _radii(spots)
    np.testing.assert_allclose(radii[:4] / radii[0], [1, np.sqrt(2), 2, np.sqrt(5)], rtol=0.03)
    assert len(spots.pixels) == 20 and spots.intensities.max() == 1.0
    assert (np.abs(spots.hkl[:, 2]) == 0).all()  # the zero-order Laue zone


def test_the_default_relrod_stops_short_of_400(silver, screen):
    """The visible difference from SpyDE's Gaussian relrod (scale 0.2 nm^-1, drawn
    down to 2 %): at the default width only {200} and {220} are drawn."""
    spots = silver.spots_on(np.eye(3), screen)
    assert len(_radii(spots)) == 2 and len(spots.pixels) == 8


def _centroid(spots):
    weights = spots.intensities[:, None]
    return (weights * (spots.pixels - spots.direct_beam)).sum(0) / weights.sum()


def test_a_stage_tilt_moves_the_laue_circle_across_the_screen(silver, screen):
    """Alpha (about the screen's x) moves the excited spots along the rows, beta
    (about its y) along the columns; the screen stays on the beam."""
    flat = silver.spots_on(np.eye(3), screen, excitation_width=0.2)
    np.testing.assert_allclose(_centroid(flat), 0.0, atol=1e-9)
    for tilt, axis in (("sample_tilt", 0), ("tilt_beta", 1)):
        ways = []
        for degrees in (0.5, -0.5):
            tilted = dataclasses.replace(screen, **{tilt: degrees})
            spots = silver.spots_on(np.eye(3), tilted, excitation_width=0.2)
            np.testing.assert_allclose(spots.direct_beam, flat.direct_beam)
            shift = _centroid(spots)
            assert abs(shift[axis]) > 5 and abs(shift[1 - axis]) < 1e-6, (tilt, shift)
            ways.append(np.sign(shift[axis]))
        assert ways[0] == -ways[1]
