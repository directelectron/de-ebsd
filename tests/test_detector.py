"""Detector geometry: direction cosines, kikuchipy convention parity."""

import numpy as np
import pytest
import torch

from de_ebsd import EBSDDetector


def test_direction_cosines_shape_and_norm():
    det = EBSDDetector(shape=(48, 64), pattern_center=(0.45, 0.55, 0.6), sample_tilt=70.0, tilt=8.0)
    dc = det.direction_cosines(dtype=torch.float64)
    assert dc.shape == (48 * 64, 3)
    assert torch.allclose(dc.norm(dim=-1), torch.ones(len(dc), dtype=torch.float64), atol=1e-12)


def test_gnomonic_bounds_center_pc():
    det = EBSDDetector(shape=(100, 100), pattern_center=(0.5, 0.5, 0.5))
    x_min, x_max, y_min, y_max = det.gnomonic_bounds
    assert x_min == pytest.approx(-1.0)
    assert x_max == pytest.approx(1.0)
    assert y_min == pytest.approx(-1.0)
    assert y_max == pytest.approx(1.0)


def test_matches_kikuchipy_direction_cosines():
    kp = pytest.importorskip("kikuchipy")
    shape, pattern_center = (37, 41), (0.42, 0.58, 0.64)
    sample_tilt, tilt, azimuthal = 70.0, 7.5, 3.0

    ours = EBSDDetector(
        shape=shape, pattern_center=pattern_center, sample_tilt=sample_tilt, tilt=tilt, azimuthal=azimuthal
    ).direction_cosines(dtype=torch.float64).numpy()

    from kikuchipy.signals.util._master_pattern import _get_direction_cosines_from_detector

    kp_det = kp.detectors.EBSDDetector(
        shape=shape, pc=pattern_center, sample_tilt=sample_tilt, tilt=tilt, azimuthal=azimuthal
    )
    theirs = _get_direction_cosines_from_detector(kp_det)
    assert np.allclose(ours, theirs, atol=1e-10)
