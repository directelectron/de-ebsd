"""de-ebsd's sample frame is OIM's (EDAX TSL): its orientations go out unturned."""

import warnings

import numpy as np
import pytest
from orix.quaternion import Rotation

from de_ebsd.core import frames
from de_ebsd.core.rotation import euler_to_matrix, matrix_to_euler

EULER = np.radians([[35.0, 48.0, 112.0]])


def test_an_orientation_is_written_as_the_indexer_returns_it():
    """OIM's Euler angles are the matrix's own, with nothing turned between."""
    g = euler_to_matrix(EULER)
    assert np.allclose(matrix_to_euler(g), EULER, atol=1e-12)
    assert np.allclose(Rotation.from_euler(matrix_to_euler(g)).to_matrix(), g, atol=1e-12)


def test_the_turn_is_gone():
    """The old 90 degree turn about the surface normal is not applied anywhere."""
    assert not hasattr(frames, "SCAN_TURN")
    np.random.seed(3)
    g = Rotation.random(4).to_matrix()
    stretch = np.diag([1e-3, -1e-3, 0.0])
    with pytest.warns(DeprecationWarning, match="OIM's"):
        assert np.allclose(euler_to_matrix(frames.scan_euler(g)), g, atol=1e-9)
    with pytest.warns(DeprecationWarning, match="OIM's"):
        assert np.allclose(frames.to_scan_frame(EULER), EULER)
    with pytest.warns(DeprecationWarning, match="OIM's"):
        assert np.allclose(frames.scan_tensor(stretch), stretch)


def test_the_deprecated_names_still_import_from_the_pipeline():
    from de_ebsd.pipeline.orientation import indexing

    assert indexing.scan_euler is frames.scan_euler
    assert indexing.to_scan_frame is frames.to_scan_frame
    assert indexing.scan_tensor is frames.scan_tensor
    assert not hasattr(indexing, "SCAN_TURN")


def test_a_match_s_orientation_is_its_euler_angles():
    from de_ebsd.pipeline.orientation.indexing import Match

    match = Match(0, EULER[0], 0.5)
    with pytest.warns(DeprecationWarning, match="Match.euler"):
        assert match.orientation is match.euler
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.allclose(match.euler, EULER[0])
