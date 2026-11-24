"""The live fit against a reference (:mod:`de_ebsd.index.live.reference`).

Test patterns are rendered straight from a band master with the lattice
deformed (``test_hrebsd._render``), so the ground truth does not share the
fit's projection. The fit is the fused CUDA kernel's: these need a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_hrebsd import _BETA, _render, _rotation
from test_reference_fit import _deviatoric_stretch

from band_master import make_band_master
from de_ebsd import EBSDDetector
from de_ebsd.core.geometry import sample_to_detector_tensor
from de_ebsd.index import _kernels
from de_ebsd.index.live.reference import StrainReference, fit_shape

pytestmark = pytest.mark.skipif(not _kernels.available("cuda"), reason="needs CUDA and CuPy")

DT = torch.float64
SIDE = 256


@pytest.fixture(scope="module")
def master():
    return make_band_master(npx=600, width=0.03, n_bands=24, seed=11)


@pytest.fixture(scope="module")
def detector():
    return EBSDDetector(
        shape=(SIDE, SIDE), pattern_center=(0.5, 0.5, 0.6), sample_tilt=70.0, tilt=0.0
    )


def _envelope(dx=0.0):
    """A backscatter envelope, as the camera's: the fit divides it out."""
    y, x = torch.meshgrid(torch.arange(SIDE, dtype=DT), torch.arange(SIDE, dtype=DT), indexing="ij")
    return 1 + torch.exp(-((x - 140 - dx) ** 2 + (y - 90) ** 2) / (2 * 120.0**2))


def _in_sample_frame(turn, detector):
    om = sample_to_detector_tensor(detector, "cpu", DT)
    return om.T @ turn @ om


@pytest.mark.parametrize(("deg", "tolerance"), [(0.0, 1e-4), (2.0, 4e-4)])
def test_a_strain_comes_back_seeded_from_the_orientations(master, detector, deg, tolerance):
    """A 2e-3 strain on a lattice turned by *deg*, seeded with the turn the
    two points' orientations give: the strain and the turn come back. Each
    pattern is flattened by its own blur, which moves a little with the
    lattice: measured, 0.4e-4 off with no turn and 2.7e-4 at 2 degrees."""
    turn = _rotation([0.2, 0.9, -0.4], np.deg2rad(deg))
    f = turn @ (torch.eye(3, dtype=DT) + 2e-3 * _BETA)
    ref = _envelope() * (4 + _render(master, detector))
    test = _envelope(6) * (4 + _render(master, detector, f))
    # orientations take sample to crystal: the reference's the identity, the
    # test's the inverse of its lattice's turn in the sample frame
    turned = _in_sample_frame(turn, detector)
    reference = StrainReference(ref.float(), detector, rotation=np.eye(3), symmetry=np.eye(3)[None])
    found = reference.fit(test.float()[None], rotations=turned.T[None].numpy())
    want = _deviatoric_stretch(f, detector)
    assert want.abs().max() > 1e-3
    error = (found.strain[0].double().cpu() - want).abs().max()
    assert error < tolerance, error
    # the deformation's own rotation, its polar part, in the sample frame
    u, _, vt = torch.linalg.svd(f)
    om = sample_to_detector_tensor(detector, "cpu", DT)
    rotation_vector = Rotation.from_matrix((om.T @ u @ vt @ om).numpy()).as_rotvec()
    error = np.abs(found.rotation_vector[0].cpu().numpy() - rotation_vector).max()
    assert error < 2e-4, error
    assert found.score[0] > 0.9


def test_unstrained_patterns_at_another_center_have_no_strain(master, detector):
    """The test's pattern center differs from the reference's: told so, the fit
    finds the geometry, not a strain; not told, it finds a strain."""
    center = (0.506, 0.495, 0.604)
    ref = 4 + _render(master, detector)
    test = 4 + _render(master, detector, pattern_center=center)
    reference = StrainReference(ref.float(), detector)
    told = reference.fit(test.float()[None], pattern_centers=[center])
    assert told.strain.abs().max() < 3e-4, told.strain.abs().max()
    assert told.rotation_vector.abs().max() < 3e-4
    untold = reference.fit(test.float()[None])
    assert untold.strain.abs().max() > 1e-3


def test_frames_are_binned_to_the_fit_shape():
    assert fit_shape((2048, 2048)) == (1024, 1024)
    assert fit_shape((1024, 1024)) == (1024, 1024)
    assert fit_shape((512, 640)) == (512, 640)
    assert fit_shape((2048, 2048), most=500) == (256, 256)


def test_a_seed_without_the_reference_orientation_is_refused(master, detector):
    reference = StrainReference((4 + _render(master, detector)).float(), detector)
    with pytest.raises(ValueError, match="rotation and symmetry"):
        reference.fit(torch.ones(1, SIDE, SIDE), rotations=np.eye(3)[None])
