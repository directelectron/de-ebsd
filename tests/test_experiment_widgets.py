"""Scientific geometry and shared gallery/notebook inspectors."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import traitlets

from de_ebsd import DepthDistribution, EBSDDetector, Microscope


def test_probe_footprint_and_roundtrip():
    microscope = Microscope(probe_diameter_nm=10)
    assert microscope.probe_footprint_nm == pytest.approx((29.238044, 10))
    widget = microscope.show()
    assert widget.to_microscope() == microscope
    widget.probe_diameter_nm = 20
    assert widget.to_microscope().probe_footprint_nm == pytest.approx((58.476088, 20))
    assert Microscope().probe_footprint_nm is None
    assert Microscope(sample_tilt=90, probe_diameter_nm=10).probe_footprint_nm is None
    with pytest.raises(ValueError, match="probe_diameter"):
        Microscope(probe_diameter_nm=float("nan"))
    with pytest.raises(traitlets.TraitError):
        widget.pcz = 0
    with pytest.raises(traitlets.TraitError):
        widget.pcx = float("inf")


def test_detector_physical_geometry_matches_projection():
    detector = EBSDDetector(shape=(4, 6), pattern_center=(.45, .55, .6), px_size=400,
                            sample_tilt=57.4, tilt=22.7, azimuthal=13, twist=9)
    corners = detector.corners_sample_mm
    assert np.linalg.norm(corners[1] - corners[0]) == pytest.approx(2.4)
    assert np.linalg.norm(corners[3] - corners[0]) == pytest.approx(1.6)
    assert np.linalg.norm(detector.pattern_center_sample_mm) == pytest.approx(.96)
    # Interpolate each pixel center on the physical screen; its unit vector
    # must equal the one the diffraction projection actually uses.
    centers = np.array([
        corners[0] + (j+.5)/6*(corners[1]-corners[0]) + (i+.5)/4*(corners[3]-corners[0])
        for i in range(4) for j in range(6)
    ])
    directions = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    assert np.allclose(directions, detector.direction_cosines(dtype=torch.float64).numpy())


@pytest.mark.parametrize("angles", [(70, 0, 0, 0), (57.4, 22.7, 0, 0), (64, -8, 17, 23)])
def test_browser_geometry_matches_python(angles):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; required only for renderer geometry parity")
    sigma, tilt, azimuth, twist = angles
    detector = EBSDDetector(shape=(80, 100), pattern_center=(.45, .55, .6), px_size=400,
                            sample_tilt=sigma, tilt=tilt, azimuthal=azimuth, twist=twist)
    widget = detector.show()
    state = {key: getattr(widget, key) for key in (
        "sample_tilt", "detector_tilt", "azimuthal", "twist", "detector_rows",
        "detector_cols", "pcx", "pcy", "pcz", "px_size", "probe_diameter_nm",
    )}
    from de_ebsd.viz import widgets

    module = Path(widgets.__file__).with_name("setup_widget.js").as_uri()
    code = (f"import {{setupGeometry}} from {json.dumps(module)};"
            f"console.log(JSON.stringify(setupGeometry({json.dumps(state)})));")
    result = subprocess.run([node, "--input-type=module", "-e", code],
                            capture_output=True, text=True, check=True)
    geometry = json.loads(result.stdout)
    assert np.allclose(geometry["corners"], detector.corners_sample_mm)
    assert np.allclose(geometry["pc"], detector.pattern_center_sample_mm)
    # Regression for the backwards-leaning sample in the previous schematic.
    assert geometry["tangent"][2] < 0
    assert geometry["normal"][0] > 0
    if tilt == azimuth == twist == 0:
        assert geometry["pcLab"][2] == pytest.approx(0, abs=1e-12)


def test_geometry_preserves_angles_and_rejects_mismatch():
    microscope = Microscope(sample_tilt=69, azimuthal=12, probe_diameter_nm=5)
    detector = EBSDDetector(sample_tilt=69, azimuthal=12, twist=8, tilt=4)
    widget = detector.show(microscope=microscope)
    assert widget.to_detector() == detector
    assert widget.to_microscope() == microscope
    with pytest.raises(ValueError, match="must agree"):
        EBSDDetector(sample_tilt=65).show(microscope=microscope)


def test_depth_widget_uses_real_profile_and_cutoff():
    depth = DepthDistribution.single_energy(20, thickness=40, depth_step=2, decay_length=15)
    widget = depth.show()
    assert widget.integration_bins == depth.integration_depths().tolist()
    assert widget.integration_bins == [20]
    assert np.asarray(widget.profiles).sum() == pytest.approx(1)
    widget.slice_index = 4
    assert depth.n_depths == 20  # inspecting never truncates the source
    assert "PRISM propagation point" in widget.to_html()
    assert 'application/vnd.jupyter.widget-view+json' in widget._repr_mimebundle_()[0]


def test_energy_selection_and_empty_energy():
    depth = DepthDistribution(np.array([15., 20.]), 2.,
                              np.array([0, 0, 0, 0, 1, 2, 1, 0.]).reshape(2, 4, 1, 1), 4)
    widget = depth.show(energy_kev=15)
    assert widget.energy_index == 0
    assert widget.profiles[0] == [0, 0, 0, 0]
    assert widget.to_html()
    assert depth.show().energy_index == 1


def test_portable_html_is_self_contained_and_escapes_labels():
    widget = Microscope().show()
    widget.crystal_label = '</script><script>alert("bad")</script>'
    html = widget.to_html()
    assert widget.crystal_label not in html
    assert "setupGeometry" in html
    assert "cdn.jsdelivr" not in html
    assert 'application/vnd.jupyter.widget-view+json' in widget._repr_mimebundle_()[0]
