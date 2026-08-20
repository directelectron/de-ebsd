"""Display layer (anyplotlib) and setup widget (anywidget) smoke tests.

These run headless: figures are built and serialized to HTML, widgets are
instantiated and their traitlets exercised; no browser is required.
"""

import numpy as np
import pytest

import torchebsd as te


@pytest.fixture(scope="module")
def sim():
    crystal = te.Crystal.from_parameters(
        a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))], debye_waller=0.0035
    )
    return te.MasterPatternSimulator(crystal, te.Microscope(energy=20e3), npx=24, dmin=0.12)


def test_structure_factors_show(sim):
    pytest.importorskip("anyplotlib")
    figure = sim.structure_factors().show(n=10)
    html = figure.to_html()
    assert isinstance(html, str) and len(html) > 100


def test_reflections_show(sim):
    pytest.importorskip("anyplotlib")
    assert sim.reflections().show(n=12).to_html()
    assert sim.reflections(k=(0, 0, 1)).show(n=12).to_html()


def test_dynamical_matrix_show(sim):
    pytest.importorskip("anyplotlib")
    assert sim.dynamical_matrix(k=(1, 1, 3)).show().to_html()


def test_master_pattern_show_projections(sim):
    pytest.importorskip("anyplotlib")
    master = sim.compute()
    for projection in ("lambert", "stereographic", "sphere"):
        html = master.show(projection=projection).to_html()
        assert isinstance(html, str) and len(html) > 100
    with pytest.raises(ValueError, match="projection"):
        master.show(projection="gnomonic")


def test_to_uint8_rgb():
    from torchebsd.display import to_uint8_rgb

    image = np.array([[0.0, 1.0], [2.0, 4.0]])
    rgb = to_uint8_rgb(image)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] == 0 and rgb[1, 1, 0] == 255


def test_setup_widget_round_trip():
    pytest.importorskip("anywidget")
    from torchebsd.widgets import SetupWidget

    microscope = te.Microscope(energy=15e3, sample_tilt=68.0)
    detector = te.EBSDDetector(shape=(240, 320), pc=(0.4, 0.6, 0.55), tilt=6.0, sample_tilt=68.0)
    widget = SetupWidget.from_setup(microscope=microscope, detector=detector)

    assert widget.energy_kev == pytest.approx(15.0)
    assert widget.sample_tilt == pytest.approx(68.0)
    assert (widget.detector_rows, widget.detector_cols) == (240, 320)

    widget.sample_tilt = 71.5
    widget.pcz = 0.62
    scope2 = widget.to_microscope()
    det2 = widget.to_detector()
    assert scope2.sample_tilt == pytest.approx(71.5)
    assert scope2.energy == pytest.approx(15e3)
    assert det2.pc[2] == pytest.approx(0.62)
    assert det2.shape == (240, 320)


def test_simulator_show_setup(sim):
    pytest.importorskip("anywidget")
    widget = sim.show_setup(detector=te.EBSDDetector())
    assert "Ni" in widget.crystal_label
