"""Display layer (anyplotlib) and setup widget (anywidget) smoke tests.

These run headless: figures are built and serialized to HTML, widgets are
instantiated and their traitlets exercised; no browser is required.
"""

import numpy as np
import pytest

import de_ebsd as de


@pytest.fixture(scope="module")
def sim():
    crystal = de.Crystal.from_parameters(
        a=0.35236, space_group=225, atoms=[("Ni", (0.0, 0.0, 0.0))], debye_waller=0.0035
    )
    return de.MasterPatternSimulator(crystal, de.Microscope(energy=20e3), npx=24, dmin=0.12)


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
    from de_ebsd.viz.figures import to_uint8_rgb

    image = np.array([[0.0, 1.0], [2.0, 4.0]])
    rgb = to_uint8_rgb(image)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] == 0 and rgb[1, 1, 0] == 255


def test_setup_widget_round_trip():
    pytest.importorskip("anywidget")
    from de_ebsd.viz.widgets import SetupWidget

    microscope = de.Microscope(energy=15e3, sample_tilt=68.0)
    detector = de.EBSDDetector(shape=(240, 320), pc=(0.4, 0.6, 0.55), tilt=6.0, sample_tilt=68.0)
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


def test_crystal_show_is_native_anyplotlib_figure(sim):
    pytest.importorskip("anyplotlib")
    crystal = sim.crystal
    figure = crystal.show()
    assert figure.to_html()


def test_phase_overview_composes_native_figures_without_nested_html(sim):
    pytest.importorskip("anyplotlib")
    crystal = sim.crystal
    overview = crystal.overview()
    assert overview.metadata["space_group"] == 225
    assert overview.metadata["atom_count"] == 4
    assert overview.metadata["volume_nm3"] == pytest.approx(crystal.volume)
    assert set(overview.figures) == {"phase_overview"}
    assert len(overview.figure.get_axes()) == 1

    master = sim.compute()
    linked = crystal.overview(master_pattern=master)
    assert set(linked.figures) == {"phase_overview"}
    assert len(linked.figure.get_axes()) == 2
    assert all(figure.to_html() for figure in linked.figures.values())
    assert linked.metadata["simulation_backend"] in {"bloch_prism", "kinematical", "computed"}
    assert "html" not in " ".join(linked.metadata).lower()
    with pytest.raises(ValueError, match="texture_height"):
        crystal.overview(master_pattern=master, texture_height=4)


def test_compact_microscope_widget_round_trip():
    pytest.importorskip("anywidget")
    from de_ebsd.viz.widgets import MicroscopeWidget

    microscope = de.Microscope(energy=18e3, sample_tilt=69.0, azimuthal=2.5)
    detector = de.EBSDDetector(
        shape=(256, 384), pc=(0.42, 0.58, 0.61), tilt=4.0, px_size=0.75,
        sample_tilt=69.0, azimuthal=2.5,
    )
    widget = MicroscopeWidget.from_setup(microscope=microscope, detector=detector)
    assert widget.energy_kev == pytest.approx(18.0)
    assert widget.px_size == pytest.approx(0.75)
    widget.pcz = 0.66
    assert widget.to_microscope().energy == pytest.approx(18e3)
    assert widget.to_detector().pc[2] == pytest.approx(0.66)


def test_sampling_sphere_show():
    pytest.importorskip("anyplotlib")
    from de_ebsd.diffraction.kvectors import sample_incident_directions

    grid = sample_incident_directions(4)
    html = grid.show().to_html()
    assert isinstance(html, str) and len(html) > 100


def test_simulator_show_setup(sim):
    pytest.importorskip("anywidget")
    widget = sim.show_setup(detector=de.EBSDDetector())
    assert "Ni" in widget.crystal_label
