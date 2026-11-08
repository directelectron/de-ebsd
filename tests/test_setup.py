"""The microscope set-up (:class:`de_ebsd.Setup`): one description of beam, sample
and screen, its fields for a user interface, the TEM screen, and the round trip
through a HyperSpy signal's metadata (:mod:`de_ebsd.io.hyperspy`).

Ported from SpyDE's ``test_phase_viewer`` (``TestSetup``, ``TestCalibrationInputs``).
"""

import dataclasses

import numpy as np
import pytest
import torch

from de_ebsd import EBSDDetector, Microscope, Setup, TEMDetector
from de_ebsd.sim.microscope import FIELDS


@pytest.mark.parametrize("kind", ["SEM", "TEM"])
def test_every_field_reads_back_what_was_written(kind):
    setup = Setup.default(kind, shape=(30, 40))
    assert setup.kind == kind and setup.detector.shape == (30, 40)
    assert setup.values() == {f.key: pytest.approx(f.default) for f in FIELDS[kind]}
    changed = {f.key: f.default + f.step for f in FIELDS[kind]}
    assert setup.replace(**changed).values() == pytest.approx(changed)
    assert set(setup.sources.values()) == {"default"}


def test_a_set_up_is_de_ebsds_microscope_and_detector():
    setup = Setup.default().replace(energy=15e3, sample_tilt=68.0, pcx=0.45, pcy=0.3, pcz=0.6)
    assert setup.microscope == Microscope(energy=15e3, sample_tilt=68.0)
    assert setup.detector.pattern_center == (0.45, 0.3, 0.6)
    assert setup.detector.sample_tilt == 68.0  # carried by both, kept equal
    tem = Setup.default("TEM").replace(sample_tilt=12.0)
    assert tem.microscope.sample_tilt == tem.detector.sample_tilt == 12.0
    assert tem.microscope.energy == 200e3


def test_a_set_up_refuses_a_microscope_and_detector_that_disagree():
    with pytest.raises(ValueError, match="sample_tilt must agree"):
        Setup(Microscope(sample_tilt=70.0), EBSDDetector(sample_tilt=60.0))
    with pytest.raises(TypeError, match="EBSDDetector or a TEMDetector"):
        Setup(Microscope(), object())
    with pytest.raises(KeyError, match="no SEM field"):
        Setup.default()["camera_length"]
    with pytest.raises(KeyError, match="tilt_beta"):
        Setup.from_values("SEM", {"tilt_beta": 1.0})


def test_the_sample_normal_sits_above_the_pattern_center():
    to_sample = Setup.default("SEM").detector_to_sample()
    normal_on_screen = to_sample.T @ np.array([0.0, 0.0, 1.0])
    assert normal_on_screen[1] > 0.3  # up, by sin(20 deg)
    np.testing.assert_allclose(np.degrees(np.arccos(normal_on_screen[2])), 20)


def test_the_sem_tilts_are_kikuchipys():
    """The four SEM angles follow kikuchipy, the reference for EBSD geometry."""
    kp = pytest.importorskip("kikuchipy")
    rng = np.random.default_rng(2)
    for angles in rng.uniform([0, -30, -30, -30], [90, 30, 30, 30], (20, 4)):
        theirs = kp.detectors.EBSDDetector(
            (60, 60),
            pc=(0.5, 0.5, 0.5),
            sample_tilt=angles[0],
            tilt=angles[1],
            azimuthal=angles[2],
            twist=angles[3],
        )
        ours = EBSDDetector((60, 60), (0.5, 0.5, 0.5), *angles)
        np.testing.assert_allclose(
            ours.detector_to_sample().T, theirs.sample_to_detector.to_matrix()[0], atol=1e-12
        )


def test_the_tem_screen_is_centered_on_the_beam_and_tilts_with_the_stage():
    detector = TEMDetector(shape=(5, 7), camera_length=200.0, size=14.0)
    np.testing.assert_allclose(detector.gnomonic_bounds, [-0.035, 0.035, -0.025, 0.025])
    rays = detector.direction_cosines(dtype=torch.float64).reshape(5, 7, 3).numpy()
    np.testing.assert_allclose(rays[2, 3], [0, 0, 1], atol=1e-15)  # the middle pixel
    assert rays[2, -1, 0] > 0 and rays[0, 3, 1] > 0  # x right, y up
    np.testing.assert_allclose(detector.pixels(np.array([0.0, 0.0])), [2.0, 3.0])
    np.testing.assert_allclose(detector.pixels([-0.035, 0.025]), [-0.5, -0.5])

    tilted = dataclasses.replace(detector, sample_tilt=10.0, tilt_beta=5.0)
    beam = tilted.detector_to_sample() @ [0.0, 0.0, 1.0]
    alpha = np.radians(10.0)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]]
    )
    beta = np.radians(5.0)
    ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    np.testing.assert_allclose(beam, rx @ ry @ [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        tilted.direction_cosines(dtype=torch.float64).reshape(5, 7, 3)[2, 3].numpy(), beam
    )


# --------------------------------------------------------------------------- #
# HyperSpy metadata
# --------------------------------------------------------------------------- #
@pytest.fixture
def hs():
    return pytest.importorskip("hyperspy.api")


def _sem_signal(hs, **metadata):
    signal = hs.signals.Signal2D(np.zeros((2, 2, 30, 40), np.float32))
    signal.metadata.set_item(
        "Acquisition_instrument.SEM.beam_energy", metadata.pop("beam_energy", 20.0)
    )
    for key, value in metadata.items():
        signal.metadata.set_item(key, value)
    return signal


def test_the_microscope_is_read_off_the_signal(hs):
    from de_ebsd.io.hyperspy import microscope_of

    assert microscope_of(_sem_signal(hs)) == "SEM"
    assert microscope_of(hs.signals.Signal2D(np.zeros((8, 8)))) == "TEM"


def test_values_come_from_metadata_in_de_ebsds_units_and_say_so(hs):
    from de_ebsd.io.hyperspy import read_setup

    setup = read_setup(_sem_signal(hs, beam_energy=15.0))
    assert setup.kind == "SEM" and setup["energy"] == 15e3  # kV in the metadata, eV here
    assert setup.detector.shape == (30, 40)
    assert setup.sources["energy"] == "metadata" and setup.sources["azimuthal"] == "default"
    tem = read_setup(hs.signals.Signal2D(np.zeros((8, 8))))
    assert tem.kind == "TEM" and tem["energy"] == 200e3


def test_a_recorded_detector_supplies_the_angles(hs):
    """A kikuchipy EBSD signal carries its detector: its angles and PC count as
    recorded with the data."""
    from de_ebsd.io.hyperspy import read_setup

    class Detector:
        sample_tilt, tilt, azimuthal, twist = 68.0, 3.0, 1.5, 0.5
        pc_average = np.array([0.45, 0.6, 0.58])

    signal = _sem_signal(hs)
    signal.detector = Detector()
    setup = read_setup(signal, fallback={"pcz": 0.7, "twist": 9.0})
    assert (setup["sample_tilt"], setup["tilt"], setup["azimuthal"]) == (68.0, 3.0, 1.5)
    assert setup.detector.pattern_center == (0.45, 0.6, 0.58) and setup["twist"] == 0.5
    assert setup.sources["twist"] == "data"


def test_writing_a_field_records_it_and_reads_back(hs):
    from de_ebsd.io.hyperspy import read_setup, write_field

    signal = _sem_signal(hs)
    write_field(signal, "SEM", "twist", 1.25)
    write_field(signal, "SEM", "energy", 12e3)
    assert signal.metadata.get_item("Acquisition_instrument.SEM.Detector.EBSD.twist") == 1.25
    assert signal.metadata.get_item("Acquisition_instrument.SEM.beam_energy") == 12.0
    setup = read_setup(signal)
    assert setup["twist"] == 1.25 and setup["energy"] == 12e3


def test_large_patterns_are_read_binned_to_the_same_screen(hs):
    from de_ebsd.index.refine.pattern_center_search import MAX_SIDE
    from de_ebsd.io.hyperspy import read_patterns

    signal = hs.signals.Signal2D(np.ones((2, 480, 640), np.float32))
    patterns = read_patterns(signal, [(0,), (1,)])
    assert patterns.shape == (2, 120, 160) and max(patterns.shape[1:]) <= MAX_SIDE
