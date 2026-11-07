"""A :class:`~de_ebsd.sim.microscope.Setup` read from, and written to, a HyperSpy
signal's metadata, and a scan's patterns read a few at a time.

Optional: nothing here imports HyperSpy, but every function expects a HyperSpy
(or kikuchipy) signal, so it is only useful where HyperSpy is installed.

Each field of a set-up names the metadata key it is kept under
(:data:`METADATA_KEYS`), in HyperSpy's conventions -- beam energy in kV, the
pattern center as kikuchipy's ``pc_x``/``pc_y``/``pc_z``. The set-up itself is in
de-ebsd's units (eV); the conversion happens here, once.
"""

from __future__ import annotations

import numpy as np

from de_ebsd.index.preprocess import bin_pattern
from de_ebsd.index.refine.pattern_center_search import MAX_SIDE
from de_ebsd.sim.microscope import FIELDS, Setup

__all__ = [
    "METADATA_KEYS",
    "microscope_of",
    "read_patterns",
    "read_setup",
    "recorded_detector",
    "write_field",
]

_EBSD = "Acquisition_instrument.SEM.Detector.EBSD."

#: Where each field of a set-up is kept in a signal's metadata, and the factor from
#: the metadata's unit to de-ebsd's: ``{kind: {key: (metadata key, scale)}}``.
METADATA_KEYS: dict[str, dict[str, tuple[str, float]]] = {
    "SEM": {
        "energy": ("Acquisition_instrument.SEM.beam_energy", 1e3),
        "sample_tilt": ("Acquisition_instrument.SEM.Stage.tilt_alpha", 1.0),
        "tilt": (_EBSD + "tilt", 1.0),
        "azimuthal": (_EBSD + "azimuthal", 1.0),
        "twist": (_EBSD + "twist", 1.0),
        "pcx": (_EBSD + "pc_x", 1.0),
        "pcy": (_EBSD + "pc_y", 1.0),
        "pcz": (_EBSD + "pc_z", 1.0),
    },
    "TEM": {
        "energy": ("Acquisition_instrument.TEM.beam_energy", 1e3),
        "camera_length": ("Acquisition_instrument.TEM.camera_length", 1.0),
        "size": ("Acquisition_instrument.TEM.Detector.Diffraction.size", 1.0),
        "sample_tilt": ("Acquisition_instrument.TEM.Stage.tilt_alpha", 1.0),
        "tilt_beta": ("Acquisition_instrument.TEM.Stage.tilt_beta", 1.0),
    },
}


def microscope_of(signal) -> str:
    """``"SEM"`` for an EBSD scan or a signal with SEM metadata, else ``"TEM"``."""
    try:
        if "EBSD" in str(getattr(signal, "_signal_type", "") or "").upper():
            return "SEM"
        if signal.metadata.has_item("Acquisition_instrument.SEM"):
            return "SEM"
    except AttributeError:
        pass
    return "TEM"


def recorded_detector(signal) -> dict[str, float]:
    """The SEM detector geometry recorded with a scan, by field key: a kikuchipy EBSD
    signal carries its ``EBSDDetector``. Empty when there is none."""
    detector = getattr(signal, "detector", None)
    if detector is None:
        return {}
    try:
        pc = np.asarray(detector.pc_average, dtype=float).reshape(3)
        return {
            "sample_tilt": float(detector.sample_tilt),
            "tilt": float(detector.tilt),
            "azimuthal": float(detector.azimuthal),
            "twist": float(detector.twist),
            "pcx": float(pc[0]),
            "pcy": float(pc[1]),
            "pcz": float(pc[2]),
        }
    except (AttributeError, TypeError, ValueError):
        return {}


def read_setup(signal, *, fallback=None) -> Setup:
    """The set-up recorded on *signal*, at its pattern shape.

    Each field is, in order: the metadata value, the value recorded with the scan
    (:func:`recorded_detector`, then *fallback*, field values by key -- the pattern
    center stamped on synthetic data, say), then the field's default.
    ``Setup.sources`` says which: ``"metadata"``, ``"data"`` or ``"default"``.
    """
    kind = microscope_of(signal)
    recorded = {**(fallback or {}), **recorded_detector(signal)}
    values, sources = {}, {}
    for field in FIELDS[kind]:
        path, scale = METADATA_KEYS[kind][field.key]
        try:
            stored = signal.metadata.get_item(path, None)
        except AttributeError:
            stored = None
        if stored is not None:
            values[field.key], sources[field.key] = float(stored) * scale, "metadata"
        elif field.key in recorded:
            values[field.key], sources[field.key] = float(recorded[field.key]), "data"
        else:
            sources[field.key] = "default"
    return Setup.from_values(kind, values, shape=_signal_shape(signal), sources=sources)


def write_field(signal, kind: str, key: str, value: float) -> None:
    """Record one field of a *kind* set-up, in de-ebsd's units, in *signal*'s metadata."""
    path, scale = METADATA_KEYS[kind][key]
    signal.metadata.set_item(path, float(value) / scale)


def read_patterns(signal, positions, max_side: int = MAX_SIDE) -> np.ndarray:
    """The frames at navigation *positions* (HyperSpy's ``inav`` order), as
    ``(N, rows, cols)`` float32, binned to at most *max_side* on a side
    (:func:`~de_ebsd.index.preprocess.bin_pattern`).

    Reads one frame at a time, never the whole scan: ``inav`` of a lazy signal is a
    lazy one-frame slice. The position ``()`` is the signal itself.
    """
    frames = []
    for position in positions:
        frame = signal.inav[position] if position else signal
        frames.append(bin_pattern(np.asarray(frame.data, dtype=np.float32), max_side))
    return np.stack(frames)


def _signal_shape(signal):
    try:
        return tuple(int(n) for n in signal.axes_manager.signal_shape[::-1])
    except AttributeError:
        return None
