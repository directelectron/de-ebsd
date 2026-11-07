"""Microscope: beam energy and sample geometry, and the set-up it makes with a detector.

The :class:`Microscope` collects the experimental parameters that belong to
the instrument rather than to the crystal or the detector: accelerating
voltage and the sample tilt. Detector geometry lives on
:class:`de_ebsd.sim.detector.EBSDDetector` (an SEM's EBSD screen) or
:class:`de_ebsd.sim.detector.TEMDetector` (a TEM's diffraction screen).

A :class:`Setup` is the two together -- one description of where the beam, the
sample and the screen are -- and :data:`FIELDS` names its editable numbers, with
labels, units and ranges, for a user interface to draw.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from de_ebsd.core.constants import (
    interaction_constant,
    relativistic_correction,
)

__all__ = ["FIELDS", "Field", "Microscope", "Setup"]


@dataclass(frozen=True)
class Microscope:
    """An SEM configured for EBSD, or the beam of a TEM.

    Parameters
    ----------
    energy
        Beam energy in eV (e.g. ``20e3``).
    sample_tilt
        Sample tilt from the horizontal in degrees (70 is standard for EBSD; a
        TEM's stage tilt alpha).
    azimuthal
        Detector azimuthal angle with respect to the tilt axis, in degrees.
    probe_diameter_nm
        Nominal circular beam diameter perpendicular to the beam, in nm.
        Optional geometry metadata, not a finite-probe scattering model.
    """

    energy: float = 20e3
    sample_tilt: float = 70.0
    azimuthal: float = 0.0
    probe_diameter_nm: float | None = None

    def __post_init__(self) -> None:
        if self.energy <= 0:
            raise ValueError("energy must be positive (in eV)")
        if self.probe_diameter_nm is not None and (
            not math.isfinite(self.probe_diameter_nm) or self.probe_diameter_nm <= 0
        ):
            raise ValueError("probe_diameter_nm must be finite and positive")

    @property
    def probe_footprint_nm(self) -> tuple[float, float] | None:
        """Major/minor diameters on a flat surface; None at grazing incidence.

        Geometric projection of a circular parallel probe, not an interaction
        volume, beam broadening prediction, or a measured spatial resolution.
        """
        cosine = math.cos(math.radians(self.sample_tilt))
        if self.probe_diameter_nm is None or cosine <= 1e-12:
            return None
        return self.probe_diameter_nm / cosine, self.probe_diameter_nm

    @property
    def energy_kev(self) -> float:
        """Beam energy in keV."""
        return self.energy / 1e3

    @property
    def gamma(self) -> float:
        """Relativistic correction factor 1 + e V / (m0 c^2)."""
        return relativistic_correction(self.energy_kev)

    @property
    def sigma(self) -> float:
        """Interaction constant in (V nm)^-1."""
        return interaction_constant(self.energy_kev, v0mod=0.0)

    def show(self, detector=None, crystal=None):
        """Editable beam, footprint and optional detector geometry inspector.

        When supplied, detector sample tilt/azimuth must agree with this object.
        Edits are a separate snapshot; read them with widget.to_microscope().
        """
        from de_ebsd.viz.widgets import MicroscopeWidget

        return MicroscopeWidget.from_setup(microscope=self, detector=detector, crystal=crystal)

    def __repr__(self) -> str:
        return (
            f"Microscope(energy={self.energy:g} eV, sample_tilt={self.sample_tilt:g} deg, "
            f"azimuthal={self.azimuthal:g} deg)"
        )


@dataclass(frozen=True)
class Field:
    """One editable number of a :class:`Setup`, described for a user interface.

    Parameters
    ----------
    key
        The name a :class:`Setup` is indexed and replaced by.
    label, unit
        What to call it and its unit (``""`` for a fraction of the screen).
    default
        Its value in :meth:`Setup.default`.
    step
        A sensible increment for a spin box.
    minimum, maximum
        Its range; ``None`` for unbounded.
    attribute, index
        Where it lives: the attribute of the microscope and/or the detector it is
        written to (*key* when empty), and the component of a tuple attribute
        (``pattern_center``) when not ``None``.
    """

    key: str
    label: str
    unit: str
    default: float
    step: float = 1.0
    minimum: float | None = None
    maximum: float | None = None
    attribute: str = ""
    index: int | None = None

    @property
    def target(self) -> str:
        """The attribute the field is written to."""
        return self.attribute or self.key


def _angle(key: str, label: str, default: float = 0.0) -> Field:
    return Field(key, label, "deg", default, 0.5, -90.0, 90.0)


#: The editable numbers of an SEM's and a TEM's set-up, in de-ebsd's units and
#: names: energy in eV, angles in degrees, the pattern center as the three
#: components of ``EBSDDetector.pattern_center`` (Bruker), lengths in mm.
FIELDS: dict[str, tuple[Field, ...]] = {
    "SEM": (
        Field("energy", "Beam energy", "eV", 20e3, 500.0, 1e3, 60e3),
        Field("sample_tilt", "Sample tilt", "deg", 70.0, 0.5, 0.0, 90.0),
        _angle("tilt", "Detector tilt"),
        _angle("azimuthal", "Azimuthal"),
        _angle("twist", "Twist"),
        Field("pcx", "PC x", "", 0.5, 0.005, -1.0, 2.0, "pattern_center", 0),
        Field("pcy", "PC y", "", 0.5, 0.005, -1.0, 2.0, "pattern_center", 1),
        Field("pcz", "PC z", "", 0.55, 0.005, 0.01, 5.0, "pattern_center", 2),
    ),
    "TEM": (
        Field("energy", "Beam energy", "eV", 200e3, 10e3, 1e3, 1.5e6),
        Field("camera_length", "Camera length", "mm", 300.0, 10.0, 1.0, 1e5),
        Field("size", "Detector width", "mm", 20.0, 1.0, 0.1, 1e3),
        _angle("sample_tilt", "Stage tilt alpha"),
        _angle("tilt_beta", "Stage tilt beta"),
    ),
}

#: The attributes a microscope and a detector can both carry; a set-up keeps them equal.
_SHARED = ("sample_tilt", "azimuthal")


def _detector_class(kind: str):
    from de_ebsd.sim.detector import EBSDDetector, TEMDetector

    try:
        return {"SEM": EBSDDetector, "TEM": TEMDetector}[kind]
    except KeyError:
        raise ValueError(f"kind must be 'SEM' or 'TEM', got {kind!r}") from None


@dataclass(frozen=True)
class Setup:
    """A microscope and its detector: where the beam, the sample and the screen are.

    An :class:`~de_ebsd.sim.detector.EBSDDetector` makes an SEM's set-up and a
    :class:`~de_ebsd.sim.detector.TEMDetector` a TEM's. The sample tilt (and the
    azimuthal angle, where the detector has one) is carried by both the microscope
    and the detector, as :class:`Microscope` and the detector already do; a set-up
    holds them equal.

    Its :attr:`fields` are the numbers a user edits: ``setup["pcx"]`` reads one,
    :meth:`replace` writes some, :meth:`values` reads them all.

    Parameters
    ----------
    microscope
        The :class:`Microscope`.
    detector
        The detector.
    sources
        Where each field's value came from, by key (such as ``"metadata"``,
        ``"data"`` or ``"default"``), for a reader to say which are guesses. Not
        part of the set-up's identity.
    """

    microscope: Microscope
    detector: object
    sources: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        self.kind  # noqa: B018 - checks the detector's type
        for name in _SHARED:
            if hasattr(self.detector, name) and not math.isclose(
                getattr(self.microscope, name), getattr(self.detector, name), abs_tol=1e-9
            ):
                raise ValueError(f"microscope and detector {name} must agree")

    @property
    def kind(self) -> str:
        """``"SEM"`` or ``"TEM"``, by the detector."""
        from de_ebsd.sim.detector import EBSDDetector, TEMDetector

        if isinstance(self.detector, EBSDDetector):
            return "SEM"
        if isinstance(self.detector, TEMDetector):
            return "TEM"
        raise TypeError(
            f"detector must be an EBSDDetector or a TEMDetector, got {type(self.detector)}"
        )

    @property
    def fields(self) -> tuple[Field, ...]:
        """This kind's :data:`FIELDS`."""
        return FIELDS[self.kind]

    @classmethod
    def default(cls, kind: str = "SEM", shape=None) -> Setup:
        """Every field at its default; *shape* the detector's, else its class's."""
        return cls.from_values(kind, {}, shape=shape)

    @classmethod
    def from_values(
        cls, kind: str, values: Mapping[str, float], *, shape=None, sources=None
    ) -> Setup:
        """A set-up of *kind* from field values by key, each one missing at its default.

        *sources* defaults to ``"set"`` for the fields given and ``"default"`` for the
        others.
        """
        detector = _detector_class(kind)()
        unknown = set(values) - {f.key for f in FIELDS[kind]}
        if unknown:
            raise KeyError(f"no {kind} field(s) {sorted(unknown)}")
        if shape is not None:
            detector = dataclasses.replace(detector, shape=tuple(int(n) for n in shape))
        microscope = Microscope(sample_tilt=detector.sample_tilt)
        merged = {f.key: values.get(f.key, f.default) for f in FIELDS[kind]}
        if sources is None:
            sources = {f.key: "set" if f.key in values else "default" for f in FIELDS[kind]}
        setup = cls(microscope, detector).replace(**merged)
        return cls(setup.microscope, setup.detector, dict(sources))

    def __getitem__(self, key: str) -> float:
        f = self._field(key)
        owner = self.detector if hasattr(self.detector, f.target) else self.microscope
        value = getattr(owner, f.target)
        return float(value if f.index is None else value[f.index])

    def values(self) -> dict[str, float]:
        """Every field's value, by key."""
        return {f.key: self[f.key] for f in self.fields}

    def replace(self, **values: float) -> Setup:
        """A copy with fields replaced by key, each written to the microscope and the
        detector wherever it is carried. :attr:`sources` is kept."""
        changes: tuple[dict, dict] = ({}, {})
        for key, value in values.items():
            f = self._field(key)
            for owner, changed in zip((self.microscope, self.detector), changes, strict=True):
                if not hasattr(owner, f.target):
                    continue
                if f.index is None:
                    changed[f.target] = float(value)
                else:
                    current = list(changed.get(f.target, getattr(owner, f.target)))
                    current[f.index] = float(value)
                    changed[f.target] = tuple(current)
        return Setup(
            dataclasses.replace(self.microscope, **changes[0]),
            dataclasses.replace(self.detector, **changes[1]),
            dict(self.sources),
        )

    def with_shape(self, shape) -> Setup:
        """A copy whose detector has *shape* ``(rows, cols)``."""
        detector = dataclasses.replace(self.detector, shape=tuple(int(n) for n in shape))
        return Setup(self.microscope, detector, dict(self.sources))

    def detector_to_sample(self):
        """The rotation taking detector-frame directions into the sample frame."""
        return self.detector.detector_to_sample()

    def _field(self, key: str) -> Field:
        for f in self.fields:
            if f.key == key:
                return f
        raise KeyError(f"no {self.kind} field {key!r}")
