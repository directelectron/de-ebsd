"""Portable experiment inspectors; one renderer for Jupyter, HTML, and SpyDE.

HTML controls edit a local snapshot; notebook controls synchronize via traitlets.
Scientific master-pattern and crystal views remain native anyplotlib figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import anywidget
import numpy as np
import traitlets

from de_ebsd.sim.detector import EBSDDetector
from de_ebsd.sim.microscope import Microscope

__all__ = ["DepthWidget", "MicroscopeWidget", "SetupWidget"]


class _PortableWidget(anywidget.AnyWidget):
    """Export the actual renderer without a notebook manager or CDN dependency."""

    def _repr_html_(self):
        identity = f"de-ebsd-{uuid4().hex}"
        state = {name: getattr(self, name) for name in self.traits(sync=True)
                 if not name.startswith("_") and name != "layout"}
        payload = json.dumps(state, allow_nan=False).replace("<", "\\u003c")
        renderer = str(self._esm).replace("</script", "<\\/script")
        return f'''<div id="{identity}"></div><script type="module">
{renderer}
const state = {payload};
const events = new Map();
const model = {{
  get: key => state[key],
  set(key, value) {{
    state[key] = value;
    for (const callback of events.get(`change:${{key}}`) || []) callback();
  }},
  on(key, callback) {{
    if (!events.has(key)) events.set(key, new Set());
    events.get(key).add(callback);
  }},
  off(key, callback) {{ events.get(key)?.delete(callback); }},
  save_changes() {{}},
}};
render({{model, el: document.getElementById("{identity}")}});
</script>'''

    def to_html(self):
        """Interactive standalone HTML; edits do not run Python simulations."""
        return ('<!doctype html><html><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>de-ebsd experiment inspector</title></head><body>'
                + self._repr_html_() + '</body></html>')


class MicroscopeWidget(_PortableWidget):
    """Editable geometry with laboratory side view and detector-face inset.

    Edits never mutate inputs or recompute existing master patterns. Read an
    edited setup explicitly with to_microscope()/to_detector().
    """

    _esm = Path(__file__).with_name("setup_widget.js").read_text(encoding="utf-8")
    energy_kev = traitlets.Float(20.0, min=0.001).tag(sync=True)
    sample_tilt = traitlets.Float(70.0, min=0.0, max=90.0).tag(sync=True)
    detector_tilt = traitlets.Float(0.0).tag(sync=True)
    azimuthal = traitlets.Float(0.0).tag(sync=True)
    twist = traitlets.Float(0.0).tag(sync=True)
    probe_diameter_nm = traitlets.Float(0.0, min=0.0).tag(sync=True)
    pcx = traitlets.Float(0.5).tag(sync=True)
    pcy = traitlets.Float(0.5).tag(sync=True)
    pcz = traitlets.Float(0.5, min=0.000001).tag(sync=True)
    detector_rows = traitlets.Int(480, min=1).tag(sync=True)
    detector_cols = traitlets.Int(480, min=1).tag(sync=True)
    px_size = traitlets.Float(1.0, min=0.000001).tag(sync=True)
    crystal_label = traitlets.Unicode("").tag(sync=True)
    has_detector = traitlets.Bool(False).tag(sync=True)

    @traitlets.validate("energy_kev", "sample_tilt", "detector_tilt", "azimuthal", "twist",
                        "probe_diameter_nm", "pcx", "pcy", "pcz", "px_size")
    def _finite(self, proposal):
        if not np.isfinite(proposal["value"]):
            raise traitlets.TraitError("geometry values must be finite")
        return proposal["value"]

    @classmethod
    def from_setup(cls, microscope=None, detector=None, crystal=None):
        widget = cls()
        if microscope is not None:
            widget.energy_kev = microscope.energy_kev
            widget.sample_tilt = microscope.sample_tilt
            widget.azimuthal = microscope.azimuthal
            widget.probe_diameter_nm = microscope.probe_diameter_nm or 0.0
        if detector is not None:
            widget.has_detector = True
            widget.detector_rows, widget.detector_cols = detector.shape
            widget.pcx, widget.pcy, widget.pcz = detector.pc
            widget.detector_tilt = detector.tilt
            widget.twist = detector.twist
            widget.px_size = detector.px_size
            if microscope is not None and not np.allclose(
                [microscope.sample_tilt, microscope.azimuthal],
                [detector.sample_tilt, detector.azimuthal],
            ):
                raise ValueError("microscope and detector sample_tilt/azimuthal must agree")
            widget.sample_tilt = detector.sample_tilt
            widget.azimuthal = detector.azimuthal
        if crystal is not None:
            widget.crystal_label = _phase_name(crystal)
        return widget

    def to_microscope(self):
        return Microscope(energy=self.energy_kev * 1e3, sample_tilt=self.sample_tilt,
                          azimuthal=self.azimuthal,
                          probe_diameter_nm=self.probe_diameter_nm or None)

    def to_detector(self):
        return EBSDDetector(shape=(self.detector_rows, self.detector_cols),
                            pc=(self.pcx, self.pcy, self.pcz), sample_tilt=self.sample_tilt,
                            tilt=self.detector_tilt, azimuthal=self.azimuthal,
                            twist=self.twist, px_size=self.px_size)


class SetupWidget(MicroscopeWidget):
    """Backward-compatible name for MicroscopeWidget."""


class DepthWidget(_PortableWidget):
    """Read-only depth model, with local energy and slice selection."""

    _esm = Path(__file__).with_name("depth_widget.js").read_text(encoding="utf-8")
    energies_kev = traitlets.List().tag(sync=True)
    profiles = traitlets.List().tag(sync=True)
    integration_bins = traitlets.List().tag(sync=True)
    depth_step = traitlets.Float(1.0).tag(sync=True)
    energy_index = traitlets.Int(0, min=0).tag(sync=True)
    slice_index = traitlets.Int(0, min=0).tag(sync=True)

    @classmethod
    def from_distribution(cls, depth, energy_kev=None):
        counts = np.asarray(depth.accum_z, dtype=float)
        energies = np.asarray(depth.energies_kev, dtype=float)
        if (counts.ndim != 4 or 0 in counts.shape or len(energies) != counts.shape[0]
                or not np.isfinite(counts).all() or (counts < 0).any()
                or not np.isfinite(energies).all()
                or not np.isfinite(depth.depth_step) or depth.depth_step <= 0
                or not np.isfinite(depth.n_total) or depth.n_total <= 0):
            raise ValueError("depth display requires finite nonnegative counts and positive scales")
        index = depth.n_energies - 1 if energy_kev is None else depth.energy_index(energy_kev)
        return cls(energies_kev=energies.tolist(),
                   profiles=(counts.sum(axis=(2, 3)) / depth.n_total).tolist(),
                   integration_bins=depth.integration_depths().tolist(),
                   depth_step=depth.depth_step, energy_index=index)


def _phase_name(crystal):
    return "+".join(dict.fromkeys(site.element for site in crystal.sites)) or "phase"
