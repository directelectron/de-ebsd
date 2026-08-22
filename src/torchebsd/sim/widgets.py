"""Interactive experiment-setup widget (anywidget).

:class:`SetupWidget` draws the EBSD geometry -- beam, tilted sample, detector
plane, and projection center -- as a live schematic in Jupyter, with sliders
for the angles and the projection center. The traitlets stay in sync both
ways, so the widget can both display a configuration and be used to adjust
one; read the result back with :meth:`to_microscope` / :meth:`to_detector`.
"""

from __future__ import annotations

import anywidget
import traitlets

from torchebsd.sim.detector import EBSDDetector
from torchebsd.sim.microscope import Microscope

__all__ = ["SetupWidget"]

_ESM = """
function render({ model, el }) {
  el.innerHTML = "";
  const root = document.createElement("div");
  root.style.cssText = "font-family: system-ui, sans-serif; max-width: 560px;";

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 560 340");
  svg.style.cssText = "width:100%; background:#fafafa; border:1px solid #ddd; border-radius:6px;";
  root.appendChild(svg);

  const controls = document.createElement("div");
  controls.style.cssText = "display:grid; grid-template-columns:11em 1fr 4.5em; " +
    "gap:4px 8px; margin-top:8px; font-size:13px; align-items:center;";
  root.appendChild(controls);

  const sliders = [
    ["sample_tilt", "Sample tilt [deg]", 0, 90, 0.5],
    ["detector_tilt", "Detector tilt [deg]", -45, 45, 0.5],
    ["pcx", "PC x", 0, 1, 0.005],
    ["pcy", "PC y", 0, 1, 0.005],
    ["pcz", "PC z (distance)", 0.1, 1.5, 0.005],
    ["energy_kev", "Energy [keV]", 5, 30, 0.5],
  ];
  for (const [name, label, min, max, step] of sliders) {
    const lab = document.createElement("label");
    lab.textContent = label;
    const input = document.createElement("input");
    input.type = "range";
    input.min = min; input.max = max; input.step = step;
    input.value = model.get(name);
    const val = document.createElement("span");
    val.textContent = Number(model.get(name)).toFixed(2);
    input.addEventListener("input", () => {
      model.set(name, Number(input.value));
      model.save_changes();
    });
    model.on(`change:${name}`, () => {
      input.value = model.get(name);
      val.textContent = Number(model.get(name)).toFixed(2);
    });
    controls.append(lab, input, val);
  }

  function line(x1, y1, x2, y2, stroke, width, dash) {
    const e = document.createElementNS(svgNS, "line");
    e.setAttribute("x1", x1); e.setAttribute("y1", y1);
    e.setAttribute("x2", x2); e.setAttribute("y2", y2);
    e.setAttribute("stroke", stroke); e.setAttribute("stroke-width", width);
    if (dash) e.setAttribute("stroke-dasharray", dash);
    svg.appendChild(e);
    return e;
  }
  function text(x, y, s, fill) {
    const e = document.createElementNS(svgNS, "text");
    e.setAttribute("x", x); e.setAttribute("y", y);
    e.setAttribute("fill", fill || "#333");
    e.setAttribute("font-size", "12");
    e.textContent = s;
    svg.appendChild(e);
    return e;
  }
  function circle(x, y, r, fill) {
    const e = document.createElementNS(svgNS, "circle");
    e.setAttribute("cx", x); e.setAttribute("cy", y);
    e.setAttribute("r", r); e.setAttribute("fill", fill);
    svg.appendChild(e);
    return e;
  }

  function draw() {
    svg.innerHTML = "";
    const sigma = model.get("sample_tilt") * Math.PI / 180;
    const theta = model.get("detector_tilt") * Math.PI / 180;
    const pcy = model.get("pcy");
    const pcz = model.get("pcz");

    // side view: beam travels down; sample tilted sigma from horizontal
    const ox = 200, oy = 200; // interaction point
    // beam
    line(ox, 20, ox, oy, "#1565c0", 2.5);
    text(ox + 6, 40, "beam (" + model.get("energy_kev").toFixed(1) + " keV)", "#1565c0");
    // sample surface through the interaction point
    const sLen = 130;
    const sx = Math.cos(sigma), sy = -Math.sin(sigma);
    line(ox - sLen * sx, oy - sLen * sy, ox + sLen * sx, oy + sLen * sy, "#616161", 5);
    text(ox - sLen * sx - 4, oy - sLen * sy - 8, "sample", "#616161");
    text(ox - 66, oy + 16, "tilt " + model.get("sample_tilt").toFixed(1) + "\\u00B0", "#616161");

    // detector plane: nominally vertical at distance d, tilted by theta
    const d = 120 + 140 * pcz;
    const dx0 = ox + d, dy0 = oy - 10;
    const dLen = 120;
    const tx = Math.sin(theta), ty = Math.cos(theta);
    const topX = dx0 - tx * dLen * (1 - pcy), topY = dy0 - ty * dLen * (1 - pcy);
    const botX = dx0 + tx * dLen * pcy, botY = dy0 + ty * dLen * pcy;
    line(topX, topY, botX, botY, "#2e7d32", 5);
    text(topX + 8, topY + 2, "detector", "#2e7d32");

    // pattern-center ray and marker (pcy along the detector height)
    line(ox, oy, dx0, dy0, "#d32f2f", 1.5, "5,4");
    circle(dx0, dy0, 4, "#d32f2f");
    text(dx0 + 8, dy0 + 4, "PC", "#d32f2f");

    // horizontal reference
    line(60, oy, 500, oy, "#bbb", 1, "2,4");
    text(468, oy - 6, "horizontal", "#999");

    const shape = model.get("detector_rows") + "\\u00D7" + model.get("detector_cols");
    text(16, 320, model.get("crystal_label") + "   detector " + shape +
         "   PC (" + model.get("pcx").toFixed(3) + ", " + pcy.toFixed(3) +
         ", " + pcz.toFixed(3) + ")");
  }

  for (const [name] of sliders) model.on(`change:${name}`, draw);
  model.on("change:detector_rows", draw);
  model.on("change:detector_cols", draw);
  model.on("change:crystal_label", draw);
  draw();
  el.appendChild(root);
}
export default { render };
"""


class SetupWidget(anywidget.AnyWidget):
    """Interactive schematic of the microscope / sample / detector geometry."""

    _esm = _ESM

    energy_kev = traitlets.Float(20.0).tag(sync=True)
    sample_tilt = traitlets.Float(70.0).tag(sync=True)
    detector_tilt = traitlets.Float(0.0).tag(sync=True)
    azimuthal = traitlets.Float(0.0).tag(sync=True)
    pcx = traitlets.Float(0.5).tag(sync=True)
    pcy = traitlets.Float(0.5).tag(sync=True)
    pcz = traitlets.Float(0.5).tag(sync=True)
    detector_rows = traitlets.Int(480).tag(sync=True)
    detector_cols = traitlets.Int(480).tag(sync=True)
    crystal_label = traitlets.Unicode("").tag(sync=True)

    @classmethod
    def from_setup(
        cls,
        microscope: Microscope | None = None,
        detector: EBSDDetector | None = None,
        crystal=None,
    ) -> SetupWidget:
        """Build the widget from existing configuration objects."""
        widget = cls()
        if microscope is not None:
            widget.energy_kev = microscope.energy_kev
            widget.sample_tilt = microscope.sample_tilt
            widget.azimuthal = microscope.azimuthal
        if detector is not None:
            widget.detector_rows, widget.detector_cols = detector.shape
            widget.pcx, widget.pcy, widget.pcz = detector.pc
            widget.detector_tilt = detector.tilt
            if microscope is None:
                widget.sample_tilt = detector.sample_tilt
        if crystal is not None:
            widget.crystal_label = repr(crystal)
        return widget

    def to_microscope(self) -> Microscope:
        """The current widget state as a :class:`Microscope`."""
        return Microscope(
            energy=self.energy_kev * 1e3,
            sample_tilt=self.sample_tilt,
            azimuthal=self.azimuthal,
        )

    def to_detector(self) -> EBSDDetector:
        """The current widget state as an :class:`EBSDDetector`."""
        return EBSDDetector(
            shape=(self.detector_rows, self.detector_cols),
            pc=(self.pcx, self.pcy, self.pcz),
            sample_tilt=self.sample_tilt,
            tilt=self.detector_tilt,
            azimuthal=self.azimuthal,
        )
