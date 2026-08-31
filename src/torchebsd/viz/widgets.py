"""Editable experiment setup widget for Jupyter.

Scientific plots use native anyplotlib figures. This module contains only the
control surface that genuinely benefits from trait synchronization: microscope
and detector parameters edited against a live geometry schematic.
"""

from __future__ import annotations

import anywidget
import traitlets

from torchebsd.sim.detector import EBSDDetector
from torchebsd.sim.microscope import Microscope

__all__ = ["MicroscopeWidget", "SetupWidget"]


_MICROSCOPE_ESM = r"""
function render({ model, el }) {
  el.innerHTML = "";
  const style = document.createElement("style");
  style.textContent = `
    .te-setup {
      --te-fg: var(--spyde-fg, var(--jp-ui-font-color1, #d8dee9));
      --te-muted: var(--spyde-muted, var(--jp-ui-font-color2, #8f98a8));
      --te-bg: var(--spyde-panel-bg, var(--jp-layout-color1, #20242d));
      --te-bg2: var(--spyde-raised-bg, var(--jp-layout-color2, #171a21));
      --te-border: var(--spyde-border, var(--jp-border-color2, #3b4252));
      --te-blue: var(--spyde-accent, #5ba7f7);
      color: var(--te-fg); background: var(--te-bg);
      border: 1px solid var(--te-border); border-radius: 6px; overflow: hidden;
      font: 11px var(--jp-ui-font-family, system-ui, sans-serif); max-width: 540px;
    }
    .te-head { display:flex; align-items:baseline; justify-content:space-between;
      gap:8px; padding:7px 9px; background:var(--te-bg2); border-bottom:1px solid var(--te-border); }
    .te-head strong { font-size:12px; }
    .te-subtitle { color:var(--te-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .te-geometry { display:block; width:100%; background:var(--te-bg2);
      border-bottom:1px solid var(--te-border); }
    .te-controls { display:grid; grid-template-columns:1fr 1fr; gap:5px 12px; padding:8px 9px; }
    .te-field { display:grid; grid-template-columns:66px minmax(54px,1fr) 58px; gap:5px; align-items:center; }
    .te-field > span:first-child, .te-readout { color:var(--te-muted); }
    .te-field input[type=range] { min-width:40px; width:100%; accent-color:var(--te-blue); }
    .te-field input[type=number], .te-facts input { box-sizing:border-box; width:100%; min-width:0;
      color:var(--te-fg); background:var(--te-bg2); border:1px solid var(--te-border);
      border-radius:3px; padding:2px 3px; font:inherit; font-variant-numeric:tabular-nums; }
    .te-facts { grid-column:1/-1; display:grid; grid-template-columns:repeat(3,1fr); gap:6px;
      padding-top:6px; margin-top:1px; border-top:1px solid var(--te-border); }
    .te-fact { display:grid; grid-template-columns:auto 1fr; align-items:center; gap:4px; color:var(--te-muted); }
    .te-bounds { grid-column:1/-1; color:var(--te-muted); font-variant-numeric:tabular-nums; }
  `;

  const root = document.createElement("div"); root.className = "te-setup";
  const header = document.createElement("div"); header.className = "te-head";
  const title = document.createElement("strong"); title.textContent = "Microscope / detector";
  const subtitle = document.createElement("span"); subtitle.className = "te-subtitle";
  header.append(title, subtitle);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 540 170"); svg.classList.add("te-geometry");
  const controls = document.createElement("div"); controls.className = "te-controls";
  root.append(header, svg, controls); el.append(style, root);

  const svgNS = "http://www.w3.org/2000/svg";
  const line = (x1,y1,x2,y2,color,width,dash) => {
    const node=document.createElementNS(svgNS,"line");
    for (const [key,value] of Object.entries({x1,y1,x2,y2})) node.setAttribute(key,value);
    node.setAttribute("stroke",color); node.setAttribute("stroke-width",width);
    if (dash) node.setAttribute("stroke-dasharray",dash); svg.appendChild(node);
  };
  const text = (x,y,value,color) => {
    const node=document.createElementNS(svgNS,"text"); node.setAttribute("x",x); node.setAttribute("y",y);
    node.setAttribute("fill",color); node.setAttribute("font-size","10"); node.textContent=value; svg.appendChild(node);
  };
  const circle = (x,y,r,color) => {
    const node=document.createElementNS(svgNS,"circle"); node.setAttribute("cx",x); node.setAttribute("cy",y);
    node.setAttribute("r",r); node.setAttribute("fill",color); svg.appendChild(node);
  };

  const fields = [
    ["energy_kev", "Energy", 5, 40, 0.1, "keV"],
    ["sample_tilt", "Sample tilt", 0, 90, 0.1, "°"],
    ["detector_tilt", "Detector tilt", -45, 45, 0.1, "°"],
    ["pcx", "PC x", 0, 1, 0.001, ""],
    ["pcy", "PC y", 0, 1, 0.001, ""],
    ["pcz", "PC z", 0.1, 1.5, 0.001, ""],
  ];
  const fieldNodes = {};
  let savePending = false;
  const commit = (name, value) => {
    model.set(name, Number(value));
    if (!savePending) requestAnimationFrame(() => {
      savePending = false;
      model.save_changes();
    });
    savePending = true;
  };
  for (const [name,label,min,max,step,unit] of fields) {
    const wrap=document.createElement("label"); wrap.className="te-field";
    const labelNode=document.createElement("span"); labelNode.textContent=label;
    const range=document.createElement("input"); range.type="range";
    Object.assign(range,{min,max,step,value:model.get(name)});
    const number=document.createElement("input"); number.type="number";
    Object.assign(number,{min,max,step,value:model.get(name)});
    number.title=unit; wrap.append(labelNode,range,number); controls.appendChild(wrap);
    fieldNodes[name]={range,number,step};
    range.addEventListener("input",()=>commit(name,range.value));
    number.addEventListener("change",()=>commit(name,number.value));
  }

  const facts=document.createElement("div"); facts.className="te-facts"; controls.appendChild(facts);
  const factInputs = {};
  for (const [name,label,min,step] of [
    ["detector_rows","rows",1,1], ["detector_cols","cols",1,1], ["px_size","pixel µm",0.001,0.01]
  ]) {
    const wrap=document.createElement("label"); wrap.className="te-fact";
    const labelNode=document.createElement("span"); labelNode.textContent=label;
    const input=document.createElement("input"); input.type="number"; input.min=min; input.step=step;
    input.addEventListener("change",()=>commit(name,input.value));
    wrap.append(labelNode,input); facts.appendChild(wrap); factInputs[name]=input;
  }
  const bounds=document.createElement("div"); bounds.className="te-bounds"; controls.appendChild(bounds);

  function draw() {
    const css=getComputedStyle(root), muted=css.getPropertyValue("--te-muted"), border=css.getPropertyValue("--te-border");
    svg.innerHTML="";
    const sampleTilt=Number(model.get("sample_tilt")), detectorTilt=Number(model.get("detector_tilt"));
    const tilt=sampleTilt*Math.PI/180, dtilt=detectorTilt*Math.PI/180;
    const pcy=Number(model.get("pcy")), pcz=Number(model.get("pcz"));
    const ox=175, oy=112, detectorX=Math.min(480,365+70*pcz);
    line(45,oy,510,oy,border,1,"3 4");
    line(ox,18,ox,oy,"#5ba7f7",3); text(ox+7,30,`${Number(model.get("energy_kev")).toFixed(1)} keV beam`,"#5ba7f7");
    const sx=Math.cos(tilt), sy=-Math.sin(tilt), sl=88;
    line(ox-sl*sx,oy-sl*sy,ox+sl*sx,oy+sl*sy,"#e5a05c",6);
    text(ox-48,oy+17,`sample ${sampleTilt.toFixed(1)}°`,"#e5a05c");
    const dy=Math.cos(dtilt)*58, dx=Math.sin(dtilt)*58;
    line(detectorX-dx,oy-dy,detectorX+dx,oy+dy,"#66c58a",6);
    text(detectorX+8,oy-48,`detector ${detectorTilt.toFixed(1)}°`,"#66c58a");
    line(ox,oy,detectorX,oy,"#e86b8d",1.5,"5 4"); circle(detectorX,oy,4,"#e86b8d");
    text(detectorX+7,oy+4,"PC","#e86b8d");
    text(12,158,`PC (${Number(model.get("pcx")).toFixed(3)}, ${pcy.toFixed(3)}, ${pcz.toFixed(3)})`,muted);

    for (const [name,,,step] of fields) {
      const value=Number(model.get(name)); fieldNodes[name].range.value=value;
      if (document.activeElement !== fieldNodes[name].number) fieldNodes[name].number.value=value.toFixed(step < .01 ? 3 : 1);
    }
    for (const name of Object.keys(factInputs)) if (document.activeElement !== factInputs[name]) factInputs[name].value=model.get(name);
    const rows=Number(model.get("detector_rows")), cols=Number(model.get("detector_cols"));
    const aspect=cols/rows, pcx=Number(model.get("pcx"));
    const gb=[-aspect*pcx/pcz,aspect*(1-pcx)/pcz,-(1-pcy)/pcz,pcy/pcz];
    bounds.textContent=`gnomonic x ${gb[0].toFixed(2)} … ${gb[1].toFixed(2)} · y ${gb[2].toFixed(2)} … ${gb[3].toFixed(2)}`;
    subtitle.textContent=`${model.get("crystal_label") || "sample"} · ${rows}×${cols} · ${Number(model.get("px_size")).toFixed(2)} µm px`;
  }
  for (const [name] of fields) model.on(`change:${name}`,draw);
  for (const name of ["detector_rows","detector_cols","px_size","crystal_label"]) model.on(`change:${name}`,draw);
  draw();
}
export default { render };
"""


class MicroscopeWidget(anywidget.AnyWidget):
    """Compact, editable microscope and detector geometry."""

    _esm = _MICROSCOPE_ESM

    energy_kev = traitlets.Float(20.0).tag(sync=True)
    sample_tilt = traitlets.Float(70.0).tag(sync=True)
    detector_tilt = traitlets.Float(0.0).tag(sync=True)
    azimuthal = traitlets.Float(0.0).tag(sync=True)
    pcx = traitlets.Float(0.5).tag(sync=True)
    pcy = traitlets.Float(0.5).tag(sync=True)
    pcz = traitlets.Float(0.5).tag(sync=True)
    detector_rows = traitlets.Int(480).tag(sync=True)
    detector_cols = traitlets.Int(480).tag(sync=True)
    px_size = traitlets.Float(1.0).tag(sync=True)
    crystal_label = traitlets.Unicode("").tag(sync=True)

    @classmethod
    def from_setup(
        cls,
        microscope: Microscope | None = None,
        detector: EBSDDetector | None = None,
        crystal=None,
    ) -> MicroscopeWidget:
        """Build the editor from existing microscope/detector objects."""
        widget = cls()
        if microscope is not None:
            widget.energy_kev = microscope.energy_kev
            widget.sample_tilt = microscope.sample_tilt
            widget.azimuthal = microscope.azimuthal
        if detector is not None:
            widget.detector_rows, widget.detector_cols = detector.shape
            widget.pcx, widget.pcy, widget.pcz = detector.pc
            widget.detector_tilt = detector.tilt
            widget.px_size = detector.px_size
            if microscope is None:
                widget.sample_tilt = detector.sample_tilt
                widget.azimuthal = detector.azimuthal
        if crystal is not None:
            widget.crystal_label = _phase_name(crystal)
        return widget

    def to_microscope(self) -> Microscope:
        """Read the edited state as a :class:`Microscope`."""
        return Microscope(
            energy=self.energy_kev * 1e3,
            sample_tilt=self.sample_tilt,
            azimuthal=self.azimuthal,
        )

    def to_detector(self) -> EBSDDetector:
        """Read the edited state as an :class:`EBSDDetector`."""
        return EBSDDetector(
            shape=(self.detector_rows, self.detector_cols),
            pc=(self.pcx, self.pcy, self.pcz),
            sample_tilt=self.sample_tilt,
            tilt=self.detector_tilt,
            azimuthal=self.azimuthal,
            px_size=self.px_size,
        )


class SetupWidget(MicroscopeWidget):
    """Backward-compatible name for :class:`MicroscopeWidget`."""


def _phase_name(crystal) -> str:
    elements = "+".join(dict.fromkeys(site.element for site in crystal.sites))
    return elements or "phase"
