// Detector geometry mirrors core.geometry.sample_to_detector_matrix (row vectors).
// Exported separately so parity with Python can be tested without a browser.
export function setupGeometry(s) {
  const rad = a => a * Math.PI / 180;
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  let basis = [[0,1,0],[0,0,1],[1,0,0]];
  for (const [axis, angle] of [[0,-s.sample_tilt],[0,s.detector_tilt],[1,-s.azimuthal],[2,-s.twist]]) {
    const u=basis[axis], c=Math.cos(rad(angle)), sn=Math.sin(rad(angle));
    basis=basis.map(v => {
      const uv=cross(u,v), dot=u.reduce((sum,x,i)=>sum+x*v[i],0);
      return v.map((x,i)=>x*c+uv[i]*sn+u[i]*dot*(1-c));
    });
  }
  const H=s.detector_rows*s.px_size/1000, W=s.detector_cols*s.px_size/1000, L=s.pcz*H;
  const toSample = v => [0,1,2].map(j => v.reduce((sum,x,i)=>sum+x*basis[i][j],0));
  const c=Math.cos(rad(s.sample_tilt)), sn=Math.sin(rad(s.sample_tilt));
  // Laboratory X horizontal, Z vertical; positive sample tilt slopes down right.
  const toLab = v => [c*v[0]+sn*v[2],v[1],-sn*v[0]+c*v[2]];
  const corners=[[-s.pcx*W,s.pcy*H,L],[(1-s.pcx)*W,s.pcy*H,L],
    [(1-s.pcx)*W,-(1-s.pcy)*H,L],[-s.pcx*W,-(1-s.pcy)*H,L]].map(toSample);
  const pc=toSample([0,0,L]);
  return {H,W,L,corners,lab:corners.map(toLab),pc,pcLab:toLab(pc),normal:[sn,0,c],
    tangent:[c,0,-sn], center:corners[0].map((_,j)=>corners.reduce((sum,v)=>sum+v[j],0)/4),
    planeAngle:Math.acos(Math.min(1,Math.abs(basis[2][2])))*180/Math.PI,
    footprint:s.probe_diameter_nm>0 && c>1e-12 ? [s.probe_diameter_nm/c,s.probe_diameter_nm] : null};
}

function render({model,el}) {
  const host=el.shadowRoot || el.attachShadow({mode:"open"});
  host.innerHTML=`<style>
    :host {display:block; max-width:580px; color-scheme:light dark}
    * {box-sizing:border-box}
    .setup {--fg:var(--spyde-fg,var(--jp-ui-font-color1,light-dark(#242a33,#d8dee9)));
      --bg:var(--spyde-panel-bg,var(--jp-layout-color1,light-dark(#f6f7f9,#20242d)));
      --line:var(--spyde-border,var(--jp-border-color2,light-dark(#ccd0d7,#4c5565)));
      --beam:var(--spyde-accent,light-dark(#1672b8,#5ba7f7)); --sample:light-dark(#a45c17,#e5a05c);
      --screen:light-dark(#197349,#66c58a); --pc:light-dark(#ad2858,#f080a6);
      color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;
      font:12px var(--jp-ui-font-family,system-ui,sans-serif);overflow:hidden}
    header {padding:9px 12px;border-bottom:1px solid var(--line);display:flex;gap:10px;flex-wrap:wrap}
    header span {margin-left:auto} svg {display:block;width:100%;font:12px system-ui;touch-action:manipulation}
    svg text {fill:var(--fg)} .insets {display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--line)}
    .controls {padding:10px 12px;display:grid;grid-template-columns:1fr 1fr;gap:7px 14px;border-top:1px solid var(--line)}
    label {display:flex;align-items:center;justify-content:space-between;gap:6px;min-width:0}
    input {font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:3px;
      width:84px;min-width:0;padding:3px 5px} .facts,.note {padding:7px 12px;line-height:1.55}
    .facts {border-top:1px solid var(--line);font-variant-numeric:tabular-nums} .note {font-size:11px}
    @media(max-width:400px) {.insets {grid-template-columns:1fr}.controls {gap:7px}.controls label {flex-wrap:wrap}.controls input {width:100%}}
  </style><section class="setup"><header><strong>Microscope / detector</strong><span></span></header>
  <svg class="geometry" height="250" role="img" aria-label="Laboratory side view: beam, sample normal and detector"></svg>
  <div class="insets"><svg class="face" height="175" role="img" aria-label="Detector face and Bruker pattern center"></svg>
  <svg class="probe" height="175" role="img" aria-label="Geometric probe footprint on the sample"></svg></div>
  <div class="facts" aria-live="polite"></div><div class="controls"></div>
  <div class="note">Geometry editor only: existing simulation results do not change. In Jupyter, read edits with
  <code>to_microscope()</code> / <code>to_detector()</code>, then recompute. Screen pitch includes binning.</div></section>`;
  const root=host.querySelector('.setup'), main=host.querySelector('.geometry'), face=host.querySelector('.face'),
    probe=host.querySelector('.probe'), controls=host.querySelector('.controls'), facts=host.querySelector('.facts');
  const fields=[['energy_kev','Beam (keV)',0.001,null,0.1],['sample_tilt','Incidence (°)',0,90,0.1],
    ['probe_diameter_nm','Probe Ø (nm; 0 = unset)',0,null,1],['detector_tilt','Screen tilt from vertical (°)',-90,90,0.1],
    ['pcx','PCx / width',null,null,0.001],['pcy','PCy / height',null,null,0.001],
    ['pcz','PCz / height',0.000001,null,0.001],['px_size','Screen pitch (µm)',0.000001,null,1],
    ['detector_rows','Rows',1,null,1],['detector_cols','Columns',1,null,1],
    ['azimuthal','Azimuth (°)',-180,180,0.1],['twist','Screen twist (°)',-180,180,0.1]];
  const inputs=new Map();
  for (const [key,label,min,max,step] of fields) {
    const wrapper=document.createElement('label'), input=document.createElement('input');
    wrapper.append(document.createTextNode(label)); input.type='number'; input.step=step;
    if(min!==null) input.min=min; if(max!==null) input.max=max;
    input.setAttribute('aria-label',label); wrapper.append(input); controls.append(wrapper); inputs.set(key,input);
    input.addEventListener('change',()=>{
      if(input.value.trim()==='' || !input.checkValidity() || !Number.isFinite(input.valueAsNumber)) {
        input.reportValidity(); input.value=model.get(key); return;
      }
      model.set(key,input.valueAsNumber); model.save_changes();
    });
  }
  const node=(svg,tag,attrs={},text='')=>{
    const n=document.createElementNS('http://www.w3.org/2000/svg',tag);
    for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v);
    n.textContent=text; svg.append(n); return n;
  };
  const text=(svg,x,y,value,anchor='start')=>node(svg,'text',{x,y,'text-anchor':anchor},value);
  const line=(svg,a,b,color,width=1,dash='')=>node(svg,'line',
    {x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:color,'stroke-width':width,'stroke-dasharray':dash});
  const arrow=(svg,a,b,color)=>{
    line(svg,a,b,color,2); const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);
    const p=[b,[b[0]-8*Math.cos(angle-.4),b[1]-8*Math.sin(angle-.4)],
      [b[0]-8*Math.cos(angle+.4),b[1]-8*Math.sin(angle+.4)]];
    node(svg,'polygon',{points:p.map(v=>v.join(',')).join(' '),fill:color});
  };
  const fmt=n=>Number(n).toFixed(2);
  function draw() {
    const s=Object.fromEntries(fields.map(([key])=>[key,Number(model.get(key))]));
    const g=setupGeometry(s), has=model.get('has_detector');
    const css=getComputedStyle(root), beam=css.getPropertyValue('--beam'), sample=css.getPropertyValue('--sample'),
      screen=css.getPropertyValue('--screen'), pcColor=css.getPropertyValue('--pc');
    for(const [key,input] of inputs) if(input!==host.activeElement) input.value=s[key];
    host.querySelector('header span').textContent=`${model.get('crystal_label') || 'Sample'} · ${fmt(s.energy_kev)} keV`;
    for(const svg of [main,face,probe]) {svg.replaceChildren(); svg.setAttribute('viewBox',`0 0 ${svg.clientWidth || 280} ${svg.getAttribute('height')}`);}
    const w=main.clientWidth||560, H=250, size=has?Math.max(g.H,g.W,g.L):1;
    const scaleVec=(v,n)=>v.map(x=>x*n);
    const a=scaleVec(g.tangent,-size*.42), b=scaleVec(g.tangent,size*.42),
      normal=scaleVec(g.normal,size*.40), source=[0,0,size*.72];
    const all=[a,b,normal,source,[0,0,0],...(has?g.lab:[])];
    const xmin=Math.min(...all.map(p=>p[0])),xmax=Math.max(...all.map(p=>p[0]));
    const zmin=Math.min(...all.map(p=>p[2])),zmax=Math.max(...all.map(p=>p[2]));
    const k=Math.min((w-96)/(xmax-xmin),(H-76)/(zmax-zmin));
    const P=v=>[48+(w-96-(xmax-xmin)*k)/2+(v[0]-xmin)*k,40+(zmax-v[2])*k];
    const origin=P([0,0,0]), A=P(a), B=P(b);
    text(main,12,18,'Laboratory side view · equal mm scale');
    const back=v=>v.map((x,i)=>x-size*.07*g.normal[i]);
    node(main,'polygon',{points:[a,b,back(b),back(a)].map(v=>P(v).join(',')).join(' '),fill:sample,opacity:.25});
    line(main,A,B,sample,4); arrow(main,origin,P(normal),sample); arrow(main,P(source),origin,beam);
    text(main,P(source)[0]-8,P(source)[1]+9,'beam','end');
    text(main,P(normal)[0]+5,P(normal)[1]+4,'n');
    text(main,12,H-10,`Incidence ${s.sample_tilt.toFixed(1)}° from n · grazing ${(90-s.sample_tilt).toFixed(1)}°`);
    const r=28, end=[origin[0]+r*g.normal[0],origin[1]-r*g.normal[2]];
    node(main,'path',{d:`M ${origin[0]} ${origin[1]-r} A ${r} ${r} 0 0 1 ${end[0]} ${end[1]}`,
      fill:'none',stroke:sample,'stroke-width':1.5});
    node(main,'circle',{cx:origin[0],cy:origin[1],r:3,fill:beam});
    if(has) {
      node(main,'polygon',{points:g.lab.map(v=>P(v).join(',')).join(' '),fill:screen,'fill-opacity':.2,stroke:screen,'stroke-width':3});
      const pc=P(g.pcLab); line(main,origin,pc,pcColor,1.5,'4 3');
      node(main,'circle',{cx:pc[0],cy:pc[1],r:4,fill:pcColor}); text(main,pc[0]+7,pc[1]-7,'PC');
      text(main,w-12,34,`normal distance ${fmt(g.L)} mm`,'end');
      text(main,w-12,51,`screen / sample planes: ${g.planeAngle.toFixed(1)}°`,'end');
    }
    // Detector face, independent of the side projection; row zero is the top.
    const fw=face.clientWidth||280, fh=175;
    text(face,12,20,has?'Detector face · Bruker PC':'No detector supplied');
    if(has) {
      const maxX=Math.max(1,s.pcx),minX=Math.min(0,s.pcx),maxY=Math.max(1,s.pcy),minY=Math.min(0,s.pcy);
      const kf=Math.min((fw-60)/(g.W*(maxX-minX)),98/(g.H*(maxY-minY)));
      const left=30-minX*g.W*kf, top=40-minY*g.H*kf, ww=g.W*kf, hh=g.H*kf;
      node(face,'rect',{x:left,y:top,width:ww,height:hh,fill:screen,'fill-opacity':.13,stroke:screen});
      const x=left+s.pcx*ww,y=top+s.pcy*hh;
      line(face,[x-7,y],[x+7,y],pcColor,2);line(face,[x,y-7],[x,y+7],pcColor,2);
      text(face,fw/2,fh-20,`${s.detector_cols} × ${s.detector_rows} pixels`,'middle');
      text(face,fw/2,fh-4,`${fmt(g.W)} × ${fmt(g.H)} mm · row 0 at top`,'middle');
      face.style.cursor='crosshair';
      face.onclick=event=>{
        const rect=face.getBoundingClientRect();
        model.set('pcx',Math.min(1,Math.max(0,(event.clientX-rect.left-left)/ww)));
        model.set('pcy',Math.min(1,Math.max(0,(event.clientY-rect.top-top)/hh)));model.save_changes();
      };
    }
    const pw=probe.clientWidth||280;
    text(probe,12,20,'Probe footprint · sample-plane inset');
    if(g.footprint) {
      const [major,minor]=g.footprint, rx=Math.min((pw-48)/2,82), ry=rx*minor/major;
      node(probe,'ellipse',{cx:pw/2,cy:83,rx,ry,fill:beam,'fill-opacity':.18,stroke:beam,'stroke-width':2});
      line(probe,[pw/2-rx,83],[pw/2+rx,83],beam,1,'3 3');
      text(probe,pw/2,143,`${fmt(major)} × ${fmt(minor)} nm`,'middle');
      text(probe,pw/2,161,'Parallel circular probe · separate scale','middle');
    } else {
      text(probe,pw/2,78,s.probe_diameter_nm?'No finite footprint at 90°':'Probe diameter not specified','middle');
      text(probe,pw/2,99,'Not an interaction-volume model','middle');
    }
    const centerLab=g.lab.reduce((sum,p)=>sum+p[2],0)/4;
    const minHeight=Math.min(...g.corners.map(p=>p[2]));
    facts.textContent=has?`PC (${s.pcx.toFixed(3)}, ${s.pcy.toFixed(3)}, ${s.pcz.toFixed(3)}) · screen center: ${fmt(centerLab)} mm above impact (lab), ${fmt(g.center[2])} mm above the sample plane (normal). `+
      (minHeight<0?'Warning: part of the screen lies behind the extended sample plane. ':'')+
      'Click the detector face to set PCx / PCy.':
      'Sample normal points out of the material. Probe size is descriptive metadata, not a solver input.';
  }
  const names=[...fields.map(f=>f[0]),'crystal_label','has_detector'];
  for(const name of names) model.on(`change:${name}`,draw);
  const observer=new ResizeObserver(draw); observer.observe(root); draw();
  return ()=>{observer.disconnect();for(const name of names) model.off(`change:${name}`,draw);};
}
export default {render};
