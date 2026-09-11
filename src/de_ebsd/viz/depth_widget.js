function render({model,el}) {
  const host=el.shadowRoot||el.attachShadow({mode:'open'});
  host.innerHTML=`<style>
    :host {display:block;max-width:580px;color-scheme:light dark} *{box-sizing:border-box}
    section {--fg:var(--spyde-fg,var(--jp-ui-font-color1,light-dark(#242a33,#d8dee9)));
      --bg:var(--spyde-panel-bg,var(--jp-layout-color1,light-dark(#f6f7f9,#20242d)));
      --line:var(--spyde-border,var(--jp-border-color2,light-dark(#ccd0d7,#4c5565)));
      --accent:var(--spyde-accent,light-dark(#1672b8,#5ba7f7));--cut:light-dark(#a45c17,#e5a05c);
      border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);
      font:12px var(--jp-ui-font-family,system-ui,sans-serif);overflow:hidden}
    header,.controls,.detail,.note {padding:9px 12px;line-height:1.5}
    header {border-bottom:1px solid var(--line)} .controls {display:flex;flex-wrap:wrap;gap:12px}
    label {display:flex;gap:8px;align-items:center} input {max-width:180px} select {font:inherit;color:inherit;background:var(--bg)}
    svg {display:block;width:100%;font:12px system-ui} svg text {fill:var(--fg)}
    .detail {border-top:1px solid var(--line);font-variant-numeric:tabular-nums} .note {font-size:11px}
  </style><section><header><strong>Depth model · exit statistics and propagation slices</strong></header>
  <div class="controls"><label>Energy <select aria-label="Depth energy"></select></label>
  <label>Slice <input aria-label="Depth slice" type="range" min="0" step="1"><output></output></label></div>
  <svg height="280" role="img" aria-label="Depth slices, exit yield and integration cutoff"></svg>
  <div class="detail" aria-live="polite"></div><div class="note">Bars show lateral-summed exit yield per incident electron, before absorption correction.
  Colored slices are retained by the backend's 99% rule (maximum over lateral pixels and lower energies).
  Depth is normal to the surface, not a ray path length. Selecting a slice only inspects the model.</div></section>`;
  const svg=host.querySelector('svg'), select=host.querySelector('select'), slider=host.querySelector('input');
  const energies=model.get('energies_kev');
  energies.forEach((value,index)=>{const o=document.createElement('option');o.value=index;o.textContent=`${value} keV`;select.append(o);});
  select.onchange=()=>{model.set('energy_index',Number(select.value));model.save_changes();};
  slider.oninput=()=>{model.set('slice_index',Number(slider.value));model.save_changes();};
  const node=(tag,attrs={},value='')=>{
    const n=document.createElementNS('http://www.w3.org/2000/svg',tag);
    for(const [key,v] of Object.entries(attrs))n.setAttribute(key,v);n.textContent=value;svg.append(n);return n;
  };
  const text=(x,y,value,anchor='start')=>node('text',{x,y,'text-anchor':anchor},value);
  function draw() {
    const ei=Math.min(energies.length-1,model.get('energy_index')), p=model.get('profiles')[ei], n=p.length,
      selected=Math.min(n-1,model.get('slice_index')), count=model.get('integration_bins')[ei], dz=model.get('depth_step');
    const W=svg.clientWidth||560, top=42, bottom=236, h=bottom-top;
    svg.setAttribute('viewBox',`0 0 ${W} 280`);svg.replaceChildren();
    select.value=ei;slider.max=n-1;slider.value=selected;host.querySelector('output').textContent=`${selected+1} / ${n}`;
    const left=62, slabW=55, barX=150, barW=Math.max(40,W-barX-45), max=Math.max(...p,1e-30),total=p.reduce((a,b)=>a+b,0);
    text(12,19,`${n} bins × ${dz} nm · modeled window ${n*dz} nm`);
    text(left,34,'Surface');text(barX,34,'Exit yield / bin');
    text(W-20,19,`max ${max.toPrecision(3)}`,'end');
    node('line',{x1:left-8,y1:top,x2:W-20,y2:top,stroke:'var(--fg)'});
    // Aggregate dense profiles only for painting; selected-bin values stay exact.
    const stride=Math.max(1,Math.ceil(n/180));
    for(let start=0;start<n;start+=stride) {
      const stop=Math.min(start+stride,n),y=top+h*start/n,hh=h*(stop-start)/n;
      const value=p.slice(start,stop).reduce((a,b)=>a+b,0)/(stop-start);
      const active=start<count;
      node('rect',{x:left,y,width:slabW,height:hh,fill:active?'var(--accent)':'var(--line)',
        'fill-opacity':.15+.55*value/max,stroke:'var(--bg)','stroke-width':.5});
      const bar=node('rect',{x:barX,y,width:barW*value/max,height:Math.max(.3,hh-.5),
        fill:active?'var(--accent)':'var(--line)'});
      const tip=document.createElementNS('http://www.w3.org/2000/svg','title');
      tip.textContent=`${start*dz}–${stop*dz} nm; mean yield/bin ${value.toPrecision(4)}`;bar.append(tip);
    }
    for(const fraction of [0,.5,1]) {
      const y=top+h*fraction;text(left-9,y+4,`${(n*dz*fraction).toFixed(0)}`,'end');
    }
    text(12,260,'z (nm)');
    const cy=top+h*count/n;
    node('line',{x1:left-5,y1:cy,x2:W-20,y2:cy,stroke:'var(--cut)','stroke-width':2,'stroke-dasharray':'5 3'});
    text(W-20,263,`99% window: ${count*dz} nm (${count} slices)`,'end');
    const sy=top+h*selected/n;
    node('rect',{x:left-3,y:sy,width:slabW+6,height:Math.max(2,h/n),fill:'none',stroke:'var(--fg)','stroke-width':2});
    node('line',{x1:barX,y1:sy+h/(2*n),x2:barX+barW,y2:sy+h/(2*n),stroke:'var(--fg)','stroke-width':1});
    const cum=p.slice(0,selected+1).reduce((a,b)=>a+b,0);
    host.querySelector('.detail').textContent=`Slice ${selected+1}: ${selected*dz}–${(selected+1)*dz} nm · `+
      `yield ${p[selected].toPrecision(4)} · cumulative ${total?(100*cum/total).toFixed(2):'0.00'}% of this energy's yield. `+
      `${selected<count?'Used':'Outside integration window'}. PRISM propagation point: ${(selected+1)*dz} nm.`;
  }
  const names=['energy_index','slice_index'];for(const name of names)model.on(`change:${name}`,draw);
  const observer=new ResizeObserver(draw);observer.observe(host.querySelector('section'));draw();
  return ()=>{observer.disconnect();for(const name of names)model.off(`change:${name}`,draw);};
}
export default {render};
