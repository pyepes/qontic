import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";

// ═══════════════════════════════════════════════════════════════════════════════
// Coordinate system:
//   Beam travels along +Z
//   Magnet gradient n̂(φ) lies in XY-plane: n̂ = (sinφ, cosφ, 0)
//   φ=0  → gradient along +Y  (splits up/down in view)
//   φ=90 → gradient along +X  (splits left/right)
//   Detector is a flat screen perpendicular to Z at Z_DET
//
// Magnet: two poles separated along n̂, very thin in Z (delta-function field).
//   The whole magnet group rotates around Z so poles always straddle the beam.
//   Gap between poles = 2 * POLE_GAP, wide enough for Gaussian packet + deflection.
//
// Bohmian: particle position at z=0 determines outcome.
//   Critical boundary = circle in XY plane at z=0 oriented ⊥ to n̂.
// ═══════════════════════════════════════════════════════════════════════════════

const Z_SRC   = -4.5;   // 25% closer to magnet
const Z_MAG   =  0;
const Z_DET   =  6;
const POLE_GAP = 2.4;   // wider gap
const POLE_H   = 1.75;  // half original height
const POLE_W   = 0.55;  // pole width (perpendicular to n̂ in XY)
const POLE_D   = 0.15;  // pole depth (along Z — thin for delta function)

const SIG    = 0.38;
const KICK   = 2.0;
const STEPS  = 130;
const PERIOD = 230;
const BG_N   = 55;
const MAX_P  = 20;

// ── Finite-field + free-spreading model constants ─────────────────────────────
// Rayleigh length z_R: packet doubles in width at z = ±z_R from waist
// z_R = m σ₀² / ℏ — we set it in simulation units
const Z_RAYLEIGH = 3.5;   // spreading scale (beam length ≈ 10 units total)
// Magnet half-length in simulation units (default L=1.0 → ±0.5 around Z_MAG)
const MAG_L_DEFAULT = 1.0;

// Free Gaussian spreading: returns σ(z) given waist σ₀ and Rayleigh length zR
function sigmaAt(z, sig0, zR) {
  // Waist at source Z_SRC — packet expands monotonically toward detector
  return sig0 * Math.sqrt(1 + ((z - Z_SRC) / zR) ** 2);
}

// Bohmian integrate — finite field model
// The wavefunction arms are Gaussians with continuously growing σ and
// centres moving at ±v_kick (proportional to accumulated time in field).
// kick_total = KICK * (L / MAG_L_DEFAULT) — scales with magnet length
function integrateFinite(x0, y0, theta, phi, magL) {
  const { pP, pM } = getProbs(theta, phi);
  const nn = nHat(phi);
  const pts = [];
  let tx = x0, ty = y0;
  const kickTotal = KICK * magL;          // total momentum kick ∝ magnet length
  const zMagLo = Z_MAG - magL / 2;
  const zMagHi = Z_MAG + magL / 2;

  for (let i = 0; i <= STEPS; i++) {
    const z = lerp(Z_SRC, Z_DET, i / STEPS);
    pts.push(new THREE.Vector3(tx, ty, z));

    // Post-magnet-entry: arms have separated; compute Bohmian guidance
    if (z > zMagLo) {
      // Fraction of kick accumulated so far
      const kickFrac = clamp((z - zMagLo) / (zMagHi - zMagLo + 0.001), 0, 1);
      const sep = kickFrac * kickTotal;
      // Spreading σ at this z
      const sig = sigmaAt(z, SIG, Z_RAYLEIGH);
      const rp2 = ((tx - nn.x * sep) ** 2 + (ty - nn.y * sep) ** 2) / sig ** 2;
      const rm2 = ((tx + nn.x * sep) ** 2 + (ty + nn.y * sep) ** 2) / sig ** 2;
      const rhoP = pP * Math.exp(-rp2), rhoM = pM * Math.exp(-rm2);
      const vn = (rhoP - rhoM) / (rhoP + rhoM + 1e-12) * kickTotal / STEPS;
      tx += nn.x * vn; ty += nn.y * vn;
    }
  }
  return pts;
}

function lerp(a, b, t) { return a + (b - a) * t; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function nHat(phi) {
  const p = phi * Math.PI / 180;
  return new THREE.Vector3(Math.sin(p), Math.cos(p), 0);
}
function getProbs(theta, phi) {
  const a = (theta - phi) * Math.PI / 180;
  const c = Math.cos(a / 2), s = Math.sin(a / 2);
  return { pP: c * c, pM: s * s };
}

// Bohmian trajectory integration
function integrate(x0, y0, theta, phi) {
  const { pP, pM } = getProbs(theta, phi);
  const nn = nHat(phi);
  const pts = [];
  let tx = x0, ty = y0;
  for (let i = 0; i <= STEPS; i++) {
    const ph = i / STEPS;
    pts.push(new THREE.Vector3(tx, ty, lerp(Z_SRC, Z_DET, ph)));
    if (ph >= 0.5) {
      const pp  = (ph - 0.5) / 0.5;
      const sep = pp * KICK, sig = SIG * (1 + pp * 0.5);
      const rp2 = ((tx - nn.x * sep) ** 2 + (ty - nn.y * sep) ** 2) / sig ** 2;
      const rm2 = ((tx + nn.x * sep) ** 2 + (ty + nn.y * sep) ** 2) / sig ** 2;
      const rhoP = pP * Math.exp(-rp2), rhoM = pM * Math.exp(-rm2);
      const vn = (rhoP - rhoM) / (rhoP + rhoM + 1e-12) * KICK / STEPS;
      tx += nn.x * vn; ty += nn.y * vn;
    }
  }
  return pts;
}

function getOutcome(pts, phi) {
  const nn = nHat(phi), e = pts[pts.length - 1];
  return (e.x * nn.x + e.y * nn.y) > 0 ? 1 : -1;
}

// Binary-search for the Bohmian critical nComp in initial-position space.
// Particles starting with initial n̂-projection > critN always reach +outcome;
// particles below critN always reach -outcome.
// Search is valid because: (a) guiding velocity is strictly along n̂, so
// the n̂-component evolves independently of the transverse component;
// (b) the velocity is a smooth monotone function near the critical plane,
// guaranteeing a unique crossing.
function findCritNInitial(theta, phi, integrateFn) {
  const { pP } = getProbs(theta, phi);
  if (Math.abs(pP - 0.5) < 1e-6) return 0;  // symmetric: plane at 0 by symmetry
  const p = phi * Math.PI / 180;
  const nx = Math.sin(p), ny = Math.cos(p);
  // Returns final n̂-projection: positive means +outcome, negative means -outcome
  const testSign = nc => {
    const tpts = integrateFn(nc * nx, nc * ny);
    const last = tpts[tpts.length - 1];
    return last.x * nx + last.y * ny;
  };
  // Increasing nc always increases final nComp (order-preserving dynamics).
  // Binary search: lo gives negative sign, hi gives positive sign.
  let lo = -3, hi = 3;
  for (let iter = 0; iter < 40; iter++) {
    const mid = (lo + hi) / 2;
    if (testSign(mid) > 0) hi = mid; else lo = mid;
    if (hi - lo < 0.001) break;
  }
  return (lo + hi) / 2;
}

// Random Gaussian initial positions with Born-rule outcome pre-assignment.
// Each particle's isUp is drawn with P(isUp) = pP (exact Born rule).
// The position is then reflected in n̂ to guarantee it lies on the correct
// side of critN — the true Bohmian critical plane found by findCritNInitial.
// Because dynamics is 1-D along n̂, this guarantees getOutcome() == isUp.
function makeInitXY(n, phi, pP, critN) {
  const p = phi * Math.PI / 180;
  const nx = Math.sin(p), ny = Math.cos(p);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const isUp = Math.random() < pP;
    let x, y, attempts = 0;
    do {
      // Box-Muller
      const u1 = Math.random(), u2 = Math.random();
      const r = SIG * 0.7 * Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10)));
      const a = 2 * Math.PI * u2;
      x = r * Math.cos(a);
      y = r * Math.sin(a);
      let nComp = x * nx + y * ny;
      // Reflect n̂-component to the correct side of critN
      if ((isUp && nComp < critN) || (!isUp && nComp > critN)) {
        const shift = 2 * (critN - nComp);
        x += shift * nx; y += shift * ny;
        nComp = 2 * critN - nComp;
      }
      attempts++;
      // Reject if too close to critical plane (ambiguous zone)
      if (Math.abs(nComp - critN) > 0.04) break;
    } while (attempts < 20);
    pts.push({ x, y, isUp });
  }
  return pts;
}

const PALETTE = [
  0x44ffaa, 0xff6633, 0x66aaff, 0xffcc33, 0xee44ff, 0x44ccff,
  0xffaa44, 0xaa44ff, 0x44ffee, 0xff4488, 0x88ff44, 0xff8844,
  0x4488ff, 0xffee44, 0xff44cc, 0x44ffcc, 0xcc44ff, 0xff4444, 0x44ff44, 0x4444ff,
];


// ── SIMULATION PANEL COMPONENT (outside App so identity is stable) ──────────
const VIEWS = ['collapse','pilot','manyworlds'];
const VIEW_LABELS = {collapse:'Collapse', pilot:'Pilot Wave', manyworlds:'Many Worlds'};
const VIEW_COLORS = {collapse:'#ff9966', pilot:'#44ddff', manyworlds:'#bb88ff'};
const VIEW_DESC = {
  collapse:  'Copenhagen: wave collapses to one outcome on detection.',
  pilot:     'de Broglie-Bohm: particles follow definite trajectories guided by the wave.',
  manyworlds:'Everett: all outcomes occur in branching parallel worlds. Wave only.',
};

function fmtWorlds(exp) {
  // worlds = 2^exp — display as 10^y where y = exp * log10(2)
  if (exp === 0) return '1';
  const log10 = exp * Math.log10(2);          // exact base-10 exponent
  const floorExp = Math.floor(log10);
  const mantissa = Math.pow(10, log10 - floorExp); // 1.0 – 9.999…
  const supMap = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴',
    '5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
  const expStr = String(floorExp).split('').map(d => supMap[d] || d).join('');
  if (floorExp < 2) return Math.round(Math.pow(10, log10)).toLocaleString();
  return mantissa.toFixed(2) + ' × 10' + expStr;
}


// ── Tooltip wrapper ──────────────────────────────────────────────────────────
const Tip = ({text, children}) => {
  const [show, setShow] = React.useState(false);
  const [pos,  setPos]  = React.useState({x:0, y:0});
  const ref = React.useRef(null);
  return (
    <span ref={ref} style={{position:'relative', display:'block'}}
      onMouseEnter={e => { setShow(true); }}
      onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span style={{
          position:'absolute', bottom:'calc(100% + 6px)', left:'50%',
          transform:'translateX(-50%)',
          background:'rgba(8,20,55,0.97)', border:'1px solid rgba(80,140,255,0.4)',
          borderRadius:5, padding:'5px 9px', fontSize:11, color:'#b8d4ff',
          whiteSpace:'pre-wrap', maxWidth:200, lineHeight:1.5,
          zIndex:999, pointerEvents:'none', fontFamily:"'Courier New',monospace",
          boxShadow:'0 4px 16px rgba(0,0,30,0.7)',
        }}>{text}</span>
      )}
    </span>
  );
};

// ── Histogram inset ─────────────────────────────────────────────────────────
function Histogram({ up, down, pP, pM }) {
  const n        = up + down;
  const total    = n || 1;
  const upFrac   = up   / total;
  const downFrac = down / total;
  const upPct    = Math.round(upFrac   * 100);
  const downPct  = Math.round(downFrac * 100);
  const expUpPct   = Math.round(pP * 100);
  const expDownPct = Math.round(pM * 100);
  const sigma  = n > 0 ? Math.sqrt(upFrac * downFrac / n) * 100 : 0;
  const sigStr = n > 0 ? ' ±' + (sigma < 0.5 ? sigma.toFixed(1) : Math.round(sigma)) + '%' : '';
  return (
    <div style={{ fontFamily:"'Courier New',monospace", fontSize:11, color:'#b8d4ff', minWidth:158 }}>
      <div style={{ fontSize:10, color:'#7a9ece', textTransform:'uppercase',
        letterSpacing:'0.12em', marginBottom:8 }}>
        Detector <span style={{color:'#8aaedd'}}>n={n}</span>
      </div>
      {[{label:'▲ +n̂', color:'#44ee66', grad:'#22aa44,#44ee66', pct:upPct,   expPct:expUpPct,   count:up},
        {label:'▼ −n̂', color:'#ff5533', grad:'#aa2211,#ff5533', pct:downPct, expPct:expDownPct, count:down}
       ].map(({label,color,grad,pct,expPct,count}) => (
        <div key={label} style={{ marginBottom:9 }}>
          <div style={{ display:'flex', justifyContent:'space-between',
            marginBottom:3, alignItems:'baseline' }}>
            <span style={{ color }}>{label}</span>
            <span style={{ color:'#b8d4ff' }}>{count} · {pct}%{sigStr}</span>
          </div>
          <div style={{ height:7, background:'rgba(15,30,70,0.6)', borderRadius:3,
            position:'relative', overflow:'visible' }}>
            <div style={{ height:'100%', borderRadius:3, transition:'width 0.35s',
              width: pct+'%', background:'linear-gradient(90deg,'+grad+')' }} />
            {n > 0 && (
              <div style={{ position:'absolute', top:0, bottom:0,
                left: Math.max(0, pct - sigma)+'%',
                width: Math.min(100, 2*sigma)+'%',
                background:'rgba(255,255,255,0.09)',
                borderLeft:'1px solid rgba(255,255,255,0.28)',
                borderRight:'1px solid rgba(255,255,255,0.28)',
                borderRadius:2, pointerEvents:'none' }} />
            )}
            <div style={{ position:'absolute', top:-2, bottom:-2, width:2, borderRadius:1,
              background:'rgba(200,210,255,0.50)', left: expPct+'%',
              boxShadow:'0 0 4px rgba(180,200,255,0.3)' }} />
          </div>
        </div>
      ))}
      <div style={{ borderTop:'1px solid rgba(60,100,200,0.30)', paddingTop:4,
        fontSize:9, color:'#7a9ece' }}>│ Born rule  ±σ bracket</div>
    </div>
  );
}
const TB = ({on, onClick, children}) => (
  <button onClick={onClick} style={{
    display:'block', width:'100%', padding:'5px 8px', marginBottom:4,
    background: on ? 'rgba(40,80,180,0.5)' : 'rgba(15,30,70,0.5)',
    border:'1px solid ' + (on ? '#5588cc' : '#334466'),
    borderRadius:5, color: on ? '#c8e8ff' : '#7090b8',
    cursor:'pointer', fontSize:13, fontFamily:'monospace', textAlign:'left',
  }}>{on ? '◉' : '○'}  {children}</button>
);

const PB = ({vals, cur, onSel, ac, ab, ic}) => (
  <div style={{display:'flex', gap:3, flexWrap:'wrap', marginBottom:5}}>
    {vals.map(v => (
      <button key={v} onClick={() => onSel(v)} style={{
        flex:1, padding:'3px 0', fontSize:11,
        background: cur===v ? 'rgba('+ac+',0.25)' : 'rgba(10,22,55,0.6)',
        border:'1px solid ' + (cur===v ? ab : 'rgba(60,100,200,0.25)'),
        borderRadius:4, color: cur===v ? ic : '#7090b8',
        cursor:'pointer', fontFamily:'monospace',
      }}>{v}°</button>
    ))}
  </div>
);

const SL = ({label, tip, children}) => (
  <div style={{marginBottom:10}}>
    <Tip text={tip || null}>
      <div style={{fontSize:13, color:'#7ab8ff', marginBottom:4,
        textTransform:'uppercase', letterSpacing:'0.08em',
        cursor: tip ? 'help' : 'default',
        borderBottom: tip ? '1px dotted rgba(100,160,255,0.4)' : 'none',
        display:'inline-block'}}>{label}</div>
    </Tip>
    {children}
  </div>
);

const SimPanel = React.memo(({
  interp, setInterp, worlds,
  theta, setTheta, thetaRef,
  phi, setPhi, phiRef,
  nPart, setNPart, nPartRef,
  speed, setSpeed, speedRef,
  pP, pM,
  running, setRunning,
  showWave, setShowWave,
  showParticles, setShowParticles,
  resetHits,
  showExpert, setShowExpert,
  wSig, setWSig, wSigRef,
  wMode, setWMode,
  wBright, setWBright, wBrightRef,
  wAlpha, setWAlpha, wAlphaRef,
  fieldModel, setFieldModel,
  magL, setMagL, magLRef,
}) => {
  const vc = VIEW_COLORS[interp];
  const [leftTab, setLeftTab] = React.useState('controls');
  return (
    <div style={{display:'flex', flexDirection:'row',
      boxSizing:'border-box', fontFamily:"'Courier New',monospace", color:'#e8f2ff',
      height:'100%'}}>
      {/* Vertical tab strip on left */}
      <div style={{display:'flex', flexDirection:'column', flexShrink:0, width:22,
        background:'rgba(4,10,30,0.7)', borderRight:'1px solid rgba(40,80,180,0.35)'}}>
        {['controls','expert'].map(t => (
          <button key={t} onClick={() => setLeftTab(t)} style={{
            writingMode:'vertical-rl', transform:'rotate(180deg)',
            padding:'12px 4px', fontSize:10, fontFamily:'monospace',
            textTransform:'uppercase', letterSpacing:'0.1em', cursor:'pointer', border:'none',
            background: leftTab===t ? 'rgba(40,80,200,0.3)' : 'transparent',
            color: leftTab===t ? '#88bbff' : '#4a6a9a',
            borderLeft: leftTab===t ? '2px solid #5588ff' : '2px solid transparent',
            flex: leftTab===t ? 'none' : 'none',
          }}>{t}</button>
        ))}
      </div>
      {/* Controls tab */}
      <div style={{display: leftTab==='controls' ? 'flex' : 'none',
        flexDirection:'column', gap:10, padding:'10px 9px',
        overflowY:'auto', flex:1}}>

      <SL label="View" tip={"Quantum interpretation:\nCollapse: wavefunction collapses on measurement\nPilot Wave: particles follow definite trajectories\nMany Worlds: all outcomes happen in parallel"}>
        <Tip text={'Click to cycle views:\nCollapse → Pilot Wave → Many Worlds'}>
        <button onClick={() => setInterp(VIEWS[(VIEWS.indexOf(interp)+1)%3])} style={{
          display:'block', width:'100%', padding:'7px 10px', marginBottom:5,
          background:'rgba('+(interp==='collapse'?'200,80,40':interp==='pilot'?'30,160,220':'120,70,220')+',0.18)',
          border:'2px solid '+vc, borderRadius:6, color:vc,
          cursor:'pointer', fontSize:13, fontFamily:'monospace', fontWeight:700, textAlign:'center',
        }}>{'>'} {VIEW_LABELS[interp]}</button>
        </Tip>
        <div style={{fontSize:12, color:'#99b8e8', lineHeight:1.6}}>{VIEW_DESC[interp]}</div>
      </SL>

      {interp === 'manyworlds' && (
        <div style={{background:'rgba(100,60,220,0.15)', border:'1px solid rgba(160,120,255,0.5)',
          borderRadius:7, padding:'8px 10px'}}>
          <div style={{fontSize:12, color:'#c8b8ff', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:4}}>Parallel worlds</div>
          <div style={{fontSize:22, fontWeight:700, color:'#ffffff'}}>{fmtWorlds(worlds)}</div>
          <div style={{fontSize:11, color:'#d0c4f8', marginTop:3}}>{Math.round(worlds / Math.max(nPart, 1))} cycles × {nPart} particles</div>
        </div>
      )}

      <SL label={'Spin θ = ' + theta + '°'} tip={'Initial spin direction of the particle\nθ=0°: spin-up |+z⟩\nθ=90°: equal superposition\nθ=180°: spin-down |-z⟩'}>
        <input type="range" min={0} max={180} step={1} defaultValue={theta}
          ref={thetaRef}
          onInput={e => setTheta(+e.target.value)}
          style={{width:'100%', accentColor:'#5090f0', marginBottom:5}}/>
        <PB vals={[0,45,90,135,180]} cur={theta} onSel={setTheta}
          ac="60,100,255" ab="rgba(80,140,255,0.7)" ic="#aaccff"
/>
        <div style={{fontSize:12, color:'#a0c0ee'}}>
          {theta===0 ? '|+z⟩ - spin up' : theta===180 ? '|-z⟩ - spin down'
            : theta===90 ? 'Equal superposition'
            : 'cos('+(theta/2).toFixed(0)+'°)|+⟩ + sin('+(theta/2).toFixed(0)+'°)|-⟩'}
        </div>
      </SL>

      <SL label={'Magnet φ = ' + phi + '°'} tip={'Orientation of the Stern-Gerlach magnet\ngradient axis in the XY plane\nφ=0°: splits along +y\nφ=90°: splits along +x'}>
        <input type="range" min={0} max={180} step={1} defaultValue={phi}
          ref={phiRef}
          onInput={e => setPhi(+e.target.value)}
          style={{width:'100%', accentColor:'#ff8844', marginBottom:5}}/>
        <PB vals={[0,45,90,135,180]} cur={phi} onSel={setPhi}
          ac="255,136,68" ab="rgba(255,136,68,0.7)" ic="#ffaa66"
/>
      </SL>

      <SL label={'Particles N = ' + nPart} tip={'Number of particles per simulation cycle\nEach particle follows a Bohmian trajectory\ndetermined by its initial position in |\u03c8|²'}>
        <input type="range" min={1} max={20} step={1} defaultValue={nPart}
          ref={nPartRef}
          onInput={e => setNPart(+e.target.value)}
          style={{width:'100%', accentColor:'#44ffaa'}}/>
        <div style={{fontSize:12, color:'#e0eeff', marginTop:2}}>
          {interp==='pilot' ? 'trajectories shown'
            : interp==='manyworlds' ? '2^'+nPart+' branches/cycle'
            : nPart+' hits per cycle'}
        </div>
      </SL>

      <SL label={'Speed ×' + speed.toFixed(1)} tip={'Simulation playback speed\n×1 = normal  ×3 = fast  ×0.25 = slow motion'}>
        <input type="range" min={0.25} max={4} step={0.25} defaultValue={speed}
          ref={speedRef}
          onInput={e => setSpeed(+e.target.value)}
          style={{width:'100%', accentColor:'#ffcc44'}}/>
        <div style={{display:'flex', justifyContent:'space-between', fontSize:10, color:'#506080', marginTop:2}}>
          <span>slow</span><span>normal</span><span>fast</span>
        </div>
      </SL>

      <SL label="Beam split" tip={"Born rule probabilities\nP(±n̂) = cos²/sin²((θ-φ)/2)\nFraction of particles landing\non each detector"}>
        <div style={{fontSize:13, lineHeight:1.9}}>
          <span style={{color:'rgba(50,220,120,0.9)'}}>P(+n̂) </span>
          <b>{Math.round(pP*100)}%</b>{'  '}
          <span style={{color:'rgba(255,80,40,0.9)'}}>P(-n̂) </span>
          <b>{Math.round(pM*100)}%</b>
        </div>
        <div style={{height:5, background:'rgba(20,40,100,0.5)', borderRadius:3}}>
          <div style={{height:'100%', borderRadius:3, transition:'width 0.2s',
            width:Math.round(pP*100)+'%', background:'linear-gradient(90deg,#22cc66,#44ff88)'}}/>
        </div>
        <div style={{fontSize:11, color:'#88aadd', marginTop:2}}>cos²((θ-φ)/2)</div>
      </SL>

      <SL label="Controls" tip={"Simulation controls:\nPlay/Pause the animation\nToggle wave and trajectory display\nClear detector hits"}>
        {/* Row 1: Play/Pause + Clear */}
        <div style={{display:'flex', gap:4, marginBottom:4}}>
          <Tip text="Pause or resume the animation">
            <button onClick={() => setRunning(!running)} style={{
              flex:1, padding:'6px 4px', textAlign:'center',
              background: running ? 'rgba(20,55,130,0.6)' : 'rgba(25,80,40,0.6)',
              border:'1px solid '+(running ? 'rgba(70,130,255,0.4)' : 'rgba(60,200,80,0.35)'),
              borderRadius:5, color: running ? '#88bbff' : '#66dd88',
              cursor:'pointer', fontSize:13, fontFamily:'monospace',
            }}>{running ? '⏸ Pause' : '▶ Play'}</button>
          </Tip>
          <Tip text={'Clear all detector hits\nand reset world counter'}>
            <button onClick={resetHits} style={{
              flex:1, padding:'6px 4px', textAlign:'center',
              background:'rgba(15,30,70,0.5)', border:'1px solid #334466',
              borderRadius:5, color:'#b0ccee',
              cursor:'pointer', fontSize:13, fontFamily:'monospace',
            }}>✕ Clear</button>
          </Tip>
        </div>
        {/* Row 2: Wave + Trajectories (pilot only) */}
        <div style={{display:'flex', gap:4}}>
          <Tip text={'Show/hide the |ψ|² wave packet\nvisualization (volumetric slab)'}>
            <button onClick={() => setShowWave(!showWave)} style={{
              flex:1, padding:'5px 4px', textAlign:'center',
              background: showWave ? 'rgba(40,80,180,0.5)' : 'rgba(15,30,70,0.5)',
              border:'1px solid '+(showWave ? '#5588cc' : '#334466'),
              borderRadius:5, color: showWave ? '#c8e8ff' : '#7090b8',
              cursor:'pointer', fontSize:12, fontFamily:'monospace',
            }}>{showWave ? '◉' : '○'} Wave</button>
          </Tip>
          {interp === 'pilot' && (
            <Tip text={'Show/hide Bohmian particle\ntrajectories guided by the wave'}>
              <button onClick={() => setShowParticles(!showParticles)} style={{
                flex:1, padding:'5px 4px', textAlign:'center',
                background: showParticles ? 'rgba(40,80,180,0.5)' : 'rgba(15,30,70,0.5)',
                border:'1px solid '+(showParticles ? '#5588cc' : '#334466'),
                borderRadius:5, color: showParticles ? '#c8e8ff' : '#7090b8',
                cursor:'pointer', fontSize:12, fontFamily:'monospace',
              }}>{showParticles ? '◉' : '○'} Trajectories</button>
            </Tip>
          )}
        </div>
      </SL>



      <div style={{fontSize:11, color:'#9ab8dd', lineHeight:1.8, marginTop:'auto',
        borderTop:'1px solid rgba(50,80,180,0.15)', paddingTop:8}}>
        <div style={{color:'#7890b0'}}>Drag: orbit  Right: pan  Scroll: zoom</div>
      </div>
      </div>{/* end controls tab */}

      {/* Expert tab */}
      <div style={{display: leftTab==='expert' ? 'flex' : 'none',
        flexDirection:'column', gap:14, padding:'10px 9px',
        overflowY:'auto', flex:1}}>

        <div>
          <div style={{fontSize:11, color:'#7ab8ff', marginBottom:5, textTransform:'uppercase', letterSpacing:'0.07em'}}>Wave style</div>
          <div style={{display:'flex', gap:4, marginBottom:4}}>
            {['old','new'].map(m => (
              <Tip key={m} text={m==='old' ? 'Bright core, alpha ∝ dens²\nSharp Gaussian falloff' : 'Uniform brightness\nPhase ripples clearly visible'}>
                <button onClick={() => setWMode(m)} style={{
                  flex:1, padding:'5px 0', fontSize:11, fontFamily:'monospace',
                  background: wMode===m ? 'rgba(60,100,255,0.3)' : 'rgba(10,22,55,0.6)',
                  border:'1px solid '+(wMode===m ? 'rgba(80,140,255,0.8)' : 'rgba(60,100,200,0.25)'),
                  borderRadius:4, color: wMode===m ? '#aaccff' : '#7090b8', cursor:'pointer',
                }}>{m==='old' ? '⬛ Dense' : '〰 Wavefront'}</button>
              </Tip>
            ))}
          </div>
          <div style={{fontSize:10, color:'#506080'}}>
            {wMode==='old' ? 'Bright core, sharp falloff' : 'Uniform — phase ripples visible'}
          </div>
        </div>

        <div>
          <div style={{fontSize:11, color:'#7ab8ff', marginBottom:5, textTransform:'uppercase', letterSpacing:'0.07em'}}>Field model</div>
          <div style={{display:'flex', gap:4, marginBottom:4}}>
            {[['delta','δ Delta'],['finite','▬ Finite']].map(([m,label]) => (
              <Tip key={m} text={m==='delta'
                ? 'Impulsive kick at z=0\n(Norsen 2014 model)\nInstant separation'
                : 'Finite magnet length L\n+ free Gaussian spreading\nGradual separation + dispersion'}>
                <button onClick={() => setFieldModel(m)} style={{
                  flex:1, padding:'5px 0', fontSize:11, fontFamily:'monospace',
                  background: fieldModel===m ? 'rgba(60,100,255,0.3)' : 'rgba(10,22,55,0.6)',
                  border:'1px solid '+(fieldModel===m ? 'rgba(80,140,255,0.8)' : 'rgba(60,100,200,0.25)'),
                  borderRadius:4, color: fieldModel===m ? '#aaccff' : '#7090b8', cursor:'pointer',
                }}>{label}</button>
              </Tip>
            ))}
          </div>
          {fieldModel === 'finite' && (
            <div style={{marginTop:6}}>
              <div style={{fontSize:11, color:'#aaccff', marginBottom:3}}>Magnet length L = {magL.toFixed(1)}</div>
              <input type="range" min={0.2} max={4.0} step={0.1} defaultValue={magL}
                ref={magLRef} onInput={e => setMagL(+e.target.value)}
                style={{width:'100%', accentColor:'#ffcc44'}}/>
              <div style={{display:'flex', justifyContent:'space-between', fontSize:10, color:'#506080'}}>
                <span>short</span><span>long</span>
              </div>
            </div>
          )}
        </div>

        <div>
          <div style={{fontSize:11, color:'#7ab8ff', marginBottom:3, textTransform:'uppercase', letterSpacing:'0.07em'}}>Wave packet σ = {wSig.toFixed(2)}</div>
          <input type="range" min={0.2} max={1.2} step={0.05} defaultValue={wSig}
            ref={wSigRef} onInput={e => setWSig(+e.target.value)}
            style={{width:'100%', accentColor:'#88aaff'}}/>
          <div style={{display:'flex', justifyContent:'space-between', fontSize:10, color:'#506080', marginTop:2}}>
            <span>narrow</span><span>wide</span>
          </div>
        </div>

        <div>
          <div style={{fontSize:11, color:'#7ab8ff', marginBottom:3, textTransform:'uppercase', letterSpacing:'0.07em'}}>Brightness = {wBright.toFixed(2)}</div>
          <input type="range" min={0.1} max={3.0} step={0.05} defaultValue={0.45}
            ref={wBrightRef} onInput={e => setWBright(+e.target.value)}
            style={{width:'100%', accentColor:'#ffaa44'}}/>
          <div style={{display:'flex', justifyContent:'space-between', fontSize:10, color:'#506080', marginTop:2}}>
            <span>dim</span><span>bright</span>
          </div>
        </div>

        <div>
          <div style={{fontSize:11, color:'#7ab8ff', marginBottom:3, textTransform:'uppercase', letterSpacing:'0.07em'}}>Max opacity = {wAlpha.toFixed(2)}</div>
          <input type="range" min={0.05} max={0.95} step={0.05} defaultValue={0.20}
            ref={wAlphaRef} onInput={e => setWAlpha(+e.target.value)}
            style={{width:'100%', accentColor:'#44ffaa'}}/>
          <div style={{display:'flex', justifyContent:'space-between', fontSize:10, color:'#506080', marginTop:2}}>
            <span>ghost</span><span>solid</span>
          </div>
        </div>

        <div style={{fontSize:11, color:'#9ab8dd', lineHeight:1.8, marginTop:'auto',
          borderTop:'1px solid rgba(50,80,180,0.15)', paddingTop:8}}>
          <div style={{color:'#7890b0'}}>Drag: orbit  Right: pan  Scroll: zoom</div>
        </div>
      </div>{/* end expert tab */}

    </div>
  );
});


// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const mountRef = useRef(null);
  const thetaRef = useRef(null);
  const phiRef   = useRef(null);
  const nPartRef = useRef(null);
  const wSigRef    = useRef(null);
  const wBrightRef  = useRef(null);
  const wAlphaRef   = useRef(null);
  const magLRef     = useRef(null);
  const speedRef    = useRef(null);
  const T        = useRef(null);   // Three.js objects
  const S        = useRef({        // mutable simulation state (no re-render)
    theta: 90, phi: 0, nPart: 6, wSig: 0.55, wMode: 'old', wBright: 0.45, wAlpha: 0.20,
    fieldModel: 'delta', magL: 1.0, speed: 1,
    showWave: true, showParticles: true, running: true,
    interp: 'pilot',  // 'collapse' | 'pilot' | 'manyworlds'
    tick: 0, dirty: true,
    camR: 15, camTheta: -2.5, camPhi: 0.22,
    target: new THREE.Vector3(0, 0, 0),
    drag: null,
    hits: [],
    hitStats: { up: 0, down: 0 },
    worlds: 0,
  });

  const [theta,       setThetaUI]   = useState(90);
  const [phi,         setPhiUI]     = useState(0);
  const [nPart,       setNPartUI]   = useState(6);
  const [showWave,    setWaveUI]    = useState(true);
  const [showParticles, setShowPartUI] = useState(true);
  const [interp,      setInterpUI]  = useState('pilot');
  const [running,     setRunUI]     = useState(true);
  const [probs,       setProbs]     = useState({ pP: 0.5, pM: 0.5 });
  const [worlds,      setWorldsUI]  = useState(0);
  const [wSig,        setWSigUI]    = useState(0.55);
  const [wMode,       setWModeUI]   = useState('old');
  const [fieldModel,  setFieldModelUI] = useState('delta');
  const [magL,        setMagLUI]       = useState(1.0);
  const [wBright,     setWBrightUI] = useState(0.45);
  const [wAlpha,      setWAlphaUI]  = useState(0.20);
  const [activeTab,   setActiveTab] = useState('sim');
  const [showExpert,  setShowExpert] = useState(false);
  const [hitCounts,   setHitCountsUI] = useState({ up: 0, down: 0 });
  const [camPreset,   setCamPreset]   = useState('3d');
  const [speed,       setSpeedUI]     = useState(1);

  const setTheta   = v => { S.current.theta   = v; S.current.dirty = true; setThetaUI(v);  setProbs(getProbs(v, S.current.phi)); if (T.current) T.current.clearHits(); if (thetaRef.current) thetaRef.current.value = v; };
  const setPhi     = v => { S.current.phi     = v; S.current.dirty = true; setPhiUI(v);    setProbs(getProbs(S.current.theta, v)); if (T.current) T.current.clearHits(); if (phiRef.current) phiRef.current.value = v; };
  const setNPart   = v => { S.current.nPart   = v; S.current.dirty = true; setNPartUI(v); if (nPartRef.current) nPartRef.current.value = v; };
  const setWSig    = v => { S.current.wSig    = v; setWSigUI(v); };
  const setWMode      = v => { S.current.wMode      = v; setWModeUI(v); };
  const setFieldModel = v => { S.current.fieldModel = v; setFieldModelUI(v); S.current.dirty = true; };
  const setMagL       = v => { S.current.magL       = v; setMagLUI(v);       S.current.dirty = true; };
  const setWBright = v => { S.current.wBright = v; setWBrightUI(v); };
  const setWAlpha  = v => { S.current.wAlpha  = v; setWAlphaUI(v); };
  const setShowWave     = v => { S.current.showWave      = v; setWaveUI(v); };
  const setShowParticles= v => { S.current.showParticles = v; setShowPartUI(v); };
  const setInterp       = v => {
    S.current.interp = v; setInterpUI(v);
    // Reset worlds when switching interpretation
    S.current.worlds = 0; setWorldsUI(0);
    if (T.current) {
      T.current.clearHits();
      // Immediately hide Bohmian particle dots/glows/lines so they don't
      // flash for one frame when switching away from pilot-wave mode.
      T.current.fDots.forEach(m  => { m.visible = false; });
      T.current.fGlows.forEach(m => { m.visible = false; });
      T.current.fLines.forEach(fl => { fl.line.visible = false; });
    }
  };
  const setRunning = v => { S.current.running = v; setRunUI(v); };
  const setSpeed   = v => { S.current.speed = v; setSpeedUI(v); if (speedRef.current) speedRef.current.value = v; };
  const resetHits  = () => {
    S.current.hits = [];
    S.current.hitStats = { up: 0, down: 0 };
    S.current.worlds = 0; setWorldsUI(0);
    setHitCountsUI({ up: 0, down: 0 });
    if (T.current) T.current.clearHits();
  };

  // ── Camera presets ─────────────────────────────────────────────────────────
  const CAM_PRESETS = {
    '3d': { camR: 15,  camTheta: -2.5,          camPhi:  0.22 },
    'xy': { camR: 14,  camTheta:  0,             camPhi:  0    },  // from +Z → see XY (detector face)
    'xz': { camR: 14,  camTheta:  0,             camPhi:  Math.PI / 2 - 0.01 }, // from +Y → see XZ (top)
    'yz': { camR: 14,  camTheta: -Math.PI / 2,   camPhi:  0    },  // from −X → see YZ (side)
  };
  const setCameraPreset = preset => {
    const p = CAM_PRESETS[preset];
    if (!p) return;
    Object.assign(S.current, p);
    if (T.current?.updateCam) T.current.updateCam();
    setCamPreset(preset);
  };

  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    // ── Renderer ──────────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setClearColor(0x07101e, 1);
    renderer.domElement.style.cssText = 'display:block;width:100%;height:100%;';
    el.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 200);

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const sun = new THREE.DirectionalLight(0x88aaff, 0.9);
    sun.position.set(3, 5, 3); scene.add(sun);

    // Resize
    function resize() {
      const w = el.clientWidth || 700, h = el.clientHeight || 440;
      renderer.setSize(w, h, false);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    resize();
    const ro = new ResizeObserver(resize); ro.observe(el);

    // Camera
    function updateCam() {
      const s = S.current;
      const { camR: r, camTheta: th, camPhi: ph, target: tg } = s;
      camera.position.set(
        tg.x + r * Math.sin(th) * Math.cos(ph),
        tg.y + r * Math.sin(ph),
        tg.z + r * Math.cos(th) * Math.cos(ph)
      );
      camera.lookAt(tg);
    }
    updateCam();

    // ── Beam axis ─────────────────────────────────────────────────────────────
    scene.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, Z_SRC), new THREE.Vector3(0, 0, Z_DET),
      ]),
      new THREE.LineBasicMaterial({ color: 0x1a3a6e, transparent: true, opacity: 0.4 })
    ));

    // ── Source ────────────────────────────────────────────────────────────────
    const src = new THREE.Mesh(
      new THREE.SphereGeometry(0.13, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0x88ccff })
    );
    src.position.z = Z_SRC; scene.add(src);

    // Spin direction arrow at source — shows initial spin state in XY plane
    // For state cos(θ/2)|+z⟩ + sin(θ/2)|−z⟩, the Bloch vector in our XY view
    // points at angle θ from +Y: direction = (sin θ, cos θ, 0)
    const spinDir = new THREE.Vector3(0, 1, 0); // updated each frame
    const spinArrow = new THREE.ArrowHelper(
      spinDir, new THREE.Vector3(0, 0, Z_SRC),
      0.9, 0xffffff, 0.22, 0.12
    );
    scene.add(spinArrow);

    // ── Detector screen — PlaneGeometry in XY, no rotation needed ─────────────
    // A PlaneGeometry default lies in XY plane → its normal points along +Z → faces beam. Correct.
    const detGrp = new THREE.Group();
    detGrp.position.z = Z_DET;
    scene.add(detGrp);

    detGrp.add(new THREE.Mesh(
      new THREE.PlaneGeometry(5, 5),
      new THREE.MeshBasicMaterial({
        color: 0x081830, transparent: true, opacity: 0.55,
        side: THREE.DoubleSide, depthWrite: false,
      })
    ));
    // Border frame
    const detBorder = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-2.5,-2.5,0), new THREE.Vector3( 2.5,-2.5,0),
        new THREE.Vector3( 2.5, 2.5,0), new THREE.Vector3(-2.5, 2.5,0),
      ]),
      new THREE.LineBasicMaterial({ color: 0x2244aa, transparent: true, opacity: 0.7 })
    );
    detGrp.add(detBorder);
    // "DETECTOR" crosshair
    [[[-2,0],[2,0]], [[0,-2],[0,2]]].forEach(([a,b]) => {
      detGrp.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a,0), new THREE.Vector3(...b,0)]),
        new THREE.LineBasicMaterial({ color: 0x112244, transparent: true, opacity: 0.4 })
      ));
    });

    // Hit splashes on detector — pool of ring meshes
    const HIT_POOL = 60;
    const hitMeshes = Array.from({ length: HIT_POOL }, () => {
      const m = new THREE.Mesh(
        new THREE.RingGeometry(0.05, 0.22, 24),
        new THREE.MeshBasicMaterial({
          color: 0xffffff, transparent: true, opacity: 0,
          side: THREE.DoubleSide, depthWrite: false,
        })
      );
      detGrp.add(m); return m;
    });
    const hitDots = Array.from({ length: HIT_POOL }, () => {
      const m = new THREE.Mesh(
        new THREE.CircleGeometry(0.07, 16),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide })
      );
      detGrp.add(m); return m;
    });
    let hitCount = 0;

    function addHit(x, y, color) {
      const idx = hitCount % HIT_POOL;
      hitMeshes[idx].position.set(x, y, 0.01);
      hitMeshes[idx].material.color.set(color);
      hitMeshes[idx].material.opacity = 0.75;
      hitDots[idx].position.set(x, y, 0.02);
      hitDots[idx].material.color.set(color);
      hitDots[idx].material.opacity = 0.95;
      hitCount++;
    }
    function clearHitsFunc() {
      hitMeshes.forEach(m => m.material.opacity = 0);
      hitDots.forEach(m => m.material.opacity = 0);
      hitCount = 0;
      S.current.hitStats = { up: 0, down: 0 };
      setHitCountsUI({ up: 0, down: 0 });
    }

    // ── Magnet group ──────────────────────────────────────────────────────────
    // magGrp rotates around Z axis to align poles with n̂(phi).
    // In local space: poles are always at (0, ±POLE_GAP, 0).
    // Pole geometry: tall along local-Y, thin along local-Z (beam), medium along local-X.
    const magGrp = new THREE.Group(); scene.add(magGrp);

    const poleGeo = new THREE.BoxGeometry(POLE_W, POLE_H, POLE_D);
    const matN = new THREE.MeshPhongMaterial({ color: 0x1133cc, transparent: true, opacity: 0.88 });
    const matS = new THREE.MeshPhongMaterial({ color: 0xcc1122, transparent: true, opacity: 0.88 });
    const poleN = new THREE.Mesh(poleGeo, matN);
    const poleS = new THREE.Mesh(poleGeo, matS);
    poleN.position.y =  POLE_GAP;
    poleS.position.y = -POLE_GAP;
    [poleN, poleS].forEach(p => {
      p.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(poleGeo),
        new THREE.LineBasicMaterial({ color: 0x4466dd, transparent: true, opacity: 0.4 })
      ));
      magGrp.add(p);
    });

    // n̂ arrow — placed outside magnet so it doesn't rotate with it
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0.5),
      1.2, 0x66aaff, 0.26, 0.14
    );
    scene.add(arrow);

    // ── Wave packet — volumetric slab with GLSL shader ──────────────────────
    // A stack of translucent XY planes rendered with an analytical |ψ|² shader.
    // The shader computes the Gaussian intensity + phase colour per fragment.
    const N_SLABS = 40;           // fewer slabs — less accumulation at centre
    const SLAB_HALF = 3.2;        // half-size of each plane in XY
    const K_WAVE = 3.5;

    const wSlabUniforms = {
      uSigXY:  { value: 0.55 },
      uSigZ:   { value: 0.88 },
      uCx:     { value: 0.0 },   // centre X (pre-split) / arm+ X (post)
      uCy:     { value: 0.0 },
      uCx2:    { value: 0.0 },   // arm− centre X (post-split)
      uCy2:    { value: 0.0 },
      uPp:     { value: 0.5 },   // P(+n̂)
      uPm:     { value: 0.5 },   // P(-n̂)
      uWz:     { value: 0.0 },   // packet centre Z
      uSlabZ:  { value: 0.0 },   // this slab's Z position
      uPhase:  { value: 0.0 },   // phase advance
      uIsPost: { value: 0.0 },   // 0=pre-split, 1=post-split
      uOpacity:{ value: 0.9 },
      uMode:   { value: 1.0 },  // 0=old(dens^3 bright), 1=new(sqrt dim)
      uBright: { value: 1.0 },
      uAlphaMax:{ value: 0.38 },
    };

    const wSlabVert = `
      varying vec2 vUv;
      varying vec3 vPos;
      void main(){
        vUv = uv;
        vPos = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const wSlabFrag = `
      uniform float uSigXY, uSigZ, uCx, uCy, uCx2, uCy2;
      uniform float uPp, uPm, uWz, uSlabZ, uPhase, uIsPost, uOpacity;
      uniform float uMode, uBright, uAlphaMax;
      varying vec2 vUv;
      varying vec3 vPos;

      float gauss2(float x, float y, float cx, float cy, float sig){
        float dx=(x-cx)/sig, dy=(y-cy)/sig;
        return exp(-0.5*(dx*dx+dy*dy));
      }
      float gaussZ(float z, float cz, float sigz){
        float dz=(z-cz)/sigz;
        return exp(-0.5*dz*dz);
      }

      void main(){
        float x = vPos.x, y = vPos.y, z = uSlabZ;
        float gz = gaussZ(z, uWz, uSigZ);

        vec3 col;
        float dens;
        float phase = cos(${K_WAVE.toFixed(1)} * (z - uWz));
        float cp = phase * 0.5 + 0.5;

        if(uIsPost < 0.5){
          // Pre-split: single packet
          float g = gauss2(x, y, uCx, uCy, uSigXY);
          dens = g * gz;
          // Colour: dark blue trough → cyan peak, phase in colour only not brightness
          col = vec3(mix(0.02, 0.40, cp),
                     mix(0.20, 0.85, cp),
                     mix(0.50, 1.00, cp));
        } else {
          // Post-split: two arms, each normalised by its own probability weight
          float gp = gauss2(x, y, uCx,  uCy,  uSigXY);
          float gm = gauss2(x, y, uCx2, uCy2, uSigXY);
          // Weight each arm equally in density so both appear at same brightness
          dens = max(gp, gm) * gz;
          float tp = (gp * uPp) / (gp * uPp + gm * uPm + 1e-6);
          vec3 colP = vec3(mix(0.0,0.25,cp), mix(0.4,1.0,cp), mix(0.0,0.35,cp));
          vec3 colM = vec3(mix(0.4,1.0,cp), mix(0.0,0.2,cp), mix(0.0,0.08,cp));
          col = mix(colM, colP, tp);
        }

        vec2 uvC = vUv - 0.5;
        float vig = 1.0 - smoothstep(0.38, 0.50, length(uvC));
        float alpha;
        vec3 finalCol;
        if(uMode < 0.5){
          // OLD style: dense glowing cloud, brighter core
          if(dens < 0.08) discard;
          float d2 = dens * dens;
          alpha = d2 * vig * uOpacity * uBright * 8.0;
          alpha = clamp(alpha, 0.0, uAlphaMax * 1.8);
          finalCol = col * (0.5 + 0.5 * dens) * uBright;
        } else {
          // NEW style: translucent, particles visible through wave
          if(dens < 0.12) discard;
          alpha = sqrt(dens) * gz * vig * uOpacity * uBright * 3.5;
          alpha = clamp(alpha, 0.0, uAlphaMax);
          finalCol = col * 0.85 * uBright;
        }
        if(alpha < 0.006) discard;
        gl_FragColor = vec4(finalCol, alpha);
      }
    `;

    // Create one material with shared uniforms; each slab mesh clones it
    // but we use instanced approach: N_SLABS plane meshes sharing the material,
    // each with its own uSlabZ set via onBeforeRender
    const wSlabMeshes = [];
    const wSlabGeo = new THREE.PlaneGeometry(SLAB_HALF*2, SLAB_HALF*2, 1, 1);
    for(let i = 0; i < N_SLABS; i++){
      const mat = new THREE.ShaderMaterial({
        vertexShader: wSlabVert,
        fragmentShader: wSlabFrag,
        uniforms: THREE.UniformsUtils.clone(wSlabUniforms),
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(wSlabGeo, mat);
      // Fixed Z position spanning full beam — never moved, only uniforms updated
      const fixedZ = Z_SRC + (i + 0.5) / N_SLABS * (Z_DET - Z_SRC);
      mesh.position.z = fixedZ;
      mat.uniforms.uSlabZ.value = fixedZ;
      scene.add(mesh);
      wSlabMeshes.push(mesh);
    }
    // wSamples kept as empty stub for compatibility
    const wSamples = [];
    const wPos = new Float32Array(0);
    const wCol = new Float32Array(0);
    const wGeo = new THREE.BufferGeometry();
    const wMat = new THREE.PointsMaterial({size:0.001});
    const wPoints = new THREE.Points(wGeo, wMat);
    // Don't add wPoints to scene — we use slabs instead

    // Critical ring at z=0
    const cRingPts = Array.from({ length: 65 }, (_, i) => {
      const a = i / 64 * Math.PI * 2;
      return new THREE.Vector3(Math.cos(a) * 0.9, Math.sin(a) * 0.9, 0);
    });
    const cRing = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(cRingPts),
      new THREE.LineBasicMaterial({ color: 0xffee44, transparent: true, opacity: 0 })
    );
    scene.add(cRing);

    // ── Background cloud ──────────────────────────────────────────────────────
    const bgPos = new Float32Array(BG_N * 3);
    const bgCol = new Float32Array(BG_N * 3);
    const bgSz  = new Float32Array(BG_N);
    const bgGeo = new THREE.BufferGeometry();
    bgGeo.setAttribute("position", new THREE.BufferAttribute(bgPos, 3));
    bgGeo.setAttribute("color",    new THREE.BufferAttribute(bgCol, 3));
    bgGeo.setAttribute("size",     new THREE.BufferAttribute(bgSz,  1));
    const bgMat = new THREE.ShaderMaterial({
      vertexShader: `
        attribute float size; attribute vec3 color; varying vec3 vColor;
        void main(){
          vColor=color;
          vec4 mv=modelViewMatrix*vec4(position,1.0);
          gl_PointSize=size*(280.0/-mv.z);
          gl_Position=projectionMatrix*mv;
        }`,
      fragmentShader: `
        varying vec3 vColor;
        void main(){
          float d=length(gl_PointCoord-vec2(0.5));
          if(d>0.5)discard;
          gl_FragColor=vec4(vColor,(1.0-smoothstep(0.15,0.5,d))*0.82);
        }`,
      transparent: true, vertexColors: true, depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const bgPoints = new THREE.Points(bgGeo, bgMat);
    scene.add(bgPoints);
    const bgSt = Array.from({ length: BG_N }, (_, i) => ({
      phase: i / BG_N,
      x0: (Math.random() - 0.5) * SIG * 1.4,
      y0: (Math.random() - 0.5) * SIG * 1.4,
      dx: 0, dy: 0,
    }));

    // ── Featured Bohmian particles ────────────────────────────────────────────
    const fDots = Array.from({ length: MAX_P }, (_, i) => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(0.10, 10, 10),
        new THREE.MeshBasicMaterial({ color: PALETTE[i], transparent: true, opacity: 0 })
      );
      scene.add(m); return m;
    });
    const fGlows = Array.from({ length: MAX_P }, (_, i) => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(0.23, 10, 10),
        new THREE.MeshBasicMaterial({ color: PALETTE[i], transparent: true, opacity: 0, depthWrite: false })
      );
      scene.add(m); return m;
    });
    const fLines = Array.from({ length: MAX_P }, () => {
      const pos = new Float32Array((STEPS + 1) * 3);
      const col = new Float32Array((STEPS + 1) * 3);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color",    new THREE.BufferAttribute(col, 3));
      const line = new THREE.Line(geo,
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 })
      );
      scene.add(line);
      return { geo, pos, col, line };
    });

    // Track whether each featured particle has registered a hit this cycle
    const hitRegistered = new Array(MAX_P).fill(false);

    let trajs = [];
    function rebuild() {
      const s = S.current;
      const integrateFn = (x, y) => s.fieldModel === 'finite'
        ? integrateFinite(x, y, s.theta, s.phi, s.magL || MAG_L_DEFAULT)
        : integrate(x, y, s.theta, s.phi);
      const { pP: pPr } = getProbs(s.theta, s.phi);
      const critN = findCritNInitial(s.theta, s.phi, integrateFn);
      const xy = makeInitXY(s.nPart, s.phi, pPr, critN);
      trajs = xy.map(({ x, y, isUp }) => ({ pts: integrateFn(x, y), isUp }));
      const mid = Math.round(STEPS * 0.5);
      trajs.forEach(({ pts, isUp }, i) => {
        const fl = fLines[i];
        const oc = isUp ? 1 : -1;
        const [ar, ag, ab] = oc > 0
          ? [0x22/255, 0xee/255, 0x66/255]
          : [1.0,      0x44/255, 0x22/255];
        pts.forEach((p, j) => {
          fl.pos[j*3]=p.x; fl.pos[j*3+1]=p.y; fl.pos[j*3+2]=p.z;
          const t = clamp((j - mid) / 15, 0, 1);
          fl.col[j*3]  = lerp(0.67, ar, t);
          fl.col[j*3+1]= lerp(0.80, ag, t);
          fl.col[j*3+2]= lerp(1.00, ab, t);
        });
        fl.geo.attributes.position.needsUpdate = true;
        fl.geo.attributes.color.needsUpdate    = true;
        fl.geo.setDrawRange(0, pts.length);
        // Only show trajectory lines when in pilot-wave mode
        fl.line.visible = (s.interp === 'pilot');
        hitRegistered[i] = false;
      });
      for (let i = s.nPart; i < MAX_P; i++) {
        fLines[i].line.visible = false;
        fDots[i].visible  = false;
        fGlows[i].visible = false;
      }
    }
    rebuild();

    // ── Input ─────────────────────────────────────────────────────────────────
    function onDown(e) {
      S.current.drag = { btn: e.button ?? 0, x: e.clientX, y: e.clientY };
      el.setPointerCapture(e.pointerId);
    }
    function onMove(e) {
      const s = S.current;
      if (!s.drag) return;
      const dx = e.clientX - s.drag.x, dy = e.clientY - s.drag.y;
      s.drag.x = e.clientX; s.drag.y = e.clientY;
      if (s.drag.btn === 0) {
        s.camTheta -= dx * 0.007;
        s.camPhi = clamp(s.camPhi + dy * 0.007, -1.2, 1.2);
      } else {
        const fwd   = new THREE.Vector3().subVectors(s.target, camera.position).normalize();
        const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0,1,0)).normalize();
        const up    = new THREE.Vector3().crossVectors(right, fwd).normalize();
        const spd   = s.camR * 0.001;
        s.target.addScaledVector(right, -dx * spd);
        s.target.addScaledVector(up,     dy * spd);
      }
      updateCam();
    }
    function onUp(e) {
      S.current.drag = null;
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
    }

    function onWheel(e) {
      e.preventDefault();
      const s = S.current;
      const factor = e.deltaY > 0 ? 1.12 : 0.89;
      // Zoom toward cursor: unproject mouse onto a sphere around target
      const rect = el.getBoundingClientRect();
      const ndcX = ((e.clientX - rect.left) / rect.width)  *  2 - 1;
      const ndcY = ((e.clientY - rect.top)  / rect.height) * -2 + 1;
      const ray  = new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
      // Closest point on ray to current target
      const oc = ray.ray.origin.clone().sub(s.target);
      const b  = oc.dot(ray.ray.direction);
      const hit = ray.ray.origin.clone().addScaledVector(ray.ray.direction, -b);
      // Pull target toward hit by (1-factor) — positive when zooming in
      if (factor < 1) s.target.lerp(hit, clamp(1 - factor, 0, 0.2));
      s.camR = clamp(s.camR * factor, 2, 45);
      updateCam();
    }

    const noCtx = e => e.preventDefault();

    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup",   onUp);
    el.addEventListener("wheel",      onWheel, { passive: false });
    el.addEventListener("contextmenu",noCtx);

    // Touch orbit (1 finger) + pinch zoom (2 fingers)
    let t1 = null, pinch0 = null;
    const onTD = e => {
      if (e.touches.length === 1) t1 = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        pinch0 = Math.sqrt(dx*dx+dy*dy);
        t1 = null;
      }
    };
    const onTM = e => {
      if (e.touches.length === 1 && t1) {
        const dx = e.touches[0].clientX - t1.x, dy = e.touches[0].clientY - t1.y;
        S.current.camTheta -= dx * 0.007;
        S.current.camPhi = clamp(S.current.camPhi + dy * 0.007, -1.2, 1.2);
        t1 = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        updateCam();
      } else if (e.touches.length === 2 && pinch0) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const d = Math.sqrt(dx*dx+dy*dy);
        S.current.camR = clamp(S.current.camR * (pinch0 / d), 2, 45);
        pinch0 = d;
        updateCam();
      }
    };
    const onTE = () => { t1 = null; pinch0 = null; };
    el.addEventListener("touchstart", onTD, { passive: true });
    window.addEventListener("touchmove", onTM, { passive: true });
    window.addEventListener("touchend", onTE);

    T.current = {
      scene, camera, renderer,
      magGrp, arrow, spinArrow,
      wSlabMeshes, wPoints, wPos, wCol, wGeo, wSamples, cRing,
      bgSt, bgPos, bgCol, bgSz, bgGeo, bgPoints,
      fDots, fGlows, fLines,
      addHit, clearHits: clearHitsFunc,
      rebuild, trajs: () => trajs,
      updateCam,
      setHitCounts: uc => setHitCountsUI({ ...uc }),
    };

    // ── Render loop ───────────────────────────────────────────────────────────
    let raf;
    function animate() {
      raf = requestAnimationFrame(animate);
      const s = S.current;
      const Tr = T.current;
      if (!Tr) return;

      if (s.dirty) { Tr.rebuild(); s.dirty = false; }
      if (s.running) s.tick += s.speed;
      updateCam();

      const { pP, pM } = getProbs(s.theta, s.phi);
      const nn      = nHat(s.phi);
      const phiRad  = s.phi * Math.PI / 180;
      const frac    = (s.tick % PERIOD) / PERIOD;
      const wZ      = lerp(Z_SRC, Z_DET, frac);
      const isPost  = frac >= 0.5;
      const pp      = isPost ? (frac - 0.5) / 0.5 : 0;
      const tIdx    = clamp(Math.round(frac * STEPS), 0, STEPS);
      const sw      = s.showWave;
      Tr.bgPoints.visible = sw;

      // ── Magnet: rotate group around Z so n̂ = (sinφ, cosφ, 0) ───────────────
      // magGrp.rotation.z = phiRad means local-Y axis → world n̂.
      // Poles are always at local (0, ±POLE_GAP, 0), so they end up at ±n̂ * POLE_GAP.
      // nHat(phi)=(sinφ,cosφ,0). Rotating CCW by phiRad maps local+Y to (-sinφ,cosφ,0).
      // Need CW rotation (negative) so local+Y → (sinφ,cosφ,0) = nHat.
      Tr.magGrp.rotation.z = -phiRad;
      // Scale magnet depth visually to match field model
      const mDepth = s.fieldModel === 'finite' ? (s.magL || 1.0) / MAG_L_DEFAULT : 1.0;
      Tr.magGrp.scale.z = mDepth;
      // ArrowHelper.setDirection breaks when dir=(0,±1,0); nudge slightly
      const arrowDir = nn.clone();
      if (Math.abs(arrowDir.x) < 0.001 && Math.abs(arrowDir.z) < 0.001)
        arrowDir.x = 0.001;
      arrowDir.normalize();
      Tr.arrow.setDirection(arrowDir);

      // Update spin arrow at source: Bloch vector direction for θ
      const thetaRad = s.theta * Math.PI / 180;
      const spinDir = new THREE.Vector3(Math.sin(thetaRad), Math.cos(thetaRad), 0).normalize();
      Tr.spinArrow.setDirection(spinDir);

      // ── Wave packet — fixed-Z slab update (uniforms only, no position changes) ─
      if (!sw) {
        Tr.wSlabMeshes.forEach(m => { m.visible = false; });
      } else {
        const wSig_ = s.wSig || 0.55;
        const isFinite_ = s.fieldModel === 'finite';
        // Finite model: σ grows with distance from magnet (free spreading)
        // Waist at Z_SRC — spread increases monotonically from source to detector
        const spreadFactor = isFinite_ ? Math.sqrt(1 + ((wZ - Z_SRC) / Z_RAYLEIGH) ** 2) : 1.0;
        const sigXY_ = wSig_ * spreadFactor;
        const sigZ_  = wSig_ * 1.6 * spreadFactor;
        const kickFrac_ = isFinite_
          ? clamp((wZ - (Z_MAG - (s.magL||1.0)/2)) / ((s.magL||1.0) + 0.001), 0, 1)
          : (isPost ? pp : 0);
        const sigPst = wSig_ * spreadFactor * (1.0 + kickFrac_ * 0.40);
        const sigZP  = wSig_ * 1.6 * spreadFactor;
        // No phAdv — phase purely spatial: cos(k*(slabZ-wZ)) in shader

        // Arm centres from actual trajectory means
        const td = Tr.trajs();
        let cPx=0,cPy=0,cMx=0,cMy=0,nP=0,nM=0;
        td.forEach(({ pts, isUp }) => {
          const oc = isUp ? 1 : -1;
          const pt = pts[tIdx];
          if(oc>0){ cPx+=pt.x; cPy+=pt.y; nP++; }
          else    { cMx+=pt.x; cMy+=pt.y; nM++; }
        });
        const mPx = nP>0 ? cPx/nP : nn.x*pp*KICK;
        const mPy = nP>0 ? cPy/nP : nn.y*pp*KICK;
        const mMx = nM>0 ? cMx/nM : -nn.x*pp*KICK;
        const mMy = nM>0 ? cMy/nM : -nn.y*pp*KICK;

        // Per-slab opacity budget spread over all slabs
        const slabOpacity = 1.0;

        Tr.wSlabMeshes.forEach((mesh) => {
          const u = mesh.material.uniforms;
          // uSlabZ was set at creation and never changes — the shader computes
          // the Z-Gaussian falloff based on distance from uWz each frame
          u.uSigXY.value  = isPost ? sigPst : sigXY_;
          u.uSigZ.value   = isPost ? sigZP  : sigZ_;
          u.uCx.value     = isPost ? mPx : 0;
          u.uCy.value     = isPost ? mPy : 0;
          u.uCx2.value    = mMx;
          u.uCy2.value    = mMy;
          u.uPp.value     = pP;
          u.uPm.value     = pM;
          u.uWz.value     = wZ;
          u.uPhase.value  = 0.0;
          const isPostSlab = s.fieldModel === 'finite'
          ? (wZ > Z_MAG - (s.magL||1.0) / 2)
          : isPost;
        u.uIsPost.value = isPostSlab ? 1.0 : 0.0;
          u.uOpacity.value  = slabOpacity;
          u.uMode.value     = s.wMode === 'old' ? 0.0 : 1.0;
          u.uBright.value   = s.wBright || 1.0;
          u.uAlphaMax.value = s.wAlpha  || 0.38;
          mesh.visible      = true;
        });
      }

      // Critical ring at z=0, rotated so it's ⊥ to n̂ in XY
      const nr = Math.max(0, 0.8 - Math.abs(frac - 0.5) * 13);
      Tr.cRing.material.opacity = sw ? nr * 0.65 : 0;
      // Ring lives in XY plane at z=0 — rotate around Z to align gap with n̂
      Tr.cRing.rotation.z = -phiRad;

      // ── Background cloud ───────────────────────────────────────────────────
      Tr.bgSt.forEach((p, i) => {
        if (s.running) p.phase = (p.phase + s.speed / PERIOD) % 1;
        const bpp = p.phase > 0.5 ? (p.phase - 0.5) / 0.5 : 0;
        let bx = p.x0, by = p.y0;
        if (p.phase >= 0.5) {
          const sep2 = bpp*KICK, sig2 = SIG*(1+bpp*0.5);
          const rp2 = ((bx-nn.x*sep2)**2+(by-nn.y*sep2)**2)/sig2**2;
          const rm2 = ((bx+nn.x*sep2)**2+(by+nn.y*sep2)**2)/sig2**2;
          const vn  = (pP*Math.exp(-rp2)-pM*Math.exp(-rm2)) /
                      (pP*Math.exp(-rp2)+pM*Math.exp(-rm2)+1e-12) * KICK*s.speed/PERIOD;
          p.dx += nn.x*vn; p.dy += nn.y*vn; bx += p.dx; by += p.dy;
        }
        if (p.phase < 1/PERIOD) {
          p.x0=(Math.random()-0.5)*SIG*1.4; p.y0=(Math.random()-0.5)*SIG*1.4;
          p.dx=0; p.dy=0;
        }
        Tr.bgPos[i*3]=bx; Tr.bgPos[i*3+1]=by; Tr.bgPos[i*3+2]=lerp(Z_SRC,Z_DET,p.phase);
        // Color: cyan→(green or red) based on local density
        const mix = clamp(bpp*3, 0, 1);
        let cr=0.55, cg=0.80, cb=1.0;
        if (bpp > 0) {
          const sep2=bpp*KICK, sig2=SIG*(1+bpp*0.5);
          const rp2=((bx-nn.x*sep2)**2+(by-nn.y*sep2)**2)/sig2**2;
          const rm2=((bx+nn.x*sep2)**2+(by+nn.y*sep2)**2)/sig2**2;
          const fP=(pP*Math.exp(-rp2))/(pP*Math.exp(-rp2)+pM*Math.exp(-rm2)+1e-12);
          cr=lerp(0.55,lerp(1.0,0.15,fP),mix);
          cg=lerp(0.80,lerp(0.28,0.92,fP),mix);
          cb=lerp(1.00,lerp(0.12,0.45,fP),mix);
        }
        Tr.bgCol[i*3]=cr; Tr.bgCol[i*3+1]=cg; Tr.bgCol[i*3+2]=cb;
        Tr.bgSz[i] = sw ? 0.20+bpp*0.10 : 0;
      });
      Tr.bgGeo.attributes.position.needsUpdate=true;
      Tr.bgGeo.attributes.color.needsUpdate=true;
      Tr.bgGeo.attributes.size.needsUpdate=true;

      // ── Featured particles ─────────────────────────────────────────────────
      const td   = Tr.trajs();

      // Detect wrap-around (new cycle) — reset hit flags
      if (tIdx <= 1) { hitRegistered.fill(false); s.dirty = true; }

      const showP = s.showParticles && s.interp === 'pilot';
      td.forEach(({ pts, isUp }, i) => {
        // Outcome pre-assigned by Born rule; position guaranteed on correct side of critN
        const oc   = i < s.nPart ? (isUp ? 1 : -1) : 1;
        const armC = oc > 0 ? 0x22ee66 : 0xff4422;

        // Hide visual elements when not in pilot view or out of range
        if (i >= s.nPart || !showP) {
          Tr.fDots[i].visible  = false;
          Tr.fGlows[i].visible = false;
          if (i < s.nPart) Tr.fLines[i].line.visible = false;
        } else {
          // Pilot wave: show dots, glows, trajectory lines
          Tr.fLines[i].line.visible = true;
          const pt = pts[tIdx];
          Tr.fDots[i].visible  = true;
          Tr.fGlows[i].visible = true;
          Tr.fDots[i].position.copy(pt);
          Tr.fGlows[i].position.copy(pt);
          Tr.fDots[i].material.opacity  = 0.95;
          Tr.fGlows[i].material.opacity = 0.18;
          const sepF = clamp((frac - 0.5) / 0.12, 0, 1);
          const col  = new THREE.Color().lerpColors(new THREE.Color(0xaaccff), new THREE.Color(armC), sepF);
          Tr.fDots[i].material.color.copy(col);
          Tr.fGlows[i].material.color.copy(col);
          Tr.fLines[i].geo.setDrawRange(0, tIdx + 1);
        }

        // Hit registration — ALL views, ALL particles in range
        if (i < s.nPart && tIdx >= STEPS - 2 && !hitRegistered[i]) {
          hitRegistered[i] = true;
          const endPt = pts[STEPS];
          // All views use the actual trajectory endpoint — same distribution as pilot wave
          Tr.addHit(endPt.x, endPt.y, armC);
          if (oc > 0) s.hitStats.up++; else s.hitStats.down++;
          Tr.setHitCounts(s.hitStats);
          // Many-worlds: each particle measurement adds 1 to the branching exponent (worlds = 2^exp)
          if (s.interp === 'manyworlds') {
            s.worlds += 1;
            setWorldsUI(s.worlds);
          }
        }
      });

      Tr.renderer.render(Tr.scene, Tr.camera);
    }
    animate();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup",   onUp);
      el.removeEventListener("wheel",      onWheel);
      el.removeEventListener("contextmenu",noCtx);
      el.removeEventListener("touchstart", onTD);
      window.removeEventListener("touchmove", onTM);
      window.removeEventListener("touchend",  onTE);
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, []);

  const { pP, pM } = probs;

  const simPanel = (
    <SimPanel
      interp={interp} setInterp={setInterp} worlds={worlds}
      theta={theta} setTheta={setTheta} thetaRef={thetaRef}
      phi={phi} setPhi={setPhi} phiRef={phiRef}
      nPart={nPart} setNPart={setNPart} nPartRef={nPartRef}
      speed={speed} setSpeed={setSpeed} speedRef={speedRef}
      pP={pP} pM={pM}
      running={running} setRunning={setRunning}
      showWave={showWave} setShowWave={setShowWave}
      showParticles={showParticles} setShowParticles={setShowParticles}
      resetHits={resetHits}
      showExpert={showExpert} setShowExpert={setShowExpert}
      wSig={wSig} setWSig={setWSig} wSigRef={wSigRef}
      wMode={wMode} setWMode={setWMode}
      wBright={wBright} setWBright={setWBright} wBrightRef={wBrightRef}
      wAlpha={wAlpha} setWAlpha={setWAlpha} wAlphaRef={wAlphaRef}
      fieldModel={fieldModel} setFieldModel={setFieldModel}
      magL={magL} setMagL={setMagL} magLRef={magLRef}
    />
  );

  // ── THEORY PANEL — iframe with MathJax loaded inside ────────────────────────
  const theoryHtml = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body { margin:0; padding:22px 26px; background:#040a1c; color:#cce0ff;
    font-family:'Georgia',serif; font-size:14px; line-height:1.9; }
  h1 { font-size:20px; color:#aaccff; margin-bottom:4px; }
  .ref { font-size:12px; color:#5878a0; font-style:italic; margin-bottom:24px; }
  h2 { font-size:15px; color:#7ab8ff; font-weight:700; margin:24px 0 8px;
    border-bottom:1px solid rgba(60,120,255,0.25); padding-bottom:5px; }
  h3 { font-size:13px; color:#88ccff; font-weight:700; margin:16px 0 6px; }
  p { margin:8px 0 12px; }
  .eq { margin:12px 0; padding:10px 20px; text-align:center;
    background:rgba(20,45,110,0.5); border:1px solid rgba(80,140,255,0.25);
    border-radius:7px; font-size:15px; overflow-x:auto; }
  .model-box { border-radius:8px; padding:12px 16px; margin:14px 0; }
  .delta-box { background:rgba(20,60,40,0.4); border:1px solid rgba(60,180,100,0.3); }
  .finite-box { background:rgba(20,40,80,0.4); border:1px solid rgba(80,140,255,0.3); }
  .model-label { font-size:11px; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:6px; }
  .delta-label { color:#66dd88; }
  .finite-label { color:#66aaff; }
  a { color:#4488aa; }
  b.collapse { color:#ff9966; }
  b.pilot    { color:#44ddff; }
  b.manyworlds { color:#bb88ff; }
</style>
<script>
MathJax = {
  tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']] },
  options: { skipHtmlTags: ['script','noscript','style','textarea'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
<h1>Theory of the Stern&ndash;Gerlach Effect</h1>
<div class="ref">Based on Norsen (2014), <em>Am. J. Phys.</em> 82, 337</div>

<h2>1. The Quantum State</h2>
<p>A spin-&frac12; particle entering the magnet is prepared in a superposition of spin
eigenstates along the measurement axis $\\hat{n}(\\varphi) = (\\sin\\varphi, \\cos\\varphi, 0)$.</p>
<div style="background:rgba(20,50,90,0.5);border:1px solid rgba(80,140,255,0.3);border-radius:7px;padding:10px 14px;margin:10px 0 14px;font-size:13px;line-height:1.7">
  <b style="color:#7ab8ff">What particle is this?</b><br/>
  The simulation is <em>species-agnostic</em>: it models any spin-&frac12; particle in dimensionless units.
  <ul style="margin:6px 0 0 16px;padding:0">
    <li><b style="color:#f0c060">Silver atoms</b> — the original 1922 Stern&ndash;Gerlach experiment. The unpaired
      4<em>d</em> electron gives the atom spin-&frac12;. The heavy nuclear mass makes
      deflections small and the trajectory nearly classical, which is why the experiment
      was interpretable without quantum mechanics being fully developed.</li>
    <li style="margin-top:6px"><b style="color:#88ddff">Neutrons</b> — the cleanest modern realisation.
      Being electrically neutral, neutrons feel <em>only</em> the magnetic dipole force
      ($\\boldsymbol{\\mu}_n\\cdot\\nabla\\mathbf{B}$) with no Lorentz deflection or
      stray-electric-field contamination. Slow (cold) neutrons have a de&nbsp;Broglie
      wavelength of nanometres, making wave-packet effects and Bohmian trajectory
      experiments directly observable. They are the standard probe in
      fundamental spin-measurement tests today.</li>
    <li style="margin-top:6px"><b style="color:#aaffaa">Electrons</b> — spin-&frac12; but charged, so a
      longitudinal magnetic field also bends the beam via the Lorentz force, masking
      the pure spin deflection. Impractical for a clean Stern&ndash;Gerlach measurement
      in free space.</li>
  </ul>
</div>
<div class="eq">$$|\\Psi\\rangle = \\cos\\tfrac{\\theta}{2}\\,|{+}\\hat{n}\\rangle + \\sin\\tfrac{\\theta}{2}\\,|{-}\\hat{n}\\rangle$$</div>

<h3>Initial wave packet</h3>
<div class="model-box delta-box">
<div class="model-label delta-label">&#x03B4; Delta model</div>
<p style="margin:0">The spatial part is a Gaussian with <em>constant</em> width $\\sigma$, moving rigidly along $z$:</p>
<div class="eq">$$\\psi_0(\\mathbf{r}) \\propto \\exp\\!\\left[-\\frac{x^2+y^2}{4\\sigma^2} - \\frac{(z-z_0)^2}{4\\sigma_z^2}\\right] e^{ikz}$$</div>
</div>

<div class="model-box finite-box">
<div class="model-label finite-label">&#x25AC; Finite field model</div>
<p style="margin:0">The packet spreads freely during propagation. With waist $\\sigma_0$ at the source $z_0$ and Rayleigh length $z_R = m\\sigma_0^2/\\hbar$, the transverse width grows as:</p>
<div class="eq">$$\\sigma(z) = \\sigma_0\\,\\sqrt{1 + \\left(\\frac{z - z_0}{z_R}\\right)^{\\!2}}$$</div>
<p style="margin:4px 0 0">So the full pre-magnet wavefunction is:</p>
<div class="eq">$$\\psi(\\mathbf{r},t) \\propto \\frac{1}{\\sigma(z)}\\exp\\!\\left[-\\frac{x^2+y^2}{4\\sigma(z)^2} - \\frac{(z-vt)^2}{4\\sigma_z^2}\\right] e^{ikz}$$</div>
</div>

<h2>2. The Magnetic Field</h2>

<div class="model-box delta-box">
<div class="model-label delta-label">&#x03B4; Delta model</div>
<p style="margin:0">Following Norsen (2014), the field is an impulsive delta function at $z=0$:</p>
<div class="eq">$$\\mathbf{B}(\\mathbf{r}) = B_0\\,\\delta(z)\\,\\hat{n}$$</div>
<p style="margin:4px 0 0">This imparts an <em>instantaneous</em> momentum kick to each spin component:</p>
<div class="eq">$$\\Delta p_{\\hat{n}} = \\begin{cases} +\\hbar\\kappa & |{+}\\hat{n}\\rangle \\ -\\hbar\\kappa & |{-}\\hat{n}\\rangle \\end{cases}$$</div>
</div>

<div class="model-box finite-box">
<div class="model-label finite-label">&#x25AC; Finite field model</div>
<p style="margin:0">The field has a constant gradient $\\alpha$ over a finite length $L$ centred at $z=0$:</p>
<div class="eq">$$\\mathbf{B}(\\mathbf{r}) = \\alpha\\,(\\hat{n}\\cdot\\mathbf{r}_\\perp)\\,\\hat{n}\\cdot\\mathbf{1}_{|z| \\leq L/2}$$</div>
<p style="margin:4px 0 0">The transverse momentum kick accumulates <em>continuously</em> as the particle traverses the field. The total kick after full traversal equals $\\hbar\\kappa$ (same as delta model), but is spread over length $L$:</p>
<div class="eq">$$\\frac{dp_{\\hat{n}}}{dz} = \\pm\\frac{\\hbar\\kappa}{L}, \\quad |z| \\leq \\tfrac{L}{2}$$</div>
<p style="margin:4px 0 0">The packet width $\\sigma(z)$ continues to grow through and beyond the magnet,
so the two arms emerge broadened and with a smooth continuous separation.</p>
</div>

<h2>3. Post-Magnet Wavefunction</h2>
<p>In both models, after the magnet the state is a superposition of two separating arms:</p>
<div class="eq">$$\\Psi(\\mathbf{r},t) = \\cos\\tfrac{\\theta}{2}\\;\\psi_+(\\mathbf{r},t)\\,|{+}\\hat{n}\\rangle + \\sin\\tfrac{\\theta}{2}\\;\\psi_-(\\mathbf{r},t)\\,|{-}\\hat{n}\\rangle$$</div>

<div class="model-box delta-box">
<div class="model-label delta-label">&#x03B4; Delta model &mdash; arm centres</div>
<div class="eq">$$\\langle\\hat{n}\\rangle_\\pm(t) = \\pm\\frac{\\hbar\\kappa}{m}\\,(t - t_{\\rm mag})$$</div>
<p style="margin:4px 0 0">Width $\\sigma$ remains constant (no free spreading).</p>
</div>

<div class="model-box finite-box">
<div class="model-label finite-label">&#x25AC; Finite model &mdash; arm centres</div>
<div class="eq">$$\\langle\\hat{n}\\rangle_\\pm(z) = \\pm\\kappa\\,\\frac{\\min(z,\\,L/2)}{1}\\cdot\\frac{z - z_{\\rm entry}}{L}$$</div>
<p style="margin:4px 0 0">Width $\\sigma(z)$ keeps growing after the magnet via free evolution.</p>
</div>

<h2>4. Born Rule Probabilities</h2>
<div class="eq">$$P(+\\hat{n}) = \\cos^2\\!\\tfrac{\\theta-\\varphi}{2}, \\qquad P(-\\hat{n}) = \\sin^2\\!\\tfrac{\\theta-\\varphi}{2}$$</div>
<p>These probabilities are <em>identical</em> in both field models &mdash; the Born rule
depends only on the initial spin state and measurement axis, not on field details.</p>

<h2>5. Pilot-Wave (Bohmian) Trajectories</h2>
<p>In the de Broglie&ndash;Bohm interpretation each particle has a definite position
$\\mathbf{Q}(t)$ at all times, evolving under the <em>guidance equation</em>:</p>
<div class="eq">$$\\dot{\\mathbf{Q}} = \\frac{\\hbar}{m}\\,\\operatorname{Im}\\!\\left[\\frac{\\Psi^*\\nabla\\Psi}{|\\Psi|^2}\\right]_{\\mathbf{r}=\\mathbf{Q}}$$</div>
<p>For the two-component state this reduces along $\\hat{n}$ to:</p>
<div class="eq">$$\\dot{Q}_{\\hat{n}} = \\frac{\\rho_+\\, v_+ + \\rho_-\\, v_-}{\\rho_+ + \\rho_-}$$</div>
<p>where $\\rho_\\pm(\\mathbf{Q},t) = |\\psi_\\pm(\\mathbf{Q},t)|^2$.</p>

<div class="model-box delta-box">
<div class="model-label delta-label">&#x03B4; Delta model</div>
<p style="margin:0">Guidance begins at $z=0$. Before the magnet trajectories are straight lines.
The critical boundary is a sharp ring in the $z=0$ plane.</p>
</div>

<div class="model-box finite-box">
<div class="model-label finite-label">&#x25AC; Finite model</div>
<p style="margin:0">Guidance begins at $z = -L/2$ (magnet entry). Because $\\sigma(z)$ grows
continuously and the separation builds gradually, trajectories curve more gently.
The spreading arms overlap for longer, so the critical boundary is a <em>fuzzy zone</em>
rather than a sharp ring.</p>
</div>

<h2>6. The Critical Boundary</h2>
<p>The yellow ring marks where $\\rho_+ = \\rho_-$:</p>
<div class="eq">$$P_+\\,|\\psi_+(\\mathbf{Q})|^2 = P_-\\,|\\psi_-(\\mathbf{Q})|^2$$</div>
<p>Particles on the $+\\hat{n}$ side reach the $+\\hat{n}$ detector. Because initial positions
are $|\\Psi|^2$-distributed (quantum equilibrium), the Born rule is reproduced exactly
in both models.</p>

<h2>7. The Three Interpretations</h2>
<p><b class="collapse">Collapse (Copenhagen):</b> Wavefunction collapses on measurement.
No trajectory prior to detection. The field model affects the <em>shape</em> of collapse
but not its probability.</p>
<p><b class="pilot">Pilot Wave (de Broglie&ndash;Bohm):</b> Both arms evolve; the particle
follows one trajectory. In the finite model the trajectory curves gradually through the
magnet rather than kinking sharply at $z=0$.</p>
<p><b class="manyworlds">Many Worlds (Everett):</b> Both branches are real. After $n$ particles:</p>
<div class="eq">$$N_{\\rm worlds} = 2^n$$</div>

<p style="font-size:12px; color:#445566; border-top:1px solid rgba(40,70,140,0.25);
  padding-top:12px; margin-top:16px;">
  <strong style="color:#607090">References:</strong>
  T. Norsen, &ldquo;The pilot-wave perspective on spin,&rdquo;
  <em>Am. J. Phys.</em> <strong>82</strong>, 337 (2014).
  <a href="https://doi.org/10.1119/1.4848217" target="_blank">doi:10.1119/1.4848217</a>
  &nbsp;&mdash;&nbsp;
  J. D&iacute;az Bulnes &amp; I.S. Oliveira,
  &ldquo;Construction of exact solutions for the Stern-Gerlach effect,&rdquo;
  <em>Braz. J. Phys.</em> <strong>31</strong>, 4 (2001).
</p>
</body>
</html>`
  const theoryPanel = (
    <iframe
      srcDoc={theoryHtml}
      style={{width:'100%', height:'100%', border:'none'}}
      title="Theory"
    />
  );

  const [panelW, setPanelW] = useState(254);
  const resizing = useRef(false);

  const resizeHandleRef = useRef(null);
  useEffect(() => {
    const handle = resizeHandleRef.current;
    if (!handle) return;
    const onMove = e => setPanelW(Math.max(180, Math.min(520, e.clientX)));
    const onUp   = e => { handle.releasePointerCapture(e.pointerId); handle.removeEventListener('pointermove', onMove); };
    const onDown = e => { e.preventDefault(); handle.setPointerCapture(e.pointerId); handle.addEventListener('pointermove', onMove); handle.addEventListener('pointerup', onUp, {once:true}); };
    handle.addEventListener('pointerdown', onDown);
    return () => handle.removeEventListener('pointerdown', onDown);
  }, []);

  return (
    <div style={{width:'100%', height:'100%', overflow:'hidden',
      background:'#07101e', display:'flex', flexDirection:'column'}}>
      <style>{`
        .tbb{padding:8px 18px;cursor:pointer;font-family:monospace;font-size:13px;
          border:none;border-bottom:3px solid transparent;background:transparent;color:#6888aa;}
        .tba{color:#aaddff;border-bottom-color:#4488ff;}
        .tbb:hover{color:#cce0ff;}
        .rh{width:5px;cursor:col-resize;background:rgba(40,80,200,0.15);flex-shrink:0;
          transition:background 0.15s;touch-action:none;user-select:none;}
        .rh:hover,.rh:active{background:rgba(80,140,255,0.4);}
        input[type=range]{touch-action:auto;pointer-events:auto;cursor:pointer;}
      `}</style>
      {/* Tab bar */}
      <div style={{display:'flex', alignItems:'center', height:38, flexShrink:0,
        background:'rgba(4,10,30,0.98)', borderBottom:'1px solid rgba(40,80,180,0.3)',
        paddingLeft:12, gap:4}}>
        <span style={{fontSize:11, color:'#4060a0', fontFamily:'monospace',
          letterSpacing:'0.08em', marginRight:12}}>STERN-GERLACH 3D</span>
        <button className={'tbb'+(activeTab==='sim'?' tba':'')}
          onClick={() => setActiveTab('sim')}>Simulation</button>
        <button className={'tbb'+(activeTab==='theory'?' tba':'')}
          onClick={() => setActiveTab('theory')}>Theory</button>
      </div>
      {/* Main area */}
      <div style={{flex:1, display:'flex', flexDirection:'row', overflow:'hidden', minHeight:0}}>
        {/* Left panel — always controls */}
        <div style={{width:panelW, minWidth:180, flexShrink:0,
          background:'rgba(8,18,45,0.98)', overflowY:'auto', height:'100%'}}>
          {simPanel}
        </div>
        {/* Resize handle */}
        <div className="rh" ref={resizeHandleRef}/>
        {/* Right area — canvas OR theory */}
        <div style={{flex:1, minWidth:0, position:'relative', height:'100%'}}>
          <div ref={mountRef} style={{width:'100%', height:'100%', cursor:'grab',
            display: activeTab==='sim' ? 'block' : 'none'}}/>
          {/* Camera preset buttons — top left */}
          {activeTab==='sim' && (
            <div style={{ position:'absolute', top:10, left:10, display:'flex', gap:4, zIndex:10 }}>
              {[['3d','3D'],['xy','XY'],['xz','XZ'],['yz','YZ']].map(([k,label]) => (
                <button key={k} onClick={() => setCameraPreset(k)} style={{
                  padding:'3px 10px', fontSize:11, fontFamily:'monospace',
                  background: camPreset===k ? 'rgba(80,140,255,0.30)' : 'rgba(4,10,30,0.75)',
                  border:'1px solid '+(camPreset===k ? '#5588ff' : 'rgba(80,140,255,0.28)'),
                  borderRadius:4, color: camPreset===k ? '#aaddff' : '#6888aa',
                  cursor:'pointer', backdropFilter:'blur(6px)', transition:'all 0.13s',
                }}>{label}</button>
              ))}
            </div>
          )}
          {/* Histogram overlay — top right */}
          {activeTab==='sim' && (
            <div style={{ position:'absolute', top:10, right:12, zIndex:10,
              background:'rgba(4,10,30,0.88)', border:'1px solid rgba(80,140,255,0.28)',
              borderRadius:7, padding:'8px 12px',
              backdropFilter:'blur(10px)', boxShadow:'0 4px 18px rgba(0,0,20,0.65)' }}>
              <Histogram up={hitCounts.up} down={hitCounts.down} pP={probs.pP} pM={probs.pM} />
            </div>
          )}
          {activeTab==='theory' && (
            <div style={{position:'absolute', inset:0, overflowY:'auto'}}>
              {theoryPanel}
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
