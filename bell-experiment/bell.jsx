import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";

// ═══════════════════════════════════════════════════════════════════════════
// Bell Experiment — Two entangled spin-½ particles flying apart along ±Z.
//
// Source at origin. Particle A flies toward -Z (left detector at Z=-7).
//                   Particle B flies toward +Z (right detector at Z=+7).
//
// Singlet state: |Ψ⁻⟩ = (1/√2)(|+⟩_A|−⟩_B − |−⟩_A|+⟩_B)
//
// Each particle carries TWO colored wave-packet branches superimposed:
//   Branch ↑ (green):  spin-up component of that particle's reduced state
//   Branch ↓ (orange): spin-down component
//
// Detector A measures along n̂_A(φ_A), Detector B along n̂_B(φ_B).
//
// COPENHAGEN: When A is measured, one branch vanishes instantly on both sides.
//             B's wave function collapses to the correlated outcome.
//
// BOHMIAN: Both branches persist. A's particle position determines its outcome.
//          The joint wave function's guidance equation then forces B's particle
//          to the correlated branch (non-local quantum potential).
// ═══════════════════════════════════════════════════════════════════════════

const Z_SRC  =  0;
const Z_DET  =  7.0;   // detector distance from source
const SIG    =  0.40;  // initial Gaussian width
const KICK   =  1.6;   // transverse separation at detector
const STEPS  =  120;
const PERIOD = 260;    // animation ticks per cycle

// Colors for the two spinor branches
const COLOR_UP   = new THREE.Color(0x22ee66);  // green  — spin ↑
const COLOR_DOWN = new THREE.Color(0xff5522);  // orange — spin ↓
const COLOR_ENTANGLED = new THREE.Color(0x55aaff); // blue — superposition

function lerp(a, b, t) { return a + (b - a) * t; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Measurement basis unit vector in the XY plane
function nHat(phi) {
  const p = phi * Math.PI / 180;
  return new THREE.Vector3(Math.sin(p), Math.cos(p), 0);
}

// Born-rule probabilities for measuring spin-up along n̂ given initial state θ
// For singlet state, if A measures along φ_A and B along φ_B:
//   P(A=+, B=+) = P(A=-, B=-) = sin²(Δφ/2)/2
//   P(A=+, B=-) = P(A=-, B=+) = cos²(Δφ/2)/2
// The correlation is C(φ_A,φ_B) = -cos(φ_A - φ_B)
function singletProbs(phiA, phiB) {
  const delta = (phiA - phiB) * Math.PI / 180;
  const pp = 0.5 * Math.sin(delta / 2) ** 2;  // P(+,+)
  const pm = 0.5 * Math.cos(delta / 2) ** 2;  // P(+,-)
  return { pp, pm, mp: pm, mm: pp };
}

// ── Bohmian guidance for one particle in the Bell setup ──────────────────────
// After measurement of A, the effective wave function of B collapses to the
// correlated spinor. We model the pre-measurement trajectory as guided by
// the superposition of two transverse Gaussians (the two spinor branches).
// The particle's transverse position determines which branch it "follows."
function integrateBohmian(x0, isUpOutcome, phi, side) {
  // side: +1 = particle B (goes to +Z), -1 = particle A (goes to -Z)
  const nn = nHat(phi);
  const pts = [];
  let tx = x0 * nn.x, ty = x0 * nn.y;
  const zSign = side; // direction of travel

  // The two branch centres separate as particle flies outward
  for (let i = 0; i <= STEPS; i++) {
    const frac = i / STEPS;
    const z = zSign * lerp(0, Z_DET, frac);
    pts.push(new THREE.Vector3(tx, ty, z));

    if (frac >= 0.5) {
      const pp2 = (frac - 0.5) / 0.5;
      const sep = pp2 * KICK;
      const sig = SIG * (1 + pp2 * 0.5);
      // Upper branch centre (spin-up along n̂)
      const upCx = nn.x * sep, upCy = nn.y * sep;
      // Lower branch centre (spin-down along n̂)
      const dnCx = -nn.x * sep, dnCy = -nn.y * sep;
      const rUp2 = ((tx - upCx) ** 2 + (ty - upCy) ** 2) / sig ** 2;
      const rDn2 = ((tx - dnCx) ** 2 + (ty - dnCy) ** 2) / sig ** 2;
      const rhoUp = 0.5 * Math.exp(-rUp2);
      const rhoDn = 0.5 * Math.exp(-rDn2);
      const vn = (rhoUp - rhoDn) / (rhoUp + rhoDn + 1e-12) * KICK / STEPS;
      tx += nn.x * vn;
      ty += nn.y * vn;
    }
  }
  return pts;
}

// ── Tooltip ──────────────────────────────────────────────────────────────────
const Tip = ({ text, children }) => {
  const [show, setShow] = useState(false);
  return (
    <span style={{ position: 'relative', display: 'block' }}
      onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && text && (
        <span style={{
          position: 'absolute', bottom: 'calc(100% + 6px)', left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(8,20,55,0.97)', border: '1px solid rgba(80,140,255,0.4)',
          borderRadius: 5, padding: '5px 9px', fontSize: 11, color: '#b8d4ff',
          whiteSpace: 'pre-wrap', maxWidth: 220, lineHeight: 1.5,
          zIndex: 999, pointerEvents: 'none',
          fontFamily: "'Courier New',monospace",
          boxShadow: '0 4px 16px rgba(0,0,30,0.7)',
        }}>{text}</span>
      )}
    </span>
  );
};

// ── Angle preset buttons ─────────────────────────────────────────────────────
const PB = ({ vals, cur, onSel }) => (
  <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', marginBottom: 5 }}>
    {vals.map(v => (
      <button key={v} onClick={() => onSel(v)} style={{
        flex: 1, padding: '3px 0', fontSize: 11,
        background: cur === v ? 'rgba(80,140,255,0.25)' : 'rgba(10,22,55,0.6)',
        border: '1px solid ' + (cur === v ? 'rgba(80,140,255,0.7)' : 'rgba(60,100,200,0.25)'),
        borderRadius: 4, color: cur === v ? '#aaccff' : '#7090b8',
        cursor: 'pointer', fontFamily: 'monospace',
      }}>{v}°</button>
    ))}
  </div>
);

// ── Section label ────────────────────────────────────────────────────────────
const SL = ({ label, tip, children }) => (
  <div style={{ marginBottom: 10 }}>
    <Tip text={tip || null}>
      <div style={{
        fontSize: 13, color: '#7ab8ff', marginBottom: 4,
        textTransform: 'uppercase', letterSpacing: '0.08em',
        cursor: tip ? 'help' : 'default',
        borderBottom: tip ? '1px dotted rgba(100,160,255,0.4)' : 'none',
        display: 'inline-block',
      }}>{label}</div>
    </Tip>
    {children}
  </div>
);

// ── Correlation histogram ────────────────────────────────────────────────────
function CorrelationPanel({ counts, phiA, phiB }) {
  const total = counts.pp + counts.pm + counts.mp + counts.mm || 1;
  const { pp: ePP, pm: ePM, mp: eMP, mm: eMM } = singletProbs(phiA, phiB);
  const corr = (counts.pp + counts.mm - counts.pm - counts.mp) / total;
  const expCorr = -Math.cos((phiA - phiB) * Math.PI / 180);
  const n = counts.pp + counts.pm + counts.mp + counts.mm;
  const rows = [
    { label: '▲▼ (+,−)', color: '#55ddaa', count: counts.pm, exp: ePM },
    { label: '▼▲ (−,+)', color: '#55aadd', count: counts.mp, exp: eMP },
    { label: '▲▲ (+,+)', color: '#ffaa44', count: counts.pp, exp: ePP },
    { label: '▼▼ (−,−)', color: '#ff4466', count: counts.mm, exp: eMM },
  ];
  return (
    <div style={{ fontFamily: "'Courier New',monospace", fontSize: 11, color: '#b8d4ff', minWidth: 168 }}>
      <div style={{ fontSize: 10, color: '#7a9ece', textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 8 }}>
        Coincidences <span style={{ color: '#8aaedd' }}>n={n}</span>
      </div>
      {rows.map(({ label, color, count, exp }) => {
        const pct = Math.round(count / total * 100);
        const expPct = Math.round(exp * 100);
        return (
          <div key={label} style={{ marginBottom: 7 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
              <span style={{ color }}>{label}</span>
              <span>{count} · {pct}%</span>
            </div>
            <div style={{ height: 6, background: 'rgba(15,30,70,0.6)', borderRadius: 3, position: 'relative' }}>
              <div style={{ height: '100%', borderRadius: 3, width: pct + '%', background: color, opacity: 0.7 }} />
              <div style={{ position: 'absolute', top: -2, bottom: -2, width: 2, borderRadius: 1,
                background: 'rgba(200,210,255,0.5)', left: expPct + '%' }} />
            </div>
          </div>
        );
      })}
      <div style={{ borderTop: '1px solid rgba(60,100,200,0.3)', paddingTop: 5, marginTop: 4 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#7a9ece' }}>C(observed)</span>
          <span style={{ color: n > 0 ? '#e8f2ff' : '#405070' }}>{n > 0 ? corr.toFixed(3) : '—'}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#7a9ece' }}>C(QM) = −cos Δφ</span>
          <span style={{ color: '#aaddff' }}>{expCorr.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// ── Control Panel ────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════
const VIEWS = ['collapse', 'bohmian'];
const VIEW_LABELS = { collapse: 'Copenhagen', bohmian: 'Bohmian' };
const VIEW_COLORS = { collapse: '#ff9966', bohmian: '#44ddff' };
const VIEW_DESC = {
  collapse: 'Measurement collapses the wave function. One branch vanishes instantly on both sides.',
  bohmian:  'Both branches persist. Particle A\'s position is guided to an outcome; the joint wave function non-locally steers particle B to the correlated branch.',
};

const ControlPanel = React.memo(({
  interp, setInterp,
  phiA, setPhiA, phiARef,
  phiB, setPhiB, phiBRef,
  speed, setSpeed, speedRef,
  running, setRunning,
  showWave, setShowWave,
  showParticles, setShowParticles,
  resetCounts,
  measuredSide, setMeasuredSide,
  counts, phiAVal, phiBVal,
}) => {
  const vc = VIEW_COLORS[interp];
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 10,
      padding: '10px 9px', overflowY: 'auto', flex: 1,
      fontFamily: "'Courier New',monospace", color: '#e8f2ff',
    }}>
      {/* Interpretation */}
      <SL label="Interpretation" tip="Toggle between quantum interpretations">
        <button onClick={() => setInterp(VIEWS[(VIEWS.indexOf(interp) + 1) % 2])} style={{
          display: 'block', width: '100%', padding: '7px 10px', marginBottom: 5,
          background: 'rgba(' + (interp === 'collapse' ? '200,80,40' : '30,160,220') + ',0.18)',
          border: '2px solid ' + vc, borderRadius: 6, color: vc,
          cursor: 'pointer', fontSize: 13, fontFamily: 'monospace', fontWeight: 700, textAlign: 'center',
        }}>{'>'} {VIEW_LABELS[interp]}</button>
        <div style={{ fontSize: 12, color: '#99b8e8', lineHeight: 1.6 }}>{VIEW_DESC[interp]}</div>
      </SL>

      {/* Measured side — which particle gets measured first */}
      <SL label="Measure first" tip={"Which particle is measured first.\nThe other is the 'remote' particle\nwhose wave function changes non-locally."}>
        <div style={{ display: 'flex', gap: 4 }}>
          {[['A', 'Left (A)'], ['B', 'Right (B)']].map(([k, label]) => (
            <button key={k} onClick={() => setMeasuredSide(k)} style={{
              flex: 1, padding: '5px 0', fontSize: 12, fontFamily: 'monospace',
              background: measuredSide === k ? 'rgba(80,140,255,0.25)' : 'rgba(10,22,55,0.6)',
              border: '1px solid ' + (measuredSide === k ? 'rgba(80,140,255,0.7)' : 'rgba(60,100,200,0.25)'),
              borderRadius: 4, color: measuredSide === k ? '#aaddff' : '#7090b8', cursor: 'pointer',
            }}>{label}</button>
          ))}
        </div>
      </SL>

      {/* Detector A angle */}
      <SL label={'Detector A  φ_A = ' + phiA + '°'} tip={'Measurement axis of detector A\n(left side, particle flying toward −Z)\nφ=0°: measure along +Y\nφ=90°: measure along +X'}>
        <input type="range" min={0} max={180} step={1} defaultValue={phiA}
          ref={phiARef} onInput={e => setPhiA(+e.target.value)}
          style={{ width: '100%', accentColor: '#44ddff', marginBottom: 5 }} />
        <PB vals={[0, 30, 60, 90, 120, 150]} cur={phiA} onSel={setPhiA} />
      </SL>

      {/* Detector B angle */}
      <SL label={'Detector B  φ_B = ' + phiB + '°'} tip={'Measurement axis of detector B\n(right side, particle flying toward +Z)\nφ=0°: measure along +Y\nφ=90°: measure along +X'}>
        <input type="range" min={0} max={180} step={1} defaultValue={phiB}
          ref={phiBRef} onInput={e => setPhiB(+e.target.value)}
          style={{ width: '100%', accentColor: '#ff9966', marginBottom: 5 }} />
        <PB vals={[0, 30, 60, 90, 120, 150]} cur={phiB} onSel={setPhiB} />
      </SL>

      {/* Angular difference readout */}
      <div style={{
        background: 'rgba(20,50,90,0.5)', border: '1px solid rgba(80,140,255,0.3)',
        borderRadius: 6, padding: '7px 10px', fontSize: 12,
      }}>
        <div style={{ color: '#7a9ece', marginBottom: 4 }}>Bell correlation</div>
        <div>Δφ = {Math.abs(phiA - phiB)}°</div>
        <div>C(QM) = −cos(Δφ) = <b style={{ color: '#aaddff' }}>{(-Math.cos((phiA - phiB) * Math.PI / 180)).toFixed(3)}</b></div>
        <div style={{ fontSize: 11, color: '#506090', marginTop: 3 }}>
          {Math.abs(phiA - phiB) === 0 ? 'Perfect anti-correlation' :
           Math.abs(phiA - phiB) === 90 ? 'Uncorrelated' :
           Math.abs(phiA - phiB) === 180 ? 'Perfect correlation' :
           'Partial correlation'}
        </div>
      </div>

      {/* Speed */}
      <SL label={'Speed ×' + speed.toFixed(1)}>
        <input type="range" min={0.25} max={4} step={0.25} defaultValue={speed}
          ref={speedRef} onInput={e => setSpeed(+e.target.value)}
          style={{ width: '100%', accentColor: '#ffcc44' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#506080', marginTop: 2 }}>
          <span>slow</span><span>normal</span><span>fast</span>
        </div>
      </SL>

      {/* Controls */}
      <SL label="Controls">
        <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
          <button onClick={() => setRunning(!running)} style={{
            flex: 1, padding: '6px 4px', textAlign: 'center',
            background: running ? 'rgba(20,55,130,0.6)' : 'rgba(25,80,40,0.6)',
            border: '1px solid ' + (running ? 'rgba(70,130,255,0.4)' : 'rgba(60,200,80,0.35)'),
            borderRadius: 5, color: running ? '#88bbff' : '#66dd88',
            cursor: 'pointer', fontSize: 13, fontFamily: 'monospace',
          }}>{running ? '⏸ Pause' : '▶ Play'}</button>
          <button onClick={resetCounts} style={{
            flex: 1, padding: '6px 4px', textAlign: 'center',
            background: 'rgba(15,30,70,0.5)', border: '1px solid #334466',
            borderRadius: 5, color: '#b0ccee', cursor: 'pointer', fontSize: 13, fontFamily: 'monospace',
          }}>✕ Clear</button>
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button onClick={() => setShowWave(!showWave)} style={{
            flex: 1, padding: '5px 4px', textAlign: 'center',
            background: showWave ? 'rgba(40,80,180,0.5)' : 'rgba(15,30,70,0.5)',
            border: '1px solid ' + (showWave ? '#5588cc' : '#334466'),
            borderRadius: 5, color: showWave ? '#c8e8ff' : '#7090b8',
            cursor: 'pointer', fontSize: 12, fontFamily: 'monospace',
          }}>{showWave ? '◉' : '○'} Wave</button>
          {interp === 'bohmian' && (
            <button onClick={() => setShowParticles(!showParticles)} style={{
              flex: 1, padding: '5px 4px', textAlign: 'center',
              background: showParticles ? 'rgba(40,80,180,0.5)' : 'rgba(15,30,70,0.5)',
              border: '1px solid ' + (showParticles ? '#5588cc' : '#334466'),
              borderRadius: 5, color: showParticles ? '#c8e8ff' : '#7090b8',
              cursor: 'pointer', fontSize: 12, fontFamily: 'monospace',
            }}>{showParticles ? '◉' : '○'} Particles</button>
          )}
        </div>
      </SL>

      {/* Coincidence statistics */}
      <CorrelationPanel counts={counts} phiA={phiAVal} phiB={phiBVal} />

      <div style={{
        fontSize: 11, color: '#9ab8dd', lineHeight: 1.8, marginTop: 'auto',
        borderTop: '1px solid rgba(50,80,180,0.15)', paddingTop: 8,
      }}>
        <div style={{ color: '#7890b0' }}>Drag: orbit  Scroll: zoom</div>
      </div>
    </div>
  );
});

// ═══════════════════════════════════════════════════════════════════════════
// ── Theory Panel ─────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════
const theoryHtml = `<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  body{margin:0;padding:22px 26px;background:#040a1c;color:#cce0ff;
    font-family:'Georgia',serif;font-size:14px;line-height:1.9;}
  h1{font-size:20px;color:#aaccff;margin-bottom:4px;}
  h2{font-size:15px;color:#7ab8ff;font-weight:700;margin:24px 0 8px;
    border-bottom:1px solid rgba(60,120,255,0.25);padding-bottom:5px;}
  h3{font-size:13px;color:#88ccff;font-weight:700;margin:16px 0 6px;}
  p{margin:8px 0 12px;}
  .eq{margin:12px 0;padding:10px 20px;text-align:center;
    background:rgba(20,45,110,0.5);border:1px solid rgba(80,140,255,0.25);
    border-radius:7px;font-size:15px;overflow-x:auto;}
  .box{border-radius:8px;padding:12px 16px;margin:14px 0;}
  .cph{background:rgba(60,20,10,0.4);border:1px solid rgba(255,120,60,0.3);}
  .bhm{background:rgba(10,40,60,0.4);border:1px solid rgba(60,180,220,0.3);}
  .lbl{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;}
  .lbl-c{color:#ff9966;} .lbl-b{color:#44ddff;}
  a{color:#4488aa;}
</style>
<script>
MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},
  options:{skipHtmlTags:['script','noscript','style','textarea']}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head><body>
<h1>Bell Experiment &mdash; Theory</h1>

<h2>1. The Singlet State</h2>
<p>Two spin-&frac12; particles are produced in the maximally entangled singlet state:</p>
<div class="eq">$$|\\Psi^-\\rangle = \\frac{1}{\\sqrt{2}}\\bigl(|{+}\\rangle_A|{-}\\rangle_B - |{-}\\rangle_A|{+}\\rangle_B\\bigr)$$</div>
<p>Particle A flies toward the left detector (angle $\\varphi_A$), particle B toward the right ($\\varphi_B$). Each particle's reduced state is an <em>equal</em> superposition of spin-up and spin-down &mdash; the two colored wave-packet branches you see superimposed on each particle before measurement.</p>

<h2>2. Quantum Correlations</h2>
<p>Born-rule probabilities for the singlet:</p>
<div class="eq">$$P({+},{-}) = P({-},{+}) = \\frac{1}{2}\\cos^2\\!\\tfrac{\\Delta\\varphi}{2}, \\quad P({+},{+}) = P({-},{-}) = \\frac{1}{2}\\sin^2\\!\\tfrac{\\Delta\\varphi}{2}$$</div>
<p>where $\\Delta\\varphi = \\varphi_A - \\varphi_B$. The quantum correlation function is:</p>
<div class="eq">$$C(\\varphi_A,\\varphi_B) = \\langle \\sigma_A\\cdot\\hat{n}_A\\;\\sigma_B\\cdot\\hat{n}_B\\rangle = -\\cos(\\Delta\\varphi)$$</div>

<h2>3. Bell's Inequality</h2>
<p>Any local hidden-variable theory must satisfy Bell's inequality (CHSH form):</p>
<div class="eq">$$|E(a,b) - E(a,b') + E(a',b) + E(a',b')| \\leq 2$$</div>
<p>Quantum mechanics predicts a maximum of $2\\sqrt{2} \\approx 2.828$ (the Tsirelson bound), violating the inequality. The canonical Bell angles are $a=0°, a'=45°, b=22.5°, b'=67.5°$, giving the quantum value $2\\sqrt{2}$.</p>

<h2>4. Two Interpretations</h2>

<div class="box cph">
<div class="lbl lbl-c">Copenhagen &mdash; Wavefunction Collapse</div>
<p style="margin:0">Before measurement both spinor branches (green ↑ and orange ↓) are superimposed on each particle. When detector A fires, the wave function collapses <em>instantaneously</em>: one branch is eliminated on <em>both</em> sides simultaneously. The green branch might survive on A and simultaneously the orange branch might survive on B (anti-correlation for $\\Delta\\varphi=0$). No signal travels between detectors &mdash; but there is a genuine non-local correlation in the Born statistics.</p>
</div>

<div class="box bhm">
<div class="lbl lbl-b">Bohmian &mdash; Pilot-Wave Mechanics</div>
<p style="margin:0">Both branches always persist. Particle A has a definite position guided by the joint wave function $\\Psi(\\mathbf{r}_A,\\mathbf{r}_B,t)$. Its position selects which branch it "rides." Because the guidance equation depends on the <em>full joint</em> wave function, particle B's guiding field changes <em>non-locally and instantaneously</em> when particle A's position changes &mdash; even if A and B are light-years apart. No hidden signaling is possible because positions are not directly observable in equilibrium, but the non-local influence is real and ensures the correct correlations.</p>
<p style="margin:8px 0 0">The guidance equation for particle B:</p>
<div class="eq" style="margin:6px 0 0">$$\\dot{\\mathbf{Q}}_B = \\frac{\\hbar}{m}\\operatorname{Im}\\!\\left[\\frac{\\Psi^*(\\mathbf{Q}_A,\\mathbf{Q}_B)\\,\\nabla_{\\mathbf{r}_B}\\Psi(\\mathbf{Q}_A,\\mathbf{Q}_B)}{|\\Psi(\\mathbf{Q}_A,\\mathbf{Q}_B)|^2}\\right]$$</div>
<p style="margin:6px 0 0">When $Q_A$ crosses the critical surface, the dominant branch in the denominator changes, abruptly changing B's velocity even though nothing physically travels to B.</p>
</div>

<h2>5. What the Colors Mean</h2>
<p><b style="color:#22ee66">Green branch (↑)</b>: the spin-up component of each particle's spinor along the local detector axis.<br/>
<b style="color:#ff5522">Orange branch (↓)</b>: the spin-down component.<br/>
<b style="color:#55aaff">Blue superposition</b>: while both branches overlap during flight, the wave packets are shown in blue to indicate the entangled superposition.<br/>
After measurement, only one branch remains (Copenhagen) or the particle is pushed into one branch (Bohmian).</p>

<h2>6. No Faster-Than-Light Signaling</h2>
<p>Although the correlations are non-local, they cannot be used to transmit information. Observer A cannot control which outcome she gets, so she cannot encode a message in her results. Observer B sees a perfectly random sequence of ↑/↓ outcomes regardless of what A does or what angle A chooses. The correlations only become visible when A and B <em>compare</em> their results classically (at light speed or slower).</p>

<p style="font-size:12px;color:#445566;border-top:1px solid rgba(40,70,140,0.25);
  padding-top:12px;margin-top:16px;">
  <strong style="color:#607090">References:</strong>
  J.S. Bell, &ldquo;On the Einstein–Podolsky–Rosen paradox,&rdquo; <em>Physics</em> <strong>1</strong>, 195 (1964). &mdash;
  J.F. Clauser et al., <em>Phys. Rev. Lett.</em> <strong>23</strong>, 880 (1969). &mdash;
  A. Aspect et al., <em>Phys. Rev. Lett.</em> <strong>49</strong>, 91 (1982). &mdash;
  D. Bohm, <em>Phys. Rev.</em> <strong>85</strong>, 166 (1952).
</p>
</body></html>`;

// ═══════════════════════════════════════════════════════════════════════════
// ── Main App ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════
export default function App() {
  const mountRef   = useRef(null);
  const phiARef    = useRef(null);
  const phiBRef    = useRef(null);
  const speedRef   = useRef(null);

  // Mutable simulation state (no re-render on every tick)
  const S = useRef({
    phiA: 0, phiB: 90,
    speed: 1.0, running: true,
    interp: 'collapse',         // 'collapse' | 'bohmian'
    measuredSide: 'A',
    showWave: true, showParticles: true,
    tick: 0, dirty: true,
    camR: 20, camTheta: 0.3, camPhi: 0.18,
    target: new THREE.Vector3(0, 0, 0),
    drag: null,
    counts: { pp: 0, pm: 0, mp: 0, mm: 0 },
    collapsed: false,      // whether collapse has happened this cycle
    outcomeA: null,        // +1 or -1
    outcomeB: null,
    collapseAtFrac: 0.72,  // fraction into animation when measurement fires
  });

  const [phiA,         setPhiAUI]      = useState(0);
  const [phiB,         setPhiBUI]      = useState(90);
  const [speed,        setSpeedUI]     = useState(1);
  const [running,      setRunUI]       = useState(true);
  const [interp,       setInterpUI]    = useState('collapse');
  const [measuredSide, setMeasuredSideUI] = useState('A');
  const [showWave,     setShowWaveUI]  = useState(true);
  const [showParticles,setShowPartUI]  = useState(true);
  const [counts,       setCountsUI]    = useState({ pp: 0, pm: 0, mp: 0, mm: 0 });
  const [activeTab,    setActiveTab]   = useState('sim');
  const [panelW,       setPanelW]      = useState(270);

  const T = useRef(null); // Three.js objects

  // Setters sync both React state and mutable S ref
  const setPhiA    = v => { S.current.phiA = v; S.current.dirty = true; setPhiAUI(v); if (phiARef.current) phiARef.current.value = v; };
  const setPhiB    = v => { S.current.phiB = v; S.current.dirty = true; setPhiBUI(v); if (phiBRef.current) phiBRef.current.value = v; };
  const setSpeed   = v => { S.current.speed = v; setSpeedUI(v); if (speedRef.current) speedRef.current.value = v; };
  const setRunning = v => { S.current.running = v; setRunUI(v); };
  const setInterp  = v => { S.current.interp = v; setInterpUI(v); S.current.dirty = true; };
  const setMeasuredSide = v => { S.current.measuredSide = v; setMeasuredSideUI(v); };
  const setShowWave      = v => { S.current.showWave = v; setShowWaveUI(v); };
  const setShowParticles = v => { S.current.showParticles = v; setShowPartUI(v); };
  const resetCounts = () => {
    S.current.counts = { pp: 0, pm: 0, mp: 0, mm: 0 };
    setCountsUI({ pp: 0, pm: 0, mp: 0, mm: 0 });
  };

  // Resize handle
  const resizeHandleRef = useRef(null);
  useEffect(() => {
    const handle = resizeHandleRef.current;
    if (!handle) return;
    const onMove = e => setPanelW(Math.max(200, Math.min(500, e.clientX)));
    const onUp   = e => { handle.releasePointerCapture(e.pointerId); handle.removeEventListener('pointermove', onMove); };
    const onDown = e => { e.preventDefault(); handle.setPointerCapture(e.pointerId); handle.addEventListener('pointermove', onMove); handle.addEventListener('pointerup', onUp, { once: true }); };
    handle.addEventListener('pointerdown', onDown);
    return () => handle.removeEventListener('pointerdown', onDown);
  }, []);

  // ── Three.js scene setup ─────────────────────────────────────────────────
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setClearColor(0x07101e, 1);
    renderer.domElement.style.cssText = 'display:block;width:100%;height:100%;';
    el.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(44, 1, 0.1, 200);
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const sun = new THREE.DirectionalLight(0x88aaff, 0.8);
    sun.position.set(3, 5, 3);
    scene.add(sun);

    function resize() {
      const w = el.clientWidth || 700, h = el.clientHeight || 440;
      renderer.setSize(w, h, false);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(el);

    function updateCam() {
      const { camR: r, camTheta: th, camPhi: ph, target: tg } = S.current;
      camera.position.set(
        tg.x + r * Math.sin(th) * Math.cos(ph),
        tg.y + r * Math.sin(ph),
        tg.z + r * Math.cos(th) * Math.cos(ph),
      );
      camera.lookAt(tg);
    }
    updateCam();

    // ── Scene geometry ───────────────────────────────────────────────────────

    // Beam axes (A=left=-Z, B=right=+Z)
    scene.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, -Z_DET - 0.5),
        new THREE.Vector3(0, 0,  Z_DET + 0.5),
      ]),
      new THREE.LineBasicMaterial({ color: 0x1a3a6e, transparent: true, opacity: 0.35 })
    ));

    // Source sphere at origin
    const srcMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    scene.add(srcMesh);

    // Source glow ring (entanglement symbol)
    const glowRingPts = Array.from({ length: 65 }, (_, i) => {
      const a = i / 64 * Math.PI * 2;
      return new THREE.Vector3(Math.cos(a) * 0.45, Math.sin(a) * 0.45, 0);
    });
    const glowRing = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(glowRingPts),
      new THREE.LineBasicMaterial({ color: 0xaaddff, transparent: true, opacity: 0.5 })
    );
    scene.add(glowRing);

    // ── Detectors ────────────────────────────────────────────────────────────
    function makeDetector(zPos, color) {
      const grp = new THREE.Group();
      grp.position.z = zPos;
      // Screen
      grp.add(new THREE.Mesh(
        new THREE.PlaneGeometry(4, 4),
        new THREE.MeshBasicMaterial({
          color: color, transparent: true, opacity: 0.10,
          side: THREE.DoubleSide, depthWrite: false,
        })
      ));
      // Border
      grp.add(new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-2, -2, 0), new THREE.Vector3(2, -2, 0),
          new THREE.Vector3(2,  2, 0), new THREE.Vector3(-2,  2, 0),
        ]),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 })
      ));
      // Crosshair
      [[[-1.6, 0], [1.6, 0]], [[0, -1.6], [0, 1.6]]].forEach(([a, b]) => {
        grp.add(new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a, 0), new THREE.Vector3(...b, 0)]),
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.25 })
        ));
      });
      scene.add(grp);
      return grp;
    }

    const detA = makeDetector(-Z_DET, 0x44ddff); // left, cyan
    const detB = makeDetector( Z_DET, 0xff9966); // right, orange

    // Detector axis arrows (show measurement direction n̂)
    const arrA = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, -Z_DET + 0.3), 1.2, 0x44ddff, 0.26, 0.14);
    const arrB = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0,  Z_DET - 0.3), 1.2, 0xff9966, 0.26, 0.14);
    scene.add(arrA, arrB);

    // Hit pools on each detector
    function makeHitPool(detGrp, n) {
      const splashes = Array.from({ length: n }, () => {
        const m = new THREE.Mesh(
          new THREE.RingGeometry(0.05, 0.20, 20),
          new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false })
        );
        detGrp.add(m); return m;
      });
      const dots = Array.from({ length: n }, () => {
        const m = new THREE.Mesh(
          new THREE.CircleGeometry(0.06, 14),
          new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide })
        );
        detGrp.add(m); return m;
      });
      let count = 0;
      return {
        add(x, y, color) {
          const i = count % n;
          splashes[i].position.set(x, y, 0.01);
          splashes[i].material.color.set(color); splashes[i].material.opacity = 0.7;
          dots[i].position.set(x, y, 0.02);
          dots[i].material.color.set(color); dots[i].material.opacity = 0.95;
          count++;
        },
        clear() {
          splashes.forEach(m => m.material.opacity = 0);
          dots.forEach(m => m.material.opacity = 0);
          count = 0;
        },
      };
    }
    const hitsA = makeHitPool(detA, 60);
    const hitsB = makeHitPool(detB, 60);

    // ── Wave-packet slabs (volumetric, shader) ────────────────────────────────
    // Two sets: one for particle A (left side, z<0), one for B (z>0).
    // Each slab stack renders the TWO spinor branches as colored Gaussians.
    const N_SLABS = 36;
    const SLAB_H  = 2.8;
    const K_WAVE  = 3.2;

    const slabVert = `
      varying vec2 vUv; varying vec3 vPos;
      void main(){ vUv=uv; vPos=position; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }
    `;
    // Fragment: renders two Gaussian branches (up=green, down=orange) + blend in superposition zone
    const slabFrag = `
      uniform float uSigXY, uSigZ;
      uniform float uCUpX, uCUpY, uCDnX, uCDnY;  // branch centres
      uniform float uWz, uSlabZ, uPhase;
      uniform float uIsPost;   // 0=superposition, 1=separated, 2=collapsed(+), 3=collapsed(-)
      uniform float uBright, uAlphaMax;
      varying vec2 vUv; varying vec3 vPos;

      float g2(float x,float y,float cx,float cy,float s){
        float dx=(x-cx)/s, dy=(y-cy)/s;
        return exp(-0.5*(dx*dx+dy*dy));
      }
      float gz(float z,float cz,float sz){ float d=(z-cz)/sz; return exp(-0.5*d*d); }

      void main(){
        float x=vPos.x, y=vPos.y, z=uSlabZ;
        float gzV=gz(z,uWz,uSigZ);
        float phase=cos(${K_WAVE.toFixed(1)}*(z-uWz));
        float cp=phase*0.5+0.5;

        vec3 col; float dens;
        vec2 uvC=vUv-0.5;
        float vig=1.0-smoothstep(0.38,0.50,length(uvC));

        if(uIsPost < 0.5){
          // superposition — single blue-tinted merged packet
          float g=g2(x,y,0.0,0.0,uSigXY);
          dens=g*gzV;
          col=vec3(mix(0.05,0.20,cp), mix(0.30,0.65,cp), mix(0.55,1.00,cp));
        } else if(uIsPost < 1.5){
          // separated: two colored branches
          float gUp=g2(x,y,uCUpX,uCUpY,uSigXY);
          float gDn=g2(x,y,uCDnX,uCDnY,uSigXY);
          dens=max(gUp,gDn)*gzV;
          float tp=gUp/(gUp+gDn+1e-6);
          vec3 cUp=vec3(mix(0.02,0.13,cp), mix(0.55,1.00,cp), mix(0.10,0.40,cp)); // green
          vec3 cDn=vec3(mix(0.50,1.00,cp), mix(0.10,0.35,cp), mix(0.02,0.08,cp)); // orange
          col=mix(cDn,cUp,tp);
        } else if(uIsPost < 2.5){
          // collapsed to spin-up branch
          float gUp=g2(x,y,uCUpX,uCUpY,uSigXY);
          dens=gUp*gzV;
          col=vec3(mix(0.02,0.13,cp), mix(0.55,1.00,cp), mix(0.10,0.40,cp));
        } else {
          // collapsed to spin-down branch
          float gDn=g2(x,y,uCDnX,uCDnY,uSigXY);
          dens=gDn*gzV;
          col=vec3(mix(0.50,1.00,cp), mix(0.10,0.35,cp), mix(0.02,0.08,cp));
        }

        if(dens<0.08) discard;
        float d2=dens*dens;
        float alpha=d2*vig*uBright*7.0;
        alpha=clamp(alpha,0.0,uAlphaMax);
        if(alpha<0.005) discard;
        gl_FragColor=vec4(col*(0.5+0.5*dens)*uBright, alpha);
      }
    `;

    const baseUniforms = {
      uSigXY:   { value: SIG * 1.1 },
      uSigZ:    { value: SIG * 1.5 },
      uCUpX:    { value: 0 }, uCUpY: { value: 0 },
      uCDnX:    { value: 0 }, uCDnY: { value: 0 },
      uWz:      { value: 0 },
      uSlabZ:   { value: 0 },
      uPhase:   { value: 0 },
      uIsPost:  { value: 0 },
      uBright:  { value: 0.55 },
      uAlphaMax:{ value: 0.32 },
    };
    const slabGeo = new THREE.PlaneGeometry(SLAB_H * 2, SLAB_H * 2, 1, 1);

    function makeSlabStack(zFrom, zTo) {
      return Array.from({ length: N_SLABS }, (_, i) => {
        const mat = new THREE.ShaderMaterial({
          vertexShader: slabVert,
          fragmentShader: slabFrag,
          uniforms: THREE.UniformsUtils.clone(baseUniforms),
          transparent: true, depthWrite: false,
          blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(slabGeo, mat);
        const fixedZ = zFrom + (i + 0.5) / N_SLABS * (zTo - zFrom);
        mesh.position.z = fixedZ;
        mat.uniforms.uSlabZ.value = fixedZ;
        scene.add(mesh);
        return mesh;
      });
    }

    const slabsA = makeSlabStack(-Z_DET, Z_SRC - 0.3); // left particle
    const slabsB = makeSlabStack( Z_SRC + 0.3, Z_DET);  // right particle

    // ── Bohmian particles (dots + trajectory lines) ───────────────────────────
    const N_BOHM = 6;
    const TRAJ_COLOR_UP   = [0x22 / 255, 0xee / 255, 0x66 / 255];
    const TRAJ_COLOR_DOWN = [1.0, 0x44 / 255, 0x22 / 255];

    function makeBohmParticles(n) {
      const dots  = Array.from({ length: n }, () => {
        const m = new THREE.Mesh(new THREE.SphereGeometry(0.10, 10, 10),
          new THREE.MeshBasicMaterial({ color: 0xaaddff, transparent: true, opacity: 0 }));
        scene.add(m); return m;
      });
      const glows = Array.from({ length: n }, () => {
        const m = new THREE.Mesh(new THREE.SphereGeometry(0.22, 10, 10),
          new THREE.MeshBasicMaterial({ color: 0xaaddff, transparent: true, opacity: 0, depthWrite: false }));
        scene.add(m); return m;
      });
      const lines = Array.from({ length: n }, () => {
        const pos = new Float32Array((STEPS + 1) * 3);
        const col = new Float32Array((STEPS + 1) * 3);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('color',    new THREE.BufferAttribute(col, 3));
        const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5 }));
        scene.add(line); return { geo, pos, col, line };
      });
      return { dots, glows, lines };
    }
    const bohmA = makeBohmParticles(N_BOHM);
    const bohmB = makeBohmParticles(N_BOHM);

    // Pre-compute Bohmian trajectories for a given set of params
    let trajsA = [], trajsB = [];
    let hitRegistered = new Array(N_BOHM).fill(false);

    function buildTrajectories() {
      const s = S.current;
      const nnA = nHat(s.phiA);
      const nnB = nHat(s.phiB);

      // N_BOHM particles: alternate up/down outcomes (Born-rule split ≈ 50/50 for singlet)
      trajsA = []; trajsB = [];
      for (let i = 0; i < N_BOHM; i++) {
        const isUpA = i < N_BOHM / 2; // first half → A measures up
        const isUpB = !isUpA;           // singlet: anti-correlated on same axis

        // Initial transverse offset along n̂: place inside Gaussian
        const offset = (isUpA ? 1 : -1) * SIG * 0.55;
        trajsA.push({ pts: integrateBohmian(offset, isUpA, s.phiA, -1), isUp: isUpA });
        trajsB.push({ pts: integrateBohmian(-offset, isUpB, s.phiB,  1), isUp: isUpB });
      }

      // Upload trajectory geometry
      [{ trajs: trajsA, bohm: bohmA, nn: nnA }, { trajs: trajsB, bohm: bohmB, nn: nnB }]
        .forEach(({ trajs, bohm }) => {
          trajs.forEach(({ pts, isUp }, i) => {
            const fl = bohm.lines[i];
            const [cr, cg, cb] = isUp ? TRAJ_COLOR_UP : TRAJ_COLOR_DOWN;
            pts.forEach((p, j) => {
              fl.pos[j * 3] = p.x; fl.pos[j * 3 + 1] = p.y; fl.pos[j * 3 + 2] = p.z;
              fl.col[j * 3] = cr; fl.col[j * 3 + 1] = cg; fl.col[j * 3 + 2] = cb;
            });
            fl.geo.attributes.position.needsUpdate = true;
            fl.geo.attributes.color.needsUpdate = true;
            fl.geo.setDrawRange(0, pts.length);
            fl.line.visible = false;
          });
        });

      hitRegistered.fill(false);
    }
    buildTrajectories();

    // ── Input ────────────────────────────────────────────────────────────────
    function onDown(e) { S.current.drag = { btn: e.button ?? 0, x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); }
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
        const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0)).normalize();
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
      S.current.camR = clamp(S.current.camR * (e.deltaY > 0 ? 1.10 : 0.91), 3, 50);
      updateCam();
    }
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup',   onUp);
    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('contextmenu', e => e.preventDefault());

    T.current = {
      scene, camera, renderer,
      detA, detB, arrA, arrB,
      slabsA, slabsB,
      bohmA, bohmB,
      hitsA, hitsB,
      buildTrajectories,
      updateCam,
      setCountsUI: c => setCountsUI({ ...c }),
    };

    // ── Render loop ──────────────────────────────────────────────────────────
    let raf;
    function animate() {
      raf = requestAnimationFrame(animate);
      const s = S.current;
      const Tr = T.current;
      if (!Tr) return;

      if (s.dirty) { Tr.buildTrajectories(); s.dirty = false; }
      if (s.running) s.tick += s.speed;

      updateCam();

      const frac = (s.tick % PERIOD) / PERIOD; // 0→1 each cycle
      const tIdx = clamp(Math.round(frac * STEPS), 0, STEPS);

      // Reset collapse state at start of new cycle
      if (tIdx <= 1 && s.collapsed) {
        s.collapsed = false;
        s.outcomeA = null;
        s.outcomeB = null;
        hitRegistered.fill(false);
        s.dirty = true;
      }

      // ── Determine collapse/outcome ────────────────────────────────────────
      const collFrac = s.collapseAtFrac;
      const hasFired = frac >= collFrac;

      if (hasFired && !s.collapsed) {
        s.collapsed = true;
        // Single outcome drawn from Born rule using singlet correlations
        const { pp } = singletProbs(s.phiA, s.phiB);
        const rand = Math.random();
        // outcomes: pp=P(+,+), pm=P(+,-), mp=P(-,+), mm=P(-,-)
        const { pm } = singletProbs(s.phiA, s.phiB);
        if (rand < pp) {
          s.outcomeA = 1; s.outcomeB = 1;
          s.counts.pp++;
        } else if (rand < pp + pm) {
          s.outcomeA = 1; s.outcomeB = -1;
          s.counts.pm++;
        } else if (rand < pp + pm + pm) {
          s.outcomeA = -1; s.outcomeB = 1;
          s.counts.mp++;
        } else {
          s.outcomeA = -1; s.outcomeB = -1;
          s.counts.mm++;
        }
        Tr.setCountsUI(s.counts);
      }

      const collapsed   = s.collapsed;
      const outcomeA    = s.outcomeA; // +1 or -1
      const outcomeB    = s.outcomeB;

      // ── Detector axis arrows ─────────────────────────────────────────────
      const nnA = nHat(s.phiA);
      const nnB = nHat(s.phiB);
      const safeDir = v => { if (Math.abs(v.x) < 0.001 && Math.abs(v.z) < 0.001) v.x = 0.001; return v.normalize(); };
      Tr.arrA.setDirection(safeDir(nnA.clone()));
      Tr.arrB.setDirection(safeDir(nnB.clone()));

      // ── Wave-packet slab update ───────────────────────────────────────────
      const sw = s.showWave;
      // Compute transverse branch separation at this frac
      const postFrac = frac > 0.5 ? (frac - 0.5) / 0.5 : 0;
      const sep = postFrac * KICK;
      const sigXY = SIG * (1 + postFrac * 0.4);
      const sigZ  = SIG * 1.5;

      // For each side (A=particle moving to -Z, B=moving to +Z)
      // wZ = current packet centre Z position
      const wZa = lerp(-Z_DET, 0, frac > 1 ? 1 : frac * 2); // A starts at 0, moves to -Z_DET
      const wZb = lerp( 0, Z_DET, frac > 1 ? 1 : frac * 2); // B starts at 0, moves to +Z_DET
      // More natural: packets fly apart continuously
      const wZA = lerp(0, -Z_DET, frac);
      const wZB = lerp(0,  Z_DET, frac);

      const isPost = frac >= 0.5;

      function updateSlabs(slabs, wZ, phiVal, outcome, isParticleA) {
        const nn = nHat(phiVal);
        const upCx =  nn.x * sep, upCy =  nn.y * sep;
        const dnCx = -nn.x * sep, dnCy = -nn.y * sep;

        // isPost value for shader:
        // 0 = superposition (pre-split)
        // 1 = separated (both branches visible)
        // 2 = collapsed to up
        // 3 = collapsed to down
        let isPostVal = 0;
        if (isPost) {
          if (s.interp === 'collapse' && collapsed && outcome !== null) {
            isPostVal = outcome === 1 ? 2 : 3;
          } else {
            isPostVal = 1;
          }
        }

        slabs.forEach(mesh => {
          if (!sw) { mesh.visible = false; return; }
          const u = mesh.material.uniforms;
          u.uSigXY.value   = sigXY;
          u.uSigZ.value    = sigZ;
          u.uCUpX.value    = upCx; u.uCUpY.value = upCy;
          u.uCDnX.value    = dnCx; u.uCDnY.value = dnCy;
          u.uWz.value      = wZ;
          u.uIsPost.value  = isPostVal;
          u.uBright.value  = 0.55;
          u.uAlphaMax.value= 0.32;
          mesh.visible     = true;
        });
      }

      updateSlabs(slabsA, wZA, s.phiA, outcomeA, true);
      updateSlabs(slabsB, wZB, s.phiB, outcomeB, false);

      // ── Bohmian particles ─────────────────────────────────────────────────
      const showP = s.showParticles && s.interp === 'bohmian';

      [
        { trajs: trajsA, bohm: bohmA, hitPool: hitsA, outcome: outcomeA },
        { trajs: trajsB, bohm: bohmB, hitPool: hitsB, outcome: outcomeB },
      ].forEach(({ trajs, bohm, hitPool, outcome }, side) => {
        trajs.forEach(({ pts, isUp }, i) => {
          const col = isUp ? 0x22ee66 : 0xff5522;
          if (!showP) {
            bohm.dots[i].visible  = false;
            bohm.glows[i].visible = false;
            bohm.lines[i].line.visible = false;
            return;
          }
          const pt = pts[tIdx];
          bohm.dots[i].position.copy(pt);
          bohm.glows[i].position.copy(pt);
          bohm.dots[i].visible  = true;
          bohm.glows[i].visible = true;
          bohm.dots[i].material.color.set(col);
          bohm.glows[i].material.color.set(col);
          bohm.dots[i].material.opacity  = 0.95;
          bohm.glows[i].material.opacity = 0.15;
          bohm.lines[i].line.visible = true;
          bohm.lines[i].geo.setDrawRange(0, tIdx + 1);

          // Hit registration
          if (tIdx >= STEPS - 2 && !hitRegistered[i + side * N_BOHM]) {
            hitRegistered[i + side * N_BOHM] = true;
            const endPt = pts[STEPS];
            hitPool.add(endPt.x, endPt.y, col);
          }
        });
      });

      // In collapse mode, add hits when collapse fires
      if (s.interp === 'collapse' && collapsed && !hitRegistered[N_BOHM * 2]) {
        hitRegistered[N_BOHM * 2] = true;
        const colA = outcomeA === 1 ? 0x22ee66 : 0xff5522;
        const colB = outcomeB === 1 ? 0x22ee66 : 0xff5522;
        // Random position on detector (Gaussian, within branch)
        const nnAv = nHat(s.phiA), nnBv = nHat(s.phiB);
        const r = () => (Math.random() - 0.5) * SIG * 1.0;
        const signA = outcomeA === 1 ? 1 : -1;
        const signB = outcomeB === 1 ? 1 : -1;
        hitsA.add(nnAv.x * KICK * signA + r(), nnAv.y * KICK * signA + r(), colA);
        hitsB.add(nnBv.x * KICK * signB + r(), nnBv.y * KICK * signB + r(), colB);
      }

      Tr.renderer.render(Tr.scene, Tr.camera);
    }
    animate();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerup', onUp);
      el.removeEventListener('wheel', onWheel);
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Layout ────────────────────────────────────────────────────────────────
  return (
    <div style={{ width: '100%', height: '100%', overflow: 'hidden', background: '#07101e', display: 'flex', flexDirection: 'column' }}>
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
      <div style={{
        display: 'flex', alignItems: 'center', height: 38, flexShrink: 0,
        background: 'rgba(4,10,30,0.98)', borderBottom: '1px solid rgba(40,80,180,0.3)',
        paddingLeft: 12, gap: 4,
      }}>
        <span style={{ fontSize: 11, color: '#4060a0', fontFamily: 'monospace', letterSpacing: '0.08em', marginRight: 12 }}>
          BELL EXPERIMENT
        </span>
        <button className={'tbb' + (activeTab === 'sim' ? ' tba' : '')} onClick={() => setActiveTab('sim')}>Simulation</button>
        <button className={'tbb' + (activeTab === 'theory' ? ' tba' : '')} onClick={() => setActiveTab('theory')}>Theory</button>
      </div>

      {/* Main area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'row', overflow: 'hidden', minHeight: 0 }}>
        {/* Left panel */}
        <div style={{ width: panelW, minWidth: 200, flexShrink: 0, background: 'rgba(8,18,45,0.98)', overflowY: 'auto', height: '100%' }}>
          <ControlPanel
            interp={interp} setInterp={setInterp}
            phiA={phiA} setPhiA={setPhiA} phiARef={phiARef}
            phiB={phiB} setPhiB={setPhiB} phiBRef={phiBRef}
            speed={speed} setSpeed={setSpeed} speedRef={speedRef}
            running={running} setRunning={setRunning}
            showWave={showWave} setShowWave={setShowWave}
            showParticles={showParticles} setShowParticles={setShowParticles}
            resetCounts={resetCounts}
            measuredSide={measuredSide} setMeasuredSide={setMeasuredSide}
            counts={counts}
            phiAVal={phiA} phiBVal={phiB}
          />
        </div>

        {/* Resize handle */}
        <div className="rh" ref={resizeHandleRef} />

        {/* Right: canvas or theory */}
        <div style={{ flex: 1, minWidth: 0, position: 'relative', height: '100%' }}>
          <div ref={mountRef} style={{
            width: '100%', height: '100%', cursor: 'grab',
            display: activeTab === 'sim' ? 'block' : 'none',
          }} />

          {/* Overlay labels */}
          {activeTab === 'sim' && (
            <>
              <div style={{ position: 'absolute', top: 10, left: 12, zIndex: 10,
                fontFamily: 'monospace', fontSize: 12, color: '#44ddff',
                background: 'rgba(4,10,30,0.75)', borderRadius: 5, padding: '4px 8px',
                border: '1px solid rgba(68,221,255,0.3)' }}>
                ← Particle A  φ_A={phiA}°
              </div>
              <div style={{ position: 'absolute', top: 10, right: 12, zIndex: 10,
                fontFamily: 'monospace', fontSize: 12, color: '#ff9966',
                background: 'rgba(4,10,30,0.75)', borderRadius: 5, padding: '4px 8px',
                border: '1px solid rgba(255,153,102,0.3)' }}>
                Particle B  φ_B={phiB}°  →
              </div>
              {/* Legend */}
              <div style={{ position: 'absolute', bottom: 12, left: 12, zIndex: 10,
                fontFamily: 'monospace', fontSize: 11, color: '#b8d4ff',
                background: 'rgba(4,10,30,0.80)', borderRadius: 6, padding: '6px 10px',
                border: '1px solid rgba(80,140,255,0.25)' }}>
                <div><span style={{ color: '#55aaff' }}>■</span> superposition (entangled)</div>
                <div><span style={{ color: '#22ee66' }}>■</span> spin ↑ branch</div>
                <div><span style={{ color: '#ff5522' }}>■</span> spin ↓ branch</div>
              </div>
            </>
          )}

          {activeTab === 'theory' && (
            <div style={{ position: 'absolute', inset: 0, overflowY: 'auto' }}>
              <iframe srcDoc={theoryHtml} style={{ width: '100%', height: '100%', border: 'none' }} title="Theory" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
