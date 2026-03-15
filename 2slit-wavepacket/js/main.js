import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GPUSim } from './gpu-sim.js?v=5';

// ─────────────────────────────────────────────────────────────
//  Wave colormaps  (dynamic LUT, 512 entries)
// ─────────────────────────────────────────────────────────────
const WAVE_PALETTES = {
  plasma: [
    [0.15, 0.05, 0.60], [0.28, 0.02, 0.70], [0.47, 0.01, 0.72],
    [0.63, 0.10, 0.65], [0.78, 0.16, 0.54], [0.90, 0.24, 0.42],
    [0.97, 0.36, 0.30], [0.99, 0.50, 0.17], [0.99, 0.65, 0.05],
    [0.95, 0.81, 0.04], [0.95, 0.99, 0.15],
  ],
  viridis: [
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.253, 0.265, 0.530],
    [0.163, 0.371, 0.558], [0.128, 0.467, 0.558], [0.134, 0.559, 0.555],
    [0.153, 0.651, 0.510], [0.302, 0.745, 0.416], [0.525, 0.833, 0.288],
    [0.762, 0.876, 0.137], [0.993, 0.906, 0.144],
  ],
  inferno: [
    [0.10, 0.02, 0.18], [0.180, 0.040, 0.320], [0.298, 0.024, 0.373],
    [0.460, 0.066, 0.310], [0.621, 0.120, 0.200], [0.768, 0.224, 0.089],
    [0.886, 0.368, 0.027], [0.968, 0.532, 0.020], [0.993, 0.715, 0.139],
    [0.987, 0.891, 0.429], [0.988, 1.000, 0.643],
  ],
  magma: [
    [0.09, 0.03, 0.22], [0.130, 0.020, 0.280], [0.246, 0.022, 0.349],
    [0.417, 0.046, 0.377], [0.576, 0.125, 0.338], [0.718, 0.215, 0.321],
    [0.844, 0.336, 0.329], [0.942, 0.484, 0.381], [0.980, 0.643, 0.497],
    [0.994, 0.804, 0.652], [0.998, 0.965, 0.845],
  ],
  ocean: [
    [0.005, 0.005, 0.170], [0.018, 0.038, 0.420], [0.042, 0.108, 0.600],
    [0.055, 0.252, 0.710], [0.068, 0.400, 0.748], [0.072, 0.556, 0.763],
    [0.095, 0.711, 0.790], [0.210, 0.848, 0.868], [0.510, 0.920, 0.940],
    [0.800, 0.962, 0.982], [1.000, 1.000, 1.000],
  ],
  fire: [
    [0.14, 0.02, 0.04], [0.320, 0.008, 0.008], [0.570, 0.010, 0.000],
    [0.820, 0.080, 0.000], [0.970, 0.220, 0.000], [1.000, 0.400, 0.000],
    [1.000, 0.590, 0.000], [1.000, 0.770, 0.050], [1.000, 0.910, 0.300],
    [1.000, 0.970, 0.650], [1.000, 1.000, 1.000],
  ],
  neon: [
    [0.02, 0.08, 0.10], [0.010, 0.140, 0.140], [0.010, 0.300, 0.200],
    [0.020, 0.550, 0.180], [0.060, 0.750, 0.150], [0.200, 0.900, 0.300],
    [0.400, 0.980, 0.500], [0.400, 1.000, 0.800], [0.600, 1.000, 0.950],
    [0.800, 1.000, 1.000], [1.000, 1.000, 1.000],
  ],
  coolwarm: [
    [0.085, 0.532, 0.201], [0.192, 0.629, 0.758], [0.380, 0.761, 0.969],
    [0.610, 0.869, 1.000], [0.820, 0.940, 1.000], [1.000, 1.000, 1.000],
    [1.000, 0.880, 0.800], [1.000, 0.680, 0.540], [0.940, 0.420, 0.310],
    [0.780, 0.160, 0.160], [0.600, 0.040, 0.040],
  ],
};

let currentPaletteKey = 'ocean';
const LUT_SIZE = 512;
const colorLUT = new Float32Array(LUT_SIZE * 3);

function buildLUT(stops) {
  const n = stops.length - 1;
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1);
    const s = t * n;
    const lo = Math.min(Math.floor(s), n - 1);
    const f = s - lo;
    const A = stops[lo], B = stops[lo + 1];
    colorLUT[i * 3 + 0] = A[0] + f * (B[0] - A[0]);
    colorLUT[i * 3 + 1] = A[1] + f * (B[1] - A[1]);
    colorLUT[i * 3 + 2] = A[2] + f * (B[2] - A[2]);
  }
}
function setWavePalette(key) {
  currentPaletteKey = key;
  buildLUT(WAVE_PALETTES[key]);
  // Redraw immediately even when paused
  if (lastRho) {
    if (viewMode === 'spacetime') updateSpacetime(lastRho, NX, NY, lastTime);
    else updateSurface(lastRho, NX, NY);
  }
}
buildLUT(WAVE_PALETTES.ocean);

function sampleLUT(t) {
  const idx = Math.max(0, Math.min(LUT_SIZE - 1, Math.round(t * (LUT_SIZE - 1))));
  return [colorLUT[idx * 3], colorLUT[idx * 3 + 1], colorLUT[idx * 3 + 2]];
}

// ─────────────────────────────────────────────────────────────
//  HSV → RGB  (for phase-coloured 2D view)
// ─────────────────────────────────────────────────────────────
function hsvToRgb(h, s, v) {
  h = ((h % 1) + 1) % 1;   // wrap to [0,1]
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    case 5: return [v, p, q];
  }
  return [0,0,0];
}

// ─────────────────────────────────────────────────────────────
//  Scene / renderer / camera
// ─────────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.setClearColor(0x090918);

const scene = new THREE.Scene();
// Fog is added only in 3D surface mode; starts off for default 2D view.
// scene.fog set by setCameraView()

const camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.001, 100);
camera.position.set(0, 0, 5.5);   // start top-down for 2D mode
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

// ─── Camera view helpers ─────────────────────────────────────
// "Cover" fit: plane fills the entire canvas, no black borders.
// Equivalent to CSS background-size:cover applied to the 4×2 scene plane.
function fitCamera2D() {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const aspect = W / H;
  camera.aspect = aspect;
  camera.updateProjectionMatrix();
  // Crop Y to the non-absorbing inner region so the absorber strip is
  // outside the viewport — the wave simply exits the frame cleanly.
  const absT     = (useGPU && gpuSim && gpuSim._absThick) ? gpuSim._absThick : 0.15;
  const innerHalfH = (1 - 2 * absT) * SURFACE_SCALE / 2;
  const innerHalfW = (1 - 2 * absT) * SS_X / 2;
  // "Cover" fit: ensure both X and Y fully fill the canvas.
  const halfH = Math.max(innerHalfH, innerHalfW / aspect);
  const fovRad = camera.fov * Math.PI / 180;
  const d = halfH / Math.tan(fovRad / 2);
  camera.position.set(0, 0, d);
  camera.up.set(0, 1, 0);
  camera.lookAt(0, 0, 0);
  controls.target.set(0, 0, 0);
  controls.update();
}

function setCameraView(mode) {
  if (mode === '2d') {
    controls.enableRotate = false;
    controls.enableZoom   = true;
    scene.fog = null;
    grid.visible = false;
    fitCamera2D();
    if (bohmPoints) bohmPoints.material.size = 0.036;
  } else {
    // 3D perspective – pull back to show the 4×2 plane
    controls.enableRotate = true;
    controls.enableZoom   = true;
    camera.position.set(0, -4.0, 3.0);
    camera.up.set(0, 0, 1);
    controls.target.set(0, 0, 0);
    controls.maxPolarAngle = Math.PI;
    controls.minPolarAngle = 0;
    scene.fog = new THREE.FogExp2(0x090918, 0.25);
    grid.visible = true;
    if (bohmPoints) bohmPoints.material.size = 0.024;
  }
  camera.lookAt(controls.target);
  controls.update();
}

// Dim grid helper
const grid = new THREE.GridHelper(6, 24, 0x1a2a3a, 0x0d1520);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

// Lights (needed for MeshLambertMaterial)
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(1, 1, 3);
scene.add(dirLight);

// ─────────────────────────────────────────────────────────────
//  Coordinate axes overlay
// ─────────────────────────────────────────────────────────────
const axesCanvas = document.getElementById('axes-canvas');
const axCtx = axesCanvas.getContext('2d');

function drawAxes() {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  axesCanvas.width  = W;
  axesCanvas.height = H;
  axCtx.clearRect(0, 0, W, H);
  if (viewMode !== 'density' && viewMode !== 'phase') return;

  // Fixed bottom-left inset: axes drawn in canvas-pixel space
  const PAD   = 36;   // left/bottom margin for labels
  const LEN_X = 120;  // pixel length of x-axis
  const LEN_Y = 80;   // pixel length of y-axis
  const ox = PAD;           // origin x (pixels from left)
  const oy = H - PAD;       // origin y (pixels from top)

  // Physical scale from pixel length
  const _v = new THREE.Vector3();
  function physToCanvas(x_nm, y_nm) {
    const sx = -2.0 + (x_nm / 200.0) * 4.0;
    const sy =  1.0 - (y_nm / 100.0) * 2.0;
    _v.set(sx, sy, 0).project(camera);
    return [(_v.x * 0.5 + 0.5) * W, (-_v.y * 0.5 + 0.5) * H];
  }
  // domain pixel extents (for scaling)
  const [canvX0] = physToCanvas(0,   50);
  const [canvX1] = physToCanvas(200, 50);
  const [, canvY0] = physToCanvas(100, 0);
  const [, canvY1] = physToCanvas(100, 100);
  const pxPerNmX = (canvX1 - canvX0) / 200;
  const pxPerNmY = (canvY1 - canvY0) / 100;

  // Fixed axis spans
  const nmX = 25;  // always 25 nm
  const nmY = Math.round(LEN_Y / Math.abs(pxPerNmY) / 25) * 25 || 25;
  const axLenX = nmX * pxPerNmX;
  const axLenY = nmY * Math.abs(pxPerNmY);

  axCtx.lineWidth   = 1.5;
  axCtx.strokeStyle = 'rgba(0,204,255,0.70)';
  axCtx.fillStyle   = 'rgba(0,204,255,0.90)';
  axCtx.font        = '10px Inter, system-ui, sans-serif';

  // X axis
  axCtx.beginPath(); axCtx.moveTo(ox, oy); axCtx.lineTo(ox + axLenX, oy); axCtx.stroke();
  // ticks at 0 and nmX
  [0, nmX].forEach(v => {
    const px = ox + v * pxPerNmX;
    axCtx.beginPath(); axCtx.moveTo(px, oy - 4); axCtx.lineTo(px, oy + 4); axCtx.stroke();
    axCtx.textAlign = 'center'; axCtx.textBaseline = 'top';
    axCtx.fillText(`${v}`, px, oy + 6);
  });
  // x axis label at tip
  axCtx.textAlign = 'left'; axCtx.textBaseline = 'middle';
  axCtx.fillStyle = 'rgba(0,204,255,0.95)';
  axCtx.fillText('x (nm)', ox + axLenX + 5, oy);

  // Y axis (upward)
  axCtx.beginPath(); axCtx.moveTo(ox, oy); axCtx.lineTo(ox, oy - axLenY); axCtx.stroke();
  [0, nmY].forEach(v => {
    const py = oy - v * Math.abs(pxPerNmY);
    axCtx.beginPath(); axCtx.moveTo(ox - 4, py); axCtx.lineTo(ox + 4, py); axCtx.stroke();
    axCtx.textAlign = 'right'; axCtx.textBaseline = 'middle';
    axCtx.fillText(`${v}`, ox - 7, py);
  });
  // y axis label (rotated) at mid-axis, to the left
  axCtx.save();
  axCtx.translate(ox - 24, oy - axLenY * 0.5);
  axCtx.rotate(-Math.PI / 2);
  axCtx.textAlign = 'center'; axCtx.textBaseline = 'middle';
  axCtx.fillStyle = 'rgba(0,204,255,0.95)';
  axCtx.fillText('y (nm)', 0, 0);
  axCtx.restore();

  // Simulation time — top-left corner of canvas
  if (typeof gpuSim !== 'undefined' && gpuSim && gpuSim.simTime != null) {
    const t_fs = (gpuSim.simTime * 1e15).toFixed(1);
    axCtx.font = '11px Inter, system-ui, sans-serif';
    axCtx.textAlign = 'left'; axCtx.textBaseline = 'top';
    axCtx.fillStyle = 'rgba(0,204,255,0.85)';
    axCtx.fillText(`t = ${t_fs} fs`, 10, 10);
  }
}

// ─────────────────────────────────────────────────────────────
//  Shared simulation state
// ─────────────────────────────────────────────────────────────
let NX = 256, NY = 128;
const SURFACE_SCALE = 2.0;   // Y scene extent  (maps to Ly = 100 nm)
const SS_X          = 4.0;   // X scene extent  (maps to Lx = 200 nm)
const HEIGHT_PEAK   = 0.9;

let viewMode    = 'density';  // default: flat 2D density (plasma)
let showParticles = true;
let showTrails    = true;
let trailDisplayLen = 20;    // max visible trail length in nm
let autoStop = true;         // auto-pause when wave exits domain
let solverMode = 'rk4';  // 'fft' | 'fd' | 'rk4'

// Performance tracking
let _perfFrameCount = 0;
let _perfLastTime   = 0;
let _perfLastSimT   = 0;
let _perfFPS        = 0;
let _perfSimSpeed   = 0;  // simulated fs per real second
let _perfStepsPerS  = 0;
let paused      = false;
let stepsPerFrame = 16;
let lastRho  = null;   // most recent rho snapshot for palette-change redraws
let lastTime = 0;

// Physics backend
let gpuSim    = null;   // GPUSim instance
let worker    = null;   // fallback web worker
let useGPU    = false;
let frameReady = false;

// Bohmian state (CPU, fed by GPU psi readback)
const HB = 1.054571817e-34;
const ME = 9.10938356e-31;
let bohmPosX = null, bohmPosY = null;
let bohmNp   = 300;
let bohmPassedSlit    = null;  // Uint8Array: 1 once particle has crossed the barrier
let bohmAnyPassedSlit = false; // true once any particle has crossed — stops respawning

// Trajectory trail ring-buffer
const TRAIL_LEN = 2000;            // steps stored per particle (covers full simulation run)
let trajBufX  = null;             // Float32Array[MAX_PARTICLES * TRAIL_LEN]
let trajBufY  = null;
let trajHead  = null;             // Int32Array[MAX_PARTICLES]
let trajFilled = null;            // Int32Array[MAX_PARTICLES]

// Particle / trail colour presets  { label, rgb:[R,G,B] }
const PARTICLE_PRESETS = [
  { label: '🟡 Yellow',   rgb: [1.00, 0.85, 0.15] },
  { label: '🩵 Cyan',     rgb: [0.00, 0.95, 1.00] },
  { label: '⚪ White',    rgb: [1.00, 1.00, 1.00] },
  { label: '🟢 Green',    rgb: [0.20, 1.00, 0.40] },
  { label: '🟣 Magenta',  rgb: [1.00, 0.30, 1.00] },
  { label: '🟠 Orange',   rgb: [1.00, 0.52, 0.05] },
  { label: '🔴 Red',      rgb: [1.00, 0.18, 0.08] },
  { label: '🔵 Blue',     rgb: [0.25, 0.55, 1.00] },
  { label: '🌸 Pink',     rgb: [1.00, 0.55, 0.82] },
  { label: '💜 Lavender', rgb: [0.72, 0.55, 1.00] },
  { label: '✨ Gold',     rgb: [1.00, 0.84, 0.00] },
  { label: '🍀 Lime',     rgb: [0.55, 1.00, 0.10] },
];
let trailColorIdx = 6;  // default: Red

// Wall / barrier colour presets
const WALL_PRESETS = [
  { label: 'Brick',      hex: 0x8B3A2B },
  { label: 'Terracotta', hex: 0xBF5A33 },
  { label: 'Brown',      hex: 0x5E3A1A },
  { label: 'Slate',      hex: 0x5A5060 },
  { label: 'Steel',      hex: 0x4466AA },
  { label: 'Concrete',   hex: 0x7A7570 },
];
let wallColorIdx = 0;  // default: Brick
function wallHex() { return WALL_PRESETS[wallColorIdx].hex; }

let trailGeo, trailMesh;

// ─────────────────────────────────────────────────────────────
//  Height-surface mesh  (view = 'surface')
// ─────────────────────────────────────────────────────────────
let surfaceGeo, surfaceMesh;

function buildSurface() {
  if (surfaceMesh) { scene.remove(surfaceMesh); surfaceGeo.dispose(); }
  surfaceGeo = new THREE.PlaneGeometry(SS_X, SURFACE_SCALE, NX - 1, NY - 1);
  // The geometry positions are in XY plane, Z is height.
  // Pre-fill color buffer (THREE.PlaneGeometry has no color attribute by default)
  const N = NX * NY;
  const colArr = new Float32Array(N * 3);
  surfaceGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
  const mat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    wireframe: false,
  });
  surfaceMesh = new THREE.Mesh(surfaceGeo, mat);
  scene.add(surfaceMesh);
}

function updateSurface(rho, Nx, Ny) {
  const posArr = surfaceGeo.attributes.position.array;
  const colArr = surfaceGeo.attributes.color.array;
  const N = Nx * Ny;
  const flat = (viewMode === 'density' || viewMode === 'phase');
  const absT = (useGPU && gpuSim && gpuSim._absThick) ? gpuSim._absThick : 0.15;

  let rhoMax = 0;
  for (let i = 0; i < N; i++) if (rho[i] > rhoMax) rhoMax = rho[i];
  const heightScale = (!flat && rhoMax > 0) ? HEIGHT_PEAK / rhoMax : 0;

  // For phase coloring we need the complex psi from GPU readback
  const psi = (viewMode === 'phase' && useGPU && gpuSim) ? gpuSim.psi : null;

  for (let ix = 0; ix < Nx; ix++) {
    for (let iy = 0; iy < Ny; iy++) {
      const vIdx = iy * Nx + ix;   // Three.js vertex index (row-major)
      const rIdx = ix * Ny + iy;   // rho column-major index

      // Mask absorber zone: zero color so the wave visually stops at the boundary
      const fx = ix / (Nx - 1);
      const fy = iy / (Ny - 1);
      if (fx < absT || fx > 1 - absT || fy < absT || fy > 1 - absT) {
        posArr[vIdx * 3 + 2] = 0;
        colArr[vIdx * 3 + 0] = 0;
        colArr[vIdx * 3 + 1] = 0;
        colArr[vIdx * 3 + 2] = 0;
        continue;
      }

      const t = rhoMax > 0 ? Math.min(1, rho[rIdx] / rhoMax) : 0;
      posArr[vIdx * 3 + 2] = flat ? 0 : rho[rIdx] * heightScale;

      let r, g, b;
      if (psi) {
        const re     = psi[vIdx * 2];
        const ampMax = Math.sqrt(rhoMax) || 1;
        const reNorm = 0.5 + 0.5 * (re / ampMax);
        [r, g, b] = sampleLUT(reNorm);
      } else if (viewMode === 'density') {
        [r, g, b] = sampleLUT(Math.pow(t, 0.45));
      } else {
        [r, g, b] = sampleLUT(Math.sqrt(t));
      }
      colArr[vIdx * 3 + 0] = r;
      colArr[vIdx * 3 + 1] = g;
      colArr[vIdx * 3 + 2] = b;
    }
  }
  surfaceGeo.attributes.position.needsUpdate = true;
  surfaceGeo.attributes.color.needsUpdate = true;
}

// ─────────────────────────────────────────────────────────────
//  Spacetime waterfall  (view = 'spacetime')
//  Displays the marginal density ρ_y(y,t) = ∫ |ψ|² dx
//  as stacked curves displaced along the time axis (3D-Z)
// ─────────────────────────────────────────────────────────────
const MAX_SLICES = 80;
const sliceHistory = [];             // [{rhoY: Float32Array(Ny), time}]
let spacetimeGroup;

function buildSpacetime() {
  if (spacetimeGroup) scene.remove(spacetimeGroup);
  spacetimeGroup = new THREE.Group();
  scene.add(spacetimeGroup);
}

function updateSpacetime(rho, Nx, Ny, time) {
  // Compute marginal density over x
  const rhoY = new Float32Array(Ny);
  for (let ix = 0; ix < Nx; ix++)
    for (let iy = 0; iy < Ny; iy++)
      rhoY[iy] += rho[ix * Ny + iy];

  sliceHistory.push({ rhoY, time });
  if (sliceHistory.length > MAX_SLICES) sliceHistory.shift();

  // Rebuild all line geometry (cheap — MAX_SLICES × Ny = 80 × 256 = 20k pts)
  while (spacetimeGroup.children.length) {
    const c = spacetimeGroup.children[0];
    c.geometry.dispose();
    spacetimeGroup.remove(c);
  }

  let rhoMax = 0;
  for (const s of sliceHistory) for (let iy = 0; iy < Ny; iy++) if (s.rhoY[iy] > rhoMax) rhoMax = s.rhoY[iy];

  const nSlices = sliceHistory.length;
  const zRange = 2.0;
  const heightAmp = 0.8;

  sliceHistory.forEach((slice, si) => {
    const pts = new Float32Array(Ny * 3);
    const cols = new Float32Array(Ny * 3);
    const z = -1.0 + (si / Math.max(1, nSlices - 1)) * zRange;
    const age = si / Math.max(1, nSlices - 1);
    for (let iy = 0; iy < Ny; iy++) {
      const y = -1.0 + (iy / (Ny - 1)) * 2.0;
      const h = rhoMax > 0 ? (slice.rhoY[iy] / rhoMax) * heightAmp : 0;
      pts[iy * 3 + 0] = y;     // x3d ← sim y
      pts[iy * 3 + 1] = h;     // y3d ← height
      pts[iy * 3 + 2] = z;     // z3d ← time axis
      const [r, g, b] = sampleLUT(age * 0.5 + (rhoMax > 0 ? slice.rhoY[iy] / rhoMax : 0) * 0.5);
      cols[iy * 3 + 0] = r;
      cols[iy * 3 + 1] = g;
      cols[iy * 3 + 2] = b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(cols, 3));
    const mat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 1 });
    spacetimeGroup.add(new THREE.Line(geo, mat));
  });
}

// ─────────────────────────────────────────────────────────────
//  Barrier mesh
// ─────────────────────────────────────────────────────────────
let barrierMesh;

function buildBarrier(slitX, slitCenterY1, slitCenterY2, slitHalfWidth) {
  if (barrierMesh) {
    scene.remove(barrierMesh);
    barrierMesh.traverse(c => { if (c.geometry) c.geometry.dispose(); });
  }

  // SS_X=4.0 maps to Lx=200nm, so scene-units per nm = 4/200 = 0.02; 6px = 4.7nm ≈ 0.06 scene-units
  const wallThick = 0.06;
  const slit1Lo = slitCenterY1 - slitHalfWidth;
  const slit1Hi = slitCenterY1 + slitHalfWidth;
  const slit2Lo = slitCenterY2 - slitHalfWidth;
  const slit2Hi = slitCenterY2 + slitHalfWidth;

  const bGroup = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color: wallHex(), transparent: true, opacity: 0.80 });

  // Clamp wall extent to the visible (non-absorber) band in Y.
  const absT   = (useGPU && gpuSim && gpuSim._absThick) ? gpuSim._absThick : 0.15;
  const yMin   = absT;
  const yMax   = 1.0 - absT;

  // Wall segments in fractional coords: [lo, hi], clamped to visible band
  const rawSegs = [
    [yMin,    slit1Lo],
    [slit1Hi, slit2Lo],
    [slit2Hi, yMax],
  ];
  const segments = rawSegs.map(([lo, hi]) => [Math.max(lo, yMin), Math.min(hi, yMax)])
                          .filter(([lo, hi]) => hi > lo + 1e-6);

  // Fractional f → scene coordinates.
  // X: sim x → sceneX = -SS_X/2 + f * SS_X
  // Y: sim y → sceneY = +1 - f * SURFACE_SCALE
  const toSceneX = f => -SS_X / 2 + f * SS_X;
  const toSceneY = f =>  1.0 - f * SURFACE_SCALE;

  segments.forEach(([lo, hi]) => {
    if (hi <= lo + 1e-6) return;
    const h    = (hi - lo) * SURFACE_SCALE;
    const yCen = toSceneY((lo + hi) / 2);
    const xPos = toSceneX(slitX);
    const geo  = new THREE.BoxGeometry(wallThick, h, 0.05);
    const mesh = new THREE.Mesh(geo, mat.clone());
    mesh.position.set(xPos, yCen, 0.025);
    bGroup.add(mesh);
  });

  scene.add(bGroup);
  barrierMesh = bGroup;
}

// ─────────────────────────────────────────────────────────────
//  Bohmian particle dots + trajectory trails
// ─────────────────────────────────────────────────────────────
let bohmPoints, bohmGeo;
const MAX_PARTICLES = 500;

function buildBohmianMesh() {
  if (bohmPoints) { scene.remove(bohmPoints); bohmGeo.dispose(); }
  bohmGeo = new THREE.BufferGeometry();
  bohmGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3));
  bohmGeo.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3));
  const ptMat = new THREE.PointsMaterial({
    vertexColors: true, size: 0.028, sizeAttenuation: true, depthTest: false,
  });
  bohmPoints = new THREE.Points(bohmGeo, ptMat);
  bohmPoints.renderOrder = 11;
  bohmPoints.visible = showParticles;
  scene.add(bohmPoints);

  // Trail mesh — LineSegments so the trail is always a solid line
  // regardless of steps/frame. Each consecutive pair of ring-buffer positions
  // becomes one segment (2 vertices). Max segments = MAX_PARTICLES * TRAIL_LEN.
  if (trailMesh) { scene.remove(trailMesh); trailGeo.dispose(); }
  const maxSegVerts = MAX_PARTICLES * TRAIL_LEN * 2; // 2 verts per segment
  trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxSegVerts * 3), 3));
  trailGeo.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(maxSegVerts * 3), 3));
  trailGeo.setDrawRange(0, 0);
  trailMesh = new THREE.LineSegments(trailGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false }));
  trailMesh.renderOrder = 10;
  trailMesh.visible = showTrails;
  scene.add(trailMesh);
}

function updateBohmianMesh(posX, posY, Np) {
  const pts  = bohmGeo.attributes.position.array;
  const cols = bohmGeo.attributes.color.array;
  const [cr, cg, cb] = PARTICLE_PRESETS[trailColorIdx].rgb;
  let n = 0;
  for (let p = 0; p < Math.min(Np, MAX_PARTICLES); p++) {
    if (!posX || isNaN(posX[p])) continue;
    pts[n * 3    ] = -SS_X / 2 + posX[p] * SS_X;
    pts[n * 3 + 1] =  1.0 - posY[p] * SURFACE_SCALE;
    pts[n * 3 + 2] = 0.03;
    cols[n*3] = cr; cols[n*3+1] = cg; cols[n*3+2] = cb; // dot = full trail color
    n++;
  }
  bohmGeo.setDrawRange(0, n);
  bohmGeo.attributes.position.needsUpdate = true;
  bohmGeo.attributes.color.needsUpdate    = true;
}

// Append current positions to ring buffer (called every physics frame)
function appendTrailPositions(posX, posY, Np) {
  if (!trajBufX || trajBufX.length < MAX_PARTICLES * TRAIL_LEN) {
    trajBufX  = new Float32Array(MAX_PARTICLES * TRAIL_LEN).fill(NaN);
    trajBufY  = new Float32Array(MAX_PARTICLES * TRAIL_LEN).fill(NaN);
    trajHead  = new Int32Array(MAX_PARTICLES);
    trajFilled = new Int32Array(MAX_PARTICLES);
  }
  for (let p = 0; p < Math.min(Np, MAX_PARTICLES); p++) {
    if (isNaN(posX[p])) {
      // Dead particle – clear its trail
      trajBufX.fill(NaN, p * TRAIL_LEN, p * TRAIL_LEN + TRAIL_LEN);
      trajFilled[p] = 0;
      continue;
    }
    const h = trajHead[p];
    trajBufX[p * TRAIL_LEN + h] = posX[p];
    trajBufY[p * TRAIL_LEN + h] = posY[p];
    trajHead[p]  = (h + 1) % TRAIL_LEN;
    if (trajFilled[p] < TRAIL_LEN) trajFilled[p]++;
  }
}

// Rebuild trail geometry for rendering
function updateTrailMesh(Np) {
  if (!trailGeo || !trajBufX) return;
  const pts  = trailGeo.attributes.position.array;
  const cols = trailGeo.attributes.color.array;
  const [cr, cg, cb] = PARTICLE_PRESETS[trailColorIdx].rgb;
  const LX_NM = 200, LY_NM = 100; // physical domain size in nm
  let n = 0;
  for (let p = 0; p < Math.min(Np, MAX_PARTICLES); p++) {
    const filled = trajFilled[p];
    if (filled < 2) continue;
    const head = trajHead[p];

    // Walk backwards from newest to find how many steps fit within trailDisplayLen nm
    let kStart = filled - 2;
    let totalDist = 0;
    for (let k = filled - 2; k >= 0; k--) {
      const idxA = (head - filled + k     + TRAIL_LEN) % TRAIL_LEN;
      const idxB = (head - filled + k + 1 + TRAIL_LEN) % TRAIL_LEN;
      const ax = trajBufX[p * TRAIL_LEN + idxA], ay = trajBufY[p * TRAIL_LEN + idxA];
      const bx = trajBufX[p * TRAIL_LEN + idxB], by = trajBufY[p * TRAIL_LEN + idxB];
      if (isNaN(ax) || isNaN(bx)) break;
      const ddx = (bx - ax) * LX_NM, ddy = (by - ay) * LY_NM;
      totalDist += Math.sqrt(ddx * ddx + ddy * ddy);
      if (totalDist > trailDisplayLen) break;
      kStart = k;
    }

    // Emit segments from kStart → newest
    const segCount = Math.max(1, filled - 2 - kStart);
    for (let k = kStart; k < filled - 1; k++) {
      const idxA = (head - filled + k     + TRAIL_LEN) % TRAIL_LEN;
      const idxB = (head - filled + k + 1 + TRAIL_LEN) % TRAIL_LEN;
      const ax = trajBufX[p * TRAIL_LEN + idxA], ay = trajBufY[p * TRAIL_LEN + idxA];
      const bx = trajBufX[p * TRAIL_LEN + idxB], by = trajBufY[p * TRAIL_LEN + idxB];
      if (isNaN(ax) || isNaN(bx)) continue;
      const t  = (k - kStart) / segCount;   // 0 = oldest, 1 = newest
      const br = 0.25 + 0.75 * t;
      pts[n*3  ] = -SS_X/2 + ax*SS_X;  pts[n*3+1] = 1.0 - ay*SURFACE_SCALE;  pts[n*3+2] = 0.022;
      cols[n*3  ] = cr*br; cols[n*3+1] = cg*br; cols[n*3+2] = cb*br;
      n++;
      pts[n*3  ] = -SS_X/2 + bx*SS_X;  pts[n*3+1] = 1.0 - by*SURFACE_SCALE;  pts[n*3+2] = 0.022;
      cols[n*3  ] = cr*br; cols[n*3+1] = cg*br; cols[n*3+2] = cb*br;
      n++;
    }
  }
  trailGeo.setDrawRange(0, n);
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate    = true;
}

// ─────────────────────────────────────────────────────────────
//  CPU Bohmian mechanics (driven by GPU psi readback)
//  psi: Float32Array, layout psi[(iy*Nx+ix)*2] = Re (row-major)
//  bohmPosX/Y: fractional [0..1]
// ─────────────────────────────────────────────────────────────
function initBohmianPositions(rho, Np, Nx, Ny) {
  const cdf = new Float32Array(Nx * Ny);
  let tot = 0;
  for (let i = 0; i < Nx * Ny; i++) tot += rho[i];
  let cum = 0;
  for (let i = 0; i < Nx * Ny; i++) { cum += rho[i] / tot; cdf[i] = cum; }

  // Reset trail buffers
  trajBufX  = new Float32Array(MAX_PARTICLES * TRAIL_LEN).fill(NaN);
  trajBufY  = new Float32Array(MAX_PARTICLES * TRAIL_LEN).fill(NaN);
  trajHead  = new Int32Array(MAX_PARTICLES);
  trajFilled = new Int32Array(MAX_PARTICLES);

  bohmPosX = new Float32Array(Np);
  bohmPosY = new Float32Array(Np);
  bohmPassedSlit    = new Uint8Array(Np);
  bohmAnyPassedSlit = false;
  for (let p = 0; p < Np; p++) {
    const u = Math.random();
    let lo = 0, hi = Nx * Ny - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
    // rho is column-major: index = ix*Ny+iy, so ix=floor(lo/Ny), iy=lo%Ny
    const ix = Math.floor(lo / Ny), iy = lo % Ny;
    bohmPosX[p] = Math.max(0.01, Math.min(0.99, (ix + (Math.random() - 0.5)) / (Nx - 1)));
    bohmPosY[p] = Math.max(0.01, Math.min(0.99, (iy + (Math.random() - 0.5)) / (Ny - 1)));
  }
}

// Respawn any dead (NaN) Bohmian particles by sampling from the current density.
// rho is column-major (ix*Ny+iy), same layout as initBohmianPositions.
function respawnDeadParticles(rho, Np, Nx, Ny, absThick) {
  // Once any particle has passed the slit, stop respawning so the
  // simulation naturally winds down as particles exit the domain.
  if (bohmAnyPassedSlit) return;
  const dead = absThick || 0.05;
  // Collect indices of dead particles
  const deadList = [];
  for (let p = 0; p < Np; p++) {
    if (isNaN(bohmPosX[p])) deadList.push(p);
  }
  if (deadList.length === 0) return;

  // Build CDF from density, excluding absorber zone
  const cdf = new Float32Array(Nx * Ny);
  let tot = 0;
  const x0d = Math.ceil(dead * (Nx - 1));
  const x1d = Math.floor((1 - dead) * (Nx - 1));
  const y0d = Math.ceil(dead * (Ny - 1));
  const y1d = Math.floor((1 - dead) * (Ny - 1));
  for (let ix = x0d; ix <= x1d; ix++) {
    for (let iy = y0d; iy <= y1d; iy++) {
      tot += rho[ix * Ny + iy];
    }
  }
  if (tot < 1e-60) return; // no density — skip
  let cum = 0;
  for (let i = 0; i < Nx * Ny; i++) {
    const ix = Math.floor(i / Ny), iy = i % Ny;
    const inZone = ix >= x0d && ix <= x1d && iy >= y0d && iy <= y1d;
    cum += inZone ? rho[i] / tot : 0;
    cdf[i] = cum;
  }

  // Sample a new position for each dead particle
  for (const p of deadList) {
    const u = Math.random();
    let lo = 0, hi = Nx * Ny - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
    const ix = Math.floor(lo / Ny), iy = lo % Ny;
    bohmPosX[p] = Math.max(dead + 0.01, Math.min(1 - dead - 0.01, (ix + (Math.random() - 0.5)) / (Nx - 1)));
    bohmPosY[p] = Math.max(dead + 0.01, Math.min(1 - dead - 0.01, (iy + (Math.random() - 0.5)) / (Ny - 1)));
    // Clear the old (dead) trail so the fresh particle starts a new one
    if (trajBufX) {
      trajBufX.fill(NaN, p * TRAIL_LEN, p * TRAIL_LEN + TRAIL_LEN);
      trajBufY.fill(NaN, p * TRAIL_LEN, p * TRAIL_LEN + TRAIL_LEN);
      trajFilled[p] = 0;
    }
  }
}

function stepBohmian(psi, Np, Nx, Ny, Lx, Ly, Dt, absThick) {
  // psi: row-major, psi[(iy*Nx+ix)*2]=Re, *2+1=Im
  const Dx = Lx / (Nx - 1), Dy = Ly / (Ny - 1);
  const hbOverM = HB / ME;
  // Particles entering the absorbing boundary region are marked dead (NaN).
  // They vanish from the display rather than reflecting back.
  // dead zone matches the actual absorber — no extra margin
  const dead = (absThick || 0.05);
  for (let p = 0; p < Np; p++) {
    const px = bohmPosX[p], py = bohmPosY[p];
    if (isNaN(px)) continue; // already dead
    // Mark particle as having passed through the barrier (x > 0.52)
    if (!bohmPassedSlit[p] && px > 0.52) {
      bohmPassedSlit[p] = 1;
      bohmAnyPassedSlit = true;
    }
    // Kill particle if it drifts into absorber zone
    if (px < dead || px > 1 - dead || py < dead || py > 1 - dead) {
      bohmPosX[p] = NaN;
      continue;
    }
    let ix = Math.round(px * (Nx - 1));
    let iy = Math.round(py * (Ny - 1));
    ix = Math.max(1, Math.min(Nx - 2, ix));
    iy = Math.max(1, Math.min(Ny - 2, iy));
    const ci   = (iy * Nx + ix) * 2;
    const re   = psi[ci], im = psi[ci + 1];
    const mod2 = re * re + im * im;
    if (mod2 < 1e-60) { bohmPosX[p] = NaN; continue; }

    const dxRe = (psi[(iy * Nx + ix + 1) * 2    ] - psi[(iy * Nx + ix - 1) * 2    ]) / (2 * Dx);
    const dxIm = (psi[(iy * Nx + ix + 1) * 2 + 1] - psi[(iy * Nx + ix - 1) * 2 + 1]) / (2 * Dx);
    const vx   = hbOverM * (dxIm * re - dxRe * im) / mod2;

    const dyRe = (psi[((iy + 1) * Nx + ix) * 2    ] - psi[((iy - 1) * Nx + ix) * 2    ]) / (2 * Dy);
    const dyIm = (psi[((iy + 1) * Nx + ix) * 2 + 1] - psi[((iy - 1) * Nx + ix) * 2 + 1]) / (2 * Dy);
    const vy   = hbOverM * (dyIm * re - dyRe * im) / mod2;

    bohmPosX[p] = px + vx * Dt / Lx;
    bohmPosY[p] = py + vy * Dt / Ly;
  }
}

// ─────────────────────────────────────────────────────────────
//  GPU backend init
// ─────────────────────────────────────────────────────────────
function initGPU() {
  try {
    gpuSim = new GPUSim();
    useGPU = true;
    document.getElementById('hud-backend').textContent = '⚡ GPU';
    return true;
  } catch (e) {
    console.warn('GPU physics unavailable, falling back to CPU worker:', e.message);
    const el = document.getElementById('hud-backend');
    el.textContent = '🖥 CPU';
    el.title = e.message;  // hover tooltip shows exact reason
    // Show a dismissible banner so the user knows what failed
    const banner = document.createElement('div');
    banner.id = 'gpu-fallback-banner';
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#7a2020;color:#fff;font:12px/1.5 monospace;padding:6px 36px 6px 10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
    banner.textContent = '⚠ GPU unavailable — running on CPU (slow). Cause: ' + e.message;
    const cls = document.createElement('span');
    cls.textContent = '✕';
    cls.style.cssText = 'position:absolute;right:10px;top:4px;cursor:pointer;font-size:14px';
    cls.onclick = () => banner.remove();
    banner.appendChild(cls);
    document.body.appendChild(banner);
    // Auto-reduce resolution and speed to keep CPU manageable
    const resEl = document.getElementById('resolution');
    const spEl  = document.getElementById('speed');
    if (resEl && parseInt(resEl.value) > 64) { resEl.value = 64; resEl.dispatchEvent(new Event('input')); }
    if (spEl  && parseInt(spEl.value)  > 4)  { spEl.value  = 4;  spEl.dispatchEvent(new Event('input')); }
    return false;
  }
}

// ─────────────────────────────────────────────────────────────
//  CPU worker fallback
// ─────────────────────────────────────────────────────────────
function spawnWorker() {
  if (worker) worker.terminate();
  worker = new Worker(new URL('./worker.js', import.meta.url));
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type !== 'frame') return;
    NX = msg.Nx; NY = msg.Ny;
    dispatchFrame(msg.rho, msg.trajX, msg.trajY, msg.time, msg.norm);
    frameReady = true;
  };
}

// ─────────────────────────────────────────────────────────────
//  Frame dispatch (GPU and CPU paths merge here)
// ─────────────────────────────────────────────────────────────
function dispatchFrame(rho, trajX, trajY, time, norm) {
  lastRho  = rho;
  lastTime = time;
  if (viewMode === 'spacetime') updateSpacetime(rho, NX, NY, time);
  else                          updateSurface(rho, NX, NY);

  const Np = trajX ? Math.min(trajX.length, MAX_PARTICLES) : 0;
  if (Np) appendTrailPositions(trajX, trajY, Np);
  updateBohmianMesh(trajX, trajY, Np);
  updateTrailMesh(Np);

  // Auto-stop: particle-based criterion fires early when all post-slit particles exit;
  // time-based limit always applies as a hard cap (prevents stranded-particle hangs).
  if (!paused && useGPU && autoStop) {
    const particleStop = (showParticles || showTrails) && bohmAnyPassedSlit && bohmPosX &&
      !bohmPosX.some((x, i) => !isNaN(x) && bohmPassedSlit[i]);
    const timeStop = gpuSim.simTime >= gpuSim.stopTime;
    if (particleStop || timeStop) {
      paused = true;
      const btn = document.getElementById('btn-pause');
      btn.textContent = '\u25b6 Start';
      btn.classList.remove('btn-running');
    }
  }

  document.getElementById('hud-time').textContent = `t = ${(time * 1e15).toFixed(1)} fs`;
  document.getElementById('hud-norm').textContent = `‖ψ‖² = ${norm.toFixed(6)}`;
  drawAxes();
}

// ─────────────────────────────────────────────────────────────
//  GPU physics tick
// ─────────────────────────────────────────────────────────────
function gpuTick(n) {
  const { dispNx: Nx, Ny, Lx, Ly, Dt } = gpuSim;
  if (n > 0) {
    gpuSim.step(n);
    if ((showParticles || showTrails) && bohmPosX && gpuSim.psi) {
      for (let s = 0; s < n; s++)
        stepBohmian(gpuSim.psi, bohmNp, Nx, Ny, Lx, Ly, Dt, gpuSim._absThick);
      // Respawn any particles that died this frame so the count stays at bohmNp
      respawnDeadParticles(gpuSim.rho, bohmNp, Nx, Ny, gpuSim._absThick);
    }
  }
  dispatchFrame(gpuSim.rho, bohmPosX, bohmPosY, gpuSim.simTime, gpuSim.norm);
}

// ─────────────────────────────────────────────────────────────
//  Config from UI
// ─────────────────────────────────────────────────────────────
function getConfig() {
  const g = id => parseFloat(document.getElementById(id).value);
  const res = parseInt(document.getElementById('resolution').value);
  const Ly = 100e-9;  // physical domain height [m]
  // Slit positions are given as nm offsets from the centre of the domain
  const slitCenterY1 = 0.5 + parseInt(document.getElementById('slit1').value) * 1e-9 / Ly;
  const slitCenterY2 = 0.5 + parseInt(document.getElementById('slit2').value) * 1e-9 / Ly;
  const slitHalfWidth = g('slitWidth') * 1e-9 / Ly;
  const Dt = parseInt(document.getElementById('dt').value) * 0.05e-15;
  const xfrac = parseFloat(document.getElementById('x0').value) / 200; // nm → fraction of Lx=200nm
  return {
    Nx: 2 * res, Ny: res,
    slitCenterY1,
    slitCenterY2,
    slitHalfWidth,
    Dt,
    xfrac,
    velox         : g('momentum') * 1e5,
    sigmax        : g('sigmax') * 1e-9,
    sigmay        : g('sigmay') * 1e-9,
    Np            : parseInt(document.getElementById('np').value),
    stepsPerFrame : parseInt(document.getElementById('speed').value),
    absThick      : g('absthick'),
    absStrength   : parseFloat(document.getElementById('absstrength').value),
    fdMode        : solverMode === 'fd',
    rkMode        : solverMode === 'rk4',
  };
}

// ─────────────────────────────────────────────────────────────
//  Reset simulation
// ─────────────────────────────────────────────────────────────
function resetDefaults() {
  // Reset every <input type="range"> and <select> to its HTML defaultValue
  document.querySelectorAll('#panel input[type="range"]').forEach(el => {
    el.value = el.defaultValue;
    el.dispatchEvent(new Event('input'));
  });
  // Reset dual-slit to defaults and re-fire update
  const s1 = document.getElementById('slit1');
  const s2 = document.getElementById('slit2');
  s1.value = s1.defaultValue;
  s2.value = s2.defaultValue;
  s1.dispatchEvent(new Event('input'));
  s2.dispatchEvent(new Event('input'));
  resetSim();
}

function resetSim(keepPaused = false) {
  paused = keepPaused ? paused : false;
  const pauseBtn = document.getElementById('btn-pause');
  if (pauseBtn) {
    pauseBtn.textContent = paused ? '\u25b6 Start' : '\u23f8 Stop';
    pauseBtn.classList.toggle('btn-running', !paused);
  }

  const cfg = getConfig();
  stepsPerFrame = cfg.stepsPerFrame;
  bohmNp = cfg.Np;
  sliceHistory.length = 0;
  frameReady = true;

  if (cfg.Nx !== NX || cfg.Ny !== NY) {
    NX = cfg.Nx; NY = cfg.Ny;
    buildSurface();
    buildBohmianMesh();
  }

buildBarrier(0.5, cfg.slitCenterY1, cfg.slitCenterY2, cfg.slitHalfWidth);

  if (useGPU) {
    gpuSim.init(cfg);
    if (viewMode === 'density' || viewMode === 'phase') fitCamera2D();
    initBohmianPositions(gpuSim.rho, bohmNp, NX, NY);
    gpuTick(0);
  } else {
    frameReady = false;
    worker.postMessage({ type: 'init', config: cfg });
  }
}

// ─────────────────────────────────────────────────────────────
//  Visual palette & particle-colour pickers
// ─────────────────────────────────────────────────────────────
const PALETTE_META = {
  plasma:   'Plasma',
  viridis:  'Viridis',
  inferno:  'Inferno',
  magma:    'Magma',
  ocean:    'Ocean',
  fire:     'Fire',
  neon:     'Neon',
  coolwarm: 'Cool–Warm',
};

function paletteCssGrad(key) {
  const stops = WAVE_PALETTES[key];
  const parts = stops.map((c, i) => {
    const p = Math.round(i / (stops.length - 1) * 100);
    return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)}) ${p}%`;
  });
  return `linear-gradient(to right, ${parts.join(', ')})`;
}

function buildPaletteUI() {
  const btn   = document.getElementById('palette-btn');
  const strip = document.getElementById('palette-strip-cur');
  const lbl   = document.getElementById('palette-label-cur'); // may be null in compact layout
  const popup = document.getElementById('palette-popup');
  let open = false;

  // Populate popup with one row per palette
  Object.keys(PALETTE_META).forEach(key => {
    const opt = document.createElement('div');
    opt.className = 'palette-option' + (key === currentPaletteKey ? ' active' : '');
    opt.dataset.key = key;
    const s = document.createElement('div');
    s.className = 'p-strip';
    s.style.background = paletteCssGrad(key);
    const n = document.createElement('span');
    n.className = 'p-name';
    n.textContent = PALETTE_META[key];
    opt.appendChild(s);
    opt.appendChild(n);
    popup.appendChild(opt);
    opt.addEventListener('click', e => {
      e.stopPropagation();
      setWavePalette(key);
      strip.style.background = paletteCssGrad(key);
      if (lbl) lbl.textContent = PALETTE_META[key];
      popup.querySelectorAll('.palette-option').forEach(o =>
        o.classList.toggle('active', o.dataset.key === key));
      closePopup();
    });
  });

  strip.style.background = paletteCssGrad(currentPaletteKey);

  function openPopup() {
    const r = btn.getBoundingClientRect();
    // Try to open below; if it would clip the viewport, open above
    const popH = 8 * 30; // approx height
    const below = r.bottom + 6;
    popup.style.left = r.left + 'px';
    popup.style.top  = (below + popH > window.innerHeight ? r.top - popH - 6 : below) + 'px';
    popup.style.display = 'block';
    open = true;
  }
  function closePopup() { popup.style.display = 'none'; open = false; }

  btn.addEventListener('click', e => { e.stopPropagation(); open ? closePopup() : openPopup(); });
  document.addEventListener('click', () => { if (open) closePopup(); });
}

function buildParticleUI() {
  const btn      = document.getElementById('particle-btn');
  const dotCur   = document.getElementById('particle-dot-cur');
  const lblCur   = document.getElementById('particle-label-cur'); // may be null in compact layout
  const popup    = document.getElementById('particle-popup');
  let open = false;

  function rgbCss(rgb) {
    const [r, g, b] = rgb;
    return `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
  }
  function setActive(i) {
    trailColorIdx = i;
    dotCur.style.background = rgbCss(PARTICLE_PRESETS[i].rgb);
    if (lblCur) lblCur.textContent = PARTICLE_PRESETS[i].label.replace(/^\S+\s*/, '');
    popup.querySelectorAll('.p-dot').forEach((d, j) =>
      d.classList.toggle('active', j === i));
  }

  // Build popup grid
  PARTICLE_PRESETS.forEach((p, i) => {
    const d = document.createElement('div');
    d.className = 'p-dot' + (i === trailColorIdx ? ' active' : '');
    d.style.background = rgbCss(p.rgb);
    d.title = p.label.replace(/^\S+\s*/, '');
    d.addEventListener('click', e => { e.stopPropagation(); setActive(i); closePopup(); });
    popup.appendChild(d);
  });

  // Init current display
  setActive(trailColorIdx);

  function openPopup() {
    const r = btn.getBoundingClientRect();
    const popW = 200, popH = 90;
    const below = r.bottom + 6;
    const left = Math.max(4, Math.min(r.right - popW, window.innerWidth - popW - 4));
    popup.style.left  = left + 'px';
    popup.style.top   = (below + popH > window.innerHeight ? r.top - popH - 6 : below) + 'px';
    popup.style.display = 'flex';
    open = true;
  }
  function closePopup() { popup.style.display = 'none'; open = false; }

  btn.addEventListener('click', e => { e.stopPropagation(); open ? closePopup() : openPopup(); });
  document.addEventListener('click', () => { if (open) closePopup(); });
}

function buildWallUI() {
  const btn    = document.getElementById('wall-btn');
  const dotCur = document.getElementById('wall-dot-cur');
  const popup  = document.getElementById('wall-popup');
  if (!btn || !popup) return;
  let open = false;

  function hexCss(hex) {
    return `#${hex.toString(16).padStart(6,'0')}`;
  }
  function setActive(i) {
    wallColorIdx = i;
    dotCur.style.background = hexCss(WALL_PRESETS[i].hex);
    popup.querySelectorAll('.p-dot').forEach((d, j) =>
      d.classList.toggle('active', j === i));
    // Live-update existing barrier colour without full rebuild
    if (barrierMesh) {
      barrierMesh.traverse(c => {
        if (c.isMesh) c.material.color.setHex(WALL_PRESETS[i].hex);
      });
    }
  }

  WALL_PRESETS.forEach((p, i) => {
    const d = document.createElement('div');
    d.className = 'p-dot' + (i === wallColorIdx ? ' active' : '');
    d.style.background = hexCss(p.hex);
    d.title = p.label;
    d.addEventListener('click', e => { e.stopPropagation(); setActive(i); closePopup(); });
    popup.appendChild(d);
  });

  setActive(wallColorIdx);

  function openPopup() {
    const r = btn.getBoundingClientRect();
    const popW = 200, popH = 90;
    const below = r.bottom + 6;
    // Right-align to button's right edge, clamped inside viewport
    const left = Math.max(4, Math.min(r.right - popW, window.innerWidth - popW - 4));
    popup.style.left  = left + 'px';
    popup.style.top   = (below + popH > window.innerHeight ? r.top - popH - 6 : below) + 'px';
    popup.style.display = 'flex';
    open = true;
  }
  function closePopup() { popup.style.display = 'none'; open = false; }

  btn.addEventListener('click', e => { e.stopPropagation(); open ? closePopup() : openPopup(); });
  document.addEventListener('click', () => { if (open) closePopup(); });
}

// ─────────────────────────────────────────────────────────────
//  UI wiring
// ─────────────────────────────────────────────────────────────
function initUI() {
  const sliders = ['slitWidth','momentum','sigmax','sigmay','trailen','np','speed','dt','absthick','absstrength','x0'];
  sliders.forEach(id => {
    const el  = document.getElementById(id);
    const lbl = document.getElementById(`${id}-val`);
    const update = () => {
      if      (id === 'slitWidth') lbl.textContent = `${parseFloat(el.value).toFixed(1)} nm`;
      else if (id === 'momentum') lbl.textContent = `${parseFloat(el.value).toFixed(2)} × 10⁵ m/s`;
      else if (id === 'sigmax')   lbl.textContent = `${parseFloat(el.value).toFixed(1)} nm`;
      else if (id === 'sigmay')   lbl.textContent = `${parseFloat(el.value).toFixed(1)} nm`;
      else if (id === 'trailen')  lbl.textContent = `${el.value} nm`;
      else if (id === 'np')       lbl.textContent = el.value;
      else if (id === 'speed')    lbl.textContent = `${el.value}×`;
      else if (id === 'dt')       lbl.textContent = `${(parseInt(el.value) * 0.05).toFixed(2)} fs`;
      else if (id === 'absthick')    lbl.textContent = `${(parseFloat(el.value)*100).toFixed(0)}%`;
      else if (id === 'absstrength') lbl.textContent = parseFloat(el.value).toFixed(0);
      else if (id === 'x0')          lbl.textContent = `${parseInt(el.value)} nm`;
      else                        lbl.textContent = parseFloat(el.value).toFixed(2);
    };
    update();
    el.addEventListener('input', update);
  });

  // Trail length slider also triggers a mesh refresh (works while paused)
  document.getElementById('trailen').addEventListener('input', (e) => {
    trailDisplayLen = parseInt(e.target.value);
    updateTrailMesh(bohmNp);
  });

  const s1 = document.getElementById('slit1');
  const s2 = document.getElementById('slit2');
  const slitsVal = document.getElementById('slits-val');
  const dualTrack = document.getElementById('dual-track');
  const slitWidthEl = document.getElementById('slitWidth');

  function previewBarrier() {
    const Ly = 100e-9;
    const v1 = parseInt(s1.value), v2 = parseInt(s2.value);
    const hw = parseFloat(slitWidthEl.value) * 1e-9 / Ly;
    buildBarrier(0.5, 0.5 + v1 * 1e-9 / Ly, 0.5 + v2 * 1e-9 / Ly, hw);
  }

  function updateDualSlit() {
    let v1 = parseInt(s1.value), v2 = parseInt(s2.value);
    // enforce minimum separation of 3 nm
    if (v1 > -2) { v1 = -2; s1.value = v1; }
    if (v2 <  2) { v2 =  2; s2.value = v2; }
    if (v1 >= v2 - 3) { v1 = v2 - 3; s1.value = v1; }
    const pct1 = (v1 + 45) / 90 * 100;
    const pct2 = (v2 + 45) / 90 * 100;
    dualTrack.style.background =
      `linear-gradient(to right,` +
      ` rgba(0,204,255,0.15) 0%, rgba(0,204,255,0.15) ${pct1}%,` +
      ` rgba(255,107,107,0.45) ${pct1}%, rgba(255,107,107,0.45) ${pct2}%,` +
      ` rgba(0,204,255,0.15) ${pct2}%, rgba(0,204,255,0.15) 100%)`;
    const sign2 = v2 >= 0 ? '+' : '';
    slitsVal.textContent = `${v1} / ${sign2}${v2} nm`;
    previewBarrier();
  }
  updateDualSlit();
  s1.addEventListener('input', updateDualSlit);
  s2.addEventListener('input', updateDualSlit);
  slitWidthEl.addEventListener('input', previewBarrier);

  // lift the active handle above the other
  s1.addEventListener('pointerdown', () => { s1.style.zIndex = 3; s2.style.zIndex = 2; });
  s2.addEventListener('pointerdown', () => { s2.style.zIndex = 3; s1.style.zIndex = 2; });

  // While dragging slit/width: pause physics; on release: full reset + resume
  ['slit1', 'slit2', 'slitWidth'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('pointerdown', () => { paused = true; });
    el.addEventListener('change', () => {
      paused = false;
      const btn = document.getElementById('btn-pause');
      btn.textContent = '⏸ Stop';
      btn.classList.add('btn-running');
      resetSim();
    });
  });

  // All remaining physics sliders: pause while dragging, live-preview on input, resume on release
  const liveReset = ['momentum','sigmax','sigmay','np','absthick','absstrength','dt','x0'];
  liveReset.forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('pointerdown', () => { paused = true; });
    el.addEventListener('input', resetSim);
    el.addEventListener('change', () => {
      paused = false;
      const btn = document.getElementById('btn-pause');
      btn.textContent = '⏸ Stop';
      btn.classList.add('btn-running');
    });
  });

  document.getElementById('resolution').addEventListener('change', e => {
    // Automatically tone down steps/frame for the heavier grid
    const speedEl = document.getElementById('speed');
    const speedLbl = document.getElementById('speed-val');
    const res = parseInt(e.target.value);
    if (res >= 256) {
      speedEl.value = '8'; speedLbl.textContent = '8×';
    } else {
      speedEl.value = '16'; speedLbl.textContent = '16×';
    }
    resetSim();
  });
  document.getElementById('btn-reset').addEventListener('click', () => resetSim(paused));
  document.getElementById('btn-defaults').addEventListener('click', resetDefaults);

  document.getElementById('btn-pause').addEventListener('click', () => {
    paused = !paused;
    const btn = document.getElementById('btn-pause');
    btn.textContent = paused ? '\u25b6 Start' : '\u23f8 Stop';
    btn.classList.toggle('btn-running', !paused);
  });

  document.getElementById('btn-particles').addEventListener('click', () => {
    showParticles = !showParticles;
    const showing = viewMode !== 'spacetime';
    bohmPoints.visible = showParticles && showing;
    document.getElementById('btn-particles').classList.toggle('btn-off', !showParticles);
  });

  document.getElementById('btn-trails').addEventListener('click', () => {
    showTrails = !showTrails;
    const showing = viewMode !== 'spacetime';
    trailMesh.visible = showTrails && showing;
    document.getElementById('btn-trails').classList.toggle('btn-off', !showTrails);
  });

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const panel = document.getElementById('panel');
      if (btn.dataset.tab === 'expert') panel.classList.add('show-expert');
      else panel.classList.remove('show-expert');
    });
  });

  // Auto-stop toggle
  document.getElementById('btn-autostop').addEventListener('click', () => {
    autoStop = !autoStop;
    document.getElementById('btn-autostop').textContent = autoStop ? 'Auto-stop: ON' : 'Auto-stop: OFF';
    document.getElementById('btn-autostop').classList.toggle('btn-off', !autoStop);
  });

  document.getElementById('btn-fdmode').addEventListener('click', () => {
    const modes  = ['fft', 'fd', 'rk4'];
    const labels = { fft: 'Solver: FFT split-op', fd: 'Solver: FD leapfrog', rk4: 'Solver: RK4' };
    solverMode = modes[(modes.indexOf(solverMode) + 1) % modes.length];
    document.getElementById('btn-fdmode').textContent = labels[solverMode];
    resetSim(paused);
  });

  buildPaletteUI();
  buildParticleUI();
  buildWallUI();

  document.getElementById('btn-view').addEventListener('click', () => {
    const cycle = ['density','phase','surface'];
    viewMode = cycle[(cycle.indexOf(viewMode) + 1) % cycle.length];

    const isFlat = (viewMode === 'density' || viewMode === 'phase');
    surfaceMesh.visible    = true;
    spacetimeGroup.visible = false;
    bohmPoints.visible     = showParticles;
    trailMesh.visible      = showTrails;

    if (isFlat) setCameraView('2d');
    else        setCameraView('3d');

    if (lastRho) updateSurface(lastRho, NX, NY);

    drawAxes();

    const labels = {
      density: '2D Density',
      phase:   'Re(\u03c8)',
      surface: '3D Surface',
    };
    document.getElementById('btn-view').textContent = labels[viewMode];
  });

  document.getElementById('btn-wire').addEventListener('click', () => {
    const mat = surfaceMesh.material;
    mat.wireframe = !mat.wireframe;
    document.getElementById('btn-wire').textContent = mat.wireframe ? 'Solid' : 'Wireframe';
  });
}

window.addEventListener('resize', () => {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (viewMode === 'density' || viewMode === 'phase') fitCamera2D();
  drawAxes();
});

// ─────────────────────────────────────────────────────────────
//  Animation loop
// ─────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();

  const now = performance.now();
  _perfFrameCount++;
  const elapsed = now - _perfLastTime;
  if (elapsed >= 1000) {
    _perfFPS = (_perfFrameCount / elapsed * 1000).toFixed(1);
    const simElapsed = useGPU && gpuSim ? (gpuSim.simTime - _perfLastSimT) * 1e15 : 0;
    _perfSimSpeed   = (simElapsed / elapsed * 1000).toFixed(0);  // fs/s
    _perfStepsPerS  = ((stepsPerFrame * _perfFrameCount) / elapsed * 1000).toFixed(0);
    _perfLastTime   = now;
    _perfLastSimT   = useGPU && gpuSim ? gpuSim.simTime : 0;
    _perfFrameCount = 0;
    const hud = document.getElementById('perf-hud');
    if (hud) hud.innerHTML =
      `FPS: ${_perfFPS}<br>` +
      `Steps/s: ${_perfStepsPerS}<br>` +
      `Sim speed: ${_perfSimSpeed} fs/s`;
  }

  if (!paused) {
    stepsPerFrame = parseInt(document.getElementById('speed').value);
    if (useGPU) {
      gpuTick(stepsPerFrame);
    } else if (frameReady) {
      frameReady = false;
      worker.postMessage({ type: 'step', stepsPerFrame });
    }
  }

  renderer.render(scene, camera);
}

// ─────────────────────────────────────────────────────────────
//  Boot
// ─────────────────────────────────────────────────────────────
function boot() {
  buildSurface();
  buildSpacetime();
  buildBohmianMesh();
  buildBarrier(0.5, 0.5 + (-8e-9 / 100e-9), 0.5 + (8e-9 / 100e-9), 4e-9 / 100e-9);
  initUI();

  useGPU = initGPU();
  if (!useGPU) spawnWorker();

  _perfLastTime = performance.now();

  // Start in 2D density view
  spacetimeGroup.visible = false;
  surfaceMesh.visible    = true;
  setCameraView('2d');
  document.getElementById('btn-view').textContent = '2D Density';

  try {
    resetSim();
  } catch (err) {
    console.error('resetSim failed:', err);
  }

  drawAxes();
  animate();
}

boot();

