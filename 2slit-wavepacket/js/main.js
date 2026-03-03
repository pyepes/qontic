import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GPUSim } from './gpu-sim.js';

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
    [0.001, 0.000, 0.014], [0.120, 0.027, 0.270], [0.298, 0.024, 0.373],
    [0.460, 0.066, 0.310], [0.621, 0.120, 0.200], [0.768, 0.224, 0.089],
    [0.886, 0.368, 0.027], [0.968, 0.532, 0.020], [0.993, 0.715, 0.139],
    [0.987, 0.891, 0.429], [0.988, 1.000, 0.643],
  ],
  magma: [
    [0.001, 0.000, 0.014], [0.086, 0.006, 0.189], [0.246, 0.022, 0.349],
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
    [0.000, 0.000, 0.000], [0.280, 0.000, 0.000], [0.570, 0.010, 0.000],
    [0.820, 0.080, 0.000], [0.970, 0.220, 0.000], [1.000, 0.400, 0.000],
    [1.000, 0.590, 0.000], [1.000, 0.770, 0.050], [1.000, 0.910, 0.300],
    [1.000, 0.970, 0.650], [1.000, 1.000, 1.000],
  ],
  neon: [
    [0.000, 0.000, 0.000], [0.000, 0.100, 0.100], [0.000, 0.300, 0.200],
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

let currentPaletteKey = 'plasma';
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
}
buildLUT(WAVE_PALETTES.plasma);

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
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.setClearColor(0x090918);

const scene = new THREE.Scene();
// Fog is added only in 3D surface mode; starts off for default 2D view.
// scene.fog set by setCameraView()

const camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.001, 100);
camera.position.set(0, 0, 3.0);   // start top-down for 2D mode
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

// ─── Camera view helpers ─────────────────────────────────────
function setCameraView(mode) {
  if (mode === '2d') {
    // Top-down: camera far enough to see the 4×2 scene plane
    camera.position.set(0, 0, 5.5);
    camera.up.set(0, 1, 0);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    controls.enableRotate = false;
    scene.fog = null;
    if (bohmPoints) bohmPoints.material.size = 0.028;
  } else {
    // 3D perspective – pull back to show the 4×2 plane
    controls.enableRotate = true;
    camera.position.set(0, -4.0, 3.0);
    camera.up.set(0, 0, 1);
    controls.target.set(0, 0, 0);
    controls.maxPolarAngle = Math.PI;
    controls.minPolarAngle = 0;
    scene.fog = new THREE.FogExp2(0x090918, 0.25);
    if (bohmPoints) bohmPoints.material.size = 0.018;
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
//  Shared simulation state
// ─────────────────────────────────────────────────────────────
let NX = 256, NY = 256;
const SURFACE_SCALE = 2.0;   // Y scene extent  (maps to Ly = 100 nm)
const SS_X          = 4.0;   // X scene extent  (maps to Lx = 200 nm, 2× wider)
const HEIGHT_PEAK   = 0.9;

let viewMode    = 'density';  // default: flat 2D density (plasma)
let showBohmian = true;
let paused      = false;
let stepsPerFrame = 16;

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

// Trajectory trail ring-buffer
const TRAIL_LEN = 100;             // steps stored per particle
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
let trailColorIdx = 0;
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

  let rhoMax = 0;
  for (let i = 0; i < N; i++) if (rho[i] > rhoMax) rhoMax = rho[i];
  const heightScale = (!flat && rhoMax > 0) ? HEIGHT_PEAK / rhoMax : 0;

  // For phase coloring we need the complex psi from GPU readback
  const psi = (viewMode === 'phase' && useGPU && gpuSim) ? gpuSim.psi : null;

  for (let ix = 0; ix < Nx; ix++) {
    for (let iy = 0; iy < Ny; iy++) {
      const vIdx = iy * Nx + ix;   // Three.js vertex index (row-major)
      const rIdx = ix * Ny + iy;   // rho column-major index
      const t    = rhoMax > 0 ? Math.min(1, rho[rIdx] / rhoMax) : 0;

      posArr[vIdx * 3 + 2] = flat ? 0 : rho[rIdx] * heightScale;

      let r, g, b;
      if (psi) {
        // Phase view: hue = arg(ψ)/(2π),  brightness = |ψ|^0.45
        const re  = psi[vIdx * 2];
        const im  = psi[vIdx * 2 + 1];
        const hue = Math.atan2(im, re) / (2 * Math.PI); // −0.5 … +0.5
        const val = Math.pow(t, 0.45);
        [r, g, b] = hsvToRgb(hue, 0.92, val);
      } else if (viewMode === 'density') {
        // Density view: plasma colormap on |ψ|², gamma=0.45 for fringe visibility
        [r, g, b] = sampleLUT(Math.pow(t, 0.45));
      } else {
        // 3D surface: plasma with sqrt gamma
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

  const wallThick = 0.04;   // scene-unit thickness along x
  const slit1Lo = slitCenterY1 - slitHalfWidth;
  const slit1Hi = slitCenterY1 + slitHalfWidth;
  const slit2Lo = slitCenterY2 - slitHalfWidth;
  const slit2Hi = slitCenterY2 + slitHalfWidth;

  const bGroup = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color: 0x00ccff, transparent: true, opacity: 0.55 });

  // Wall segments in fractional coords: [lo, hi]
  const segments = [
    [0,       slit1Lo],
    [slit1Hi, slit2Lo],
    [slit2Hi, 1.0],
  ];

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
  bohmPoints.visible = showBohmian;
  scene.add(bohmPoints);

  // Trail mesh
  if (trailMesh) { scene.remove(trailMesh); trailGeo.dispose(); }
  const maxTrailPts = MAX_PARTICLES * TRAIL_LEN;
  trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxTrailPts * 3), 3));
  trailGeo.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(maxTrailPts * 3), 3));
  trailGeo.setDrawRange(0, 0);
  trailMesh = new THREE.Points(trailGeo,
    new THREE.PointsMaterial({ vertexColors: true, size: 0.007, sizeAttenuation: true, depthTest: false }));
  trailMesh.renderOrder = 10;
  trailMesh.visible = showBohmian;
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
  let n = 0;
  for (let p = 0; p < Math.min(Np, MAX_PARTICLES); p++) {
    const filled = trajFilled[p];
    if (!filled) continue;
    const head = trajHead[p];
    for (let k = 0; k < filled; k++) {
      // k=0 is oldest entry, k=filled-1 is most recent
      const ringIdx = (head - filled + k + TRAIL_LEN) % TRAIL_LEN;
      const bx = trajBufX[p * TRAIL_LEN + ringIdx];
      const by = trajBufY[p * TRAIL_LEN + ringIdx];
      if (isNaN(bx)) continue;
      pts[n * 3    ] = -SS_X / 2 + bx * SS_X;
      pts[n * 3 + 1] =  1.0 - by * SURFACE_SCALE;
      pts[n * 3 + 2] = 0.022;
      const age = k / Math.max(1, filled - 1);   // 0=oldest … 1=newest
      const bright = age * age;                   // quadratic fade
      cols[n * 3    ] = cr * bright;
      cols[n * 3 + 1] = cg * bright;
      cols[n * 3 + 2] = cb * bright;
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

function stepBohmian(psi, Np, Nx, Ny, Lx, Ly, Dt, absThick) {
  // psi: row-major, psi[(iy*Nx+ix)*2]=Re, *2+1=Im
  const Dx = Lx / (Nx - 1), Dy = Ly / (Ny - 1);
  const hbOverM = HB / ME;
  // Particles entering the absorbing boundary region are marked dead (NaN).
  // They vanish from the display rather than reflecting back.
  const dead = (absThick || 0.05) + 0.015;
  for (let p = 0; p < Np; p++) {
    const px = bohmPosX[p], py = bohmPosY[p];
    if (isNaN(px)) continue; // already dead
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
    document.getElementById('hud-backend').textContent = '🖥 CPU';
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
  if (viewMode === 'spacetime') updateSpacetime(rho, NX, NY, time);
  else                          updateSurface(rho, NX, NY);

  const Np = trajX ? Math.min(trajX.length, MAX_PARTICLES) : 0;
  if (Np) appendTrailPositions(trajX, trajY, Np);
  updateBohmianMesh(trajX, trajY, Np);
  updateTrailMesh(Np);

  // Auto-stop when sim time reaches the natural end point (like MATLAB Nt limit)
  if (!paused && useGPU && gpuSim.simTime >= gpuSim.stopTime) {
    paused = true;
    const btn = document.getElementById('btn-pause');
    btn.textContent = '\u25b6 Start';
    btn.classList.remove('btn-running');
  }

  document.getElementById('hud-time').textContent = `t = ${(time * 1e15).toFixed(1)} fs`;
  document.getElementById('hud-norm').textContent = `‖ψ‖² = ${norm.toFixed(6)}`;
}

// ─────────────────────────────────────────────────────────────
//  GPU physics tick
// ─────────────────────────────────────────────────────────────
function gpuTick(n) {
  const { Nx, Ny, Lx, Ly, Dt } = gpuSim;
  if (n > 0) {
    gpuSim.step(n);
    if (showBohmian && bohmPosX && gpuSim.psi)
      for (let s = 0; s < n; s++)
        stepBohmian(gpuSim.psi, bohmNp, Nx, Ny, Lx, Ly, Dt, gpuSim._absThick);
  }
  dispatchFrame(gpuSim.rho, bohmPosX, bohmPosY, gpuSim.simTime, gpuSim.norm);
}

// ─────────────────────────────────────────────────────────────
//  Config from UI
// ─────────────────────────────────────────────────────────────
function getConfig() {
  const g = id => parseFloat(document.getElementById(id).value);
  const res = parseInt(document.getElementById('resolution').value);
  return {
    Nx: res, Ny: res,
    slitCenterY1  : g('slit1'),
    slitCenterY2  : g('slit2'),
    slitHalfWidth : g('slitWidth'),
    velox         : g('momentum') * 1e5,
    sigmax        : g('sigmax') * 1e-9,
    Np            : parseInt(document.getElementById('np').value),
    stepsPerFrame : parseInt(document.getElementById('speed').value),
    absThick      : g('absthick'),
  };
}

// ─────────────────────────────────────────────────────────────
//  Reset simulation
// ─────────────────────────────────────────────────────────────
function resetSim() {
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
  const lbl   = document.getElementById('palette-label-cur');
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
      lbl.textContent = PALETTE_META[key];
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
  const container = document.getElementById('particle-picker');
  PARTICLE_PRESETS.forEach((p, i) => {
    const dot = document.createElement('div');
    dot.className = 'particle-dot' + (i === trailColorIdx ? ' active' : '');
    const [r, g, b] = p.rgb;
    dot.style.background = `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
    // Tooltip: strip leading emoji
    dot.title = p.label.replace(/^\S+\s*/, '');
    dot.addEventListener('click', () => {
      trailColorIdx = i;
      container.querySelectorAll('.particle-dot').forEach((d, j) =>
        d.classList.toggle('active', j === i));
    });
    container.appendChild(dot);
  });
}

// ─────────────────────────────────────────────────────────────
//  UI wiring
// ─────────────────────────────────────────────────────────────
function initUI() {
  const sliders = ['slit1','slit2','slitWidth','momentum','sigmax','np','speed','absthick'];
  sliders.forEach(id => {
    const el  = document.getElementById(id);
    const lbl = document.getElementById(`${id}-val`);
    const update = () => {
      if      (id === 'momentum') lbl.textContent = `${parseFloat(el.value).toFixed(2)} × 10⁵ m/s`;
      else if (id === 'sigmax')   lbl.textContent = `${parseFloat(el.value).toFixed(1)} nm`;
      else if (id === 'np')       lbl.textContent = el.value;
      else if (id === 'speed')    lbl.textContent = `${el.value}×`;
      else if (id === 'absthick') lbl.textContent = `${(parseFloat(el.value)*100).toFixed(0)}%`;
      else                        lbl.textContent = parseFloat(el.value).toFixed(2);
    };
    update();
    el.addEventListener('input', update);
  });

  document.getElementById('resolution').addEventListener('change', resetSim);
  document.getElementById('btn-reset').addEventListener('click', resetSim);

  document.getElementById('btn-pause').addEventListener('click', () => {
    paused = !paused;
    const btn = document.getElementById('btn-pause');
    btn.textContent = paused ? '▶ Start' : '⏸ Stop';
    btn.classList.toggle('btn-running', !paused);
  });

  document.getElementById('btn-bohmian').addEventListener('click', () => {
    showBohmian = !showBohmian;
    const showing = viewMode !== 'spacetime';
    bohmPoints.visible = showBohmian && showing;
    trailMesh.visible  = showBohmian && showing;
    document.getElementById('btn-bohmian').textContent =
      showBohmian ? 'Hide Particles' : 'Show Particles';
  });

  buildPaletteUI();
  buildParticleUI();

  document.getElementById('btn-view').addEventListener('click', () => {
    // Cycle: density → phase → 3D surface → spacetime → density
    const cycle = ['density','phase','surface','spacetime'];
    viewMode = cycle[(cycle.indexOf(viewMode) + 1) % cycle.length];

    const isFlat = (viewMode === 'density' || viewMode === 'phase');
    const isSurf = isFlat || viewMode === 'surface';
    surfaceMesh.visible    = isSurf;
    spacetimeGroup.visible = (viewMode === 'spacetime');
    bohmPoints.visible     = isSurf && showBohmian;
    trailMesh.visible      = isSurf && showBohmian;

    if (isFlat) setCameraView('2d');
    else        setCameraView('3d');

    const labels = {
      density:    '� 2D Density',
      phase:      '🌈 2D Phase',
      surface:    '🌊 3D Surface',
      spacetime:  '🕐 Spacetime',
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
});

// ─────────────────────────────────────────────────────────────
//  Animation loop
// ─────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();

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
  buildBarrier(0.5, 0.422, 0.578, 0.039);
  initUI();

  useGPU = initGPU();
  if (!useGPU) spawnWorker();

  // Start in 2D density view
  spacetimeGroup.visible = false;
  surfaceMesh.visible    = true;
  setCameraView('2d');
  document.getElementById('btn-view').textContent = '🗺 2D Density';

  try {
    resetSim();
  } catch (err) {
    console.error('resetSim failed:', err);
  }

  animate();
}

boot();

