'use strict';
// ============================================================
//  GPU-accelerated split-operator quantum simulation
//  Uses WebGL2 fragment shaders + ping-pong textures
//
//  Wavefunction stored as RGBA32F textures:  R=Re(ψ), G=Im(ψ)
//  Physics: split-operator with Stockham out-of-place FFT
//  Absorbing BC: cos^8 ramp on all 4 edges (no reflections)
// ============================================================

const HB = 1.054571817e-34;
const ME = 9.10938356e-31;
const QE = 1.602176634e-19;

// ─── Vertex shader (full-screen quad, no uniforms needed) ────
const VS = `#version 300 es
layout(location=0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

// ─── Pointwise complex multiply: out = A * B ─────────────────
const FS_CMUL = `#version 300 es
precision highp float;
uniform sampler2D uA, uB;
out vec4 oC;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec2 a = texelFetch(uA, c, 0).rg;
  vec2 b = texelFetch(uB, c, 0).rg;
  oC = vec4(a.x*b.x - a.y*b.y,
            a.x*b.y + a.y*b.x, 0.0, 1.0);
}`;

// ─── Stockham out-of-place FFT – one butterfly pass ──────────
//
//  For pass p (0-indexed), stride s = 2^p, N2 = N/2:
//    output index `out_i` (along x for horiz, y for vert)
//    r       = out_i mod (2s)
//    q       = out_i div (2s)
//    k       = r mod s
//    isSecond = r >= s
//    aIdx = q*s + k          (first  input index)
//    bIdx = aIdx + N2        (second input index)
//    angle = dir * (-π) * k / s   (dir=+1 forward, -1 inverse)
//    out  = isSecond ? (a - w*b) : (a + w*b)
//
//  After log2(N) passes the FFT is complete, in natural order.
const FS_FFT = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform float uPass;
uniform float uN;
uniform float uHoriz;  // 1.0 = horizontal (x), 0.0 = vertical (y)
uniform float uDir;    // +1.0 = forward, -1.0 = inverse
out vec4 oC;
const float PI = 3.141592653589793;
void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int out_i = (uHoriz > 0.5) ? coord.x : coord.y;

  int s  = 1 << int(uPass);
  int N2 = int(uN) >> 1;
  int r  = out_i - (out_i / (2*s)) * (2*s); // out_i % (2s)
  int q  = out_i / (2*s);
  int k  = r - (r / s) * s;                  // r % s
  bool isSecond = (r >= s);

  int aIdx = q * s + k;
  int bIdx = aIdx + N2;

  ivec2 ca = (uHoriz > 0.5) ? ivec2(aIdx, coord.y) : ivec2(coord.x, aIdx);
  ivec2 cb = (uHoriz > 0.5) ? ivec2(bIdx, coord.y) : ivec2(coord.x, bIdx);

  vec2 a  = texelFetch(uTex, ca, 0).rg;
  vec2 b  = texelFetch(uTex, cb, 0).rg;

  float ang = uDir * (-PI) * float(k) / float(s);
  vec2  w   = vec2(cos(ang), sin(ang));
  vec2  wb  = vec2(w.x*b.x - w.y*b.y,  w.x*b.y + w.y*b.x);

  oC = vec4(isSecond ? (a - wb) : (a + wb), 0.0, 1.0);
}`;

// ─── Scale by 1/(Nx*Ny) — absorption handled by CAP in vprop ─
// The sin² mask is NOT applied here; it caused the initial packet's
// absorber-zone tails to be over-killed when combined with the CAP.
// The capdecay texture is used separately by the FD leapfrog path.
const FS_ABS = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform float uScale; // 1 / (Nx * Ny)
out vec4 oC;
void main() {
  ivec2 c  = ivec2(gl_FragCoord.xy);
  vec2  psi = texelFetch(uTex, c, 0).rg * uScale;
  oC = vec4(psi, 0.0, 1.0);
}`;

// ─── Compute |ψ|² → R channel ────────────────────────────────
const FS_RHO = `#version 300 es
precision highp float;
uniform sampler2D uTex;
out vec4 oC;
void main() {
  vec2 p = texelFetch(uTex, ivec2(gl_FragCoord.xy), 0).rg;
  oC = vec4(p.x*p.x + p.y*p.y, 0.0, 0.0, 1.0);
}`;

// ─── Barrier mask: multiply ψ by real mask (R channel) ───────
const FS_MASK = `#version 300 es
precision highp float;
uniform sampler2D uPsi;
uniform sampler2D uMask;
out vec4 oC;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec2 psi = texelFetch(uPsi, c, 0).rg;
  float m   = texelFetch(uMask, c, 0).r;
  oC = vec4(psi * m, 0.0, 1.0);
}`;

// ─── FD leapfrog pass 1: update R using current I ────────────
// Schrödinger: ∂R/∂t = +(ℏ/2m)∇²I
// R_half = R + uAX*(I_xp+I_xm-2I) + uAY*(I_yp+I_ym-2I)
const FS_FD_R = `#version 300 es
precision highp float;
uniform sampler2D uPsi;   // (R, I)
uniform sampler2D uMask;  // 1=free, 0=barrier/boundary
uniform float uAX;
uniform float uAY;
out vec4 oC;
void main() {
  ivec2 c  = ivec2(gl_FragCoord.xy);
  ivec2 sz = textureSize(uPsi, 0);
  if (c.x<=0||c.x>=sz.x-1||c.y<=0||c.y>=sz.y-1||texelFetch(uMask,c,0).r<0.5){
    oC=vec4(0.0,0.0,0.0,1.0); return;
  }
  vec2  C   = texelFetch(uPsi, c, 0).rg;
  float I   = C.g;
  float Ixp = texelFetch(uPsi, c+ivec2(1,0), 0).g;
  float Ixm = texelFetch(uPsi, c-ivec2(1,0), 0).g;
  float Iyp = texelFetch(uPsi, c+ivec2(0,1), 0).g;
  float Iym = texelFetch(uPsi, c-ivec2(0,1), 0).g;
  float laplI = uAX*(Ixp+Ixm-2.0*I) + uAY*(Iyp+Iym-2.0*I);
  oC = vec4(C.r - laplI, I, 0.0, 1.0);  // R_half = R - (hbar/2m)*lapl(I)
}`;

// ─── FD leapfrog pass 2: update I using R_half, apply CAP ────────────
// Schrödinger:  ∂I/∂t = -(ℏ/2m)∇²R
//   I_new = I - uAX*(R_xp+R_xm-2R) - uAY*(R_yp+R_ym-2R)
// Then multiply both (R_half, I_new) by per-pixel CAP decay
const FS_FD_I = `#version 300 es
precision highp float;
uniform sampler2D uHalf;     // (R_half, I_orig) from pass 1
uniform sampler2D uCapDecay; // per-pixel exp(-Γ·dt)
uniform sampler2D uMask;     // 1=free, 0=barrier/boundary
uniform float uAX;
uniform float uAY;
out vec4 oC;
void main() {
  ivec2 c  = ivec2(gl_FragCoord.xy);
  ivec2 sz = textureSize(uHalf, 0);
  if (c.x<=0||c.x>=sz.x-1||c.y<=0||c.y>=sz.y-1||texelFetch(uMask,c,0).r<0.5){
    oC=vec4(0.0,0.0,0.0,1.0); return;
  }
  vec2  H   = texelFetch(uHalf, c, 0).rg;
  float R   = H.r;
  float Rxp = texelFetch(uHalf, c+ivec2(1,0), 0).r;
  float Rxm = texelFetch(uHalf, c-ivec2(1,0), 0).r;
  float Ryp = texelFetch(uHalf, c+ivec2(0,1), 0).r;
  float Rym = texelFetch(uHalf, c-ivec2(0,1), 0).r;
  float laplR = uAX*(Rxp+Rxm-2.0*R) + uAY*(Ryp+Rym-2.0*R);
  float I_new = H.g + laplR;  // I_new = I + (hbar/2m)*lapl(R_half)
  float decay = texelFetch(uCapDecay, c, 0).r;
  oC = vec4(R*decay, I_new*decay, 0.0, 1.0);
}`;

// ─── RK4: dψ/dt = (iℏ/2m)∇²ψ in split real/imag form ───────
// Computes (dR/dt, dI/dt) = (-cY·∇²I, +cX·∇²R) using the 5-point stencil.
// Dirichlet BC enforced via uMask: ψ=0 at boundary and barrier pixels.
// Only 5 texelFetch calls (reads rg pair per neighbour).
// dψ/dt = (iℏ/2m)∇²ψ  (free Schrödinger, no CAP here)
// CAP is applied as an unconditionally-stable operator-split exp(-Γ·dt)
// in FS_RK_FINAL. Embedding Γ here would require Γ·dt ≤ 2.785 (RK4 stability
// limit) but our Γ·dt can reach absStrength ≫ 1, causing blow-up.
const FS_RK_DERIV = `#version 300 es
precision highp float;
uniform sampler2D uPsi;
uniform sampler2D uMask;
uniform float uCX;   // hbar / (2m * dx^2)
uniform float uCY;   // hbar / (2m * dy^2)
out vec4 oC;
void main() {
  ivec2 c  = ivec2(gl_FragCoord.xy);
  ivec2 sz = textureSize(uPsi, 0);
  if (c.x<=0||c.x>=sz.x-1||c.y<=0||c.y>=sz.y-1||texelFetch(uMask,c,0).r<0.5){
    oC = vec4(0.0,0.0,0.0,1.0); return;
  }
  vec2 C  = texelFetch(uPsi, c,               0).rg;
  vec2 Xp = texelFetch(uPsi, c+ivec2( 1, 0), 0).rg;
  vec2 Xm = texelFetch(uPsi, c+ivec2(-1, 0), 0).rg;
  vec2 Yp = texelFetch(uPsi, c+ivec2( 0, 1), 0).rg;
  vec2 Ym = texelFetch(uPsi, c+ivec2( 0,-1), 0).rg;
  vec2 lapl = uCX*(Xp+Xm-2.0*C) + uCY*(Yp+Ym-2.0*C);
  oC = vec4(-lapl.g, lapl.r, 0.0, 1.0);
}`;

// ─── RK4: dst = base + scale * delta (staging & accumulation) ─
const FS_RK_AXPY = `#version 300 es
precision highp float;
uniform sampler2D uBase;
uniform sampler2D uDelta;
uniform float uScale;
out vec4 oC;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec2 b  = texelFetch(uBase,  c, 0).rg;
  vec2 d  = texelFetch(uDelta, c, 0).rg;
  oC = vec4(b + uScale*d, 0.0, 1.0);
}`;

// ─── RK4: final step — psi_new = (psi_n + dt6*accum) * capdecay ───
// Operator split: free SE integrated by RK4; then CAP exp(-Γ·dt) applied
// multiplicatively (unconditionally stable for any Γ·dt).
const FS_RK_FINAL = `#version 300 es
precision highp float;
uniform sampler2D uPsi;
uniform sampler2D uAccum;
uniform sampler2D uMask;
uniform sampler2D uCapDecay;
uniform float uDt6;
out vec4 oC;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  if (texelFetch(uMask, c, 0).r < 0.5) { oC = vec4(0.0,0.0,0.0,1.0); return; }
  vec2 psi   = texelFetch(uPsi,   c, 0).rg;
  vec2 acc   = texelFetch(uAccum, c, 0).rg;
  float dec  = texelFetch(uCapDecay, c, 0).r;
  oC = vec4((psi + uDt6*acc) * dec, 0.0, 1.0);
}`;

// ─── Export class ─────────────────────────────────────────────
export class GPUSim {
  constructor() {
    // Create a hidden canvas solely for physics computation
    const canvas = document.createElement('canvas');
    canvas.style.display = 'none';
    document.body.appendChild(canvas);
    this._canvas = canvas;

    const gl = canvas.getContext('webgl2', { powerPreference: 'high-performance', failIfMajorPerformanceCaveat: false });
    if (!gl) throw new Error('WebGL2 not available (hardware acceleration may be disabled in your browser)');
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) throw new Error('EXT_color_buffer_float not supported (GPU driver may not support float render targets)');
    // Log renderer info for diagnostics
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    if (dbg) {
      const vendor   = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL);
      const renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
      console.info(`[GPUSim] WebGL2 renderer: ${vendor} / ${renderer}`);
      if (/SwiftShader|llvmpipe|softpipe|Microsoft Basic/i.test(renderer))
        throw new Error(`Software renderer detected (${renderer}) — GPU driver not accelerating WebGL`);
    }
    this.gl = gl;

    this._buildQuad();
    this._progs = {
      cmul    : this._prog(VS, FS_CMUL),
      fft     : this._prog(VS, FS_FFT),
      abs     : this._prog(VS, FS_ABS),
      rho     : this._prog(VS, FS_RHO),
      mask    : this._prog(VS, FS_MASK),
      fdR     : this._prog(VS, FS_FD_R),
      fdI     : this._prog(VS, FS_FD_I),
      rkDeriv : this._prog(VS, FS_RK_DERIV),
      rkAxpy  : this._prog(VS, FS_RK_AXPY),
      rkFinal : this._prog(VS, FS_RK_FINAL),
    };

    this.simTime = 0;
    this.Nx = 0; this.Ny = 0;
    this.rho = null;    // Float32Array (column-major, matches worker convention)
    this._psi = null;   // Float32Array (row-major GL readback, length Nx*Ny*2)
    this._tex = null;
    this._fbo = null;
  }

  // ── WebGL helpers ─────────────────────────────────────────
  _shader(type, src) {
    const gl = this.gl;
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error('Shader compile error:\n' + gl.getShaderInfoLog(s) + '\n' + src.slice(0, 200));
    return s;
  }

  _prog(vs, fs) {
    const gl = this.gl;
    const p = gl.createProgram();
    gl.attachShader(p, this._shader(gl.VERTEX_SHADER,   vs));
    gl.attachShader(p, this._shader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      throw new Error('Program link error: ' + gl.getProgramInfoLog(p));
    return p;
  }

  _buildQuad() {
    const gl = this.gl;
    this._vao = gl.createVertexArray();
    gl.bindVertexArray(this._vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    this._quadBuf = buf;
    gl.bindVertexArray(null);
  }

  // Create an RGBA32F texture (Re=R, Im=G, BA ignored). Data optional.
  _mkTex(w, h, data) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // Use RGBA32F for maximum read-back compatibility
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0,
                  gl.RGBA, gl.FLOAT, data || null);
    return t;
  }

  _mkFbo(tex) {
    const gl = this.gl;
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, tex, 0);
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE)
      throw new Error('Framebuffer incomplete: ' + st);
    return fb;
  }

  // Run one fullscreen shader pass: read srcTextures, write to dstFBO
  _pass(prog, dstFBO, w, h, uniforms) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
    gl.viewport(0, 0, w, h);
    gl.useProgram(prog);

    gl.bindVertexArray(this._vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._quadBuf);
    const posLoc = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    let unit = 0;
    for (const [name, val] of Object.entries(uniforms)) {
      const loc = gl.getUniformLocation(prog, name);
      if (loc === null) continue;
      if (val instanceof WebGLTexture) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, val);
        gl.uniform1i(loc, unit++);
      } else {
        gl.uniform1f(loc, val);
      }
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  // ── Stockham 2D FFT ───────────────────────────────────────
  // Reads from srcTex, ping-pongs through ping/pong textures only.
  // Returns the WebGLTexture holding the result.
  // dir: +1.0 = forward, -1.0 = inverse
  _fft2d(srcTex, dir) {
    const { Nx, Ny } = this;
    const T = this._tex, F = this._fbo;
    const prog = this._progs.fft;

    let src = srcTex;

    // ── Horizontal passes (FFT along x, uHoriz=1, size=Nx) ──
    // Choose initial write target to avoid read/write to same texture
    let dstH = (src === T.ping) ? 1 : 0;
    for (let p = 0; p < this._Nx_log2; p++) {
      const dstTex = dstH === 0 ? T.ping : T.pong;
      const dstFBO = dstH === 0 ? F.ping : F.pong;
      this._pass(prog, dstFBO, Nx, Ny,
        { uTex: src, uPass: p, uN: Nx, uHoriz: 1, uDir: dir });
      src = dstTex;
      dstH ^= 1;
    }

    // ── Vertical passes (FFT along y, uHoriz=0, size=Ny) ────
    let dstV = (src === T.ping) ? 1 : 0;
    for (let p = 0; p < this._Ny_log2; p++) {
      const dstTex = dstV === 0 ? T.ping : T.pong;
      const dstFBO = dstV === 0 ? F.ping : F.pong;
      this._pass(prog, dstFBO, Nx, Ny,
        { uTex: src, uPass: p, uN: Ny, uHoriz: 0, uDir: dir });
      src = dstTex;
      dstV ^= 1;
    }

    return src; // ping or pong texture holding the result
  }

  // ── Accessors for current/next psi ────────────────────────
  _curTex() { return this._cur === 0 ? this._tex.psi0 : this._tex.psi1; }
  _nxtTex() { return this._cur === 0 ? this._tex.psi1 : this._tex.psi0; }
  _curFBO() { return this._cur === 0 ? this._fbo.psi0 : this._fbo.psi1; }
  _nxtFBO() { return this._cur === 0 ? this._fbo.psi1 : this._fbo.psi0; }

  // ── Initialise simulation ────────────────────────────────
  init(cfg) {
    const gl = this.gl;
    const {
      Nx = 128, Ny = 128, Lx = 200e-9, Ly = 100e-9, Dt = 0.05e-15,
      velox = 1e5, veloy = 0,
      sigmax = 6e-9, sigmay = 10e-9, xfrac = 0.20,
      slitX = 0.5, slitCenterY1 = 0.422, slitCenterY2 = 0.578,
      slitHalfWidth = 0.039,
      absThick = 0.25, absStrength = 120,
      fdMode = false,
      rkMode = false,
    } = cfg;
    this._fdMode = fdMode;
    this._rkMode = rkMode;

    // Clean up old GL resources
    if (this._tex) {
      Object.values(this._tex).forEach(t => gl.deleteTexture(t));
      Object.values(this._fbo).forEach(f => gl.deleteFramebuffer(f));
    }

    // Double the domain in X transparently: GPU runs simNx×Ny,
    // readback exports only the central Nx columns (the displayed region).
    // This gives the wave a hidden absorption margin on both sides.
    const simNx    = Nx * 2;
    const dispOffX = Nx / 2;       // first displayed column in the sim grid
    this.Nx        = simNx;        // used by GL steps / FFT / readback
    this.dispNx    = Nx;           // exported display width
    this._dispOffX = dispOffX;
    this.Ny = Ny;
    this.Lx = Lx; this.Ly = Ly; this.Dt = Dt;  // Lx/Ly are display physical sizes
    this._Nx_log2 = Math.round(Math.log2(simNx));
    this._Ny_log2 = Math.round(Math.log2(Ny));
    this._absThick    = absThick;
    this._absStrength = absStrength;
    this._canvas.width = simNx; this._canvas.height = Ny;

    const Dx = Lx / (Nx - 1), Dy = Ly / (Ny - 1);
    const simLx = Dx * (simNx - 1);   // exact same Δx, domain ≈ 2×Lx
    // Convert display-fraction → sim-fraction: sim_frac = (dispOffX + f*(Nx-1))/(simNx-1)
    const toSimFrac = f => (dispOffX + f * (Nx - 1)) / (simNx - 1);
    const simXfrac  = toSimFrac(xfrac);
    const simSlitX  = toSimFrac(slitX);
    const dkx = 2 * Math.PI / simLx;
    const dky = 2 * Math.PI / Ly;

    // ── Initial wavefunction ────────────────────────────────
    // Layout: RGBA32F tex, pixel (ix, iy) = grid point (ix, iy)
    // texImage2D upload is row-major: data[iy*Nx+ix] → pixel (ix, iy=row)
    const psi0Data = new Float32Array(simNx * Ny * 4);
    const x0 = simXfrac * simLx, y0 = Ly / 2;
    const noX = Math.pow(1 / (2 * Math.PI * sigmax * sigmax), 0.25) / Math.SQRT2;
    const noY = Math.pow(1 / (2 * Math.PI * sigmay * sigmay), 0.25) / Math.SQRT2;
    for (let ix = 0; ix < simNx; ix++) {
      const x   = ix * Dx;
      const gx  = noX * Math.exp(-((x - x0) ** 2) / (4 * sigmax ** 2));
      const phx = ME * velox * (x - x0) / HB;
      const pxR = gx * Math.cos(phx), pxI = gx * Math.sin(phx);
      for (let iy = 0; iy < Ny; iy++) {
        const y   = iy * Dy;
        const gy  = noY * Math.exp(-((y - y0) ** 2) / (4 * sigmay ** 2));
        const phy = ME * veloy * (y - y0) / HB;
        const pyR = gy * Math.cos(phy), pyI = gy * Math.sin(phy);
        const i4  = (iy * simNx + ix) * 4;        // row-major RGBA
        psi0Data[i4    ] = pxR * pyR - pxI * pyI; // Re
        psi0Data[i4 + 1] = pxR * pyI + pxI * pyR; // Im
        // B, A channels unused → 0, 1
        psi0Data[i4 + 3] = 1.0;
      }
    }
    // Normalise
    let norm2 = 0;
    for (let i = 0; i < simNx * Ny * 4; i += 4)
      norm2 += psi0Data[i] ** 2 + psi0Data[i + 1] ** 2;
    const invN = 1 / Math.sqrt(norm2 * Dx * Dy);
    for (let i = 0; i < simNx * Ny * 4; i += 4) {
      psi0Data[i    ] *= invN;
      psi0Data[i + 1] *= invN;
    }

    // ── V propagator (half-step: exp(-i V Δt / 2ℏ)) ────────
    const vpropData = new Float32Array(simNx * Ny * 4);
    const QE_local = 1.602176634e-19;
    const barrierV = 10 * QE_local;
    const barrierThick = 6;
    const biX0 = Math.floor(simSlitX * (simNx - 1));
    const j1c  = Math.floor(slitCenterY1 * (Ny - 1));
    const j2c  = Math.floor(slitCenterY2 * (Ny - 1));
    const hw   = Math.floor(slitHalfWidth * (Ny - 1));
    for (let ix = 0; ix < simNx; ix++) {
      for (let iy = 0; iy < Ny; iy++) {
        const i4 = (iy * simNx + ix) * 4;
        let v = 0;
        if (ix >= biX0 && ix < biX0 + barrierThick) {
          const inSlit1 = Math.abs(iy - j1c) <= hw;
          const inSlit2 = Math.abs(iy - j2c) <= hw;
          if (!inSlit1 && !inSlit2) v = barrierV;
        }
        const ang = v * Dt / (2 * HB);
        vpropData[i4    ] =  Math.cos(ang);
        vpropData[i4 + 1] = -Math.sin(ang);
        vpropData[i4 + 3] = 1.0;
      }
    }

    // ── Bake Complex Absorbing Potential (CAP) into vprop ──
    // Applied once per V-half-step → fires BEFORE and AFTER the kinetic
    // propagation, so waves can never sneak through in one T-step.
    // Decay per half-step = exp(-absStrength/2 * d²), where d = 1-sin(depth·π/2).
    // Total per full step = exp(-absStrength * d²) — same magnitude as before,
    // but correctly bracketing the FFT.
    const HPI = Math.PI / 2;
    for (let ix = 0; ix < simNx; ix++) {
      for (let iy = 0; iy < Ny; iy++) {
        const fx = ix / Math.max(simNx - 1, 1);
        const fy = iy / Math.max(Ny - 1, 1);
        let sx = 1.0, sy = 1.0;
        if (fx < absThick)       sx = Math.sin(fx              / absThick * HPI);
        else if (fx > 1-absThick) sx = Math.sin((1 - fx)        / absThick * HPI);
        if (fy < absThick)       sy = Math.sin(fy              / absThick * HPI);
        else if (fy > 1-absThick) sy = Math.sin((1 - fy)        / absThick * HPI);
        const dx = 1 - sx, dy = 1 - sy;
        const decay = Math.exp(-absStrength * 0.5 * (dx * dx + dy * dy));
        const i4 = (iy * simNx + ix) * 4;
        vpropData[i4    ] *= decay;
        vpropData[i4 + 1] *= decay;
      }
    }

    // bmask = all ones (no barrier absorption mask needed; domain-edge absorber handles it)
    const bmaskData = new Float32Array(simNx * Ny * 4);
    for (let i = 0; i < simNx * Ny * 4; i += 4) {
      bmaskData[i    ] = 1.0;
      bmaskData[i + 3] = 1.0;
    }

    // ── T propagator (full-step in k-space, natural FFT order) ─
    const tpropData = new Float32Array(simNx * Ny * 4);
    for (let ix = 0; ix < simNx; ix++) {
      const kx = ix < simNx / 2 ? ix * dkx : (ix - simNx) * dkx;
      for (let iy = 0; iy < Ny; iy++) {
        const ky  = iy < Ny / 2 ? iy * dky : (iy - Ny) * dky;
        const ang = HB * (kx * kx + ky * ky) / (2 * ME) * Dt;
        const i4  = (iy * simNx + ix) * 4;
        tpropData[i4    ] =  Math.cos(ang);
        tpropData[i4 + 1] = -Math.sin(ang);
        tpropData[i4 + 3] = 1.0;
      }
    }

    // ── FD: barrier mask (0 at barrier pixels, 1 elsewhere) ─
    // Also used in FD leapfrog to enforce ψ=0 inside walls.
    const fdmaskData = new Float32Array(simNx * Ny * 4);
    for (let ix = 0; ix < simNx; ix++) {
      for (let iy = 0; iy < Ny; iy++) {
        const i4 = (iy * simNx + ix) * 4;
        let isWall = 0;
        if (ix >= biX0 && ix < biX0 + barrierThick) {
          const inSlit1 = Math.abs(iy - j1c) <= hw;
          const inSlit2 = Math.abs(iy - j2c) <= hw;
          if (!inSlit1 && !inSlit2) isWall = 1;
        }
        fdmaskData[i4    ] = isWall ? 0.0 : 1.0;
        fdmaskData[i4 + 3] = 1.0;
      }
    }

    // ── FD: per-pixel CAP decay = exp(-absStrength * d²) ──────
    // d = 1 - sin(depth*π/2), ranges 0 (interior edge) → 1 (boundary)
    // Applied once per full step, provides smooth attenuation before
    // the hard Dirichlet zero at the boundary pixel.
    const capdecayData = new Float32Array(simNx * Ny * 4);
    const HPI2 = Math.PI / 2;
    for (let ix = 0; ix < simNx; ix++) {
      for (let iy = 0; iy < Ny; iy++) {
        const fx = ix / Math.max(simNx - 1, 1);
        const fy = iy / Math.max(Ny - 1, 1);
        let sx = 1.0, sy = 1.0;
        if (fx < absThick)        sx = Math.sin(fx        / absThick * HPI2);
        else if (fx > 1-absThick) sx = Math.sin((1-fx)    / absThick * HPI2);
        if (fy < absThick)        sy = Math.sin(fy        / absThick * HPI2);
        else if (fy > 1-absThick) sy = Math.sin((1-fy)    / absThick * HPI2);
        const ddx = 1 - sx, ddy = 1 - sy;
        const decay = Math.exp(-absStrength * (ddx * ddx + ddy * ddy));
        const i4 = (iy * simNx + ix) * 4;
        capdecayData[i4    ] = decay;
        capdecayData[i4 + 3] = 1.0;
      }
    }

    // ── FD coefficients (stored for use in _stepFDOnce and _stepRK4Once) ───────
    // Dx and Dy already declared above
    this._AX = HB * Dt / (2 * ME * Dx * Dx);  // ℏ·dt / (2m·dx²)  (FD leapfrog)
    this._AY = HB * Dt / (2 * ME * Dy * Dy);
    this._CX = HB / (2 * ME * Dx * Dx);        // ℏ / (2m·dx²)  (RK4 derivative, no Dt)
    this._CY = HB / (2 * ME * Dy * Dy);

    // ── Allocate textures ───────────────────────────────────
    const T = {
      psi0     : this._mkTex(simNx, Ny, psi0Data),
      psi1     : this._mkTex(simNx, Ny, null),
      ping     : this._mkTex(simNx, Ny, null),
      pong     : this._mkTex(simNx, Ny, null),
      vprop    : this._mkTex(simNx, Ny, vpropData),
      tprop    : this._mkTex(simNx, Ny, tpropData),
      bmask    : this._mkTex(simNx, Ny, bmaskData),
      fdmask   : this._mkTex(simNx, Ny, fdmaskData),
      capdecay : this._mkTex(simNx, Ny, capdecayData),
      // RK4 scratch: rkz = permanent zero texture (read-only),
      //              rkk = current ki,  rka/rkb = accumulator ping-pong.
      // caprate  = Γ/ℏ per pixel (CAP rate baked for use in DERIV shader).
      caprate  : this._mkTex(simNx, Ny, (() => {
        const d = new Float32Array(simNx * Ny * 4);
        for (let ix = 0; ix < simNx; ix++) {
          for (let iy = 0; iy < Ny; iy++) {
            const fx = ix / Math.max(simNx - 1, 1);
            const fy = iy / Math.max(Ny - 1, 1);
            let sx = 1.0, sy = 1.0;
            if (fx < absThick)        sx = Math.sin(fx     / absThick * HPI2);
            else if (fx > 1-absThick) sx = Math.sin((1-fx) / absThick * HPI2);
            if (fy < absThick)        sy = Math.sin(fy     / absThick * HPI2);
            else if (fy > 1-absThick) sy = Math.sin((1-fy) / absThick * HPI2);
            const ddx = 1 - sx, ddy = 1 - sy;
            // Γ (s⁻¹) such that exp(-Γ·Dt) = exp(-absStrength·d²) per step.
            // i.e. Γ = absStrength * d² / Dt
            const i4 = (iy * simNx + ix) * 4;
            d[i4    ] = absStrength * (ddx * ddx + ddy * ddy) / Dt;
            d[i4 + 3] = 1.0;
          }
        }
        return d;
      })()),

      rkz      : this._mkTex(simNx, Ny, new Float32Array(simNx * Ny * 4)),  // explicit zeros
      rkk      : this._mkTex(simNx, Ny, null),
      rka      : this._mkTex(simNx, Ny, null),
      rkb      : this._mkTex(simNx, Ny, null),
    };
    this._tex = T;

    // ── Allocate framebuffers ───────────────────────────────────
    this._fbo = {
      psi0 : this._mkFbo(T.psi0),
      psi1 : this._mkFbo(T.psi1),
      ping : this._mkFbo(T.ping),
      pong : this._mkFbo(T.pong),
      rkk  : this._mkFbo(T.rkk),
      rka  : this._mkFbo(T.rka),
      rkb  : this._mkFbo(T.rkb),
    };

    this._cur = 0;   // psi is in psi0
    this.simTime = 0;
    // Auto-stop: time for wave to cross the sim domain AND for any FFT wrap-around
    // reflection to travel back to the display's left edge (dispOffX columns at Dx each).
    this.stopTime = (simLx * (1 - simXfrac) + dispOffX * Dx) / velox;

    // CPU-side buffers (pixBuf = full sim; rho/psi = display region only)
    this._pixBuf = new Float32Array(simNx * Ny * 4); // RGBA readback (full sim)
    this.rho = new Float32Array(Nx * Ny);             // column-major, display only
    this._psi = null;                                 // set after first readback

    // ── Async PBO readback (double-buffered) ──────────────
    const byteSize = simNx * Ny * 4 * 4; // RGBA × float32 (full sim)
    this._pbo = [gl.createBuffer(), gl.createBuffer()];
    this._pboFence = [null, null];  // fenceSync per PBO slot
    this._pboIdx = 0;
    this._pboReady = false;
    for (const b of this._pbo) {
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, b);
      gl.bufferData(gl.PIXEL_PACK_BUFFER, byteSize, gl.STREAM_READ);
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);

    // Synchronous read for frame-0 (so rho/psi are immediately valid)
    this._readbackSync();
  }

  // ── Single split-operator step ───────────────────────────
  //
  //  Step layout (cur=A, nxt=B throughout a single call):
  //    1. V-half:   cmul(psi[A], vprop)  → psi[B];  swap A↔B
  //    2. Fwd FFT:  fft(psi[A])          → kTex       (ping/pong, A unused after)
  //    3. T-step:   cmul(kTex, tprop)    → psi[B]    (B is free after step 1 swap)
  //    4. Inv FFT:  ifft(psi[B])         → kTex2      (ping/pong)
  //    5. Abs/scale: scale+mask(kTex2)   → psi[A]    (A is free after step 2)
  //    6. V-half:   cmul(psi[A], vprop)  → psi[B];  swap A↔B
  //  End: psi is in psi[A]  (same index as start after two swaps)
  _stepOnce() {
    const { Nx, Ny, Dt } = this;
    const p = this._progs, T = this._tex;

    // Split-operator: V/2 → FFT → T → IFFT → V/2  (matches MATLAB exactly)
    // Step 1: V half-step
    this._pass(p.cmul, this._nxtFBO(), Nx, Ny,
      { uA: this._curTex(), uB: T.vprop });
    this._cur ^= 1;

    // Step 2: Forward 2D FFT
    const kTex = this._fft2d(this._curTex(), 1.0);

    // Step 3: T full-step
    this._pass(p.cmul, this._nxtFBO(), Nx, Ny,
      { uA: kTex, uB: T.tprop });

    // Step 4: Inverse 2D FFT
    const kTex2 = this._fft2d(this._nxtTex(), -1.0);

    // Step 5: Normalise IFFT result (CAP baked into vprop handles absorption)
    this._pass(p.abs, this._curFBO(), Nx, Ny, {
      uTex  : kTex2,
      uScale: 1.0 / (Nx * Ny),
    });

    // Step 6: V half-step
    this._pass(p.cmul, this._nxtFBO(), Nx, Ny,
      { uA: this._curTex(), uB: T.vprop });
    this._cur ^= 1;

    this.simTime += Dt;
  }

  // ── FD leapfrog single step (2 passes, true Dirichlet BC) ──
  // Schrödinger: ∂R/∂t = -(ℏ/2m)∇²I,  ∂I/∂t = +(ℏ/2m)∇²R
  // No FFT → no periodic wrap-around → wave is fully absorbed at walls.
  _stepFDOnce() {
    const { Nx, Ny, Dt } = this;
    const p = this._progs, T = this._tex, F = this._fbo;
    // Pass 1: R_half = R + AX·∇²I
    this._pass(p.fdR, F.ping, Nx, Ny, {
      uPsi : this._curTex(), uMask: T.fdmask,
      uAX  : this._AX, uAY: this._AY,
    });
    // Pass 2: I_new = I + AX·∇²R_half; apply CAP decay to both channels
    this._pass(p.fdI, this._nxtFBO(), Nx, Ny, {
      uHalf    : T.ping, uCapDecay: T.capdecay,
      uMask    : T.fdmask,
      uAX      : this._AX, uAY: this._AY,
    });
    this._cur ^= 1;
    this.simTime += Dt;
  }

  // ── RK4 single step (11 GPU passes, 4th-order, Dirichlet BC) ──
  // \u03c8_{n+1} = \u03c8_n + (Dt/6)(k1 + 2k2 + 2k3 + k4)  then apply CAP decay.
  // Staging: \u03c8_tmp = \u03c8_n + scale\u00b7ki,  accumulated in rka/rkb ping-pong.
  // Barrier: Dirichlet \u03c8=0 enforced by fdmask (same as FD leapfrog).
  _stepRK4Once() {
    const { Nx, Ny, Dt } = this;
    const p = this._progs, T = this._tex, F = this._fbo;
    const cX = this._CX, cY = this._CY;
    const psi_n = this._curTex();
    const rkd = { uMask: T.fdmask, uCX: cX, uCY: cY };

    // k1 = f(psi_n)
    this._pass(p.rkDeriv, F.rkk,  Nx, Ny, { uPsi: psi_n,  ...rkd });
    this._pass(p.rkAxpy,  F.rka,  Nx, Ny, { uBase: T.rkz,  uDelta: T.rkk, uScale: 1.0 });
    this._pass(p.rkAxpy,  F.ping, Nx, Ny, { uBase: psi_n,  uDelta: T.rkk, uScale: Dt * 0.5 });

    // k2 = f(psi_n + dt/2 * k1)
    this._pass(p.rkDeriv, F.rkk,  Nx, Ny, { uPsi: T.ping,  ...rkd });
    this._pass(p.rkAxpy,  F.rkb,  Nx, Ny, { uBase: T.rka,  uDelta: T.rkk, uScale: 2.0 });
    this._pass(p.rkAxpy,  F.ping, Nx, Ny, { uBase: psi_n,  uDelta: T.rkk, uScale: Dt * 0.5 });

    // k3 = f(psi_n + dt/2 * k2)
    this._pass(p.rkDeriv, F.rkk,  Nx, Ny, { uPsi: T.ping,  ...rkd });
    this._pass(p.rkAxpy,  F.rka,  Nx, Ny, { uBase: T.rkb,  uDelta: T.rkk, uScale: 2.0 });
    this._pass(p.rkAxpy,  F.ping, Nx, Ny, { uBase: psi_n,  uDelta: T.rkk, uScale: Dt });

    // k4 = f(psi_n + dt * k3)
    this._pass(p.rkDeriv, F.rkk,  Nx, Ny, { uPsi: T.ping,  ...rkd });
    this._pass(p.rkAxpy,  F.rkb,  Nx, Ny, { uBase: T.rka,  uDelta: T.rkk, uScale: 1.0 });

    // psi_{n+1} = psi_n + (Dt/6)*(k1 + 2k2 + 2k3 + k4)
    // No separate decay pass: CAP is already embedded in each ki above.
    this._pass(p.rkFinal, this._nxtFBO(), Nx, Ny, {
      uPsi: psi_n, uAccum: T.rkb, uMask: T.fdmask, uCapDecay: T.capdecay, uDt6: Dt / 6.0,
    });
    this._cur ^= 1;
    this.simTime += Dt;
  }

  // ── Public: advance n steps then read back ───────────────
  step(n) {
    const stepFn = this._rkMode ? () => this._stepRK4Once()
                 : this._fdMode ? () => this._stepFDOnce()
                                : () => this._stepOnce();
    for (let i = 0; i < n; i++) stepFn();
    this._readback();
  }

  // ── Sync readback (init only) ────────────────────────────
  _readbackSync() {
    const gl = this.gl;
    const { Nx, Ny } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._curFBO());
    gl.readPixels(0, 0, Nx, Ny, gl.RGBA, gl.FLOAT, this._pixBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this._processBuf();
  }

  // ── Async double-buffered PBO readback (every step) ──────
  // Kick off a non-blocking DMA copy into curPBO, then retrieve
  // the *previous* frame's data from prevPBO — only after its fence signals.
  _readback() {
    const gl = this.gl;
    const { Nx, Ny } = this;

    const curSlot  =  this._pboIdx;
    const prevSlot =  this._pboIdx ^ 1;
    const curPBO   = this._pbo[curSlot];
    const prevPBO  = this._pbo[prevSlot];

    // Retrieve previous frame's pixels — only once the GPU fence signals
    if (this._pboReady) {
      const fence = this._pboFence[prevSlot];
      let ready = !fence; // no fence means synchronous init path already consumed it
      if (fence) {
        const status = gl.clientWaitSync(fence, 0, 0); // non-blocking poll
        if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) {
          gl.deleteSync(fence);
          this._pboFence[prevSlot] = null;
          ready = true;
        }
      }
      if (ready) {
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, prevPBO);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this._pixBuf);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        this._processBuf();
      }
    }

    // Enqueue non-blocking copy of current frame into curPBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._curFBO());
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, curPBO);
    gl.readPixels(0, 0, Nx, Ny, gl.RGBA, gl.FLOAT, 0); // 0 = offset into PBO
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Place a fence so we only read back once the DMA transfer is complete
    if (this._pboFence[curSlot]) gl.deleteSync(this._pboFence[curSlot]);
    this._pboFence[curSlot] = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush(); // ensure the fence is submitted to the GPU command queue

    this._pboIdx  ^= 1;
    this._pboReady = true;
  }

  // ── Unpack pixBuf → rho + psi ─────────────────────────────
  _processBuf() {
    // pixBuf spans the full simNx×Ny grid. Extract only the displayed columns.
    const simNx    = this.Nx;         // internal sim width
    const dispNx   = this.dispNx;     // displayed columns
    const dispOffX = this._dispOffX;  // first displayed column
    const Ny       = this.Ny;
    const buf = this._pixBuf;
    const rho = this.rho;
    // rho: column-major [dix*Ny+iy] where dix = display ix ∈ [0, dispNx)
    for (let dix = 0; dix < dispNx; dix++) {
      const ix = dix + dispOffX;
      for (let iy = 0; iy < Ny; iy++) {
        const glIdx = (iy * simNx + ix) * 4;
        const re = buf[glIdx], im = buf[glIdx + 1];
        rho[dix * Ny + iy] = re * re + im * im;
      }
    }
    // psi: row-major [(iy*dispNx+dix)*2] for Bohmian gradient
    if (!this._psi || this._psi.length !== dispNx * Ny * 2) {
      this._psi = new Float32Array(dispNx * Ny * 2);
    }
    for (let dix = 0; dix < dispNx; dix++) {
      const ix = dix + dispOffX;
      for (let iy = 0; iy < Ny; iy++) {
        const glIdx  = (iy * simNx + ix) * 4;
        const psiIdx = (iy * dispNx + dix) * 2;
        this._psi[psiIdx    ] = buf[glIdx    ];
        this._psi[psiIdx + 1] = buf[glIdx + 1];
      }
    }
    this.psi = this._psi;
  }

  // Compute norm integral (used for HUD display)
  get norm() {
    if (!this.rho) return 0;
    let s = 0;
    const dNx = this.dispNx;
    for (let i = 0; i < dNx * this.Ny; i++) s += this.rho[i];
    return s * (this.Lx / (dNx - 1)) * (this.Ly / (this.Ny - 1));
  }
}
