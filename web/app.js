/* fruit-fly-BFM replay viewer — vanilla JS, no dependencies.
 *
 * The view is a perspective camera on a z-up world.  Aircraft are drawn as
 * fixed-size icons (a real fighter is 9 m long in a 12 km arena — anything
 * drawn to scale is invisible), with a vertical line down to the ground so
 * altitude is readable.  Everything else is HUD.
 *
 * Extra panels:
 *  - brainView: population rates per group (and per-type when full CNS)
 *  - flyView: stick & throttle cartoon driven by DN readout
 *
 * Live training extension (2026-09):
 *  - polls /api/train/status, /api/train/log, /api/train/live
 *  - training runs in terminal (server subprocess), survives window close
 *  - while training, viewer shows live match of a random fly in training
 */

"use strict";

const COL = { blue: "#4da3ff", red: "#ff5d5d", amber: "#ffc857", green: "#5ddc8a", dim: "#7b8a9c", cyan: "#5de0ff", magenta: "#ff7ac0" };
const ARENA_R = 12000.0;

const state = {
  manifest: null, replay: null, frames: [], t: 0, dur: 1, playing: true, speed: 1,
  cam: { yaw: -0.9, pitch: 0.42, dist: 2600, mode: "chase-blue", target: [0, 0, 4000] },
  lastTS: 0, drag: null, strip: null,
  liveGen: -1,
  training: { running: false, pid: null, status: null },
};

// ---------------------------------------------------------------- utilities
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, u) => a + (b - a) * u;
const fmt = (v, n = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(n);

function norm3(v) { const n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n]; }
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function mul3(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

// ---------------------------------------------------------------- the view
function makeCamera() {
  const c = state.cam;
  const dir = [Math.cos(c.pitch) * Math.cos(c.yaw), Math.cos(c.pitch) * Math.sin(c.yaw), Math.sin(c.pitch)];
  const eye = sub3(c.target, mul3(dir, c.dist));
  const fwd = norm3(sub3(c.target, eye));
  const right = norm3(cross3([0, 0, 1], fwd));
  const up = norm3(cross3(fwd, right));
  return { eye, fwd, right, up, focal: 0.9 * canvasH() };
}

function canvasH() { return $("view").clientHeight || 600; }

function project(p, cam) {
  const d = sub3(p, cam.eye);
  const depth = dot3(d, cam.fwd);
  if (depth < 1.0) return null;
  const x = dot3(d, cam.right), y = dot3(d, cam.up);
  const k = cam.focal / depth;
  return [canvasW() / 2 + x * k, canvasH() / 2 - y * k, depth];
}
function canvasW() { return $("view").clientWidth || 900; }

function drawLine(ctx, a, b, cam, style, width) {
  const pa = project(a, cam), pb = project(b, cam);
  if (!pa || !pb) return;
  ctx.strokeStyle = style; ctx.lineWidth = width || 1;
  ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
}

function fillQuad(ctx, cam, a, b, c, d, style) {
  const pts = [a, b, c, d].map((p) => project(p, cam));
  if (pts.some((p) => !p)) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.fillStyle = style;
  ctx.fill();
}

// ------------------------------------------------------------- interpolation
function sampleAt(t) {
  const f = state.frames;
  if (!f.length) return null;
  let i = 0;
  while (i < f.length - 1 && f[i + 1].t <= t) i++;
  const a = f[i], b = f[Math.min(i + 1, f.length - 1)];
  const span = (b.t - a.t) || 1;
  const u = clamp((t - a.t) / span, 0, 1);
  const mix = (key) => {
    const A = a[key], B = b[key];
    if (!A || !B) return A || B;
    return {
      p: [lerp(A.p[0], B.p[0], u), lerp(A.p[1], B.p[1], u), lerp(A.p[2], B.p[2], u)],
      psi: A.psi, gam: A.gam, v: lerp(A.v, B.v, u), mu: lerp(A.mu, B.mu, u),
      n: lerp(A.n, B.n, u), thr: lerp(A.thr, B.thr, u), ammo: A.ammo,
      es: lerp(A.es || 0, B.es || 0, u), cmd: A.cmd, hp: A.hp,
    };
  };
  // brain data is not interpolated, just taken from nearest frame
  const brain = a.brain || b.brain || null;
  const brain_red = a.brain_red || b.brain_red || null;
  return { t, b: mix("b"), r: mix("r"), g: a.g, hp: a.hp, raw: a, brain, brain_red };
}

// ------------------------------------------------------------------ low-poly world
let MOUNTAINS = null;
function initMountains() {
  if (MOUNTAINS) return MOUNTAINS;
  const rng = (seed) => {
    let x = seed;
    return () => { x = (x * 1664525 + 1013904223) % 0x100000000; return x / 0x100000000; };
  };
  const rand = rng(12345);
  const mts = [];
  const N = 32; // more outer mountains for confusion reduction
  for (let i = 0; i < N; i++) {
    const ang = (i / N) * Math.PI * 2 + (rand() - 0.5) * 0.35;
    const r = ARENA_R * (0.80 + rand() * 0.28);
    const cx = Math.cos(ang) * r;
    const cy = Math.sin(ang) * r;
    const h = 800 + rand() * 1800 + (i % 4 === 0 ? 900 : 0);
    const baseR = 800 + rand() * 1000;
    const perpAng = ang + Math.PI / 2;
    const bx1 = cx + Math.cos(perpAng) * baseR * 0.75;
    const by1 = cy + Math.sin(perpAng) * baseR * 0.75;
    const bx2 = cx + Math.cos(perpAng + Math.PI) * baseR * 0.75;
    const by2 = cy + Math.sin(perpAng + Math.PI) * baseR * 0.75;
    const shade = 0.55 + 0.45 * Math.cos(ang - 0.8);
    mts.push({ cx, cy, h, bx1, by1, bx2, by2, shade, ang, snow: h > 1800 });
  }
  for (let i = 0; i < 14; i++) {
    const ang = rand() * Math.PI * 2;
    const r = ARENA_R * (0.42 + rand() * 0.28);
    const cx = Math.cos(ang) * r;
    const cy = Math.sin(ang) * r;
    const h = 180 + rand() * 420;
    const baseR = 450 + rand() * 600;
    const perpAng = ang + Math.PI / 2;
    const bx1 = cx + Math.cos(perpAng) * baseR * 0.65;
    const by1 = cy + Math.sin(perpAng) * baseR * 0.65;
    const bx2 = cx + Math.cos(perpAng + Math.PI) * baseR * 0.65;
    const by2 = cy + Math.sin(perpAng + Math.PI) * baseR * 0.65;
    const shade = 0.45 + 0.35 * Math.cos(ang - 0.8);
    mts.push({ cx, cy, h, bx1, by1, bx2, by2, shade, ang, inner: true });
  }
  MOUNTAINS = mts;
  return mts;
}

let STARS = null;
function initStars() {
  if (STARS) return STARS;
  const s = [];
  for (let i=0;i<180;i++) {
    s.push({ x: Math.random(), y: Math.random()*0.65, r: Math.random()*1.2+0.3, a: 0.15+Math.random()*0.65 });
  }
  STARS = s;
  return s;
}

function drawSkybox(ctx) {
  const W = canvasW(), H = canvasH();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0.0, "#04070e");
  grad.addColorStop(0.12, "#080e1c");
  grad.addColorStop(0.28, "#0f1e38");
  grad.addColorStop(0.48, "#1a2f52");
  grad.addColorStop(0.66, "#253d5f");
  grad.addColorStop(0.76, "#324a6a");
  grad.addColorStop(0.84, "#4a5e78");
  grad.addColorStop(0.92, "#5a6d82");
  grad.addColorStop(1.0, "#2a3a4a");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);
  // stars
  const stars = initStars();
  for (const st of stars) {
    ctx.fillStyle = `rgba(180, 200, 255, ${st.a})`;
    ctx.beginPath();
    ctx.arc(st.x*W, st.y*H, st.r, 0, Math.PI*2);
    ctx.fill();
  }
  // moon / sun glow
  ctx.fillStyle = "rgba(255, 220, 160, 0.12)";
  ctx.beginPath();
  ctx.arc(W * 0.74, H * 0.20, 88, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(255, 230, 180, 0.06)";
  ctx.beginPath();
  ctx.arc(W * 0.74, H * 0.20, 160, 0, Math.PI * 2);
  ctx.fill();
  // horizon glow
  const hg = ctx.createLinearGradient(0, H*0.72, 0, H);
  hg.addColorStop(0, "rgba(255, 180, 120, 0)");
  hg.addColorStop(0.5, "rgba(255, 150, 80, 0.04)");
  hg.addColorStop(1, "rgba(255, 120, 60, 0.08)");
  ctx.fillStyle = hg;
  ctx.fillRect(0, H*0.72, W, H*0.28);
}

function drawMountains(ctx, cam) {
  const mts = initMountains();
  const withDepth = [];
  for (const m of mts) {
    const dx = m.cx - cam.eye[0];
    const dy = m.cy - cam.eye[1];
    const dz = (m.h * 0.5) - cam.eye[2];
    const depth = dx * cam.fwd[0] + dy * cam.fwd[1] + dz * cam.fwd[2];
    if (depth < 80) continue;
    withDepth.push({ m, depth });
  }
  withDepth.sort((a, b) => b.depth - a.depth);
  for (const { m } of withDepth) {
    const pApex = project([m.cx, m.cy, m.h], cam);
    const pB1 = project([m.bx1, m.by1, 0], cam);
    const pB2 = project([m.bx2, m.by2, 0], cam);
    const pBase = project([m.cx, m.cy, 0], cam);
    if (!pApex || !pB1 || !pB2) continue;
    const dist = Math.hypot(m.cx, m.cy);
    const fog = clamp((dist - 5000) / 9000, 0, 0.6);
    const baseAlpha = m.inner ? 0.38 : 0.62;
    const alpha = (1 - fog) * baseAlpha;
    // left face - brighter
    ctx.beginPath();
    ctx.moveTo(pB1[0], pB1[1]);
    ctx.lineTo(pApex[0], pApex[1]);
    ctx.lineTo(pBase ? pBase[0] : (pB1[0] + pB2[0]) / 2, pBase ? pBase[1] : (pB1[1] + pB2[1]) / 2);
    ctx.closePath();
    const lShade = m.shade;
    ctx.fillStyle = `rgba(${Math.round(75 + 65 * lShade)}, ${Math.round(85 + 55 * lShade)}, ${Math.round(105 + 45 * lShade)}, ${alpha})`;
    ctx.fill();
    ctx.strokeStyle = `rgba(90, 110, 135, ${alpha*0.3})`;
    ctx.lineWidth = 0.6;
    ctx.stroke();
    // right face - darker
    ctx.beginPath();
    ctx.moveTo(pB2[0], pB2[1]);
    ctx.lineTo(pApex[0], pApex[1]);
    ctx.lineTo(pBase ? pBase[0] : (pB1[0] + pB2[0]) / 2, pBase ? pBase[1] : (pB1[1] + pB2[1]) / 2);
    ctx.closePath();
    const rShade = m.shade * 0.72;
    ctx.fillStyle = `rgba(${Math.round(50 + 50 * rShade)}, ${Math.round(62 + 45 * rShade)}, ${Math.round(80 + 40 * rShade)}, ${alpha * 0.88})`;
    ctx.fill();
    // base shadow
    if (pB1 && pB2 && pBase) {
      ctx.fillStyle = `rgba(8, 12, 18, ${alpha * 0.32})`;
      ctx.beginPath();
      ctx.moveTo(pB1[0], pB1[1]);
      ctx.lineTo(pB2[0], pB2[1]);
      ctx.lineTo(pBase[0], pBase[1]);
      ctx.closePath();
      ctx.fill();
    }
    // snow cap for tall peaks
    if (m.snow && pApex) {
      const snowH = m.h * 0.18;
      const pSnowBase = project([m.cx, m.cy, m.h - snowH], cam);
      if (pSnowBase) {
        ctx.fillStyle = `rgba(220, 230, 245, ${alpha*0.85})`;
        ctx.beginPath();
        ctx.moveTo(pApex[0], pApex[1]);
        ctx.lineTo((pB1[0]+pApex[0])/2, (pB1[1]+pApex[1])/2 + 4);
        ctx.lineTo(pSnowBase[0], pSnowBase[1]);
        ctx.lineTo((pB2[0]+pApex[0])/2, (pB2[1]+pApex[1])/2 + 4);
        ctx.closePath();
        ctx.fill();
      }
    }
  }
}

// ------------------------------------------------------------------ drawing
function drawWorld(ctx, cam, s) {
  const W = canvasW(), H = canvasH();
  drawSkybox(ctx);
  drawMountains(ctx, cam);

  const cx = Math.round(cam.eye[0] + cam.fwd[0] * cam.dist / 1000), cy = Math.round(cam.eye[1] + cam.fwd[1] * cam.dist / 1000);
  const step = 2000, span = 18;
  const gx = Math.round(cx / step) * step, gy = Math.round(cy / step) * step;
  const half = span * step;

  // deep under layer - darkest
  fillQuad(ctx, cam,
    [gx - half * 2.2, gy - half * 2.2, -4], [gx + half * 2.2, gy - half * 2.2, -4],
    [gx + half * 2.2, gy + half * 2.2, -4], [gx - half * 2.2, gy + half * 2.2, -4],
    "rgba(8, 12, 20, 0.96)");

  // main ground - slightly brighter for visibility
  fillQuad(ctx, cam,
    [gx - half * 1.8, gy - half * 1.8, -1], [gx + half * 1.8, gy - half * 1.8, -1],
    [gx + half * 1.8, gy + half * 1.8, -1], [gx - half * 1.8, gy + half * 1.8, -1],
    "rgba(16, 24, 36, 0.94)");

  fillQuad(ctx, cam,
    [gx - half, gy - half, 0], [gx + half, gy - half, 0],
    [gx + half, gy + half, 0], [gx - half, gy + half, 0],
    "rgba(28, 40, 58, .88)");

  // secondary layer with slight blue tint
  fillQuad(ctx, cam,
    [gx - ARENA_R * 0.7, gy - ARENA_R * 0.7, 1], [gx + ARENA_R * 0.7, gy - ARENA_R * 0.7, 1],
    [gx + ARENA_R * 0.7, gy + ARENA_R * 0.7, 1], [gx - ARENA_R * 0.7, gy + ARENA_R * 0.7, 1],
    "rgba(32, 48, 70, 0.42)");

  // grid - more visible
  for (let i = -span; i <= span; i++) {
    const x = gx + i * step, y = gy + i * step;
    drawLine(ctx, [x, gy - half, 0], [x, gy + half, 0], cam, "rgba(70, 95, 130, .38)", 1);
    drawLine(ctx, [gx - half, y, 0], [gx + half, y, 0], cam, "rgba(70, 95, 130, .38)", 1);
  }
  for (let i = -span; i <= span; i += 5) {
    drawLine(ctx, [gx + i * step, gy - half, 0], [gx + i * step, gy + half, 0], cam, "rgba(110, 140, 180, .52)", 1.3);
    drawLine(ctx, [gx - half, gy + i * step, 0], [gx + half, gy + i * step, 0], cam, "rgba(110, 140, 180, .52)", 1.3);
  }
  // center cross for orientation
  drawLine(ctx, [gx, gy - 600, 0], [gx, gy + 600, 0], cam, "rgba(77,163,255,.55)", 1.5);
  drawLine(ctx, [gx - 600, gy, 0], [gx + 600, gy, 0], cam, "rgba(77,163,255,.55)", 1.5);

  const gl = project([gx, gy, 0], cam);
  if (gl) {
    ctx.fillStyle = "rgba(150,172,200,.55)";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("GROUND  z = 0 m", gl[0] + 6, gl[1] + 12);
  }
  ctx.beginPath();
  let started = false;
  for (let a = 0; a <= 360; a += 2) {
    const rad = a * Math.PI / 180;
    const p = project([ARENA_R * Math.cos(rad), ARENA_R * Math.sin(rad), 0], cam);
    if (!p) { started = false; continue; }
    if (!started) { ctx.moveTo(p[0], p[1]); started = true; } else ctx.lineTo(p[0], p[1]);
  }
  ctx.strokeStyle = "rgba(120,150,190,.28)"; ctx.lineWidth = 1.5; ctx.setLineDash([6, 8]); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); started = false;
  for (let a = 0; a <= 360; a += 2) {
    const rad = a * Math.PI / 180;
    const p = project([ARENA_R * Math.cos(rad), ARENA_R * Math.sin(rad), 2], cam);
    if (!p) { started = false; continue; }
    if (!started) { ctx.moveTo(p[0], p[1]); started = true; } else ctx.lineTo(p[0], p[1]);
  }
  ctx.strokeStyle = "rgba(77,163,255,.08)"; ctx.lineWidth = 3; ctx.stroke();

  drawTrail(ctx, cam, "b", COL.blue);
  drawTrail(ctx, cam, "r", COL.red);
}

function drawTrail(ctx, cam, key, color) {
  const f = state.frames;
  ctx.strokeStyle = color; ctx.globalAlpha = 0.30; ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  for (const fr of f) {
    if (fr.t > state.t) break;
    const p = project(fr[key].p, cam);
    if (!p) { started = false; continue; }
    if (!started) { ctx.moveTo(p[0], p[1]); started = true; } else ctx.lineTo(p[0], p[1]);
  }
  ctx.stroke(); ctx.globalAlpha = 1;
}

function drawAircraft(ctx, cam, ac, color, label, s) {
  const p = project(ac.p, cam);
  if (!p) return;
  const [x, y] = p;
  const ahead = add3(ac.p, mul3([Math.cos(ac.psi * Math.PI / 180), Math.sin(ac.psi * Math.PI / 180), 0], 500));
  const q = project(ahead, cam);
  const ang = q ? Math.atan2(q[1] - y, q[0] - x) : 0;
  const size = 18;
  const gp = project([ac.p[0], ac.p[1], 0], cam);
  if (gp) {
    ctx.fillStyle = "rgba(0,0,0,0.28)";
    ctx.beginPath();
    ctx.ellipse(gp[0], gp[1], size * 1.0, size * 0.5, ang, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(ang);
  // F-16 Fighting Falcon - more accurate low-poly
  // fuselage
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.96;
  ctx.beginPath();
  ctx.moveTo(size * 1.65, 0); // nose
  ctx.lineTo(size * 1.15, size * 0.10);
  ctx.lineTo(size * 0.25, size * 0.13);
  ctx.lineTo(-size * 0.65, size * 0.16);
  ctx.lineTo(-size * 1.15, size * 0.09);
  ctx.lineTo(-size * 1.15, -size * 0.09);
  ctx.lineTo(-size * 0.65, -size * 0.16);
  ctx.lineTo(size * 0.25, -size * 0.13);
  ctx.lineTo(size * 1.15, -size * 0.10);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = 0.9; ctx.stroke();
  // bubble canopy - F-16 distinctive
  ctx.fillStyle = "rgba(180, 210, 255, 0.72)";
  ctx.beginPath();
  ctx.ellipse(size * 0.52, 0, size * 0.36, size * 0.13, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(220, 240, 255, 0.35)";
  ctx.beginPath();
  ctx.ellipse(size * 0.58, -0.02*size, size * 0.18, size * 0.06, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.5)"; ctx.lineWidth = 0.7; ctx.stroke();
  // side intakes - F-16 single intake under, but we show side intakes for visibility
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.beginPath();
  ctx.ellipse(size * 0.15, size * 0.14, size * 0.18, size * 0.05, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(size * 0.15, -size * 0.14, size * 0.18, size * 0.05, 0, 0, Math.PI * 2);
  ctx.fill();
  // wings - clipped delta, F-16 style
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.94;
  ctx.beginPath();
  ctx.moveTo(size * 0.28, size * 0.13);
  ctx.lineTo(-size * 0.35, size * 1.28);
  ctx.lineTo(-size * 0.62, size * 1.18);
  ctx.lineTo(size * 0.02, size * 0.18);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineWidth = 0.7; ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(size * 0.28, -size * 0.13);
  ctx.lineTo(-size * 0.35, -size * 1.28);
  ctx.lineTo(-size * 0.62, -size * 1.18);
  ctx.lineTo(size * 0.02, -size * 0.18);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  // leading edge stripes
  ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.lineWidth = 0.8;
  ctx.beginPath(); ctx.moveTo(size*0.28, size*0.13); ctx.lineTo(-size*0.35, size*1.28); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(size*0.28, -size*0.13); ctx.lineTo(-size*0.35, -size*1.28); ctx.stroke();
  // horizontal stabs
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(-size * 0.68, size * 0.10);
  ctx.lineTo(-size * 0.92, size * 0.52);
  ctx.lineTo(-size * 1.06, size * 0.44);
  ctx.lineTo(-size * 0.78, size * 0.06);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineWidth=0.6; ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(-size * 0.68, -size * 0.10);
  ctx.lineTo(-size * 0.92, -size * 0.52);
  ctx.lineTo(-size * 1.06, -size * 0.44);
  ctx.lineTo(-size * 0.78, -size * 0.06);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  // vertical tail - single, tall
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(-size * 0.75, 0);
  ctx.lineTo(-size * 0.95, size * 0.08);
  ctx.lineTo(-size * 1.18, size * 0.68);
  ctx.lineTo(-size * 1.28, size * 0.60);
  ctx.lineTo(-size * 0.85, -0.02);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  // exhaust
  ctx.fillStyle = "rgba(0,0,0,0.32)";
  ctx.fillRect(-size * 1.18, -size * 0.05, size * 0.38, size * 0.10);
  const thr = ac.thr || 0;
  if (thr > 0.55) {
    ctx.globalAlpha = 0.6 + thr * 0.35;
    const glow = ctx.createRadialGradient(-size * 1.22, 0, 0, -size * 1.22, 0, size * 0.42);
    glow.addColorStop(0, `rgba(255, ${190 + thr * 45}, 90, 0.95)`);
    glow.addColorStop(0.25, `rgba(255, 160, 50, 0.6)`);
    glow.addColorStop(0.55, `rgba(255, 110, 30, 0.22)`);
    glow.addColorStop(1, "rgba(255, 80, 20, 0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(-size * 1.22, 0, size * 0.42, 0, Math.PI * 2);
    ctx.fill();
    // inner blue core for afterburner
    if (thr > 0.85) {
      ctx.fillStyle = `rgba(120, 200, 255, ${0.5 + (thr-0.85)*2})`;
      ctx.beginPath();
      ctx.arc(-size*1.22, 0, size*0.18, 0, Math.PI*2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  ctx.restore();
  const bank = ac.mu * Math.PI / 180;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(ang + bank * 0.6);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(-size * 1.1, 0);
  ctx.lineTo(size * 1.1, 0);
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(0, 0, size * 0.65, -Math.PI * 0.7, Math.PI * 0.7);
  ctx.stroke();
  ctx.restore();
  if (gp) {
    ctx.strokeStyle = "rgba(120,150,190,.28)"; ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(gp[0], gp[1]); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(150,172,200,.9)";
    ctx.beginPath();
    ctx.moveTo(gp[0], gp[1]);
    ctx.lineTo(gp[0] - 4, gp[1] - 5);
    ctx.lineTo(gp[0] + 4, gp[1] - 5);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = COL.dim; ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(`${label} ${(ac.p[2] / 1000).toFixed(1)}km AGL`, gp[0] + 6, gp[1] + 11);
  }
  if (s && s.hp) {
    const hp = ac === s.b ? s.hp[0] : s.hp[1];
    if (hp < 0.5) {
      ctx.fillStyle = `rgba(255,80,80,${0.15 + (1 - hp) * 0.25})`;
      ctx.beginPath();
      ctx.arc(x, y, size * (1.2 + (1 - hp)), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawEngagement(ctx, cam, s) {
  for (const [ac, color] of [[s.b, COL.blue], [s.r, COL.red]]) {
    const nose = add3(ac.p, mul3([Math.cos(ac.psi * Math.PI / 180) * Math.cos(ac.gam * Math.PI / 180),
      Math.sin(ac.psi * Math.PI / 180) * Math.cos(ac.gam * Math.PI / 180), Math.sin(ac.gam * Math.PI / 180)], 900));
    drawLine(ctx, ac.p, nose, cam, color + "66", 1.5);
  }
  if (s.b.cmd && s.b.cmd[3]) {
    const dir = [Math.cos(s.b.psi * Math.PI / 180) * Math.cos(s.b.gam * Math.PI / 180),
      Math.sin(s.b.psi * Math.PI / 180) * Math.cos(s.b.gam * Math.PI / 180), Math.sin(s.b.gam * Math.PI / 180)];
    const len = 300 + 900 * ((state.t * 3) % 1);
    drawLine(ctx, s.b.p, add3(s.b.p, mul3(dir, len)), cam, COL.amber, 2);
  }
  drawLine(ctx, s.b.p, s.r.p, cam, "rgba(160,180,210,.35)", 1);
  const inW = s.g && s.g.wez;
  ctx.strokeStyle = inW ? COL.green : "rgba(120,150,190,.25)";
  const p = project(s.r.p, cam);
  if (p) {
    ctx.lineWidth = inW ? 2 : 1;
    ctx.beginPath(); ctx.arc(p[0], p[1], inW ? 26 : 18, 0, Math.PI * 2); ctx.stroke();
  }
}

// ------------------------------------------------------------------ the HUD
function updateHUD(s) {
  const g = s.g || {};
  const rp = state.replay || {};
  const bName = rp.blue_name || "fly";
  const rName = rp.red_name || "opponent";
  const wez = g.wez ? `<span class="ok">GUN WEZ</span>` : `<span class="warn">—</span>`;
  $("hud").innerHTML = `
    <div><span class="b">BLUE · FLY</span> <span class="dim">${bName}</span> v${fmt(s.b.v, 0)}m/s n${fmt(s.b.n, 1)} ${fmt(s.b.mu, 0)}°bank
      thr ${fmt(s.b.thr * 100, 0)}% ammo ${s.b.ammo}</div>
    <div><span class="r">RED · OPP</span> <span class="dim">${rName}</span> v${fmt(s.r.v, 0)}m/s n${fmt(s.r.n, 1)} ${fmt(s.r.mu, 0)}°bank
      thr ${fmt(s.r.thr * 100, 0)}% ammo ${s.r.ammo}</div>
    <div>RNG ${fmt(g.rng, 0)}m  AA ${fmt(g.aa, 1)}°  TAA ${fmt(g.taa, 0)}°  HCA ${fmt(g.hca, 0)}°
      CLOS ${fmt(g.clo, 0)}m/s  ${wez}</div>`;
}

function cmdBar(label, value, lo, hi, neg) {
  const span = hi - lo;
  const frac = clamp((value - lo) / span, 0, 1);
  const zero = clamp((0 - lo) / span, 0, 1);
  const left = Math.min(frac, zero) * 100, width = Math.abs(frac - zero) * 100;
  return `<div class="cmdrow"><span>${label}</span>
    <div class="bar"><span class="${neg ? "neg" : ""}" style="left:${left}%; width:${width}%"></span>
    <i class="mid"></i></div><span style="text-align:right">${fmt(value, 2)}</span></div>`;
}

function updatePanel(s) {
  const c = s.b.cmd || [0, 0, 0, false];
  $("commands").innerHTML =
    cmdBar("roll", c[0], -1, 1) + cmdBar("pull", c[1], -1, 1) +
    cmdBar("throttle", c[2], 0, 1) +
    `<div class="cmdrow"><span>trigger</span><div class="bar"><span style="left:0;width:${c[3] ? 100 : 0}%\\"></span></div>
      <span style="text-align:right">${c[3] ? "FIRE" : "safe"}</span></div>`;

  const g = s.g || {};
  const rows = [
    ["range", fmt(g.rng, 0) + " m"],
    ["tracking error (AA)", fmt(g.aa, 1) + "°"],
    ["target aspect", fmt(g.taa, 0) + "°"],
    ["heading crossing", fmt(g.hca, 0) + "°"],
    ["closure", fmt(g.clo, 0) + " m/s"],
    ["required lead", fmt(g.lead, 1) + "°"],
    ["blue specific energy", fmt(s.b.es, 0) + " m"],
    ["red specific energy", fmt(s.r.es, 0) + " m"],
    ["health", `${fmt(s.hp[0] * 100, 0)}% / ${fmt(s.hp[1] * 100, 0)}%`],
  ];
  $("telemetry").innerHTML = rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("");
}

function updateEvents() {
  const rp = state.replay;
  if (!rp) return;
  const shown = (rp.events || []).filter((e) => e.t <= state.t + 0.01).slice(-14).reverse();
  $("events").innerHTML = shown.map((e) => {
    const cls = /shot down|kill/i.test(e.msg) ? "kill" : (/hits/.test(e.msg) ? "hit" : "");
    return `<li class="${cls}">[${fmt(e.t, 1)}s] ${e.msg}</li>`;
  }).join("");
}

// --------------------------------------------------------------- brain + fly views
function drawBrainView(s) {
  const cv = $("brainView");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = cv.width, H = cv.height;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const w = W/dpr, h = H/dpr;
  ctx.clearRect(0,0,w,h);

  const brain = s.brain;
  const hasBrain = !!brain;
  $("brainMeta").textContent = hasBrain ? `${fmt(brain.population_hz,2)} Hz pop · ${brain.spikes_last} spikes` : "no brain field — run --blue brain";

  const groups = hasBrain ? brain.groups : null;
  const perType = hasBrain ? brain.per_type : null;

  if (!hasBrain) {
    const c = s.b.cmd || [0,0,0,false];
    const roll = c[0], pull = c[1], thr = c[2];
    const pseudo = [
      { label: "turn_left", value: Math.max(0, -roll), color: COL.blue },
      { label: "turn_right", value: Math.max(0, roll), color: COL.red },
      { label: "pitch_up", value: Math.max(0, pull), color: COL.green },
      { label: "pitch_down", value: Math.max(0, -pull), color: COL.amber },
      { label: "speed", value: thr, color: COL.cyan },
      { label: "trigger", value: c[3]?1:0, color: COL.magenta },
    ];
    drawBarGroup(ctx, w, h, pseudo, "pseudo from commands (no brain field)");
    $("brainLegend").innerHTML = `<span class="dim">No brain data in this replay. Run <span class="mono">python -m flybfm fight --blue brain --replay web/replays/brain.json</span> to record real DN rates.</span>`;
    return;
  }

  const order = ["turn_left","turn_right","vis_left","vis_right","pitch_up","pitch_down","speed","trigger"];
  const bars = [];
  for (const k of order) {
    if (groups && groups[k] !== undefined) {
      let col = COL.dim;
      if (k.includes("left")) col = COL.blue;
      if (k.includes("right")) col = COL.red;
      if (k.includes("pitch_up")) col = COL.green;
      if (k.includes("pitch_down")) col = COL.amber;
      if (k==="speed") col = COL.cyan;
      if (k==="trigger") col = COL.magenta;
      bars.push({ label: k, value: groups[k], color: col });
    }
  }
  const extra = Object.keys(groups || {}).filter(k=>!order.includes(k)).slice(0,4);
  for (const k of extra) bars.push({ label: k, value: groups[k], color: COL.dim });

  drawBarGroup(ctx, w, h, bars, "DN + STMD groups");

  if (perType && Object.keys(perType).length) {
    const top = Object.entries(perType).sort((a,b)=>b[1]-a[1]).slice(0,10);
    $("brainLegend").innerHTML = top.map(([t,v])=>`<span style="color:${COL.blue}">${t}</span> ${fmt(v,2)}Hz`).join(" · ");
  } else {
    $("brainLegend").innerHTML = `<span class="dim">groups: ${Object.keys(groups||{}).join(", ")}</span>`;
  }
}

function drawBarGroup(ctx, w, h, bars, title) {
  const pad = 10, labelW = 88, gap = 4;
  const maxV = Math.max(0.01, ...bars.map(b=>Math.abs(b.value)));
  const barH = (h - pad*2 - 16) / Math.max(bars.length,1);
  ctx.fillStyle = "#8aa0b8"; ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(title, pad, pad+8);
  bars.forEach((b,i)=>{
    const y = pad+16 + i*(barH+gap);
    const x0 = pad+labelW;
    const bw = (w - x0 - pad) * (Math.abs(b.value)/maxV);
    ctx.fillStyle = "#7b8a9c"; ctx.textAlign="right";
    ctx.fillText(b.label, x0-6, y+barH*0.6);
    ctx.textAlign="left";
    ctx.fillStyle = "rgba(22,32,43,0.9)";
    ctx.fillRect(x0, y, w - x0 - pad, barH);
    ctx.fillStyle = b.color;
    ctx.fillRect(x0, y, bw, barH);
    ctx.fillStyle = "#cfd8e3"; ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(fmt(b.value,2), x0 + bw + 4, y+barH*0.6);
  });
}

function drawFlyView(s) {
  const cv = $("flyView");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = cv.width, H = cv.height;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const w = W/dpr, h = H/dpr;
  ctx.clearRect(0,0,w,h);
  const c = s.b.cmd || [0,0,0,false];
  const roll = c[0], pull = c[1], thr = c[2], trig = c[3];
  const brain = s.brain;
  const hasBrain = !!brain;
  $("flyMeta").textContent = `${roll>=0?'roll right':'roll left'} ${fmt(Math.abs(roll),2)} · ${pull>=0?'pull up':'push down'} ${fmt(Math.abs(pull),2)} · thr ${fmt(thr,2)} ${trig?'· FIRE':''}`;
  // enhanced cockpit - basic imported cockpit look, with panels
  const bgGrad = ctx.createLinearGradient(0,0,0,h);
  bgGrad.addColorStop(0, "#0a121e");
  bgGrad.addColorStop(0.18, "#0f1c2a");
  bgGrad.addColorStop(0.38, "#13202e");
  bgGrad.addColorStop(0.68, "#101a26");
  bgGrad.addColorStop(1, "#0a0f18");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0,0,w,h);
  // canopy bow
  ctx.fillStyle = "rgba(30,45,65,0.65)";
  ctx.fillRect(0,0,w, 20);
  ctx.fillStyle = "rgba(77,163,255,0.10)";
  ctx.fillRect(0,20,w, 4);
  // canopy reflection
  ctx.fillStyle = "rgba(100, 160, 220, 0.06)";
  ctx.beginPath();
  ctx.ellipse(w*0.5, h*0.10, w*0.48, 26, 0, 0, Math.PI*2);
  ctx.fill();
  // instrument glare
  ctx.fillStyle = "rgba(77,163,255,0.04)";
  ctx.beginPath();
  ctx.ellipse(w*0.22, h*0.88, w*0.18, 12, 0, 0, Math.PI*2);
  ctx.fill();
  // side consoles - more detailed cockpit
  ctx.fillStyle = "#0e1a28";
  ctx.fillRect(0, h*0.76, w*0.40, h*0.24);
  ctx.fillRect(w*0.60, h*0.76, w*0.40, h*0.24);
  ctx.strokeStyle = "#1e2e42"; ctx.lineWidth=1.2;
  ctx.strokeRect(0, h*0.76, w*0.40, h*0.24);
  ctx.strokeRect(w*0.60, h*0.76, w*0.40, h*0.24);
  // buttons on side consoles
  ctx.fillStyle = "#1a2a3d";
  for (let i=0;i<3;i++) {
    for (let j=0;j<4;j++) {
      ctx.fillStyle = (i+j)%2===0 ? "#1e2e44" : "#1a2a3d";
      ctx.fillRect(6 + j*18, h*0.78 + i*16, 10, 8);
    }
    for (let j=0;j<4;j++) {
      ctx.fillStyle = (i+j)%2===0 ? "#1e2e44" : "#1a2a3d";
      ctx.fillRect(w*0.62 + 8 + j*18, h*0.78 + i*16, 10, 8);
    }
  }
  // seat - more detailed, with ejection seat look
  const seatX = w*0.60, seatY = h*0.28, seatW = w*0.36, seatH = h*0.54;
  ctx.fillStyle = "#141f2f";
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(seatX, seatY, seatW, seatH, 14);
  else ctx.rect(seatX, seatY, seatW, seatH);
  ctx.fill();
  ctx.strokeStyle = "#2a3a52"; ctx.lineWidth=1.6; ctx.stroke();
  // seat belts
  ctx.strokeStyle = "rgba(80, 90, 60, 0.45)"; ctx.lineWidth=3;
  ctx.beginPath(); ctx.moveTo(seatX+8, seatY+12); ctx.lineTo(seatX+seatW*0.45, seatY+seatH*0.55); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(seatX+seatW-8, seatY+12); ctx.lineTo(seatX+seatW*0.55, seatY+seatH*0.55); ctx.stroke();
  // seat texture lines
  ctx.strokeStyle = "rgba(60,84,116,0.28)"; ctx.lineWidth=1;
  for (let i=1;i<5;i++){
    const yy = seatY + i*seatH/5.5;
    ctx.beginPath(); ctx.moveTo(seatX+10, yy); ctx.lineTo(seatX+seatW-10, yy); ctx.stroke();
  }
  // headrest
  ctx.fillStyle = "#1c2a3a";
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(seatX+seatW*0.12, seatY-12, seatW*0.76, 24, 8);
  else ctx.rect(seatX+seatW*0.12, seatY-12, seatW*0.76, 24);
  ctx.fill();
  ctx.strokeStyle = "#2a3a52"; ctx.lineWidth=1; ctx.stroke();
  const stickBaseX = w*0.30, stickBaseY = h*0.62;
  const stickRange = 58;
  const sx = stickBaseX + roll*stickRange;
  const sy = stickBaseY + pull*stickRange*0.7;
  ctx.fillStyle = "#0a141e";
  ctx.beginPath(); ctx.arc(stickBaseX, stickBaseY, 74, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = "#1e2e42"; ctx.lineWidth=2; ctx.stroke();
  ctx.strokeStyle = "rgba(77,122,160,0.22)"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(stickBaseX-74, stickBaseY); ctx.lineTo(stickBaseX+74, stickBaseY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(stickBaseX, stickBaseY-74); ctx.lineTo(stickBaseX, stickBaseY+74); ctx.stroke();
  ctx.strokeStyle = "rgba(77,163,255,0.18)"; ctx.setLineDash([4,6]);
  ctx.beginPath(); ctx.arc(stickBaseX, stickBaseY, 42, 0, Math.PI*2); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#2a3f5a"; ctx.lineWidth=10; ctx.lineCap="round";
  ctx.beginPath(); ctx.moveTo(stickBaseX, stickBaseY+18); ctx.lineTo(sx, sy+10); ctx.stroke();
  ctx.strokeStyle = "#3a5578"; ctx.lineWidth=6; ctx.beginPath(); ctx.moveTo(stickBaseX, stickBaseY+12); ctx.lineTo(sx, sy+6); ctx.stroke();
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(roll*0.25);
  ctx.fillStyle = "#1c2e44";
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(-10, -26, 20, 42, 6);
  else ctx.rect(-10, -26, 20, 42);
  ctx.fill();
  ctx.strokeStyle = "#2f4a67"; ctx.lineWidth=1.2; ctx.stroke();
  ctx.fillStyle = trig ? "#ff3b3b" : "#4a5a70";
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(-4, 6, 8, 12, 3);
  else ctx.rect(-4,6,8,12);
  ctx.fill();
  if (trig) {
    ctx.fillStyle = "rgba(255,60,60,0.35)";
    ctx.beginPath(); ctx.arc(0, 12, 18, 0, Math.PI*2); ctx.fill();
  }
  ctx.fillStyle = "#5a708a";
  ctx.beginPath(); ctx.arc(0, -14, 5, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = "#1a2a3a"; ctx.lineWidth=0.8; ctx.stroke();
  ctx.fillStyle = "#2a3f5a";
  ctx.beginPath(); ctx.ellipse(0, -24, 9, 6, 0, 0, Math.PI*2); ctx.fill();
  ctx.restore();
  ctx.fillStyle = "rgba(0,0,0,0.25)";
  ctx.beginPath(); ctx.ellipse(stickBaseX, stickBaseY+20, 22, 8, 0, 0, Math.PI*2); ctx.fill();
  const tx = w*0.10, ty0 = h*0.18, ty1 = h*0.74;
  ctx.fillStyle = "#0e1a27";
  ctx.fillRect(tx-14, ty0-8, 28, ty1-ty0+16);
  ctx.strokeStyle = "#1e2e42"; ctx.lineWidth=1.2; ctx.strokeRect(tx-14, ty0-8, 28, ty1-ty0+16);
  ctx.strokeStyle = "rgba(80,110,140,0.25)"; ctx.lineWidth=1;
  for (let i=0;i<=4;i++){
    const yy = ty0 + i*(ty1-ty0)/4;
    ctx.beginPath(); ctx.moveTo(tx-10, yy); ctx.lineTo(tx+10, yy); ctx.stroke();
  }
  const ty = ty1 - thr*(ty1-ty0);
  ctx.fillStyle = "#1a2a3d";
  ctx.fillRect(tx-16, ty-16, 32, 10);
  ctx.fillStyle = "#5de0ff";
  ctx.fillRect(tx-18, ty-14, 36, 6);
  ctx.strokeStyle = "#8ad8ff"; ctx.lineWidth=0.8; ctx.strokeRect(tx-18, ty-14, 36, 6);
  ctx.fillStyle = "#cfd8e3";
  ctx.beginPath(); ctx.arc(tx, ty-14, 7, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = "#1a2a3a"; ctx.lineWidth=1; ctx.stroke();
  ctx.fillStyle = "#5a708a"; ctx.font="9px ui-monospace, monospace";
  ctx.fillText("IDLE", tx+16, ty1+2);
  ctx.fillText("MIL", tx+16, ty0+4 + (ty1-ty0)*0.25);
  ctx.fillText("MAX", tx+16, ty0+4);
  ctx.fillStyle = "#7b8a9c"; ctx.font="10px ui-monospace, monospace";
  ctx.fillText("THR", tx-14, ty0-14);
  ctx.fillText(fmt(thr*100,0)+"%", tx-16, ty1+18);
  const fx = w*0.80, fy = h*0.50;
  const flap = Math.sin(Date.now()*0.022 * (0.6 + thr*2.5)) * (12 + thr*20);
  const flap2 = Math.cos(Date.now()*0.022 * (0.6 + thr*2.5)) * (6 + thr*10);
  ctx.save();
  ctx.translate(fx, fy);
  ctx.rotate(pull*0.12);
  // shadow under fly
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.beginPath(); ctx.ellipse(0, 34, 20, 7, 0, 0, Math.PI*2); ctx.fill();
  // legs - 6 legs, Drosophila has 3 pairs, front pair manipulates stick
  function drawLeg(ax, ay, bx, by, cx, cy, thick=1.9, highlight=false) {
    ctx.strokeStyle = highlight ? "#4a3522" : "#2a1a12"; ctx.lineWidth=thick; ctx.lineCap="round"; ctx.lineJoin="round";
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(cx, cy); ctx.stroke();
    ctx.fillStyle = highlight ? "#6a4a32" : "#1a0f0a"; ctx.beginPath(); ctx.arc(cx, cy, 1.3, 0, Math.PI*2); ctx.fill();
  }
  // hind legs
  drawLeg(3, 16, 12, 24, 16, 36, 1.8);
  drawLeg(-3, 16, -12, 24, -16, 36, 1.8);
  // middle legs
  drawLeg(4, 6, 14, 10, 18, 22, 1.7);
  drawLeg(-4, 6, -14, 10, -18, 22, 1.7);
  // halteres - Drosophila distinctive, small club-shaped balancing organs behind wings
  ctx.fillStyle = "rgba(200, 180, 160, 0.55)";
  ctx.strokeStyle = "rgba(160, 140, 120, 0.6)"; ctx.lineWidth=0.6;
  ctx.beginPath(); ctx.moveTo(-2, 2); ctx.lineTo(-5, 6); ctx.stroke();
  ctx.beginPath(); ctx.arc(-5.5, 7.5, 1.8, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(2, 2); ctx.lineTo(5, 6); ctx.stroke();
  ctx.beginPath(); ctx.arc(5.5, 7.5, 1.8, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  // wings - transparent, veined, Drosophila-like
  ctx.save();
  ctx.rotate(flap * 0.018 + flap2*0.005);
  ctx.fillStyle = "rgba(175, 205, 235, 0.18)";
  ctx.strokeStyle = "rgba(115, 155, 185, 0.42)"; ctx.lineWidth=0.7;
  ctx.beginPath();
  ctx.moveTo(-1, -1);
  ctx.bezierCurveTo(-10, -9, -28, -8, -36, 1);
  ctx.bezierCurveTo(-32, 9, -16, 12, -1, 3);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
  // wing veins
  ctx.strokeStyle = "rgba(90, 130, 165, 0.55)"; ctx.lineWidth=0.45;
  ctx.beginPath(); ctx.moveTo(-1,0); ctx.lineTo(-34,0.5); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-4,-2); ctx.lineTo(-26,-5); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-3,1); ctx.lineTo(-22,6); ctx.stroke();
  // wing membrane highlight
  ctx.fillStyle = "rgba(200, 220, 255, 0.08)";
  ctx.beginPath(); ctx.ellipse(-18, -1, 10, 5, -0.15, 0, Math.PI*2); ctx.fill();
  ctx.restore();
  ctx.save();
  ctx.rotate(-flap * 0.018 - flap2*0.005);
  ctx.fillStyle = "rgba(175, 205, 235, 0.18)";
  ctx.strokeStyle = "rgba(115, 155, 185, 0.42)"; ctx.lineWidth=0.7;
  ctx.beginPath();
  ctx.moveTo(1, -1);
  ctx.bezierCurveTo(10, -9, 28, -8, 36, 1);
  ctx.bezierCurveTo(32, 9, 16, 12, 1, 3);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.strokeStyle = "rgba(90, 130, 165, 0.55)"; ctx.lineWidth=0.45;
  ctx.beginPath(); ctx.moveTo(1,0); ctx.lineTo(34,0.5); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(4,-2); ctx.lineTo(26,-5); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(3,1); ctx.lineTo(22,6); ctx.stroke();
  ctx.fillStyle = "rgba(200, 220, 255, 0.08)";
  ctx.beginPath(); ctx.ellipse(18, -1, 10, 5, 0.15, 0, Math.PI*2); ctx.fill();
  ctx.restore();
  // abdomen - segmented, yellow with dark stripes, Drosophila melanogaster
  ctx.fillStyle = "#e6c85a";
  ctx.strokeStyle = "#5a4a2a"; ctx.lineWidth=0.6;
  ctx.beginPath();
  ctx.ellipse(0, 16, 7.5, 15, 0, 0, Math.PI*2);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#3a2a18";
  for (let i=0;i<6;i++){
    const yy = 8 + i*4.2;
    const ww = 6.8 - i*0.55;
    ctx.fillRect(-ww, yy, ww*2, 1.5);
  }
  // thorax
  ctx.fillStyle = "#6a5a4a";
  ctx.strokeStyle = "#2a1e12"; ctx.lineWidth=0.7;
  ctx.beginPath();
  ctx.ellipse(0, 1, 7.2, 10, 0, 0, Math.PI*2);
  ctx.fill(); ctx.stroke();
  // thorax bristles
  ctx.strokeStyle = "#1a120a"; ctx.lineWidth=0.55;
  for (let i=-2;i<=2;i++){
    const bx = i*2.2;
    ctx.beginPath(); ctx.moveTo(bx, -3); ctx.lineTo(bx + (Math.random()-0.5)*0.8, -9.5); ctx.stroke();
  }
  // head
  ctx.fillStyle = "#4a3a2a";
  ctx.beginPath();
  ctx.ellipse(0, -9.5, 6.0, 5.5, 0, 0, Math.PI*2);
  ctx.fill();
  ctx.strokeStyle = "#1a0f08"; ctx.lineWidth=0.6; ctx.stroke();
  // eyes - large, red, faceted, Drosophila
  const eyeGrad = ctx.createRadialGradient(-4.8, -10.5, 0.5, -4.8, -10.5, 5.8);
  eyeGrad.addColorStop(0, "#ff8a7a");
  eyeGrad.addColorStop(0.15, "#ff5a4a");
  eyeGrad.addColorStop(0.35, "#cc2220");
  eyeGrad.addColorStop(0.7, "#7a0f0f");
  eyeGrad.addColorStop(1, "#4a0808");
  ctx.fillStyle = eyeGrad;
  ctx.beginPath(); ctx.ellipse(-4.2, -10.5, 4.6, 5.6, -0.18, 0, 0, Math.PI*2); ctx.fill();
  // facets highlight
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  for (let i=0;i<3;i++) {
    ctx.beginPath(); ctx.arc(-4.2 + (Math.random()-0.5)*2, -10.5 + (Math.random()-0.5)*3, 0.6, 0, Math.PI*2); ctx.fill();
  }
  const eyeGradR = ctx.createRadialGradient(4.8, -10.5, 0.5, 4.8, -10.5, 5.8);
  eyeGradR.addColorStop(0, "#ff8a7a");
  eyeGradR.addColorStop(0.15, "#ff5a4a");
  eyeGradR.addColorStop(0.35, "#cc2220");
  eyeGradR.addColorStop(0.7, "#7a0f0f");
  eyeGradR.addColorStop(1, "#4a0808");
  ctx.fillStyle = eyeGradR;
  ctx.beginPath(); ctx.ellipse(4.2, -10.5, 4.6, 5.6, 0.18, 0, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  for (let i=0;i<3;i++) {
    ctx.beginPath(); ctx.arc(4.2 + (Math.random()-0.5)*2, -10.5 + (Math.random()-0.5)*3, 0.6, 0, Math.PI*2); ctx.fill();
  }
  // eye shine
  ctx.fillStyle = "rgba(255,210,190,0.22)";
  ctx.beginPath(); ctx.arc(-4.8, -11.5, 1.3, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.arc(3.6, -11.5, 1.3, 0, Math.PI*2); ctx.fill();
  // antennae
  ctx.strokeStyle = "#1a0f08"; ctx.lineWidth=0.9;
  ctx.beginPath(); ctx.moveTo(-1.8, -12.5); ctx.lineTo(-2.8, -15.8); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(1.8, -12.5); ctx.lineTo(2.8, -15.8); ctx.stroke();
  ctx.fillStyle = "#1a0f08";
  ctx.beginPath(); ctx.arc(-2.9, -16.0, 1.1, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.arc(2.9, -16.0, 1.1, 0, Math.PI*2); ctx.fill();
  // arista (feathery)
  ctx.strokeStyle = "#1a0f08"; ctx.lineWidth=0.4;
  ctx.beginPath(); ctx.moveTo(-2.9,-16); ctx.lineTo(-4.5,-15.2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(2.9,-16); ctx.lineTo(4.5,-15.2); ctx.stroke();
  // proboscis
  ctx.fillStyle = "#2a1a0f";
  ctx.beginPath(); ctx.ellipse(0, -6.0, 1.3, 1.8, 0, 0, Math.PI*2); ctx.fill();
  // front legs manipulating stick - more realistic, with joints
  const stickVecX = (stickBaseX - fx) + roll*stickRange;
  const stickVecY = (stickBaseY - fy) + pull*stickRange*0.7;
  // left front leg reaching stick
  ctx.strokeStyle = "#3a2a1a"; ctx.lineWidth=2.0; ctx.lineCap="round"; ctx.lineJoin="round";
  ctx.beginPath();
  ctx.moveTo(-2, -1.5);
  ctx.quadraticCurveTo(-7 + stickVecX*0.18, -3 + stickVecY*0.12, stickVecX*0.52 -3, stickVecY*0.52);
  ctx.stroke();
  ctx.strokeStyle = "#4a3a2a"; ctx.lineWidth=1.4;
  ctx.beginPath();
  ctx.moveTo(-2, -1.5);
  ctx.quadraticCurveTo(-7 + stickVecX*0.18, -3 + stickVecY*0.12, stickVecX*0.52 -3, stickVecY*0.52);
  ctx.stroke();
  // right front leg
  ctx.strokeStyle = "#3a2a1a"; ctx.lineWidth=2.0;
  ctx.beginPath();
  ctx.moveTo(2, -1.5);
  ctx.quadraticCurveTo(7 + stickVecX*0.18, -3 + stickVecY*0.12, stickVecX*0.52 +3, stickVecY*0.52);
  ctx.stroke();
  ctx.strokeStyle = "#4a3a2a"; ctx.lineWidth=1.4;
  ctx.beginPath();
  ctx.moveTo(2, -1.5);
  ctx.quadraticCurveTo(7 + stickVecX*0.18, -3 + stickVecY*0.12, stickVecX*0.52 +3, stickVecY*0.52);
  ctx.stroke();
  // tarsi gripping stick
  ctx.fillStyle = "#5a4a32";
  ctx.beginPath(); ctx.arc(stickVecX*0.52-3, stickVecY*0.52, 2.2, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = "#2a1a12"; ctx.lineWidth=0.6; ctx.stroke();
  ctx.beginPath(); ctx.arc(stickVecX*0.52+3, stickVecY*0.52, 2.2, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  ctx.restore();
  ctx.fillStyle = "rgba(77,163,255,0.12)";
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillText(`STICK ${fmt(roll,2)} / ${fmt(pull,2)}`, w*0.02, h*0.92);
  ctx.fillText(`THR ${fmt(thr*100,0)}% ${trig?'FIRE':''}`, w*0.02, h*0.96);
  const status = [];
  if (Math.abs(roll)>0.1) status.push(roll>0?"→ rolling right":"← rolling left");
  else status.push("→ wings level");
  if (Math.abs(pull)>0.1) status.push(pull>0?"↑ pulling up":"↓ pushing down");
  else status.push("↔ pitch neutral");
  status.push(`throttle ${fmt(thr*100,0)}%`);
  if (trig) status.push("🔴 TRIGGER DOWN");
  if (hasBrain) {
    const g = brain.groups;
    const lr = (g.turn_right||0)-(g.turn_left||0);
    const vdiff = (g.vis_right||0)-(g.vis_left||0);
    status.push(`brain: DN L/R ${fmt(lr,2)}Hz, STMD L/R ${fmt(vdiff,2)}Hz`);
  }
  $("flyStatus").innerHTML = status.map(s=>`<span>${s}</span>`).join(" · ");
}

// --------------------------------------------------------------- strip chart
function buildStrip() {
  const cv = $("strip");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const off = document.createElement("canvas");
  off.width = cv.width; off.height = cv.height;
  const ctx = off.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const f = state.frames;
  if (!f.length) { state.strip = { off, w, h, dpr }; return; }
  const pad = 6, laneH = (h - pad * 4) / 3;
  const X = (t) => pad + (t / state.dur) * (w - pad * 2);

  const lanes = [
    { title: "range (log 100m–20km)", color: "#9fb4cc", y: (v) => {
        const u = (Math.log10(clamp(v, 100, 20000)) - 2) / (Math.log10(20000) - 2);
        return pad + (1 - u) * laneH * 0.92; } },
    { title: "nose error AA (°, ±180)", color: COL.blue, y: (v) => {
        const u = (clamp(v, -180, 180) + 180) / 360; return pad + laneH + 22 + (1 - u) * laneH * 0.92; } },
    { title: "speed m/s (blue) / TAA (°)", color: COL.amber, y: (v) => {
        const u = clamp(v / 400, 0, 1); return pad * 2 + (laneH + 22) * 2 + (1 - u) * laneH * 0.92; } },
  ];
  lanes.forEach((ln) => {
    ctx.strokeStyle = "rgba(50,70,95,.6)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, ln.y(0)); ctx.lineTo(w - pad, ln.y(0)); ctx.stroke();
    ctx.fillStyle = COL.dim; ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(ln.title, pad + 2, ln.y(0) - 4);
  });
  const series = [
    { lane: 0, get: (fr) => fr.g.rng, color: "#9fb4cc" },
    { lane: 1, get: (fr) => fr.g.aa, color: COL.blue },
    { lane: 2, get: (fr) => fr.b.v, color: COL.amber },
    { lane: 2, get: (fr) => fr.g.taa, color: COL.red },
  ];
  for (const se of series) {
    const ln = lanes[se.lane];
    ctx.strokeStyle = se.color; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.9;
    ctx.beginPath();
    f.forEach((fr, i) => {
      const v = se.get(fr);
      if (v === undefined || v === null) return;
      const x = X(fr.t), y = ln.y(v);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  for (const e of (state.replay.events || [])) {
    const x = X(e.t);
    ctx.strokeStyle = /shot down/.test(e.msg) ? COL.green : "rgba(255,200,87,.5)";
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
  }
  state.strip = { off, w, h, dpr, X };
}

function drawStripPlayhead() {
  const st = state.strip;
  if (!st) return;
  const cv = $("strip"), ctx = cv.getContext("2d");
  ctx.setTransform(st.dpr, 0, 0, st.dpr, 0, 0);
  ctx.clearRect(0, 0, st.w, st.h);
  ctx.drawImage(st.off, 0, 0, st.w, st.h);
  const x = st.X(state.t);
  ctx.strokeStyle = "#e8f1ff"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, st.h); ctx.stroke();
}

// --------------------------------------------------------------- main loop
function frame(ts) {
  const dt = Math.min((ts - state.lastTS) / 1000 || 0, 0.1);
  state.lastTS = ts;
  if (state.playing && state.dur > 0) {
    state.t += dt * state.speed;
    if (state.t >= state.dur) { state.t = state.dur; setPlaying(false); }
    $("scrub").value = String(Math.round((state.t / state.dur) * 1000));
  }
  const s = sampleAt(state.t);
  if (s) {
    const cv = $("view"), ctx = cv.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== cv.clientWidth * dpr) { cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    state.cam.target = [(s.b.p[0] + s.r.p[0]) / 2, (s.b.p[1] + s.r.p[1]) / 2, (s.b.p[2] + s.r.p[2]) / 2];
    if (state.cam.mode === "chase-blue") { state.cam.yaw = s.b.psi * Math.PI / 180; state.cam.pitch = 0.35; state.cam.dist = 2200; }
    else if (state.cam.mode === "chase-red") { state.cam.yaw = s.r.psi * Math.PI / 180; state.cam.pitch = 0.35; state.cam.dist = 2200; }
    else if (state.cam.mode === "top") { state.cam.pitch = 1.45; state.cam.dist = 5000; }

    const cam = makeCamera();
    drawWorld(ctx, cam, s);
    drawEngagement(ctx, cam, s);
    drawAircraft(ctx, cam, s.b, COL.blue, "B", s);
    drawAircraft(ctx, cam, s.r, COL.red, "R", s);

    updateHUD(s); updatePanel(s); updateEvents();
    drawStripPlayhead();
    drawBrainView(s);
    drawFlyView(s);
    $("clock").textContent = fmt(s.t,1)+"s / "+fmt(state.dur,1)+"s";
  }
  requestAnimationFrame(frame);
}

function setPlaying(p) { state.playing = p; $("play").textContent = p ? "⏸" : "▶"; }

// ------------------------------------------------------------------ loading
async function loadManifest() {
  const res = await fetch("replays/index.json", { cache: "no-store" });
  const man = await res.json();
  state.manifest = man;
  // try to include live.json even if not in manifest yet (training may have created it)
  let replays = [...man.replays];
  try {
    const liveRes = await fetch("replays/live.json", { method: "HEAD", cache: "no-store" });
    if (liveRes.ok && !replays.some(r => r.file === "live.json")) {
      replays.unshift({ file: "live.json", tag: "live_best", result: "live", duration_s: 0, blue: "best", red: "live", frames: 0, manoeuvres: 0 });
    }
  } catch {}
  // also check API
  try {
    const apiLive = await fetch("/api/train/live", { cache: "no-store" });
    if (apiLive.ok) {
      const j = await apiLive.json();
      if (j && j.generation !== undefined && !replays.some(r => r.file === "live.json")) {
        replays.unshift({ file: "__live_api__", tag: `live gen ${j.generation}`, result: j.result || "live", duration_s: j.duration_s || 0, blue: j.blue_name || "best", red: j.red_name || "opp", frames: (j.frames||[]).length, manoeuvres: 0, _liveData: j });
      }
    }
  } catch {}
  const sel = $("replaySel");
  sel.innerHTML = replays.map((r, i) =>
    `<option value="${r.file}" ${r.file==="live.json"?"style=\"color:#5ddc8a;font-weight:600\"":""}>${r.file==="live.json"?"🔴 LIVE — ":""}${r.tag} · ${r.result} · ${fmt(r.duration_s, 0)}s · ${r.blue} v ${r.red}${r.file==="live.json"?" · best result":""}</option>`).join("");
  // stash for selector handler
  state.manifest.replays = replays;
  if (replays.length) {
    // prefer live if training was running recently
    const liveOpt = replays.find(r => r.file === "live.json" || r.file === "__live_api__");
    if (liveOpt && state.autoWatchLive) {
      if (liveOpt.file === "__live_api__" && liveOpt._liveData) await loadReplayData(liveOpt._liveData);
      else await loadReplay(liveOpt.file);
    } else {
      await loadReplay(replays[0].file);
    }
  }
}

async function loadReplay(file) {
  if (file === "__live_api__") {
    const live = await fetchLiveReplay();
    if (live) { await loadReplayData(live); return; }
  }
  // support live.json via API fallback
  let rp = null;
  try {
    const res = await fetch("replays/" + file, { cache: "no-store" });
    if (!res.ok) throw new Error("not found in replays/");
    rp = await res.json();
  } catch {
    if (file === "live.json") {
      rp = await fetchLiveReplay();
    }
  }
  if (!rp) {
    console.warn("replay not found", file);
    return;
  }
  await loadReplayData(rp);
}

async function loadReplayData(rp) {
  state.replay = rp;
  state.frames = rp.frames || [];
  state.dur = (state.frames.length ? state.frames[state.frames.length - 1].t : 1) || 1;
  state.t = 0;
  $("scenTitle").textContent = rp.scenario.tag;
  $("scenDesc").textContent = rp.scenario.desc;
  const rc = rp.result;
  const label = rc === "blue" ? `BLUE ${rp.blue_name || "blue"} wins` : rc === "red" ? `RED ${rp.red_name || "red"} wins` : `draw / ${rc}`;
  $("result").textContent = label;
  $("result").className = "result " + (rc === "blue" ? "blue" : rc === "red" ? "red" : "draw");
  const m = rp.manoeuvres || {};
  const mix = m.pursuit_mix || {};
  $("manoeuvres").innerHTML =
    `<div><b>manoeuvre content</b> ${fmt(m.manoeuvre_content, 2)} — fraction of the fight in a recognised tactic</div>` +
    `<div><b>pursuit mix</b> lead ${fmt((mix.lead || 0) * 100, 0)}% · pure ${fmt((mix.pure || 0) * 100, 0)}% · lag ${fmt((mix.lag || 0) * 100, 0)}%</div>` +
    `<div><b>counted</b> ${Object.entries(m.counts || {}).map(([k, v]) => `${k}×${v}`).join(", ") || "none"}</div>`;
  const rate = state.frames.length > 1 ? (state.dur / (state.frames.length - 1)) : 0;
  $("rate").textContent = `${fmt(rate, 2)} s (${state.frames.length} frames)`;
  buildStrip();
  setPlaying(true);
  // show live badge if this is a live replay
  if (rp.generation !== undefined) {
    $("liveBadge").classList.remove("hidden");
    $("liveBadge").textContent = `LIVE gen ${rp.generation} · ${rp.blue_name || "fly"} vs ${rp.red_name || "opp"} · ${rc}`;
  } else {
    $("liveBadge").classList.add("hidden");
  }
}

// -------------------------------------------------------------- live training
let benchCache = null;
async function fetchBench() {
  try {
    const res = await fetch("/api/bench", { cache: "no-store" });
    if (!res.ok) return null;
    const j = await res.json();
    benchCache = j;
    return j;
  } catch { return null; }
}
async function fetchTrainStatus() {
  try {
    const res = await fetch("/api/train/status", { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}
async function fetchTrainLog() {
  try {
    const res = await fetch("/api/train/log?lines=120", { cache: "no-store" });
    if (!res.ok) return null;
    const j = await res.json();
    return j.log || "";
  } catch { return null; }
}
async function fetchLiveReplay() {
  try {
    const res = await fetch("/api/train/live", { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

function fmtTime(s) {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s/60).toFixed(1)} min`;
  return `${(s/3600).toFixed(1)}h (${(s/60).toFixed(0)} min)`;
}

function updateEstTime() {
  const gens = parseInt($("trainGens").value, 10) || 16;
  const pop = parseInt($("trainPop").value, 10) || 64;
  const eps = parseInt($("trainEps").value, 10) || 8;
  const totalFights = gens * pop * eps;
  const bench = benchCache;
  let avgFightS = 0.09; // fallback 90ms now 60s mix (was 50ms 20s random)
  let fightsPerSec = 11;
  let cpuCount = navigator.hardwareConcurrency || 4;
  let platformStr = "";
  if (bench) {
    avgFightS = bench.avg_fight_s || 0.09;
    fightsPerSec = bench.fights_per_second || (1/avgFightS);
    cpuCount = bench.cpu_count || cpuCount;
    platformStr = bench.platform || "";
  }
  // curriculum makes early fights shorter (15-20s) ~0.75x, plus eval overhead ~1.1x
  const estS = totalFights * avgFightS * 0.75 * 1.1 + gens * 2.0;
  const benchInfo = bench ? `${(avgFightS*1000).toFixed(0)}ms/fight, ${fightsPerSec.toFixed(1)} fights/s, ${cpuCount} CPUs, mix ${bench.easy_frac||0.4}/${bench.defensive_frac||0.3}/${bench.random_frac||0.3} 60s` : "no bench yet — using 90ms/fight fallback (60s mix)";
  const sysInfo = bench ? `${cpuCount} CPUs · ${platformStr.split('-')[0] || ''} · ${benchInfo}` : `${cpuCount} CPUs (browser) · ${benchInfo}`;
  $("estTime").innerHTML = `⏱ <b>${totalFights} fights</b> (${gens}×${pop}×${eps}) — est <b>${fmtTime(estS)}</b> on this machine (${benchInfo}). ` +
    `Gunnery-final: 22d obs az/el separate, 40/30/30 base but curriculum 100/0/0→80/5/15→40/30/30→20/40/40 + opponent level→all + episode 15-20s→60-90s. ` +
    `Reward tracking 1° w12 + 30° w2 inverse-quad, hit 10 kill 100, good trigger +2 bad -0.01, good_solution +0.5 in_wez +0.05. Linear 92 params pop64. ` +
    `Prioritizes evals over length. <span class="dim">Live fights/s from history, ETA from bench.</span>`;
  $("sysSpecs").textContent = sysInfo;
  const def = $("estDefault");
  if (def) def.textContent = `~${fmtTime(16*64*8*0.09*0.75)} on this machine (8192 fights)`;
}

function updateTrainUI(data) {
  if (!data) {
    $("trainStatus").textContent = "idle (no server or no training) — run python -m flybfm serve";
    $("trainStatus").className = "trainbar-status mono";
    return;
  }
  state.training = data;
  const running = data.running;
  const st = data.status || {};
  const gen = st.generation;
  const total = st.total_generations || "?";
  const best = st.best_return;
  const mean = st.mean_return;
  if (running) {
    $("trainStatus").textContent = `running pid ${data.pid} — gen ${gen !== undefined ? gen : "?"} / ${total} best ${best !== undefined ? best.toFixed(1) : "?"} mean ${mean !== undefined ? mean.toFixed(1) : "?"}`;
    $("trainStatus").className = "trainbar-status mono running";
    // auto-show training bar when training is running
    const bar = $("trainBar");
    if (bar && bar.classList.contains("hidden")) {
      bar.classList.remove("hidden");
    }
  } else {
    if (st && st.generation !== undefined) {
      $("trainStatus").textContent = `stopped — last gen ${gen} / ${total} best ${fmt(best,1)} — best result is live replay (auto-loaded below)`;
      // when training just finished, ensure best result is shown
      if (state.liveGen >= 0 && state.replay && state.replay.generation === undefined) {
        // will be loaded by pollTraining's finished check
      }
    } else {
      $("trainStatus").textContent = "idle — no training running. Click ⚙ train to configure and start.";
    }
    $("trainStatus").className = "trainbar-status mono";
  }
  // progress panel — shows best result details + live fights/s from history + ETA
  if (st && (st.history || st.generation !== undefined)) {
    const hist = (st.history || []).slice(-10);
    // compute live fights/s from history wall_s
    let totalFightsHist = 0;
    let totalWall = 0;
    const cfg = st.config || {};
    const pop = cfg.population || 24;
    const eps = cfg.episodes_per_candidate || 8;
    for (const h of hist) {
      if (h.wall_s) {
        totalWall += h.wall_s;
        totalFightsHist += pop * eps;
      }
    }
    const liveFps = totalWall > 0 ? totalFightsHist / totalWall : 0;
    const gens = cfg.generations || 16;
    const remainingGens = Math.max(0, gens - (st.generation !== undefined ? st.generation+1 : 0));
    const remainingFights = remainingGens * pop * eps;
    const etaS = liveFps > 0 ? remainingFights / liveFps : 0;

    const lines = hist.map(h => {
      const ev = h.eval ? ` win ${(h.eval.win_rate*100).toFixed(0)}% hit ${(h.eval.hit_rate*100).toFixed(1)}%` : "";
      const fps = h.wall_s ? ((pop*eps)/h.wall_s).toFixed(1) : "—";
      return `gen ${h.generation} best ${fmt(h.best_return,1)} mean ${fmt(h.mean_return,1)} ${h.max_time_s ? h.max_time_s.toFixed(0)+'s' : ''} ${fps} fights/s${ev}`;
    }).join("\n");
    const poolLines = (st.pool || []).slice(0,4).map(r => `${r.name}: ${r.learner_wins}W ${r.opponent_wins}L`).join(" · ");
    const bestInfo = st.best ? `best score ${fmt(st.best.score,2)} — ${st.best.params ? st.best.params.length + " params" : ""}` : "";
    const liveInfo = liveFps > 0 ? `live ${liveFps.toFixed(1)} fights/s, ETA ${fmtTime(etaS)} for ${remainingFights} fights remaining` : "live fights/s estimating…";
    $("trainProgress").innerHTML = `<b>gen ${st.generation}/${st.total_generations || "?"}</b> best ${fmt(st.best_return,2)} mean ${fmt(st.mean_return,2)} — ${liveInfo}<br>` +
      (bestInfo ? `${bestInfo}<br>` : "") +
      (st.eval ? `eval win ${fmt(st.eval.win_rate*100,0)}% loss ${fmt(st.eval.loss_rate*100,0)}% hit ${fmt(st.eval.hit_rate*100,1)}% — <b>best result</b><br>` : "") +
      `<span class="dim">${lines.replace(/\n/g,"<br>")}</span>` +
      (poolLines ? `<br><span class="dim">${poolLines}</span>` : "") +
      `<br><button id="loadBestBtn" class="ghost" style="margin-top:6px">👁 show best result replay</button>`;
    const btn = $("loadBestBtn");
    if (btn) btn.onclick = () => watchLive();
  }
}

async function pollTraining() {
  const status = await fetchTrainStatus();
  updateTrainUI(status);
  const log = await fetchTrainLog();
  if (log !== null) {
    if (log.length) {
      $("trainLog").textContent = log;
      $("trainLog").scrollTop = $("trainLog").scrollHeight;
    } else if (!status || !status.running) {
      // keep previous log if empty and not running
    }
  }
  // live replay handling — both while training and after it finishes (best result)
  const live = await fetchLiveReplay();
  if (live && live.generation !== undefined) {
    const isNewGen = live.generation !== state.liveGen;
    if (isNewGen) state.liveGen = live.generation;
    // auto-load if:
    // - user enabled autoWatchLive (clicked start or watch live)
    // - we are already showing a live replay
    // - training just finished and we have no replay showing best yet (show best result)
    const shouldAutoLoad = state.autoWatchLive || (state.replay && state.replay.generation !== undefined) || (!status.running && status.status && status.status.generation !== undefined && !state.hasShownBest);
    if (shouldAutoLoad && isNewGen) {
      await loadReplayData(live);
      state.hasShownBest = true;
    } else if (isNewGen && !shouldAutoLoad) {
      // hint that new live is available — don't steal user's current replay
      $("liveBadge").classList.remove("hidden");
      $("liveBadge").textContent = `LIVE gen ${live.generation} available — click 👁 watch live fly (best result)`;
      $("liveBadge").onclick = () => watchLive();
    }
    // if training finished and we never showed best, show it now (best result requirement)
    if (!status.running && live && !state.hasShownBest) {
      await loadReplayData(live);
      state.hasShownBest = true;
      state.autoWatchLive = true;
    }
  }
}

async function startTraining() {
  const cfg = {
    policy: $("trainPolicy").value,
    basis: $("trainBasis").value,
    generations: parseInt($("trainGens").value, 10),
    population: parseInt($("trainPop").value, 10),
    episodes: parseInt($("trainEps").value, 10),
    seed: parseInt($("trainSeed").value, 10),
    easy_frac: parseInt($("trainEasy").value, 10) / 100.0,
    defensive_frac: parseInt($("trainDef").value, 10) / 100.0,
    random_frac: parseInt($("trainRand").value, 10) / 100.0,
    no_curriculum: $("trainNoCurr").checked,
  };
  // normalize fracs
  const tot = cfg.easy_frac + cfg.defensive_frac + cfg.random_frac;
  if (tot > 0) {
    cfg.easy_frac /= tot; cfg.defensive_frac /= tot; cfg.random_frac /= tot;
  }
  $("trainLog").textContent = `starting training: ${JSON.stringify(cfg, null, 2)}\n...`;
  try {
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    const j = await res.json();
    if (!res.ok) {
      $("trainLog").textContent = `failed to start: ${j.error || res.statusText}\n${JSON.stringify(j, null, 2)}`;
    } else {
      $("trainLog").textContent = `started pid ${j.pid}\ncmd: ${j.cmd}\nlog: ${j.log}\nlive replay: ${j.live_replay}\n\nThis process runs in the terminal (server process), not in the browser thread. Closing the window doesn't kill it. The viewer will show live matches while training.`;
      state.autoWatchLive = true;
      // immediate poll
      setTimeout(pollTraining, 1000);
    }
  } catch (e) {
    $("trainLog").textContent = `error: ${e}\n\nIf you are opening file:// directly, the API won't work. Run:\n  python -m flybfm serve --port 8000\nand open http://localhost:8000\n\nOr start training in terminal:\n  python -m flybfm train --live-replay web/replays/live.json --live-status runs/live/status.json --generations ${cfg.generations} --population ${cfg.population} --episodes ${cfg.episodes}`;
  }
}

async function stopTraining() {
  try {
    const res = await fetch("/api/train/stop", { method: "POST" });
    const j = await res.json();
    $("trainLog").textContent = `stop result: ${JSON.stringify(j, null, 2)}\n${$("trainLog").textContent}`;
    setTimeout(pollTraining, 500);
  } catch (e) {
    $("trainLog").textContent = `stop error: ${e}`;
  }
}

async function watchLive() {
  state.autoWatchLive = true;
  const live = await fetchLiveReplay();
  if (!live) {
    $("trainLog").textContent = "no live replay yet — training hasn't written one, or not running. Check /api/train/status and log.";
    return;
  }
  state.liveGen = live.generation;
  await loadReplayData(live);
}

// -------------------------------------------------------------------- events
function bind() {
  $("play").onclick = () => setPlaying(!state.playing);
  $("speed").onchange = (e) => { state.speed = parseFloat(e.target.value); };
  $("cam").onchange = (e) => { state.cam.mode = e.target.value; };
  $("replaySel").onchange = (e) => { state.autoWatchLive = false; loadReplay(e.target.value); };
  $("reload").onclick = () => loadManifest().catch((err) => console.error(err));
  $("scrub").oninput = (e) => {
    state.t = (parseFloat(e.target.value) / 1000) * state.dur;
    setPlaying(false);
  };
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
    if (e.code === "ArrowRight") { state.t = clamp(state.t + 0.5, 0, state.dur); setPlaying(false); }
    if (e.code === "ArrowLeft") { state.t = clamp(state.t - 0.5, 0, state.dur); setPlaying(false); }
  });
  const cv = $("view");
  cv.addEventListener("pointerdown", (e) => { state.drag = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointerup", () => { state.drag = null; });
  cv.addEventListener("pointermove", (e) => {
    if (!state.drag) return;
    const dx = e.clientX - state.drag.x, dy = e.clientY - state.drag.y;
    state.drag = { x: e.clientX, y: e.clientY };
    state.cam.yaw -= dx * 0.006;
    state.cam.pitch = clamp(state.cam.pitch + dy * 0.005, -0.2, 1.5);
    if (state.cam.mode !== "free") { state.cam.mode = "free"; $("cam").value = "free"; }
  });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    state.cam.dist = clamp(state.cam.dist * (1 + Math.sign(e.deltaY) * 0.12), 300, 30000);
  }, { passive: false });
  window.addEventListener("resize", () => { if (state.replay) buildStrip(); });

  // training controls
  const toggle = $("toggleTrain");
  if (toggle) {
    toggle.onclick = () => {
      $("trainBar").classList.toggle("hidden");
      // when opening, refresh est time
      if (!$("trainBar").classList.contains("hidden")) {
        updateEstTime();
        fetchBench().then(() => updateEstTime());
      }
    };
  }
  const startBtn = $("startTrain");
  if (startBtn) startBtn.onclick = () => startTraining();
  const stopBtn = $("stopTrain");
  if (stopBtn) stopBtn.onclick = () => stopTraining();
  const watchBtn = $("watchLive");
  if (watchBtn) watchBtn.onclick = () => watchLive();

  // estimated time — update on any training input change
  const estInputs = ["trainGens", "trainPop", "trainEps", "trainPolicy", "trainBasis"];
  for (const id of estInputs) {
    const el = $(id);
    if (el) {
      el.addEventListener("input", () => updateEstTime());
      el.addEventListener("change", () => updateEstTime());
    }
  }
}

bind();
loadManifest().catch((err) => {
  $("scenTitle").textContent = "no replay";
  $("scenDesc").textContent = "Run `python -m flybfm fight --replay web/replays/<name>.json` to produce one.";
  console.error(err);
});
requestAnimationFrame(frame);

// initial bench + est time
fetchBench().then(() => updateEstTime());
updateEstTime();

// start polling training status every 2s
setInterval(pollTraining, 2000);
setTimeout(pollTraining, 800);
setInterval(() => { fetchBench().then(() => updateEstTime()); }, 15000);
