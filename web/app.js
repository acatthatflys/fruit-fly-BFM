/* fruit-fly-BFM replay viewer — vanilla JS, no dependencies.
 *
 * The view is a perspective camera on a z-up world.  Aircraft are drawn as
 * fixed-size icons (a real fighter is 9 m long in a 12 km arena — anything
 * drawn to scale is invisible), with a vertical line down to the ground so
 * altitude is readable.  Everything else is HUD.
 */
"use strict";

const COL = { blue: "#4da3ff", red: "#ff5d5d", amber: "#ffc857", green: "#5ddc8a", dim: "#7b8a9c" };
const ARENA_R = 12000.0;

const state = {
  manifest: null, replay: null, frames: [], t: 0, dur: 1, playing: true, speed: 1,
  cam: { yaw: -0.9, pitch: 0.42, dist: 2600, mode: "chase-blue", target: [0, 0, 4000] },
  lastTS: 0, drag: null, strip: null,
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
  // camera sits behind the look-at point, at (yaw, pitch), distance `dist`
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

/* Filled world-space quadrilateral.  Used for the ground surface; skipped
 * entirely when any corner is behind the camera rather than half-drawn. */
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
  return { t, b: mix("b"), r: mix("r"), g: a.g, hp: a.hp, raw: a };
}

// ------------------------------------------------------------------ drawing
function drawWorld(ctx, cam, s) {
  const W = canvasW(), H = canvasH();
  ctx.clearRect(0, 0, W, H);

  // ---- the ground plane ------------------------------------------------
  // z = 0 is 250 m below the hard floor and 13 km below the ceiling, so the
  // ground is the one reference that makes "up" and "down" unambiguous on a
  // 2-D canvas.  Draw it as an actual filled surface (not just a wire grid),
  // shaded by height above it, then label it.
  const cx = Math.round(cam.eye[0] + cam.fwd[0] * cam.dist / 1000), cy = Math.round(cam.eye[1] + cam.fwd[1] * cam.dist / 1000);
  const step = 2000, span = 16;
  const gx = Math.round(cx / step) * step, gy = Math.round(cy / step) * step;
  const half = span * step;
  fillQuad(ctx, cam,
    [gx - half, gy - half, 0], [gx + half, gy - half, 0],
    [gx + half, gy + half, 0], [gx - half, gy + half, 0],
    "rgba(22,31,45,.72)");
  for (let i = -span; i <= span; i++) {
    const x = gx + i * step, y = gy + i * step;
    drawLine(ctx, [x, gy - half, 0], [x, gy + half, 0], cam, "rgba(60,84,116,.45)", 1);
    drawLine(ctx, [gx - half, y, 0], [gx + half, y, 0], cam, "rgba(60,84,116,.45)", 1);
  }
  // a bold outlined square every 10 km so scale is readable without counting lines
  for (let i = -span; i <= span; i += 5) {
    drawLine(ctx, [gx + i * step, gy - half, 0], [gx + i * step, gy + half, 0], cam, "rgba(96,126,164,.55)", 1.5);
    drawLine(ctx, [gx - half, gy + i * step, 0], [gx + half, gy + i * step, 0], cam, "rgba(96,126,164,.55)", 1.5);
  }
  const gl = project([gx, gy, 0], cam);
  if (gl) {
    ctx.fillStyle = "rgba(150,172,200,.55)";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("GROUND  z = 0 m", gl[0] + 6, gl[1] + 12);
  }
  // arena boundary ring
  ctx.beginPath();
  let started = false;
  for (let a = 0; a <= 360; a += 4) {
    const rad = a * Math.PI / 180;
    const p = project([ARENA_R * Math.cos(rad), ARENA_R * Math.sin(rad), 0], cam);
    if (!p) { started = false; continue; }
    if (!started) { ctx.moveTo(p[0], p[1]); started = true; } else ctx.lineTo(p[0], p[1]);
  }
  ctx.strokeStyle = "rgba(120,150,190,.35)"; ctx.setLineDash([4, 6]); ctx.stroke(); ctx.setLineDash([]);

  // trails
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
  // heading on screen, from the horizontal projection of the velocity
  const ahead = add3(ac.p, mul3([Math.cos(ac.psi * Math.PI / 180), Math.sin(ac.psi * Math.PI / 180), 0], 500));
  const q = project(ahead, cam);
  const ang = q ? Math.atan2(q[1] - y, q[0] - x) : 0;
  const size = 13;

  ctx.save();
  ctx.translate(x, y); ctx.rotate(ang);
  // silhouette: swept-wing dart
  ctx.beginPath();
  ctx.moveTo(size, 0);
  ctx.lineTo(-size * 0.7, size * 0.62);
  ctx.lineTo(-size * 0.35, 0);
  ctx.lineTo(-size * 0.7, -size * 0.62);
  ctx.closePath();
  ctx.fillStyle = color; ctx.globalAlpha = 0.92; ctx.fill();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(0,0,0,.65)"; ctx.lineWidth = 1; ctx.stroke();
  ctx.restore();

  // bank indicator: a wing line rotated about the icon
  const bank = ac.mu * Math.PI / 180;
  ctx.save();
  ctx.translate(x, y); ctx.rotate(ang + bank);
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(-size * 0.95, 0); ctx.lineTo(size * 0.95, 0); ctx.stroke();
  ctx.restore();

  // altitude leader down to the ground, marked so the direction of "down" is
  // explicit: a small caret on the ground surface plus an AGL readout.
  const gp = project([ac.p[0], ac.p[1], 0], cam);
  if (gp) {
    ctx.strokeStyle = "rgba(120,150,190,.30)"; ctx.lineWidth = 1;
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
}

function drawEngagement(ctx, cam, s) {
  // nose reference lines
  for (const [ac, color] of [[s.b, COL.blue], [s.r, COL.red]]) {
    const nose = add3(ac.p, mul3([Math.cos(ac.psi * Math.PI / 180) * Math.cos(ac.gam * Math.PI / 180),
      Math.sin(ac.psi * Math.PI / 180) * Math.cos(ac.gam * Math.PI / 180), Math.sin(ac.gam * Math.PI / 180)], 900));
    drawLine(ctx, ac.p, nose, cam, color + "66", 1.5);
  }
  // gun line + tracer when the trigger is down
  if (s.b.cmd && s.b.cmd[3]) {
    const dir = [Math.cos(s.b.psi * Math.PI / 180) * Math.cos(s.b.gam * Math.PI / 180),
      Math.sin(s.b.psi * Math.PI / 180) * Math.cos(s.b.gam * Math.PI / 180), Math.sin(s.b.gam * Math.PI / 180)];
    const len = 300 + 900 * ((state.t * 3) % 1);
    drawLine(ctx, s.b.p, add3(s.b.p, mul3(dir, len)), cam, COL.amber, 2);
  }
  // line of sight
  drawLine(ctx, s.b.p, s.r.p, cam, "rgba(160,180,210,.35)", 1);
  // weapon-engagement-zone bubble around the target
  const r = s.g ? s.g.rng : 0;
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
  // Blue is our fly (the connectome-constrained controller / learner); red is
  // the opponent.  Show each side's policy name so a replay is self-describing.
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
    `<div class="cmdrow"><span>trigger</span><div class="bar"><span style="left:0;width:${c[3] ? 100 : 0}%"></span></div>
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
  const sel = $("replaySel").value;
  const rp = state.replay;
  if (!rp) return;
  const shown = (rp.events || []).filter((e) => e.t <= state.t + 0.01).slice(-14).reverse();
  $("events").innerHTML = shown.map((e) => {
    const cls = /shot down|kill/i.test(e.msg) ? "kill" : (/hits/.test(e.msg) ? "hit" : "");
    return `<li class="${cls}">[${fmt(e.t, 1)}s] ${e.msg}</li>`;
  }).join("");
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
  // event ticks
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

    // camera target: midpoint, biased toward the aircraft in chase mode
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
  }
  requestAnimationFrame(frame);
}

function setPlaying(p) { state.playing = p; $("play").textContent = p ? "⏸" : "▶"; }

// ------------------------------------------------------------------ loading
async function loadManifest() {
  const res = await fetch("replays/index.json", { cache: "no-store" });
  const man = await res.json();
  state.manifest = man;
  const sel = $("replaySel");
  sel.innerHTML = man.replays.map((r, i) =>
    `<option value="${r.file}">${r.tag} · ${r.result} · ${fmt(r.duration_s, 0)}s · ${r.blue} v ${r.red}</option>`).join("");
  if (man.replays.length) await loadReplay(man.replays[0].file);
}

async function loadReplay(file) {
  const res = await fetch("replays/" + file, { cache: "no-store" });
  const rp = await res.json();
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
}

// -------------------------------------------------------------------- events
function bind() {
  $("play").onclick = () => setPlaying(!state.playing);
  $("speed").onchange = (e) => { state.speed = parseFloat(e.target.value); };
  $("cam").onchange = (e) => { state.cam.mode = e.target.value; };
  $("replaySel").onchange = (e) => loadReplay(e.target.value);
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
}

bind();
loadManifest().catch((err) => {
  $("scenTitle").textContent = "no replay";
  $("scenDesc").textContent = "Run `python -m flybfm fight --replay web/replays/<name>.json` to produce one.";
  console.error(err);
});
requestAnimationFrame(frame);
