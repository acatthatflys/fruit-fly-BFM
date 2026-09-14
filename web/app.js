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

// ------------------------------------------------------------------ drawing
function drawWorld(ctx, cam, s) {
  const W = canvasW(), H = canvasH();
  ctx.clearRect(0, 0, W, H);

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
  ctx.beginPath();
  let started = false;
  for (let a = 0; a <= 360; a += 4) {
    const rad = a * Math.PI / 180;
    const p = project([ARENA_R * Math.cos(rad), ARENA_R * Math.sin(rad), 0], cam);
    if (!p) { started = false; continue; }
    if (!started) { ctx.moveTo(p[0], p[1]); started = true; } else ctx.lineTo(p[0], p[1]);
  }
  ctx.strokeStyle = "rgba(120,150,190,.35)"; ctx.setLineDash([4, 6]); ctx.stroke(); ctx.setLineDash([]);

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
  const size = 13;

  ctx.save();
  ctx.translate(x, y); ctx.rotate(ang);
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

  const bank = ac.mu * Math.PI / 180;
  ctx.save();
  ctx.translate(x, y); ctx.rotate(ang + bank);
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(-size * 0.95, 0); ctx.lineTo(size * 0.95, 0); ctx.stroke();
  ctx.restore();

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

  ctx.fillStyle = "#0e141c"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = "rgba(60,84,116,.25)"; ctx.lineWidth=1;
  for (let i=0;i<=4;i++){ const x = w*0.15 + i*(w*0.5/4); ctx.beginPath(); ctx.moveTo(x, h*0.1); ctx.lineTo(x, h*0.85); ctx.stroke(); }
  for (let i=0;i<=4;i++){ const y = h*0.1 + i*(h*0.75/4); ctx.beginPath(); ctx.moveTo(w*0.15, y); ctx.lineTo(w*0.65, y); ctx.stroke(); }

  const cx = w*0.4, cy = h*0.5;
  const range = 60;
  const sx = cx + roll*range;
  const sy2 = cy + pull*range*0.6;

  ctx.fillStyle = "#16202b"; ctx.beginPath(); ctx.arc(cx, cy, 70, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = "#22303f"; ctx.lineWidth=2; ctx.stroke();
  ctx.strokeStyle = "rgba(123,138,156,.3)"; ctx.beginPath(); ctx.moveTo(cx-70, cy); ctx.lineTo(cx+70, cy); ctx.moveTo(cx, cy-70); ctx.lineTo(cx, cy+70); ctx.stroke();
  ctx.strokeStyle = "#4da3ff"; ctx.lineWidth=4; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(sx, sy2); ctx.stroke();
  ctx.fillStyle = trig ? "#ff5d5d" : "#4da3ff"; ctx.beginPath(); ctx.arc(sx, sy2, 12, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = "#0a0f15"; ctx.lineWidth=2; ctx.stroke();

  const fx = w*0.82, fy = h*0.45;
  const flap = Math.sin(Date.now()*0.02 * (0.5 + thr*2)) * (10 + thr*15);
  ctx.save(); ctx.translate(fx, fy);
  ctx.fillStyle = "#cfd8e3"; ctx.beginPath(); ctx.ellipse(0,0, 10, 22, 0,0,Math.PI*2); ctx.fill();
  ctx.fillStyle = "#ff5d5d"; ctx.beginPath(); ctx.arc(0, -18, 6, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = "rgba(93,220,138,.7)";
  ctx.beginPath(); ctx.ellipse(-14, 2, 18, 6, -0.3 + flap*0.02, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.ellipse(14, 2, 18, 6, 0.3 - flap*0.02, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = "#7b8a9c"; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(-4, -6); ctx.lineTo(cx - fx + (sx-cx)*0.3, cy - fy + (sy2-cy)*0.3); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(4, -6); ctx.lineTo(cx - fx + (sx-cx)*0.7, cy - fy + (sy2-cy)*0.7); ctx.stroke();
  ctx.restore();

  const tx = w*0.92, ty0 = h*0.15, ty1 = h*0.85;
  ctx.fillStyle = "#16202b"; ctx.fillRect(tx-8, ty0, 16, ty1-ty0);
  ctx.strokeStyle = "#22303f"; ctx.strokeRect(tx-8, ty0, 16, ty1-ty0);
  const ty = ty1 - thr*(ty1-ty0);
  ctx.fillStyle = "#5de0ff"; ctx.fillRect(tx-12, ty-6, 24, 12);
  ctx.fillStyle = "#7b8a9c"; ctx.font="10px ui-monospace, monospace"; ctx.fillText("THR", tx-14, ty0-6);
  ctx.fillText(fmt(thr*100,0)+"%", tx-16, ty1+12);

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
  const pop = parseInt($("trainPop").value, 10) || 24;
  const eps = parseInt($("trainEps").value, 10) || 8;
  const totalFights = gens * pop * eps;
  const bench = benchCache;
  let avgFightS = 0.05; // fallback 50ms
  let fightsPerSec = 20;
  let cpuCount = navigator.hardwareConcurrency || 4;
  let platformStr = "";
  if (bench) {
    avgFightS = bench.avg_fight_s || 0.05;
    fightsPerSec = bench.fights_per_second || (1/avgFightS);
    cpuCount = bench.cpu_count || cpuCount;
    platformStr = bench.platform || "";
  }
  // curriculum makes early fights shorter (15-20s) ~0.75x, plus eval overhead ~1.1x
  const estS = totalFights * avgFightS * 0.75 * 1.1 + gens * 2.0; // +2s per gen for eval/live replay
  const benchInfo = bench ? `${(avgFightS*1000).toFixed(0)}ms/fight, ${fightsPerSec.toFixed(1)} fights/s, ${cpuCount} CPUs` : "no bench yet — using 50ms/fight fallback";
  const sysInfo = bench ? `${cpuCount} CPUs · ${platformStr.split('-')[0] || ''} · ${benchInfo}` : `${cpuCount} CPUs (browser) · ${benchInfo}`;
  $("estTime").innerHTML = `⏱ <b>${totalFights} fights</b> (${gens}×${pop}×${eps}) — est <b>${fmtTime(estS)}</b> on this machine (${benchInfo}). ` +
    `Gunnery-focused: 40/30/30 base but curriculum 70/10/20→40/30/30→20/40/40 + opponent level→all + episode 15-20s→60-90s. ` +
    `Prioritizes evals over length. <span class="dim">Longer max_time adds ~linear cost; more eps/pop is cheaper than longer fights for learning.</span>`;
  $("sysSpecs").textContent = sysInfo;
  const def = $("estDefault");
  if (def) def.textContent = `~${fmtTime(16*24*8*0.05*0.75)} on this machine`;
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
  // progress panel — shows best result details
  if (st && (st.history || st.generation !== undefined)) {
    const hist = (st.history || []).slice(-8);
    const lines = hist.map(h => {
      const ev = h.eval ? ` win ${(h.eval.win_rate*100).toFixed(0)}% hit ${(h.eval.hit_rate*100).toFixed(1)}%` : "";
      return `gen ${h.generation} best ${fmt(h.best_return,1)} mean ${fmt(h.mean_return,1)} ${h.max_time_s ? h.max_time_s.toFixed(0)+'s' : ''}${ev}`;
    }).join("\n");
    const poolLines = (st.pool || []).slice(0,4).map(r => `${r.name}: ${r.learner_wins}W ${r.opponent_wins}L`).join(" · ");
    const bestInfo = st.best ? `best score ${fmt(st.best.score,2)} — ${st.best.params ? st.best.params.length + " params" : ""}` : "";
    $("trainProgress").innerHTML = `<b>gen ${st.generation}/${st.total_generations || "?"}</b> best ${fmt(st.best_return,2)} mean ${fmt(st.mean_return,2)}<br>` +
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
