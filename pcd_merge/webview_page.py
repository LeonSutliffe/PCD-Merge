"""The browser walk-through's page (see :mod:`webview`): HTML, CSS and
JavaScript in one string. ``{{TITLE}}``, ``{{META}}`` (JSON) and ``{{DATA}}``
(the point data's script blocks) are filled in when a page is written.

Plain WebGL 1 (works everywhere, WebGL 2 not needed), no libraries and
nothing loaded from the internet, so the file works offline and from a
USB stick or an email attachment. Controls follow the app's walk-through.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>{{TITLE}}</title>
<style>
:root {
  --bg: #121217; --panel: rgba(22, 22, 28, 0.86); --line: rgba(255, 255, 255, 0.12);
  --text: #e8e8ee; --muted: #9a9aa8; --accent: #5ad7ff; --warn: #ff8c19;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; overflow: hidden; }
canvas { position: fixed; inset: 0; width: 100%; height: 100%; display: block; }
#ov { pointer-events: none; }
#gl { touch-action: none; outline: none; }
.bar { position: fixed; left: 8px; right: 8px; display: flex; flex-wrap: wrap; gap: 6px;
  align-items: center; pointer-events: none; }
.bar > * { pointer-events: auto; }
#top { top: 8px; }
#bottom { bottom: 8px; justify-content: space-between; }
.box { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 6px 10px; backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); }
#title { font-weight: 600; max-width: 40ch; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.group { display: flex; gap: 4px; align-items: center; }
button, select { font: inherit; color: var(--text); background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; cursor: pointer; }
button:hover, select:hover { background: rgba(255, 255, 255, 0.12); }
button.on { background: var(--accent); color: #06202b; border-color: var(--accent); }
select option { background: #222; }
label.small { color: var(--muted); font-size: 12px; }
input[type=range] { width: 80px; accent-color: var(--accent); }
#hint { color: var(--muted); font-size: 13px; max-width: 70%; white-space: pre-wrap; }
#info { color: var(--muted); font-size: 12px; text-align: right; white-space: nowrap; }
#meas { position: fixed; left: 8px; bottom: 52px; max-width: min(360px, calc(100% - 16px));
  max-height: 40vh; overflow: auto; display: none; }
#meas h3 { margin: 0 0 4px; font-size: 13px; color: var(--muted); font-weight: 600; }
#meas .row { display: flex; gap: 8px; align-items: baseline; padding: 3px 0;
  border-top: 1px solid var(--line); }
#meas .row:first-of-type { border-top: 0; }
#meas .v { font-weight: 600; color: var(--accent); white-space: nowrap; }
#meas .d { color: var(--muted); font-size: 12px; flex: 1; }
#meas button { padding: 0 6px; font-size: 12px; }
#help { position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
  background: rgba(0, 0, 0, 0.5); }
#help .box { max-width: 560px; margin: 16px; padding: 16px 20px; max-height: 90vh; overflow: auto; }
#help h2 { margin: 0 0 8px; font-size: 17px; }
#help table { border-collapse: collapse; width: 100%; }
#help td { padding: 3px 8px 3px 0; vertical-align: top; }
#help td:first-child { white-space: nowrap; color: var(--accent); font-weight: 600; }
#help p { color: var(--muted); margin: 10px 0 0; font-size: 13px; }
#loading { position: fixed; inset: 0; display: flex; flex-direction: column; gap: 12px;
  align-items: center; justify-content: center; background: var(--bg); }
#loading .track { width: min(320px, 80vw); height: 6px; background: var(--line); border-radius: 3px; }
#loading .fill { height: 100%; width: 0; background: var(--accent); border-radius: 3px;
  transition: width 0.15s; }
#loading .msg { color: var(--muted); }
#cross { position: fixed; left: 50%; top: 50%; width: 22px; height: 22px; margin: -11px 0 0 -11px;
  display: none; pointer-events: none; }
@media (max-width: 700px) {
  #title { max-width: 100%; flex-basis: 100%; }
  #hint { display: none; }
  .wide { display: none; }
}
</style>
</head>
<body>
<canvas id="gl" tabindex="0"></canvas>
<canvas id="ov"></canvas>
<svg id="cross" viewBox="-11 -11 22 22"><g stroke="white" stroke-width="1.5" opacity="0.85">
<line x1="-10" y1="0" x2="-4" y2="0"/><line x1="4" y1="0" x2="10" y2="0"/>
<line x1="0" y1="-10" x2="0" y2="-4"/><line x1="0" y1="4" x2="0" y2="10"/></g></svg>

<div id="top" class="bar">
  <div id="title" class="box"></div>
  <div class="box group">
    <button id="bWalk" class="on" title="Walk through the site (W A S D and the mouse)">Walk</button>
    <button id="bOrbit" title="Look at the whole site from outside (O)">Overview</button>
  </div>
  <div class="box group">
    <button id="bMeasure" title="Click two points to measure the distance between them (M)">Measure</button>
  </div>
  <div class="box group wide">
    <label class="small" for="sColour">Colour</label><select id="sColour"></select>
    <label class="small" for="sDetail">Detail</label><select id="sDetail"></select>
    <label class="small" for="rSize">Size</label><input id="rSize" type="range" min="5" max="30" value="16">
  </div>
  <div class="box group">
    <select id="sJump" title="Jump to where a scan was taken (keys 1-9)"></select>
    <button id="bStart" title="Back to the start (Home)">Start</button>
    <button id="bHelp" title="Controls (H)">?</button>
  </div>
</div>
<div id="bottom" class="bar">
  <div id="hint" class="box"></div>
  <div id="info" class="box"></div>
</div>
<div id="meas" class="box"></div>
<div id="help"><div class="box">
  <h2>Walking through the site</h2>
  <table>
    <tr><td>Click the view</td><td>Look around with the mouse (Esc gives the mouse back). Or drag.</td></tr>
    <tr><td>W A S D / arrows</td><td>Move</td></tr>
    <tr><td>Space or E / C or Q</td><td>Up / down</td></tr>
    <tr><td>Shift</td><td>Hold to go 4 times faster</td></tr>
    <tr><td>Mouse wheel</td><td>Walking speed (Overview: zoom)</td></tr>
    <tr><td>1 to 9</td><td>Jump to where a scan was taken</td></tr>
    <tr><td>Home</td><td>Back to the start</td></tr>
    <tr><td>O</td><td>Overview of the whole site, and back</td></tr>
    <tr><td>M</td><td>Measure: click two points. Esc cancels the first point.</td></tr>
    <tr><td>Touch</td><td>One finger looks around, two fingers move (Overview: rotate, pinch to zoom)</td></tr>
  </table>
  <p id="about"></p>
  <p style="text-align:right;margin-top:12px"><button id="bHelpClose">Close</button></p>
</div></div>
<div id="loading"><div class="msg" id="loadMsg">Loading the points…</div>
  <div class="track"><div class="fill" id="loadFill"></div></div></div>

<script type="application/json" id="meta">{{META}}</script>
{{DATA}}
<script>
"use strict";
(function () {
const META = JSON.parse(document.getElementById("meta").textContent);
const $ = (id) => document.getElementById(id);
const canvas = $("gl"), ov = $("ov"), octx = ov.getContext("2d");
const BG = [0.07, 0.07, 0.09];
const LOOK = 0.15;               // degrees per pixel
const LEVELS = 12;
const COARSE = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
const DETAIL = { "Low": [3.0, 1000000], "Medium": [2.0, 2500000], "High": [1.4, 5000000],
                 "Maximum": [1.0, 12000000] };

document.title = META.title || "Walk-through";
$("title").textContent = META.title || "Walk-through";
$("about").textContent = [META.note, "Made " + META.made + " with " + META.app + ".",
  "Points shown: " + (META.points / 1e6).toFixed(1) + " million."].filter(Boolean).join(" ");

function fail(msg) {
  $("loadMsg").textContent = msg;
  $("loading").style.display = "flex";
  document.querySelector("#loading .track").style.display = "none";
}

// ------------------------------------------------------------------ data

let C = 0, N = 0, table, counts, starts, q, rgb = null, inten = null;

async function loadData() {
  const blocks = Array.from(document.querySelectorAll("script.pts"));
  const parts = [];
  for (let i = 0; i < blocks.length; i++) {
    const s = atob(blocks[i].textContent.trim());
    const u = new Uint8Array(s.length);
    for (let k = 0; k < s.length; k++) u[k] = s.charCodeAt(k);
    parts.push(u);
    blocks[i].textContent = "";
    $("loadFill").style.width = (60 * (i + 1) / blocks.length) + "%";
    await new Promise((r) => setTimeout(r, 0));
  }
  if (typeof DecompressionStream === "undefined")
    throw new Error("This browser is too old to open the walk-through. Please use a current " +
                    "Chrome, Edge, Firefox or Safari.");
  $("loadMsg").textContent = "Unpacking…";
  const stream = new Blob(parts).stream().pipeThrough(new DecompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  $("loadFill").style.width = "85%";
  C = META.chunks; N = META.points;
  const pad4 = (n) => n + ((4 - n % 4) % 4);
  let off = 0;
  table = new Float32Array(buf, off, C * 11); off += C * 44;
  counts = new Uint32Array(buf, off, C); off += C * 4;
  q = new Uint16Array(buf, off, N * 3); off += pad4(N * 6);
  if (META.has_rgb) { rgb = new Uint8Array(buf, off, N * 3); off += pad4(N * 3); }
  if (META.has_intensity) { inten = new Uint8Array(buf, off, N); off += pad4(N); }
  starts = new Uint32Array(C);
  for (let i = 1; i < C; i++) starts[i] = starts[i - 1] + counts[i - 1];
}

// ------------------------------------------------------------------ maths (row-major 4x4)

function mul(a, b) {
  const o = new Float64Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
    o[r * 4 + c] = s;
  }
  return o;
}
function colMajor(m) {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) o[c * 4 + r] = m[r * 4 + c];
  return o;
}
const rad = (d) => d * Math.PI / 180;
const norm = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// ------------------------------------------------------------------ camera

const cam = { pos: META.start.pos.slice(), yaw: META.start.yaw, pitch: META.start.pitch, fov: 75,
              near: Math.max(META.near, 0.01), far: META.far };
const lo = META.lo, hi = META.hi;
const centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
let mode = "walk";                    // or "orbit"
const orbit = { target: centre.slice(), dist: diag * 0.9 };
let speed = META.speed;

function forward() {
  const y = rad(cam.yaw), p = rad(cam.pitch);
  return [Math.cos(p) * Math.cos(y), Math.cos(p) * Math.sin(y), Math.sin(p)];
}
function rightVec() { const y = rad(cam.yaw); return [Math.sin(y), -Math.cos(y), 0]; }
function look(dx, dy) {
  cam.yaw = ((cam.yaw - dx) % 360 + 360) % 360;
  cam.pitch = Math.max(-89, Math.min(89, cam.pitch - dy));
  if (mode === "orbit") placeOrbit();
}
function lookAt(t) {
  const d = [t[0] - cam.pos[0], t[1] - cam.pos[1], t[2] - cam.pos[2]];
  cam.yaw = (Math.atan2(d[1], d[0]) * 180 / Math.PI + 360) % 360;
  cam.pitch = Math.atan2(d[2], Math.hypot(d[0], d[1])) * 180 / Math.PI;
}
function placeOrbit() {
  const f = forward();
  for (let k = 0; k < 3; k++) cam.pos[k] = orbit.target[k] - f[k] * orbit.dist;
}
function viewMatrix() {
  const f = forward(), s = norm(cross(f, [0, 0, 1])), u = cross(s, f), p = cam.pos;
  return Float64Array.from([s[0], s[1], s[2], -dot(s, p), u[0], u[1], u[2], -dot(u, p),
    -f[0], -f[1], -f[2], dot(f, p), 0, 0, 0, 1]);
}
function projMatrix(aspect) {
  const t = 1 / Math.tan(rad(cam.fov) / 2), n = cam.near, f = cam.far;
  return Float64Array.from([t / aspect, 0, 0, 0, 0, t, 0, 0,
    0, 0, (f + n) / (n - f), 2 * f * n / (n - f), 0, 0, -1, 0]);
}

// ------------------------------------------------------------------ WebGL

const gl = canvas.getContext("webgl", { antialias: false, alpha: false, preserveDrawingBuffer: false })
  || canvas.getContext("experimental-webgl");

const VS = `
attribute vec3 a_q; attribute vec3 a_rgb; attribute float a_int;
uniform mat4 u_mvp; uniform mat4 u_mv;
uniform vec3 u_lo; uniform vec3 u_span;
uniform float u_size; uniform float u_focal; uniform float u_max; uniform float u_fog;
uniform float u_zlo; uniform float u_zhi; uniform float u_mode; uniform vec3 u_bg;
varying vec3 v_col;
vec3 ramp(float t) {
  vec3 c0 = vec3(40., 60., 160.), c1 = vec3(40., 150., 200.), c2 = vec3(80., 200., 110.),
       c3 = vec3(235., 210., 60.), c4 = vec3(230., 80., 50.);
  vec3 c = t < 0.25 ? mix(c0, c1, t / 0.25) : t < 0.5 ? mix(c1, c2, (t - 0.25) / 0.25)
         : t < 0.75 ? mix(c2, c3, (t - 0.5) / 0.25) : mix(c3, c4, (t - 0.75) / 0.25);
  return c / 255.;
}
void main() {
  vec4 p = vec4(u_lo + a_q * u_span, 1.0);
  gl_Position = u_mvp * p;
  float d = max(-(u_mv * p).z, 1e-3);
  gl_PointSize = clamp(u_size * u_focal / d, 1.0, u_max);
  vec3 c;
  if (u_mode < 0.5) c = a_rgb;
  else if (u_mode < 1.5) c = vec3(30. + 225. * a_int, 28. + 222. * a_int, 26. + 214. * a_int) / 255.;
  else c = ramp(clamp((p.z - u_zlo) / max(u_zhi - u_zlo, 1e-6), 0., 1.));
  float f = clamp(d / u_fog, 0., 1.);
  v_col = mix(c, u_bg, 0.8 * f * f);
}`;
const FS = `
precision mediump float;
varying vec3 v_col;
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  if (dot(q, q) > 1.0) discard;
  gl_FragColor = vec4(v_col, 1.0);
}`;

let prog, U = {}, maxPoint = 64;
function setupGL() {
  const sh = (src, kind) => {
    const s = gl.createShader(kind);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  prog = gl.createProgram();
  gl.attachShader(prog, sh(VS, gl.VERTEX_SHADER));
  gl.attachShader(prog, sh(FS, gl.FRAGMENT_SHADER));
  gl.bindAttribLocation(prog, 0, "a_q");
  gl.bindAttribLocation(prog, 1, "a_rgb");
  gl.bindAttribLocation(prog, 2, "a_int");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  for (const n of ["u_mvp", "u_mv", "u_lo", "u_span", "u_size", "u_focal", "u_max", "u_fog",
                   "u_zlo", "u_zhi", "u_mode", "u_bg"]) U[n] = gl.getUniformLocation(prog, n);
  maxPoint = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1] || 64;
  const buffer = (data, loc, size, type) => {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, type, true, 0, 0);
  };
  buffer(q, 0, 3, gl.UNSIGNED_SHORT);
  if (rgb) buffer(rgb, 1, 3, gl.UNSIGNED_BYTE); else gl.vertexAttrib3f(1, 0.8, 0.8, 0.8);
  if (inten) buffer(inten, 2, 1, gl.UNSIGNED_BYTE); else gl.vertexAttrib1f(2, 0.7);
  if (gl.getError() === gl.OUT_OF_MEMORY) throw new Error("The graphics card ran out of memory.");
  gl.useProgram(prog);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(BG[0], BG[1], BG[2], 1);
}

// ------------------------------------------------------------------ drawing

let pxTarget = 1.4, maxBudget = 5000000, budget = 5000000, pointScale = 1.6, colourMode = 2;
let moving = false, lastInput = 0, lastFrame = 0, fps = 0, drawn = 0, dirty = true;
let W = 1, H = 1, DPR = 1, MVP = null, MV = null;

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = Math.max(1, Math.round(innerWidth * DPR)); H = Math.max(1, Math.round(innerHeight * DPR));
  canvas.width = W; canvas.height = H; ov.width = W; ov.height = H;
  dirty = true;
}

function frustum(m) {
  const P = [];
  for (const [a, s] of [[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1]]) {
    const p = [m[12] + s * m[a * 4], m[13] + s * m[a * 4 + 1], m[14] + s * m[a * 4 + 2], m[15] + s * m[a * 4 + 3]];
    const l = Math.hypot(p[0], p[1], p[2]) || 1;
    P.push([p[0] / l, p[1] / l, p[2] / l, p[3] / l]);
  }
  return P;
}

function draw() {
  const now = performance.now();
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  if (dt > 0 && dt < 1) {
    fps = fps ? 0.9 * fps + 0.1 / dt : 1 / dt;
    if (moving) {
      if (dt > 1 / 28) budget = Math.max(200000, Math.floor(budget * 0.8));
      else if (dt < 1 / 50) budget = Math.min(maxBudget, Math.floor(budget * 1.15));
    }
  }
  gl.viewport(0, 0, W, H);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  MV = viewMatrix();
  MVP = mul(projMatrix(W / H), MV);
  const focal = H / 2 / Math.tan(rad(cam.fov) / 2);
  gl.uniformMatrix4fv(U.u_mvp, false, colMajor(MVP));
  gl.uniformMatrix4fv(U.u_mv, false, colMajor(MV));
  gl.uniform1f(U.u_focal, focal);
  gl.uniform1f(U.u_max, Math.min(maxPoint, 40 * DPR));
  gl.uniform1f(U.u_fog, diag * 1.2);
  gl.uniform1f(U.u_zlo, lo[2]); gl.uniform1f(U.u_zhi, hi[2]);
  gl.uniform1f(U.u_mode, colourMode);
  gl.uniform3f(U.u_bg, BG[0], BG[1], BG[2]);

  // which chunks, and how much of each (as in the app: flyscene.plan_frame)
  const P = frustum(MVP), eye = cam.pos, use = moving ? budget : maxBudget;
  const vis = [], frac = [];
  let total = 0;
  for (let i = 0; i < C; i++) {
    const o = i * 11, cx = table[o + 6], cy = table[o + 7], cz = table[o + 8], r = table[o + 9];
    let inside = true;
    for (let k = 0; k < 6; k++) {
      const p = P[k];
      if (p[0] * cx + p[1] * cy + p[2] * cz + p[3] < -r) { inside = false; break; }
    }
    if (!inside) continue;
    const d = Math.max(Math.hypot(cx - eye[0], cy - eye[1], cz - eye[2]) - r, 1e-3);
    const want = d * pxTarget / focal, sp = Math.max(table[o + 10], 1e-9);
    const f = Math.min(1, (sp / want) * (sp / want));
    vis.push(i); frac.push(f); total += f * counts[i];
  }
  const scale = total > use ? use / total : 1;
  drawn = 0;
  for (let j = 0; j < vis.length; j++) {
    const i = vis[j], o = i * 11;
    const f = Math.min(1, frac[j] * scale);
    const lv = Math.max(0, Math.min(LEVELS - 1, Math.round(-Math.log2(Math.max(f, Math.pow(2, -(LEVELS - 1)))))));
    const n = Math.max(1, Math.ceil(counts[i] * Math.pow(0.5, lv)));
    gl.uniform3f(U.u_lo, table[o], table[o + 1], table[o + 2]);
    gl.uniform3f(U.u_span, table[o + 3] * 65535, table[o + 4] * 65535, table[o + 5] * 65535);
    gl.uniform1f(U.u_size, table[o + 10] * Math.sqrt(Math.pow(2, lv)) * pointScale);
    gl.drawArrays(gl.POINTS, starts[i], n);
    drawn += n;
  }
  drawOverlay();
  const txt = (drawn / 1e6).toFixed(1) + " M of " + (N / 1e6).toFixed(1) + " M points drawn";
  $("info").textContent = moving && fps ? txt + "   " + fps.toFixed(0) + " fps" : txt;
}

// ------------------------------------------------------------------ measuring

const measurements = (META.measurements || []).map((m) => ({ name: m.name, kind: m.kind, points: m.points, saved: true }));
let measuring = false, pending = null;

function project(p) {
  const m = MVP, x = p[0], y = p[1], z = p[2];
  const w = m[12] * x + m[13] * y + m[14] * z + m[15];
  if (w <= cam.near * 0.5) return null;
  return [((m[0] * x + m[1] * y + m[2] * z + m[3]) / w + 1) / 2 * W,
          (1 - (m[4] * x + m[5] * y + m[6] * z + m[7]) / w) / 2 * H, w];
}
function clipSegment(a, b) {
  const m = MVP, near = cam.near;
  const wOf = (p) => m[12] * p[0] + m[13] * p[1] + m[14] * p[2] + m[15];
  const wa = wOf(a), wb = wOf(b);
  if (wa <= near && wb <= near) return null;
  if (wa <= near || wb <= near) {
    const t = (near - wa) / (wb - wa), mid = [0, 1, 2].map((k) => a[k] + t * (b[k] - a[k]));
    if (wa <= near) a = mid; else b = mid;
  }
  const pa = project(a), pb = project(b);
  return pa && pb ? [pa, pb] : null;
}
const fmt = (v) => v.toFixed(3) + " m";
function describe(m) {
  if (m.kind === "point") {
    const o = META.origin, p = m.points[0];
    return ["(" + [0, 1, 2].map((k) => (p[k] + o[k]).toFixed(3)).join(", ") + ")", ""];
  }
  const a = m.points[0], b = m.points[1], d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const len = Math.hypot(d[0], d[1], d[2]), hz = Math.hypot(d[0], d[1]);
  const slope = len ? Math.atan2(d[2], hz) * 180 / Math.PI : 0;
  return [fmt(len), "horizontal " + fmt(hz) + ", height " + (d[2] >= 0 ? "+" : "") + d[2].toFixed(3) +
          " m, slope " + (slope >= 0 ? "+" : "") + slope.toFixed(1) + "°"];
}

function drawOverlay() {
  octx.setTransform(1, 0, 0, 1, 0, 0);
  octx.clearRect(0, 0, W, H);
  octx.lineWidth = 2 * DPR;
  octx.font = "600 " + (13 * DPR) + "px system-ui, sans-serif";
  for (const m of measurements) {
    const pts = m.points;
    if (m.kind === "distance") {
      const seg = clipSegment(pts[0], pts[1]);
      if (seg) {
        octx.strokeStyle = "#5ad7ff";
        octx.beginPath(); octx.moveTo(seg[0][0], seg[0][1]); octx.lineTo(seg[1][0], seg[1][1]); octx.stroke();
      }
    }
    octx.fillStyle = "#5ad7ff";
    for (const p of pts) {
      const s = project(p);
      if (s) { octx.beginPath(); octx.arc(s[0], s[1], 4 * DPR, 0, 7); octx.fill(); }
    }
    const mid = [0, 1, 2].map((k) => pts.reduce((a, p) => a + p[k], 0) / pts.length);
    const s = project(mid);
    if (s) {
      const text = (m.name ? m.name + ": " : "") + describe(m)[0];
      const w = octx.measureText(text).width;
      octx.fillStyle = "rgba(10, 12, 16, 0.75)";
      octx.fillRect(s[0] + 6 * DPR, s[1] - 20 * DPR, w + 10 * DPR, 18 * DPR);
      octx.fillStyle = "#c8f0ff";
      octx.fillText(text, s[0] + 11 * DPR, s[1] - 6 * DPR);
    }
  }
  if (pending) {
    const s = project(pending);
    if (s) {
      octx.fillStyle = "#ff8c19";
      octx.beginPath(); octx.arc(s[0], s[1], 5.5 * DPR, 0, 7); octx.fill();
      if (mouse && !locked()) {
        octx.strokeStyle = "#ff8c19"; octx.setLineDash([6 * DPR, 5 * DPR]);
        octx.beginPath(); octx.moveTo(s[0], s[1]); octx.lineTo(mouse[0] * DPR, mouse[1] * DPR); octx.stroke();
        octx.setLineDash([]);
      }
    }
  }
}

function pick(sx, sy) {
  // the nearest point within 7 px of (sx, sy) (CSS pixels), or null
  const m = MVP, px = sx * DPR, py = sy * DPR, R = 7 * DPR, R2 = R * R;
  const focal = H / 2 / Math.tan(rad(cam.fov) / 2);
  let best = null, bestW = Infinity;
  for (let i = 0; i < C; i++) {
    const o = i * 11, c = [table[o + 6], table[o + 7], table[o + 8]], r = table[o + 9];
    const w = m[12] * c[0] + m[13] * c[1] + m[14] * c[2] + m[15];
    if (w > r) {
      const s = project(c);
      if (!s) continue;
      if (Math.hypot(s[0] - px, s[1] - py) > r * focal / w + R) continue;
    } else if (w < -r) continue;
    const lx = table[o], ly = table[o + 1], lz = table[o + 2];
    const tx = table[o + 3], ty = table[o + 4], tz = table[o + 5];
    const end = starts[i] + counts[i];
    for (let v = starts[i]; v < end; v++) {
      const x = lx + q[v * 3] * tx, y = ly + q[v * 3 + 1] * ty, z = lz + q[v * 3 + 2] * tz;
      const pw = m[12] * x + m[13] * y + m[14] * z + m[15];
      if (pw <= cam.near || pw >= bestW) continue;
      const qx = ((m[0] * x + m[1] * y + m[2] * z + m[3]) / pw + 1) / 2 * W - px;
      const qy = (1 - (m[4] * x + m[5] * y + m[6] * z + m[7]) / pw) / 2 * H - py;
      if (qx * qx + qy * qy <= R2) { best = [x, y, z]; bestW = pw; }
    }
  }
  return best;
}

function measureAt(sx, sy) {
  const p = pick(sx, sy);
  if (!p) { setHint("No point there: click on the scan itself."); return; }
  if (!pending) {
    pending = p;
    setHint("Now click the second point.   Esc: start again");
  } else {
    const m = { name: "", kind: "distance", points: [pending, p] };
    measurements.push(m);
    pending = null;
    setHint("Measured " + describe(m)[0] + ". Click two more points for another.");
    refreshList();
  }
  dirty = true;
}

function refreshList() {
  const box = $("meas");
  box.innerHTML = "";
  if (!measurements.length) { box.style.display = "none"; return; }
  box.style.display = "block";
  const h = document.createElement("h3");
  h.textContent = "Measurements";
  box.appendChild(h);
  measurements.forEach((m, i) => {
    const row = document.createElement("div");
    row.className = "row";
    const [v, d] = describe(m);
    const a = document.createElement("span"); a.className = "v"; a.textContent = (m.name ? m.name + ": " : "") + v;
    const b = document.createElement("span"); b.className = "d"; b.textContent = d;
    const x = document.createElement("button"); x.textContent = "×"; x.title = "Remove";
    x.onclick = () => { measurements.splice(i, 1); refreshList(); dirty = true; x.blur(); };
    row.append(a, b, x);
    box.appendChild(row);
  });
}

// ------------------------------------------------------------------ input

const keys = new Set();
let mouse = null, drag = null, dragMoved = false, touches = new Map(), pinch = null;
const locked = () => document.pointerLockElement === canvas;
const hintText = () => {
  if (measuring) return locked() ? "Click: measure at the cross   W A S D: move   Esc: free the mouse"
    : "Click two points to measure   Drag: " + (mode === "orbit" ? "rotate" : "look around") + "   M: stop measuring";
  if (mode === "orbit") return "Drag: rotate   Right-drag: move sideways   Wheel: zoom   O: walk again";
  return locked() ? "W A S D: move   Mouse: look   Space/C: up/down   Shift: faster   Wheel: speed   Esc: free the mouse"
    : (COARSE ? "One finger: look around   Two fingers: move   ? for help"
              : "Click the view to look around   W A S D: move   Space/C: up/down   ? for help");
};
function setHint(t) { $("hint").textContent = t || hintText(); }
function poke() {
  lastInput = performance.now();
  if (!moving) { moving = true; budget = maxBudget; }
  dirty = true;
}

function tick(dt) {
  const k = keys;
  const fwd = (k.has("KeyW") || k.has("ArrowUp")) - (k.has("KeyS") || k.has("ArrowDown"));
  const rt = (k.has("KeyD") || k.has("ArrowRight")) - (k.has("KeyA") || k.has("ArrowLeft"));
  const up = (k.has("Space") || k.has("KeyE") || k.has("PageUp")) - (k.has("KeyC") || k.has("KeyQ") || k.has("PageDown"));
  if (!(fwd || rt || up)) return false;
  let v = speed * (k.has("ShiftLeft") || k.has("ShiftRight") ? 4 : 1) * dt;
  if (fwd && rt) v /= Math.SQRT2;
  if (mode === "orbit") {  // move what the overview turns around
    const r = rightVec(), f = forward(), h = norm([f[0], f[1], 0]), k = v * 3;
    for (let i = 0; i < 3; i++) orbit.target[i] += fwd * k * h[i] + rt * k * r[i] + (i === 2 ? up * k : 0);
    placeOrbit();
    return true;
  }
  let f = forward();
  f = norm([f[0], f[1], 0]);
  const r = rightVec();
  for (let i = 0; i < 3; i++) cam.pos[i] += fwd * v * f[i] + rt * v * r[i] + (i === 2 ? up * v : 0);
  return true;
}

window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "SELECT" || e.target.tagName === "INPUT")) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const c = e.code;
  if (c === "Escape") { if (pending) { pending = null; setHint(); dirty = true; } if ($("help").style.display === "flex") showHelp(false); return; }
  if (c === "KeyM") { setMeasuring(!measuring); return; }
  if (c === "KeyO") { setMode(mode === "walk" ? "orbit" : "walk"); return; }
  if (c === "KeyH" || e.key === "?") { showHelp($("help").style.display !== "flex"); return; }
  if (c === "Home") { goStart(); return; }
  if (/^Digit[1-9]$/.test(c)) { jump(+c.slice(5) - 1); return; }
  if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown"].includes(c)) e.preventDefault();
  keys.add(c);
  poke();
});
window.addEventListener("keyup", (e) => keys.delete(e.code));
window.addEventListener("blur", () => keys.clear());

canvas.addEventListener("mousedown", (e) => {
  canvas.focus();
  if (locked()) { if (measuring && e.button === 0) measureAt(innerWidth / 2, innerHeight / 2); return; }
  drag = [e.clientX, e.clientY, e.button]; dragMoved = false;
});
window.addEventListener("mouseup", (e) => {
  if (!drag) return;
  const d = drag; drag = null;
  if (dragMoved || d[2] !== 0) return;
  if (measuring) measureAt(e.clientX, e.clientY);
  else if (mode === "walk" && canvas.requestPointerLock) canvas.requestPointerLock();
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
window.addEventListener("mousemove", (e) => {
  mouse = [e.clientX, e.clientY];
  if (locked()) { look(e.movementX * LOOK, e.movementY * LOOK); poke(); return; }
  if (!drag) { if (pending) dirty = true; return; }
  const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
  if (Math.abs(dx) + Math.abs(dy) > 3) dragMoved = true;
  if (!dragMoved) return;
  drag[0] = e.clientX; drag[1] = e.clientY;
  if (mode === "orbit" && (drag[2] === 2 || e.shiftKey)) pan(dx, dy); else look(dx * LOOK, dy * LOOK);
  poke();
});
document.addEventListener("pointerlockchange", () => { setHint(); $("cross").style.display = locked() ? "block" : "none"; dirty = true; });
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const notches = -Math.sign(e.deltaY) * Math.min(3, Math.abs(e.deltaY) / 100 || 1);
  if (mode === "orbit") { orbit.dist = Math.max(0.3, orbit.dist / Math.pow(1.15, notches)); placeOrbit(); }
  else { speed = Math.max(0.01, speed * Math.pow(1.25, notches)); setHint("Speed " + speed.toFixed(2) + " m/s"); }
  poke();
}, { passive: false });

function pan(dx, dy) {
  const r = rightVec(), f = forward(), s = orbit.dist * 0.0015;
  const u = cross(r, f);
  for (let i = 0; i < 3; i++) orbit.target[i] += (-dx * r[i] + dy * u[i]) * s;
  placeOrbit();
}

// touch: one finger looks (Overview: rotates), two move (Overview: pinch zooms)
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  for (const t of e.changedTouches) touches.set(t.identifier, [t.clientX, t.clientY, t.clientX, t.clientY, performance.now()]);
  pinch = null;
}, { passive: false });
canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  const before = new Map(Array.from(touches, ([k, v]) => [k, v.slice()]));
  for (const t of e.changedTouches) { const v = touches.get(t.identifier); if (v) { v[0] = t.clientX; v[1] = t.clientY; } }
  const list = Array.from(touches.values());
  if (list.length === 1) {
    const id = touches.keys().next().value, a = before.get(id), b = touches.get(id);
    look((b[0] - a[0]) * LOOK * 1.3, (b[1] - a[1]) * LOOK * 1.3);
  } else if (list.length >= 2) {
    const [a, b] = list;
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (pinch) {
      const dx = mx - pinch[0], dy = my - pinch[1];
      if (mode === "orbit") { orbit.dist = Math.max(0.3, orbit.dist * pinch[2] / Math.max(dist, 1)); pan(dx, dy); placeOrbit(); }
      else {
        let f = forward(); f = norm([f[0], f[1], 0]);
        const r = rightVec(), k = speed * 0.01;
        for (let i = 0; i < 3; i++) cam.pos[i] += -dy * k * f[i] + dx * k * r[i];
        cam.pos[2] += (dist - pinch[2]) * k * 0.5;
      }
    }
    pinch = [mx, my, dist];
  }
  poke();
}, { passive: false });
canvas.addEventListener("touchend", (e) => {
  for (const t of e.changedTouches) {
    const v = touches.get(t.identifier);
    touches.delete(t.identifier);
    if (v && measuring && touches.size === 0 && Math.hypot(v[0] - v[2], v[1] - v[3]) < 8 && performance.now() - v[4] < 400)
      measureAt(v[0], v[1]);
  }
  pinch = null;
});

// ------------------------------------------------------------------ controls

function setMode(m) {
  if (m === mode) return;
  if (locked()) document.exitPointerLock();
  mode = m;
  if (m === "orbit") {
    orbit.target = centre.slice(); orbit.dist = diag * 0.9;
    if (cam.pitch > -15) cam.pitch = -35;
    placeOrbit();
  }
  $("bWalk").classList.toggle("on", m === "walk");
  $("bOrbit").classList.toggle("on", m === "orbit");
  setHint(); poke();
}
function setMeasuring(on) {
  measuring = on; pending = null;
  $("bMeasure").classList.toggle("on", on);
  canvas.style.cursor = on ? "crosshair" : "";
  setHint(); dirty = true;
}
function goStart() {
  if (mode !== "walk") setMode("walk");
  cam.pos = META.start.pos.slice(); cam.yaw = META.start.yaw; cam.pitch = META.start.pitch; poke();
}
function jump(i) {
  const s = (META.scans || [])[i];
  if (!s) return;
  if (mode !== "walk") setMode("walk");
  cam.pos = s[1].slice(); poke();
}
function showHelp(on) { $("help").style.display = on ? "flex" : "none"; }

function controls() {
  const colours = [];
  if (META.has_rgb) colours.push(["Colour", 0]);
  if (META.has_intensity) colours.push(["Intensity", 1]);
  colours.push(["Height", 2]);
  for (const [n, v] of colours) $("sColour").add(new Option(n, v));
  colourMode = colours[0][1];
  $("sColour").onchange = (e) => { colourMode = +e.target.value; dirty = true; e.target.blur(); };
  for (const n of Object.keys(DETAIL)) $("sDetail").add(new Option(n, n));
  const start = COARSE ? "Medium" : "High";
  $("sDetail").value = start;
  [pxTarget, maxBudget] = DETAIL[start]; budget = maxBudget;
  $("sDetail").onchange = (e) => { [pxTarget, maxBudget] = DETAIL[e.target.value]; budget = maxBudget; dirty = true; e.target.blur(); };
  $("rSize").oninput = (e) => { pointScale = e.target.value / 10; dirty = true; };
  $("rSize").onchange = (e) => e.target.blur();
  const scans = META.scans || [];
  $("sJump").add(new Option(scans.length ? "Jump to a scan…" : "No scan positions", ""));
  scans.forEach((s, i) => $("sJump").add(new Option((i < 9 ? i + 1 + ": " : "") + s[0], i)));
  $("sJump").disabled = !scans.length;
  $("sJump").onchange = (e) => { if (e.target.value !== "") jump(+e.target.value); e.target.value = ""; e.target.blur(); };
  const btn = (id, f) => { $(id).onclick = (e) => { f(); e.currentTarget.blur(); canvas.focus(); }; };
  btn("bWalk", () => setMode("walk"));
  btn("bOrbit", () => setMode("orbit"));
  btn("bMeasure", () => setMeasuring(!measuring));
  btn("bStart", goStart);
  btn("bHelp", () => showHelp(true));
  btn("bHelpClose", () => showHelp(false));
  $("help").onclick = (e) => { if (e.target === $("help")) showHelp(false); };
}

function loop(now) {
  const dt = Math.min(0.1, (now - (loop.last || now)) / 1000);
  loop.last = now;
  if (tick(dt)) { lastInput = now; dirty = true; moving = true; }
  if (moving && now - lastInput > 250 && !keys.size) { moving = false; dirty = true; }
  if (dirty) { dirty = false; draw(); }
  requestAnimationFrame(loop);
}

async function main() {
  if (!gl) { fail("This browser or computer cannot show 3D (WebGL is off or not available)."); return; }
  try { await loadData(); } catch (e) { fail("The points could not be loaded: " + e.message); return; }
  $("loadMsg").textContent = "Sending the points to the graphics card…";
  await new Promise((r) => setTimeout(r, 0));
  try { setupGL(); } catch (e) { fail("The 3D view could not start: " + e.message); return; }
  controls();
  resize();
  window.addEventListener("resize", resize);
  refreshList();
  setHint();
  $("loading").style.display = "none";
  canvas.focus();
  requestAnimationFrame(loop);
  window.__walk = { cam, pick, measurements, get drawn() { return drawn; }, draw, setMode,
                    setMeasuring, measureAt, get mode() { return mode; } };
}
main();
})();
</script>
</body>
</html>
"""
