const $ = (id) => document.getElementById(id);

const state = {
  jobId: null,
  mesh: null,
  labels: new Map(),
  rotation: [-0.55, 0.72],
  pan: [0, 0],
  zoom: 1,
  dragging: false,
  dragMode: "rotate",
  lastPointer: [0, 0],
  meshVersion: 0,
  mode: "multiview",
  defaults: null,
};

const els = {
  canvas: $("glCanvas"),
  status: $("statusBadge"),
  meshInfo: $("meshInfo"),
  logs: $("logs"),
  summary: $("summaryList"),
  tooltip: $("faceTooltip"),
  loadBtn: $("loadBtn"),
  chooseFileBtn: $("chooseFileBtn"),
  stepFile: $("stepFile"),
  runBtn: $("runBtn"),
  resetBtn: $("resetViewBtn"),
  multiviewModeBtn: $("multiviewModeBtn"),
  singleviewModeBtn: $("singleviewModeBtn"),
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || JSON.stringify(data));
  return data;
}

function setStatus(text) {
  els.status.textContent = text;
}

function fillDefaults(defaults) {
  state.defaults = defaults;
  state.mode = defaults.mode || "multiview";
  applyModeDefaults(state.mode);
  $("m2fRoot").value = defaults.m2f_root;
  $("device").value = defaults.device;
  $("scoreThreshold").value = defaults.score_threshold;
  $("minRatio").value = defaults.min_ratio;
  $("minFaceArea").value = defaults.min_face_area;
}

function setMode(mode) {
  state.mode = mode;
  applyModeDefaults(mode);
}

function applyModeDefaults(mode) {
  const modes = (state.defaults && state.defaults.modes) || {};
  const item = modes[mode] || {};
  $("weightsPath").value = item.weights_path || (state.defaults && state.defaults.weights_path) || "";
  $("configPath").value = item.config_path || (state.defaults && state.defaults.config_path) || "";
  els.multiviewModeBtn.classList.toggle("active", mode === "multiview");
  els.singleviewModeBtn.classList.toggle("active", mode === "singleview");
}

function rgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function renderSummary(items) {
  els.summary.innerHTML = "";
  for (const item of items || []) {
    const row = document.createElement("div");
    row.className = "summary-item";
    row.innerHTML = `
      <span class="swatch" style="background:${rgb(item.color)}"></span>
      <span>${item.class_id} ${item.class_name}</span>
      <strong>${item.faces}</strong>
    `;
    els.summary.appendChild(row);
  }
}

function updateLogs(job) {
  els.logs.textContent = (job.logs || []).join("\n");
  els.logs.scrollTop = els.logs.scrollHeight;
}

function setMesh(mesh) {
  state.mesh = mesh;
  state.labels.clear();
  let triangles = 0;
  for (const face of mesh.faces) {
    state.labels.set(face.face_id, face);
    triangles += face.triangles.length / 3;
  }
  els.meshInfo.textContent = `${mesh.faces.length} faces · ${triangles} triangles`;
  buildBuffers();
  draw();
}

async function refreshMeshIfNeeded(job) {
  if (!state.jobId || job.mesh_version === state.meshVersion) return;
  const mesh = await api(`/api/jobs/${state.jobId}/mesh`);
  state.meshVersion = job.mesh_version;
  setMesh(mesh);
}

function collectRunPayload() {
  return {
    job_id: state.jobId,
    mode: state.mode,
    weights_path: $("weightsPath").value.trim(),
    config_path: $("configPath").value.trim(),
    m2f_root: $("m2fRoot").value.trim(),
    output_dir: $("outputDir").value.trim(),
    device: $("device").value.trim(),
    score_threshold: Number($("scoreThreshold").value),
    min_ratio: Number($("minRatio").value),
    min_face_area: Number($("minFaceArea").value),
  };
}

els.multiviewModeBtn.addEventListener("click", () => setMode("multiview"));
els.singleviewModeBtn.addEventListener("click", () => setMode("singleview"));

function onStepLoaded(data) {
  state.jobId = data.job.id;
  state.meshVersion = data.job.mesh_version || 0;
  setMesh(data.mesh);
  updateLogs(data.job);
  renderSummary([]);
  els.runBtn.disabled = false;
  setStatus("已加载");
}

async function uploadStepFile(file) {
  const form = new FormData();
  form.append("step_file", file);
  const res = await fetch("/api/upload-step", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || JSON.stringify(data));
  return data;
}

els.chooseFileBtn.addEventListener("click", () => {
  els.stepFile.click();
});

els.stepFile.addEventListener("change", async () => {
  const file = els.stepFile.files && els.stepFile.files[0];
  if (!file) return;
  try {
    setStatus("上传中");
    els.chooseFileBtn.disabled = true;
    els.loadBtn.disabled = true;
    $("stepPath").value = file.name;
    const data = await uploadStepFile(file);
    onStepLoaded(data);
  } catch (err) {
    setStatus("失败");
    alert(err.message);
  } finally {
    els.chooseFileBtn.disabled = false;
    els.loadBtn.disabled = false;
    els.stepFile.value = "";
  }
});

els.loadBtn.addEventListener("click", async () => {
  try {
    setStatus("加载中");
    els.loadBtn.disabled = true;
    const stepPath = $("stepPath").value.trim();
    const data = await api("/api/load-step", {
      method: "POST",
      body: JSON.stringify({ step_path: stepPath }),
    });
    onStepLoaded(data);
  } catch (err) {
    setStatus("失败");
    alert(err.message);
  } finally {
    els.loadBtn.disabled = false;
  }
});

els.runBtn.addEventListener("click", async () => {
  try {
    setStatus("推理中");
    els.runBtn.disabled = true;
    await api("/api/run", { method: "POST", body: JSON.stringify(collectRunPayload()) });
    pollJob();
  } catch (err) {
    setStatus("失败");
    els.runBtn.disabled = false;
    alert(err.message);
  }
});

async function pollJob() {
  if (!state.jobId) return;
  const data = await api(`/api/jobs/${state.jobId}`);
  updateLogs(data);
  setStatus(data.status);
  if (data.status === "done") {
    const mesh = await api(`/api/jobs/${state.jobId}/mesh`);
    state.meshVersion = data.mesh_version || state.meshVersion;
    setMesh(mesh);
    renderSummary(data.result.summary);
    els.runBtn.disabled = false;
    setStatus("完成");
    return;
  }
  await refreshMeshIfNeeded(data);
  renderSummary(data.summary);
  if (data.status === "failed") {
    els.runBtn.disabled = false;
    setStatus("失败");
    alert(data.error || "推理失败");
    return;
  }
  window.setTimeout(pollJob, 1000);
}

// ---- Minimal WebGL mesh viewer ------------------------------------------------
let gl;
let program;
let buffers = [];
let matrixLocation;
let lightLocation;
let positionLocation;
let colorLocation;
let normalLocation;

const vs = `
attribute vec3 aPosition;
attribute vec3 aColor;
attribute vec3 aNormal;
uniform mat4 uMatrix;
varying vec3 vColor;
varying vec3 vNormal;
void main() {
  gl_Position = uMatrix * vec4(aPosition, 1.0);
  vColor = aColor;
  vNormal = normalize(aNormal);
}`;

const fs = `
precision mediump float;
varying vec3 vColor;
varying vec3 vNormal;
uniform vec3 uLightDir;
void main() {
  vec3 n = normalize(vNormal);
  vec3 lightDir = normalize(uLightDir);
  float diffuse = max(dot(n, lightDir), 0.0);
  float backShade = max(dot(n, vec3(-0.35, -0.25, -0.85)), 0.0);
  float rim = pow(1.0 - abs(n.z), 2.0) * 0.18;
  vec3 lit = vColor * (0.34 + diffuse * 0.62 - backShade * 0.16);
  lit += vec3(1.0, 0.96, 0.88) * pow(diffuse, 18.0) * 0.18;
  lit += vec3(rim);
  gl_FragColor = vec4(clamp(lit, 0.0, 1.0), 1.0);
}`;

function createShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function initGL() {
  gl = els.canvas.getContext("webgl", { antialias: true });
  if (!gl) {
    els.meshInfo.textContent = "浏览器不支持 WebGL";
    return;
  }
  program = gl.createProgram();
  gl.attachShader(program, createShader(gl.VERTEX_SHADER, vs));
  gl.attachShader(program, createShader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  gl.useProgram(program);
  positionLocation = gl.getAttribLocation(program, "aPosition");
  colorLocation = gl.getAttribLocation(program, "aColor");
  normalLocation = gl.getAttribLocation(program, "aNormal");
  matrixLocation = gl.getUniformLocation(program, "uMatrix");
  lightLocation = gl.getUniformLocation(program, "uLightDir");
  gl.enable(gl.DEPTH_TEST);
}

function buildBuffers() {
  if (!gl || !state.mesh) return;
  for (const item of buffers) {
    gl.deleteBuffer(item.position);
    gl.deleteBuffer(item.color);
    gl.deleteBuffer(item.normal);
  }
  buffers = [];
  for (const face of state.mesh.faces) {
    const points = face.points;
    const tris = face.triangles;
    const vertices = [];
    const colors = [];
    const normals = [];
    const c = face.color.map((x) => x / 255);
    for (let i = 0; i < tris.length; i += 3) {
      const ia = tris[i] * 3;
      const ib = tris[i + 1] * 3;
      const ic = tris[i + 2] * 3;
      const ax = points[ia];
      const ay = points[ia + 1];
      const az = points[ia + 2];
      const bx = points[ib];
      const by = points[ib + 1];
      const bz = points[ib + 2];
      const cx = points[ic];
      const cy = points[ic + 1];
      const cz = points[ic + 2];
      const ux = bx - ax;
      const uy = by - ay;
      const uz = bz - az;
      const vx = cx - ax;
      const vy = cy - ay;
      const vz = cz - az;
      let nx = uy * vz - uz * vy;
      let ny = uz * vx - ux * vz;
      let nz = ux * vy - uy * vx;
      const invLen = 1 / Math.max(Math.hypot(nx, ny, nz), 1e-8);
      nx *= invLen;
      ny *= invLen;
      nz *= invLen;
      for (const idx of [ia, ib, ic]) {
        vertices.push(points[idx], points[idx + 1], points[idx + 2]);
        colors.push(c[0], c[1], c[2]);
        normals.push(nx, ny, nz);
      }
    }
    const position = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, position);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    const color = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, color);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
    const normal = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normal);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
    buffers.push({ position, color, normal, count: vertices.length / 3, face });
  }
}

function identity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
    }
  }
  return out;
}

function translate(x, y, z) {
  const m = identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

function scale(s) {
  const m = identity();
  m[0] = s;
  m[5] = s;
  m[10] = s;
  return m;
}

function rotateX(a) {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
}

function rotateY(a) {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1];
}

function projection() {
  if (!state.mesh) return identity();
  const b = state.mesh.bounds;
  const cx = (b[0] + b[1]) / 2;
  const cy = (b[2] + b[3]) / 2;
  const cz = (b[4] + b[5]) / 2;
  const span = Math.max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1e-6);
  const fit = (1.65 / span) * state.zoom;
  let m = identity();
  m = multiply(m, translate(state.pan[0], state.pan[1], 0));
  m = multiply(m, scale(fit));
  m = multiply(m, rotateX(state.rotation[0]));
  m = multiply(m, rotateY(state.rotation[1]));
  m = multiply(m, translate(-cx, -cy, -cz));
  return m;
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (els.canvas.width !== width || els.canvas.height !== height) {
    els.canvas.width = width;
    els.canvas.height = height;
  }
  gl.viewport(0, 0, width, height);
}

function draw() {
  if (!gl) return;
  resizeCanvas();
  gl.clearColor(0.87, 0.9, 0.92, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.uniformMatrix4fv(matrixLocation, false, new Float32Array(projection()));
  gl.uniform3f(lightLocation, -0.35, 0.55, 0.76);
  for (const item of buffers) {
    gl.bindBuffer(gl.ARRAY_BUFFER, item.position);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, item.color);
    gl.enableVertexAttribArray(colorLocation);
    gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, item.normal);
    gl.enableVertexAttribArray(normalLocation);
    gl.vertexAttribPointer(normalLocation, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, item.count);
  }
}

els.canvas.addEventListener("pointerdown", (event) => {
  state.dragging = true;
  state.dragMode = event.shiftKey || event.button === 1 ? "pan" : "rotate";
  state.lastPointer = [event.clientX, event.clientY];
  els.canvas.setPointerCapture(event.pointerId);
});

els.canvas.addEventListener("pointermove", (event) => {
  if (!state.dragging) return;
  const dx = event.clientX - state.lastPointer[0];
  const dy = event.clientY - state.lastPointer[1];
  state.lastPointer = [event.clientX, event.clientY];
  if (state.dragMode === "pan") {
    state.pan[0] += dx / els.canvas.clientWidth * 2;
    state.pan[1] -= dy / els.canvas.clientHeight * 2;
  } else {
    state.rotation[1] += dx * 0.01;
    state.rotation[0] += dy * 0.01;
  }
  draw();
});

els.canvas.addEventListener("pointerup", () => {
  state.dragging = false;
});

els.canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  state.zoom *= Math.exp(-event.deltaY * 0.001);
  state.zoom = Math.min(20, Math.max(0.05, state.zoom));
  draw();
});

els.resetBtn.addEventListener("click", () => {
  state.rotation = [-0.55, 0.72];
  state.pan = [0, 0];
  state.zoom = 1;
  draw();
});

window.addEventListener("resize", draw);

async function boot() {
  initGL();
  const defaults = await api("/api/defaults");
  fillDefaults(defaults);
  draw();
}

boot().catch((err) => alert(err.message));
