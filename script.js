import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

/* =========================
   MediaPipe
========================= */
let poseLandmarker = null;
let activeWorkspaceAnalysisId = null;
let workspaceUidCounter = 0;
let poseLastVideoTimestampMs = 0;

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const IDX = {
  NOSE: 0,
  LEFT_EYE: 2,
  RIGHT_EYE: 5,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32
};

const POSE_CONNECTIONS = [
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31],
  [24, 26], [26, 28], [28, 30], [30, 32]
];

/* =========================
   Utility
========================= */
function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round(value, digits = 2) {
  const p = 10 ** digits;
  return Math.round(value * p) / p;
}

function isFiniteNumber(v) {
  return Number.isFinite(v);
}

function mean(values) {
  const valid = values.filter(isFiniteNumber);
  if (!valid.length) return NaN;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function median(values) {
  const valid = values.filter(isFiniteNumber).slice().sort((a, b) => a - b);
  if (!valid.length) return NaN;
  const mid = Math.floor(valid.length / 2);
  return valid.length % 2 === 0
    ? (valid[mid - 1] + valid[mid]) / 2
    : valid[mid];
}

function movingAverage(arr, windowSize = 5) {
  const n = arr.length;
  const out = new Array(n).fill(NaN);
  const half = Math.floor(windowSize / 2);

  for (let i = 0; i < n; i++) {
    const vals = [];
    for (let j = i - half; j <= i + half; j++) {
      const idx = clamp(j, 0, n - 1);
      if (isFiniteNumber(arr[idx])) vals.push(arr[idx]);
    }
    out[i] = vals.length ? mean(vals) : NaN;
  }
  return out;
}

function interpolateSeries(arr) {
  const out = arr.slice();
  let lastValid = -1;
  let filledCount = 0;

  for (let i = 0; i < out.length; i++) {
    if (isFiniteNumber(out[i])) {
      lastValid = i;
      continue;
    }

    let nextValid = -1;
    for (let j = i + 1; j < out.length; j++) {
      if (isFiniteNumber(out[j])) {
        nextValid = j;
        break;
      }
    }

    if (lastValid === -1 && nextValid === -1) {
      out[i] = NaN;
    } else if (lastValid === -1) {
      out[i] = out[nextValid];
      filledCount += 1;
    } else if (nextValid === -1) {
      out[i] = out[lastValid];
      filledCount += 1;
    } else {
      const ratio = (i - lastValid) / (nextValid - lastValid);
      out[i] = out[lastValid] + (out[nextValid] - out[lastValid]) * ratio;
      filledCount += 1;
    }
  }

  return {
    values: out,
    filledCount
  };
}

function gradient(arr, dt) {
  const out = new Array(arr.length).fill(NaN);
  for (let i = 0; i < arr.length; i++) {
    if (i === 0) out[i] = (arr[i + 1] - arr[i]) / dt;
    else if (i === arr.length - 1) out[i] = (arr[i] - arr[i - 1]) / dt;
    else out[i] = (arr[i + 1] - arr[i - 1]) / (2 * dt);
  }
  return out;
}

function filenameBase(name) {
  const idx = name.lastIndexOf(".");
  return idx === -1 ? name : name.slice(0, idx);
}

function waitEvent(target, eventName) {
  return new Promise((resolve) => {
    const handler = () => {
      target.removeEventListener(eventName, handler);
      resolve();
    };
    target.addEventListener(eventName, handler, { once: true });
  });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function drawTextBlock(ctx, lines, x, y) {
  ctx.save();
  ctx.font = "bold 20px sans-serif";
  ctx.textBaseline = "top";

  const padding = 10;
  const lineH = 26;
  const width = Math.max(...lines.map((t) => ctx.measureText(t).width)) + padding * 2;
  const height = lineH * lines.length + padding * 2;

  ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
  ctx.fillRect(x, y, width, height);

  ctx.fillStyle = "white";
  lines.forEach((line, i) => {
    ctx.fillText(line, x + padding, y + padding + i * lineH);
  });

  ctx.restore();
}

function csvEscape(value) {
  if (value == null) return "";
  const text = `${value}`;
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replace(/"/g, "\"\"")}"`;
}

function buildSampleTimes(duration, fps) {
  const safeFps = Math.max(1, Math.floor(fps));
  const frameCount = Math.max(1, Math.floor(duration * safeFps) + 1);
  const times = [];

  for (let i = 0; i < frameCount; i++) {
    const t = Math.min(i / safeFps, duration);
    if (!times.length || Math.abs(t - times[times.length - 1]) > 1e-7) {
      times.push(t);
    }
  }

  if (times[times.length - 1] < duration) {
    times.push(duration);
  }

  return times;
}

function safeRange(values) {
  const valid = values.filter(isFiniteNumber);
  if (!valid.length) return NaN;
  return Math.max(...valid) - Math.min(...valid);
}

function safeMax(values) {
  const valid = values.filter(isFiniteNumber);
  if (!valid.length) return NaN;
  return Math.max(...valid);
}

/* =========================
   MediaPipe helpers
========================= */
async function ensurePoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  return poseLandmarker;
}

function getLandmarkXY(landmarks, idx, width, height) {
  if (!landmarks || !landmarks[idx]) return null;
  const lm = landmarks[idx];
  if (isFiniteNumber(lm.visibility) && lm.visibility < 0.3) return null;
  return {
    x: lm.x * width,
    y: lm.y * height,
    z: lm.z ?? 0
  };
}

function midpoint(p1, p2) {
  if (!p1 && !p2) return null;
  if (!p1) return { ...p2 };
  if (!p2) return { ...p1 };
  return {
    x: (p1.x + p2.x) / 2,
    y: (p1.y + p2.y) / 2,
    z: ((p1.z ?? 0) + (p2.z ?? 0)) / 2
  };
}

function estimateBodyHeightPx(landmarks, width, height) {
  if (!landmarks) return NaN;

  const topCandidates = [
    IDX.NOSE, IDX.LEFT_EYE, IDX.RIGHT_EYE, IDX.LEFT_EAR, IDX.RIGHT_EAR
  ].map((i) => getLandmarkXY(landmarks, i, width, height)).filter(Boolean);

  const bottomCandidates = [
    IDX.LEFT_ANKLE, IDX.RIGHT_ANKLE,
    IDX.LEFT_HEEL, IDX.RIGHT_HEEL,
    IDX.LEFT_FOOT_INDEX, IDX.RIGHT_FOOT_INDEX
  ].map((i) => getLandmarkXY(landmarks, i, width, height)).filter(Boolean);

  if (!topCandidates.length || !bottomCandidates.length) return NaN;

  const minY = Math.min(...topCandidates.map((p) => p.y));
  const maxY = Math.max(...bottomCandidates.map((p) => p.y));
  const bodyHeight = maxY - minY;

  return bodyHeight > 40 ? bodyHeight : NaN;
}

/* =========================
   Drawing
========================= */
function drawVideoFrame(ctx, video, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

function drawPoseSkeleton(ctx, landmarks, width, height) {
  if (!landmarks) return;

  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(56, 189, 248, 0.9)";
  ctx.fillStyle = "rgba(34, 197, 94, 0.95)";

  for (const [a, b] of POSE_CONNECTIONS) {
    const p1 = getLandmarkXY(landmarks, a, width, height);
    const p2 = getLandmarkXY(landmarks, b, width, height);
    if (!p1 || !p2) continue;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  for (let i = 0; i < landmarks.length; i++) {
    const p = getLandmarkXY(landmarks, i, width, height);
    if (!p) continue;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawTrail(ctx, trail, trailLength) {
  const start = Math.max(0, trail.length - trailLength);

  ctx.save();
  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(255, 208, 0, 0.95)";
  ctx.beginPath();

  let started = false;
  for (let i = start; i < trail.length; i++) {
    const p = trail[i];
    if (!p || !isFiniteNumber(p.x) || !isFiniteNumber(p.y)) continue;
    if (!started) {
      ctx.moveTo(p.x, p.y);
      started = true;
    } else {
      ctx.lineTo(p.x, p.y);
    }
  }
  ctx.stroke();

  const latest = trail[trail.length - 1];
  if (latest && isFiniteNumber(latest.x) && isFiniteNumber(latest.y)) {
    ctx.fillStyle = "#ff3b30";
    ctx.beginPath();
    ctx.arc(latest.x, latest.y, 8, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

/* =========================
   Charts
========================= */
function clearChart(ctx, canvas, title = "") {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (title) {
    ctx.fillStyle = "#111827";
    ctx.font = "bold 24px sans-serif";
    ctx.fillText(title, 24, 34);
  }
}

function drawTrajectoryChart(ctx, canvas, result) {
  clearChart(ctx, canvas, "腰軌道 XY");
  if (!result || !result.frames.length) return;

  const x = result.frames.map((f) => f.hipDxCmSmooth);
  const y = result.frames.map((f) => f.hipDyCmSmooth);

  const valid = x
    .map((vx, i) => ({ x: vx, y: y[i] }))
    .filter((p) => isFiniteNumber(p.x) && isFiniteNumber(p.y));

  if (!valid.length) return;

  const padL = 90;
  const padR = 40;
  const padT = 60;
  const padB = 70;
  const w = canvas.width - padL - padR;
  const h = canvas.height - padT - padB;

  const minX = Math.min(...valid.map((p) => p.x));
  const maxX = Math.max(...valid.map((p) => p.x));
  const minY = Math.min(...valid.map((p) => p.y));
  const maxY = Math.max(...valid.map((p) => p.y));

  const x0 = minX === maxX ? minX - 1 : minX;
  const x1 = minX === maxX ? maxX + 1 : maxX;
  const y0 = minY === maxY ? minY - 1 : minY;
  const y1 = minY === maxY ? maxY + 1 : maxY;

  function sx(v) {
    return padL + ((v - x0) / (x1 - x0)) * w;
  }
  function sy(v) {
    return padT + h - ((v - y0) / (y1 - y0)) * h;
  }

  ctx.strokeStyle = "#d1d5db";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#111827";
  ctx.font = "14px sans-serif";

  for (let i = 0; i <= 5; i++) {
    const yy = padT + (h / 5) * i;
    const yVal = y1 - ((y1 - y0) * i) / 5;
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(padL + w, yy);
    ctx.stroke();
    ctx.fillText(`${round(yVal, 1)} cm`, 10, yy + 4);
  }
  for (let i = 0; i <= 5; i++) {
    const xx = padL + (w / 5) * i;
    const xVal = x0 + ((x1 - x0) * i) / 5;
    ctx.beginPath();
    ctx.moveTo(xx, padT);
    ctx.lineTo(xx, padT + h);
    ctx.stroke();
    ctx.fillText(`${round(xVal, 1)} cm`, xx - 24, padT + h + 24);
  }

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 3;
  ctx.beginPath();
  valid.forEach((p, i) => {
    const px = sx(p.x);
    const py = sy(p.y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();

  const first = valid[0];
  const last = valid[valid.length - 1];

  ctx.fillStyle = "#16a34a";
  ctx.beginPath();
  ctx.arc(sx(first.x), sy(first.y), 7, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#dc2626";
  ctx.beginPath();
  ctx.arc(sx(last.x), sy(last.y), 7, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#111827";
  ctx.font = "16px sans-serif";
  ctx.fillText("前後移動 (cm)", padL + w / 2 - 42, canvas.height - 20);

  ctx.save();
  ctx.translate(24, padT + h / 2 + 40);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("上下移動 (cm, 上が＋)", 0, 0);
  ctx.restore();
}

function drawTimeSeriesChart(ctx, canvas, result) {
  clearChart(ctx, canvas, "時系列");
  if (!result || !result.frames.length) return;

  const frames = result.frames;
  const times = frames.map((f) => f.timeSec);
  const xVals = frames.map((f) => f.hipDxCmSmooth);
  const yVals = frames.map((f) => f.hipDyCmSmooth);

  const validVals = [...xVals, ...yVals].filter(isFiniteNumber);
  if (!validVals.length) return;

  const padL = 80;
  const padR = 40;
  const padT = 60;
  const padB = 60;
  const w = canvas.width - padL - padR;
  const h = canvas.height - padT - padB;

  const minT = Math.min(...times);
  const maxT = Math.max(...times);
  const minV = Math.min(...validVals);
  const maxV = Math.max(...validVals);

  function sx(t) {
    return padL + ((t - minT) / Math.max(0.0001, maxT - minT)) * w;
  }
  function sy(v) {
    return padT + h - ((v - minV) / Math.max(0.0001, maxV - minV)) * h;
  }

  ctx.strokeStyle = "#d1d5db";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 5; i++) {
    const yy = padT + (h / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(padL + w, yy);
    ctx.stroke();
  }

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 3;
  ctx.beginPath();
  let startedX = false;
  xVals.forEach((v, i) => {
    if (!isFiniteNumber(v)) return;
    const px = sx(times[i]);
    const py = sy(v);
    if (!startedX) {
      ctx.moveTo(px, py);
      startedX = true;
    } else {
      ctx.lineTo(px, py);
    }
  });
  ctx.stroke();

  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = 3;
  ctx.beginPath();
  let startedY = false;
  yVals.forEach((v, i) => {
    if (!isFiniteNumber(v)) return;
    const px = sx(times[i]);
    const py = sy(v);
    if (!startedY) {
      ctx.moveTo(px, py);
      startedY = true;
    } else {
      ctx.lineTo(px, py);
    }
  });
  ctx.stroke();

  ctx.fillStyle = "#111827";
  ctx.font = "16px sans-serif";
  ctx.fillText("青: 前後移動", padL, 28);
  ctx.fillText("赤: 上下移動", padL + 160, 28);
  ctx.fillText("時間 (s)", padL + w / 2 - 20, canvas.height - 18);
}

/* =========================
   Workspace class
========================= */
class AnalysisWorkspace {
  constructor(rootEl, index) {
    this.root = rootEl;
    this.index = index;
    this.stopRequested = false;
    this.currentVideoFile = null;
    this.latestResult = null;
    this.videoObjectUrl = null;
    this.uid = ++workspaceUidCounter;

    this.bindDom();
    this.attachEvents();
    this.rename(index);
    this.initCanvas();
  }

  bindDom() {
    this.titleEl = this.root.querySelector(".workspace-title");
    this.subtitleEl = this.root.querySelector(".workspace-subtitle");

    this.videoFileInput = this.root.querySelector(".videoFile");
    this.heightCmInput = this.root.querySelector(".heightCm");
    this.throwingHandSelect = this.root.querySelector(".throwingHand");
    this.analysisFpsInput = this.root.querySelector(".analysisFps");
    this.smoothWindowInput = this.root.querySelector(".smoothWindow");
    this.trailLengthInput = this.root.querySelector(".trailLength");

    this.loadBtn = this.root.querySelector(".loadBtn");
    this.analyzeBtn = this.root.querySelector(".analyzeBtn");
    this.stopBtn = this.root.querySelector(".stopBtn");
    this.duplicateBtn = this.root.querySelector(".duplicateWorkspaceBtn");
    this.removeBtn = this.root.querySelector(".removeWorkspaceBtn");

    this.progressBar = this.root.querySelector(".progressBar");
    this.statusText = this.root.querySelector(".statusText");
    this.exportStatus = this.root.querySelector(".exportStatus");

    this.overlayCanvas = this.root.querySelector(".overlayCanvas");
    this.overlayCtx = this.overlayCanvas.getContext("2d");
    this.sourceVideo = this.root.querySelector(".sourceVideo");

    this.trajectoryCanvas = this.root.querySelector(".trajectoryCanvas");
    this.trajCtx = this.trajectoryCanvas.getContext("2d");
    this.timeseriesCanvas = this.root.querySelector(".timeseriesCanvas");
    this.tsCtx = this.timeseriesCanvas.getContext("2d");

    this.metricStepWidth = this.root.querySelector(".metricStepWidth");
    this.metricHipXRange = this.root.querySelector(".metricHipXRange");
    this.metricHipYRange = this.root.querySelector(".metricHipYRange");
    this.metricHipSpeed = this.root.querySelector(".metricHipSpeed");
    this.metricLegLength = this.root.querySelector(".metricLegLength");
    this.metricStepRatio = this.root.querySelector(".metricStepRatio");
    this.metricHipXRangePct = this.root.querySelector(".metricHipXRangePct");
    this.metricHipYRangePct = this.root.querySelector(".metricHipYRangePct");

    this.downloadCsvBtn = this.root.querySelector(".downloadCsvBtn");
    this.downloadPngBtn = this.root.querySelector(".downloadPngBtn");
    this.exportVideoBtn = this.root.querySelector(".exportVideoBtn");
  }

  attachEvents() {
    this.loadBtn.addEventListener("click", () => this.handleLoadOnly());
    this.analyzeBtn.addEventListener("click", () => this.analyze());
    this.stopBtn.addEventListener("click", () => {
      this.stopRequested = true;
      this.stopBtn.disabled = true;
    });

    this.downloadCsvBtn.addEventListener("click", () => this.downloadCsv());
    this.downloadPngBtn.addEventListener("click", () => this.downloadPng());
    this.exportVideoBtn.addEventListener("click", () => this.exportAnnotatedVideo());

    this.duplicateBtn.addEventListener("click", () => duplicateWorkspace(this));
    this.removeBtn.addEventListener("click", () => removeWorkspace(this));
  }

  rename(index) {
    this.index = index;
    this.titleEl.textContent = `解析ページ ${index}`;
  }

  initCanvas() {
    this.overlayCanvas.width = 960;
    this.overlayCanvas.height = 540;
    this.overlayCtx.fillStyle = "black";
    this.overlayCtx.fillRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    this.overlayCtx.fillStyle = "white";
    this.overlayCtx.font = "24px sans-serif";
    this.overlayCtx.fillText("動画を選択するとここにプレビューが表示されます", 40, 60);

    clearChart(this.trajCtx, this.trajectoryCanvas, "腰軌道 XY");
    clearChart(this.tsCtx, this.timeseriesCanvas, "時系列");
    this.setStatus("未初期化");
    this.setExportStatus("未出力");
  }

  setStatus(text) {
    this.statusText.textContent = text;
  }

  setExportStatus(text) {
    this.exportStatus.textContent = text;
  }

  clearVideoUrl() {
    if (this.videoObjectUrl) {
      URL.revokeObjectURL(this.videoObjectUrl);
      this.videoObjectUrl = null;
    }
  }

  async loadVideoFile(file) {
    if (!file) throw new Error("動画ファイルが選択されていません。");

    this.clearVideoUrl();
    this.currentVideoFile = file;

    this.sourceVideo.pause();
    this.sourceVideo.removeAttribute("src");
    this.sourceVideo.load();

    this.videoObjectUrl = URL.createObjectURL(file);
    this.sourceVideo.src = this.videoObjectUrl;

    await waitEvent(this.sourceVideo, "loadedmetadata");

    this.overlayCanvas.width = this.sourceVideo.videoWidth;
    this.overlayCanvas.height = this.sourceVideo.videoHeight;
    drawVideoFrame(this.overlayCtx, this.sourceVideo, this.overlayCanvas);

    this.setStatus(`動画読込完了: ${file.name} (${round(this.sourceVideo.duration, 2)}秒)`);
  }

  async handleLoadOnly() {
    try {
      const file = this.videoFileInput.files?.[0];
      if (!file) {
        alert("まず動画ファイルを選択してください。");
        return;
      }
      this.setStatus("モデル初期化中...");
      await ensurePoseLandmarker();
      await this.loadVideoFile(file);
    } catch (err) {
      console.error(err);
      alert(err.message || "動画の準備に失敗しました。");
      this.setStatus("読込失敗");
    }
  }

  buildCsv(result) {
    const header = [
      "frame",
      "time_sec",
      "landmark_detected",
      "hip_x_px",
      "hip_y_px",
      "body_height_px",
      "px_per_cm",
      "depth_correction",
      "hip_dx_cm_raw",
      "hip_dy_cm_raw",
      "hip_dx_cm_corr",
      "hip_dy_cm_corr",
      "hip_dx_cm_smooth",
      "hip_dy_cm_smooth",
      "hip_vx_cm_s",
      "hip_vy_cm_s",
      "hip_speed_cm_s",
      "left_ankle_x_px",
      "right_ankle_x_px",
      "ankle_dx_cm_corr"
    ];

    const rows = result.frames.map((f) => [
      f.index,
      f.timeSec,
      f.landmarkDetected ? 1 : 0,
      f.hipXpx,
      f.hipYpx,
      f.bodyHeightPx,
      f.pxPerCm,
      f.depthCorrection,
      f.hipDxCmRaw,
      f.hipDyCmRaw,
      f.hipDxCmCorr,
      f.hipDyCmCorr,
      f.hipDxCmSmooth,
      f.hipDyCmSmooth,
      f.hipVxCmS,
      f.hipVyCmS,
      f.hipSpeedCmS,
      f.leftAnkleXpx,
      f.rightAnkleXpx,
      f.ankleDxCmCorr
    ]);

    return [header, ...rows]
      .map((row) => row.map(csvEscape).join(","))
      .join("\n");
  }

  updateSummary(result) {
    const m = result.metrics;
    const setText = (el, text) => {
      if (el) el.textContent = text;
    };

    setText(this.metricStepWidth, `${round(m.stepWidthCm, 1)} cm`);
    setText(this.metricHipXRange, `${round(m.hipXRangeCm, 1)} cm`);
    setText(this.metricHipYRange, `${round(m.hipYRangeCm, 1)} cm`);
    setText(this.metricHipSpeed, `${round(m.maxHipSpeedCmS, 1)} cm/s`);
    setText(this.metricLegLength, `${round(m.legLengthCm, 1)} cm`);
    setText(this.metricStepRatio, round(m.stepRatio, 3).toString());
    setText(this.metricHipXRangePct, isFiniteNumber(m.hipXRangePctLeg)
      ? `${round(m.hipXRangePctLeg, 1)} %`
      : "-");
    setText(this.metricHipYRangePct, isFiniteNumber(m.hipYRangePctLeg)
      ? `${round(m.hipYRangePctLeg, 1)} %`
      : "-");
  }

  async renderAnnotatedFrameAtIndex(frameIndex) {
    if (!this.latestResult) return;

    const frame = this.latestResult.frames[clamp(frameIndex, 0, this.latestResult.frames.length - 1)];
    const t = frame.timeSec;

    this.sourceVideo.currentTime = t;
    await waitEvent(this.sourceVideo, "seeked");

    drawVideoFrame(this.overlayCtx, this.sourceVideo, this.overlayCanvas);

    if (frame.landmarks) {
      drawPoseSkeleton(this.overlayCtx, frame.landmarks, this.overlayCanvas.width, this.overlayCanvas.height);
    }

    const trail = this.latestResult.frames
      .slice(0, frameIndex + 1)
      .map((f) =>
        isFiniteNumber(f.hipXpx) && isFiniteNumber(f.hipYpx)
          ? { x: f.hipXpx, y: f.hipYpx }
          : null
      );

    drawTrail(this.overlayCtx, trail, this.latestResult.trailLength);

    const lines = [
      `time: ${round(frame.timeSec, 2)} s`,
      `hip x: ${round(frame.hipDxCmSmooth, 1)} cm`,
      `hip y: ${round(frame.hipDyCmSmooth, 1)} cm`,
      `hip speed: ${round(frame.hipSpeedCmS, 1)} cm/s`,
      `depth corr: ${round(frame.depthCorrection, 3)}`
    ];

    if (frame.index === this.latestResult.metrics.stepFrameIndex) {
      lines.push("max step width");
    }

    drawTextBlock(this.overlayCtx, lines, 18, 18);
  }

  async analyze() {
    if (activeWorkspaceAnalysisId != null && activeWorkspaceAnalysisId !== this.uid) {
      alert("別の解析ページで解析中です。完了または停止してから実行してください。");
      return;
    }

    const file = this.videoFileInput.files?.[0];
    const heightCm = parseFloat(this.heightCmInput.value);
    const throwingHand = this.throwingHandSelect.value;
    const analysisFps = parseFloat(this.analysisFpsInput.value);
    const smoothWindow = parseInt(this.smoothWindowInput.value, 10);
    const trailLength = parseInt(this.trailLengthInput.value, 10);

    if (!file) {
      alert("動画ファイルを選択してください。");
      return;
    }
    if (!isFiniteNumber(heightCm) || heightCm <= 0) {
      alert("身長を正しく入力してください。");
      return;
    }
    if (!isFiniteNumber(analysisFps) || analysisFps <= 0) {
      alert("解析FPSを正しく入力してください。");
      return;
    }

    this.stopRequested = false;
    activeWorkspaceAnalysisId = this.uid;
    this.latestResult = null;
    this.interpolationStats = { total: 0, filled: 0 };
    this.progressBar.value = 0;
    this.analyzeBtn.disabled = true;
    this.stopBtn.disabled = false;
    this.downloadCsvBtn.disabled = true;
    this.downloadPngBtn.disabled = true;
    this.exportVideoBtn.disabled = true;
    this.setExportStatus("未出力");

    let analysisEndTimestampMs = null;

    try {
      this.setStatus("モデル初期化中...");
      await ensurePoseLandmarker();
      await this.loadVideoFile(file);

      const duration = this.sourceVideo.duration;
      const dt = 1 / analysisFps;
      const times = buildSampleTimes(duration, analysisFps);
      const runStartTimestampMs = poseLastVideoTimestampMs + 1000;

      const frames = [];
      const trail = [];

      this.setStatus("解析中...");

      for (let i = 0; i < times.length; i++) {
        if (this.stopRequested) {
          this.setStatus("停止しました");
          break;
        }

        const t = times[i];
        this.sourceVideo.currentTime = t;
        await waitEvent(this.sourceVideo, "seeked");

        const tsMs = runStartTimestampMs + Math.round(t * 1000);
        const result = poseLandmarker.detectForVideo(this.sourceVideo, tsMs);
        drawVideoFrame(this.overlayCtx, this.sourceVideo, this.overlayCanvas);

        let landmarks = null;
        if (result.landmarks && result.landmarks.length > 0) {
          landmarks = result.landmarks[0];
        }

        const width = this.overlayCanvas.width;
        const height = this.overlayCanvas.height;

        let hipMid = null;
        let shoulderMid = null;
        let leftAnkle = null;
        let rightAnkle = null;
        let bodyHeightPx = NaN;

        if (landmarks) {
          const leftHip = getLandmarkXY(landmarks, IDX.LEFT_HIP, width, height);
          const rightHip = getLandmarkXY(landmarks, IDX.RIGHT_HIP, width, height);
          hipMid = midpoint(leftHip, rightHip);

          const leftShoulder = getLandmarkXY(landmarks, IDX.LEFT_SHOULDER, width, height);
          const rightShoulder = getLandmarkXY(landmarks, IDX.RIGHT_SHOULDER, width, height);
          shoulderMid = midpoint(leftShoulder, rightShoulder);

          leftAnkle = getLandmarkXY(landmarks, IDX.LEFT_ANKLE, width, height);
          rightAnkle = getLandmarkXY(landmarks, IDX.RIGHT_ANKLE, width, height);

          bodyHeightPx = estimateBodyHeightPx(landmarks, width, height);
          drawPoseSkeleton(this.overlayCtx, landmarks, width, height);
        }

        if (hipMid) trail.push({ x: hipMid.x, y: hipMid.y });
        else trail.push(null);

        drawTrail(this.overlayCtx, trail, trailLength);

        frames.push({
          index: i,
          timeSec: t,
          landmarkDetected: Boolean(landmarks),
          landmarks,
          hipXpx: hipMid ? hipMid.x : NaN,
          hipYpx: hipMid ? hipMid.y : NaN,
          shoulderXpx: shoulderMid ? shoulderMid.x : NaN,
          shoulderYpx: shoulderMid ? shoulderMid.y : NaN,
          leftAnkleXpx: leftAnkle ? leftAnkle.x : NaN,
          leftAnkleYpx: leftAnkle ? leftAnkle.y : NaN,
          rightAnkleXpx: rightAnkle ? rightAnkle.x : NaN,
          rightAnkleYpx: rightAnkle ? rightAnkle.y : NaN,
          bodyHeightPx
        });

        drawTextBlock(
          this.overlayCtx,
          [
            `time: ${round(t, 2)} s`,
            `frame: ${i + 1} / ${times.length}`
          ],
          18,
          18
        );

        this.progressBar.value = (i + 1) / times.length;
        this.setStatus(`解析中... ${i + 1} / ${times.length}`);
      }

      if (!frames.length || this.stopRequested) {
        return;
      }

      const cols = [
        "hipXpx", "hipYpx",
        "shoulderXpx", "shoulderYpx",
        "leftAnkleXpx", "leftAnkleYpx",
        "rightAnkleXpx", "rightAnkleYpx",
        "bodyHeightPx"
      ];

      for (const col of cols) {
        const series = frames.map((f) => f[col]);
        const interp = interpolateSeries(series);
        interp.values.forEach((v, idx) => {
          frames[idx][col] = v;
        });
        if (!this.interpolationStats) this.interpolationStats = { total: 0, filled: 0 };
        this.interpolationStats.total += series.length;
        this.interpolationStats.filled += interp.filledCount;
      }

      const refBodyHeightPx = median(frames.map((f) => f.bodyHeightPx));
      if (!isFiniteNumber(refBodyHeightPx) || refBodyHeightPx < 40) {
        throw new Error("見かけ身長の推定に失敗しました。全身が映る、より横向きの動画を使ってください。");
      }

      const refPxPerCm = refBodyHeightPx / heightCm;
      const firstValidFrame = frames.find(
        (f) => isFiniteNumber(f.hipXpx) && isFiniteNumber(f.hipYpx)
      );

      if (!firstValidFrame) {
        throw new Error("腰のランドマークが取得できませんでした。");
      }

      const originX = firstValidFrame.hipXpx;
      const originY = firstValidFrame.hipYpx;
      const handednessXSign = throwingHand === "left" ? -1 : 1;

      for (const f of frames) {
        f.pxPerCm = f.bodyHeightPx / heightCm;
        f.depthCorrection = clamp(refPxPerCm / f.pxPerCm, 0.8, 1.25);

        const dxCmRaw = ((f.hipXpx - originX) / f.pxPerCm) * handednessXSign;
        const dyCmRaw = -(f.hipYpx - originY) / f.pxPerCm;

        f.hipDxCmRaw = dxCmRaw;
        f.hipDyCmRaw = dyCmRaw;
        f.hipDxCmCorr = dxCmRaw * f.depthCorrection;
        f.hipDyCmCorr = dyCmRaw * f.depthCorrection;

        const ankleDxPx = Math.abs(f.leftAnkleXpx - f.rightAnkleXpx);
        f.ankleDxCmCorr = (ankleDxPx / f.pxPerCm) * f.depthCorrection;
      }

      const smoothX = movingAverage(frames.map((f) => f.hipDxCmCorr), smoothWindow);
      const smoothY = movingAverage(frames.map((f) => f.hipDyCmCorr), smoothWindow);

      smoothX.forEach((v, i) => (frames[i].hipDxCmSmooth = v));
      smoothY.forEach((v, i) => (frames[i].hipDyCmSmooth = v));

      const vx = gradient(smoothX, dt);
      const vy = gradient(smoothY, dt);

      vx.forEach((v, i) => (frames[i].hipVxCmS = v));
      vy.forEach((v, i) => (frames[i].hipVyCmS = v));
      frames.forEach((f) => {
        f.hipSpeedCmS = Math.sqrt(f.hipVxCmS ** 2 + f.hipVyCmS ** 2);
      });

      const stepFrame = frames.reduce((best, cur) =>
        !best || cur.ankleDxCmCorr > best.ankleDxCmCorr ? cur : best, null);

      const estimatedStepWidthCm = stepFrame?.ankleDxCmCorr ?? NaN;
      const estimatedLegLengthCm = heightCm * 0.53;
      const stepRatio = estimatedStepWidthCm / estimatedLegLengthCm;

      const xVals = frames.map((f) => f.hipDxCmSmooth);
      const yVals = frames.map((f) => f.hipDyCmSmooth);
      const speedVals = frames.map((f) => f.hipSpeedCmS);
      const detectedCount = frames.filter((f) => f.landmarkDetected).length;
      const interpolationRate = this.interpolationStats?.total
        ? this.interpolationStats.filled / this.interpolationStats.total
        : NaN;

      const metrics = {
        stepWidthCm: estimatedStepWidthCm,
        legLengthCm: estimatedLegLengthCm,
        stepRatio,
        hipXRangeCm: safeRange(xVals),
        hipYRangeCm: safeRange(yVals),
        hipXRangePctLeg: (safeRange(xVals) / estimatedLegLengthCm) * 100,
        hipYRangePctLeg: (safeRange(yVals) / estimatedLegLengthCm) * 100,
        maxHipSpeedCmS: safeMax(speedVals),
        stepFrameIndex: stepFrame?.index ?? -1,
        stepTimeSec: stepFrame?.timeSec ?? NaN,
        refBodyHeightPx,
        detectedFrameRate: detectedCount / frames.length,
        interpolationRate
      };

      this.latestResult = {
        sourceName: file.name,
        throwingHand,
        heightCm,
        analysisFps,
        smoothWindow,
        trailLength,
        frames,
        metrics,
        video: {
          width: this.overlayCanvas.width,
          height: this.overlayCanvas.height,
          duration: this.sourceVideo.duration
        }
      };

      this.updateSummary(this.latestResult);
      drawTrajectoryChart(this.trajCtx, this.trajectoryCanvas, this.latestResult);
      drawTimeSeriesChart(this.tsCtx, this.timeseriesCanvas, this.latestResult);
      await this.renderAnnotatedFrameAtIndex(0);

      this.downloadCsvBtn.disabled = false;
      this.downloadPngBtn.disabled = false;
      this.exportVideoBtn.disabled = false;

      this.setStatus("解析完了");
      analysisEndTimestampMs = runStartTimestampMs + Math.round(duration * 1000) + 1000;
    } catch (err) {
      console.error(err);
      alert(err.message || "解析に失敗しました。");
      this.setStatus("解析失敗");
    } finally {
      if (analysisEndTimestampMs != null) {
        poseLastVideoTimestampMs = Math.max(poseLastVideoTimestampMs, analysisEndTimestampMs);
      }
      activeWorkspaceAnalysisId = null;
      this.analyzeBtn.disabled = false;
      this.stopBtn.disabled = true;
    }
  }

  downloadCsv() {
    if (!this.latestResult) return;
    const csv = this.buildCsv(this.latestResult);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    downloadBlob(blob, `${filenameBase(this.latestResult.sourceName)}_hip_track.csv`);
  }

  downloadPng() {
    this.trajectoryCanvas.toBlob((blob) => {
      if (!blob || !this.latestResult) return;
      downloadBlob(blob, `${filenameBase(this.latestResult.sourceName)}_trajectory.png`);
    });
  }

  async exportAnnotatedVideo() {
    if (!this.latestResult) return;

    try {
      this.exportVideoBtn.disabled = true;
      this.setExportStatus("注釈付き動画を書き出し中...");

      const stream = this.overlayCanvas.captureStream(this.latestResult.analysisFps);
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";

      const recorder = new MediaRecorder(stream, { mimeType });
      const chunks = [];

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };

      const done = new Promise((resolve) => {
        recorder.onstop = resolve;
      });

      recorder.start();

      for (let i = 0; i < this.latestResult.frames.length; i++) {
        await this.renderAnnotatedFrameAtIndex(i);
        await new Promise((r) => setTimeout(r, 1000 / this.latestResult.analysisFps));
        this.setExportStatus(`注釈付き動画を書き出し中... ${i + 1}/${this.latestResult.frames.length}`);
      }

      recorder.stop();
      await done;

      const blob = new Blob(chunks, { type: mimeType });
      downloadBlob(blob, `${filenameBase(this.latestResult.sourceName)}_hip_track.webm`);
      this.setExportStatus("注釈付き動画を書き出しました");
    } catch (err) {
      console.error(err);
      alert(err.message || "動画書き出しに失敗しました。");
      this.setExportStatus("動画書き出し失敗");
    } finally {
      this.exportVideoBtn.disabled = false;
    }
  }

  applyConfigFrom(other) {
    this.heightCmInput.value = other.heightCmInput.value;
    this.throwingHandSelect.value = other.throwingHandSelect.value;
    this.analysisFpsInput.value = other.analysisFpsInput.value;
    this.smoothWindowInput.value = other.smoothWindowInput.value;
    this.trailLengthInput.value = other.trailLengthInput.value;
  }

  destroy() {
    this.clearVideoUrl();
    this.sourceVideo.pause();
    this.sourceVideo.removeAttribute("src");
    this.sourceVideo.load();
  }
}

/* =========================
   Workspace management
========================= */
const workspaceContainer = document.getElementById("workspaceContainer");
const workspaceTemplate = document.getElementById("workspaceTemplate");
const addWorkspaceBtn = document.getElementById("addWorkspaceBtn");

const workspaceList = [];

function refreshWorkspaceTitles() {
  workspaceList.forEach((ws, idx) => ws.rename(idx + 1));
}

function createWorkspace(copyFrom = null) {
  const node = workspaceTemplate.content.firstElementChild.cloneNode(true);
  workspaceContainer.appendChild(node);

  const ws = new AnalysisWorkspace(node, workspaceList.length + 1);

  if (copyFrom) {
    ws.applyConfigFrom(copyFrom);
  }

  workspaceList.push(ws);
  refreshWorkspaceTitles();
  return ws;
}

function duplicateWorkspace(sourceWorkspace) {
  const ws = createWorkspace(sourceWorkspace);
  ws.root.scrollIntoView({ behavior: "smooth", block: "start" });
}

function removeWorkspace(workspace) {
  if (workspaceList.length === 1) {
    alert("解析ページは最低1つ残してください。");
    return;
  }

  const idx = workspaceList.indexOf(workspace);
  if (idx === -1) return;

  workspace.destroy();
  workspace.root.remove();
  workspaceList.splice(idx, 1);
  refreshWorkspaceTitles();
}

addWorkspaceBtn.addEventListener("click", () => {
  const ws = createWorkspace();
  ws.root.scrollIntoView({ behavior: "smooth", block: "start" });
});

/* =========================
   Initial
========================= */
createWorkspace();
