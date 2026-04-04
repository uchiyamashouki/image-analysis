import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

/* =========================
   DOM
========================= */
const videoFileInput = document.getElementById("videoFile");
const heightCmInput = document.getElementById("heightCm");
const throwingHandSelect = document.getElementById("throwingHand");
const analysisFpsInput = document.getElementById("analysisFps");
const smoothWindowInput = document.getElementById("smoothWindow");
const trailLengthInput = document.getElementById("trailLength");

const loadDemoBtn = document.getElementById("loadDemoBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const stopBtn = document.getElementById("stopBtn");

const progressBar = document.getElementById("progressBar");
const statusText = document.getElementById("statusText");
const exportStatus = document.getElementById("exportStatus");

const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d");
const sourceVideo = document.getElementById("sourceVideo");

const trajectoryCanvas = document.getElementById("trajectoryCanvas");
const trajCtx = trajectoryCanvas.getContext("2d");
const timeseriesCanvas = document.getElementById("timeseriesCanvas");
const tsCtx = timeseriesCanvas.getContext("2d");

const metricStepWidth = document.getElementById("metricStepWidth");
const metricHipXRange = document.getElementById("metricHipXRange");
const metricHipYRange = document.getElementById("metricHipYRange");
const metricHipSpeed = document.getElementById("metricHipSpeed");
const metricLegLength = document.getElementById("metricLegLength");
const metricStepRatio = document.getElementById("metricStepRatio");

const downloadCsvBtn = document.getElementById("downloadCsvBtn");
const downloadPngBtn = document.getElementById("downloadPngBtn");
const exportVideoBtn = document.getElementById("exportVideoBtn");

/* =========================
   MediaPipe
========================= */
let poseLandmarker = null;

/* =========================
   State
========================= */
let currentVideoFile = null;
let latestResult = null;
let stopRequested = false;

/* =========================
   定数
========================= */
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

// MediaPipe Pose Landmarker 33点のインデックス
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
   共通
========================= */
function setStatus(text) {
  statusText.textContent = text;
}

function setExportStatus(text) {
  exportStatus.textContent = text;
}

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
    } else if (nextValid === -1) {
      out[i] = out[lastValid];
    } else {
      const ratio = (i - lastValid) / (nextValid - lastValid);
      out[i] = out[lastValid] + (out[nextValid] - out[lastValid]) * ratio;
    }
  }
  return out;
}

function gradient(arr, dt) {
  const out = new Array(arr.length).fill(NaN);
  for (let i = 0; i < arr.length; i++) {
    if (i === 0) {
      out[i] = (arr[i + 1] - arr[i]) / dt;
    } else if (i === arr.length - 1) {
      out[i] = (arr[i] - arr[i - 1]) / dt;
    } else {
      out[i] = (arr[i + 1] - arr[i - 1]) / (2 * dt);
    }
  }
  return out;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
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

/* =========================
   MediaPipe 初期化
========================= */
async function ensurePoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  setStatus("MediaPipeモデルを初期化中...");
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

  setStatus("モデル初期化完了");
  return poseLandmarker;
}

/* =========================
   ランドマーク補助
========================= */
function getLandmarkXY(landmarks, idx, width, height) {
  if (!landmarks || !landmarks[idx]) return null;
  const lm = landmarks[idx];
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
   描画
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
   グラフ
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

function drawTrajectoryChart(result) {
  clearChart(trajCtx, trajectoryCanvas, "腰軌道 XY");

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
  const w = trajectoryCanvas.width - padL - padR;
  const h = trajectoryCanvas.height - padT - padB;

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

  trajCtx.strokeStyle = "#d1d5db";
  trajCtx.lineWidth = 1;

  for (let i = 0; i <= 5; i++) {
    const yy = padT + (h / 5) * i;
    trajCtx.beginPath();
    trajCtx.moveTo(padL, yy);
    trajCtx.lineTo(padL + w, yy);
    trajCtx.stroke();
  }
  for (let i = 0; i <= 5; i++) {
    const xx = padL + (w / 5) * i;
    trajCtx.beginPath();
    trajCtx.moveTo(xx, padT);
    trajCtx.lineTo(xx, padT + h);
    trajCtx.stroke();
  }

  trajCtx.strokeStyle = "#2563eb";
  trajCtx.lineWidth = 3;
  trajCtx.beginPath();
  valid.forEach((p, i) => {
    const px = sx(p.x);
    const py = sy(p.y);
    if (i === 0) trajCtx.moveTo(px, py);
    else trajCtx.lineTo(px, py);
  });
  trajCtx.stroke();

  const first = valid[0];
  const last = valid[valid.length - 1];

  trajCtx.fillStyle = "#16a34a";
  trajCtx.beginPath();
  trajCtx.arc(sx(first.x), sy(first.y), 7, 0, Math.PI * 2);
  trajCtx.fill();

  trajCtx.fillStyle = "#dc2626";
  trajCtx.beginPath();
  trajCtx.arc(sx(last.x), sy(last.y), 7, 0, Math.PI * 2);
  trajCtx.fill();

  trajCtx.fillStyle = "#111827";
  trajCtx.font = "16px sans-serif";
  trajCtx.fillText("前後移動 (cm)", padL + w / 2 - 42, trajectoryCanvas.height - 20);

  trajCtx.save();
  trajCtx.translate(24, padT + h / 2 + 40);
  trajCtx.rotate(-Math.PI / 2);
  trajCtx.fillText("上下移動 (cm, 上が＋)", 0, 0);
  trajCtx.restore();
}

function drawTimeSeriesChart(result) {
  clearChart(tsCtx, timeseriesCanvas, "時系列");

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
  const w = timeseriesCanvas.width - padL - padR;
  const h = timeseriesCanvas.height - padT - padB;

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

  tsCtx.strokeStyle = "#d1d5db";
  tsCtx.lineWidth = 1;

  for (let i = 0; i <= 5; i++) {
    const yy = padT + (h / 5) * i;
    tsCtx.beginPath();
    tsCtx.moveTo(padL, yy);
    tsCtx.lineTo(padL + w, yy);
    tsCtx.stroke();
  }

  tsCtx.strokeStyle = "#2563eb";
  tsCtx.lineWidth = 3;
  tsCtx.beginPath();
  xVals.forEach((v, i) => {
    if (!isFiniteNumber(v)) return;
    const px = sx(times[i]);
    const py = sy(v);
    if (i === 0) tsCtx.moveTo(px, py);
    else tsCtx.lineTo(px, py);
  });
  tsCtx.stroke();

  tsCtx.strokeStyle = "#dc2626";
  tsCtx.lineWidth = 3;
  tsCtx.beginPath();
  yVals.forEach((v, i) => {
    if (!isFiniteNumber(v)) return;
    const px = sx(times[i]);
    const py = sy(v);
    if (i === 0) tsCtx.moveTo(px, py);
    else tsCtx.lineTo(px, py);
  });
  tsCtx.stroke();

  tsCtx.fillStyle = "#111827";
  tsCtx.font = "16px sans-serif";
  tsCtx.fillText("青: 前後移動", padL, 28);
  tsCtx.fillText("赤: 上下移動", padL + 160, 28);
  tsCtx.fillText("時間 (s)", padL + w / 2 - 20, timeseriesCanvas.height - 18);
}

/* =========================
   入力動画準備
========================= */
async function loadVideoFile(file) {
  if (!file) throw new Error("動画ファイルが選択されていません。");

  currentVideoFile = file;
  sourceVideo.pause();
  sourceVideo.removeAttribute("src");
  sourceVideo.load();

  const objectUrl = URL.createObjectURL(file);
  sourceVideo.src = objectUrl;

  await waitEvent(sourceVideo, "loadedmetadata");

  overlayCanvas.width = sourceVideo.videoWidth;
  overlayCanvas.height = sourceVideo.videoHeight;
  drawVideoFrame(overlayCtx, sourceVideo, overlayCanvas);

  setStatus(`動画読込完了: ${file.name} (${round(sourceVideo.duration, 2)}秒)`);
}

/* =========================
   解析本体
========================= */
async function analyzeCurrentVideo() {
  const file = videoFileInput.files?.[0];
  const heightCm = parseFloat(heightCmInput.value);
  const throwingHand = throwingHandSelect.value;
  const analysisFps = parseFloat(analysisFpsInput.value);
  const smoothWindow = parseInt(smoothWindowInput.value, 10);
  const trailLength = parseInt(trailLengthInput.value, 10);

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

  stopRequested = false;
  latestResult = null;
  progressBar.value = 0;
  analyzeBtn.disabled = true;
  stopBtn.disabled = false;
  downloadCsvBtn.disabled = true;
  downloadPngBtn.disabled = true;
  exportVideoBtn.disabled = true;
  setExportStatus("未出力");

  try {
    await ensurePoseLandmarker();
    await loadVideoFile(file);

    const duration = sourceVideo.duration;
    const dt = 1 / analysisFps;
    const times = [];
    for (let t = 0; t <= duration; t += dt) {
      times.push(Math.min(t, duration));
    }

    const frames = [];
    const trail = [];

    setStatus("解析中...");

    for (let i = 0; i < times.length; i++) {
      if (stopRequested) {
        setStatus("停止しました");
        break;
      }

      const t = times[i];
      sourceVideo.currentTime = t;
      await waitEvent(sourceVideo, "seeked");

      const result = poseLandmarker.detectForVideo(sourceVideo, Math.round(t * 1000));

      drawVideoFrame(overlayCtx, sourceVideo, overlayCanvas);

      let landmarks = null;
      if (result.landmarks && result.landmarks.length > 0) {
        landmarks = result.landmarks[0];
      }

      const width = overlayCanvas.width;
      const height = overlayCanvas.height;

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

        drawPoseSkeleton(overlayCtx, landmarks, width, height);
      }

      if (hipMid) {
        trail.push({ x: hipMid.x, y: hipMid.y });
      } else {
        trail.push(null);
      }

      drawTrail(overlayCtx, trail, trailLength);

      frames.push({
        index: i,
        timeSec: t,
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
        overlayCtx,
        [
          `time: ${round(t, 2)} s`,
          `frame: ${i + 1} / ${times.length}`
        ],
        18,
        18
      );

      progressBar.value = (i + 1) / times.length;
      setStatus(`解析中... ${i + 1} / ${times.length}`);
    }

    if (!frames.length || stopRequested) {
      analyzeBtn.disabled = false;
      stopBtn.disabled = true;
      return;
    }

    // 補間
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
      interp.forEach((v, idx) => {
        frames[idx][col] = v;
      });
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

    for (const f of frames) {
      f.pxPerCm = f.bodyHeightPx / heightCm;
      f.depthCorrection = clamp(refPxPerCm / f.pxPerCm, 0.8, 1.25);

      const dxCmRaw = (f.hipXpx - originX) / f.pxPerCm;
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

    const xVals = frames.map((f) => f.hipDxCmSmooth).filter(isFiniteNumber);
    const yVals = frames.map((f) => f.hipDyCmSmooth).filter(isFiniteNumber);
    const speedVals = frames.map((f) => f.hipSpeedCmS).filter(isFiniteNumber);

    const metrics = {
      stepWidthCm: estimatedStepWidthCm,
      legLengthCm: estimatedLegLengthCm,
      stepRatio,
      hipXRangeCm: Math.max(...xVals) - Math.min(...xVals),
      hipYRangeCm: Math.max(...yVals) - Math.min(...yVals),
      maxHipSpeedCmS: Math.max(...speedVals),
      stepFrameIndex: stepFrame?.index ?? -1,
      stepTimeSec: stepFrame?.timeSec ?? NaN,
      refBodyHeightPx
    };

    latestResult = {
      sourceName: file.name,
      throwingHand,
      heightCm,
      analysisFps,
      smoothWindow,
      trailLength,
      frames,
      metrics,
      video: {
        width: overlayCanvas.width,
        height: overlayCanvas.height,
        duration: sourceVideo.duration
      }
    };

    updateSummary(latestResult);
    drawTrajectoryChart(latestResult);
    drawTimeSeriesChart(latestResult);
    await renderAnnotatedFrameAtIndex(0);

    downloadCsvBtn.disabled = false;
    downloadPngBtn.disabled = false;
    exportVideoBtn.disabled = false;

    setStatus("解析完了");
  } catch (err) {
    console.error(err);
    alert(err.message || "解析に失敗しました。");
    setStatus("解析失敗");
  } finally {
    analyzeBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

/* =========================
   結果描画
========================= */
function updateSummary(result) {
  const m = result.metrics;
  metricStepWidth.textContent = `${round(m.stepWidthCm, 1)} cm`;
  metricHipXRange.textContent = `${round(m.hipXRangeCm, 1)} cm`;
  metricHipYRange.textContent = `${round(m.hipYRangeCm, 1)} cm`;
  metricHipSpeed.textContent = `${round(m.maxHipSpeedCmS, 1)} cm/s`;
  metricLegLength.textContent = `${round(m.legLengthCm, 1)} cm`;
  metricStepRatio.textContent = round(m.stepRatio, 3).toString();
}

async function renderAnnotatedFrameAtIndex(frameIndex) {
  if (!latestResult) return;

  const frame = latestResult.frames[clamp(frameIndex, 0, latestResult.frames.length - 1)];
  const t = frame.timeSec;
  sourceVideo.currentTime = t;
  await waitEvent(sourceVideo, "seeked");

  drawVideoFrame(overlayCtx, sourceVideo, overlayCanvas);

  if (frame.landmarks) {
    drawPoseSkeleton(overlayCtx, frame.landmarks, overlayCanvas.width, overlayCanvas.height);
  }

  const trail = latestResult.frames
    .slice(0, frameIndex + 1)
    .map((f) =>
      isFiniteNumber(f.hipXpx) && isFiniteNumber(f.hipYpx)
        ? { x: f.hipXpx, y: f.hipYpx }
        : null
    );

  drawTrail(overlayCtx, trail, latestResult.trailLength);

  const lines = [
    `time: ${round(frame.timeSec, 2)} s`,
    `hip x: ${round(frame.hipDxCmSmooth, 1)} cm`,
    `hip y: ${round(frame.hipDyCmSmooth, 1)} cm`,
    `hip speed: ${round(frame.hipSpeedCmS, 1)} cm/s`,
    `depth corr: ${round(frame.depthCorrection, 3)}`
  ];

  if (frame.index === latestResult.metrics.stepFrameIndex) {
    lines.push("max step width");
  }

  drawTextBlock(overlayCtx, lines, 18, 18);
}

/* =========================
   CSV / PNG / Video
========================= */
function buildCsv(result) {
  const header = [
    "frame",
    "time_sec",
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
    .map((row) =>
      row
        .map((v) => (v === null || v === undefined ? "" : `${v}`))
        .join(",")
    )
    .join("\n");
}

async function exportAnnotatedVideo(result) {
  setExportStatus("注釈付き動画を書き出し中...");

  const stream = overlayCanvas.captureStream(result.analysisFps);
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

  for (let i = 0; i < result.frames.length; i++) {
    await renderAnnotatedFrameAtIndex(i);
    await new Promise((r) => setTimeout(r, 1000 / result.analysisFps));
    setExportStatus(`注釈付き動画を書き出し中... ${i + 1}/${result.frames.length}`);
  }

  recorder.stop();
  await done;

  const blob = new Blob(chunks, { type: mimeType });
  downloadBlob(blob, `${filenameBase(result.sourceName)}_hip_track.webm`);
  setExportStatus("注釈付き動画を書き出しました");
}

/* =========================
   ボタン
========================= */
loadDemoBtn.addEventListener("click", async () => {
  try {
    const file = videoFileInput.files?.[0];
    if (!file) {
      alert("まず動画ファイルを選択してください。");
      return;
    }
    await ensurePoseLandmarker();
    await loadVideoFile(file);
  } catch (err) {
    console.error(err);
    alert(err.message || "動画の準備に失敗しました。");
  }
});

analyzeBtn.addEventListener("click", analyzeCurrentVideo);

stopBtn.addEventListener("click", () => {
  stopRequested = true;
  stopBtn.disabled = true;
});

downloadCsvBtn.addEventListener("click", () => {
  if (!latestResult) return;
  const csv = buildCsv(latestResult);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  downloadBlob(blob, `${filenameBase(latestResult.sourceName)}_hip_track.csv`);
});

downloadPngBtn.addEventListener("click", () => {
  trajectoryCanvas.toBlob((blob) => {
    if (!blob || !latestResult) return;
    downloadBlob(blob, `${filenameBase(latestResult.sourceName)}_trajectory.png`);
  });
});

exportVideoBtn.addEventListener("click", async () => {
  if (!latestResult) return;
  try {
    exportVideoBtn.disabled = true;
    await exportAnnotatedVideo(latestResult);
  } catch (err) {
    console.error(err);
    alert(err.message || "動画書き出しに失敗しました。");
    setExportStatus("動画書き出し失敗");
  } finally {
    exportVideoBtn.disabled = false;
  }
});

/* =========================
   初期表示
========================= */
overlayCanvas.width = 960;
overlayCanvas.height = 540;
overlayCtx.fillStyle = "black";
overlayCtx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
overlayCtx.fillStyle = "white";
overlayCtx.font = "24px sans-serif";
overlayCtx.fillText("動画を選択するとここにプレビューが表示されます", 40, 60);

clearChart(trajCtx, trajectoryCanvas, "腰軌道 XY");
clearChart(tsCtx, timeseriesCanvas, "時系列");
setStatus("未初期化");
setExportStatus("未出力");
