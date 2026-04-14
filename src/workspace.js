import { IDX } from "./constants.js";
import { drawTimeSeriesChart, drawTrajectoryChart, clearChart } from "./charts.js";
import { drawPoseSkeleton, drawTrail, drawVideoFrame } from "./drawing.js";
import { detectPoseForVideo, ensurePoseLandmarker, estimateBodyHeightPx, getLandmarkXY, midpoint } from "./pose.js";
import {
  buildSampleTimes,
  clamp,
  csvEscape,
  downloadBlob,
  drawTextBlock,
  filenameBase,
  gradient,
  interpolateSeries,
  isFiniteNumber,
  median,
  movingAverage,
  round,
  safeMax,
  safeRange,
  waitEvent
} from "./utils.js";

let activeWorkspaceAnalysisId = null;
let workspaceUidCounter = 0;
let poseLastVideoTimestampMs = 0;

export class AnalysisWorkspace {
  constructor(rootEl, index, handlers) {
    this.root = rootEl;
    this.index = index;
    this.stopRequested = false;
    this.currentVideoFile = null;
    this.latestResult = null;
    this.videoObjectUrl = null;
    this.uid = ++workspaceUidCounter;
    this.handlers = handlers;

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
    this.metricCameraAngle = this.root.querySelector(".metricCameraAngle");

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

    this.duplicateBtn.addEventListener("click", () => this.handlers.onDuplicate(this));
    this.removeBtn.addEventListener("click", () => this.handlers.onRemove(this));
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

    clearChart(this.trajCtx, this.trajectoryCanvas, "腰軌道 (横軸: 投球方向 / 縦軸: 鉛直方向)");
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
      "hip_forward_cm_raw",
      "hip_side_cm_raw",
      "hip_vertical_cm_raw",
      "hip_forward_cm_smooth",
      "hip_side_cm_smooth",
      "hip_vertical_cm_smooth",
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
      f.hipForwardCmRaw,
      f.hipSideCmRaw,
      f.hipVerticalCmRaw,
      f.hipForwardCmSmooth,
      f.hipSideCmSmooth,
      f.hipVerticalCmSmooth,
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
    setText(this.metricCameraAngle, isFiniteNumber(m.cameraAngleDeg)
      ? `${round(m.cameraAngleDeg, 1)}°`
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
      `hip 0°: ${round(frame.hipForwardCmSmooth, 1)} cm`,
      `hip 鉛直: ${round(frame.hipVerticalCmSmooth, 1)} cm`,
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
      const axisAnkleIdx = throwingHand === "right" ? IDX.RIGHT_ANKLE : IDX.LEFT_ANKLE;
      const leadAnkleIdx = throwingHand === "right" ? IDX.LEFT_ANKLE : IDX.RIGHT_ANKLE;

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
        const result = detectPoseForVideo(this.sourceVideo, tsMs);
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
        let axisAnkle = null;
        let leadAnkle = null;
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
          axisAnkle = getLandmarkXY(landmarks, axisAnkleIdx, width, height);
          leadAnkle = getLandmarkXY(landmarks, leadAnkleIdx, width, height);

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
          axisAnkleXpx: axisAnkle ? axisAnkle.x : NaN,
          axisAnkleYpx: axisAnkle ? axisAnkle.y : NaN,
          leadAnkleXpx: leadAnkle ? leadAnkle.x : NaN,
          leadAnkleYpx: leadAnkle ? leadAnkle.y : NaN,
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
        "axisAnkleXpx", "axisAnkleYpx",
        "leadAnkleXpx", "leadAnkleYpx",
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
      const firstValidAxisFrame = frames.find(
        (f) => isFiniteNumber(f.axisAnkleXpx) && isFiniteNumber(f.axisAnkleYpx)
      );

      if (!firstValidFrame || !firstValidAxisFrame) {
        throw new Error("腰または軸足ランドマークが取得できませんでした。");
      }

      const originHipX = firstValidFrame.hipXpx;
      const originHipY = firstValidFrame.hipYpx;
      const originAxisX = firstValidAxisFrame.axisAnkleXpx;
      const originAxisY = firstValidAxisFrame.axisAnkleYpx;

      for (const f of frames) {
        f.pxPerCm = f.bodyHeightPx / heightCm;
        f.depthCorrection = clamp(refPxPerCm / f.pxPerCm, 0.8, 1.25);

        const dxCmRaw = (f.hipXpx - originHipX) / f.pxPerCm;
        const dyCmRaw = -(f.hipYpx - originHipY) / f.pxPerCm;

        f.hipDxCmRaw = dxCmRaw;
        f.hipDyCmRaw = dyCmRaw;
        f.hipDxCmCorr = dxCmRaw * f.depthCorrection;
        f.hipDyCmCorr = dyCmRaw * f.depthCorrection;

        const ankleDxPx = Math.abs(f.leftAnkleXpx - f.rightAnkleXpx);
        f.ankleDxCmCorr = (ankleDxPx / f.pxPerCm) * f.depthCorrection;
      }

      const stepFrame = frames.reduce((best, cur) =>
        !best || cur.ankleDxCmCorr > best.ankleDxCmCorr ? cur : best, null);

      const stepAxisToLead = stepFrame
        ? {
            x: ((stepFrame.leadAnkleXpx - stepFrame.axisAnkleXpx) / stepFrame.pxPerCm) * stepFrame.depthCorrection,
            y: -((stepFrame.leadAnkleYpx - stepFrame.axisAnkleYpx) / stepFrame.pxPerCm) * stepFrame.depthCorrection
          }
        : null;

      const dirNorm = stepAxisToLead ? Math.hypot(stepAxisToLead.x, stepAxisToLead.y) : 0;
      const forwardUnit = dirNorm > 1e-6
        ? { x: stepAxisToLead.x / dirNorm, y: stepAxisToLead.y / dirNorm }
        : { x: 1, y: 0 };
      const sideUnit = { x: -forwardUnit.y, y: forwardUnit.x };

      const cameraAngleDeg = (Math.acos(clamp(forwardUnit.x, -1, 1)) * 180) / Math.PI;

      for (const f of frames) {
        const hipFromAxisX = ((f.hipXpx - originAxisX) / f.pxPerCm) * f.depthCorrection;
        const hipFromAxisY = -((f.hipYpx - originAxisY) / f.pxPerCm) * f.depthCorrection;
        f.hipForwardCmRaw = hipFromAxisX * forwardUnit.x + hipFromAxisY * forwardUnit.y;
        f.hipSideCmRaw = hipFromAxisX * sideUnit.x + hipFromAxisY * sideUnit.y;
        f.hipVerticalCmRaw = hipFromAxisY;
      }

      const smoothX = movingAverage(frames.map((f) => f.hipForwardCmRaw), smoothWindow);
      const smoothY = movingAverage(frames.map((f) => f.hipSideCmRaw), smoothWindow);
      const smoothVertical = movingAverage(frames.map((f) => f.hipVerticalCmRaw), smoothWindow);

      smoothX.forEach((v, i) => {
        frames[i].hipForwardCmSmooth = v;
        frames[i].hipDxCmSmooth = v;
      });
      smoothY.forEach((v, i) => {
        frames[i].hipSideCmSmooth = v;
        frames[i].hipDyCmSmooth = v;
      });
      smoothVertical.forEach((v, i) => {
        frames[i].hipVerticalCmSmooth = v;
      });

      const vx = gradient(smoothX, dt);
      const vy = gradient(smoothY, dt);

      vx.forEach((v, i) => (frames[i].hipVxCmS = v));
      vy.forEach((v, i) => (frames[i].hipVyCmS = v));
      frames.forEach((f) => {
        f.hipSpeedCmS = Math.sqrt(f.hipVxCmS ** 2 + f.hipVyCmS ** 2);
      });

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
        cameraAngleDeg,
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
