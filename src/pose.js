import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

import { IDX, MODEL_URL, WASM_URL } from "./constants.js";
import { isFiniteNumber } from "./utils.js";

let poseLandmarker = null;

export async function ensurePoseLandmarker() {
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

export function detectPoseForVideo(videoEl, timestampMs) {
  if (!poseLandmarker) {
    throw new Error("PoseLandmarker is not initialized.");
  }
  return poseLandmarker.detectForVideo(videoEl, timestampMs);
}

export function getLandmarkXY(landmarks, idx, width, height) {
  if (!landmarks || !landmarks[idx]) return null;
  const lm = landmarks[idx];
  if (isFiniteNumber(lm.visibility) && lm.visibility < 0.3) return null;
  return {
    x: lm.x * width,
    y: lm.y * height,
    z: lm.z ?? 0
  };
}

export function midpoint(p1, p2) {
  if (!p1 && !p2) return null;
  if (!p1) return { ...p2 };
  if (!p2) return { ...p1 };
  return {
    x: (p1.x + p2.x) / 2,
    y: (p1.y + p2.y) / 2,
    z: ((p1.z ?? 0) + (p2.z ?? 0)) / 2
  };
}

export function estimateBodyHeightPx(landmarks, width, height) {
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
