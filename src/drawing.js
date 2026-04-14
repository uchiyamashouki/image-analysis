import { POSE_CONNECTIONS } from "./constants.js";
import { getLandmarkXY } from "./pose.js";
import { isFiniteNumber } from "./utils.js";

export function drawVideoFrame(ctx, video, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

export function drawPoseSkeleton(ctx, landmarks, width, height) {
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

export function drawTrail(ctx, trail, trailLength) {
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
