import { isFiniteNumber, round } from "./utils.js";

export function clearChart(ctx, canvas, title = "") {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (title) {
    ctx.fillStyle = "#111827";
    ctx.font = "bold 24px sans-serif";
    ctx.fillText(title, 24, 34);
  }
}

export function drawTrajectoryChart(ctx, canvas, result) {
  clearChart(ctx, canvas, "腰軌道 (横軸: 投球方向 / 縦軸: 鉛直方向)");
  if (!result || !result.frames.length) return;

  const x = result.frames.map((f) => f.hipForwardCmSmooth);
  const y = result.frames.map((f) => f.hipVerticalCmSmooth);

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
  ctx.fillText("投球方向 (cm)", padL + w / 2 - 45, canvas.height - 20);

  ctx.save();
  ctx.translate(24, padT + h / 2 + 40);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("鉛直方向 (cm, 上方向が＋)", 0, 0);
  ctx.restore();
}

export function drawTimeSeriesChart(ctx, canvas, result) {
  clearChart(ctx, canvas, "時系列");
  if (!result || !result.frames.length) return;

  const frames = result.frames;
  const times = frames.map((f) => f.timeSec);
  const xVals = frames.map((f) => f.hipForwardCmSmooth);
  const yVals = frames.map((f) => f.hipVerticalCmSmooth);

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
  ctx.fillText("青: 投球方向", padL, 28);
  ctx.fillText("赤: 鉛直方向", padL + 160, 28);
  ctx.fillText("時間 (s)", padL + w / 2 - 20, canvas.height - 18);
}
