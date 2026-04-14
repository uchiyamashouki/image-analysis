export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function round(value, digits = 2) {
  const p = 10 ** digits;
  return Math.round(value * p) / p;
}

export function isFiniteNumber(v) {
  return Number.isFinite(v);
}

export function mean(values) {
  const valid = values.filter(isFiniteNumber);
  if (!valid.length) return NaN;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

export function median(values) {
  const valid = values.filter(isFiniteNumber).slice().sort((a, b) => a - b);
  if (!valid.length) return NaN;
  const mid = Math.floor(valid.length / 2);
  return valid.length % 2 === 0
    ? (valid[mid - 1] + valid[mid]) / 2
    : valid[mid];
}

export function movingAverage(arr, windowSize = 5) {
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

export function interpolateSeries(arr) {
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

export function gradient(arr, dt) {
  const out = new Array(arr.length).fill(NaN);
  for (let i = 0; i < arr.length; i++) {
    if (i === 0) out[i] = (arr[i + 1] - arr[i]) / dt;
    else if (i === arr.length - 1) out[i] = (arr[i] - arr[i - 1]) / dt;
    else out[i] = (arr[i + 1] - arr[i - 1]) / (2 * dt);
  }
  return out;
}

export function filenameBase(name) {
  const idx = name.lastIndexOf(".");
  return idx === -1 ? name : name.slice(0, idx);
}

export function waitEvent(target, eventName) {
  return new Promise((resolve) => {
    const handler = () => {
      target.removeEventListener(eventName, handler);
      resolve();
    };
    target.addEventListener(eventName, handler, { once: true });
  });
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function drawTextBlock(ctx, lines, x, y) {
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

export function csvEscape(value) {
  if (value == null) return "";
  const text = `${value}`;
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replace(/"/g, '""')}"`;
}

export function buildSampleTimes(duration, fps) {
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

export function safeRange(values) {
  const valid = values.filter(isFiniteNumber);
  if (!valid.length) return NaN;
  return Math.max(...valid) - Math.min(...valid);
}

export function safeMax(values) {
  const valid = values.filter(isFiniteNumber);
  if (!valid.length) return NaN;
  return Math.max(...valid);
}
