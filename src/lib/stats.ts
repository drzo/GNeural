export interface Bin {
  x: number;
  label: string;
  count: number;
}

export function histogram(values: ArrayLike<number>, bins = 24): Bin[] {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!values.length || !Number.isFinite(lo)) return [];
  if (hi === lo) hi = lo + 1e-6;
  const w = (hi - lo) / bins;
  const out = Array.from({ length: bins }, (_, i) => ({ x: lo + (i + 0.5) * w, label: (lo + (i + 0.5) * w).toPrecision(2), count: 0 }));
  for (let i = 0; i < values.length; i++) out[Math.min(bins - 1, Math.floor((values[i] - lo) / w))].count++;
  return out;
}

export function stats(values: ArrayLike<number>) {
  const n = values.length;
  if (!n) return { mean: 0, std: 0, min: 0, max: 0, zeros: 0 };
  let s = 0, s2 = 0, min = Infinity, max = -Infinity, zeros = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    s += v;
    s2 += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
    if (Math.abs(v) < 1e-8) zeros++;
  }
  const mean = s / n;
  return { mean, std: Math.sqrt(Math.max(0, s2 / n - mean * mean)), min, max, zeros: zeros / n };
}
