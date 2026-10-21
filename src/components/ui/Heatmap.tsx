import { useEffect, useRef } from 'react';

const STOPS: [number, number, number][] = [
  [15, 17, 23],
  [31, 58, 120],
  [79, 142, 247],
  [46, 204, 113],
  [243, 156, 18],
  [255, 245, 220],
];

function color(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x));
  const f = x - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function diverging(t: number): [number, number, number] {
  const v = Math.max(-1, Math.min(1, t));
  return v >= 0 ? [26 + 53 * v, 29 + 113 * v, 39 + 208 * v] : [26 - 205 * v, 29 + 47 * -v, 39 + 21 * -v];
}

/** Renders a matrix of values as a crisp pixel heatmap. `signed` uses a blue/red diverging scale. */
export function Heatmap({ values, rows, cols, size = 120, signed = false, gray = false, className = '', title }: {
  values: ArrayLike<number>;
  rows: number;
  cols: number;
  size?: number;
  signed?: boolean;
  gray?: boolean;
  className?: string;
  title?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext('2d');
    if (!c || !ctx || rows <= 0 || cols <= 0) return;
    c.width = cols;
    c.height = rows;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < rows * cols; i++) {
      const v = values[i] ?? 0;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const absMax = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    const span = hi - lo || 1;
    const img = ctx.createImageData(cols, rows);
    for (let i = 0; i < rows * cols; i++) {
      const v = values[i] ?? 0;
      const [r, g, b] = signed ? diverging(v / absMax) : gray ? [255 * ((v - lo) / span), 255 * ((v - lo) / span), 255 * ((v - lo) / span)] : color((v - lo) / span);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [values, rows, cols, signed, gray]);

  const aspect = cols / rows;
  const w = aspect >= 1 ? size : size * aspect;
  const h = aspect >= 1 ? size / aspect : size;
  return <canvas ref={ref} title={title} className={`rounded-md border border-line ${className}`} style={{ width: w, height: Math.max(h, 6), imageRendering: 'pixelated' }} />;
}

/** Interprets a flat activation by its shape and returns a 2D grid (or channel grids) suitable for display. */
export function activationGrids(values: Float32Array, shape: number[] | null, maxMaps = 8): { rows: number; cols: number; data: Float32Array; label?: string }[] {
  if (!shape || !shape.length) return [{ rows: 1, cols: values.length, data: values }];
  if (shape.length === 3) {
    const [h, w, c] = shape;
    return Array.from({ length: Math.min(c, maxMaps) }, (_, k) => {
      const d = new Float32Array(h * w);
      for (let i = 0; i < h * w; i++) d[i] = values[i * c + k];
      return { rows: h, cols: w, data: d, label: `ch ${k}` };
    });
  }
  if (shape.length === 2) return [{ rows: shape[0], cols: shape[1], data: values }];
  const n = shape[0];
  if (n > 32) {
    const cols = Math.ceil(Math.sqrt(n));
    const data = new Float32Array(cols * Math.ceil(n / cols));
    data.set(values.subarray(0, n));
    return [{ rows: Math.ceil(n / cols), cols, data }];
  }
  return [{ rows: 1, cols: n, data: values }];
}
