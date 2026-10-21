import type { Activation, Initializer, Param, Shape } from './types';
import { size } from './shapes';

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type Rng = () => number;

export function randn(rng: Rng) {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

export function makeParam(name: string, shape: Shape, init: Initializer | 'ones' | 'bias', rng: Rng, fanIn: number, fanOut: number, l1 = 0, l2 = 0): Param {
  const n = size(shape);
  const value = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    switch (init) {
      case 'glorot_uniform': {
        const lim = Math.sqrt(6 / (fanIn + fanOut));
        value[i] = (rng() * 2 - 1) * lim;
        break;
      }
      case 'he_normal':
        value[i] = randn(rng) * Math.sqrt(2 / fanIn);
        break;
      case 'lecun_normal':
        value[i] = randn(rng) * Math.sqrt(1 / fanIn);
        break;
      case 'gneural':
        value[i] = rng() * 0.2 - 0.1;
        break;
      case 'ones':
        value[i] = 1;
        break;
      default:
        value[i] = 0;
    }
  }
  return { name, shape, value, grad: new Float32Array(n), l1, l2 };
}

/** C[m,n] (+)= A[m,k] * B[k,n] */
export function matmul(A: Float32Array, B: Float32Array, m: number, k: number, n: number, C = new Float32Array(m * n), acc = false) {
  if (!acc) C.fill(0);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i * k + p];
      if (a === 0) continue;
      const bo = p * n, co = i * n;
      for (let j = 0; j < n; j++) C[co + j] += a * B[bo + j];
    }
  }
  return C;
}

/** C[k,n] += A[m,k]^T * B[m,n] */
export function matmulTA(A: Float32Array, B: Float32Array, m: number, k: number, n: number, C: Float32Array) {
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i * k + p];
      if (a === 0) continue;
      const co = p * n, bo = i * n;
      for (let j = 0; j < n; j++) C[co + j] += a * B[bo + j];
    }
  }
  return C;
}

/** C[m,k] (+)= A[m,n] * B[k,n]^T */
export function matmulTB(A: Float32Array, B: Float32Array, m: number, n: number, k: number, C = new Float32Array(m * k), acc = false) {
  if (!acc) C.fill(0);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      let s = 0;
      const ao = i * n, bo = p * n;
      for (let j = 0; j < n; j++) s += A[ao + j] * B[bo + j];
      C[i * k + p] += s;
    }
  }
  return C;
}

const SQRT_2_PI = Math.sqrt(2 / Math.PI);
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Applies an activation in-place-safe over rows of length `dim` (softmax is per row). */
export function activate(act: Activation, z: Float32Array, dim: number): Float32Array {
  const y = new Float32Array(z.length);
  switch (act) {
    case 'relu':
      for (let i = 0; i < z.length; i++) y[i] = z[i] > 0 ? z[i] : 0;
      return y;
    case 'leaky_relu':
      for (let i = 0; i < z.length; i++) y[i] = z[i] > 0 ? z[i] : 0.01 * z[i];
      return y;
    case 'sigmoid':
      for (let i = 0; i < z.length; i++) y[i] = sigmoid(z[i]);
      return y;
    case 'tanh':
      for (let i = 0; i < z.length; i++) y[i] = Math.tanh(z[i]);
      return y;
    case 'gelu':
      for (let i = 0; i < z.length; i++) {
        const x = z[i];
        y[i] = 0.5 * x * (1 + Math.tanh(SQRT_2_PI * (x + 0.044715 * x * x * x)));
      }
      return y;
    case 'softmax':
      for (let r = 0; r < z.length; r += dim) {
        let max = -Infinity;
        for (let j = 0; j < dim; j++) max = Math.max(max, z[r + j]);
        let sum = 0;
        for (let j = 0; j < dim; j++) sum += y[r + j] = Math.exp(z[r + j] - max);
        for (let j = 0; j < dim; j++) y[r + j] /= sum;
      }
      return y;
    default:
      y.set(z);
      return y;
  }
}

export function activateBackward(act: Activation, z: Float32Array, y: Float32Array, dy: Float32Array, dim: number): Float32Array {
  const dz = new Float32Array(z.length);
  switch (act) {
    case 'relu':
      for (let i = 0; i < z.length; i++) dz[i] = z[i] > 0 ? dy[i] : 0;
      return dz;
    case 'leaky_relu':
      for (let i = 0; i < z.length; i++) dz[i] = z[i] > 0 ? dy[i] : 0.01 * dy[i];
      return dz;
    case 'sigmoid':
      for (let i = 0; i < z.length; i++) dz[i] = dy[i] * y[i] * (1 - y[i]);
      return dz;
    case 'tanh':
      for (let i = 0; i < z.length; i++) dz[i] = dy[i] * (1 - y[i] * y[i]);
      return dz;
    case 'gelu':
      for (let i = 0; i < z.length; i++) {
        const x = z[i];
        const u = SQRT_2_PI * (x + 0.044715 * x * x * x);
        const t = Math.tanh(u);
        const du = SQRT_2_PI * (1 + 3 * 0.044715 * x * x);
        dz[i] = dy[i] * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du);
      }
      return dz;
    case 'softmax':
      for (let r = 0; r < z.length; r += dim) {
        let dot = 0;
        for (let j = 0; j < dim; j++) dot += dy[r + j] * y[r + j];
        for (let j = 0; j < dim; j++) dz[r + j] = y[r + j] * (dy[r + j] - dot);
      }
      return dz;
    default:
      dz.set(dy);
      return dz;
  }
}

export { sigmoid };
