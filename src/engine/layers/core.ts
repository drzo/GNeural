import { activate, activateBackward, makeParam, matmul, matmulTA, matmulTB, type Rng } from '../math';
import type { Activation, Initializer, Param } from '../types';

export interface InitOpts {
  init: Initializer;
  l1: number;
  l2: number;
}

/** Fully connected transform applied to each row of length `inDim`. */
export class Linear {
  W: Param;
  b: Param | null;
  params: Param[];
  private x: Float32Array = new Float32Array(0);
  private z: Float32Array = new Float32Array(0);
  private y: Float32Array = new Float32Array(0);

  constructor(
    public inDim: number,
    public outDim: number,
    public act: Activation,
    rng: Rng,
    o: InitOpts,
    bias = true,
    prefix = '',
  ) {
    this.W = makeParam(`${prefix}kernel`, [inDim, outDim], o.init, rng, inDim, outDim, o.l1, o.l2);
    this.b = bias ? makeParam(`${prefix}bias`, [outDim], o.init === 'gneural' ? 'ones' : 'zeros', rng, inDim, outDim) : null;
    this.params = this.b ? [this.W, this.b] : [this.W];
  }

  forward(x: Float32Array) {
    const rows = x.length / this.inDim;
    this.x = x;
    const z = matmul(x, this.W.value, rows, this.inDim, this.outDim);
    if (this.b) {
      const b = this.b.value;
      for (let r = 0; r < rows; r++) for (let j = 0; j < this.outDim; j++) z[r * this.outDim + j] += b[j];
    }
    this.z = z;
    this.y = activate(this.act, z, this.outDim);
    return this.y;
  }

  backward(dy: Float32Array) {
    const rows = this.x.length / this.inDim;
    const dz = activateBackward(this.act, this.z, this.y, dy, this.outDim);
    matmulTA(this.x, dz, rows, this.inDim, this.outDim, this.W.grad);
    if (this.b) {
      const g = this.b.grad;
      for (let r = 0; r < rows; r++) for (let j = 0; j < this.outDim; j++) g[j] += dz[r * this.outDim + j];
    }
    return matmulTB(dz, this.W.value, rows, this.outDim, this.inDim);
  }
}

/** Normalizes each row of length `dim`, then scales and shifts. */
export class LayerNorm {
  gamma: Param;
  beta: Param;
  params: Param[];
  private xhat = new Float32Array(0);
  private inv = new Float32Array(0);

  constructor(public dim: number, rng: Rng, prefix: string) {
    this.gamma = makeParam(`${prefix}gamma`, [dim], 'ones', rng, dim, dim);
    this.beta = makeParam(`${prefix}beta`, [dim], 'zeros', rng, dim, dim);
    this.params = [this.gamma, this.beta];
  }

  forward(x: Float32Array) {
    const d = this.dim, rows = x.length / d;
    const y = new Float32Array(x.length);
    this.xhat = new Float32Array(x.length);
    this.inv = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let mean = 0, v = 0;
      for (let j = 0; j < d; j++) mean += x[r * d + j];
      mean /= d;
      for (let j = 0; j < d; j++) v += (x[r * d + j] - mean) ** 2;
      const inv = 1 / Math.sqrt(v / d + 1e-5);
      this.inv[r] = inv;
      for (let j = 0; j < d; j++) {
        const xh = (x[r * d + j] - mean) * inv;
        this.xhat[r * d + j] = xh;
        y[r * d + j] = xh * this.gamma.value[j] + this.beta.value[j];
      }
    }
    return y;
  }

  backward(dy: Float32Array) {
    const d = this.dim, rows = dy.length / d;
    const dx = new Float32Array(dy.length);
    for (let r = 0; r < rows; r++) {
      let m1 = 0, m2 = 0;
      for (let j = 0; j < d; j++) {
        const i = r * d + j;
        this.gamma.grad[j] += dy[i] * this.xhat[i];
        this.beta.grad[j] += dy[i];
        const dxh = dy[i] * this.gamma.value[j];
        m1 += dxh;
        m2 += dxh * this.xhat[i];
      }
      m1 /= d;
      m2 /= d;
      for (let j = 0; j < d; j++) {
        const i = r * d + j;
        dx[i] = this.inv[r] * (dy[i] * this.gamma.value[j] - m1 - this.xhat[i] * m2);
      }
    }
    return dx;
  }
}

/** Per-channel normalization using running statistics, with learnable scale and shift. */
export class BatchNormCore {
  gamma: Param;
  beta: Param;
  mean: Param;
  variance: Param;
  params: Param[];
  private xhat = new Float32Array(0);

  constructor(public channels: number, public momentum: number, rng: Rng) {
    this.gamma = makeParam('gamma', [channels], 'ones', rng, channels, channels);
    this.beta = makeParam('beta', [channels], 'zeros', rng, channels, channels);
    this.mean = { ...makeParam('running_mean', [channels], 'zeros', rng, 1, 1), frozen: true };
    this.variance = { ...makeParam('running_var', [channels], 'ones', rng, 1, 1), frozen: true };
    this.params = [this.gamma, this.beta, this.mean, this.variance];
  }

  forward(x: Float32Array, training: boolean) {
    const c = this.channels, n = x.length / c;
    if (training) {
      const m = this.momentum;
      for (let j = 0; j < c; j++) {
        const rm = this.mean.value[j];
        let mu = 0, v = 0;
        for (let r = 0; r < n; r++) {
          mu += x[r * c + j];
          v += (x[r * c + j] - rm) ** 2;
        }
        // Deviation from the running mean keeps the estimate meaningful for single-sample updates.
        this.mean.value[j] = m * rm + (1 - m) * (mu / n);
        this.variance.value[j] = m * this.variance.value[j] + (1 - m) * (v / n);
      }
    }
    const y = new Float32Array(x.length);
    this.xhat = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const j = i % c;
      const xh = (x[i] - this.mean.value[j]) / Math.sqrt(this.variance.value[j] + 1e-3);
      this.xhat[i] = xh;
      y[i] = xh * this.gamma.value[j] + this.beta.value[j];
    }
    return y;
  }

  backward(dy: Float32Array) {
    const c = this.channels;
    const dx = new Float32Array(dy.length);
    for (let i = 0; i < dy.length; i++) {
      const j = i % c;
      this.gamma.grad[j] += dy[i] * this.xhat[i];
      this.beta.grad[j] += dy[i];
      dx[i] = (dy[i] * this.gamma.value[j]) / Math.sqrt(this.variance.value[j] + 1e-3);
    }
    return dx;
  }
}
