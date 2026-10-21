import { activate, activateBackward, makeParam, type Rng } from '../math';
import type { Activation, Param } from '../types';
import type { InitOpts } from './core';

export interface Grid {
  h: number;
  w: number;
  c: number;
}

const samePad = (n: number, k: number, s: number) => {
  const out = Math.ceil(n / s);
  return Math.floor(Math.max((out - 1) * s + k - n, 0) / 2);
};

/** 2D convolution over [H, W, C]; 1D convolution uses H = 1 and kh = 1. */
export class ConvCore {
  W: Param;
  b: Param;
  params: Param[];
  oh: number;
  ow: number;
  private pt: number;
  private pl: number;
  private x: Float32Array = new Float32Array(0);
  private z: Float32Array = new Float32Array(0);
  private y: Float32Array = new Float32Array(0);

  constructor(
    public g: Grid,
    public kh: number,
    public kw: number,
    public sh: number,
    public sw: number,
    public filters: number,
    padding: string,
    public act: Activation,
    rng: Rng,
    o: InitOpts,
  ) {
    const fanIn = kh * kw * g.c;
    this.W = makeParam('kernel', [kh, kw, g.c, filters], o.init, rng, fanIn, kh * kw * filters, o.l1, o.l2);
    this.b = makeParam('bias', [filters], o.init === 'gneural' ? 'ones' : 'zeros', rng, fanIn, filters);
    this.params = [this.W, this.b];
    const same = padding === 'same';
    this.oh = same ? Math.ceil(g.h / sh) : Math.floor((g.h - kh) / sh) + 1;
    this.ow = same ? Math.ceil(g.w / sw) : Math.floor((g.w - kw) / sw) + 1;
    this.pt = same ? samePad(g.h, kh, sh) : 0;
    this.pl = same ? samePad(g.w, kw, sw) : 0;
  }

  forward(x: Float32Array) {
    const { h, w, c } = this.g, K = this.filters;
    const W = this.W.value;
    const z = new Float32Array(this.oh * this.ow * K);
    for (let oy = 0; oy < this.oh; oy++) {
      for (let ox = 0; ox < this.ow; ox++) {
        const zo = (oy * this.ow + ox) * K;
        for (let k = 0; k < K; k++) z[zo + k] = this.b.value[k];
        for (let ky = 0; ky < this.kh; ky++) {
          const iy = oy * this.sh + ky - this.pt;
          if (iy < 0 || iy >= h) continue;
          for (let kx = 0; kx < this.kw; kx++) {
            const ix = ox * this.sw + kx - this.pl;
            if (ix < 0 || ix >= w) continue;
            const xo = (iy * w + ix) * c, wo = (ky * this.kw + kx) * c * K;
            for (let ci = 0; ci < c; ci++) {
              const xv = x[xo + ci];
              if (xv === 0) continue;
              const wb = wo + ci * K;
              for (let k = 0; k < K; k++) z[zo + k] += xv * W[wb + k];
            }
          }
        }
      }
    }
    this.x = x;
    this.z = z;
    this.y = activate(this.act, z, K);
    return this.y;
  }

  backward(dy: Float32Array) {
    const { h, w, c } = this.g, K = this.filters;
    const dz = activateBackward(this.act, this.z, this.y, dy, K);
    const dx = new Float32Array(this.x.length);
    const W = this.W.value, dW = this.W.grad, db = this.b.grad, x = this.x;
    for (let oy = 0; oy < this.oh; oy++) {
      for (let ox = 0; ox < this.ow; ox++) {
        const zo = (oy * this.ow + ox) * K;
        for (let k = 0; k < K; k++) db[k] += dz[zo + k];
        for (let ky = 0; ky < this.kh; ky++) {
          const iy = oy * this.sh + ky - this.pt;
          if (iy < 0 || iy >= h) continue;
          for (let kx = 0; kx < this.kw; kx++) {
            const ix = ox * this.sw + kx - this.pl;
            if (ix < 0 || ix >= w) continue;
            const xo = (iy * w + ix) * c, wo = (ky * this.kw + kx) * c * K;
            for (let ci = 0; ci < c; ci++) {
              const xv = x[xo + ci];
              const wb = wo + ci * K;
              let acc = 0;
              for (let k = 0; k < K; k++) {
                const g = dz[zo + k];
                dW[wb + k] += xv * g;
                acc += W[wb + k] * g;
              }
              dx[xo + ci] += acc;
            }
          }
        }
      }
    }
    return dx;
  }
}

export class PoolCore {
  oh: number;
  ow: number;
  private argmax = new Int32Array(0);
  private inLen = 0;

  constructor(public g: Grid, public ph: number, public pw: number, public mode: 'max' | 'avg') {
    this.oh = Math.floor(g.h / ph);
    this.ow = Math.floor(g.w / pw);
  }

  forward(x: Float32Array) {
    const { w, c } = this.g;
    const y = new Float32Array(this.oh * this.ow * c);
    this.argmax = new Int32Array(y.length);
    this.inLen = x.length;
    const area = this.ph * this.pw;
    for (let oy = 0; oy < this.oh; oy++) {
      for (let ox = 0; ox < this.ow; ox++) {
        for (let ci = 0; ci < c; ci++) {
          let best = -Infinity, bi = 0, sum = 0;
          for (let ky = 0; ky < this.ph; ky++) {
            for (let kx = 0; kx < this.pw; kx++) {
              const i = ((oy * this.ph + ky) * w + ox * this.pw + kx) * c + ci;
              sum += x[i];
              if (x[i] > best) (best = x[i]), (bi = i);
            }
          }
          const o = (oy * this.ow + ox) * c + ci;
          y[o] = this.mode === 'max' ? best : sum / area;
          this.argmax[o] = bi;
        }
      }
    }
    return y;
  }

  backward(dy: Float32Array) {
    const { w, c } = this.g;
    const dx = new Float32Array(this.inLen);
    const area = this.ph * this.pw;
    for (let oy = 0; oy < this.oh; oy++) {
      for (let ox = 0; ox < this.ow; ox++) {
        for (let ci = 0; ci < c; ci++) {
          const o = (oy * this.ow + ox) * c + ci;
          if (this.mode === 'max') {
            dx[this.argmax[o]] += dy[o];
            continue;
          }
          for (let ky = 0; ky < this.ph; ky++) {
            for (let kx = 0; kx < this.pw; kx++) dx[((oy * this.ph + ky) * w + ox * this.pw + kx) * c + ci] += dy[o] / area;
          }
        }
      }
    }
    return dx;
  }
}
