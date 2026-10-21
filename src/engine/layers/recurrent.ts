import { makeParam, matmul, matmulTB, sigmoid, type Rng } from '../math';
import type { Param } from '../types';
import type { InitOpts } from './core';

export type Cell = 'LSTM' | 'GRU' | 'SimpleRNN';

interface StepCache {
  x: Float32Array;
  hPrev: Float32Array;
  cPrev: Float32Array;
  gates: Float32Array;
  rhN: Float32Array;
  c: Float32Array;
  tc: Float32Array;
  h: Float32Array;
}

/**
 * Runs one recurrent cell across a [T, F] sequence (optionally reversed) and returns [T, U],
 * where output row t is the state after reading step t.
 */
export class RecurrentCore {
  W: Param;
  R: Param;
  b: Param;
  params: Param[];
  private G: number;
  private steps: StepCache[] = [];

  constructor(
    public cell: Cell,
    public T: number,
    public F: number,
    public U: number,
    public reverse: boolean,
    rng: Rng,
    o: InitOpts,
    prefix = '',
  ) {
    this.G = cell === 'LSTM' ? 4 : cell === 'GRU' ? 3 : 1;
    const GU = this.G * U;
    this.W = makeParam(`${prefix}kernel`, [F, GU], o.init, rng, F, GU, o.l1, o.l2);
    this.R = makeParam(`${prefix}recurrent`, [U, GU], o.init === 'zeros' ? 'zeros' : 'glorot_uniform', rng, U, GU, o.l1, o.l2);
    this.b = makeParam(`${prefix}bias`, [GU], 'zeros', rng, F, GU);
    if (cell === 'LSTM') for (let j = U; j < 2 * U; j++) this.b.value[j] = 1;
    this.params = [this.W, this.R, this.b];
  }

  private order() {
    const idx = Array.from({ length: this.T }, (_, i) => i);
    return this.reverse ? idx.reverse() : idx;
  }

  forward(x: Float32Array) {
    const { F, U, G } = this;
    const GU = G * U;
    const out = new Float32Array(this.T * U);
    let h = new Float32Array(U);
    let c = new Float32Array(U);
    this.steps = [];
    for (const t of this.order()) {
      const xt = x.subarray(t * F, (t + 1) * F);
      const ax = matmul(xt, this.W.value, 1, F, GU);
      for (let j = 0; j < GU; j++) ax[j] += this.b.value[j];
      const rh = matmul(h, this.R.value, 1, U, GU);
      const gates = new Float32Array(GU);
      const hn = new Float32Array(U);
      const cn = new Float32Array(U);
      const tc = new Float32Array(U);
      let rhN = new Float32Array(0);
      if (this.cell === 'SimpleRNN') {
        for (let j = 0; j < U; j++) hn[j] = gates[j] = Math.tanh(ax[j] + rh[j]);
      } else if (this.cell === 'LSTM') {
        for (let j = 0; j < U; j++) {
          const i = sigmoid(ax[j] + rh[j]);
          const f = sigmoid(ax[U + j] + rh[U + j]);
          const g = Math.tanh(ax[2 * U + j] + rh[2 * U + j]);
          const o = sigmoid(ax[3 * U + j] + rh[3 * U + j]);
          gates[j] = i, gates[U + j] = f, gates[2 * U + j] = g, gates[3 * U + j] = o;
          cn[j] = f * c[j] + i * g;
          tc[j] = Math.tanh(cn[j]);
          hn[j] = o * tc[j];
        }
      } else {
        rhN = rh.slice(2 * U);
        for (let j = 0; j < U; j++) {
          const z = sigmoid(ax[j] + rh[j]);
          const r = sigmoid(ax[U + j] + rh[U + j]);
          const n = Math.tanh(ax[2 * U + j] + r * rhN[j]);
          gates[j] = z, gates[U + j] = r, gates[2 * U + j] = n;
          hn[j] = (1 - z) * n + z * h[j];
        }
      }
      this.steps.push({ x: xt, hPrev: h, cPrev: c, gates, rhN, c: cn, tc, h: hn });
      out.set(hn, t * U);
      h = hn;
      c = cn;
    }
    return out;
  }

  /** dOut is [T, U] aligned with forward output rows. Returns dX [T, F]. */
  backward(dOut: Float32Array) {
    const { F, U, G } = this;
    const GU = G * U;
    const dx = new Float32Array(this.T * F);
    let dhNext = new Float32Array(U);
    let dcNext = new Float32Array(U);
    const order = this.order();
    for (let s = this.steps.length - 1; s >= 0; s--) {
      const t = order[s];
      const st = this.steps[s];
      const dh = new Float32Array(U);
      for (let j = 0; j < U; j++) dh[j] = dOut[t * U + j] + dhNext[j];
      const dax = new Float32Array(GU);
      const drh = new Float32Array(GU);
      const dhPrev = new Float32Array(U);
      const dcPrev = new Float32Array(U);
      const g = st.gates;
      if (this.cell === 'SimpleRNN') {
        for (let j = 0; j < U; j++) dax[j] = drh[j] = dh[j] * (1 - g[j] * g[j]);
      } else if (this.cell === 'LSTM') {
        for (let j = 0; j < U; j++) {
          const i = g[j], f = g[U + j], gg = g[2 * U + j], o = g[3 * U + j];
          const dc = dh[j] * o * (1 - st.tc[j] * st.tc[j]) + dcNext[j];
          dax[3 * U + j] = dh[j] * st.tc[j] * o * (1 - o);
          dax[j] = dc * gg * i * (1 - i);
          dax[U + j] = dc * st.cPrev[j] * f * (1 - f);
          dax[2 * U + j] = dc * i * (1 - gg * gg);
          dcPrev[j] = dc * f;
        }
        drh.set(dax);
      } else {
        for (let j = 0; j < U; j++) {
          const z = g[j], r = g[U + j], n = g[2 * U + j];
          const dn = dh[j] * (1 - z) * (1 - n * n);
          const dz = dh[j] * (st.hPrev[j] - n) * z * (1 - z);
          const dr = dn * st.rhN[j] * r * (1 - r);
          dax[j] = dz, dax[U + j] = dr, dax[2 * U + j] = dn;
          drh[j] = dz, drh[U + j] = dr, drh[2 * U + j] = dn * r;
          dhPrev[j] = dh[j] * z;
        }
      }
      for (let p = 0; p < F; p++) {
        const xv = st.x[p];
        if (xv === 0) continue;
        for (let j = 0; j < GU; j++) this.W.grad[p * GU + j] += xv * dax[j];
      }
      for (let p = 0; p < U; p++) {
        const hv = st.hPrev[p];
        if (hv === 0) continue;
        for (let j = 0; j < GU; j++) this.R.grad[p * GU + j] += hv * drh[j];
      }
      for (let j = 0; j < GU; j++) this.b.grad[j] += dax[j];
      dx.set(matmulTB(dax, this.W.value, 1, GU, F), t * F);
      const dhR = matmulTB(drh, this.R.value, 1, GU, U);
      for (let j = 0; j < U; j++) dhPrev[j] += dhR[j];
      dhNext = dhPrev;
      dcNext = dcPrev;
    }
    return dx;
  }

  /** Index of the output row holding the final state. */
  get finalRow() {
    return this.reverse ? 0 : this.T - 1;
  }
}
