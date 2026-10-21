import type { Rng } from '../math';
import type { Param } from '../types';
import { LayerNorm, Linear, type InitOpts } from './core';

/** Multi-head scaled dot-product self-attention over a [T, D] sequence. */
export class AttentionCore {
  q: Linear;
  k: Linear;
  v: Linear;
  o: Linear | null;
  params: Param[];
  attn: Float32Array[] = [];
  private Q: Float32Array = new Float32Array(0);
  private K: Float32Array = new Float32Array(0);
  private V: Float32Array = new Float32Array(0);
  private dk: number;

  constructor(public T: number, public D: number, public heads: number, project: boolean, rng: Rng, opts: InitOpts, prefix = '') {
    this.q = new Linear(D, D, 'linear', rng, opts, true, `${prefix}query_`);
    this.k = new Linear(D, D, 'linear', rng, opts, true, `${prefix}key_`);
    this.v = new Linear(D, D, 'linear', rng, opts, true, `${prefix}value_`);
    this.o = project ? new Linear(D, D, 'linear', rng, opts, true, `${prefix}out_`) : null;
    this.dk = D / heads;
    this.params = [...this.q.params, ...this.k.params, ...this.v.params, ...(this.o?.params ?? [])];
  }

  forward(x: Float32Array) {
    const { T, D, dk } = this;
    this.Q = this.q.forward(x);
    this.K = this.k.forward(x);
    this.V = this.v.forward(x);
    const O = new Float32Array(T * D);
    const scale = 1 / Math.sqrt(dk);
    this.attn = [];
    for (let h = 0; h < this.heads; h++) {
      const off = h * dk;
      const A = new Float32Array(T * T);
      for (let i = 0; i < T; i++) {
        let max = -Infinity;
        for (let j = 0; j < T; j++) {
          let s = 0;
          for (let c = 0; c < dk; c++) s += this.Q[i * D + off + c] * this.K[j * D + off + c];
          A[i * T + j] = s * scale;
          max = Math.max(max, A[i * T + j]);
        }
        let sum = 0;
        for (let j = 0; j < T; j++) sum += A[i * T + j] = Math.exp(A[i * T + j] - max);
        for (let j = 0; j < T; j++) A[i * T + j] /= sum;
        for (let j = 0; j < T; j++) {
          const a = A[i * T + j];
          for (let c = 0; c < dk; c++) O[i * D + off + c] += a * this.V[j * D + off + c];
        }
      }
      this.attn.push(A);
    }
    return this.o ? this.o.forward(O) : O;
  }

  backward(dy: Float32Array) {
    const { T, D, dk } = this;
    const dO = this.o ? this.o.backward(dy) : dy;
    const dQ = new Float32Array(T * D), dK = new Float32Array(T * D), dV = new Float32Array(T * D);
    const scale = 1 / Math.sqrt(dk);
    for (let h = 0; h < this.heads; h++) {
      const off = h * dk;
      const A = this.attn[h];
      for (let i = 0; i < T; i++) {
        const dA = new Float32Array(T);
        let dot = 0;
        for (let j = 0; j < T; j++) {
          let s = 0;
          for (let c = 0; c < dk; c++) {
            s += dO[i * D + off + c] * this.V[j * D + off + c];
            dV[j * D + off + c] += A[i * T + j] * dO[i * D + off + c];
          }
          dA[j] = s;
          dot += s * A[i * T + j];
        }
        for (let j = 0; j < T; j++) {
          const ds = A[i * T + j] * (dA[j] - dot) * scale;
          if (ds === 0) continue;
          for (let c = 0; c < dk; c++) {
            dQ[i * D + off + c] += ds * this.K[j * D + off + c];
            dK[j * D + off + c] += ds * this.Q[i * D + off + c];
          }
        }
      }
    }
    const dx = this.q.backward(dQ);
    const dxk = this.k.backward(dK);
    const dxv = this.v.backward(dV);
    for (let i = 0; i < dx.length; i++) dx[i] += dxk[i] + dxv[i];
    return dx;
  }
}

/** Post-norm Transformer encoder block: LN(x + MHA(x)) then LN(h + FFN(h)). */
export class TransformerCore {
  mha: AttentionCore;
  ln1: LayerNorm;
  ln2: LayerNorm;
  ff1: Linear;
  ff2: Linear;
  params: Param[];

  constructor(T: number, D: number, heads: number, ffDim: number, rng: Rng, opts: InitOpts) {
    this.mha = new AttentionCore(T, D, heads, true, rng, opts, 'attn_');
    this.ln1 = new LayerNorm(D, rng, 'ln1_');
    this.ff1 = new Linear(D, ffDim, 'relu', rng, opts, true, 'ffn1_');
    this.ff2 = new Linear(ffDim, D, 'linear', rng, opts, true, 'ffn2_');
    this.ln2 = new LayerNorm(D, rng, 'ln2_');
    this.params = [...this.mha.params, ...this.ln1.params, ...this.ff1.params, ...this.ff2.params, ...this.ln2.params];
  }

  forward(x: Float32Array) {
    const a = this.mha.forward(x);
    const r1 = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) r1[i] = x[i] + a[i];
    const h = this.ln1.forward(r1);
    const f = this.ff2.forward(this.ff1.forward(h));
    const r2 = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) r2[i] = h[i] + f[i];
    return this.ln2.forward(r2);
  }

  backward(dy: Float32Array) {
    const dr2 = this.ln2.backward(dy);
    const dh = this.ff1.backward(this.ff2.backward(dr2));
    for (let i = 0; i < dh.length; i++) dh[i] += dr2[i];
    const dr1 = this.ln1.backward(dh);
    const dx = this.mha.backward(dr1);
    for (let i = 0; i < dx.length; i++) dx[i] += dr1[i];
    return dx;
  }
}
