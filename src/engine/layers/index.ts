import { activate, activateBackward, type Rng } from '../math';
import type { Activation, Initializer, LayerSpec, LayerTrace, Param, RuntimeLayer, Shape } from '../types';
import { size } from '../shapes';
import { AttentionCore, TransformerCore } from './attention';
import { ConvCore, PoolCore, type Grid } from './conv';
import { BatchNormCore, Linear, type InitOpts } from './core';
import { RecurrentCore, type Cell } from './recurrent';

interface Core {
  params: Param[];
  forward(x: Float32Array, training: boolean): Float32Array;
  backward(dy: Float32Array): Float32Array;
}

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);

function make(spec: LayerSpec, inShape: Shape, outShape: Shape, core: Core, trace?: () => LayerTrace): RuntimeLayer {
  return {
    spec,
    inShape,
    outShape,
    params: core.params,
    forward: (x, t) => core.forward(x, t),
    backward: (dy) => core.backward(dy),
    trace,
  };
}

const identity: Core = { params: [], forward: (x) => x, backward: (d) => d };

function activationCore(act: Activation, dim: number): Core {
  let z: Float32Array = new Float32Array(0), y: Float32Array = new Float32Array(0);
  return {
    params: [],
    forward: (x) => ((z = x), (y = activate(act, x, dim))),
    backward: (dy) => activateBackward(act, z, y, dy, dim),
  };
}

function dropoutCore(rate: number, rng: Rng): Core {
  let mask: Float32Array | null = null;
  return {
    params: [],
    forward(x, training) {
      if (!training || rate <= 0) return (mask = null), x;
      mask = new Float32Array(x.length);
      const y = new Float32Array(x.length);
      const keep = 1 / (1 - rate);
      for (let i = 0; i < x.length; i++) if (rng() >= rate) (mask[i] = keep), (y[i] = x[i] * keep);
      return y;
    },
    backward(dy) {
      if (!mask) return dy;
      const dx = new Float32Array(dy.length);
      for (let i = 0; i < dy.length; i++) dx[i] = dy[i] * mask[i];
      return dx;
    },
  };
}

export class CustomCodeError extends Error {}

function customCore(code: string, trusted: boolean): Core {
  if (!trusted) return identity;
  let fn: (x: number) => number;
  try {
    fn = new Function('x', code) as (x: number) => number;
    const probe = fn(0.5);
    if (typeof probe !== 'number') throw new Error('Expression must return a number');
  } catch (e) {
    throw new CustomCodeError(`Custom layer code failed: ${(e as Error).message}`);
  }
  let x0: Float32Array = new Float32Array(0);
  const h = 1e-3;
  return {
    params: [],
    forward(x) {
      x0 = x;
      const y = new Float32Array(x.length);
      for (let i = 0; i < x.length; i++) y[i] = fn(x[i]);
      return y;
    },
    backward(dy) {
      const dx = new Float32Array(dy.length);
      for (let i = 0; i < dy.length; i++) dx[i] = (dy[i] * (fn(x0[i] + h) - fn(x0[i] - h))) / (2 * h);
      return dx;
    },
  };
}

function recurrentLayer(spec: LayerSpec, inShape: Shape, outShape: Shape, rng: Rng, o: InitOpts, cell: Cell, bidirectional: boolean): RuntimeLayer {
  const [T, F] = inShape;
  const U = num(spec.params.units, 16);
  const seq = Boolean(spec.params.returnSequences);
  const cores = bidirectional
    ? [new RecurrentCore(cell, T, F, U, false, rng, o, 'fwd_'), new RecurrentCore(cell, T, F, U, true, rng, o, 'bwd_')]
    : [new RecurrentCore(cell, T, F, U, false, rng, o)];
  const D = U * cores.length;
  return make(spec, inShape, outShape, {
    params: cores.flatMap((c) => c.params),
    forward(x) {
      const outs = cores.map((c) => c.forward(x));
      const y = new Float32Array(seq ? T * D : D);
      outs.forEach((out, ci) => {
        if (seq) for (let t = 0; t < T; t++) y.set(out.subarray(t * U, (t + 1) * U), t * D + ci * U);
        else y.set(out.subarray(cores[ci].finalRow * U, (cores[ci].finalRow + 1) * U), ci * U);
      });
      return y;
    },
    backward(dy) {
      const dx = new Float32Array(T * F);
      cores.forEach((c, ci) => {
        const dOut = new Float32Array(T * U);
        if (seq) for (let t = 0; t < T; t++) dOut.set(dy.subarray(t * D + ci * U, t * D + (ci + 1) * U), t * U);
        else dOut.set(dy.subarray(ci * U, (ci + 1) * U), c.finalRow * U);
        const d = c.backward(dOut);
        for (let i = 0; i < dx.length; i++) dx[i] += d[i];
      });
      return dx;
    },
  });
}

export function createLayer(spec: LayerSpec, inShape: Shape, outShape: Shape, rng: Rng): RuntimeLayer {
  const p = spec.params;
  const o: InitOpts = {
    init: (p.initializer as Initializer) ?? 'glorot_uniform',
    l1: num(p.l1, 0),
    l2: num(p.l2, 0),
  };
  const last = inShape[inShape.length - 1];
  const act = (p.activation as Activation) ?? 'linear';
  switch (spec.type) {
    case 'Input':
    case 'Flatten':
    case 'Reshape':
      return make(spec, inShape, outShape, identity);
    case 'Dense':
      return make(spec, inShape, outShape, new Linear(last, num(p.units, 1), act, rng, o, p.useBias !== false));
    case 'Dropout':
      return make(spec, inShape, outShape, dropoutCore(num(p.rate, 0), rng));
    case 'ReLU':
    case 'Sigmoid':
    case 'Tanh':
    case 'GELU':
    case 'Softmax':
      return make(spec, inShape, outShape, activationCore(spec.type.toLowerCase() as Activation, last));
    case 'Custom':
      return make(spec, inShape, outShape, customCore(String(p.code ?? 'return x;'), p.trusted === true));
    case 'BatchNorm':
      return make(spec, inShape, outShape, new BatchNormCore(last, num(p.momentum, 0.9), rng));
    case 'Conv2D':
    case 'Conv1D': {
      const is2d = spec.type === 'Conv2D';
      const g: Grid = is2d ? { h: inShape[0], w: inShape[1], c: inShape[2] } : { h: 1, w: inShape[0], c: inShape[1] };
      const k = num(p.kernel, 3), s = num(p.stride, 1);
      const core = new ConvCore(g, is2d ? k : 1, k, is2d ? s : 1, s, num(p.filters, 8), String(p.padding), act, rng, o);
      return make(spec, inShape, outShape, core, () => ({ featureShape: is2d ? outShape : [1, ...outShape] }));
    }
    case 'MaxPooling':
    case 'AveragePooling': {
      const pool = num(p.pool, 2);
      const is2d = inShape.length === 3;
      const g: Grid = is2d ? { h: inShape[0], w: inShape[1], c: inShape[2] } : { h: 1, w: inShape[0], c: inShape[1] };
      const core = new PoolCore(g, is2d ? pool : 1, pool, spec.type === 'MaxPooling' ? 'max' : 'avg');
      return make(spec, inShape, outShape, { params: [], forward: (x) => core.forward(x), backward: (d) => core.backward(d) }, () => ({
        featureShape: is2d ? outShape : [1, ...outShape],
      }));
    }
    case 'LSTM':
    case 'GRU':
    case 'SimpleRNN':
      return recurrentLayer(spec, inShape, outShape, rng, o, spec.type, false);
    case 'Bidirectional':
      return recurrentLayer(spec, inShape, outShape, rng, o, (p.cell as Cell) ?? 'LSTM', true);
    case 'MultiHeadAttention':
    case 'SelfAttention': {
      const multi = spec.type === 'MultiHeadAttention';
      const core = new AttentionCore(inShape[0], last, multi ? num(p.heads, 2) : 1, multi, rng, o);
      return make(spec, inShape, outShape, core, () => ({ attention: core.attn }));
    }
    case 'TransformerBlock': {
      const core = new TransformerCore(inShape[0], last, num(p.heads, 2), num(p.ffDim, 32), rng, o);
      return make(spec, inShape, outShape, core, () => ({ attention: core.mha.attn }));
    }
  }
}

export const outputSize = (l: RuntimeLayer) => size(l.outShape);
