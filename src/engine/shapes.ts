import { parseShape } from './catalog';
import type { LayerSpec, Shape } from './types';

export interface LayerAnalysis {
  id: string;
  inShape: Shape | null;
  outShape: Shape | null;
  params: number;
  error?: string;
  warnings: string[];
  bottleneck?: boolean;
}

export const size = (s: Shape) => s.reduce((a, b) => a * b, 1);
export const fmtShape = (s: Shape | null) => (s ? `[${s.join(', ')}]` : '[?]');

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);

const convOut = (n: number, k: number, s: number, pad: string) =>
  pad === 'same' ? Math.ceil(n / s) : Math.floor((n - k) / s) + 1;

const gates = (cell: string) => (cell === 'LSTM' ? 4 : cell === 'GRU' ? 3 : 1);

export function recurrentParams(cell: string, f: number, u: number) {
  return (f + u + 1) * gates(cell) * u;
}

/** Shape and parameter math for a single layer. Throws a readable message on incompatible input. */
function inferLayer(spec: LayerSpec, inp: Shape): { out: Shape; params: number } {
  const p = spec.params;
  const rank = inp.length;
  const last = inp[rank - 1];
  const needRank = (r: number, what: string) => {
    if (rank !== r) throw new Error(`${spec.type} expects ${what} (rank ${r}) but received [${inp.join(', ')}]`);
  };
  switch (spec.type) {
    case 'Input':
      return { out: inp, params: 0 };
    case 'Dense': {
      const u = num(p.units, 1);
      return { out: [...inp.slice(0, -1), u], params: last * u + (p.useBias === false ? 0 : u) };
    }
    case 'Dropout':
    case 'ReLU':
    case 'Sigmoid':
    case 'Softmax':
    case 'Tanh':
    case 'GELU':
    case 'Custom':
      return { out: inp, params: 0 };
    case 'Flatten':
      return { out: [size(inp)], params: 0 };
    case 'Reshape': {
      const t = parseShape(p.target);
      if (!t) throw new Error('Reshape target shape is invalid');
      if (size(t) !== size(inp)) throw new Error(`Cannot reshape ${size(inp)} values into [${t.join(', ')}] (${size(t)} values)`);
      return { out: t, params: 0 };
    }
    case 'Conv2D': {
      needRank(3, 'an image [height, width, channels]');
      const k = num(p.kernel, 3), s = num(p.stride, 1), f = num(p.filters, 8), pad = String(p.padding);
      const h = convOut(inp[0], k, s, pad), w = convOut(inp[1], k, s, pad);
      if (h <= 0 || w <= 0) throw new Error(`Kernel ${k} is larger than the ${inp[0]}x${inp[1]} input`);
      return { out: [h, w, f], params: k * k * last * f + f };
    }
    case 'Conv1D': {
      needRank(2, 'a sequence [steps, features]');
      const k = num(p.kernel, 3), s = num(p.stride, 1), f = num(p.filters, 8);
      const l = convOut(inp[0], k, s, String(p.padding));
      if (l <= 0) throw new Error(`Kernel ${k} is longer than the sequence (${inp[0]})`);
      return { out: [l, f], params: k * last * f + f };
    }
    case 'MaxPooling':
    case 'AveragePooling': {
      if (rank !== 2 && rank !== 3) throw new Error(`${spec.type} expects [H, W, C] or [T, C] but received [${inp.join(', ')}]`);
      const pool = num(p.pool, 2);
      const out = rank === 3 ? [Math.floor(inp[0] / pool), Math.floor(inp[1] / pool), last] : [Math.floor(inp[0] / pool), last];
      if (out.some((d) => d <= 0)) throw new Error(`Pool size ${pool} is larger than the input`);
      return { out, params: 0 };
    }
    case 'BatchNorm':
      return { out: inp, params: 2 * last };
    case 'LSTM':
    case 'GRU':
    case 'SimpleRNN': {
      needRank(2, 'a sequence [steps, features]');
      const u = num(p.units, 16);
      return { out: p.returnSequences ? [inp[0], u] : [u], params: recurrentParams(spec.type, last, u) };
    }
    case 'Bidirectional': {
      needRank(2, 'a sequence [steps, features]');
      const u = num(p.units, 8);
      return { out: p.returnSequences ? [inp[0], 2 * u] : [2 * u], params: 2 * recurrentParams(String(p.cell), last, u) };
    }
    case 'MultiHeadAttention':
    case 'SelfAttention':
    case 'TransformerBlock': {
      needRank(2, 'a sequence [steps, features]');
      const h = spec.type === 'SelfAttention' ? 1 : num(p.heads, 2);
      if (last % h !== 0) throw new Error(`Feature size ${last} is not divisible by ${h} heads`);
      const d = last;
      const attn = (spec.type === 'SelfAttention' ? 3 : 4) * (d * d + d);
      if (spec.type !== 'TransformerBlock') return { out: inp, params: attn };
      const ff = num(p.ffDim, 32);
      return { out: inp, params: attn + 4 * d + d * ff + ff + ff * d + d };
    }
  }
}

/** Walks an ordered chain of layers, inferring shapes and flagging problems. */
export function analyzeChain(chain: LayerSpec[], targetShape?: Shape | null): LayerAnalysis[] {
  const result: LayerAnalysis[] = [];
  let cur: Shape | null = null;
  chain.forEach((spec, i) => {
    const a: LayerAnalysis = { id: spec.id, inShape: cur, outShape: null, params: 0, warnings: [] };
    result.push(a);
    if (spec.type === 'Input') {
      cur = parseShape(spec.params.shape);
      if (!cur) a.error = 'Input shape is invalid. Use numbers like "8,8,1".';
      a.inShape = cur;
      a.outShape = cur;
      return;
    }
    if (!cur) {
      a.error = 'Waiting for a valid shape from the previous layer';
      return;
    }
    try {
      const { out, params } = inferLayer(spec, cur);
      a.outShape = out;
      a.params = params;
      const isLast = i === chain.length - 1;
      if (spec.type === 'Dense' && !isLast && size(out) * 8 <= size(cur) && size(cur) >= 32) {
        a.bottleneck = true;
        a.warnings.push(`Bottleneck: compresses ${size(cur)} values down to ${size(out)}`);
      }
      if (!isLast && (spec.type === 'Softmax' || spec.params.activation === 'softmax')) {
        a.warnings.push('Softmax is usually only used on the final layer');
      }
      if (params > 1_000_000) a.warnings.push('Very large layer; training in the browser may be slow');
      const declared = parseShape(spec.params.declaredShape);
      if (spec.params.declaredShape && declared && declared.join() !== out.join()) {
        a.error = `Declared shape [${declared.join(', ')}] does not match computed shape [${out.join(', ')}]`;
      }
      cur = a.error ? null : out;
    } catch (e) {
      a.error = (e as Error).message;
      cur = null;
    }
  });
  const last = result[result.length - 1];
  if (targetShape && last?.outShape && last.outShape.join() !== targetShape.join() && size(last.outShape) !== size(targetShape)) {
    last.error = `Output ${fmtShape(last.outShape)} does not match dataset target ${fmtShape(targetShape)}`;
  }
  return result;
}
