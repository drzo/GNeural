import type { Model } from '../engine/model';
import type { LayerAnalysis } from '../engine/shapes';
import type { LayerSpec } from '../engine/types';
import type { GneuralArch, GneuralWeights, WorkerRequest, WorkerResponse } from './protocol';
import GdWorker from './gd.worker?worker';

export type Compatibility = { ok: true; arch: GneuralArch } | { ok: false; reason: string };

/** Gneural runs plain ReLU MLPs: Input[n] -> Dense(relu)+ -> Dense(linear for one hidden layer, else relu). */
export function gneuralCompatibility(chain: LayerSpec[], analysis: LayerAnalysis[]): Compatibility {
  if (analysis.some((a) => a.error)) return { ok: false, reason: 'Fix the network errors first.' };
  const layers = chain.filter((l) => l.type !== 'Dropout');
  const [input, ...dense] = layers;
  if (!input || input.type !== 'Input') return { ok: false, reason: 'The network needs an Input layer.' };
  const inShape = analysis[0].outShape;
  if (!inShape || inShape.length !== 1) return { ok: false, reason: 'Geometry Dash export needs a flat input like [2], not an image or sequence.' };
  if (dense.length < 2) return { ok: false, reason: 'Needs at least one hidden Dense layer and one output Dense layer.' };
  const bad = dense.find((l) => l.type !== 'Dense');
  if (bad) return { ok: false, reason: `"${bad.name}" (${bad.type}) cannot run in Geometry Dash. Only Dense and Dropout layers are supported.` };
  if (dense.some((l) => l.params.useBias === false)) return { ok: false, reason: 'All Dense layers must use a bias.' };
  const hidden = dense.slice(0, -1);
  const out = dense[dense.length - 1];
  const nonRelu = hidden.find((l) => l.params.activation !== 'relu');
  if (nonRelu) return { ok: false, reason: `Hidden layer "${nonRelu.name}" must use ReLU, the only activation Gneural implements.` };
  const outAct = out.params.activation;
  const units = (l: LayerSpec) => Number(l.params.units);
  const arch = (kind: GneuralArch['kind']): Compatibility => ({
    ok: true,
    arch: { inputs: inShape[0], hidden: hidden.map(units), outputs: units(out), kind },
  });
  if (hidden.length === 1 && outAct === 'linear') return arch('mlp');
  if (outAct === 'relu') return arch('dense');
  return { ok: false, reason: 'The output layer must use ReLU (or Linear with exactly one hidden layer).' };
}

function denseLayers(model: Model) {
  return model.layers.filter((l) => l.spec.type === 'Dense');
}

export function toGneuralWeights(model: Model): GneuralWeights {
  return {
    layers: denseLayers(model).map((l) => {
      const [W, b] = l.params;
      const [rows, cols] = W.shape;
      return {
        W: Array.from({ length: rows }, (_, i) => Array.from(W.value.subarray(i * cols, (i + 1) * cols))),
        b: Array.from(b.value),
      };
    }),
  };
}

export function applyGneuralWeights(model: Model, w: GneuralWeights) {
  denseLayers(model).forEach((l, i) => {
    const src = w.layers[i];
    if (!src) return;
    const [W, b] = l.params;
    W.value.set(src.W.flat());
    b.value.set(src.b);
  });
}

/** Runs one request in a fresh worker (g.js keeps global level state, so every job gets a clean slate). */
export function runGneural(req: WorkerRequest, onMessage: (m: WorkerResponse) => void) {
  const worker = new GdWorker();
  let finish = () => {};
  const done = new Promise<void>((resolve) => {
    finish = () => {
      worker.terminate();
      resolve();
    };
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      onMessage(e.data);
      if (e.data.type !== 'progress') finish();
    };
    worker.onerror = (e) => {
      onMessage({ type: 'error', message: e.message || 'The Geometry Dash worker crashed' });
      finish();
    };
  });
  worker.postMessage(req);
  return { done, cancel: () => finish() };
}
