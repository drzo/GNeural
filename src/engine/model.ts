import { createLayer } from './layers';
import { mulberry32 } from './math';
import { size, type LayerAnalysis } from './shapes';
import type { LayerSpec, LayerTrace, Param, RuntimeLayer } from './types';

export type SerializedWeights = Record<string, Record<string, string>>;

export interface ForwardTrace {
  outputs: Float32Array[];
  traces: (LayerTrace | undefined)[];
}

function toBase64(a: Float32Array) {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function fromBase64(s: string) {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export class Model {
  layers: RuntimeLayer[];
  /** Gradient of the loss w.r.t. each layer's output, from the latest backward pass. */
  gradOut: Float32Array[] = [];

  constructor(chain: LayerSpec[], analysis: LayerAnalysis[], seed: number) {
    const rng = mulberry32(seed);
    this.layers = chain.map((spec, i) => createLayer(spec, analysis[i].inShape!, analysis[i].outShape!, rng));
  }

  get params(): Param[] {
    return this.layers.flatMap((l) => l.params);
  }

  get paramCount() {
    return this.params.filter((p) => !p.frozen).reduce((a, p) => a + p.value.length, 0);
  }

  get inputSize() {
    return size(this.layers[0].outShape);
  }

  get outputShape() {
    return this.layers[this.layers.length - 1].outShape;
  }

  /** The activation producing the final output, used by losses to decide logits vs probabilities. */
  get outputActivation(): string {
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const s = this.layers[i].spec;
      if (['Softmax', 'Sigmoid', 'ReLU', 'Tanh', 'GELU'].includes(s.type)) return s.type.toLowerCase();
      if (typeof s.params.activation === 'string') return s.params.activation;
      if (!['Reshape', 'Flatten', 'Dropout'].includes(s.type)) return 'linear';
    }
    return 'linear';
  }

  forward(x: Float32Array, training = false) {
    let h = x;
    for (const l of this.layers) h = l.forward(h, training);
    return h;
  }

  /** Forward pass that records every layer's output, optionally stopping after layer `stopAt`. */
  forwardTrace(x: Float32Array, stopAt = this.layers.length - 1): ForwardTrace {
    const outputs: Float32Array[] = [];
    const traces: (LayerTrace | undefined)[] = [];
    let h = x;
    for (let i = 0; i <= stopAt; i++) {
      h = this.layers[i].forward(h, false);
      outputs.push(h);
      traces.push(this.layers[i].trace?.());
    }
    return { outputs, traces };
  }

  backward(dy: Float32Array) {
    this.gradOut = new Array(this.layers.length);
    let g = dy;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      this.gradOut[i] = g;
      g = this.layers[i].backward(g);
    }
    return g;
  }

  zeroGrad() {
    for (const p of this.params) p.grad.fill(0);
  }

  serialize(): SerializedWeights {
    const out: SerializedWeights = {};
    for (const l of this.layers) {
      if (!l.params.length) continue;
      out[l.spec.id] = Object.fromEntries(l.params.map((p) => [p.name, toBase64(p.value)]));
    }
    return out;
  }

  /** Loads matching tensors; returns how many layers were restored. */
  load(weights: SerializedWeights) {
    let restored = 0;
    for (const l of this.layers) {
      const saved = weights[l.spec.id];
      if (!saved) continue;
      let ok = true;
      for (const p of l.params) {
        const enc = saved[p.name];
        const arr = enc ? fromBase64(enc) : null;
        if (arr && arr.length === p.value.length) p.value.set(arr);
        else ok = false;
      }
      if (ok) restored++;
    }
    return restored;
  }
}
