import { useEffect, useState } from 'react';
import type { ForwardTrace, Model } from '../../engine/model';
import { ensureModel } from '../../store/actions';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';

export interface ModelRun {
  model: Model;
  trace: ForwardTrace;
}

/** Runs the current model on `x`, re-running whenever the graph or weights change. */
export function useModelRun(x: Float32Array | null) {
  const net = useNetwork();
  const weightsVersion = useStore((s) => s.weightsVersion);
  const [run, setRun] = useState<ModelRun | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!x || !net.valid) {
      setRun(null);
      setError(net.valid ? null : 'Fix the network errors to run inference.');
      return;
    }
    const model = ensureModel(net);
    if (!model) return setError('The model could not be built.');
    if (x.length !== model.inputSize) return setError(`Input has ${x.length} values but the network expects ${model.inputSize}.`);
    try {
      setRun({ model, trace: model.forwardTrace(x) });
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [x, net, weightsVersion]);

  return { run, error, net };
}

export function probabilities(y: Float32Array, act: string) {
  if (act === 'softmax' || act === 'sigmoid') return Array.from(y);
  const max = Math.max(...y);
  const e = Array.from(y, (v) => Math.exp(v - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
