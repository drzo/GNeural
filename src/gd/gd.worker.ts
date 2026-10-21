import './shims/node-globals.js';
import type { GneuralArch, GneuralWeights, Sample, WorkerRequest, WorkerResponse } from './protocol';
import type { MLP as MLPType } from 'gneural';

type Counter = ReturnType<MLPType['predict']>['input'][number];

const post = (msg: WorkerResponse) => (self as unknown as Worker).postMessage(msg);

async function loadLibrary() {
  await import('@g-js-api/g.js');
  return import('gneural');
}

function createNetwork(lib: Awaited<ReturnType<typeof loadLibrary>>, arch: GneuralArch): MLPType {
  if (arch.kind === 'mlp') return new lib.MLP(arch.inputs, arch.hidden[0], arch.outputs);
  return new lib.MLPDense(arch.inputs, arch.hidden.length === 1 ? arch.hidden[0] : arch.hidden, arch.outputs);
}

const copy = (W: number[][]) => W.map((r) => [...r]);

function applyWeights(net: MLPType, arch: GneuralArch, w: GneuralWeights) {
  const out = w.layers[w.layers.length - 1];
  net.weightsHiddenOutput = copy(out.W);
  net.biasOutput = [...out.b];
  if (arch.kind === 'mlp') {
    net.weightsInputHidden = copy(w.layers[0].W);
    net.biasHidden = [...w.layers[0].b];
    return;
  }
  const dense = net as unknown as { weightsHidden: number[][][]; biasHidden: number[][] };
  dense.weightsHidden = w.layers.slice(0, -1).map((l) => copy(l.W));
  dense.biasHidden = w.layers.slice(0, -1).map((l) => [...l.b]);
}

function readWeights(net: MLPType, arch: GneuralArch): GneuralWeights {
  const out = { W: copy(net.weightsHiddenOutput), b: [...net.biasOutput] };
  if (arch.kind === 'mlp') {
    return { layers: [{ W: copy(net.weightsInputHidden), b: [...(net.biasHidden as number[])] }, out] };
  }
  const dense = net as unknown as { weightsHidden: number[][][]; biasHidden: number[][] };
  return { layers: [...dense.weightsHidden.map((W, i) => ({ W: copy(W), b: [...dense.biasHidden[i]] })), out] };
}

function meanError(net: MLPType, data: Sample[]) {
  if (!data.length) return NaN;
  let total = 0;
  for (const s of data) {
    net.feedForward(s.input);
    total += net.calculateError(s.output);
  }
  return total / data.length;
}

async function exportLevel(net: MLPType, req: Extract<WorkerRequest, { type: 'export' }>) {
  applyWeights(net, req.arch, req.weights);
  net.feedForward(req.input);
  const outputs = [...net.outputOutputs];
  const layers = net.predict(...req.input);
  const sp = req.spacing;
  const hidden = layers.hidden as Counter[] | Counter[][];
  const rows: Counter[][] = Array.isArray(hidden[0]) ? (hidden as Counter[][]) : [hidden as Counter[]];
  layers.input.forEach((c, i) => c.display(i * sp + 45, 165));
  rows.forEach((row, r) => row.forEach((c, i) => c.display(i * sp + 45, 135 - r * 30)));
  layers.output.forEach((c, i) => c.display(i * sp + 45, 135 - rows.length * 30));
  const g = globalThis as unknown as { $: { exportConfig: (c: unknown) => Promise<string> } };
  const levelString = await g.$.exportConfig({ type: 'levelstring', options: { info: false } });
  const groups = new Set<string>();
  for (const m of levelString.matchAll(/(?:^|,)57,([\d.]+)/g)) m[1].split('.').forEach((id) => groups.add(id));
  post({ type: 'exported', levelString, objects: levelString.split(';').length - 1, groups: groups.size, outputs });
}

async function handle(req: WorkerRequest) {
  const lib = await loadLibrary();
  const net = createNetwork(lib, req.arch);
  if (req.type === 'export') return exportLevel(net, req);
  if (req.weights) applyWeights(net, req.arch, req.weights);
  for (let epoch = 1; epoch <= req.epochs; epoch++) {
    net.train(req.train, 1, req.learningRate);
    if (epoch % req.reportEvery === 0 || epoch === req.epochs) {
      post({
        type: 'progress',
        epoch,
        trainLoss: meanError(net, req.train),
        valLoss: meanError(net, req.val),
        weights: readWeights(net, req.arch),
      });
    }
  }
  post({ type: 'trained', weights: readWeights(net, req.arch) });
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  handle(e.data).catch((err: unknown) => post({ type: 'error', message: err instanceof Error ? err.message : String(err) }));
};
