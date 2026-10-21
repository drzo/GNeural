import type { Edge } from '@xyflow/react';
import { makeAugmenter, splitDataset } from '../data/datasets';
import { Model, type SerializedWeights } from '../engine/model';
import { TrainingRun } from '../engine/trainer';
import type { Sample } from '../engine/types';
import { applyGneuralWeights, gneuralCompatibility, runGneural, toGneuralWeights } from '../gd/bridge';
import type { Sample as GdSample } from '../gd/protocol';
import { supabase } from '../lib/supabase';
import type { LayerNode } from './graph';
import { computeNetwork, type NetworkInfo } from './useNetwork';
import { defaultConfig, useStore, type NetworkConfig } from './useStore';

const get = useStore.getState;

function signature(net: NetworkInfo) {
  return JSON.stringify(net.chain.map((c, i) => [c.id, c.type, c.params, net.analysis[i].inShape, net.analysis[i].outShape]));
}

export function currentNetwork() {
  const s = get();
  return computeNetwork(s.nodes, s.edges, s.config);
}

/** Returns a model matching the current graph, rebuilding (and restoring saved weights) when the graph changed. */
export function ensureModel(net = currentNetwork()): Model | null {
  if (!net.valid) return null;
  const s = get();
  const sig = signature(net);
  if (s.model && s.modelSignature === sig) return s.model;
  try {
    const model = new Model(net.chain, net.analysis, s.config.train.seed);
    if (s.pendingWeights) {
      const n = model.load(s.pendingWeights);
      if (n) s.log('info', `Restored saved weights for ${n} layer${n === 1 ? '' : 's'}`, 'model');
      useStore.setState({ pendingWeights: null });
    } else if (s.model) {
      const n = model.load(s.model.serialize());
      if (n) s.log('info', `Kept trained weights for ${n} unchanged layer${n === 1 ? '' : 's'}`, 'model');
    }
    s.setModel(model, sig);
    return model;
  } catch (e) {
    s.log('error', `Could not build the model: ${(e as Error).message}`, 'model');
    return null;
  }
}

export function resetWeights() {
  useStore.setState({ model: null, modelSignature: '', pendingWeights: null });
  get().resetTraining();
  ensureModel();
  get().log('info', 'Weights re-initialized', 'model');
}

function reportProblems(net: NetworkInfo) {
  const { log } = get();
  if (net.datasetError) log('error', net.datasetError, 'dataset');
  for (const i of net.issues) log('error', i, 'graph');
  for (const e of net.errors) log('error', `${e.name}: ${e.message}`, 'shapes');
  if (net.chain.length < 2) log('error', 'Add at least one layer after the Input layer', 'graph');
}

export function startTraining() {
  const s = get();
  if (s.training.status === 'paused') return resumeTraining();
  if (s.training.status === 'running') return;
  const net = currentNetwork();
  if (!net.valid || !net.dataset) return reportProblems(net);
  const model = ensureModel(net);
  if (!model) return;
  const cfg = s.config.train;
  const ds = net.dataset;
  const { train, val } = splitDataset(ds, s.config.splits);
  if (!train.length) return s.log('error', 'The training split is empty. Increase the train percentage.', 'dataset');
  const startEpoch = s.training.history.length ? s.training.history[s.training.history.length - 1].epoch : 0;
  s.setTraining({ status: 'running', engine: cfg.engine, totalEpochs: startEpoch + cfg.epochs, batch: 0, batches: 0 });
  s.setView('training');

  if (cfg.engine === 'gneural') return trainWithGneural(net, model, train, val, startEpoch);

  s.log('info', `Training on ${train.length} samples (${val.length} validation) for ${cfg.epochs} epochs with ${cfg.optimizer.toUpperCase()}`, 'trainer');
  const run = new TrainingRun(
    model, train, val, cfg,
    { kind: ds.task, classes: ds.classes.length, augment: makeAugmenter(ds, s.config.augmentation) },
    {
      onBatch: (r) => get().pushBatch(r),
      onEpoch: (r) => {
        get().pushEpoch(r);
        const acc = r.valAcc !== null ? `, val acc ${(r.valAcc * 100).toFixed(1)}%` : '';
        get().log('info', `Epoch ${r.epoch}: loss ${r.trainLoss.toFixed(4)}, val loss ${r.valLoss.toFixed(4)}${acc}`, 'trainer');
      },
      onLog: (level, msg) => get().log(level, msg, 'trainer'),
    },
    startEpoch,
  );
  s.setRun({ pause: () => run.pause(), resume: () => run.resume(), stop: () => run.stop() });
  run.run().then((result) => {
    get().setTraining({ status: 'idle' });
    get().setRun(null);
    if (result === 'done') get().log('success', 'Training complete', 'trainer');
    if (result === 'stopped') get().log('warn', 'Training stopped', 'trainer');
  });
}

const toGd = (samples: Sample[]): GdSample[] =>
  samples.map((s) => ({ input: Array.from(s.x), output: Array.from(s.y) }));

function trainWithGneural(net: NetworkInfo, model: Model, train: Sample[], val: Sample[], startEpoch: number) {
  const s = get();
  const compat = gneuralCompatibility(net.chain, net.analysis);
  if (!compat.ok) {
    s.setTraining({ status: 'idle' });
    return s.log('error', `Gneural engine: ${compat.reason}`, 'gneural');
  }
  const cfg = s.config.train;
  const started = performance.now();
  s.log('info', `Training with the Gneural library (${compat.arch.inputs}-${compat.arch.hidden.join('-')}-${compat.arch.outputs}) in a background worker`, 'gneural');
  const job = runGneural(
    {
      type: 'train', arch: compat.arch, weights: toGneuralWeights(model), train: toGd(train), val: toGd(val),
      epochs: cfg.epochs, learningRate: cfg.learningRate, reportEvery: 1,
    },
    (m) => {
      const st = get();
      if (m.type === 'progress') {
        applyGneuralWeights(model, m.weights);
        st.pushBatch({ epoch: startEpoch + m.epoch, batch: 1, batches: 1, loss: m.trainLoss, lr: cfg.learningRate });
        st.pushEpoch({
          epoch: startEpoch + m.epoch, trainLoss: m.trainLoss, valLoss: m.valLoss, trainAcc: null, valAcc: null,
          metrics: {}, lr: cfg.learningRate, gradNorms: {}, seconds: (performance.now() - started) / 1000 / m.epoch,
        });
        if (m.epoch % Math.max(1, Math.round(cfg.epochs / 10)) === 0) st.log('info', `Epoch ${startEpoch + m.epoch}: error ${m.trainLoss.toFixed(5)}, val ${m.valLoss.toFixed(5)}`, 'gneural');
      } else if (m.type === 'trained') {
        applyGneuralWeights(model, m.weights);
        st.bumpWeights();
        st.log('success', 'Gneural training complete', 'gneural');
      } else if (m.type === 'error') {
        st.log('error', m.message, 'gneural');
      }
    },
  );
  s.setRun({ stop: () => { job.cancel(); get().log('warn', 'Gneural training stopped', 'gneural'); } });
  job.done.then(() => {
    get().setTraining({ status: 'idle' });
    get().setRun(null);
  });
}

export function pauseTraining() {
  const s = get();
  if (s.training.status !== 'running' || !s.run?.pause) return;
  s.run.pause();
  s.setTraining({ status: 'paused' });
  s.log('info', 'Training paused', 'trainer');
}

export function resumeTraining() {
  const s = get();
  if (s.training.status !== 'paused' || !s.run?.resume) return;
  s.run.resume();
  s.setTraining({ status: 'running' });
  s.log('info', 'Training resumed', 'trainer');
}

export function stopTraining() {
  const s = get();
  s.run?.stop();
  s.setRun(null);
  s.setTraining({ status: 'idle' });
}

export function toggleRun() {
  const st = get().training.status;
  if (st === 'running') pauseTraining();
  else startTraining();
}

export function flashFlow() {
  get().setFlowing(true);
  window.setTimeout(() => get().setFlowing(false), 1600);
}

export interface NetworkRow {
  id: string;
  name: string;
  description: string;
  layer_count: number;
  param_count: number;
  updated_at: string;
}

interface SavedGraph {
  nodes: LayerNode[];
  edges: Edge[];
}

function requireDb() {
  if (!supabase) throw new Error('Cloud storage is not configured');
  return supabase;
}

export async function saveNetwork(asNew = false) {
  const s = get();
  try {
    const db = requireDb();
    const net = currentNetwork();
    const model = s.model && s.modelSignature === signature(net) ? s.model : null;
    const row = {
      name: s.networkName.trim() || 'Untitled network',
      description: net.chain.map((c) => c.type).join(' > ').slice(0, 300),
      graph: { nodes: s.nodes.map(({ id, type, position, data }) => ({ id, type, position, data })), edges: s.edges } satisfies SavedGraph,
      config: s.config,
      weights: model ? model.serialize() : s.pendingWeights,
      layer_count: net.chain.length,
      param_count: net.totalParams,
      updated_at: new Date().toISOString(),
    };
    const query = s.networkId && !asNew
      ? db.from('networks').update(row).eq('id', s.networkId).select('id').maybeSingle()
      : db.from('networks').insert(row).select('id').maybeSingle();
    const { data, error } = await query;
    if (error) throw error;
    if (!data) throw new Error('The network could not be found. It may have been deleted.');
    useStore.setState({ networkId: data.id, dirty: false });
    s.log('success', `Saved "${row.name}"${row.weights ? ' with trained weights' : ''}`, 'storage');
    return true;
  } catch (e) {
    s.log('error', `Save failed: ${(e as Error).message}`, 'storage');
    return false;
  }
}

export async function listNetworks(): Promise<NetworkRow[]> {
  const { data, error } = await requireDb()
    .from('networks')
    .select('id, name, description, layer_count, param_count, updated_at')
    .order('updated_at', { ascending: false })
    .limit(100);
  if (error) throw error;
  return (data ?? []) as NetworkRow[];
}

export async function loadNetwork(id: string) {
  const s = get();
  try {
    const { data, error } = await requireDb().from('networks').select('*').eq('id', id).maybeSingle();
    if (error) throw error;
    if (!data) throw new Error('Network not found');
    const graph = data.graph as SavedGraph | null;
    if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) throw new Error('The saved graph is malformed');
    const base = defaultConfig();
    const cfg = (data.config ?? {}) as Partial<NetworkConfig>;
    s.replaceNetwork({
      id: data.id,
      name: data.name,
      nodes: graph.nodes.map((n) => (n.data?.layerType === 'Custom' ? { ...n, data: { ...n.data, params: { ...n.data.params, trusted: false } } } : n)),
      edges: graph.edges,
      config: { ...base, ...cfg, train: { ...base.train, ...cfg.train } },
      weights: (data.weights as SerializedWeights | null) ?? null,
    });
    s.setModal(null);
    s.log('success', `Loaded "${data.name}"`, 'storage');
  } catch (e) {
    s.log('error', `Load failed: ${(e as Error).message}`, 'storage');
  }
}

export async function deleteNetwork(id: string) {
  const { error } = await requireDb().from('networks').delete().eq('id', id);
  if (error) throw error;
  if (get().networkId === id) useStore.setState({ networkId: null, dirty: true });
}

export function exportJson() {
  const s = get();
  const net = currentNetwork();
  const model = s.model && s.modelSignature === signature(net) ? s.model : null;
  return JSON.stringify(
    {
      format: 'gneural-workbench/1',
      name: s.networkName,
      layers: net.chain.map((c, i) => ({ ...c, inShape: net.analysis[i].inShape, outShape: net.analysis[i].outShape, params_count: net.analysis[i].params })),
      totalParams: net.totalParams,
      training: s.config,
      weights: model?.serialize() ?? null,
    },
    null,
    2,
  );
}

export interface GdExport {
  levelString: string;
  objects: number;
  groups: number;
  outputs: number[];
}

export function exportGeometryDash(input: number[], spacing: number): Promise<GdExport> {
  const net = currentNetwork();
  const compat = gneuralCompatibility(net.chain, net.analysis);
  if (!compat.ok) return Promise.reject(new Error(compat.reason));
  const model = ensureModel(net);
  if (!model) return Promise.reject(new Error('The model could not be built'));
  return new Promise((resolve, reject) => {
    runGneural({ type: 'export', arch: compat.arch, weights: toGneuralWeights(model), input, spacing }, (m) => {
      if (m.type === 'exported') {
        get().log('success', `Generated Geometry Dash level: ${m.objects} objects, ${m.groups} groups`, 'gneural');
        resolve(m);
      } else if (m.type === 'error') reject(new Error(m.message));
    });
  });
}
