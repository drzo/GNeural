import { applyEdgeChanges, applyNodeChanges, type Connection, type Edge, type EdgeChange, type NodeChange } from '@xyflow/react';
import { create } from 'zustand';
import { defaultParams, layerDef } from '../engine/catalog';
import type { Model, SerializedWeights } from '../engine/model';
import type { BatchReport, EpochReport } from '../engine/trainer';
import type { LayerParams, LayerType, TrainConfig } from '../engine/types';
import type { Augmentation, Splits } from '../data/datasets';
import { TEMPLATES, templateParams, type TemplateDef } from '../data/templates';
import { autoLayout, extractChain, makeEdge, makeNode, type LayerNode } from './graph';

export type ViewName = 'graph' | 'architecture' | 'training' | 'inference' | 'debug' | 'dataset';
export type LogLevel = 'info' | 'warn' | 'error' | 'success';
export type Modal = 'templates' | 'load' | 'export' | null;

export interface LogEntry {
  id: number;
  ts: number;
  level: LogLevel;
  source: string;
  msg: string;
}

export interface NetworkConfig {
  train: TrainConfig;
  datasetId: string;
  csv: { text: string; name: string } | null;
  splits: Splits;
  augmentation: Augmentation;
}

export interface TrainingState {
  status: 'idle' | 'running' | 'paused';
  engine: TrainConfig['engine'];
  epoch: number;
  totalEpochs: number;
  batch: number;
  batches: number;
  history: EpochReport[];
  batchLoss: { step: number; loss: number }[];
}

interface Snapshot {
  nodes: LayerNode[];
  edges: Edge[];
}

export const DEFAULT_TRAIN: TrainConfig = {
  engine: 'workbench',
  optimizer: 'adam',
  loss: 'crossentropy',
  metrics: ['accuracy'],
  epochs: 15,
  batchSize: 32,
  learningRate: 0.005,
  schedule: 'constant',
  momentum: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-7,
  clipNorm: 0,
  seed: 1,
};

export const defaultConfig = (): NetworkConfig => ({
  train: { ...DEFAULT_TRAIN },
  datasetId: 'digits',
  csv: null,
  splits: { train: 0.7, val: 0.15 },
  augmentation: { noise: false, shift: false, flip: false, scale: false },
});

const emptyTraining = (): TrainingState => ({
  status: 'idle', engine: 'workbench', epoch: 0, totalEpochs: 0, batch: 0, batches: 0, history: [], batchLoss: [],
});

interface RunHandle {
  pause?: () => void;
  resume?: () => void;
  stop: () => void;
}

export interface AppState {
  networkId: string | null;
  networkName: string;
  dirty: boolean;
  nodes: LayerNode[];
  edges: Edge[];
  selectedId: string | null;
  past: Snapshot[];
  future: Snapshot[];
  view: ViewName;
  mode: 'beginner' | 'expert';
  theme: 'dark' | 'light';
  modal: Modal;
  walkthrough: number | null;
  config: NetworkConfig;
  logs: LogEntry[];
  training: TrainingState;
  run: RunHandle | null;
  model: Model | null;
  modelSignature: string;
  pendingWeights: SerializedWeights | null;
  weightsVersion: number;
  flowing: boolean;

  log(level: LogLevel, msg: string, source?: string): void;
  clearLogs(): void;
  setView(v: ViewName): void;
  setMode(m: AppState['mode']): void;
  toggleTheme(): void;
  setModal(m: Modal): void;
  setWalkthrough(step: number | null): void;
  setName(n: string): void;
  select(id: string | null): void;

  onNodesChange(changes: NodeChange<LayerNode>[]): void;
  onEdgesChange(changes: EdgeChange[]): void;
  onConnect(c: Connection): void;
  addLayer(type: LayerType, position?: { x: number; y: number }, params?: LayerParams, name?: string): void;
  appendLayer(type: LayerType): void;
  updateLayer(id: string, patch: { name?: string; params?: LayerParams }): void;
  removeNode(id: string): void;
  removeSelected(): void;
  undo(): void;
  redo(): void;
  checkpoint(): void;
  arrange(): void;
  setConfig(patch: Partial<NetworkConfig>): void;
  setTrain(patch: Partial<TrainConfig>): void;
  loadTemplate(t: TemplateDef): void;
  newNetwork(): void;
  replaceNetwork(n: { id: string | null; name: string; nodes: LayerNode[]; edges: Edge[]; config: NetworkConfig; weights: SerializedWeights | null }): void;
  setTraining(patch: Partial<TrainingState>): void;
  pushBatch(r: BatchReport): void;
  pushEpoch(r: EpochReport): void;
  resetTraining(): void;
  setRun(r: RunHandle | null): void;
  setModel(m: Model | null, signature: string): void;
  bumpWeights(): void;
  setFlowing(f: boolean): void;
}

const HISTORY_LIMIT = 100;
let logSeq = 0;

function templateGraph(t: TemplateDef) {
  const pos = autoLayout(t.layers.length);
  const nodes = t.layers.map(([type, params, name], i) => makeNode(type, pos[i], templateParams(type, params), name));
  const edges = nodes.slice(1).map((n, i) => makeEdge(nodes[i].id, n.id));
  return { nodes, edges };
}

const initial = templateGraph(TEMPLATES[0]);

export const useStore = create<AppState>()((set, get) => {
  const snapshot = (): Snapshot => ({ nodes: get().nodes, edges: get().edges });
  const commit = (next: Partial<AppState>) =>
    set((s) => ({ ...next, past: [...s.past, { nodes: s.nodes, edges: s.edges }].slice(-HISTORY_LIMIT), future: [], dirty: true }));
  let lastEdit = { id: '', at: 0 };

  return {
    networkId: null,
    networkName: 'Digit classifier',
    dirty: false,
    nodes: initial.nodes,
    edges: initial.edges,
    selectedId: null,
    past: [],
    future: [],
    view: 'graph',
    mode: 'beginner',
    theme: 'dark',
    modal: null,
    walkthrough: null,
    config: { ...defaultConfig(), train: { ...DEFAULT_TRAIN, ...TEMPLATES[0].train } },
    logs: [],
    training: emptyTraining(),
    run: null,
    model: null,
    modelSignature: '',
    pendingWeights: null,
    weightsVersion: 0,
    flowing: false,

    log: (level, msg, source = 'app') =>
      set((s) => ({ logs: [...s.logs, { id: ++logSeq, ts: Date.now(), level, source, msg }].slice(-600) })),
    clearLogs: () => set({ logs: [] }),
    setView: (view) => set({ view }),
    setMode: (mode) => set({ mode }),
    toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
    setModal: (modal) => set({ modal }),
    setWalkthrough: (walkthrough) => set({ walkthrough }),
    setName: (networkName) => set({ networkName, dirty: true }),
    select: (selectedId) => set({ selectedId }),

    onNodesChange: (all) => {
      const changes = all.filter((c) => c.type !== 'select');
      if (!changes.length) return;
      const meaningful = changes.filter((c) => c.type !== 'dimensions');
      const dragEnd = changes.some((c) => c.type === 'position' && c.dragging === false);
      const removal = changes.some((c) => c.type === 'remove');
      const nodes = applyNodeChanges(changes, get().nodes);
      if (removal) return commit({ nodes });
      if (dragEnd) return set({ nodes, dirty: true });
      set({ nodes, dirty: get().dirty || meaningful.some((c) => c.type === 'position') });
    },
    onEdgesChange: (changes) => {
      const edges = applyEdgeChanges(changes, get().edges);
      if (changes.some((c) => c.type === 'remove')) commit({ edges });
      else set({ edges });
    },
    onConnect: (c) => {
      if (!c.source || !c.target || c.source === c.target) return;
      const edges = get().edges.filter((e) => e.target !== c.target);
      if (edges.some((e) => e.source === c.source && e.target === c.target)) return;
      commit({ edges: [...edges, makeEdge(c.source, c.target)] });
      get().log('info', 'Connected layers', 'graph');
    },
    checkpoint: () => set((s) => ({ past: [...s.past, snapshot()].slice(-HISTORY_LIMIT), future: [] })),
    arrange: () => {
      const { nodes, edges } = get();
      const { chain, orphans } = extractChain(nodes, edges);
      const order = [...chain.map((c) => c.id), ...orphans];
      const pos = autoLayout(chain.length);
      const placed = new Map(order.map((id, i) => [id, i < chain.length ? pos[i] : { x: (i - chain.length) * 250, y: (Math.ceil(chain.length / 5) + 0.5) * 190 }]));
      commit({ nodes: nodes.map((n) => ({ ...n, position: placed.get(n.id) ?? n.position })) });
    },
    addLayer: (type, position, params, name) => {
      const nodes = get().nodes;
      const pos = position ?? { x: 80 + nodes.length * 30, y: 260 + nodes.length * 20 };
      const node = makeNode(type, pos, params ? { ...params } : defaultParams(type), name);
      commit({ nodes: [...nodes.map((n) => ({ ...n, selected: false })), { ...node, selected: true }], selectedId: node.id });
      get().log('info', `Added ${layerDef(type).label} layer`, 'graph');
    },
    appendLayer: (type) => {
      const { nodes, edges } = get();
      const { chain } = extractChain(nodes, edges);
      const tail = chain.length ? nodes.find((n) => n.id === chain[chain.length - 1].id) : undefined;
      const pos = tail ? { x: tail.position.x + 270, y: tail.position.y } : { x: 0, y: 0 };
      const node = makeNode(type, pos, defaultParams(type));
      commit({
        nodes: [...nodes.map((n) => ({ ...n, selected: false })), { ...node, selected: true }],
        edges: tail ? [...edges, makeEdge(tail.id, node.id)] : edges,
        selectedId: node.id,
      });
      get().log('info', `Appended ${layerDef(type).label}${tail ? ` after "${tail.data.name}"` : ''}`, 'graph');
    },
    updateLayer: (id, patch) => {
      const nodes = get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, name: patch.name ?? n.data.name, params: patch.params ? { ...n.data.params, ...patch.params } : n.data.params } } : n,
      );
      const now = Date.now();
      const coalesce = lastEdit.id === id && now - lastEdit.at < 800;
      lastEdit = { id, at: now };
      if (coalesce) set({ nodes, dirty: true });
      else commit({ nodes });
    },
    removeNode: (id) => {
      const { nodes, edges } = get();
      const node = nodes.find((n) => n.id === id);
      if (!node) return;
      const inc = edges.find((e) => e.target === id);
      const outs = edges.filter((e) => e.source === id);
      const kept = edges.filter((e) => e.source !== id && e.target !== id);
      const bridged = inc ? outs.map((o) => makeEdge(inc.source, o.target)) : [];
      commit({ nodes: nodes.filter((n) => n.id !== id), edges: [...kept, ...bridged], selectedId: get().selectedId === id ? null : get().selectedId });
      get().log('info', `Removed "${node.data.name}"`, 'graph');
    },
    removeSelected: () => {
      const id = get().selectedId ?? get().nodes.find((n) => n.selected)?.id;
      if (id) get().removeNode(id);
    },
    undo: () => {
      const { past } = get();
      if (!past.length) return;
      const prev = past[past.length - 1];
      set((s) => ({ ...prev, past: s.past.slice(0, -1), future: [snapshot(), ...s.future], dirty: true }));
      lastEdit = { id: '', at: 0 };
    },
    redo: () => {
      const { future } = get();
      if (!future.length) return;
      const next = future[0];
      set((s) => ({ ...next, future: s.future.slice(1), past: [...s.past, snapshot()], dirty: true }));
    },

    setConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch }, dirty: true })),
    setTrain: (patch) => set((s) => ({ config: { ...s.config, train: { ...s.config.train, ...patch } }, dirty: true })),
    loadTemplate: (t) => {
      get().run?.stop();
      const { nodes, edges } = templateGraph(t);
      commit({
        nodes, edges, selectedId: null, networkId: null, networkName: t.name, modal: null,
        config: { ...defaultConfig(), datasetId: t.datasetId, train: { ...DEFAULT_TRAIN, ...t.train } },
        model: null, modelSignature: '', pendingWeights: null, training: emptyTraining(), run: null,
      });
      get().log('success', `Loaded template "${t.name}" with ${nodes.length} layers`, 'templates');
    },
    newNetwork: () => {
      get().run?.stop();
      const input = makeNode('Input', { x: 0, y: 0 }, defaultParams('Input'));
      commit({
        nodes: [input], edges: [], selectedId: input.id, networkId: null, networkName: 'Untitled network',
        config: { ...defaultConfig(), datasetId: 'xor' }, model: null, modelSignature: '', pendingWeights: null,
        training: emptyTraining(), run: null,
      });
      get().log('info', 'Started a new network', 'app');
    },
    replaceNetwork: (n) => {
      get().run?.stop();
      set({
        networkId: n.id, networkName: n.name, nodes: n.nodes, edges: n.edges, config: n.config, selectedId: null,
        past: [], future: [], dirty: false, model: null, modelSignature: '', pendingWeights: n.weights,
        training: emptyTraining(), run: null,
      });
    },
    setTraining: (patch) => set((s) => ({ training: { ...s.training, ...patch } })),
    pushBatch: (r) =>
      set((s) => {
        const step = (r.epoch - 1) * r.batches + r.batch;
        const every = Math.max(1, Math.floor((s.training.totalEpochs * r.batches) / 400));
        const batchLoss = step % every === 0 ? [...s.training.batchLoss, { step, loss: r.loss }] : s.training.batchLoss;
        return { training: { ...s.training, epoch: r.epoch, batch: r.batch, batches: r.batches, batchLoss } };
      }),
    pushEpoch: (r) => set((s) => ({ training: { ...s.training, epoch: r.epoch, history: [...s.training.history, r] }, weightsVersion: s.weightsVersion + 1 })),
    resetTraining: () => set({ training: emptyTraining() }),
    setRun: (run) => set({ run }),
    setModel: (model, modelSignature) => set({ model, modelSignature, weightsVersion: get().weightsVersion + 1 }),
    bumpWeights: () => set((s) => ({ weightsVersion: s.weightsVersion + 1 })),
    setFlowing: (flowing) => set({ flowing }),
  };
});
