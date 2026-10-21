import { mulberry32, randn, type Rng } from '../engine/math';
import type { Sample, Shape, TaskKind } from '../engine/types';

export type DataKind = 'image' | 'sequence' | 'text' | 'tabular';

export interface Dataset {
  id: string;
  name: string;
  description: string;
  kind: DataKind;
  task: TaskKind;
  inputShape: Shape;
  targetShape: Shape;
  classes: string[];
  featureNames: string[];
  samples: Sample[];
  normalize?: { mean: number[]; std: number[] };
}

export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  kind: DataKind;
}

const oneHot = (i: number, n: number) => {
  const v = new Float32Array(n);
  v[i] = 1;
  return v;
};

const GLYPHS = [
  ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  ['01110', '10001', '00001', '00010', '00100', '01000', '11111'],
  ['11111', '00010', '00100', '00010', '00001', '10001', '01110'],
  ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
  ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
  ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
  ['01110', '10001', '10001', '01111', '00001', '00010', '01100'],
];

export function renderDigit(d: number, rng: Rng, clean = false) {
  const img = new Float32Array(64);
  const ox = clean ? 1 : Math.floor(rng() * 4);
  const oy = clean ? 0 : Math.floor(rng() * 2);
  const ink = clean ? 1 : 0.7 + rng() * 0.3;
  GLYPHS[d].forEach((row, y) =>
    [...row].forEach((bit, x) => {
      if (bit === '1' && (clean || rng() > 0.06)) img[(y + oy) * 8 + x + ox] = ink;
    }),
  );
  if (!clean) for (let i = 0; i < 64; i++) img[i] = Math.min(1, Math.max(0, img[i] + randn(rng) * 0.06));
  return img;
}

function digits(rng: Rng): Dataset {
  const samples: Sample[] = [];
  for (let i = 0; i < 800; i++) {
    const d = i % 10;
    samples.push({ x: renderDigit(d, rng), y: oneHot(d, 10), label: d });
  }
  return {
    id: 'digits', name: 'Digits 8x8', description: 'Handwritten-style digits 0-9 rendered at 8x8 pixels with jitter and noise.',
    kind: 'image', task: 'classification', inputShape: [8, 8, 1], targetShape: [10],
    classes: GLYPHS.map((_, i) => String(i)), featureNames: [], samples,
  };
}

function digitsReconstruction(rng: Rng): Dataset {
  const base = digits(rng);
  return {
    ...base, id: 'digits-ae', name: 'Digits (reconstruct)', description: 'The digits dataset where the target is the image itself, for autoencoders.',
    task: 'reconstruction', targetShape: [8, 8, 1], classes: [],
    samples: base.samples.map((s) => ({ ...s, y: s.x.slice(), label: -1 })),
  };
}

function spiral(rng: Rng): Dataset {
  const samples: Sample[] = [];
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < 120; i++) {
      const r = i / 120;
      const t = c * 2.1 + r * 4 + randn(rng) * 0.2;
      samples.push({ x: new Float32Array([r * Math.sin(t), r * Math.cos(t)]), y: oneHot(c, 3), label: c });
    }
  }
  return {
    id: 'spiral', name: 'Spirals', description: 'Three interleaved spiral arms in 2D. A classic test of non-linear decision boundaries.',
    kind: 'tabular', task: 'classification', inputShape: [2], targetShape: [3], classes: ['Arm A', 'Arm B', 'Arm C'], featureNames: ['x', 'y'], samples,
  };
}

function xor(rng: Rng): Dataset {
  const samples: Sample[] = [];
  for (let i = 0; i < 240; i++) {
    const a = rng() > 0.5 ? 1 : 0, b = rng() > 0.5 ? 1 : 0;
    const label = a ^ b;
    samples.push({ x: new Float32Array([a + randn(rng) * 0.12, b + randn(rng) * 0.12]), y: oneHot(label, 2), label });
  }
  return {
    id: 'xor', name: 'XOR', description: 'Outputs 1 when exactly one input is on. Impossible for a single straight line to separate.',
    kind: 'tabular', task: 'classification', inputShape: [2], targetShape: [2], classes: ['Off', 'On'], featureNames: ['a', 'b'], samples,
  };
}

function sequenceNext(): Dataset {
  const samples: Sample[] = [];
  for (let a = 0; a <= 24; a++) samples.push({ x: new Float32Array([a, a + 1]), y: new Float32Array([a + 2]), label: -1 });
  return {
    id: 'next-number', name: 'Next number', description: 'Predict the next number in a sequence: [1, 2] -> 3. The Gneural README example, ready for Geometry Dash.',
    kind: 'tabular', task: 'regression', inputShape: [2], targetShape: [1], classes: [], featureNames: ['n', 'n + 1'], samples,
  };
}

export const WAVE_CLASSES = ['Sine', 'Square', 'Sawtooth', 'Triangle'];

export function waveform(kind: number, freq: number, phase: number, amp: number, steps = 16) {
  const out = new Float32Array(steps);
  for (let t = 0; t < steps; t++) {
    const p = ((t / steps) * freq + phase) % 1;
    out[t] = amp * [Math.sin(2 * Math.PI * p), p < 0.5 ? 1 : -1, 2 * p - 1, 1 - 4 * Math.abs(p - 0.5)][kind];
  }
  return out;
}

function waves(rng: Rng): Dataset {
  const samples: Sample[] = [];
  for (let i = 0; i < 480; i++) {
    const k = i % 4;
    const x = waveform(k, 1 + rng() * 1.5, rng(), 0.6 + rng() * 0.4);
    for (let t = 0; t < x.length; t++) x[t] += randn(rng) * 0.05;
    samples.push({ x, y: oneHot(k, 4), label: k });
  }
  return {
    id: 'waves', name: 'Waveforms', description: 'Short 16-step signals of four shapes with random frequency, phase and noise.',
    kind: 'sequence', task: 'classification', inputShape: [16, 1], targetShape: [4], classes: WAVE_CLASSES, featureNames: [], samples,
  };
}

const POSITIVE = ['great', 'love', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy', 'enjoyed', 'brilliant', 'perfect', 'fun', 'beautiful'];
const NEGATIVE = ['terrible', 'hate', 'awful', 'boring', 'bad', 'worst', 'sad', 'broken', 'poor', 'disappointing', 'ugly', 'annoying'];
const FILLER = ['the', 'movie', 'was', 'really', 'this', 'game', 'level', 'it', 'so', 'very', 'i', 'a', 'show', 'music', 'felt', 'quite'];
export const TEXT_STEPS = 8;
export const TEXT_DIM = 16;

function hashWord(w: string) {
  let h = 2166136261;
  for (let i = 0; i < w.length; i++) h = Math.imul(h ^ w.charCodeAt(i), 16777619);
  return h >>> 0;
}

export const tokenize = (text: string) => text.toLowerCase().match(/[a-z']+/g) ?? [];

/** Deterministic hashed word embeddings plus sinusoidal positions: text -> [8, 16]. */
export function encodeText(text: string) {
  const out = new Float32Array(TEXT_STEPS * TEXT_DIM);
  tokenize(text).slice(0, TEXT_STEPS).forEach((w, t) => {
    const r = mulberry32(hashWord(w));
    for (let d = 0; d < TEXT_DIM; d++) {
      const pos = d % 2 === 0 ? Math.sin(t / Math.pow(100, d / TEXT_DIM)) : Math.cos(t / Math.pow(100, (d - 1) / TEXT_DIM));
      out[t * TEXT_DIM + d] = randn(r) * 0.8 + pos * 0.2;
    }
  });
  return out;
}

function sentiment(rng: Rng): Dataset {
  const pick = <T,>(a: T[]) => a[Math.floor(rng() * a.length)];
  const samples: Sample[] = [];
  for (let i = 0; i < 500; i++) {
    const label = i % 2;
    const len = 4 + Math.floor(rng() * 5);
    const words = Array.from({ length: len }, () => pick(FILLER));
    const n = 1 + Math.floor(rng() * 2);
    for (let k = 0; k < n; k++) words[Math.floor(rng() * len)] = pick(label ? POSITIVE : NEGATIVE);
    const text = words.join(' ');
    samples.push({ x: encodeText(text), y: oneHot(label, 2), label, text });
  }
  return {
    id: 'sentiment', name: 'Sentiment phrases', description: 'Short review-style phrases labeled positive or negative, encoded as 8 tokens x 16 features.',
    kind: 'text', task: 'classification', inputShape: [TEXT_STEPS, TEXT_DIM], targetShape: [2], classes: ['Negative', 'Positive'], featureNames: [], samples,
  };
}

/** Parses a CSV where the last column is the target. Numeric feature columns are standardized. */
export function parseCsv(text: string, name: string): Dataset {
  const rows = text.split(/\r?\n/).map((l) => l.split(',').map((c) => c.trim())).filter((r) => r.length > 1 && r.some(Boolean));
  if (rows.length < 4) throw new Error('The CSV needs at least a few rows of data');
  const hasHeader = rows[0].slice(0, -1).some((c) => c !== '' && Number.isNaN(Number(c)));
  const header = hasHeader ? rows.shift()! : rows[0].map((_, i) => `col${i + 1}`);
  const width = header.length;
  const data = rows.filter((r) => r.length === width);
  if (data.length < 4) throw new Error('Rows have inconsistent column counts');
  const feat = data.map((r) => r.slice(0, -1).map(Number));
  if (feat.some((r) => r.some((v) => !Number.isFinite(v)))) throw new Error('All columns except the last must be numeric');
  const targets = data.map((r) => r[width - 1]);
  const uniq = [...new Set(targets)];
  const numericTarget = targets.every((t) => t !== '' && Number.isFinite(Number(t)));
  const classify = !numericTarget || (uniq.length <= 20 && targets.every((t) => Number.isInteger(Number(t))));
  const nf = width - 1;
  const mean = Array.from({ length: nf }, (_, j) => feat.reduce((a, r) => a + r[j], 0) / feat.length);
  const std = Array.from({ length: nf }, (_, j) => Math.sqrt(feat.reduce((a, r) => a + (r[j] - mean[j]) ** 2, 0) / feat.length) || 1);
  const classes = classify ? uniq.sort() : [];
  const samples = feat.map((r, i): Sample => {
    const x = new Float32Array(r.map((v, j) => (v - mean[j]) / std[j]));
    if (classify) {
      const label = classes.indexOf(targets[i]);
      return { x, y: oneHot(label, classes.length), label };
    }
    return { x, y: new Float32Array([Number(targets[i])]), label: -1 };
  });
  return {
    id: 'csv', name, description: `Uploaded CSV with ${samples.length} rows and ${nf} features.`,
    kind: 'tabular', task: classify ? 'classification' : 'regression', inputShape: [nf], targetShape: [classify ? classes.length : 1],
    classes, featureNames: header.slice(0, -1), samples, normalize: { mean, std },
  };
}

const BUILDERS: Record<string, (rng: Rng) => Dataset> = {
  digits,
  'digits-ae': digitsReconstruction,
  spiral,
  xor,
  'next-number': sequenceNext,
  waves,
  sentiment,
};

export const BUILTIN_DATASETS: DatasetInfo[] = Object.values(BUILDERS).map((b) => {
  const { id, name, description, kind } = b(mulberry32(1));
  return { id, name, description, kind };
});

const cache = new Map<string, Dataset>();

export function loadDataset(id: string, csv?: { text: string; name: string } | null): Dataset {
  if (id === 'csv') {
    if (!csv) throw new Error('No CSV has been uploaded');
    return parseCsv(csv.text, csv.name);
  }
  const hit = cache.get(id);
  if (hit) return hit;
  const build = BUILDERS[id];
  if (!build) throw new Error(`Unknown dataset "${id}"`);
  const ds = build(mulberry32(42));
  cache.set(id, ds);
  return ds;
}

export interface Splits {
  train: number;
  val: number;
}

export function splitDataset(ds: Dataset, splits: Splits, seed = 7) {
  const rng = mulberry32(seed);
  const idx = ds.samples.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTrain = Math.round(idx.length * splits.train);
  const nVal = Math.round(idx.length * splits.val);
  const pick = (a: number[]) => a.map((i) => ds.samples[i]);
  return {
    train: pick(idx.slice(0, nTrain)),
    val: pick(idx.slice(nTrain, nTrain + nVal)),
    test: pick(idx.slice(nTrain + nVal)),
  };
}

export interface Augmentation {
  noise: boolean;
  shift: boolean;
  flip: boolean;
  scale: boolean;
}

export function makeAugmenter(ds: Dataset, a: Augmentation) {
  const active = a.noise || a.scale || (ds.kind === 'image' && (a.shift || a.flip));
  if (!active) return (x: Float32Array) => x;
  const rng = mulberry32(99);
  const [h, w, c] = ds.inputShape.length === 3 ? ds.inputShape : [1, ds.inputShape[0], 1];
  return (x: Float32Array) => {
    let y = x.slice();
    if (ds.kind === 'image' && (a.shift || a.flip)) {
      const dx = a.shift ? Math.round((rng() - 0.5) * 2) : 0;
      const dy = a.shift ? Math.round((rng() - 0.5) * 2) : 0;
      const flip = a.flip && rng() < 0.5;
      const out = new Float32Array(y.length);
      for (let r = 0; r < h; r++) {
        for (let col = 0; col < w; col++) {
          const sr = r - dy, sc0 = col - dx, sc = flip ? w - 1 - sc0 : sc0;
          if (sr < 0 || sr >= h || sc < 0 || sc >= w) continue;
          for (let k = 0; k < c; k++) out[(r * w + col) * c + k] = y[(sr * w + sc) * c + k];
        }
      }
      y = out;
    }
    const s = a.scale ? 0.85 + rng() * 0.3 : 1;
    for (let i = 0; i < y.length; i++) y[i] = y[i] * s + (a.noise ? randn(rng) * 0.05 : 0);
    return y;
  };
}
