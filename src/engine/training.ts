import type { LossName, MetricName, OptimizerName, Param, ScheduleName, TrainConfig } from './types';

const EPS = 1e-7;

export interface LossResult {
  loss: number;
  grad: Float32Array;
}

/**
 * `outputAct` tells the loss whether outputs are already probabilities; if not,
 * cross-entropy losses treat them as logits and apply softmax / sigmoid internally.
 */
export function computeLoss(name: LossName, y: Float32Array, t: Float32Array, outputAct: string): LossResult {
  const n = y.length;
  const grad = new Float32Array(n);
  let loss = 0;
  switch (name) {
    case 'crossentropy': {
      if (outputAct === 'softmax') {
        for (let i = 0; i < n; i++) {
          loss -= t[i] * Math.log(y[i] + EPS);
          grad[i] = -t[i] / (y[i] + EPS);
        }
        return { loss, grad };
      }
      let max = -Infinity;
      for (let i = 0; i < n; i++) max = Math.max(max, y[i]);
      let sum = 0;
      for (let i = 0; i < n; i++) sum += Math.exp(y[i] - max);
      for (let i = 0; i < n; i++) {
        const p = Math.exp(y[i] - max) / sum;
        loss -= t[i] * Math.log(p + EPS);
        grad[i] = p - t[i];
      }
      return { loss, grad };
    }
    case 'bce': {
      const probs = outputAct === 'sigmoid';
      for (let i = 0; i < n; i++) {
        const p = probs ? Math.min(Math.max(y[i], EPS), 1 - EPS) : 1 / (1 + Math.exp(-y[i]));
        loss -= (t[i] * Math.log(p) + (1 - t[i]) * Math.log(1 - p)) / n;
        grad[i] = probs ? (p - t[i]) / (p * (1 - p)) / n : (p - t[i]) / n;
      }
      return { loss, grad };
    }
    case 'mae':
      for (let i = 0; i < n; i++) {
        const d = y[i] - t[i];
        loss += Math.abs(d) / n;
        grad[i] = Math.sign(d) / n;
      }
      return { loss, grad };
    case 'huber':
      for (let i = 0; i < n; i++) {
        const d = y[i] - t[i];
        const a = Math.abs(d);
        loss += (a <= 1 ? 0.5 * d * d : a - 0.5) / n;
        grad[i] = (a <= 1 ? d : Math.sign(d)) / n;
      }
      return { loss, grad };
    default:
      for (let i = 0; i < n; i++) {
        const d = y[i] - t[i];
        loss += (d * d) / n;
        grad[i] = (2 * d) / n;
      }
      return { loss, grad };
  }
}

export const LOSS_LABELS: Record<LossName, string> = {
  crossentropy: 'Categorical CrossEntropy',
  bce: 'Binary CrossEntropy',
  mse: 'Mean Squared Error',
  mae: 'Mean Absolute Error',
  huber: 'Huber',
};

export const OPTIMIZER_LABELS: Record<OptimizerName, string> = { adam: 'Adam', sgd: 'SGD', rmsprop: 'RMSProp', adagrad: 'AdaGrad' };

export const SCHEDULE_LABELS: Record<ScheduleName, string> = {
  constant: 'Constant',
  step: 'Step decay',
  exponential: 'Exponential decay',
  cosine: 'Cosine annealing',
  warmup_cosine: 'Warmup + cosine',
};

/** Learning rate at a fractional epoch position `progress` in [0, 1]. */
export function scheduledRate(schedule: ScheduleName, base: number, progress: number) {
  switch (schedule) {
    case 'step':
      return base * Math.pow(0.5, Math.floor(progress * 4));
    case 'exponential':
      return base * Math.pow(0.05, progress);
    case 'cosine':
      return base * (0.02 + 0.98 * 0.5 * (1 + Math.cos(Math.PI * progress)));
    case 'warmup_cosine': {
      const w = 0.1;
      if (progress < w) return base * (progress / w + 0.01);
      return base * (0.02 + 0.98 * 0.5 * (1 + Math.cos((Math.PI * (progress - w)) / (1 - w))));
    }
    default:
      return base;
  }
}

interface Slot {
  m: Float32Array;
  v: Float32Array;
}

export class Optimizer {
  private slots = new Map<Param, Slot>();
  private t = 0;

  constructor(private cfg: TrainConfig) {}

  /** Applies averaged gradients (already divided by batch size) and clears them. */
  step(params: Param[], lr: number) {
    const { optimizer, momentum, beta1, beta2, epsilon, clipNorm } = this.cfg;
    this.t++;
    let scale = 1;
    if (clipNorm > 0) {
      let sq = 0;
      for (const p of params) if (!p.frozen) for (let i = 0; i < p.grad.length; i++) sq += p.grad[i] ** 2;
      const norm = Math.sqrt(sq);
      if (norm > clipNorm) scale = clipNorm / norm;
    }
    for (const p of params) {
      if (p.frozen) continue;
      let s = this.slots.get(p);
      if (!s) this.slots.set(p, (s = { m: new Float32Array(p.value.length), v: new Float32Array(p.value.length) }));
      const { value: w, grad } = p;
      for (let i = 0; i < w.length; i++) {
        let g = grad[i] * scale + p.l2 * 2 * w[i] + p.l1 * Math.sign(w[i]);
        if (!Number.isFinite(g)) g = 0;
        switch (optimizer) {
          case 'sgd':
            s.m[i] = momentum * s.m[i] + g;
            w[i] -= lr * s.m[i];
            break;
          case 'rmsprop':
            s.v[i] = 0.9 * s.v[i] + 0.1 * g * g;
            w[i] -= (lr * g) / (Math.sqrt(s.v[i]) + epsilon);
            break;
          case 'adagrad':
            s.v[i] += g * g;
            w[i] -= (lr * g) / (Math.sqrt(s.v[i]) + epsilon);
            break;
          default: {
            s.m[i] = beta1 * s.m[i] + (1 - beta1) * g;
            s.v[i] = beta2 * s.v[i] + (1 - beta2) * g * g;
            const mh = s.m[i] / (1 - Math.pow(beta1, this.t));
            const vh = s.v[i] / (1 - Math.pow(beta2, this.t));
            w[i] -= (lr * mh) / (Math.sqrt(vh) + epsilon);
          }
        }
        grad[i] = 0;
      }
    }
  }
}

export const argmax = (a: ArrayLike<number>) => {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
};

/** Macro-averaged classification metrics from predicted / true label pairs. */
export function classificationMetrics(pred: number[], truth: number[], classes: number): Record<MetricName, number> {
  const tp = new Array(classes).fill(0), fp = new Array(classes).fill(0), fn = new Array(classes).fill(0);
  let correct = 0;
  pred.forEach((p, i) => {
    const t = truth[i];
    if (p === t) (correct++, tp[t]++);
    else (fp[p]++, fn[t]++);
  });
  let prec = 0, rec = 0, f1 = 0, seen = 0;
  for (let c = 0; c < classes; c++) {
    if (tp[c] + fn[c] === 0 && fp[c] === 0) continue;
    seen++;
    const pc = tp[c] + fp[c] ? tp[c] / (tp[c] + fp[c]) : 0;
    const rc = tp[c] + fn[c] ? tp[c] / (tp[c] + fn[c]) : 0;
    prec += pc;
    rec += rc;
    f1 += pc + rc ? (2 * pc * rc) / (pc + rc) : 0;
  }
  seen = Math.max(seen, 1);
  return { accuracy: pred.length ? correct / pred.length : 0, precision: prec / seen, recall: rec / seen, f1: f1 / seen };
}
