import type { Model } from './model';
import { argmax, classificationMetrics, computeLoss, Optimizer, scheduledRate } from './training';
import type { MetricName, Sample, TaskKind, TrainConfig } from './types';

export interface EpochReport {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number | null;
  valAcc: number | null;
  metrics: Partial<Record<MetricName, number>>;
  lr: number;
  gradNorms: Record<string, number>;
  seconds: number;
}

export interface BatchReport {
  epoch: number;
  batch: number;
  batches: number;
  loss: number;
  lr: number;
}

export interface TrainCallbacks {
  onBatch(r: BatchReport): void;
  onEpoch(r: EpochReport): void;
  onLog(level: 'info' | 'warn' | 'error' | 'success', msg: string): void;
}

export interface TrainTask {
  kind: TaskKind;
  classes: number;
  augment: (x: Float32Array) => Float32Array;
}

export interface Evaluation {
  loss: number;
  metrics: Record<MetricName, number> | null;
}

export function evaluate(model: Model, data: Sample[], cfg: TrainConfig, task: TrainTask): Evaluation {
  if (!data.length) return { loss: NaN, metrics: null };
  const act = model.outputActivation;
  let loss = 0;
  const pred: number[] = [], truth: number[] = [];
  for (const s of data) {
    const y = model.forward(s.x, false);
    loss += computeLoss(cfg.loss, y, s.y, act).loss;
    if (task.kind === 'classification') (pred.push(argmax(y)), truth.push(s.label));
  }
  return {
    loss: loss / data.length,
    metrics: task.kind === 'classification' ? classificationMetrics(pred, truth, task.classes) : null,
  };
}

const VANISH = 1e-7;
const EXPLODE = 1e3;

/** A cooperative, pausable training loop that yields to the UI between batches. */
export class TrainingRun {
  private paused = false;
  private stopped = false;
  private wake: (() => void) | null = null;

  constructor(
    private model: Model,
    private train: Sample[],
    private val: Sample[],
    private cfg: TrainConfig,
    private task: TrainTask,
    private cb: TrainCallbacks,
    private startEpoch = 0,
  ) {}

  pause() {
    this.paused = true;
  }

  resume() {
    this.paused = false;
    this.wake?.();
  }

  stop() {
    this.stopped = true;
    this.wake?.();
  }

  private async yieldToUi() {
    await new Promise((r) => setTimeout(r, 0));
    while (this.paused && !this.stopped) await new Promise<void>((r) => (this.wake = r));
  }

  /** Resolves with 'done' or 'stopped'. */
  async run(): Promise<'done' | 'stopped' | 'failed'> {
    const { model, cfg, task, cb } = this;
    const opt = new Optimizer(cfg);
    const bs = Math.max(1, Math.min(cfg.batchSize, this.train.length));
    const batches = Math.ceil(this.train.length / bs);
    const totalSteps = cfg.epochs * batches;
    const act = model.outputActivation;
    model.zeroGrad();
    let slice = performance.now();
    for (let e = 0; e < cfg.epochs; e++) {
      const started = performance.now();
      const order = this.train.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      let lossSum = 0, correct = 0;
      const gradSq: Record<string, number> = {};
      let lr = cfg.learningRate;
      for (let b = 0; b < batches; b++) {
        lr = scheduledRate(cfg.schedule, cfg.learningRate, (e * batches + b) / totalSteps);
        const idx = order.slice(b * bs, (b + 1) * bs);
        let batchLoss = 0;
        for (const k of idx) {
          const s = this.train[k];
          const x = task.augment(s.x);
          const y = model.forward(x, true);
          const target = task.kind === 'reconstruction' ? x : s.y;
          const { loss, grad } = computeLoss(cfg.loss, y, target, act);
          batchLoss += loss;
          if (task.kind === 'classification' && argmax(y) === s.label) correct++;
          model.backward(grad);
        }
        if (!Number.isFinite(batchLoss)) {
          cb.onLog('error', `Loss became ${batchLoss} at epoch ${this.startEpoch + e + 1}. Training halted: try a lower learning rate or gradient clipping.`);
          return 'failed';
        }
        const inv = 1 / idx.length;
        for (const l of model.layers) {
          let sq = 0;
          for (const p of l.params) {
            if (p.frozen) continue;
            for (let i = 0; i < p.grad.length; i++) {
              p.grad[i] *= inv;
              sq += p.grad[i] * p.grad[i];
            }
          }
          if (l.params.length) gradSq[l.spec.id] = (gradSq[l.spec.id] ?? 0) + Math.sqrt(sq);
        }
        opt.step(model.params, lr);
        lossSum += batchLoss;
        cb.onBatch({ epoch: this.startEpoch + e + 1, batch: b + 1, batches, loss: batchLoss / idx.length, lr });
        if (performance.now() - slice > 24 || this.paused) {
          await this.yieldToUi();
          slice = performance.now();
        }
        if (this.stopped) return 'stopped';
      }
      const ev = evaluate(model, this.val, cfg, task);
      const gradNorms = Object.fromEntries(Object.entries(gradSq).map(([k, v]) => [k, v / batches]));
      for (const l of model.layers) {
        const g = gradNorms[l.spec.id];
        if (g === undefined) continue;
        if (g < VANISH) cb.onLog('warn', `Vanishing gradient in "${l.spec.name}" (norm ${g.toExponential(1)})`);
        if (g > EXPLODE) cb.onLog('warn', `Exploding gradient in "${l.spec.name}" (norm ${g.toExponential(1)})`);
      }
      const metrics: Partial<Record<MetricName, number>> = {};
      if (ev.metrics) for (const m of cfg.metrics) metrics[m] = ev.metrics[m];
      cb.onEpoch({
        epoch: this.startEpoch + e + 1,
        trainLoss: lossSum / this.train.length,
        valLoss: ev.loss,
        trainAcc: task.kind === 'classification' ? correct / this.train.length : null,
        valAcc: ev.metrics ? ev.metrics.accuracy : null,
        metrics,
        lr,
        gradNorms,
        seconds: (performance.now() - started) / 1000,
      });
      await this.yieldToUi();
      slice = performance.now();
      if (this.stopped) return 'stopped';
    }
    return 'done';
  }
}
