import { CircleCheck as CheckCircle2, Cpu, Gauge, Pause, Play, RotateCcw, Square, Triangle } from 'lucide-react';
import { useMemo } from 'react';
import { LOSS_LABELS, OPTIMIZER_LABELS, SCHEDULE_LABELS } from '../../engine/training';
import type { LossName, MetricName, OptimizerName, ScheduleName, TrainConfig } from '../../engine/types';
import { gneuralCompatibility } from '../../gd/bridge';
import { flashFlow, pauseTraining, resetWeights, startTraining, stopTraining } from '../../store/actions';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';
import { HelpTip, Section, Segmented } from '../ui/primitives';

const METRICS: { id: MetricName; label: string; help: string }[] = [
  { id: 'accuracy', label: 'Accuracy', help: 'Share of samples classified correctly.' },
  { id: 'precision', label: 'Precision', help: 'Of the samples predicted as a class, how many really were (averaged across classes).' },
  { id: 'recall', label: 'Recall', help: 'Of the samples truly in a class, how many were found (averaged across classes).' },
  { id: 'f1', label: 'F1 score', help: 'The balance of precision and recall.' },
];

function Field({ label, help, children }: { label: string; help: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-center gap-1.5 text-[12px] font-medium text-muted">{label}<HelpTip text={help} /></span>
      {children}
    </label>
  );
}

function NumberInput({ value, onChange, min, max, step }: { value: number; onChange(v: number): void; min?: number; max?: number; step?: number }) {
  return (
    <input
      type="number"
      className="input font-mono"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (Number.isFinite(v)) onChange(max !== undefined ? Math.min(max, v) : v);
      }}
    />
  );
}

export function RunPanel() {
  const cfg = useStore((s) => s.config.train);
  const setTrain = useStore((s) => s.setTrain);
  const status = useStore((s) => s.training.status);
  const history = useStore((s) => s.training.history.length);
  const expert = useStore((s) => s.mode === 'expert');
  const setView = useStore((s) => s.setView);
  const net = useNetwork();
  const compat = useMemo(() => gneuralCompatibility(net.chain, net.analysis), [net]);
  const classification = net.dataset?.task === 'classification';
  const gd = cfg.engine === 'gneural';
  const locked = status !== 'idle';
  const set = <K extends keyof TrainConfig>(k: K) => (v: TrainConfig[K]) => setTrain({ [k]: v } as Partial<TrainConfig>);

  return (
    <div>
      <Section title={<span className="flex items-center gap-1.5"><Cpu size={12} /> Engine</span>}>
        <div data-tour="engine">
        <Segmented
          value={cfg.engine}
          onChange={(v) => setTrain({ engine: v })}
          options={[
            { value: 'workbench', label: 'Workbench', title: 'Full-featured in-browser engine supporting every layer type' },
            { value: 'gneural', label: 'Gneural', title: 'The Gneural library, the same math that runs inside Geometry Dash' },
          ]}
        />
        </div>
        <div className={`mt-3 flex gap-2 rounded-lg px-3 py-2 text-[12px] ${compat.ok ? 'bg-ok-500/10 text-ok-400' : 'bg-raised text-muted'}`}>
          {compat.ok ? <CheckCircle2 size={14} className="mt-0.5 shrink-0" /> : <Triangle size={13} className="mt-0.5 shrink-0 text-warn-400" />}
          <span>
            {compat.ok
              ? `Geometry Dash compatible (${compat.arch.inputs}-${compat.arch.hidden.join('-')}-${compat.arch.outputs}). Export it as a level from the Export menu.`
              : gd ? `Gneural cannot run this network: ${compat.reason}` : `Not Geometry Dash compatible: ${compat.reason}`}
          </span>
        </div>
      </Section>

      <Section title="Optimization">
        <div className="space-y-3.5">
          {!gd && (
            <Field label="Optimizer" help="The algorithm that updates weights. Adam adapts the step size per weight and is a reliable default.">
              <select className="input" value={cfg.optimizer} disabled={locked} onChange={(e) => set('optimizer')(e.target.value as OptimizerName)}>
                {Object.entries(OPTIMIZER_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </Field>
          )}
          {!gd && (
            <Field label="Loss function" help="How wrong a prediction is. CrossEntropy suits classification; MSE suits predicting numbers or reconstructing inputs.">
              <select className="input" value={cfg.loss} disabled={locked} onChange={(e) => set('loss')(e.target.value as LossName)}>
                {Object.entries(LOSS_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </Field>
          )}
          {gd && <p className="text-[12px] text-muted">Gneural trains with plain gradient descent on squared error, one sample at a time.</p>}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Epochs" help="How many full passes over the training data.">
              <NumberInput value={cfg.epochs} min={1} max={gd ? 5000 : 500} onChange={(v) => set('epochs')(Math.max(1, Math.round(v)))} />
            </Field>
            {!gd && (
              <Field label="Batch size" help="Samples averaged per weight update. Smaller is noisier but often generalizes better.">
                <NumberInput value={cfg.batchSize} min={1} max={1024} onChange={(v) => set('batchSize')(Math.max(1, Math.round(v)))} />
              </Field>
            )}
            <Field label="Learning rate" help="Step size for each update. Too high diverges, too low crawls.">
              <NumberInput value={cfg.learningRate} min={0.000001} max={2} step={0.0005} onChange={set('learningRate')} />
            </Field>
          </div>
          {!gd && (
            <Field label="Schedule" help="How the learning rate changes over training. Cosine annealing often squeezes out extra accuracy.">
              <select className="input" value={cfg.schedule} disabled={locked} onChange={(e) => set('schedule')(e.target.value as ScheduleName)}>
                {Object.entries(SCHEDULE_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </Field>
          )}
        </div>
      </Section>

      {!gd && expert && (
        <Section title="Advanced optimizer">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Momentum" help="SGD momentum factor."><NumberInput value={cfg.momentum} min={0} max={0.999} step={0.01} onChange={set('momentum')} /></Field>
            <Field label="Beta 1" help="Adam first-moment decay."><NumberInput value={cfg.beta1} min={0} max={0.9999} step={0.01} onChange={set('beta1')} /></Field>
            <Field label="Beta 2" help="Adam / RMSProp second-moment decay."><NumberInput value={cfg.beta2} min={0} max={0.99999} step={0.001} onChange={set('beta2')} /></Field>
            <Field label="Epsilon" help="Numerical stability constant."><NumberInput value={cfg.epsilon} min={0} max={1} step={1e-8} onChange={set('epsilon')} /></Field>
            <Field label="Clip norm" help="Rescale gradients whose global norm exceeds this. 0 disables clipping."><NumberInput value={cfg.clipNorm} min={0} max={100} step={0.5} onChange={set('clipNorm')} /></Field>
            <Field label="Seed" help="Random seed for weight initialization."><NumberInput value={cfg.seed} min={0} max={99999} onChange={(v) => set('seed')(Math.round(v))} /></Field>
          </div>
        </Section>
      )}

      {!gd && classification && (
        <Section title="Metrics">
          <div className="grid grid-cols-2 gap-2">
            {METRICS.map((m) => {
              const on = cfg.metrics.includes(m.id);
              return (
                <label key={m.id} className={`flex cursor-pointer items-center gap-2 rounded-lg border px-2.5 py-2 text-[12px] transition-colors ${on ? 'border-brand-500/50 bg-brand-500/10 text-fg' : 'border-line text-muted hover:bg-raised'}`}>
                  <input
                    type="checkbox"
                    className="accent-brand-500"
                    checked={on}
                    onChange={() => setTrain({ metrics: on ? cfg.metrics.filter((x) => x !== m.id) : [...cfg.metrics, m.id] })}
                  />
                  {m.label}
                  <span className="ml-auto"><HelpTip text={m.help} /></span>
                </label>
              );
            })}
          </div>
        </Section>
      )}

      <div className="space-y-2 px-4 py-4">
        {status === 'running' ? (
          <div className="flex gap-2">
            {!gd && <button className="btn-outline flex-1" onClick={pauseTraining}><Pause size={14} /> Pause</button>}
            <button className="btn-danger flex-1" onClick={stopTraining}><Square size={13} /> Stop</button>
          </div>
        ) : (
          <button className="btn-primary h-10 w-full text-[14px]" data-tour="train-button" onClick={startTraining} disabled={!net.valid}>
            <Play size={15} fill="currentColor" /> {status === 'paused' ? 'Resume training' : history ? 'Continue training' : 'Train'}
          </button>
        )}
        <div className="flex gap-2">
          <button className="btn-outline flex-1" disabled={!net.valid} onClick={() => { setView('inference'); flashFlow(); }}>
            <Gauge size={14} /> Run inference
          </button>
          <button className="btn-ghost" disabled={locked || !net.valid} onClick={resetWeights} title="Re-initialize all weights and clear training history">
            <RotateCcw size={14} />
          </button>
        </div>
        {!net.valid && <p className="text-[11px] text-bad-400">Fix the issues shown in the console before training.</p>}
      </div>
    </div>
  );
}
