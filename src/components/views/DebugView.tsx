import { TriangleAlert as AlertTriangle, Bug, ChevronLeft, ChevronRight, CircleDot, FastForward, RotateCcw, Shuffle, Skull } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { splitDataset } from '../../data/datasets';
import type { Model } from '../../engine/model';
import { fmtShape } from '../../engine/shapes';
import { computeLoss } from '../../engine/training';
import type { LayerSpec, Sample } from '../../engine/types';
import { layerColor } from '../../lib/format';
import { histogram, stats } from '../../lib/stats';
import { ensureModel } from '../../store/actions';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';
import { CHART, ChartCard, axisProps, tooltipProps } from '../ui/charts';
import { activationGrids, Heatmap } from '../ui/Heatmap';
import { Empty } from '../ui/primitives';

const VANISH = 1e-7;
const EXPLODE = 1e3;

interface GradInfo {
  norms: number[];
  paramGrads: Float32Array[][];
}

function gradientsFor(model: Model, s: Sample, loss: Parameters<typeof computeLoss>[0], reconstruction: boolean): GradInfo {
  model.zeroGrad();
  const y = model.forward(s.x, false);
  const { grad } = computeLoss(loss, y, reconstruction ? s.x : s.y, model.outputActivation);
  model.backward(grad);
  const paramGrads = model.layers.map((l) => l.params.filter((p) => !p.frozen).map((p) => p.grad.slice()));
  const norms = paramGrads.map((gs) => Math.sqrt(gs.reduce((a, g) => a + g.reduce((b, v) => b + v * v, 0), 0)));
  model.zeroGrad();
  return { norms, paramGrads };
}

/** Fraction of units in a layer's last dimension that never activate across the probe samples. */
function deadUnits(model: Model, probe: Sample[], layer: number) {
  const units = model.layers[layer].outShape[model.layers[layer].outShape.length - 1];
  const alive = new Uint8Array(units);
  for (const s of probe) {
    const out = model.forwardTrace(s.x, layer).outputs[layer];
    for (let i = 0; i < out.length; i++) if (out[i] > 1e-6) alive[i % units] = 1;
  }
  let dead = 0;
  for (const a of alive) if (!a) dead++;
  return { dead, units };
}

function relu(spec: LayerSpec) {
  return spec.type === 'ReLU' || spec.params.activation === 'relu';
}

function HistogramChart({ data, color }: { data: ReturnType<typeof histogram>; color: string }) {
  return (
    <ResponsiveContainer>
      <BarChart data={data} margin={{ left: -18, right: 4, top: 4 }} barCategoryGap={1}>
        <CartesianGrid stroke={CHART.grid} vertical={false} />
        <XAxis dataKey="label" {...axisProps} interval={5} tick={{ fontSize: 10 }} />
        <YAxis {...axisProps} allowDecimals={false} />
        <Tooltip {...tooltipProps} labelFormatter={(l) => `around ${l}`} />
        <Bar dataKey="count" fill={color} radius={[2, 2, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DebugView() {
  const net = useNetwork();
  const cfg = useStore((s) => s.config);
  const weightsVersion = useStore((s) => s.weightsVersion);
  const training = useStore((s) => s.training.status === 'running');
  const log = useStore((s) => s.log);
  const setView = useStore((s) => s.setView);
  const [cursor, setCursor] = useState(0);
  const [breakpoints, setBreakpoints] = useState<Set<string>>(new Set());
  const [sampleIdx, setSampleIdx] = useState(0);
  const [model, setModel] = useState<Model | null>(null);

  const pool = useMemo(() => (net.dataset ? splitDataset(net.dataset, cfg.splits).val : []), [net.dataset, cfg.splits]);
  const sample = pool[sampleIdx % Math.max(1, pool.length)];
  const n = net.chain.length;

  useEffect(() => {
    setModel(net.valid ? ensureModel(net) : null);
  }, [net, weightsVersion]);

  useEffect(() => {
    if (cursor >= n) setCursor(Math.max(0, n - 1));
  }, [n, cursor]);

  const trace = useMemo(() => (model && sample ? model.forwardTrace(sample.x, Math.min(cursor, model.layers.length - 1)) : null), [model, sample, cursor, weightsVersion]);
  const grads = useMemo(
    () => (model && sample && !training ? gradientsFor(model, sample, cfg.train.loss, net.dataset?.task === 'reconstruction') : null),
    [model, sample, training, cfg.train.loss, net.dataset, weightsVersion],
  );
  const dead = useMemo(() => (model && net.chain[cursor] && relu(net.chain[cursor]) ? deadUnits(model, pool.slice(0, 64), cursor) : null), [model, pool, cursor, net.chain, weightsVersion]);

  if (!net.valid || !model) {
    return <Empty icon={<Bug size={22} />} title="Nothing to debug yet" body="Build a valid network first. The debugger steps a single sample through it one layer at a time." action={<button className="btn-primary" onClick={() => setView('graph')}>Open Graph view</button>} />;
  }

  const spec = net.chain[cursor];
  const out = trace?.outputs[cursor];
  const actStats = out ? stats(out) : null;
  const layer = model.layers[cursor];
  const weights = layer.params.filter((p) => !p.frozen && p.shape.length > 1);
  const weightValues = weights.length ? weights[0].value : null;
  const gradNorm = grads?.norms[cursor];
  const gradFlat = grads?.paramGrads[cursor]?.[0];

  const runToBreakpoint = () => {
    for (let i = cursor + 1; i < n; i++) {
      if (breakpoints.has(net.chain[i].id)) {
        setCursor(i);
        log('info', `Paused at breakpoint "${net.chain[i].name}"`, 'debug');
        return;
      }
    }
    setCursor(n - 1);
    log('info', 'Forward pass completed without hitting a breakpoint', 'debug');
  };

  const toggleBp = (id: string) =>
    setBreakpoints((b) => {
      const next = new Set(b);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="flex h-full">
      <div className="flex w-[300px] shrink-0 flex-col border-r border-line">
        <div className="border-b border-line p-4">
          <div className="mb-3 flex items-center justify-between">
            <span className="label">Forward pass</span>
            <button className="btn-ghost h-7 px-2 text-[12px]" onClick={() => setSampleIdx(Math.floor(Math.random() * pool.length))} disabled={!pool.length}><Shuffle size={12} /> Sample</button>
          </div>
          <div className="flex gap-1">
            <button className="btn-outline h-8 flex-1" onClick={() => setCursor(0)} title="Reset to input"><RotateCcw size={13} /></button>
            <button className="btn-outline h-8 flex-1" onClick={() => setCursor((c) => Math.max(0, c - 1))} disabled={cursor === 0} title="Step back"><ChevronLeft size={15} /></button>
            <button className="btn-primary h-8 flex-1" onClick={() => setCursor((c) => Math.min(n - 1, c + 1))} disabled={cursor >= n - 1} title="Step forward"><ChevronRight size={15} /></button>
            <button className="btn-outline h-8 flex-1" onClick={runToBreakpoint} disabled={cursor >= n - 1} title="Run to next breakpoint"><FastForward size={14} /></button>
          </div>
          <p className="mt-2 text-[11px] text-faint">Click the dot beside a layer to set a breakpoint.</p>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {net.chain.map((l, i) => {
            const role = i === 0 ? 'input' : i === n - 1 ? 'output' : 'hidden';
            const g = grads?.norms[i];
            const has = model.layers[i].params.some((p) => !p.frozen);
            const warn = has && g !== undefined && (g < VANISH || g > EXPLODE);
            return (
              <div key={l.id} className={`flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors ${i === cursor ? 'bg-brand-500/15' : 'hover:bg-raised'} ${i > cursor ? 'opacity-45' : ''}`}>
                <button onClick={() => toggleBp(l.id)} className="flex h-4 w-4 items-center justify-center" title="Toggle breakpoint">
                  <CircleDot size={13} className={breakpoints.has(l.id) ? 'fill-bad-500 text-bad-500' : 'text-line hover:text-faint'} />
                </button>
                <button className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={() => setCursor(i)}>
                  <span className="h-5 w-1 rounded-full" style={{ background: layerColor(l.type, role) }} />
                  <span className="min-w-0 flex-1 truncate text-[12px]">{l.name}</span>
                  {warn && <AlertTriangle size={12} className="text-warn-400" />}
                  <span className="font-mono text-[10px] text-faint">{fmtShape(net.analysis[i].outShape)}</span>
                </button>
              </div>
            );
          })}
        </div>
      </div>

      <div className="min-w-0 flex-1 overflow-y-auto p-6">
        <div className="mb-5 flex items-end justify-between">
          <div>
            <div className="label">Layer {cursor} of {n - 1}</div>
            <h2 className="mt-1 text-[18px] font-semibold">{spec.name} <span className="text-[13px] font-normal text-faint">{spec.type} / {fmtShape(net.analysis[cursor].outShape)}</span></h2>
          </div>
          {sample && net.dataset?.task === 'classification' && <span className="chip bg-raised text-muted">sample label: {net.dataset.classes[sample.label]}</span>}
        </div>

        <div className="mb-4 grid grid-cols-5 gap-3">
          {actStats && (
            [['Mean', actStats.mean.toFixed(3)], ['Std', actStats.std.toFixed(3)], ['Min', actStats.min.toFixed(3)], ['Max', actStats.max.toFixed(3)], ['Zeros', `${(actStats.zeros * 100).toFixed(0)}%`]].map(([k, v]) => (
              <div key={k} className="card px-4 py-3">
                <div className="label">{k}</div>
                <div className="mt-1 font-mono text-[16px] font-semibold">{v}</div>
              </div>
            ))
          )}
        </div>

        <div className="mb-4 space-y-2">
          {gradNorm !== undefined && layer.params.some((p) => !p.frozen) && gradNorm < VANISH && (
            <div className="flex items-center gap-2 rounded-lg bg-warn-500/10 px-3 py-2 text-[12px] text-warn-400"><AlertTriangle size={14} /> Vanishing gradient: this layer barely learns from this sample (norm {gradNorm.toExponential(1)}).</div>
          )}
          {gradNorm !== undefined && gradNorm > EXPLODE && (
            <div className="flex items-center gap-2 rounded-lg bg-bad-500/10 px-3 py-2 text-[12px] text-bad-400"><AlertTriangle size={14} /> Exploding gradient (norm {gradNorm.toExponential(1)}). Lower the learning rate or enable gradient clipping.</div>
          )}
          {dead && dead.dead > 0 && (
            <div className={`flex items-center gap-2 rounded-lg px-3 py-2 text-[12px] ${dead.dead / dead.units > 0.3 ? 'bg-bad-500/10 text-bad-400' : 'bg-raised text-muted'}`}>
              <Skull size={14} /> {dead.dead} of {dead.units} units ({((dead.dead / dead.units) * 100).toFixed(0)}%) never activated across {Math.min(64, pool.length)} validation samples.
            </div>
          )}
          {training && <div className="rounded-lg bg-raised px-3 py-2 text-[12px] text-muted">Gradient inspection is paused while training runs.</div>}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <ChartCard title="Activation distribution" className="h-[240px]">
            {out ? <HistogramChart data={histogram(out)} color={CHART.blue} /> : null}
          </ChartCard>
          <ChartCard title={weightValues ? `Weight distribution (${weights[0].name})` : 'Weight distribution'} className="h-[240px]">
            {weightValues ? <HistogramChart data={histogram(weightValues)} color={CHART.violet} /> : <div className="flex h-full items-center justify-center text-[12px] text-faint">This layer has no weights.</div>}
          </ChartCard>
          <ChartCard title="Gradient distribution" right={gradNorm !== undefined && layer.params.length ? <span className="font-mono text-[11px] text-faint">norm {gradNorm.toExponential(2)}</span> : undefined} className="h-[240px]">
            {gradFlat && gradFlat.length ? <HistogramChart data={histogram(gradFlat)} color={CHART.green} /> : <div className="flex h-full items-center justify-center text-[12px] text-faint">{training ? 'Paused during training.' : 'No trainable parameters here.'}</div>}
          </ChartCard>
          <ChartCard title="Output values" className="h-[240px]">
            {out && (
              <div className="flex h-full flex-wrap content-start gap-2 overflow-y-auto">
                {activationGrids(out, net.analysis[cursor].outShape, 8).map((g, k) => (
                  <div key={k} className="text-center">
                    <Heatmap values={g.data} rows={g.rows} cols={g.cols} size={g.rows === 1 ? 380 : 88} signed />
                    {g.label && <div className="font-mono text-[10px] text-faint">{g.label}</div>}
                  </div>
                ))}
              </div>
            )}
          </ChartCard>
        </div>

        {grads && (
          <ChartCard title="Gradient norm by layer (log10)" className="mt-4 h-[220px]">
            <ResponsiveContainer>
              <BarChart data={net.chain.map((l, i) => ({ name: l.name, g: grads.norms[i] > 0 ? Math.log10(grads.norms[i]) : null }))} margin={{ left: -12, right: 8, top: 4 }}>
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="name" {...axisProps} interval={0} tick={{ fontSize: 10 }} />
                <YAxis {...axisProps} />
                <Tooltip {...tooltipProps} formatter={(v) => `10^${Number(v).toFixed(2)}`} />
                <Bar dataKey="g" fill={CHART.yellow} radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
      </div>
    </div>
  );
}
