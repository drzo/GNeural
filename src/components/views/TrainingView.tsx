import { Activity, ChartLine, Pause, Play, Square } from 'lucide-react';
import { useMemo } from 'react';
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { SCHEDULE_LABELS, scheduledRate } from '../../engine/training';
import { num, pct } from '../../lib/format';
import { pauseTraining, resumeTraining, startTraining, stopTraining } from '../../store/actions';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';
import { CHART, ChartCard, Legend, axisProps, tooltipProps } from '../ui/charts';
import { Empty, Stat } from '../ui/primitives';
import { RunHistory } from './training/RunHistory';

function Progress({ label, value, max, tone }: { label: string; value: number; max: number; tone: string }) {
  const p = max ? Math.min(1, value / max) : 0;
  return (
    <div className="min-w-0 flex-1">
      <div className="mb-1.5 flex justify-between text-[11px]">
        <span className="text-faint">{label}</span>
        <span className="font-mono text-muted">{value} / {max || '-'}</span>
      </div>
      <div className="relative h-1.5 overflow-hidden rounded-full bg-raised">
        <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${p * 100}%`, background: tone }} />
      </div>
    </div>
  );
}

export function TrainingView() {
  const t = useStore((s) => s.training);
  const cfg = useStore((s) => s.config.train);
  const net = useNetwork();
  const last = t.history[t.history.length - 1];
  const classification = net.dataset?.task === 'classification';
  const running = t.status === 'running';

  const lrCurve = useMemo(() => {
    const epochs = Math.max(1, t.totalEpochs || cfg.epochs);
    return Array.from({ length: 61 }, (_, i) => ({ epoch: +((i / 60) * epochs).toFixed(2), lr: scheduledRate(cfg.schedule, cfg.learningRate, i / 60) }));
  }, [cfg.schedule, cfg.learningRate, cfg.epochs, t.totalEpochs]);

  const history = useMemo(() => t.history.map((h) => ({ ...h, trainAccP: h.trainAcc !== null ? h.trainAcc * 100 : null, valAccP: h.valAcc !== null ? h.valAcc * 100 : null })), [t.history]);

  const grads = useMemo(() => {
    if (!last) return [];
    const names = new Map(net.chain.map((c) => [c.id, c.name]));
    return Object.entries(last.gradNorms).map(([id, g]) => ({ name: names.get(id) ?? id, g: Math.log10(Math.max(g, 1e-12)) }));
  }, [last, net.chain]);

  const bestVal = t.history.reduce<number | null>((b, h) => (h.valAcc !== null && (b === null || h.valAcc > b) ? h.valAcc : b), null);

  if (!t.history.length && t.status === 'idle') {
    return (
      <div className="flex h-full flex-col overflow-y-auto p-6">
        <div className="min-h-[320px] flex-1">
          <Empty
            icon={<ChartLine size={22} />}
            title="No training yet"
            body="Training charts appear here as soon as you start. Loss should go down; accuracy should go up."
            action={<button className="btn-primary" onClick={startTraining} disabled={!net.valid}><Play size={14} fill="currentColor" /> Start training</button>}
          />
        </div>
        <RunHistory />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="card mb-4 flex items-center gap-6 px-5 py-4">
        <div className="flex items-center gap-3">
          <span className={`flex h-9 w-9 items-center justify-center rounded-xl ${running ? 'bg-ok-500/15 text-ok-400' : t.status === 'paused' ? 'bg-warn-500/15 text-warn-400' : 'bg-raised text-faint'}`}>
            <Activity size={17} className={running ? 'animate-pulse' : ''} />
          </span>
          <div>
            <div className="text-[14px] font-semibold capitalize">{running ? 'Training' : t.status === 'paused' ? 'Paused' : 'Finished'}</div>
            <div className="text-[11px] text-faint">{t.engine === 'gneural' ? 'Gneural library' : `${cfg.optimizer.toUpperCase()}, ${SCHEDULE_LABELS[cfg.schedule].toLowerCase()} schedule`}</div>
          </div>
        </div>
        <Progress label="Epoch" value={t.epoch} max={t.totalEpochs} tone={CHART.blue} />
        <Progress label="Batch" value={t.batch} max={t.batches} tone={CHART.violet} />
        <div className="flex gap-2">
          {running && t.engine === 'workbench' && <button className="btn-outline" onClick={pauseTraining}><Pause size={14} /> Pause</button>}
          {t.status === 'paused' && <button className="btn-primary" onClick={resumeTraining}><Play size={14} fill="currentColor" /> Resume</button>}
          {t.status !== 'idle' ? (
            <button className="btn-danger" onClick={stopTraining}><Square size={13} /> Stop</button>
          ) : (
            <button className="btn-primary" onClick={startTraining} disabled={!net.valid}><Play size={14} fill="currentColor" /> Continue</button>
          )}
        </div>
      </div>

      <div className="mb-4 grid grid-cols-5 gap-3">
        <Stat label="Train loss" value={num(last?.trainLoss)} tone="brand" />
        <Stat label="Val loss" value={num(last?.valLoss)} />
        <Stat label="Val accuracy" value={classification ? pct(last?.valAcc) : '-'} sub={bestVal !== null ? `best ${pct(bestVal)}` : undefined} tone="ok" />
        <Stat label="Learning rate" value={last ? last.lr.toExponential(2) : cfg.learningRate.toExponential(2)} />
        <Stat label="Time / epoch" value={last ? `${last.seconds.toFixed(2)}s` : '-'} />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <ChartCard title="Loss" right={<Legend items={[['Train', CHART.blue], ['Validation', CHART.yellow]]} />} className="h-[280px]">
          <ResponsiveContainer>
            <LineChart data={history} margin={{ left: -12, right: 8, top: 4 }}>
              <CartesianGrid stroke={CHART.grid} vertical={false} />
              <XAxis dataKey="epoch" {...axisProps} />
              <YAxis {...axisProps} tickFormatter={(v: number) => v.toFixed(2)} />
              <Tooltip {...tooltipProps} formatter={(v) => Number(v).toFixed(4)} />
              <Line type="monotone" dataKey="trainLoss" name="Train" stroke={CHART.blue} strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="valLoss" name="Validation" stroke={CHART.yellow} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {classification ? (
          <ChartCard title="Accuracy" right={<Legend items={[['Train', CHART.green], ['Validation', CHART.violet]]} />} className="h-[280px]">
            <ResponsiveContainer>
              <LineChart data={history} margin={{ left: -12, right: 8, top: 4 }}>
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="epoch" {...axisProps} />
                <YAxis {...axisProps} domain={[0, 100]} unit="%" />
                <Tooltip {...tooltipProps} formatter={(v) => `${Number(v).toFixed(1)}%`} />
                <Line type="monotone" dataKey="trainAccP" name="Train" stroke={CHART.green} strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="valAccP" name="Validation" stroke={CHART.violet} strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        ) : (
          <ChartCard title="Batch loss" right={<span className="text-[11px] text-faint">every update</span>} className="h-[280px]">
            <BatchLoss data={t.batchLoss} />
          </ChartCard>
        )}

        <ChartCard title="Learning-rate schedule" right={<span className="text-[11px] text-faint">{SCHEDULE_LABELS[cfg.schedule]}</span>} className="h-[240px]">
          <ResponsiveContainer>
            <AreaChart data={lrCurve} margin={{ left: -4, right: 8, top: 4 }}>
              <defs>
                <linearGradient id="lrFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={CHART.violet} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={CHART.violet} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={CHART.grid} vertical={false} />
              <XAxis dataKey="epoch" type="number" domain={[0, 'dataMax']} {...axisProps} />
              <YAxis {...axisProps} tickFormatter={(v: number) => v.toExponential(0)} />
              <Tooltip {...tooltipProps} formatter={(v) => Number(v).toExponential(3)} />
              <Area type="monotone" dataKey="lr" name="Learning rate" stroke={CHART.violet} fill="url(#lrFill)" strokeWidth={2} isAnimationActive={false} />
              {last && <ReferenceDot x={last.epoch} y={last.lr} r={5} fill={CHART.violet} stroke="#fff" strokeWidth={2} />}
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        {classification ? (
          <ChartCard title="Batch loss" right={<span className="text-[11px] text-faint">every update</span>} className="h-[240px]">
            <BatchLoss data={t.batchLoss} />
          </ChartCard>
        ) : (
          <GradientChart grads={grads} />
        )}

        {classification && <GradientChart grads={grads} />}

        {classification && last && Object.keys(last.metrics).length > 0 && (
          <ChartCard title="Validation metrics (latest epoch)" className="h-[240px]">
            <div className="grid h-full grid-cols-2 content-center gap-3">
              {Object.entries(last.metrics).map(([k, v]) => (
                <div key={k} className="rounded-lg bg-raised px-4 py-3">
                  <div className="label">{k}</div>
                  <div className="mt-1 font-mono text-[22px] font-semibold">{pct(v)}</div>
                  <div className="mt-2 h-1 rounded-full bg-line"><div className="h-full rounded-full bg-ok-500 transition-all" style={{ width: `${(v ?? 0) * 100}%` }} /></div>
                </div>
              ))}
            </div>
          </ChartCard>
        )}
      </div>

      <div className="mt-4">
        <RunHistory />
      </div>
    </div>
  );
}

function BatchLoss({ data }: { data: { step: number; loss: number }[] }) {
  return (
    <ResponsiveContainer>
      <AreaChart data={data} margin={{ left: -12, right: 8, top: 4 }}>
        <defs>
          <linearGradient id="bFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={CHART.blue} stopOpacity={0.3} />
            <stop offset="100%" stopColor={CHART.blue} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={CHART.grid} vertical={false} />
        <XAxis dataKey="step" {...axisProps} />
        <YAxis {...axisProps} tickFormatter={(v: number) => v.toFixed(2)} />
        <Tooltip {...tooltipProps} formatter={(v) => Number(v).toFixed(4)} labelFormatter={(l) => `Step ${l}`} />
        <Area type="monotone" dataKey="loss" stroke={CHART.blue} fill="url(#bFill)" strokeWidth={1.5} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function GradientChart({ grads }: { grads: { name: string; g: number }[] }) {
  return (
    <ChartCard title="Gradient norm per layer" right={<span className="text-[11px] text-faint">log10, latest epoch</span>} className="h-[240px]">
      {grads.length ? (
        <ResponsiveContainer>
          <BarChart data={grads} margin={{ left: -12, right: 8, top: 4 }}>
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="name" {...axisProps} interval={0} tick={{ fontSize: 10 }} />
            <YAxis {...axisProps} />
            <Tooltip {...tooltipProps} formatter={(v) => `10^${Number(v).toFixed(2)}`} />
            <Bar dataKey="g" name="Gradient norm" fill={CHART.green} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <div className="flex h-full items-center justify-center text-[12px] text-faint">Available with the Workbench engine after the first epoch.</div>
      )}
    </ChartCard>
  );
}
