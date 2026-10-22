import { Database, FileUp, Image, ListOrdered, Table2, Type } from 'lucide-react';
import { useMemo, useRef, useState, type ChangeEvent } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { BUILTIN_DATASETS, parseCsv, splitDataset, type DataKind } from '../../data/datasets';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';
import { CHART, ChartCard, axisProps, tooltipProps } from '../ui/charts';
import { Heatmap } from '../ui/Heatmap';
import { Empty, Toggle } from '../ui/primitives';

const KIND_ICON: Record<DataKind, typeof Image> = { image: Image, sequence: ListOrdered, text: Type, tabular: Table2 };
const PALETTE = [CHART.blue, CHART.green, CHART.yellow, '#1fc7b6', '#e74c3c', '#f06292', '#8bc34a', '#ff9800', '#4dd0e1', '#a1887f'];
const MAX_CSV = 2 * 1024 * 1024;

function Sparkline({ x }: { x: Float32Array }) {
  const pts = Array.from(x, (v, i) => `${(i / Math.max(1, x.length - 1)) * 100},${30 - v * 24}`).join(' ');
  return (
    <svg viewBox="0 0 100 60" className="h-[72px] w-full">
      <line x1="0" x2="100" y1="30" y2="30" stroke="#8b93a733" />
      <polyline points={pts} fill="none" stroke="#1fc7b6" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

export function DatasetView() {
  const net = useNetwork();
  const config = useStore((s) => s.config);
  const setConfig = useStore((s) => s.setConfig);
  const log = useStore((s) => s.log);
  const running = useStore((s) => s.training.status !== 'idle');
  const fileRef = useRef<HTMLInputElement>(null);
  const [page, setPage] = useState(0);
  const ds = net.dataset;

  const split = useMemo(() => (ds ? splitDataset(ds, config.splits) : null), [ds, config.splits]);
  const distribution = useMemo(() => {
    if (!ds || ds.task !== 'classification') return [];
    const counts = ds.classes.map((name) => ({ name, count: 0 }));
    for (const s of ds.samples) if (counts[s.label]) counts[s.label].count++;
    return counts;
  }, [ds]);

  const choose = (id: string) => {
    if (running) return log('warn', 'Stop training before switching datasets', 'dataset');
    setConfig({ datasetId: id });
    setPage(0);
  };

  const upload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    if (running) return log('warn', 'Stop training before switching datasets', 'dataset');
    if (file.size > MAX_CSV) return log('error', 'CSV files are limited to 2 MB', 'dataset');
    try {
      const text = await file.text();
      const parsed = parseCsv(text, file.name);
      setConfig({ datasetId: 'csv', csv: { text, name: file.name } });
      setPage(0);
      log('success', `Loaded ${file.name}: ${parsed.samples.length} rows, ${parsed.inputShape[0]} features, ${parsed.task}`, 'dataset');
    } catch (err) {
      log('error', `Could not read ${file.name}: ${(err as Error).message}`, 'dataset');
    }
  };

  const setSplit = (key: 'train' | 'val', v: number) => {
    const other = key === 'train' ? config.splits.val : config.splits.train;
    const clamped = Math.min(v, 0.95 - other);
    setConfig({ splits: { ...config.splits, [key]: Math.max(key === 'train' ? 0.1 : 0, clamped) } });
  };

  const test = Math.max(0, 1 - config.splits.train - config.splits.val);
  const perPage = 24;
  const samples = ds ? ds.samples.slice(page * perPage, page * perPage + perPage) : [];
  const pages = ds ? Math.ceil(ds.samples.length / perPage) : 0;

  return (
    <div className="flex h-full">
      <div className="w-[320px] shrink-0 space-y-5 overflow-y-auto border-r border-line p-5">
        <div data-tour="dataset-list">
          <div className="label mb-2">Datasets</div>
          <div className="space-y-1.5">
            {BUILTIN_DATASETS.map((d) => {
              const Icon = KIND_ICON[d.kind];
              const active = config.datasetId === d.id;
              return (
                <button key={d.id} onClick={() => choose(d.id)} className={`flex w-full gap-3 rounded-xl border px-3 py-2.5 text-left transition-all ${active ? 'border-brand-500/60 bg-brand-500/10' : 'border-line hover:border-faint hover:bg-raised'}`}>
                  <Icon size={16} className={`mt-0.5 shrink-0 ${active ? 'text-brand-400' : 'text-muted'}`} />
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium">{d.name}</div>
                    <div className="line-clamp-2 text-[11px] leading-snug text-faint">{d.description}</div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div>
          <div className="label mb-2">Your data</div>
          <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden" onChange={upload} />
          <button onClick={() => fileRef.current?.click()} className={`flex w-full flex-col items-center gap-1.5 rounded-xl border border-dashed px-3 py-5 text-center transition-colors ${config.datasetId === 'csv' ? 'border-brand-500/60 bg-brand-500/10' : 'border-line hover:border-faint hover:bg-raised'}`}>
            <FileUp size={18} className="text-muted" />
            <span className="text-[12px] font-medium">{config.csv ? config.csv.name : 'Upload a CSV file'}</span>
            <span className="text-[11px] text-faint">Numeric columns, last column is the target</span>
          </button>
          {config.csv && config.datasetId !== 'csv' && (
            <button className="btn-ghost mt-1.5 h-7 w-full text-[12px]" onClick={() => choose('csv')}>Use {config.csv.name}</button>
          )}
        </div>

        <div data-tour="dataset-splits">
          <div className="label mb-3">Train / validation / test split</div>
          <div className="mb-3 flex h-2.5 overflow-hidden rounded-full">
            <div className="bg-brand-500 transition-all" style={{ width: `${config.splits.train * 100}%` }} />
            <div className="bg-warn-500 transition-all" style={{ width: `${config.splits.val * 100}%` }} />
            <div className="bg-ok-500 transition-all" style={{ width: `${test * 100}%` }} />
          </div>
          {(['train', 'val'] as const).map((k) => (
            <label key={k} className="mb-2 flex items-center gap-3 text-[12px]">
              <span className="w-16 text-muted">{k === 'train' ? 'Train' : 'Validation'}</span>
              <input type="range" min={0} max={0.95} step={0.05} value={config.splits[k]} onChange={(e) => setSplit(k, Number(e.target.value))} className={`flex-1 ${k === 'train' ? 'accent-brand-500' : 'accent-warn-500'}`} />
              <span className="w-10 text-right font-mono text-faint">{Math.round(config.splits[k] * 100)}%</span>
            </label>
          ))}
          <div className="flex items-center gap-3 text-[12px]">
            <span className="w-16 text-muted">Test</span>
            <span className="flex-1 text-faint">the remainder</span>
            <span className="w-10 text-right font-mono text-faint">{Math.round(test * 100)}%</span>
          </div>
          {split && (
            <div className="mt-3 grid grid-cols-3 gap-2 text-center">
              {[['train', split.train.length, 'text-brand-400'], ['val', split.val.length, 'text-warn-400'], ['test', split.test.length, 'text-ok-400']].map(([k, v, c]) => (
                <div key={k as string} className="rounded-lg bg-raised py-2">
                  <div className={`font-mono text-[15px] font-semibold ${c}`}>{v}</div>
                  <div className="text-[10px] uppercase tracking-wide text-faint">{k}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div data-tour="dataset-augmentation">
          <div className="label mb-1">Augmentation</div>
          <p className="mb-3 text-[11px] text-faint">Random changes applied to training samples so the network generalizes better.</p>
          <div className="space-y-2.5">
            {([
              ['noise', 'Gaussian noise', 'Adds small random jitter to every value', true],
              ['scale', 'Random scaling', 'Multiplies values by 0.85 to 1.15', true],
              ['shift', 'Random shift', 'Moves images by up to one pixel', ds?.kind === 'image'],
              ['flip', 'Horizontal flip', 'Mirrors images left to right', ds?.kind === 'image'],
            ] as const).map(([k, title, hint, enabled]) => (
              <div key={k} className={`flex items-center justify-between gap-3 ${enabled ? '' : 'opacity-40'}`}>
                <div>
                  <div className="text-[12px]">{title}</div>
                  <div className="text-[11px] text-faint">{enabled ? hint : 'Images only'}</div>
                </div>
                <Toggle checked={config.augmentation[k] && !!enabled} onChange={(v) => enabled && setConfig({ augmentation: { ...config.augmentation, [k]: v } })} />
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="min-w-0 flex-1 overflow-y-auto p-6">
        {!ds ? (
          <Empty icon={<Database size={22} />} title="Dataset unavailable" body={net.datasetError ?? 'Choose a dataset on the left.'} />
        ) : (
          <div className="space-y-5">
            <div>
              <h2 className="text-[18px] font-semibold">{ds.name}</h2>
              <p className="mt-1 max-w-2xl text-[13px] text-muted">{ds.description}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <span className="chip bg-raised text-muted">{ds.samples.length} samples</span>
                <span className="chip bg-raised text-muted">{ds.task}</span>
                <span className="chip bg-raised font-mono text-muted">input [{ds.inputShape.join(', ')}]</span>
                <span className="chip bg-raised font-mono text-muted">target [{ds.targetShape.join(', ')}]</span>
                {net.chain[0] && net.analysis[0]?.error && <span className="chip bg-bad-500/15 text-bad-400">Input layer shape does not match</span>}
              </div>
            </div>

            {distribution.length > 0 && (
              <ChartCard title="Class distribution" className="h-[220px]">
                <ResponsiveContainer>
                  <BarChart data={distribution} margin={{ left: -16, right: 8, top: 4 }}>
                    <CartesianGrid stroke={CHART.grid} vertical={false} />
                    <XAxis dataKey="name" {...axisProps} interval={0} />
                    <YAxis {...axisProps} allowDecimals={false} />
                    <Tooltip {...tooltipProps} cursor={{ fill: '#8b93a714' }} />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {distribution.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            )}

            <div className="card p-5">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-[13px] font-semibold">Samples</h3>
                <div className="flex items-center gap-2 text-[12px] text-faint">
                  <button className="btn-ghost h-7 px-2" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>Prev</button>
                  <span className="font-mono">{page + 1} / {pages}</span>
                  <button className="btn-ghost h-7 px-2" disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)}>Next</button>
                </div>
              </div>
              {ds.kind === 'tabular' ? (
                <div className="overflow-x-auto">
                  <table className="w-full font-mono text-[11px]">
                    <thead>
                      <tr className="text-left text-faint">
                        {ds.featureNames.map((f) => <th key={f} className="px-2 py-1.5 font-medium">{f}</th>)}
                        <th className="px-2 py-1.5 font-medium text-brand-400">target</th>
                      </tr>
                    </thead>
                    <tbody>
                      {samples.map((s, r) => (
                        <tr key={r} className="border-t border-line/60">
                          {Array.from(s.x, (v, j) => {
                            const raw = ds.normalize ? v * ds.normalize.std[j] + ds.normalize.mean[j] : v;
                            return <td key={j} className="px-2 py-1.5 text-muted">{Number(raw.toFixed(3))}</td>;
                          })}
                          <td className="px-2 py-1.5 text-fg">{s.label >= 0 ? ds.classes[s.label] : s.y[0].toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className={`grid gap-3 ${ds.kind === 'image' ? 'grid-cols-8' : ds.kind === 'text' ? 'grid-cols-2' : 'grid-cols-4'}`}>
                  {samples.map((s, i) => (
                    <div key={i} className="rounded-xl border border-line bg-bg p-2 transition-colors hover:border-faint">
                      {ds.kind === 'image' ? (
                        <Heatmap values={s.x} rows={ds.inputShape[0]} cols={ds.inputShape[1]} size={96} gray className="!w-full !h-auto" />
                      ) : ds.kind === 'text' ? (
                        <p className="min-h-[40px] text-[12px] leading-snug text-muted">{s.text ?? '(encoded text)'}</p>
                      ) : (
                        <Sparkline x={s.x} />
                      )}
                      <div className="mt-1.5 truncate text-center text-[11px] text-faint">
                        {s.label >= 0 ? ds.classes[s.label] : `next: ${s.y[0].toFixed(2)}`}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
