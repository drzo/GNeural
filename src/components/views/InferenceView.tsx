import { Gauge, Play, Sparkles, Target } from 'lucide-react';
import { useMemo, useState } from 'react';
import { splitDataset } from '../../data/datasets';
import { layerDef } from '../../engine/catalog';
import { argmax } from '../../engine/training';
import { fmtShape } from '../../engine/shapes';
import { layerColor, pct } from '../../lib/format';
import { flashFlow } from '../../store/actions';
import { useStore } from '../../store/useStore';
import { activationGrids, Heatmap } from '../ui/Heatmap';
import { Empty } from '../ui/primitives';
import { InputPanel, type InputValue } from './inference/InputPanel';
import { probabilities, useModelRun } from './useModelRun';

export function InferenceView() {
  const [input, setInput] = useState<InputValue | null>(null);
  const { run, error, net } = useModelRun(input?.x ?? null);
  const trained = useStore((s) => s.training.history.length > 0);
  const setView = useStore((s) => s.setView);
  const ds = net.dataset;
  const pool = useMemo(() => (ds ? splitDataset(ds, useStore.getState().config.splits).test : []), [ds]);
  const [layerIdx, setLayerIdx] = useState<number | null>(null);

  if (!ds) return <Empty icon={<Gauge size={22} />} title="No dataset" body={net.datasetError ?? 'Choose a dataset first.'} action={<button className="btn-primary" onClick={() => setView('dataset')}>Open Dataset view</button>} />;

  const out = run?.trace.outputs[run.trace.outputs.length - 1];
  const act = run?.model.outputActivation ?? 'linear';
  const probs = out && ds.task === 'classification' ? probabilities(out, act) : null;
  const ranked = probs ? probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 6) : [];
  const predicted = probs ? argmax(probs) : null;

  const attentionLayers = run ? run.trace.traces.map((t, i) => ({ t, i })).filter((x) => x.t?.attention?.length) : [];
  const featureLayers = run ? run.trace.traces.map((t, i) => ({ t, i })).filter((x) => x.t?.featureShape) : [];
  const focus = layerIdx ?? (featureLayers[0]?.i ?? null);

  return (
    <div className="flex h-full">
      <div className="w-[380px] shrink-0 space-y-5 overflow-y-auto border-r border-line p-5">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-[15px] font-semibold">Try the network</h2>
            <p className="text-[12px] text-faint">{ds.name} / {ds.task}</p>
          </div>
          <button className="btn-outline h-7 px-2 text-[12px]" onClick={flashFlow}><Play size={12} /> Animate</button>
        </div>
        {!trained && (
          <div className="flex gap-2 rounded-lg bg-warn-500/10 px-3 py-2 text-[12px] text-warn-400">
            <Sparkles size={14} className="mt-0.5 shrink-0" /> The network has not been trained yet, so predictions are random. Press Space to train.
          </div>
        )}
        <InputPanel ds={ds} pool={pool} value={input} onChange={setInput} />
      </div>

      <div className="min-w-0 flex-1 overflow-y-auto p-6">
        {error && <div className="mb-4 rounded-lg bg-bad-500/10 px-4 py-3 text-[13px] text-bad-400">{error}</div>}
        {run && out && (
          <div className="space-y-5">
            <div className="card p-5">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-[13px] font-semibold">Prediction</h3>
                {input?.label !== null && input?.label !== undefined && ds.classes[input.label] && (
                  <span className={`chip ${predicted === input.label ? 'bg-ok-500/15 text-ok-400' : 'bg-bad-500/15 text-bad-400'}`}>
                    <Target size={11} /> truth: {ds.classes[input.label]}
                  </span>
                )}
              </div>
              {probs ? (
                <div className="flex items-start gap-8">
                  <div className="text-center">
                    <div className="font-mono text-[44px] font-semibold leading-none text-brand-400">{ds.classes[predicted!] ?? predicted}</div>
                    <div className="mt-2 text-[12px] text-muted">{pct(probs[predicted!])} confidence</div>
                  </div>
                  <div className="flex-1 space-y-2">
                    {ranked.map(({ p, i }, r) => (
                      <div key={i} className="flex items-center gap-3 text-[12px]">
                        <span className="w-20 truncate text-right text-muted">{ds.classes[i] ?? i}</span>
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-raised">
                          <div className="h-full rounded-full transition-all duration-500" style={{ width: `${p * 100}%`, background: r === 0 ? '#4f8ef7' : '#8b93a7' }} />
                        </div>
                        <span className="w-12 text-right font-mono text-faint">{pct(p)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : ds.task === 'reconstruction' && ds.inputShape.length === 3 ? (
                <div className="flex items-center gap-8">
                  {[['Input', input!.x], ['Reconstruction', out]].map(([label, v]) => (
                    <div key={label as string} className="text-center">
                      <Heatmap values={v as Float32Array} rows={ds.inputShape[0]} cols={ds.inputShape[1]} size={140} gray />
                      <div className="mt-2 text-[12px] text-muted">{label as string}</div>
                    </div>
                  ))}
                  <div className="text-[12px] text-muted">
                    Mean squared error<br />
                    <span className="font-mono text-[20px] text-fg">{(Array.from(out).reduce((s, v, i) => s + (v - input!.x[i]) ** 2, 0) / out.length).toFixed(4)}</span>
                  </div>
                </div>
              ) : (
                <div className="flex flex-wrap gap-4">
                  {Array.from(out).map((v, i) => (
                    <div key={i} className="rounded-xl bg-raised px-5 py-3">
                      <div className="label">Output {i + 1}</div>
                      <div className="font-mono text-[28px] font-semibold text-brand-400">{v.toFixed(4)}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="card p-5">
              <h3 className="mb-1 text-[13px] font-semibold">Layer activations</h3>
              <p className="mb-4 text-[12px] text-faint">Each strip shows the values a layer passed on. Brighter means larger.</p>
              <div className="space-y-2.5">
                {net.chain.map((l, i) => {
                  const v = run.trace.outputs[i];
                  const shape = net.analysis[i].outShape;
                  const g = activationGrids(v, shape, 1)[0];
                  const role = i === 0 ? 'input' : i === net.chain.length - 1 ? 'output' : 'hidden';
                  return (
                    <button key={l.id} onClick={() => setLayerIdx(i)} className={`flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left transition-colors ${focus === i ? 'bg-raised' : 'hover:bg-raised/60'}`}>
                      <span className="h-6 w-1 rounded-full" style={{ background: layerColor(l.type, role) }} />
                      <span className="w-32 truncate text-[12px]">{l.name}</span>
                      <span className="w-20 font-mono text-[10px] text-faint">{fmtShape(shape)}</span>
                      <div className="min-w-0 flex-1 overflow-hidden">
                        <Heatmap values={g.data} rows={g.rows} cols={g.cols} size={g.rows === 1 ? 420 : 60} className={g.rows === 1 ? '!h-4 !w-full' : ''} />
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {focus !== null && run.trace.outputs[focus] && net.analysis[focus].outShape && (
              <div className="card p-5">
                <h3 className="mb-4 text-[13px] font-semibold">{net.chain[focus].name} <span className="font-normal text-faint">/ {layerDef(net.chain[focus].type).label} {featureLayers.some((f) => f.i === focus) ? 'feature maps' : 'output'}</span></h3>
                <div className="flex flex-wrap gap-3">
                  {activationGrids(run.trace.outputs[focus], net.analysis[focus].outShape, 16).map((g, k) => (
                    <div key={k} className="text-center">
                      <Heatmap values={g.data} rows={g.rows} cols={g.cols} size={g.rows === 1 ? 360 : 96} />
                      {g.label && <div className="mt-1 font-mono text-[10px] text-faint">{g.label}</div>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {attentionLayers.length > 0 && (
              <div className="card p-5">
                <h3 className="mb-1 text-[13px] font-semibold">Attention maps</h3>
                <p className="mb-4 text-[12px] text-faint">Row = position doing the looking, column = position being looked at.</p>
                <div className="space-y-4">
                  {attentionLayers.map(({ t, i }) => {
                    const T = Math.round(Math.sqrt(t!.attention![0].length));
                    const words = input?.note ? input.note.toLowerCase().match(/[a-z']+/g)?.slice(0, T) : undefined;
                    return (
                      <div key={i}>
                        <div className="mb-2 text-[12px] text-muted">{net.chain[i].name}</div>
                        <div className="flex flex-wrap gap-4">
                          {t!.attention!.map((a, h) => (
                            <div key={h} className="text-center">
                              <Heatmap values={a} rows={T} cols={T} size={120} />
                              <div className="mt-1 font-mono text-[10px] text-faint">head {h + 1}</div>
                            </div>
                          ))}
                        </div>
                        {words && <div className="mt-2 font-mono text-[10px] text-faint">tokens: {words.join(' / ')}</div>}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
