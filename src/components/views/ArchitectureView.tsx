import { TriangleAlert as AlertTriangle, ArrowDown, CircleCheck as CheckCircle2, Database, Funnel, GitBranch, Circle as XCircle } from 'lucide-react';
import { layerDef } from '../../engine/catalog';
import { fmtShape, size } from '../../engine/shapes';
import { compact, layerColor } from '../../lib/format';
import { useNetwork } from '../../store/useNetwork';
import { useStore } from '../../store/useStore';
import { LayerBadge } from '../ui/LayerIcon';
import { Empty, Stat } from '../ui/primitives';

export function ArchitectureView() {
  const net = useNetwork();
  const select = useStore((s) => s.select);
  const selectedId = useStore((s) => s.selectedId);
  const setView = useStore((s) => s.setView);
  const expert = useStore((s) => s.mode === 'expert');

  if (!net.chain.length) {
    return <Empty icon={<GitBranch size={22} />} title="No architecture yet" body="Add an Input layer in the Graph view and connect layers to see how data shapes flow through the network." action={<button className="btn-primary" onClick={() => setView('graph')}>Open Graph view</button>} />;
  }

  const maxParams = Math.max(1, ...net.analysis.map((a) => a.params));
  const warnings = net.analysis.reduce((s, a) => s + a.warnings.length, 0);
  const last = net.analysis[net.analysis.length - 1];
  const target = net.dataset?.targetShape;
  const outputMatches = !!(last?.outShape && target && size(last.outShape) === size(target));
  const activationBytes = net.analysis.reduce((s, a) => s + (a.outShape ? size(a.outShape) : 0), 0) * 4;

  return (
    <div className="flex h-full">
      <div className="min-w-0 flex-1 overflow-y-auto px-10 py-8">
        <div className="mx-auto max-w-[560px]">
          {net.chain.map((l, i) => {
            const a = net.analysis[i];
            const role = i === 0 ? 'input' : i === net.chain.length - 1 ? 'output' : 'hidden';
            const color = layerColor(l.type, role);
            const def = layerDef(l.type);
            return (
              <div key={l.id}>
                {i > 0 && (
                  <div className="flex flex-col items-center py-1">
                    <div className={`h-4 w-px ${a.error ? 'bg-bad-500' : 'bg-line'}`} />
                    <div className={`rounded-full border px-2 py-0.5 font-mono text-[10px] ${a.error ? 'border-bad-500/50 bg-bad-500/10 text-bad-400' : 'border-line bg-panel text-faint'}`}>
                      {fmtShape(a.inShape)}
                    </div>
                    <ArrowDown size={14} className={a.error ? 'text-bad-500' : 'text-faint'} />
                  </div>
                )}
                <button
                  onClick={() => select(l.id)}
                  className={`group relative flex w-full items-center gap-4 overflow-hidden rounded-xl border bg-panel py-3 pl-5 pr-4 text-left shadow-node transition-all hover:-translate-y-px ${
                    selectedId === l.id ? 'border-brand-500 ring-2 ring-brand-500/20' : a.error ? 'border-bad-500/60' : 'border-line hover:border-faint/60'
                  }`}
                >
                  <span className="absolute inset-y-0 left-0 w-1" style={{ background: color }} />
                  <LayerBadge type={l.type} color={color} size={34} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-[14px] font-semibold">{l.name}</span>
                      <span className="text-[11px] text-faint">{def.label}</span>
                      {a.bottleneck && <span className="chip bg-warn-500/15 text-warn-400"><Funnel size={10} /> bottleneck</span>}
                    </div>
                    <div className="mt-1 flex items-center gap-2 font-mono text-[12px]">
                      <span className="text-faint">{fmtShape(a.inShape)}</span>
                      <span className="text-faint">to</span>
                      <span className={a.error ? 'text-bad-400' : 'text-fg'}>{fmtShape(a.outShape)}</span>
                      {expert && typeof l.params.activation === 'string' && <span className="text-faint">/ {l.params.activation}</span>}
                    </div>
                    {a.params > 0 && (
                      <div className="mt-2 h-1 overflow-hidden rounded-full bg-raised">
                        <div className="h-full rounded-full transition-all" style={{ width: `${(a.params / maxParams) * 100}%`, background: color }} />
                      </div>
                    )}
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-[14px] font-semibold">{a.params ? compact(a.params) : '-'}</div>
                    <div className="text-[10px] text-faint">params</div>
                  </div>
                </button>
                {(a.error || a.warnings.length > 0) && (
                  <div className="mt-1.5 space-y-1 pl-5">
                    {a.error && <div className="flex items-center gap-1.5 text-[12px] text-bad-400"><XCircle size={12} /> {a.error}</div>}
                    {a.warnings.map((w) => <div key={w} className="flex items-center gap-1.5 text-[12px] text-warn-400"><AlertTriangle size={12} /> {w}</div>)}
                  </div>
                )}
              </div>
            );
          })}
          {net.orphans.length > 0 && (
            <p className="mt-6 rounded-lg border border-dashed border-warn-500/40 px-4 py-3 text-[12px] text-warn-400">
              {net.orphans.length} layer{net.orphans.length > 1 ? 's are' : ' is'} not connected and not shown here.
            </p>
          )}
        </div>
      </div>

      <aside className="w-[280px] shrink-0 space-y-3 overflow-y-auto border-l border-line p-5">
        <Stat label="Total parameters" value={compact(net.totalParams)} sub={`${(net.totalParams * 4 / 1024).toFixed(1)} KB of weights`} tone="brand" />
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Layers" value={net.chain.length} />
          <Stat label="Activations" value={`${(activationBytes / 1024).toFixed(0)}KB`} sub="per sample" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Errors" value={net.errors.length} tone={net.errors.length ? 'bad' : 'ok'} />
          <Stat label="Warnings" value={warnings} tone={warnings ? 'warn' : 'ok'} />
        </div>
        <div className="card p-4">
          <div className="label mb-3 flex items-center gap-1.5"><Database size={11} /> Dataset fit</div>
          {net.dataset ? (
            <div className="space-y-2.5 text-[12px]">
              <div className="flex items-center justify-between"><span className="text-muted">{net.dataset.name}</span></div>
              <Row ok={!net.errors.some((e) => e.id === net.chain[0].id)} label="Input" value={fmtShape(net.dataset.inputShape)} />
              <Row ok={outputMatches} label="Output" value={`${fmtShape(last?.outShape ?? null)} vs ${fmtShape(target ?? null)}`} />
            </div>
          ) : (
            <p className="text-[12px] text-bad-400">{net.datasetError}</p>
          )}
        </div>
        <div className="card p-4">
          <div className="label mb-3">Parameters by layer</div>
          <div className="space-y-1.5">
            {net.chain.map((l, i) => {
              const a = net.analysis[i];
              if (!a.params) return null;
              const share = a.params / Math.max(1, net.totalParams);
              return (
                <div key={l.id} className="text-[11px]">
                  <div className="flex justify-between"><span className="truncate text-muted">{l.name}</span><span className="font-mono text-faint">{(share * 100).toFixed(1)}%</span></div>
                  <div className="mt-0.5 h-1 rounded-full bg-raised"><div className="h-full rounded-full bg-brand-500" style={{ width: `${share * 100}%` }} /></div>
                </div>
              );
            })}
          </div>
        </div>
      </aside>
    </div>
  );
}

function Row({ ok, label, value }: { ok: boolean; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      {ok ? <CheckCircle2 size={13} className="text-ok-400" /> : <XCircle size={13} className="text-bad-400" />}
      <span className="w-12 text-faint">{label}</span>
      <span className="truncate font-mono text-muted">{value}</span>
    </div>
  );
}
