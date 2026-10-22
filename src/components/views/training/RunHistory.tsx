import { Factory as History, Loader as Loader2, Trash as Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { num, pct } from '../../../lib/format';
import { deleteRun, listRuns, type RunPoint, type TrainingRunRow } from '../../../store/library';
import { useStore } from '../../../store/useStore';

function Spark({ points }: { points: RunPoint[] }) {
  if (points.length < 2) return <div className="h-8 w-28" />;
  const vals = points.flatMap((p) => [p.trainLoss, p.valLoss]);
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const line = (k: 'trainLoss' | 'valLoss') =>
    points.map((p, i) => `${(i / (points.length - 1)) * 100},${30 - ((p[k] - lo) / span) * 26}`).join(' ');
  return (
    <svg viewBox="0 0 100 32" preserveAspectRatio="none" className="h-8 w-28 shrink-0">
      <polyline points={line('trainLoss')} fill="none" stroke="#4f8ef7" strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
      <polyline points={line('valLoss')} fill="none" stroke="#f5b83d" strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

const when = (iso: string) => new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
const dur = (s: number) => (s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);

export function RunHistory() {
  const networkId = useStore((s) => s.networkId);
  const version = useStore((s) => s.libraryVersion);
  const log = useStore((s) => s.log);
  const [runs, setRuns] = useState<TrainingRunRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => {
    if (!networkId) return;
    let live = true;
    setError(null);
    listRuns(networkId)
      .then((r) => live && setRuns(r))
      .catch((e: Error) => live && setError(e.message));
    return () => { live = false; };
  }, [networkId, version]);

  const remove = async (id: string) => {
    setDeleting(id);
    try {
      await deleteRun(id);
    } catch (e) {
      log('error', (e as Error).message, 'history');
    } finally {
      setDeleting(null);
    }
  };

  return (
    <section className="card p-5">
      <div className="mb-3 flex items-center gap-2">
        <History size={14} className="text-brand-400" />
        <h3 className="text-[13px] font-semibold">Run history</h3>
        {networkId && runs && <span className="text-[11px] text-faint">{runs.length} saved run{runs.length === 1 ? '' : 's'}</span>}
        <span className="ml-auto flex items-center gap-3 text-[10px] text-faint">
          <span className="flex items-center gap-1"><span className="h-0.5 w-3 rounded bg-brand-500" /> train loss</span>
          <span className="flex items-center gap-1"><span className="h-0.5 w-3 rounded bg-warn-500" /> val loss</span>
        </span>
      </div>

      {!networkId ? (
        <p className="text-[12px] text-muted">Save this network to the cloud and every training session will be recorded here so you can compare results over time.</p>
      ) : error ? (
        <p className="rounded-lg bg-bad-500/10 px-3 py-2 text-[12px] text-bad-400">{error}</p>
      ) : !runs ? (
        <div className="space-y-1.5">{[0, 1].map((i) => <div key={i} className="h-12 animate-pulse rounded-lg bg-raised" />)}</div>
      ) : !runs.length ? (
        <p className="text-[12px] text-muted">No runs yet. Finished or stopped training sessions for this network appear here automatically.</p>
      ) : (
        <div className="divide-y divide-line/60">
          {runs.map((r, i) => (
            <div key={r.id} className="group flex items-center gap-4 py-2.5">
              <span className="w-6 shrink-0 text-right font-mono text-[11px] text-faint">#{runs.length - i}</span>
              <div className="w-32 shrink-0">
                <div className="text-[12px] font-medium">{when(r.created_at)}</div>
                <div className="flex items-center gap-1.5 text-[10px] text-faint">
                  <span className={r.engine === 'gneural' ? 'text-ok-400' : 'text-brand-400'}>{r.engine === 'gneural' ? 'Gneural' : 'Workbench'}</span>
                  {r.status === 'stopped' && <span className="text-warn-400">stopped</span>}
                </div>
              </div>
              <Spark points={r.history} />
              <div className="grid flex-1 grid-cols-4 gap-3 font-mono text-[12px]">
                {([
                  ['epochs', String(r.epochs)],
                  ['val loss', num(r.final_val_loss)],
                  ['best acc', pct(r.best_val_acc)],
                  ['time', dur(r.duration_seconds)],
                ] as const).map(([k, v]) => (
                  <div key={k}>
                    <div className="text-[10px] uppercase tracking-wide text-faint">{k}</div>
                    <div className="text-fg">{v}</div>
                  </div>
                ))}
              </div>
              <button className="icon-btn h-7 w-7 shrink-0 opacity-0 transition-opacity hover:text-bad-400 group-hover:opacity-100" onClick={() => remove(r.id)} disabled={deleting === r.id} aria-label="Delete run">
                {deleting === r.id ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />}
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
