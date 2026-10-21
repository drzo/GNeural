import { FolderOpen, Layers, Loader as Loader2, Trash as Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { compact } from '../../lib/format';
import { deleteNetwork, listNetworks, loadNetwork, type NetworkRow } from '../../store/actions';
import { useStore } from '../../store/useStore';
import { Empty, Modal } from '../ui/primitives';

export function LoadModal() {
  const close = () => useStore.getState().setModal(null);
  const currentId = useStore((s) => s.networkId);
  const dirty = useStore((s) => s.dirty);
  const [rows, setRows] = useState<NetworkRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<string | null>(null);

  useEffect(() => {
    listNetworks().then(setRows, (e: Error) => setError(e.message));
  }, []);

  const open = async (id: string) => {
    setBusy(id);
    await loadNetwork(id);
    setBusy(null);
  };

  const remove = async (id: string) => {
    setBusy(id);
    try {
      await deleteNetwork(id);
      setRows((r) => r?.filter((x) => x.id !== id) ?? null);
    } catch (e) {
      setError((e as Error).message);
    }
    setBusy(null);
    setConfirm(null);
  };

  return (
    <Modal title="Open a saved network" subtitle={dirty ? 'Opening a network replaces the current one. Save first if you want to keep your changes.' : 'Networks are stored with their training settings and trained weights.'} onClose={close} width={640}>
      <div className="p-4">
        {error && <div className="mb-3 rounded-lg bg-bad-500/10 px-3 py-2 text-[13px] text-bad-400">Could not reach saved networks: {error}</div>}
        {!rows && !error && (
          <div className="space-y-2">{[0, 1, 2].map((i) => <div key={i} className="h-16 rounded-xl bg-raised"><div className="shimmer h-full w-full rounded-xl" /></div>)}</div>
        )}
        {rows && !rows.length && <div className="h-64"><Empty icon={<FolderOpen size={22} />} title="No saved networks yet" body="Press Ctrl+S or the save button in the top bar to store your current network." /></div>}
        <div className="space-y-2">
          {rows?.map((r) => (
            <div key={r.id} className={`group flex items-center gap-3 rounded-xl border px-4 py-3 transition-colors ${r.id === currentId ? 'border-brand-500/50 bg-brand-500/5' : 'border-line hover:bg-raised'}`}>
              <button className="min-w-0 flex-1 text-left" onClick={() => open(r.id)} disabled={!!busy}>
                <div className="flex items-center gap-2">
                  <span className="truncate text-[14px] font-medium">{r.name}</span>
                  {r.id === currentId && <span className="chip bg-brand-500/15 text-brand-400">Open</span>}
                </div>
                <div className="mt-0.5 truncate font-mono text-[11px] text-faint">{r.description}</div>
              </button>
              <div className="text-right text-[11px] text-faint">
                <div className="flex items-center justify-end gap-1"><Layers size={11} /> {r.layer_count} layers, {compact(Number(r.param_count))} params</div>
                <div>{new Date(r.updated_at).toLocaleString()}</div>
              </div>
              {busy === r.id ? (
                <Loader2 size={16} className="animate-spin text-faint" />
              ) : confirm === r.id ? (
                <div className="flex gap-1">
                  <button className="btn-danger h-7 px-2 text-[12px]" onClick={() => remove(r.id)}>Delete</button>
                  <button className="btn-ghost h-7 px-2 text-[12px]" onClick={() => setConfirm(null)}>Cancel</button>
                </div>
              ) : (
                <button className="icon-btn opacity-0 group-hover:opacity-100" onClick={() => setConfirm(r.id)} title="Delete"><Trash2 size={14} /></button>
              )}
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}
