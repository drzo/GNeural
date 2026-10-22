import { Cloud, FileSpreadsheet, FileUp, Loader as Loader2, Trash as Trash2 } from 'lucide-react';
import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { parseCsv } from '../../../data/datasets';
import { compact } from '../../../lib/format';
import { deleteDatasetFile, downloadDatasetFile, listDatasetFiles, MAX_CSV_BYTES, uploadDatasetFile, type DatasetFile } from '../../../store/library';
import { useStore } from '../../../store/useStore';

const kb = (n: number) => (n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`);

export function FileLibrary({ onChosen }: { onChosen(): void }) {
  const config = useStore((s) => s.config);
  const setConfig = useStore((s) => s.setConfig);
  const log = useStore((s) => s.log);
  const version = useStore((s) => s.libraryVersion);
  const running = useStore((s) => s.training.status !== 'idle');
  const fileRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<DatasetFile[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    listDatasetFiles()
      .then((f) => live && (setFiles(f), setLoadError(null)))
      .catch((e: Error) => live && setLoadError(e.message));
    return () => { live = false; };
  }, [version]);

  const use = (text: string, name: string) => {
    setConfig({ datasetId: 'csv', csv: { text, name } });
    onChosen();
  };

  const upload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    if (running) return log('warn', 'Stop training before switching datasets', 'dataset');
    if (file.size > MAX_CSV_BYTES) return log('error', 'CSV files are limited to 2 MB', 'dataset');
    let text: string;
    try {
      text = await file.text();
      const parsed = parseCsv(text, file.name);
      use(text, file.name);
      log('success', `Loaded ${file.name}: ${parsed.samples.length} rows, ${parsed.inputShape[0]} features, ${parsed.task}`, 'dataset');
    } catch (err) {
      return log('error', `Could not read ${file.name}: ${(err as Error).message}`, 'dataset');
    }
    setBusy('upload');
    try {
      await uploadDatasetFile(file);
      log('success', `Saved ${file.name} to your file library`, 'storage');
    } catch (err) {
      log('warn', `${(err as Error).message} The file is still in use on this device.`, 'storage');
    } finally {
      setBusy(null);
    }
  };

  const open = async (f: DatasetFile) => {
    if (running) return log('warn', 'Stop training before switching datasets', 'dataset');
    setBusy(f.id);
    try {
      const text = await downloadDatasetFile(f);
      parseCsv(text, f.name);
      use(text, f.name);
      log('success', `Opened ${f.name} from your file library`, 'dataset');
    } catch (err) {
      log('error', (err as Error).message, 'dataset');
    } finally {
      setBusy(null);
    }
  };

  const remove = async (f: DatasetFile) => {
    if (!window.confirm(`Delete "${f.name}" from the file library? Networks already using it keep their copy.`)) return;
    setBusy(f.id);
    try {
      await deleteDatasetFile(f);
      log('info', `Deleted ${f.name}`, 'storage');
    } catch (err) {
      log('error', (err as Error).message, 'storage');
      setBusy(null);
    }
  };

  const activeName = config.datasetId === 'csv' ? config.csv?.name : null;

  return (
    <div>
      <div className="label mb-2 flex items-center justify-between">
        <span>Your data</span>
        <span className="flex items-center gap-1 normal-case tracking-normal text-faint"><Cloud size={11} /> saved in the cloud</span>
      </div>
      <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden" onChange={upload} />
      <button
        onClick={() => fileRef.current?.click()}
        disabled={busy === 'upload'}
        className="flex w-full flex-col items-center gap-1.5 rounded-xl border border-dashed border-line px-3 py-5 text-center transition-colors hover:border-brand-500/60 hover:bg-raised disabled:opacity-60"
      >
        {busy === 'upload' ? <Loader2 size={18} className="animate-spin text-brand-400" /> : <FileUp size={18} className="text-muted" />}
        <span className="text-[12px] font-medium">{busy === 'upload' ? 'Uploading...' : 'Upload a CSV file'}</span>
        <span className="text-[11px] text-faint">Numeric columns, last column is the target, up to 2 MB</span>
      </button>

      {config.csv && config.datasetId !== 'csv' && (
        <button className="btn-ghost mt-1.5 h-7 w-full text-[12px]" onClick={() => !running && use(config.csv!.text, config.csv!.name)}>Use {config.csv.name} again</button>
      )}

      <div className="mt-2.5 space-y-1">
        {loadError && <p className="rounded-lg bg-bad-500/10 px-3 py-2 text-[11px] text-bad-400">{loadError}</p>}
        {!files && !loadError && [0, 1].map((i) => <div key={i} className="h-11 animate-pulse rounded-lg bg-raised" />)}
        {files?.length === 0 && <p className="px-1 text-[11px] text-faint">Files you upload appear here so you can reuse them in any network.</p>}
        {files?.map((f) => {
          const active = activeName === f.name;
          return (
            <div key={f.id} className={`group flex items-center gap-2.5 rounded-lg border px-2.5 py-2 transition-colors ${active ? 'border-brand-500/60 bg-brand-500/10' : 'border-transparent hover:bg-raised'}`}>
              <button className="flex min-w-0 flex-1 items-center gap-2.5 text-left" onClick={() => open(f)} disabled={!!busy}>
                {busy === f.id ? <Loader2 size={15} className="shrink-0 animate-spin text-brand-400" /> : <FileSpreadsheet size={15} className={`shrink-0 ${active ? 'text-brand-400' : 'text-muted'}`} />}
                <span className="min-w-0">
                  <span className="block truncate text-[12px] font-medium">{f.name}</span>
                  <span className="block text-[10px] text-faint">{compact(f.row_count)} rows / {f.feature_count} features / {f.task} / {kb(f.size_bytes)}</span>
                </span>
              </button>
              <button className="icon-btn h-7 w-7 shrink-0 opacity-0 transition-opacity hover:text-bad-400 group-hover:opacity-100" onClick={() => remove(f)} disabled={!!busy} aria-label={`Delete ${f.name}`}>
                <Trash2 size={13} />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
