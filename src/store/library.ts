import { BUILTIN_DATASETS, parseCsv, type DataKind, type DatasetInfo } from '../data/datasets';
import { requireDb } from '../lib/supabase';
import { useStore } from './useStore';

const BUCKET = 'datasets';
export const MAX_CSV_BYTES = 2 * 1024 * 1024;
const get = useStore.getState;

export interface DatasetFile {
  id: string;
  name: string;
  storage_path: string;
  size_bytes: number;
  row_count: number;
  feature_count: number;
  task: 'classification' | 'regression';
  created_at: string;
}

export interface RunPoint {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  valAcc: number | null;
}

export interface TrainingRunRow {
  id: string;
  engine: 'workbench' | 'gneural';
  dataset_id: string;
  epochs: number;
  final_train_loss: number | null;
  final_val_loss: number | null;
  best_val_acc: number | null;
  duration_seconds: number;
  status: 'done' | 'stopped';
  history: RunPoint[];
  created_at: string;
}

function fail(context: string, cause: unknown, message: string): never {
  console.error(context, cause);
  throw new Error(message);
}

export interface CatalogDataset extends DatasetInfo {
  sampleCount: number | null;
}

/** Built-in datasets as described in the database, limited to ones this app can generate. */
export async function listBuiltinDatasets(): Promise<CatalogDataset[]> {
  const { data, error } = await requireDb()
    .from('builtin_datasets')
    .select('id, name, description, kind, sample_count')
    .order('sort_order');
  if (error) fail('list builtin datasets failed', error, 'Could not load the dataset catalog.');
  const known = new Set(BUILTIN_DATASETS.map((d) => d.id));
  return (data ?? [])
    .filter((r) => known.has(r.id))
    .map((r) => ({ id: r.id, name: r.name, description: r.description, kind: r.kind as DataKind, sampleCount: r.sample_count }));
}

export async function listDatasetFiles(): Promise<DatasetFile[]> {
  const { data, error } = await requireDb()
    .from('datasets')
    .select('id, name, storage_path, size_bytes, row_count, feature_count, task, created_at')
    .order('created_at', { ascending: false })
    .limit(100);
  if (error) fail('list datasets failed', error, 'Could not load your saved files.');
  return (data ?? []) as DatasetFile[];
}

/** Validates the CSV locally, stores it in the bucket and records it in the library. */
export async function uploadDatasetFile(file: File): Promise<{ row: DatasetFile; text: string }> {
  if (file.size > MAX_CSV_BYTES) throw new Error('CSV files are limited to 2 MB.');
  if (!file.size) throw new Error('The file is empty.');
  const text = await file.text();
  const parsed = parseCsv(text, file.name);
  const db = requireDb();
  const path = `${crypto.randomUUID()}.csv`;
  const { error: upErr } = await db.storage.from(BUCKET).upload(path, new Blob([text], { type: 'text/csv' }), { contentType: 'text/csv', upsert: false });
  if (upErr) fail('dataset upload failed', upErr, 'Could not upload the file. Please try again.');
  const { data, error } = await db
    .from('datasets')
    .insert({
      name: file.name.slice(0, 200),
      storage_path: path,
      size_bytes: file.size,
      row_count: parsed.samples.length,
      feature_count: parsed.inputShape[0],
      task: parsed.task === 'classification' ? 'classification' : 'regression',
    })
    .select('id, name, storage_path, size_bytes, row_count, feature_count, task, created_at')
    .maybeSingle();
  if (error || !data) {
    await db.storage.from(BUCKET).remove([path]);
    fail('dataset insert failed', error, 'Could not save the file to your library.');
  }
  get().bumpLibrary();
  return { row: data as DatasetFile, text };
}

export async function downloadDatasetFile(f: DatasetFile): Promise<string> {
  const { data, error } = await requireDb().storage.from(BUCKET).download(f.storage_path);
  if (error || !data) fail('dataset download failed', error, `Could not open ${f.name}.`);
  return data.text();
}

export async function deleteDatasetFile(f: DatasetFile) {
  const db = requireDb();
  const { error } = await db.from('datasets').delete().eq('id', f.id);
  if (error) fail('dataset delete failed', error, `Could not delete ${f.name}.`);
  const { error: rmErr } = await db.storage.from(BUCKET).remove([f.storage_path]);
  if (rmErr) console.error('dataset file remove failed', rmErr);
  get().bumpLibrary();
}

export async function listRuns(networkId: string): Promise<TrainingRunRow[]> {
  const { data, error } = await requireDb()
    .from('training_runs')
    .select('id, engine, dataset_id, epochs, final_train_loss, final_val_loss, best_val_acc, duration_seconds, status, history, created_at')
    .eq('network_id', networkId)
    .order('created_at', { ascending: false })
    .limit(50);
  if (error) fail('list runs failed', error, 'Could not load the training history.');
  return ((data ?? []) as TrainingRunRow[]).map((r) => ({ ...r, history: Array.isArray(r.history) ? r.history : [] }));
}

export async function deleteRun(id: string) {
  const { error } = await requireDb().from('training_runs').delete().eq('id', id);
  if (error) fail('delete run failed', error, 'Could not delete that run.');
  get().bumpLibrary();
}

const finite = (v: number | null | undefined) => (v !== null && v !== undefined && Number.isFinite(v) ? v : null);

/** Saves the epochs produced since `fromIndex` as one run of the current saved network. */
export async function recordRun(fromIndex: number, status: 'done' | 'stopped', startedAt: number) {
  const s = get();
  const epochs = s.training.history.slice(fromIndex);
  if (!epochs.length) return;
  if (!s.networkId) {
    s.log('info', 'Save this network to keep a history of its training runs', 'history');
    return;
  }
  const last = epochs[epochs.length - 1];
  const accs = epochs.map((e) => e.valAcc).filter((v): v is number => v !== null && Number.isFinite(v));
  const step = Math.max(1, Math.ceil(epochs.length / 200));
  const history: RunPoint[] = epochs
    .filter((_, i) => i % step === 0 || i === epochs.length - 1)
    .map((e) => ({ epoch: e.epoch, trainLoss: finite(e.trainLoss) ?? 0, valLoss: finite(e.valLoss) ?? 0, valAcc: finite(e.valAcc) }));
  try {
    const { error } = await requireDb().from('training_runs').insert({
      network_id: s.networkId,
      engine: s.training.engine,
      dataset_id: s.config.datasetId,
      epochs: epochs.length,
      final_train_loss: finite(last.trainLoss),
      final_val_loss: finite(last.valLoss),
      best_val_acc: accs.length ? Math.max(...accs) : null,
      duration_seconds: Math.max(0, (performance.now() - startedAt) / 1000),
      status,
      history,
      config: s.config.train,
    });
    if (error) throw error;
    get().log('success', `Recorded training run (${epochs.length} epoch${epochs.length === 1 ? '' : 's'}) in this network's history`, 'history');
    get().bumpLibrary();
  } catch (e) {
    console.error('record run failed', e);
    get().log('warn', 'Could not save this training run to the history', 'history');
  }
}
