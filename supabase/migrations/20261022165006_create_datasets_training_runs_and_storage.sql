/*
# Dataset library, training run history and file storage

Adds a shared library of uploaded data files and a history of every training session
for saved networks. The workbench has no sign-in, so like `networks` these are shared
across everyone using the app.

1. New Tables
- `datasets` - metadata for CSV files uploaded to the `datasets` storage bucket
  - `id` (uuid, primary key)
  - `name` (text) - display name, 1-200 characters
  - `storage_path` (text, unique) - object path inside the `datasets` bucket
  - `size_bytes` (integer) - file size, 1 byte to 2 MB
  - `row_count`, `feature_count` (integer) - parsed shape, for display
  - `task` (text) - 'classification' or 'regression'
  - `created_at` (timestamptz)
- `training_runs` - one row per finished or stopped training session
  - `id` (uuid, primary key)
  - `network_id` (uuid, references networks, cascades on delete)
  - `engine` (text) - 'workbench' or 'gneural'
  - `dataset_id` (text) - which dataset was used
  - `epochs` (integer) - epochs completed in this session
  - `final_train_loss`, `final_val_loss` (double precision)
  - `best_val_acc` (double precision, nullable) - only for classification
  - `duration_seconds` (double precision)
  - `status` (text) - 'done' or 'stopped'
  - `history` (jsonb array) - compact per-epoch metrics for charts
  - `config` (jsonb) - training settings used
  - `created_at` (timestamptz)

2. Storage
- Private bucket `datasets`, 2 MB per file, CSV / plain text only.
- Files are read through the API (no public URLs).

3. Security
- RLS enabled on both tables. Shared workspace: anon + authenticated may select, insert
  and delete. No UPDATE grant or policy: rows are immutable once written.
- Storage policies on `datasets` bucket: select, insert, delete for anon + authenticated.
  No update policy, so existing files cannot be overwritten.

4. Notes
1. Index on training_runs (network_id, created_at desc) for the per-network history list.
2. Check constraints bound sizes and enumerated values server-side.
*/

CREATE TABLE IF NOT EXISTS datasets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 200),
  storage_path text NOT NULL UNIQUE CHECK (char_length(storage_path) BETWEEN 1 AND 200),
  size_bytes integer NOT NULL CHECK (size_bytes > 0 AND size_bytes <= 2097152),
  row_count integer NOT NULL DEFAULT 0 CHECK (row_count >= 0),
  feature_count integer NOT NULL DEFAULT 0 CHECK (feature_count >= 0),
  task text NOT NULL CHECK (task IN ('classification', 'regression')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS datasets_created_at_idx ON datasets (created_at DESC);

ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
GRANT SELECT, INSERT, DELETE ON datasets TO anon, authenticated;

DROP POLICY IF EXISTS "shared_select_datasets" ON datasets;
CREATE POLICY "shared_select_datasets" ON datasets FOR SELECT TO anon, authenticated USING (true);
DROP POLICY IF EXISTS "shared_insert_datasets" ON datasets;
CREATE POLICY "shared_insert_datasets" ON datasets FOR INSERT TO anon, authenticated WITH CHECK (true);
DROP POLICY IF EXISTS "shared_delete_datasets" ON datasets;
CREATE POLICY "shared_delete_datasets" ON datasets FOR DELETE TO anon, authenticated USING (true);

CREATE TABLE IF NOT EXISTS training_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  network_id uuid NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
  engine text NOT NULL CHECK (engine IN ('workbench', 'gneural')),
  dataset_id text NOT NULL DEFAULT '' CHECK (char_length(dataset_id) <= 100),
  epochs integer NOT NULL CHECK (epochs > 0 AND epochs <= 100000),
  final_train_loss double precision,
  final_val_loss double precision,
  best_val_acc double precision CHECK (best_val_acc IS NULL OR (best_val_acc >= 0 AND best_val_acc <= 1)),
  duration_seconds double precision NOT NULL DEFAULT 0 CHECK (duration_seconds >= 0),
  status text NOT NULL CHECK (status IN ('done', 'stopped')),
  history jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(history) = 'array'),
  config jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS training_runs_network_created_idx ON training_runs (network_id, created_at DESC);

ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;
GRANT SELECT, INSERT, DELETE ON training_runs TO anon, authenticated;

DROP POLICY IF EXISTS "shared_select_training_runs" ON training_runs;
CREATE POLICY "shared_select_training_runs" ON training_runs FOR SELECT TO anon, authenticated USING (true);
DROP POLICY IF EXISTS "shared_insert_training_runs" ON training_runs;
CREATE POLICY "shared_insert_training_runs" ON training_runs FOR INSERT TO anon, authenticated WITH CHECK (true);
DROP POLICY IF EXISTS "shared_delete_training_runs" ON training_runs;
CREATE POLICY "shared_delete_training_runs" ON training_runs FOR DELETE TO anon, authenticated USING (true);

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES ('datasets', 'datasets', false, 2097152, ARRAY['text/csv', 'text/plain'])
ON CONFLICT (id) DO UPDATE SET public = false, file_size_limit = 2097152, allowed_mime_types = ARRAY['text/csv', 'text/plain'];

DROP POLICY IF EXISTS "shared_select_dataset_files" ON storage.objects;
CREATE POLICY "shared_select_dataset_files" ON storage.objects FOR SELECT TO anon, authenticated
  USING (bucket_id = 'datasets');
DROP POLICY IF EXISTS "shared_insert_dataset_files" ON storage.objects;
CREATE POLICY "shared_insert_dataset_files" ON storage.objects FOR INSERT TO anon, authenticated
  WITH CHECK (bucket_id = 'datasets' AND name ~ '^[0-9a-f-]{36}\.csv$');
DROP POLICY IF EXISTS "shared_delete_dataset_files" ON storage.objects;
CREATE POLICY "shared_delete_dataset_files" ON storage.objects FOR DELETE TO anon, authenticated
  USING (bucket_id = 'datasets');
