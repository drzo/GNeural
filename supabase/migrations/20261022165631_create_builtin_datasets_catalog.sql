/*
# Built-in dataset catalog

Adds a read-only catalog of the seven datasets that ship with the Gneural workbench, so
their descriptions and properties live in the database. The sample values themselves are
generated deterministically in the app (fixed seed), so every user sees identical data.

1. New Tables
- `builtin_datasets`
  - `id` (text, primary key) - stable identifier the app uses, e.g. 'digits'
  - `name` (text) - display name
  - `description` (text) - one-line explanation
  - `kind` (text) - 'image', 'sequence', 'text' or 'tabular'
  - `task` (text) - 'classification', 'regression' or 'reconstruction'
  - `input_shape`, `target_shape` (integer[]) - tensor shapes of one sample
  - `classes` (text[]) - category names, empty for regression/reconstruction
  - `feature_names` (text[]) - column names for tabular data
  - `sample_count` (integer) - number of generated samples
  - `sort_order` (integer) - display order
  - `created_at` (timestamptz)

2. Seed data
- Digits 8x8, Digits (reconstruct), Spirals, XOR, Next number, Waveforms, Sentiment phrases.
  Upserted so re-running keeps the catalog in sync.

3. Security
- RLS enabled. Read-only reference data: anon + authenticated get SELECT only.
  No insert/update/delete grants or policies, so only migrations can change it.
*/

CREATE TABLE IF NOT EXISTS builtin_datasets (
  id text PRIMARY KEY CHECK (char_length(id) BETWEEN 1 AND 50),
  name text NOT NULL,
  description text NOT NULL DEFAULT '',
  kind text NOT NULL CHECK (kind IN ('image', 'sequence', 'text', 'tabular')),
  task text NOT NULL CHECK (task IN ('classification', 'regression', 'reconstruction')),
  input_shape integer[] NOT NULL,
  target_shape integer[] NOT NULL,
  classes text[] NOT NULL DEFAULT '{}',
  feature_names text[] NOT NULL DEFAULT '{}',
  sample_count integer NOT NULL CHECK (sample_count > 0),
  sort_order integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE builtin_datasets ENABLE ROW LEVEL SECURITY;
GRANT SELECT ON builtin_datasets TO anon, authenticated;

DROP POLICY IF EXISTS "public_read_builtin_datasets" ON builtin_datasets;
CREATE POLICY "public_read_builtin_datasets" ON builtin_datasets FOR SELECT
  TO anon, authenticated USING (true);

INSERT INTO builtin_datasets (id, name, description, kind, task, input_shape, target_shape, classes, feature_names, sample_count, sort_order) VALUES
  ('digits', 'Digits 8x8', 'Handwritten-style digits 0-9 rendered at 8x8 pixels with jitter and noise.',
    'image', 'classification', '{8,8,1}', '{10}', '{0,1,2,3,4,5,6,7,8,9}', '{}', 800, 1),
  ('digits-ae', 'Digits (reconstruct)', 'The digits dataset where the target is the image itself, for autoencoders.',
    'image', 'reconstruction', '{8,8,1}', '{8,8,1}', '{}', '{}', 800, 2),
  ('spiral', 'Spirals', 'Three interleaved spiral arms in 2D. A classic test of non-linear decision boundaries.',
    'tabular', 'classification', '{2}', '{3}', '{"Arm A","Arm B","Arm C"}', '{x,y}', 360, 3),
  ('xor', 'XOR', 'Outputs 1 when exactly one input is on. Impossible for a single straight line to separate.',
    'tabular', 'classification', '{2}', '{2}', '{Off,On}', '{a,b}', 240, 4),
  ('next-number', 'Next number', 'Predict the next number in a sequence: [1, 2] -> 3. The Gneural README example, ready for Geometry Dash.',
    'tabular', 'regression', '{2}', '{1}', '{}', '{n,"n + 1"}', 25, 5),
  ('waves', 'Waveforms', 'Short 16-step signals of four shapes with random frequency, phase and noise.',
    'sequence', 'classification', '{16,1}', '{4}', '{Sine,Square,Sawtooth,Triangle}', '{}', 480, 6),
  ('sentiment', 'Sentiment phrases', 'Short review-style phrases labeled positive or negative, encoded as 8 tokens x 16 features.',
    'text', 'classification', '{8,16}', '{2}', '{Negative,Positive}', '{}', 500, 7)
ON CONFLICT (id) DO UPDATE SET
  name = EXCLUDED.name, description = EXCLUDED.description, kind = EXCLUDED.kind, task = EXCLUDED.task,
  input_shape = EXCLUDED.input_shape, target_shape = EXCLUDED.target_shape, classes = EXCLUDED.classes,
  feature_names = EXCLUDED.feature_names, sample_count = EXCLUDED.sample_count, sort_order = EXCLUDED.sort_order;
