/*
# Create networks table

Stores neural networks designed in the Gneural workbench so they can be saved and reopened later.

1. New Tables
- `networks`
  - `id` (uuid, primary key)
  - `name` (text, not null) - display name of the network
  - `description` (text) - optional short description
  - `graph` (jsonb, not null) - layer nodes and connections from the visual editor
  - `config` (jsonb, not null) - training configuration (optimizer, loss, dataset, splits, etc.)
  - `weights` (jsonb, nullable) - trained weights, encoded per layer
  - `layer_count` (integer) - number of layers, for quick display in lists
  - `param_count` (bigint) - total parameter count, for quick display in lists
  - `created_at`, `updated_at` (timestamptz)

2. Security
- RLS enabled.
- This app has no sign-in screen; the workbench is a single shared workspace, so the anon and
  authenticated roles may read, create, update and delete saved networks.

3. Notes
1. An index on `updated_at` supports the "recently saved" list ordering.
*/

CREATE TABLE IF NOT EXISTS networks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL DEFAULT 'Untitled network',
  description text NOT NULL DEFAULT '',
  graph jsonb NOT NULL DEFAULT '{}'::jsonb,
  config jsonb NOT NULL DEFAULT '{}'::jsonb,
  weights jsonb,
  layer_count integer NOT NULL DEFAULT 0,
  param_count bigint NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS networks_updated_at_idx ON networks (updated_at DESC);

ALTER TABLE networks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "shared_select_networks" ON networks;
CREATE POLICY "shared_select_networks" ON networks FOR SELECT
  TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "shared_insert_networks" ON networks;
CREATE POLICY "shared_insert_networks" ON networks FOR INSERT
  TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "shared_update_networks" ON networks;
CREATE POLICY "shared_update_networks" ON networks FOR UPDATE
  TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "shared_delete_networks" ON networks;
CREATE POLICY "shared_delete_networks" ON networks FOR DELETE
  TO anon, authenticated USING (true);
