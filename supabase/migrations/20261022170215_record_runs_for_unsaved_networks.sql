/*
# Record every training run, including runs of unsaved networks

1. Modified Tables
- `training_runs`
  - `network_id` is now optional. Runs of a network that has not been saved yet are
    stored with no network and linked once the network is saved.
  - New `network_name` (text) - name of the network when the run happened
  - New `layer_count` (integer) and `param_count` (bigint) - network size at run time

2. Security
- New column-level UPDATE grant on `network_id` only, plus an UPDATE policy that lets a
  client attach an unlinked run to a network exactly once: the row must currently have
  no network (USING network_id IS NULL) and must end up with one (WITH CHECK NOT NULL).
  All other run columns remain immutable.

3. Notes
1. No data is removed; existing rows keep their network link.
*/

ALTER TABLE training_runs ALTER COLUMN network_id DROP NOT NULL;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'training_runs' AND column_name = 'network_name') THEN
    ALTER TABLE training_runs ADD COLUMN network_name text NOT NULL DEFAULT 'Untitled network' CHECK (char_length(network_name) <= 200);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'training_runs' AND column_name = 'layer_count') THEN
    ALTER TABLE training_runs ADD COLUMN layer_count integer NOT NULL DEFAULT 0 CHECK (layer_count >= 0);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'training_runs' AND column_name = 'param_count') THEN
    ALTER TABLE training_runs ADD COLUMN param_count bigint NOT NULL DEFAULT 0 CHECK (param_count >= 0);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS training_runs_created_idx ON training_runs (created_at DESC);

GRANT UPDATE (network_id) ON training_runs TO anon, authenticated;

DROP POLICY IF EXISTS "shared_link_training_runs" ON training_runs;
CREATE POLICY "shared_link_training_runs" ON training_runs FOR UPDATE
  TO anon, authenticated
  USING (network_id IS NULL)
  WITH CHECK (network_id IS NOT NULL);
