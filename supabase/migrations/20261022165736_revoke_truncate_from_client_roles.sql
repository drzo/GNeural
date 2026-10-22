/*
# Remove TRUNCATE / TRIGGER / REFERENCES from client roles

1. Security
- Default privileges granted TRUNCATE, TRIGGER and REFERENCES on `networks`, `datasets`
  and `training_runs` to anon and authenticated. TRUNCATE bypasses row level security and
  would let any visitor wipe a whole table, so these unused privileges are revoked.
  The SELECT/INSERT/UPDATE/DELETE grants the app relies on are unchanged.
*/

REVOKE TRUNCATE, TRIGGER, REFERENCES ON networks, datasets, training_runs FROM anon, authenticated;
