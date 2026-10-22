/*
# Remove unused UPDATE privilege on datasets and training_runs

1. Security
- Revokes UPDATE from anon and authenticated on `datasets` and `training_runs`.
  Rows in both tables are written once and never edited; there was already no UPDATE
  policy, this removes the unused table privilege as well (least privilege).
*/

REVOKE UPDATE ON datasets FROM anon, authenticated;
REVOKE UPDATE ON training_runs FROM anon, authenticated;
