/*
# Make builtin_datasets strictly read-only for clients

1. Security
- Default privileges gave anon/authenticated full table rights on `builtin_datasets`.
  TRUNCATE is not governed by RLS, so revoke every privilege except SELECT.
*/

REVOKE ALL ON builtin_datasets FROM anon, authenticated;
GRANT SELECT ON builtin_datasets TO anon, authenticated;
