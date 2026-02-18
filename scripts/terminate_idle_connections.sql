-- Run as PostgreSQL superuser when connection slots are exhausted.
-- Terminates idle connections (not the one running this script).
-- Example: psql -h 127.0.0.1 -U postgres -d postgres -f terminate_idle_connections.sql

SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname IN ('mobius_rag', 'mobius_qa')
  AND pid != pg_backend_pid()
  AND state = 'idle'
  AND state_change < NOW() - INTERVAL '5 minutes';
