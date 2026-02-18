-- Run as PostgreSQL superuser when connection slots are exhausted.
-- Example: psql -h 127.0.0.1 -U postgres -d mobius_rag -f cleanup_stale_jobs.sql

-- Mark chunking jobs stuck in 'processing' for > 30 min as failed
UPDATE chunking_jobs
SET status = 'failed', worker_id = NULL, completed_at = NOW(),
    error_message = 'Admin cleanup: stuck in processing >30min'
WHERE status = 'processing'
  AND started_at < NOW() - INTERVAL '30 minutes';

-- Mark embedding jobs stuck in 'processing' for > 30 min as failed
UPDATE embedding_jobs
SET status = 'failed', worker_id = NULL, completed_at = NOW(),
    error_message = 'Admin cleanup: stuck in processing >30min'
WHERE status = 'processing'
  AND started_at < NOW() - INTERVAL '30 minutes';
