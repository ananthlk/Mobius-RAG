"""Migration: add ``chunking_jobs.failure_count`` + ``failure_reason``.

Background — 2026-04-30 dev-smoke. The chunking pipeline kept getting
stuck in crash-loops on a small set of "bloat-trapped" docs:

  1. Worker claims a doc with leftover policy_lines from prior runs.
  2. ``clear_policy_for_document`` exceeds Postgres ``statement_timeout``
     trying to delete thousands of stale lines, RAISES (per the
     2026-04-30 fix that stopped silent cleanup failures).
  3. Worker dies; supervisor restarts.
  4. Stale-recovery resets the job to ``pending``.
  5. Goto 1.

The supervisor + stale-recovery loop has no memory of "this doc has
already crashed N times". Net effect: 3 of 8 workers gridlocked on
3 bad docs; effective parallelism drops, throughput tanks.

This migration adds:

* ``failure_count`` — incremented every time the job ends in a
  failure path (worker crash, raise-on-cleanup, exception). Reset
  to 0 only on successful completion.
* ``failure_reason`` — short categorical tag so operators can
  triage what kind of failure it is.

The worker code change (separate commit) makes stale-recovery and
the worker error path increment ``failure_count`` and, when
``failure_count >= 3``, set ``status='blocked'`` instead of
resetting to ``pending``. Workers' claim path skips ``blocked``.
A new ``/admin/list_blocked_docs`` surfaces them for human review.

Idempotent: ``ADD COLUMN IF NOT EXISTS``.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg  # noqa: E402

from app.config import DATABASE_URL  # noqa: E402


async def migrate() -> None:
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        await conn.execute("""
            ALTER TABLE chunking_jobs
            ADD COLUMN IF NOT EXISTS failure_count INTEGER NOT NULL DEFAULT 0
        """)
        await conn.execute("""
            ALTER TABLE chunking_jobs
            ADD COLUMN IF NOT EXISTS failure_reason VARCHAR(40)
        """)
        # Partial index to speed up "list blocked docs" queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_chunking_jobs_blocked
            ON chunking_jobs (failure_reason, completed_at DESC)
            WHERE status = 'blocked' OR failure_count > 0
        """)
        print("  Ensured columns: failure_count, failure_reason")
        print("  Ensured index: ix_chunking_jobs_blocked")
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(migrate())
    print("\nMigration add_chunking_failure_tracking completed.")


if __name__ == "__main__":
    main()
