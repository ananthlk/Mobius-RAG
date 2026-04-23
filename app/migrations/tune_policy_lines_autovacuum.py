"""Migration: tune autovacuum for policy_lines.

Problem (observed 2026-04-23):
    policy_lines is the hottest table in rag. A single 100-page doc
    re-run churns 70%+ of rows via DELETE+INSERT (clear-and-rewrite
    the lexicon rule matches). The default
    ``autovacuum_vacuum_scale_factor=0.20`` means Postgres waits
    until 20% of the table is dead tuples before vacuuming — on a
    300K-row table that's 60K dead tuples, at which point seq scans
    on the still-live rows are wading through hundreds of thousands
    of tombstones. We saw this cause a doom-loop where each DELETE
    retry got slower as dead-tuple count accumulated, eventually
    hitting the 10-min statement_timeout and restarting the job
    from scratch.

Fix:
    Per-table overrides that make autovacuum kick in much more
    eagerly:
      * scale_factor 0.02 (2%) — trigger at 6K dead tuples instead
        of 60K.
      * analyze_scale_factor 0.02 — keep planner stats fresh as
        row distribution shifts.
      * vacuum_cost_limit 2000 (vs default 200) — vacuum runs 10×
        faster once it starts, so it finishes before the next job
        queues up.

These are the same sorts of knobs Cloud SQL ops tuning recommends
for high-churn tables; no instance-level config change needed.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(
            """
            ALTER TABLE policy_lines SET (
                autovacuum_vacuum_scale_factor = 0.02,
                autovacuum_analyze_scale_factor = 0.02,
                autovacuum_vacuum_cost_limit = 2000
            )
            """
        )
        print("  Set per-table autovacuum knobs on policy_lines")

        # Verify and print current reloptions so ops can confirm
        row = await conn.fetchrow(
            "SELECT reloptions FROM pg_class WHERE relname='policy_lines'"
        )
        print(f"  policy_lines reloptions: {row['reloptions']}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration tune_policy_lines_autovacuum completed.")
