"""Migration: create ``source_fetch_attempts`` (Crawler — per-attempt audit trail).

DDL authored by the **DB seat** (Platform Architects) in
``docs/rag-agents/crawler-signoff/02-db-seat.md`` — Q2 ruled ACCEPTED
2026-08-12. Built under Ananth's build-now / retro-signoff instruction the
same day.

WHY THIS TABLE EXISTS
    ``discovered_sources`` keeps only the *latest* fetch state
    (last_fetch_status, last_fetch_at, fetch_attempt_count). There is no
    per-attempt history, so robots/rate-limit behavior and drift causation
    cannot be audited after the fact. That is load-bearing for the known
    403-read-as-disallow_all class of bug: without history you can only see
    the most recent verdict, which is the very thing such a bug corrupts.

    Append-only. One row per fetch attempt. Never updated.

⚠️ ONE DIVERGENCE FROM THE SIGNED DDL — flagged back to the DB seat, not
   decided unilaterally:
   The verdict's prose specifies "monthly partitions + automated drop for 90d
   retention", but the DDL as written declares ``PRIMARY KEY (attempt_id)``
   alone. Postgres requires every partition key column to be part of the
   primary key, so ``PARTITION BY RANGE (attempted_at)`` is rejected against
   that PK — the two halves of the ruling cannot both be satisfied as written.
   Resolving it needs a DB-seat call between:
     (a) composite PK ``(attempt_id, attempted_at)`` → partitioning works;
     (b) unpartitioned + scheduled ``DELETE`` on attempted_at → PK unchanged.
   This migration implements (b): plain table, plus an ``attempted_at`` index
   so retention pruning is cheap and (a) stays available later. Retention
   itself is NOT automated here — no cron is created until the seat rules.

Idempotent — safe to re-run.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.source_fetch_attempts (
    attempt_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discovered_source_id    UUID NOT NULL
                            REFERENCES public.discovered_sources(id) ON DELETE CASCADE,

    attempted_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- NULL http_status = network-level failure (timeout, conn refused, DNS).
    -- Distinguishing "no response" from "a 4xx response" is the whole point
    -- of the audit trail; do not collapse them to a sentinel.
    http_status             INTEGER,
    bytes_downloaded        INTEGER,
    latency_ms              INTEGER,

    -- Open vocabulary per the DB seat's ruling:
    --   crawlable | disallow_all | unknown | error
    -- Deliberately TEXT, not an enum: the robots gate is tri-state and still
    -- evolving; a CHECK constraint here would need a migration per new state.
    robots_decision         TEXT,

    content_hash_before     TEXT,   -- NULL on first fetch
    content_hash_after      TEXT,

    error_message           TEXT,   -- network/parse detail when http_status IS NULL
    run_id                  TEXT    -- batch correlation for one crawl/freshness pass
);
"""

_CREATE_INDEXES_SQL = [
    # Per the signed DDL.
    "CREATE INDEX IF NOT EXISTS idx_fetch_attempts_source "
    "ON public.source_fetch_attempts (discovered_source_id);",
    "CREATE INDEX IF NOT EXISTS idx_fetch_attempts_robots_decision "
    "ON public.source_fetch_attempts (robots_decision);",
    # Added for retention pruning under option (b) above — a 90d DELETE
    # without this index seq-scans the whole audit trail.
    "CREATE INDEX IF NOT EXISTS idx_fetch_attempts_attempted_at "
    "ON public.source_fetch_attempts (attempted_at);",
    # Per-URL history in time order is THE audit query ("show me every
    # attempt on this URL"); the single-column source index alone makes it
    # a sort every time.
    "CREATE INDEX IF NOT EXISTS idx_fetch_attempts_source_time "
    "ON public.source_fetch_attempts (discovered_source_id, attempted_at DESC);",
]


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'source_fetch_attempts'
        """)
        if exists:
            print("  Table public.source_fetch_attempts already exists")
        else:
            await conn.execute(_CREATE_TABLE_SQL)
            print("  Created table public.source_fetch_attempts")

        for sql in _CREATE_INDEXES_SQL:
            await conn.execute(sql)
        print(f"  Ensured {len(_CREATE_INDEXES_SQL)} indexes on source_fetch_attempts")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_source_fetch_attempts completed.")
