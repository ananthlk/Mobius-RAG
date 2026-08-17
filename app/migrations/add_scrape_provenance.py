#!/usr/bin/env python3
"""Migration: caller-agnostic scrape_provenance table.

Every web fetch the scraper performs — for ANY caller (chat web_scrape,
mobius-rag strategy_d / filler_d, appeals, crawl jobs) — records one row
here: which URL was touched, the robots decision, the fetch outcome, and
who asked. The scraper is the single egress chokepoint, so persisting
here makes web-fetch provenance universal instead of per-caller (chat's
Diagnostics tab, RAG's own view, and a fleet-wide robots audit all read
the same store). Ananth, 2026-08-17: "traced irrespective of who called."

Operational/append-only audit table — no FK into the governed corpus
data model. Indexed by correlation_id (a surface reads its own turn's
provenance) and by created_at (audits / robots-block sweeps).
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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS public.scrape_provenance (
                id              BIGSERIAL PRIMARY KEY,
                caller_id       TEXT,
                correlation_id  TEXT,
                seed_url        TEXT,
                url             TEXT NOT NULL,
                final_url       TEXT,
                robots_decision TEXT,
                fetch_status    TEXT,
                content_type    TEXT,
                mode            TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scrape_provenance_correlation "
            "ON public.scrape_provenance(correlation_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scrape_provenance_created "
            "ON public.scrape_provenance(created_at DESC)"
        )
        print("✓ scrape_provenance table + indexes ready")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
