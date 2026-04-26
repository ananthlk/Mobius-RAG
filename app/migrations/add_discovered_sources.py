"""Migration: create ``discovered_sources`` table (Phase 13.2 — curator).

Rationale and full schema reference: scripts/curator/SCHEMA.md.

Idempotent — safe to re-run. Detects table existence and column
existence per the existing migration pattern (see e.g.
``add_document_authority_level.py``).
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.discovered_sources (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity
    url                         TEXT NOT NULL UNIQUE,
    host                        VARCHAR(255) NOT NULL,
    path                        TEXT NOT NULL,

    -- Classification
    payer                       VARCHAR(100),
    state                       VARCHAR(2),
    program                     VARCHAR(100),
    inferred_authority_level    VARCHAR(100),
    curated_authority_level     VARCHAR(100),
    topic_tags                  JSONB,
    content_kind                VARCHAR(10) NOT NULL,
    extension                   VARCHAR(20),

    -- Liveness
    first_seen_at               TIMESTAMP NOT NULL DEFAULT now(),
    last_seen_at                TIMESTAMP NOT NULL DEFAULT now(),
    last_fetch_status           INTEGER,
    last_fetch_at               TIMESTAMP,
    fetch_attempt_count         INTEGER NOT NULL DEFAULT 0,

    -- Content fingerprinting
    content_type                VARCHAR(255),
    content_length              BIGINT,
    content_hash                VARCHAR(64),
    content_changed_at          TIMESTAMP,

    -- Ingestion linkage
    ingested                    BOOLEAN NOT NULL DEFAULT false,
    ingested_doc_id             UUID REFERENCES public.documents(id) ON DELETE SET NULL,
    ingested_at                 TIMESTAMP,

    -- Discovery provenance
    discovered_via              VARCHAR(20),
    seed_url                    TEXT,
    depth_from_seed             INTEGER,
    scrape_job_id               VARCHAR(64),

    -- Curation
    curation_status             VARCHAR(20) NOT NULL DEFAULT 'auto',
    curated_by                  VARCHAR(255),
    curation_notes              TEXT,
    curated_at                  TIMESTAMP,

    created_at                  TIMESTAMP NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMP NOT NULL DEFAULT now()
);
"""

# Indexes — created separately so re-running on a partial install
# still finishes the work.
_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ds_host          ON public.discovered_sources (host);",
    "CREATE INDEX IF NOT EXISTS idx_ds_payer_state   ON public.discovered_sources (payer, state);",
    "CREATE INDEX IF NOT EXISTS idx_ds_curation      ON public.discovered_sources (curation_status);",
    "CREATE INDEX IF NOT EXISTS idx_ds_ingested      ON public.discovered_sources (ingested);",
    "CREATE INDEX IF NOT EXISTS idx_ds_scrape_job_id ON public.discovered_sources (scrape_job_id);",
    # GIN index for tag array search (used by ReAct topic lookup).
    "CREATE INDEX IF NOT EXISTS idx_ds_topic_tags_gin ON public.discovered_sources USING gin (topic_tags);",
]


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        # Detect existing table to make the print line meaningful.
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'discovered_sources'
        """)
        if exists:
            print("  Table public.discovered_sources already exists")
        else:
            await conn.execute(_CREATE_TABLE_SQL)
            print("  Created table public.discovered_sources")

        for sql in _CREATE_INDEXES_SQL:
            await conn.execute(sql)
        print(f"  Ensured {len(_CREATE_INDEXES_SQL)} indexes on discovered_sources")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_discovered_sources completed.")
