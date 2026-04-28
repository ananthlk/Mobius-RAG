"""Migration: extend search_events with Phase B lexicon-aware columns.

Phase A shipped lexicon expansion in BM25.  Phase B persists the lexicon
match + expansion bag + final tsquery + per-arm hit counts on every
``/api/skills/v1/corpus_search`` call so the lexicon-management workflow
has a feeder.  The most important downstream consumer is the
``/admin/search_events?unmatched=true`` endpoint, which surfaces queries
whose lexicon match was empty — i.e. the highest-leverage candidates for
new lexicon entries.

The original ``add_search_events`` migration created the table with the
old (pre-lexicon) shape.  This migration is purely additive: every
column / index uses ``IF NOT EXISTS`` so it is safe to re-run.

New columns
-----------
  raw_query           — alias for ``query`` populated alongside it
                         (lexicon UI prefers this name)
  normalized_query    — same as bm25_normalized_query; new name for symmetry
  matched_codes       — text[] of lexicon codes that fired (e.g.
                         ``{"d:benefits.dme", "p:utilization_management.prior_authorization"}``)
  expansion_phrases   — text[] of phrases OR-joined into the tsquery
  final_tsquery       — the actual ``to_tsquery`` string executed
  bm25_hits           — raw BM25-arm result count (pre-fusion)
  vector_hits         — raw vector-arm result count (pre-fusion)
  total_chunks        — final chunks returned (post-rerank, post-assemble)
  filters             — request.filters as jsonb (payer/state/program/auth)
  domain_tags         — text[] of d:* matched_codes
  jurisdiction_tags   — text[] of j:* matched_codes
  process_tags        — text[] of p:* matched_codes
  caller_id           — optional caller-supplied request id

Plus a partial index on (created_at DESC) WHERE cardinality(matched_codes)=0
so the unmatched-query feed scans only the lexicon-coaching candidates.

Operator deploy (do NOT run from local — runs in prod via Cloud Run Job):

    gcloud run jobs execute mobius-rag-migrate \\
        --region us-central1 \\
        --args=python,-m,app.migrations.add_search_events_table

Idempotent: every statement guards with IF NOT EXISTS / IF EXISTS.
"""
from __future__ import annotations

import logging

from sqlalchemy import text

from app.database import get_engine

logger = logging.getLogger(__name__)


async def migrate() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        # Ensure base table exists (no-op when add_search_events already ran)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS search_events (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                search_id       TEXT,
                caller          TEXT NOT NULL DEFAULT 'api',
                query           TEXT NOT NULL DEFAULT '',
                mode            TEXT NOT NULL DEFAULT 'corpus',
                k               INTEGER NOT NULL DEFAULT 10,
                returned        INTEGER NOT NULL DEFAULT 0,
                total_ms        FLOAT,
                embed_ms        FLOAT,
                bm25_ms         FLOAT,
                vec_ms          FLOAT,
                rerank_ms       FLOAT,
                arm_hits        JSONB,
                arm_results     JSONB,
                scoring_trace   JSONB,
                assembly        JSONB,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))

        # ── Phase B additive columns ─────────────────────────────────────
        await conn.execute(text("""
            ALTER TABLE search_events
                ADD COLUMN IF NOT EXISTS raw_query          TEXT,
                ADD COLUMN IF NOT EXISTS normalized_query   TEXT,
                ADD COLUMN IF NOT EXISTS matched_codes      TEXT[] NOT NULL DEFAULT '{}',
                ADD COLUMN IF NOT EXISTS expansion_phrases  TEXT[] NOT NULL DEFAULT '{}',
                ADD COLUMN IF NOT EXISTS final_tsquery      TEXT,
                ADD COLUMN IF NOT EXISTS bm25_hits          INTEGER,
                ADD COLUMN IF NOT EXISTS vector_hits        INTEGER,
                ADD COLUMN IF NOT EXISTS total_chunks       INTEGER,
                ADD COLUMN IF NOT EXISTS filters            JSONB,
                ADD COLUMN IF NOT EXISTS domain_tags        TEXT[] NOT NULL DEFAULT '{}',
                ADD COLUMN IF NOT EXISTS jurisdiction_tags  TEXT[] NOT NULL DEFAULT '{}',
                ADD COLUMN IF NOT EXISTS process_tags       TEXT[] NOT NULL DEFAULT '{}',
                ADD COLUMN IF NOT EXISTS caller_id          TEXT
        """))

        # ── Indexes ──────────────────────────────────────────────────────
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS search_events_created_idx
              ON search_events (created_at DESC)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS search_events_caller_created_idx
              ON search_events (caller, created_at DESC)
        """))
        # Partial index for the lexicon-coaching feed: ?unmatched=true
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS search_events_unmatched_idx
              ON search_events (created_at DESC)
              WHERE cardinality(matched_codes) = 0
        """))

        logger.info("search_events Phase B columns + indexes ensured")


def main() -> None:
    import asyncio
    asyncio.run(migrate())
    print("Migration add_search_events_table completed.")


if __name__ == "__main__":
    main()
