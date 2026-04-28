"""Migration: create search_events table for pipeline trace persistence."""
import logging
from sqlalchemy import text
from app.database import get_engine

logger = logging.getLogger(__name__)

async def migrate():
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS search_events (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                search_id       TEXT NOT NULL,
                caller          TEXT NOT NULL DEFAULT 'api',
                query           TEXT NOT NULL,
                mode            TEXT NOT NULL,
                k               INTEGER NOT NULL,
                returned        INTEGER NOT NULL,
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
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_search_events_created_at
            ON search_events (created_at DESC)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_search_events_search_id
            ON search_events (search_id)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_search_events_caller
            ON search_events (caller)
        """))
        logger.info("search_events migration complete")
