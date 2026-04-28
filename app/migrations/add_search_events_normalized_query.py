"""Migration: add bm25_normalized_query column to search_events.

Tracks the query string actually passed to plainto_tsquery after the
BM25 normalizer strips question-lead phrases and noise quantifiers.
NULL when the normalizer made no change (query passed through unchanged).

This column is surfaced in the SearchTracePanel UI so engineers can see
at a glance whether the natural-language question was rewritten before
the BM25 arm ran.
"""
import logging
from sqlalchemy import text
from app.database import get_engine

logger = logging.getLogger(__name__)


async def migrate() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("""
            ALTER TABLE search_events
            ADD COLUMN IF NOT EXISTS bm25_normalized_query TEXT
        """))
        logger.info("search_events: bm25_normalized_query column added")
