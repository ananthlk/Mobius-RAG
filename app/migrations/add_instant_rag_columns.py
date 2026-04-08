"""Add instant-RAG columns to the documents table.

Supports the instant-rag skill: ephemeral document lifecycle, verification tiers,
agent scoping, and tagging coverage tracking.
"""
from __future__ import annotations

import logging
from sqlalchemy import text
from app.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

_STATEMENTS = [
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS instant_rag_status TEXT",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS verification_tier TEXT DEFAULT 'verified'",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS agent_scope_tags JSONB DEFAULT '[]'",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS retention_policy TEXT DEFAULT 'persist'",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS tagging_coverage JSONB",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS instant_rag_test_results JSONB",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS envelope_id UUID",
    # Indexes
    "CREATE INDEX IF NOT EXISTS idx_documents_instant_rag_status ON documents(instant_rag_status) WHERE instant_rag_status IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_documents_expires_at ON documents(expires_at) WHERE expires_at IS NOT NULL",
]


async def migrate() -> None:
    async with AsyncSessionLocal() as db:
        for stmt in _STATEMENTS:
            try:
                await db.execute(text(stmt))
            except Exception as exc:
                logger.warning("Migration statement skipped: %s — %s", stmt[:60], exc)
        await db.commit()
        logger.info("add_instant_rag_columns migration complete")
