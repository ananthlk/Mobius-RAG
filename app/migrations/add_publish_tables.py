"""
Migration: create rag_published_embeddings and publish_events tables.

- rag_published_embeddings: dbt contract table; one row per published embedding (written on user Publish).
- publish_events: audit log (document_id, published_at, published_by, rows_written).

Usage:
    python -m app.migrations.add_publish_tables
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
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("  pgvector extension enabled")
        except Exception as ext_err:
            print(f"  WARNING: Could not enable pgvector extension: {ext_err}")
            raise

        # 1. Create publish_events table
        exists_pe = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'publish_events'
        """)
        if exists_pe:
            print("  Table publish_events already exists")
        else:
            await conn.execute("""
                CREATE TABLE publish_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    published_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    published_by VARCHAR(255),
                    rows_written INTEGER NOT NULL DEFAULT 0,
                    notes TEXT
                )
            """)
            await conn.execute("""
                CREATE INDEX idx_publish_events_document_id ON publish_events(document_id)
            """)
            await conn.execute("""
                CREATE INDEX idx_publish_events_published_at ON publish_events(published_at DESC)
            """)
            print("  Created table publish_events")

        # 2. Create rag_published_embeddings table (dbt contract schema)
        exists_rpe = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'rag_published_embeddings'
        """)
        if exists_rpe:
            print("  Table rag_published_embeddings already exists")
        else:
            await conn.execute("""
                CREATE TABLE rag_published_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL,
                    source_type VARCHAR(20) NOT NULL,
                    source_id UUID NOT NULL,
                    embedding vector(1536) NOT NULL,
                    model VARCHAR(100),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    text TEXT NOT NULL DEFAULT '',
                    page_number INTEGER NOT NULL DEFAULT 0,
                    paragraph_index INTEGER NOT NULL DEFAULT 0,
                    section_path VARCHAR(500) NOT NULL DEFAULT '',
                    chapter_path VARCHAR(500) NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    document_filename VARCHAR(255) NOT NULL DEFAULT '',
                    document_display_name VARCHAR(255) NOT NULL DEFAULT '',
                    document_authority_level VARCHAR(100) NOT NULL DEFAULT '',
                    document_effective_date VARCHAR(20) NOT NULL DEFAULT '',
                    document_termination_date VARCHAR(20) NOT NULL DEFAULT '',
                    document_payer VARCHAR(100) NOT NULL DEFAULT '',
                    document_state VARCHAR(2) NOT NULL DEFAULT '',
                    document_program VARCHAR(100) NOT NULL DEFAULT '',
                    document_status VARCHAR(20) NOT NULL DEFAULT '',
                    document_created_at TIMESTAMP,
                    document_review_status VARCHAR(20) NOT NULL DEFAULT '',
                    document_reviewed_at TIMESTAMP,
                    document_reviewed_by VARCHAR(255),
                    content_sha VARCHAR(64) NOT NULL DEFAULT '',
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    source_verification_status VARCHAR(20) NOT NULL DEFAULT ''
                )
            """)
            await conn.execute("""
                CREATE INDEX idx_rag_published_embeddings_document_id ON rag_published_embeddings(document_id)
            """)
            await conn.execute("""
                CREATE INDEX idx_rag_published_embeddings_document_source ON rag_published_embeddings(document_id, source_type, source_id)
            """)
            await conn.execute("""
                CREATE INDEX idx_rag_published_embeddings_embedding ON rag_published_embeddings
                USING hnsw (embedding vector_cosine_ops)
            """)
            print("  Created table rag_published_embeddings with vector index")
    finally:
        await conn.close()


def main():
    asyncio.run(migrate())
    print("Migration add_publish_tables completed.")


if __name__ == "__main__":
    main()
