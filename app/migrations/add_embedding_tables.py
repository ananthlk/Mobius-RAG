"""
Migration: create embedding_jobs and chunk_embeddings tables.

NOTE: Embeddings are stored in Vertex AI Vector Search, not PostgreSQL.
The embedding column is JSONB for schema compatibility but not used for search.

- embedding_jobs: job queue for embedding tasks (same pattern as chunking_jobs)
- chunk_embeddings: stores embedding metadata (actual vectors in Vertex AI)

Usage:
    python -m app.migrations.add_embedding_tables
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
        # Note: pgvector extension removed - embeddings are in Vertex AI Vector Search

        # 1. Create embedding_jobs table
        exists_jobs = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'embedding_jobs'
        """)
        if exists_jobs:
            print("  Table embedding_jobs already exists")
        else:
            await conn.execute("""
                CREATE TABLE embedding_jobs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    worker_id VARCHAR(100),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    embedding_config_version VARCHAR(100),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE INDEX idx_embedding_jobs_status ON embedding_jobs(status)
            """)
            await conn.execute("""
                CREATE INDEX idx_embedding_jobs_document_id ON embedding_jobs(document_id)
            """)
            print("  Created table embedding_jobs")

        # 2. Create chunk_embeddings table (embeddings stored in Vertex AI, not here)
        exists_emb = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'chunk_embeddings'
        """)
        if exists_emb:
            print("  Table chunk_embeddings already exists")
        else:
            # Note: embedding column is JSONB for compatibility; actual vectors in Vertex AI
            await conn.execute("""
                CREATE TABLE chunk_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    source_type VARCHAR(20) NOT NULL,
                    source_id UUID NOT NULL,
                    embedding JSONB,
                    model VARCHAR(100),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE INDEX idx_chunk_embeddings_document_id ON chunk_embeddings(document_id)
            """)
            await conn.execute("""
                CREATE INDEX idx_chunk_embeddings_document_source ON chunk_embeddings(document_id, source_type)
            """)
            print("  Created table chunk_embeddings")
    finally:
        await conn.close()


def main():
    asyncio.run(migrate())
    print("Migration add_embedding_tables completed.")


if __name__ == "__main__":
    main()
