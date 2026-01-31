"""
Migration: create pgvector extension, embedding_jobs and chunk_embeddings tables.

- pgvector extension for vector type
- embedding_jobs: job queue for embedding tasks (same pattern as chunking_jobs)
- chunk_embeddings: stores embeddings with link to source (no text column)

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
        # 1. Enable pgvector extension (requires pgvector installed in PostgreSQL)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("  pgvector extension enabled")
        except Exception as ext_err:
            print(f"  WARNING: Could not enable pgvector extension: {ext_err}")
            print("  Install pgvector: https://github.com/pgvector/pgvector#installation")
            raise

        # 2. Create embedding_jobs table
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

        # 3. Create chunk_embeddings table (no text column; link back via source_id)
        exists_emb = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'chunk_embeddings'
        """)
        if exists_emb:
            print("  Table chunk_embeddings already exists")
        else:
            # Use vector(1536) - common for OpenAI text-embedding-3-small; configurable per deployment
            await conn.execute("""
                CREATE TABLE chunk_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    source_type VARCHAR(20) NOT NULL,
                    source_id UUID NOT NULL,
                    embedding vector(1536) NOT NULL,
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
            # HNSW index for cosine similarity search
            await conn.execute("""
                CREATE INDEX idx_chunk_embeddings_embedding ON chunk_embeddings
                USING hnsw (embedding vector_cosine_ops)
            """)
            print("  Created table chunk_embeddings with vector index")
    finally:
        await conn.close()


def main():
    asyncio.run(migrate())
    print("Migration add_embedding_tables completed.")


if __name__ == "__main__":
    main()
