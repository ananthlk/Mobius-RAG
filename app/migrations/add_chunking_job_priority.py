"""
Migration: add priority (INTEGER, default 10) to chunking_jobs.

Priority 0 = urgent (chat/instant-RAG user uploads — user is waiting).
Priority 10 = normal (background corpus ingestion).

Worker claim query orders by priority ASC, created_at ASC so chat
documents jump the queue ahead of the large corpus backlog.

An index on (status, priority, created_at) is added so the claim
query stays fast even with tens of thousands of pending jobs.
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
        col_exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunking_jobs' AND column_name = 'priority'"
        )
        if not col_exists:
            await conn.execute(
                "ALTER TABLE chunking_jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 10"
            )
            print("  Added column chunking_jobs.priority (default 10)")
        else:
            print("  Column chunking_jobs.priority already exists")

        idx_exists = await conn.fetchval(
            "SELECT 1 FROM pg_indexes "
            "WHERE tablename = 'chunking_jobs' AND indexname = 'ix_chunking_jobs_status_priority_created'"
        )
        if not idx_exists:
            await conn.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_chunking_jobs_status_priority_created "
                "ON chunking_jobs (status, priority, created_at)"
            )
            print("  Created index ix_chunking_jobs_status_priority_created")
        else:
            print("  Index ix_chunking_jobs_status_priority_created already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_chunking_job_priority completed.")
