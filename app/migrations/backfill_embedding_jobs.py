"""
Backfill embedding jobs for documents that have completed chunking but no embedding job.

Run once to enqueue embedding for documents that were chunked before the embedding worker existed:
    python -m app.migrations.backfill_embedding_jobs
"""
import asyncio
import sys
from pathlib import Path
from uuid import UUID

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.database import AsyncSessionLocal
from app.models import EmbeddingJob


async def backfill():
    async with AsyncSessionLocal() as db:
        # Documents with ChunkingResult status=completed (or metadata.status=completed)
        # that don't have a pending EmbeddingJob
        result = await db.execute(
            text("""
                SELECT d.id
                FROM documents d
                JOIN chunking_results cr ON cr.document_id = d.id
                WHERE (cr.metadata->>'status') = 'completed'
                  AND (SELECT COUNT(*) FROM embedding_jobs ej
                       WHERE ej.document_id = d.id AND ej.status IN ('pending', 'processing')) = 0
                ORDER BY d.created_at DESC
            """)
        )
        rows = result.fetchall()
        doc_ids = [str(row[0]) for row in rows]

        if not doc_ids:
            print("  No documents need embedding job backfill")
            return

        added = 0
        for doc_id in doc_ids:
            job = EmbeddingJob(document_id=UUID(doc_id), status="pending")
            db.add(job)
            added += 1
            print(f"  Enqueued embedding job for document {doc_id}")

        await db.commit()
        print(f"  Backfill complete: {added} embedding job(s) enqueued")


def main():
    asyncio.run(backfill())
    print("Backfill embedding jobs completed.")


if __name__ == "__main__":
    main()
