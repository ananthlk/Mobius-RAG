#!/usr/bin/env python3
"""
Kill and reset a single document stuck in extraction/embedding.

Usage:
  cd mobius-rag && uv run python scripts/kill_and_reset_document.py "Sunshine Provider Manual"
  cd mobius-rag && uv run python scripts/kill_and_reset_document.py "59G_5020_Provider_General_REQUIREMENTS.pdf"

Matches document by filename or display_name (case-insensitive contains).
- Marks chunking_jobs (pending/processing) as failed
- Marks embedding_jobs (pending/processing) as failed
- Sets chunking_results.metadata.status to "idle" so UI shows idle
"""
import asyncio
import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/kill_and_reset_document.py <filename or display_name>")
        sys.exit(1)
    name_pattern = sys.argv[1].strip()
    if not name_pattern:
        print("Provide a non-empty filename or display name to match.")
        sys.exit(1)

    from app.config import DATABASE_URL
    from app.models import Document, ChunkingJob, EmbeddingJob, ChunkingResult

    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Find document by filename or display_name (ILIKE contains)
        q = select(Document).where(
            (Document.filename.ilike(f"%{name_pattern}%")) | (Document.display_name.ilike(f"%{name_pattern}%"))
        )
        result = await db.execute(q)
        doc = result.scalar_one_or_none()
        if not doc:
            print(f"No document matching '{name_pattern}' found.")
            sys.exit(1)
        doc_id = doc.id
        label = doc.display_name or doc.filename
        print(f"Document: {label} (id={doc_id})")

        # Fail chunking jobs (pending/processing)
        chunk_result = await db.execute(
            update(ChunkingJob)
            .where(ChunkingJob.document_id == doc_id, ChunkingJob.status.in_(["pending", "processing"]))
            .values(
                status="failed",
                worker_id=None,
                completed_at=None,
                error_message="Killed and reset by kill_and_reset_document.py",
            )
            .returning(ChunkingJob.id)
        )
        chunk_ids = chunk_result.fetchall()
        chunk_count = len(chunk_ids)
        print(f"  Chunking jobs marked failed: {chunk_count}")

        # Fail embedding jobs (pending/processing)
        emb_result = await db.execute(
            update(EmbeddingJob)
            .where(EmbeddingJob.document_id == doc_id, EmbeddingJob.status.in_(["pending", "processing"]))
            .values(
                status="failed",
                worker_id=None,
                started_at=None,
                completed_at=None,
                error_message="Killed and reset by kill_and_reset_document.py",
            )
            .returning(EmbeddingJob.id)
        )
        emb_ids = emb_result.fetchall()
        emb_count = len(emb_ids)
        print(f"  Embedding jobs marked failed: {emb_count}")

        # Set chunking_results.metadata.status to "idle" so UI shows idle
        cr_result = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_id))
        cr = cr_result.scalar_one_or_none()
        if cr:
            meta = dict(cr.metadata_ or {})
            meta["status"] = "idle"
            cr.metadata_ = meta
            print("  ChunkingResult metadata set to idle.")
        else:
            print("  No ChunkingResult row (nothing to update).")

        await db.commit()
    await engine.dispose()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
