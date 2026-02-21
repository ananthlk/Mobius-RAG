#!/usr/bin/env python3
"""Bulk-publish documents that have chunk_embeddings but no rows in rag_published_embeddings.

Finds documents with embeddings that were never published, calls publish_document for each,
and records PublishEvent. Use before running the DBT pipeline so those docs sync to Chat.

Usage:
  python scripts/publish_unpublished_documents.py
  python scripts/publish_unpublished_documents.py --dry-run
  python scripts/publish_unpublished_documents.py --limit 10

Uses DATABASE_URL from mobius-rag env (same DB mobius-dbt reads from for ingest).
"""
import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

# Add repo root so app modules are importable
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from sqlalchemy import select, exists

from app.database import AsyncSessionLocal
from app.models import ChunkEmbedding, PublishEvent, RagPublishedEmbedding
from app.services.publish import PublishResult, publish_document


async def _find_unpublished_document_ids(db) -> list[UUID]:
    """Return document_ids that have chunk_embeddings but no rag_published_embeddings."""
    subq = exists().where(RagPublishedEmbedding.document_id == ChunkEmbedding.document_id)
    q = select(ChunkEmbedding.document_id).where(~subq).distinct()
    result = await db.execute(q)
    return [row[0] for row in result.fetchall()]


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish documents with embeddings but not in rag_published_embeddings"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list documents that would be published, do not write",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of documents to publish (default: no limit)",
    )
    args = parser.parse_args()

    async with AsyncSessionLocal() as db:
        doc_ids = await _find_unpublished_document_ids(db)
        if not doc_ids:
            print("No documents found with chunk_embeddings but not in rag_published_embeddings.")
            return 0

        if args.limit is not None:
            doc_ids = doc_ids[: args.limit]
            print(f"Limited to {args.limit} document(s).")
        print(f"Found {len(doc_ids)} document(s) to publish.")
        if args.dry_run:
            for i, did in enumerate(doc_ids, 1):
                print(f"  [{i}] {did}")
            print("Dry run: no changes made.")
            return 0

        ok = 0
        err = 0
        for i, doc_id in enumerate(doc_ids, 1):
            try:
                result = await publish_document(doc_id, db)
                if not isinstance(result, PublishResult):
                    raise ValueError(f"Unexpected result type: {type(result)}")
                event = PublishEvent(
                    document_id=doc_id,
                    published_by="publish_unpublished_documents.py",
                    rows_written=result.rows_written,
                    verification_passed=result.verification_passed,
                    verification_message=result.verification_message,
                )
                db.add(event)
                await db.commit()
                status = "ok" if result.verification_passed else "verify_fail"
                print(f"  [{i}/{len(doc_ids)}] {doc_id} -> {result.rows_written} rows ({status})")
                ok += 1
            except Exception as e:
                await db.rollback()
                print(f"  [{i}/{len(doc_ids)}] {doc_id} ERROR: {e}", file=sys.stderr)
                err += 1

        print(f"\nDone: {ok} published, {err} failed.")
        return 1 if err else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
