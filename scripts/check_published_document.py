#!/usr/bin/env python3
"""Check if a document landed in the published vector table (rag_published_embeddings).
Usage: python scripts/check_published_document.py [name_substring]
Example: python scripts/check_published_document.py SH-PRO-BH-General
Uses DATABASE_URL from env (Cloud SQL required; see docs/MIGRATE_LOCAL_TO_CLOUD.md).
"""
import asyncio
import sys
from pathlib import Path

# Add repo root so app.config and app.database are importable
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from sqlalchemy import select, or_, func
from app.database import AsyncSessionLocal
from app.models import Document, RagPublishedEmbedding


async def main():
    name = (sys.argv[1] if len(sys.argv) > 1 else "SH-PRO-BH-General").strip()
    if not name:
        print("Usage: python scripts/check_published_document.py [document_name_substring]")
        sys.exit(1)

    print("Dev vector index: PostgreSQL table 'rag_published_embeddings' (DB from DATABASE_URL)")
    print(f"Looking for documents matching: {name!r}")
    print()

    async with AsyncSessionLocal() as db:
        # Find documents by filename or display_name
        q = select(Document).where(
            or_(Document.filename.ilike(f"%{name}%"), Document.display_name.ilike(f"%{name}%"))
        )
        result = await db.execute(q)
        docs = result.scalars().all()

        if not docs:
            print(f"No documents found matching {name!r} in 'documents' table.")
            sys.exit(0)

        for doc in docs:
            print(f"Document: id={doc.id}")
            print(f"  filename={doc.filename!r}, display_name={doc.display_name!r}")
            # Count in rag_published_embeddings
            count_result = await db.execute(
                select(func.count()).select_from(RagPublishedEmbedding).where(RagPublishedEmbedding.document_id == doc.id)
            )
            n = count_result.scalar() or 0
            print(f"  rag_published_embeddings: {n} rows")
            if n:
                sample_result = await db.execute(
                    select(RagPublishedEmbedding).where(RagPublishedEmbedding.document_id == doc.id).limit(1)
                )
                r = sample_result.scalar_one()
                print(f"  sample: document_filename={r.document_filename!r}, document_display_name={r.document_display_name!r}, text_len={len(r.text or '')}")
            print()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
