#!/usr/bin/env python3
"""Run the same logic as GET /documents/{id}/policy/lines to see the real exception."""
import asyncio
import sys
import os

# Run from mobius-rag so app is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.orm import defer
from app.database import AsyncSessionLocal
from app.models import Document, PolicyLine, PolicyParagraph

DOCUMENT_ID = "e123a670-8be6-4d60-8ad1-132d0847e929"
SKIP = 0
LIMIT = 5000


async def main():
    doc_uuid = UUID(DOCUMENT_ID)
    async with AsyncSessionLocal() as db:
        doc_result = await db.execute(select(Document.id).where(Document.id == doc_uuid))
        if not doc_result.scalar_one_or_none():
            print("Document not found")
            return
        base = (
            select(PolicyLine)
            .join(PolicyParagraph, PolicyParagraph.id == PolicyLine.paragraph_id)
            .where(PolicyLine.document_id == doc_uuid)
        )
        total_q = (
            select(func.count(PolicyLine.id))
            .select_from(PolicyLine)
            .join(PolicyParagraph, PolicyParagraph.id == PolicyLine.paragraph_id)
            .where(PolicyLine.document_id == doc_uuid)
        )
        print("Running count query...")
        total = (await db.execute(total_q)).scalar_one() or 0
        print(f"Total: {total}")
        q = base.order_by(
            PolicyLine.page_number, PolicyParagraph.order_index, PolicyLine.order_index
        ).options(defer(PolicyLine.offset_match_quality)).offset(SKIP).limit(LIMIT)
        print("Running main query...")
        rows = (await db.execute(q)).scalars().all()
        print(f"Rows: {len(rows)}")
        lines = []
        for i, ln in enumerate(rows):
            created_at = getattr(ln, "created_at", None)
            lines.append(
                {
                    "id": str(ln.id),
                    "document_id": str(ln.document_id),
                    "page_number": ln.page_number,
                    "paragraph_id": str(ln.paragraph_id),
                    "paragraph_order_index": None,
                    "order_index": ln.order_index,
                    "parent_line_id": str(ln.parent_line_id) if getattr(ln, "parent_line_id", None) else None,
                    "heading_path": ln.heading_path,
                    "line_type": ln.line_type,
                    "text": ln.text,
                    "is_atomic": ln.is_atomic,
                    "non_atomic_reason": ln.non_atomic_reason,
                    "p_tags": ln.p_tags,
                    "d_tags": ln.d_tags,
                    "j_tags": ln.j_tags,
                    "inferred_d_tags": getattr(ln, "inferred_d_tags", None),
                    "inferred_j_tags": getattr(ln, "inferred_j_tags", None),
                    "conflict_flags": ln.conflict_flags,
                    "extracted_fields": ln.extracted_fields,
                    "start_offset": getattr(ln, "start_offset", None),
                    "end_offset": getattr(ln, "end_offset", None),
                    "offset_match_quality": None,
                    "created_at": created_at.isoformat() if created_at else None,
                }
            )
        out = {"total": int(total), "lines": lines}
        print("Serializing to JSON...")
        import json
        json.dumps(out)
        print("OK: no exception")


if __name__ == "__main__":
    asyncio.run(main())
