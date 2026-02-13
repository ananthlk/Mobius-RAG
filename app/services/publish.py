"""
Publish service: build and write rag_published_embeddings rows for a document (dbt contract).

On user Publish we load all chunk_embeddings for the document, join to chunks/facts and document,
build one row per contract schema, then DELETE existing published rows for that document and INSERT.
After write, runs an integrity check (row count + optional spot-check) and returns verification result.
"""
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Document,
    ChunkEmbedding,
    HierarchicalChunk,
    ExtractedFact,
    RagPublishedEmbedding,
)


@dataclass
class PublishResult:
    rows_written: int
    verification_passed: bool
    verification_message: str | None  # None if passed, else reason


def _build_text_for_chunk(chunk: HierarchicalChunk) -> str:
    """Same as embedding_worker: text that was embedded for a hierarchical chunk."""
    if chunk.summary and chunk.text:
        return f"{chunk.summary}\n{chunk.text}"
    return chunk.text or ""


def _build_text_for_fact(fact: ExtractedFact) -> str:
    """Same as embedding_worker: text that was embedded for a fact."""
    parts = [fact.fact_text or ""]
    extras = []
    if fact.who_eligible:
        extras.append(f"Who eligible: {fact.who_eligible}")
    if fact.how_verified:
        extras.append(f"How verified: {fact.how_verified}")
    if fact.limitations:
        extras.append(f"Limitations: {fact.limitations}")
    if extras:
        parts.append(" ".join(extras))
    return "\n".join(parts).strip() or ""


def _str_or_empty(val: str | None) -> str:
    return (val or "").strip() if val is not None else ""


def _content_sha(document_id: UUID, source_id: UUID, text: str) -> str:
    return hashlib.sha256(f"{document_id}{source_id}{text}".encode()).hexdigest()


async def publish_document(document_id: UUID, db: AsyncSession, generator_id: str | None = None) -> PublishResult:
    """
    Write all embeddings for the given document to rag_published_embeddings (dbt contract).
    Deletes existing published rows for this document first, then inserts new set.
    Runs an integrity check after write (row count + spot-check of a few rows).
    Returns PublishResult(rows_written, verification_passed, verification_message).
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    gen_input = (generator_id or "").strip().upper() or None
    if gen_input and gen_input not in ("A", "B"):
        gen_input = "A"

    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise ValueError("Document not found")

    # Resolve generator_id: use explicit value, or infer from which embeddings exist (try A then B)
    from sqlalchemy import or_
    gen: str
    if gen_input in ("A", "B"):
        gen = gen_input
    else:
        # Caller did not specify; use whichever generator has embeddings (prefer A)
        for candidate in ("A", "B"):
            if candidate == "A":
                where_gen = or_(ChunkEmbedding.generator_id.is_(None), ChunkEmbedding.generator_id == "A")
            else:
                where_gen = (ChunkEmbedding.generator_id == "B")
            ce_check = await db.execute(
                select(ChunkEmbedding).where(ChunkEmbedding.document_id == document_id, where_gen)
            )
            if ce_check.scalars().first() is not None:
                gen = candidate
                break
        else:
            raise ValueError(
                "No chunk embeddings for this document (tried generator_id A and B); run embedding first"
            )

    if gen == "A":
        where_gen = or_(ChunkEmbedding.generator_id.is_(None), ChunkEmbedding.generator_id == "A")
    else:
        where_gen = (ChunkEmbedding.generator_id == "B")
    ce_result = await db.execute(
        select(ChunkEmbedding).where(ChunkEmbedding.document_id == document_id, where_gen)
    )
    embeddings = ce_result.scalars().all()
    if not embeddings:
        raise ValueError(f"No chunk embeddings for this document (generator_id={gen}); run embedding first")

    # Document metadata (contract: empty string when null)
    doc_filename = _str_or_empty(doc.filename)
    doc_display_name = _str_or_empty(doc.display_name)
    doc_authority_level = _str_or_empty(doc.authority_level)
    doc_effective_date = _str_or_empty(doc.effective_date)
    doc_termination_date = _str_or_empty(doc.termination_date)
    doc_payer = _str_or_empty(doc.payer)
    doc_state = _str_or_empty(doc.state) if doc.state else ""
    doc_program = _str_or_empty(doc.program)
    doc_status = _str_or_empty(doc.status)
    doc_review_status = _str_or_empty(doc.review_status)
    doc_created_at = doc.created_at

    rows: list[RagPublishedEmbedding] = []

    for ce in embeddings:
        text = ""
        page_number = 0
        paragraph_index = 0
        section_path = ""
        chapter_path = ""
        summary = ""
        source_verification_status = ""

        if ce.source_type == "hierarchical":
            chunk_result = await db.execute(
                select(HierarchicalChunk).where(HierarchicalChunk.id == ce.source_id)
            )
            chunk = chunk_result.scalar_one_or_none()
            if not chunk:
                continue
            text = _build_text_for_chunk(chunk)
            page_number = chunk.page_number or 0
            paragraph_index = chunk.paragraph_index or 0
            section_path = _str_or_empty(chunk.section_path)
            chapter_path = _str_or_empty(chunk.chapter_path)
            summary = _str_or_empty(chunk.summary)
            source_verification_status = "n/a"
        else:
            fact_result = await db.execute(
                select(ExtractedFact).where(ExtractedFact.id == ce.source_id)
            )
            fact = fact_result.scalar_one_or_none()
            if not fact:
                continue
            text = _build_text_for_fact(fact)
            page_number = fact.page_number or 0
            paragraph_index = 0
            summary = _str_or_empty(fact.fact_text)
            source_verification_status = _str_or_empty(getattr(fact, "verification_status", None) or "")
            if getattr(fact, "hierarchical_chunk_id", None):
                hc_result = await db.execute(
                    select(HierarchicalChunk).where(HierarchicalChunk.id == fact.hierarchical_chunk_id)
                )
                hc = hc_result.scalar_one_or_none()
                if hc:
                    section_path = _str_or_empty(hc.section_path)
                    chapter_path = _str_or_empty(hc.chapter_path)

        content_sha = _content_sha(document_id, ce.source_id, text)
        model_str = (ce.model or "").strip() if ce.model else ""

        row = RagPublishedEmbedding(
            id=ce.id,
            document_id=document_id,
            source_type=ce.source_type,
            source_id=ce.source_id,
            embedding=ce.embedding,
            model=model_str,
            created_at=ce.created_at,
            text=text or "",
            page_number=page_number,
            paragraph_index=paragraph_index,
            section_path=section_path,
            chapter_path=chapter_path,
            summary=summary,
            document_filename=doc_filename,
            document_display_name=doc_display_name,
            document_authority_level=doc_authority_level,
            document_effective_date=doc_effective_date,
            document_termination_date=doc_termination_date,
            document_payer=doc_payer,
            document_state=doc_state,
            document_program=doc_program,
            document_status=doc_status,
            document_created_at=doc_created_at,
            document_review_status=doc_review_status,
            document_reviewed_at=None,
            document_reviewed_by=None,
            content_sha=content_sha,
            updated_at=now,
            source_verification_status=source_verification_status,
        )
        rows.append(row)

    await db.execute(delete(RagPublishedEmbedding).where(RagPublishedEmbedding.document_id == document_id))
    for row in rows:
        db.add(row)
    await db.flush()

    # Integrity check: count rows in rag_published_embeddings for this document
    expected_count = len(rows)
    count_result = await db.execute(
        select(func.count()).select_from(RagPublishedEmbedding).where(RagPublishedEmbedding.document_id == document_id)
    )
    actual_count = count_result.scalar() or 0

    if actual_count != expected_count:
        return PublishResult(
            rows_written=expected_count,
            verification_passed=False,
            verification_message=f"Row count mismatch: expected {expected_count}, found {actual_count} in rag_published_embeddings",
        )

    # Spot-check: verify a few inserted rows exist with correct content_sha and non-null embedding
    sample_size = min(5, len(rows))
    to_check = random.sample(rows, sample_size) if len(rows) > 0 else []
    for row in to_check:
        r_result = await db.execute(
            select(RagPublishedEmbedding).where(RagPublishedEmbedding.id == row.id)
        )
        published = r_result.scalar_one_or_none()
        if not published:
            return PublishResult(
                rows_written=expected_count,
                verification_passed=False,
                verification_message=f"Spot-check failed: row id {row.id} not found after insert",
            )
        if (published.content_sha or "") != (row.content_sha or ""):
            return PublishResult(
                rows_written=expected_count,
                verification_passed=False,
                verification_message=f"Spot-check failed: content_sha mismatch for id {row.id}",
            )
        if published.embedding is None:
            return PublishResult(
                rows_written=expected_count,
                verification_passed=False,
                verification_message=f"Spot-check failed: embedding null for id {row.id}",
            )

    return PublishResult(
        rows_written=expected_count,
        verification_passed=True,
        verification_message=None,
    )
