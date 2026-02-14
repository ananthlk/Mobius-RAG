"""
Chunking worker database handler.

All worker persistence goes through this module: commits, rollbacks,
reads, and writes to ChunkingResult, ChunkingEvent, HierarchicalChunk,
ExtractedFact, PolicyParagraph, PolicyLine, ProcessingError, EmbeddingJob.

Path A and Path B never call ``db.add()`` or raw SQL directly.
"""
from __future__ import annotations

import json as _json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select, delete, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    ChunkingEvent,
    ChunkingJob,
    ChunkingResult,
    Document,
    EmbeddingJob,
    ExtractedFact,
    HierarchicalChunk,
    PolicyLine,
    PolicyParagraph,
    ProcessingError,
)

logger = logging.getLogger(__name__)


def _utc_now_naive() -> datetime:
    """Naive UTC datetime for DB (TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Stale job recovery
# ---------------------------------------------------------------------------

async def recover_stale_jobs(
    db: AsyncSession,
    timeout_minutes: float = 10.0,
    worker_id: str | None = None,
) -> int:
    """Reset jobs stuck in 'processing' for longer than *timeout_minutes*.

    This catches jobs whose workers died (crash, restart, OOM) without
    transitioning the job to ``failed``.  Resets them to ``pending`` so
    a healthy worker can pick them up.

    Returns the number of recovered jobs.
    """
    from datetime import timedelta

    cutoff = _utc_now_naive() - timedelta(minutes=timeout_minutes)

    stmt = (
        update(ChunkingJob)
        .where(
            ChunkingJob.status == "processing",
            ChunkingJob.started_at < cutoff,
        )
        .values(
            status="pending",
            worker_id=None,
            started_at=None,
            error_message=f"Auto-recovered: stuck in processing >{timeout_minutes}min (by {worker_id or 'unknown'})",
        )
        .returning(ChunkingJob.id, ChunkingJob.document_id)
    )

    result = await db.execute(stmt)
    recovered = result.fetchall()
    await db.commit()

    for job_id, doc_id in recovered:
        logger.warning(
            "[stale-recovery] Reset job %s (doc %s) from processing -> pending",
            job_id, doc_id,
        )
    return len(recovered)


# ---------------------------------------------------------------------------
# ChunkingResult (progress / status)
# ---------------------------------------------------------------------------

async def upsert_chunking_result(
    db: AsyncSession,
    doc_uuid: UUID,
    results_paragraphs: dict,
    *,
    status: str = "in_progress",
    total_paragraphs: int = 0,
    total_pages: int = 0,
) -> bool:
    """Create or update the ChunkingResult row for *doc_uuid*.

    Returns True on success, False on error (session is rolled back).
    """
    try:
        # Check document still exists
        doc_check = await db.execute(select(Document).where(Document.id == doc_uuid))
        if doc_check.scalar_one_or_none() is None:
            logger.warning("[db] upsert_chunking_result: document %s gone", doc_uuid)
            return False

        # Error counts
        error_counts: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
        try:
            err_result = await db.execute(
                select(ProcessingError).where(ProcessingError.document_id == doc_uuid)
            )
            for err in err_result.scalars().all():
                error_counts[err.severity] = error_counts.get(err.severity, 0) + 1
        except Exception as err_err:
            logger.error("[db] error fetching error_counts: %s", err_err)

        result_q = await db.execute(
            select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid)
        )
        chunking_result = result_q.scalar_one_or_none()

        if not chunking_result:
            chunking_result = ChunkingResult(
                document_id=doc_uuid,
                metadata_={},
                results={},
            )
            db.add(chunking_result)

        metadata = chunking_result.metadata_ or {}
        metadata.update({
            "status": status,
            "total_paragraphs": total_paragraphs,
            "completed_count": len(results_paragraphs),
            "total_pages": total_pages,
            "error_counts": error_counts,
            "last_updated": _utc_now_naive().isoformat(),
        })
        chunking_result.metadata_ = metadata
        chunking_result.results = results_paragraphs
        chunking_result.updated_at = _utc_now_naive()

        await db.flush()
        await db.commit()
        return True
    except Exception as exc:
        await db.rollback()
        logger.error("[db] upsert_chunking_result failed: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# ChunkingEvent (emit / write_event)
# ---------------------------------------------------------------------------

async def write_event(
    db: AsyncSession,
    doc_uuid: UUID,
    event_type: str,
    event_data: dict,
) -> None:
    """Persist a single ChunkingEvent, commit, and pg_notify for SSE push."""
    try:
        event = ChunkingEvent(
            document_id=doc_uuid,
            event_type=event_type,
            event_data=event_data,
        )
        db.add(event)
        await db.commit()

        # Push real-time notification via pg_notify (payload < 8 KB)
        try:
            payload = _json.dumps({
                "id": str(event.id),
                "document_id": str(doc_uuid),
                "event_type": event_type,
                "event_data": event_data,
            }, default=str)
            await db.execute(
                text("SELECT pg_notify('chunking_events', :p)"), {"p": payload}
            )
            await db.commit()
        except Exception as notify_exc:
            # Non-fatal: SSE will still pick up on next poll/reconnect
            logger.debug("[db] pg_notify failed (non-fatal): %s", notify_exc)
    except Exception as exc:
        logger.error("[db] write_event(%s) failed: %s", event_type, exc, exc_info=True)
        await db.rollback()


# ---------------------------------------------------------------------------
# Policy data (Path B)
# ---------------------------------------------------------------------------

async def clear_policy_for_document(db: AsyncSession, doc_uuid: UUID) -> None:
    """Delete existing PolicyLine + PolicyParagraph rows for *doc_uuid*."""
    try:
        await db.execute(delete(PolicyLine).where(PolicyLine.document_id == doc_uuid))
        await db.execute(delete(PolicyParagraph).where(PolicyParagraph.document_id == doc_uuid))
        await db.flush()
        logger.info("[db] Cleared policy paragraphs/lines for %s", doc_uuid)
    except Exception as exc:
        logger.warning("[db] clear_policy_for_document (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# HierarchicalChunk (Path A & B)
# ---------------------------------------------------------------------------

async def persist_chunk(
    db: AsyncSession,
    doc_uuid: UUID,
    page_number: int,
    paragraph_index: int,
    paragraph_text: str,
    *,
    section_path: str | None = None,
    start_offset_in_page: int | None = None,
    summary: str | None = None,
    extraction_status: str = "pending",
    critique_status: str = "pending",
) -> HierarchicalChunk:
    """Get-or-create a HierarchicalChunk row and return it (flushed, has .id)."""
    q = await db.execute(
        select(HierarchicalChunk).where(
            HierarchicalChunk.document_id == doc_uuid,
            HierarchicalChunk.page_number == page_number,
            HierarchicalChunk.paragraph_index == paragraph_index,
        )
    )
    chunk = q.scalar_one_or_none()

    if not chunk:
        chunk = HierarchicalChunk(
            document_id=doc_uuid,
            page_number=page_number,
            paragraph_index=paragraph_index,
            section_path=section_path,
            text=paragraph_text,
            text_length=len(paragraph_text),
            start_offset_in_page=start_offset_in_page,
            summary=summary,
            extraction_status=extraction_status,
            critique_status=critique_status,
        )
        db.add(chunk)
        await db.flush()
    else:
        # Patch offset if it was missing
        if getattr(chunk, "start_offset_in_page", None) is None and start_offset_in_page is not None:
            chunk.start_offset_in_page = start_offset_in_page
            await db.flush()
    return chunk


# ---------------------------------------------------------------------------
# ExtractedFact (Path A)
# ---------------------------------------------------------------------------

async def persist_facts(
    db: AsyncSession,
    chunk: HierarchicalChunk,
    doc_uuid: UUID,
    facts: list[dict],
    page_md: str,
    paragraph_text: str,
    current_page_number: int,
    *,
    sanitize_fn=None,
    category_scores_fn=None,
    find_span_fn=None,
) -> int:
    """Delete existing facts for *chunk* and insert *facts*.

    Returns number of facts persisted.  Caller should commit after.

    ``sanitize_fn``, ``category_scores_fn``, ``find_span_fn`` are injected
    so this module does not import worker helpers directly.
    """
    from app.models import category_scores_dict_to_columns

    if sanitize_fn is None:
        sanitize_fn = lambda d: d  # noqa: E731
    if category_scores_fn is None:
        category_scores_fn = category_scores_dict_to_columns
    if find_span_fn is None:
        find_span_fn = lambda _md, _txt, s, e: (s, e)  # noqa: E731

    # Delete old facts
    await db.execute(
        delete(ExtractedFact).where(ExtractedFact.hierarchical_chunk_id == chunk.id)
    )

    chunk_start = getattr(chunk, "start_offset_in_page", None)

    for fact_data in facts:
        safe = sanitize_fn(fact_data)
        cat_cols = category_scores_fn(safe.get("category_scores") or fact_data.get("category_scores"))

        fact_text = (safe.get("fact_text") or "").strip()
        fact_page_number = None
        fact_start_offset = None
        fact_end_offset = None

        src_start = fact_data.get("source_start")
        src_end = fact_data.get("source_end")
        if (
            chunk_start is not None
            and src_start is not None
            and src_end is not None
            and isinstance(paragraph_text, str)
        ):
            try:
                so = int(src_start)
                eo = int(src_end)
                if 0 <= so < eo <= len(paragraph_text):
                    fact_page_number = current_page_number
                    fact_start_offset = chunk_start + so
                    fact_end_offset = chunk_start + eo
            except (TypeError, ValueError):
                pass

        if fact_text:
            fact_start_offset, fact_end_offset = find_span_fn(
                page_md, fact_text, fact_start_offset, fact_end_offset
            )
            if fact_start_offset is not None and fact_end_offset is not None and fact_page_number is None:
                fact_page_number = current_page_number

        fact_obj = ExtractedFact(
            hierarchical_chunk_id=chunk.id,
            document_id=str(doc_uuid),
            fact_text=safe.get("fact_text", "") or "",
            fact_type=safe.get("fact_type"),
            who_eligible=safe.get("who_eligible"),
            how_verified=safe.get("how_verified"),
            conflict_resolution=safe.get("conflict_resolution"),
            when_applies=safe.get("when_applies"),
            limitations=safe.get("limitations"),
            is_verified=safe.get("is_verified"),
            is_eligibility_related=safe.get("is_eligibility_related"),
            is_pertinent_to_claims_or_members=safe.get("is_pertinent_to_claims_or_members"),
            confidence=safe.get("confidence"),
            page_number=fact_page_number,
            start_offset=fact_start_offset,
            end_offset=fact_end_offset,
            **cat_cols,
        )
        db.add(fact_obj)

    await db.flush()
    return len(facts)


# ---------------------------------------------------------------------------
# EmbeddingJob (enqueue after chunking completes)
# ---------------------------------------------------------------------------

async def enqueue_embedding_job(
    db: AsyncSession,
    document_id: UUID,
    generator_id: str | None,
) -> bool:
    """Enqueue an EmbeddingJob if one is not already pending. Returns True if enqueued."""
    from sqlalchemy import or_

    gen = (generator_id or "A").strip().upper() or "A"
    if gen not in ("A", "B"):
        gen = "A"

    if gen == "A":
        where_gen = or_(EmbeddingJob.generator_id.is_(None), EmbeddingJob.generator_id == "A")
    else:
        where_gen = EmbeddingJob.generator_id == "B"

    existing = await db.execute(
        select(EmbeddingJob).where(
            EmbeddingJob.document_id == document_id,
            where_gen,
            EmbeddingJob.status == "pending",
        ).limit(1)
    )
    if existing.scalar_one_or_none() is not None:
        return False

    job = EmbeddingJob(
        document_id=document_id,
        status="pending",
        generator_id=generator_id,
    )
    db.add(job)
    return True


# ---------------------------------------------------------------------------
# ProcessingError helper
# ---------------------------------------------------------------------------

async def log_processing_error(
    db: AsyncSession,
    document_id: str,
    paragraph_id: str,
    error_type: str,
    severity: str,
    error_message: str,
    error_details: dict | None = None,
    stage: str | None = None,
) -> None:
    """Insert a ProcessingError row. Never raises."""
    try:
        err = ProcessingError(
            document_id=document_id,
            paragraph_id=paragraph_id,
            error_type=error_type,
            severity=severity,
            error_message=error_message[:2000] if error_message else "",
            error_details=error_details or {},
            stage=stage,
        )
        db.add(err)
        await db.flush()
    except Exception as exc:
        logger.error("[db] log_processing_error failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Embeddable units
# ---------------------------------------------------------------------------

async def write_embeddable_unit(
    db: AsyncSession,
    document_id: UUID,
    generator_id: str | None,
    source_type: str,
    source_id: UUID,
    text: str,
    *,
    page_number: int | None = None,
    paragraph_index: int | None = None,
    section_path: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Insert one row into embeddable_units. Caller should commit."""
    from app.models import EmbeddableUnit

    unit = EmbeddableUnit(
        document_id=document_id,
        generator_id=generator_id,
        source_type=source_type,
        source_id=source_id,
        text=text,
        page_number=page_number,
        paragraph_index=paragraph_index,
        section_path=section_path,
        metadata_=metadata or {},
    )
    db.add(unit)
    await db.flush()


async def clear_embeddable_units(
    db: AsyncSession,
    document_id: UUID,
    generator_id: str | None = None,
) -> int:
    """Delete embeddable_units for a document (optionally scoped to generator).

    Returns the number of deleted rows.
    """
    from app.models import EmbeddableUnit

    stmt = delete(EmbeddableUnit).where(EmbeddableUnit.document_id == document_id)
    if generator_id is not None:
        stmt = stmt.where(EmbeddableUnit.generator_id == generator_id)
    result = await db.execute(stmt)
    await db.flush()
    return result.rowcount if hasattr(result, "rowcount") else 0


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

async def set_config_snapshot(
    db: AsyncSession,
    job: Any,
    *,
    worker_config: Any | None = None,
    extra: dict | None = None,
) -> dict:
    """Build and persist a config snapshot on *job*.chunking_config_snapshot.

    Returns the snapshot dict.  Caller should commit.
    """
    snapshot: dict[str, Any] = {
        "threshold": getattr(job, "threshold", None),
        "critique_enabled": getattr(job, "critique_enabled", None),
        "max_retries": getattr(job, "max_retries", None),
        "extraction_enabled": getattr(job, "extraction_enabled", None),
        "generator_id": getattr(job, "generator_id", None),
        "prompt_versions": getattr(job, "prompt_versions", None),
        "llm_config_version": getattr(job, "llm_config_version", None),
    }
    if worker_config is not None:
        snapshot["worker_defaults"] = {
            "default_threshold": getattr(worker_config, "default_threshold", None),
            "default_critique_enabled": getattr(worker_config, "default_critique_enabled", None),
            "default_max_retries": getattr(worker_config, "default_max_retries", None),
            "path_b_cap_ngrams": getattr(worker_config, "path_b_cap_ngrams", None),
            "path_b_cap_abbrevs": getattr(worker_config, "path_b_cap_abbrevs", None),
            "path_b_min_occurrences": getattr(worker_config, "path_b_min_occurrences", None),
        }
    if extra:
        snapshot.update(extra)
    snapshot["snapshot_at"] = _utc_now_naive().isoformat()
    job.chunking_config_snapshot = snapshot
    return snapshot


# ---------------------------------------------------------------------------
# Commit / rollback helpers
# ---------------------------------------------------------------------------

async def safe_commit(db: AsyncSession) -> bool:
    """Commit; on failure rollback and return False."""
    try:
        await db.commit()
        return True
    except Exception as exc:
        logger.error("[db] commit failed, rolling back: %s", exc, exc_info=True)
        await db.rollback()
        return False


async def safe_rollback(db: AsyncSession) -> None:
    """Rollback; never raises."""
    try:
        await db.rollback()
    except Exception as exc:
        logger.error("[db] rollback failed: %s", exc, exc_info=True)
