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

from sqlalchemy import select, delete, func, text as sa_text
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


async def publish_document(
    document_id: UUID,
    db: AsyncSession,
    generator_id: str | None = None,
    *,
    background_sync: bool = False,
) -> PublishResult:
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

    # Batch-load the source rows once (dict lookup in the loop) instead of a
    # per-embedding SELECT. A giant doc has ~9k embeddings; the old per-row
    # fetch meant ~9k sequential round-trips, which blew the request timeout
    # (503) and stalled the worker's auto-publish. Two set-based queries here
    # keep publish O(1) round-trips regardless of doc size.
    hier_by_id: dict = {}
    fact_by_id: dict = {}
    _hier_res = await db.execute(
        select(HierarchicalChunk).where(HierarchicalChunk.document_id == document_id)
    )
    for _c in _hier_res.scalars().all():
        hier_by_id[_c.id] = _c
    _fact_res = await db.execute(
        select(ExtractedFact).where(ExtractedFact.document_id == document_id)
    )
    for _f in _fact_res.scalars().all():
        fact_by_id[_f.id] = _f

    # Batch-load chunk-level topic tags from policy_paragraphs.
    # DISTINCT ON (page_number, order_index) deduplicates repeated ingest runs
    # that append rows instead of upsert (e.g. Sunshine Provider Manual p121
    # has 14 duplicate rows). Take the latest by created_at for freshest tags.
    # Keyed by (page_number, order_index) for O(1) lookup in the row loop.
    _chunk_tags_by_pos: dict[tuple[int, int], dict] = {}
    try:
        _pp_rows = await db.execute(
            sa_text(
                """
                SELECT DISTINCT ON (page_number, order_index)
                    page_number, order_index, d_tags, p_tags, j_tags
                FROM policy_paragraphs
                WHERE document_id = :doc_id
                ORDER BY page_number, order_index, created_at DESC
                """
            ),
            {"doc_id": str(document_id)},
        )
        for _pp in _pp_rows.mappings().all():
            _chunk_tags_by_pos[(_pp["page_number"], _pp["order_index"])] = {
                "d": _pp["d_tags"],
                "p": _pp["p_tags"],
                "j": _pp["j_tags"],
            }
    except Exception:
        pass  # non-fatal; chunk_d/p/j_tags stay NULL for this doc

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
            chunk = hier_by_id.get(ce.source_id)
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
            fact = fact_by_id.get(ce.source_id)
            if not fact:
                continue
            text = _build_text_for_fact(fact)
            page_number = fact.page_number or 0
            paragraph_index = 0
            summary = _str_or_empty(fact.fact_text)
            source_verification_status = _str_or_empty(getattr(fact, "verification_status", None) or "")
            if getattr(fact, "hierarchical_chunk_id", None):
                hc = hier_by_id.get(fact.hierarchical_chunk_id)
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
            **({
                "chunk_d_tags": _ct["d"],
                "chunk_p_tags": _ct["p"],
                "chunk_j_tags": _ct["j"],
            } if (_ct := _chunk_tags_by_pos.get((page_number, paragraph_index))) else {}),
        )
        rows.append(row)

    await db.execute(delete(RagPublishedEmbedding).where(RagPublishedEmbedding.document_id == document_id))
    for row in rows:
        db.add(row)
    await db.flush()

    # Step 5 of the Chroma → pgvector migration: populate the typed
    # ``embedding_vec vector(1536)`` column alongside the JSONB
    # ``embedding`` column. Done via raw UPDATE because the ORM model
    # intentionally does not declare ``embedding_vec`` (the
    # ``pgvector.sqlalchemy.Vector`` adapter is not a runtime dep —
    # see pyproject.toml). Same text-form cast as the migration
    # backfill uses (app/migrations/add_pgvector_columns.py).
    # Batch the vec writes as one executemany instead of ~9k awaited UPDATEs
    # (the second O(N) round-trip loop that timed out giant publishes). asyncpg
    # pipelines the parameter list, so this is one driver round-trip regardless
    # of row count. Chunked to bound peak memory of the text-form vectors.
    _vec_params: list[dict] = []
    for row in rows:
        emb = row.embedding
        if not isinstance(emb, list) or not emb:
            continue
        try:
            text_form = "[" + ",".join(repr(float(x)) for x in emb) + "]"
        except (TypeError, ValueError):
            continue
        _vec_params.append({"vec": text_form, "id": str(row.id)})
    _vec_stmt = sa_text(
        "UPDATE rag_published_embeddings "
        "SET embedding_vec = CAST(:vec AS vector) "
        "WHERE id = CAST(:id AS uuid)"
    )
    for _i in range(0, len(_vec_params), 1000):
        _batch = _vec_params[_i : _i + 1000]
        try:
            await db.execute(_vec_stmt, _batch)
        except Exception:
            # Best-effort per-batch fallback: retrieval filters out rows where
            # embedding_vec IS NULL, so a missed row just sits out until the
            # next publish. Retry the batch row-by-row so one bad vector does
            # not sink the whole batch.
            import logging as _logging
            _logging.getLogger(__name__).exception(
                "[publish] embedding_vec batch UPDATE failed (rows %d-%d); retrying row-by-row",
                _i, _i + len(_batch),
            )
            for _p in _batch:
                try:
                    await db.execute(_vec_stmt, _p)
                except Exception:
                    _logging.getLogger(__name__).exception(
                        "[publish] embedding_vec UPDATE failed for id=%s; JSONB row kept", _p["id"],
                    )

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

    # ── Best-effort sync to retrieval stores ───────────────────────
    # Writes to Chroma + chat's Postgres ``published_rag_metadata`` so
    # chat can actually retrieve this document. Env-gated; no-ops if
    # CHROMA_HOST or CHAT_DATABASE_URL aren't set (test envs, early dev).
    # Errors are logged and surfaced in the result message but do NOT
    # fail the publish — the rag-side mart row is already committed,
    # downstream sync is recoverable via backfill_published_to_chat.py.
    #
    # background_sync=True (used by the inline instant pipeline): fires
    # the sync as asyncio.create_task so publish returns in ~0s and the
    # caller can commit the PublishEvent immediately. The 132s chat_pg
    # sync runs concurrently without blocking the upload response path.
    sync_summary: str | None = None
    _is_instant = doc.expires_at is not None
    if background_sync:
        import asyncio as _asyncio
        import logging as _bg_log

        async def _bg_sync() -> None:
            try:
                from app.database import AsyncSessionLocal as _ASL
                from app.services.publish_sync import sync_document_to_retrieval_stores as _sync
                async with _ASL() as _bgs:
                    _res = await _sync(document_id, _bgs, is_instant_rag=_is_instant)
                    _bg_log.getLogger(__name__).info(
                        "[publish] bg-sync done: chunks=%d chroma=%s chat_pg=%s (%.2fs)",
                        _res.chunks_synced, _res.chroma_status, _res.chat_pg_status, _res.duration_s,
                    )
            except Exception as _be:
                _bg_log.getLogger(__name__).warning("[publish] bg-sync failed (non-fatal): %s", _be)

        _asyncio.create_task(_bg_sync())
        return PublishResult(
            rows_written=expected_count,
            verification_passed=True,
            verification_message="published OK; downstream sync running in background",
        )

    try:
        from app.services.publish_sync import sync_document_to_retrieval_stores
        sync_res = await sync_document_to_retrieval_stores(
            document_id, db,
            is_instant_rag=_is_instant,
        )
        sync_summary = (
            f"sync: chunks={sync_res.chunks_synced} "
            f"chroma={sync_res.chroma_status} "
            f"chat_pg={sync_res.chat_pg_status} "
            f"({sync_res.duration_s}s)"
        )
        if not sync_res.ok:
            # publish itself succeeded; flag the partial sync in the
            # response so the caller can decide whether to retry.
            return PublishResult(
                rows_written=expected_count,
                verification_passed=True,
                verification_message=(
                    f"published OK but downstream sync incomplete — {sync_summary}; "
                    f"chroma_msg={sync_res.chroma_message!r} chat_msg={sync_res.chat_pg_message!r}"
                ),
            )
    except Exception as sync_exc:
        # Defensive: a bug in sync code path must not fail publish.
        import logging as _logging
        _logging.getLogger(__name__).exception(
            "[publish] sync_document_to_retrieval_stores raised; publish kept",
        )
        sync_summary = f"sync: raised {type(sync_exc).__name__}"

    return PublishResult(
        rows_written=expected_count,
        verification_passed=True,
        verification_message=sync_summary,
    )
