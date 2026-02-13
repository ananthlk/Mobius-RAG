"""
Path A pipeline: LLM extraction -> critique -> retry -> persist.

Called per-paragraph by the coordinator.  Receives an already-persisted
:class:`HierarchicalChunk` and enriches it with extraction results, facts,
and embeddable units.

All DB writes go through ``worker.db``, events through ``ctx.emit``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.services.chunking import split_paragraphs_from_markdown
from app.services.extraction import stream_extract_facts
from app.services.critique import critique_extraction
from app.services.utils import parse_json_response, parse_json_response_best_effort

from app.worker.context import ChunkingRunContext
from app.worker.errors import record_paragraph_error
from app.worker import db as db_handler

logger = logging.getLogger(__name__)


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_whitespace(s: str) -> str:
    return " ".join(s.split()) if s else ""


def _sanitize_fact_for_db(fact_data: dict) -> dict:
    """Return a copy safe for PostgreSQL."""
    import math
    out = {}
    for k, v in fact_data.items():
        if k == "category_scores":
            continue
        if v is None:
            out[k] = None
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif k in ("is_verified", "is_eligibility_related", "is_pertinent_to_claims_or_members", "confidence") and v is not None:
            out[k] = str(v).lower() if isinstance(v, bool) else str(v)
        else:
            out[k] = v
    return out


def _find_fact_span_in_markdown(
    fact_text: str,
    page_md: str,
    fallback_start: int | None = None,
    fallback_end: int | None = None,
) -> tuple[int | None, int | None]:
    import re
    if not fact_text or not page_md:
        return fallback_start, fallback_end
    if (
        fallback_start is not None
        and fallback_end is not None
        and 0 <= fallback_start < fallback_end <= len(page_md)
    ):
        slice_text = page_md[fallback_start:fallback_end]
        if _normalize_whitespace(slice_text) == _normalize_whitespace(fact_text):
            return fallback_start, fallback_end
    try:
        pattern = re.escape(fact_text).replace("\\ ", r"\\s+")
        m = re.search(pattern, page_md)
        if m:
            return m.start(), m.end()
    except re.error:
        pass
    idx = page_md.find(fact_text)
    if idx >= 0:
        return idx, idx + len(fact_text)
    return fallback_start, fallback_end


# ---------------------------------------------------------------------------
# Per-paragraph processor (called by coordinator)
# ---------------------------------------------------------------------------

async def process_paragraph(
    ctx: ChunkingRunContext,
    chunk,  # HierarchicalChunk — already persisted
    para_id: str,
    paragraph_text: str,
    page_md: str,
    *,
    section_path: str | None = None,
    para_start_offset: int | None = None,
    page_number: int,
    para_idx: int,
    threshold: float = 0.6,
    critique_enabled: bool = True,
    max_retries: int = 2,
    extraction_prompt_body: str | None = None,
    retry_extraction_prompt_body: str | None = None,
    critique_prompt_body: str | None = None,
    llm: Any = None,
) -> None:
    """Enrich an existing HierarchicalChunk with LLM extraction + critique.

    On success: updates chunk status, persists facts, writes embeddable units.
    On failure: records error, marks chunk as failed, but does not raise.
    """
    doc_id = ctx.document_id
    doc_uuid = ctx.doc_uuid
    db = ctx.db

    # ── Stage 1: Extract ──────────────────────────────────────────────
    extraction_result = await _extract(
        ctx, para_id, paragraph_text, section_path,
        extraction_prompt_body=extraction_prompt_body,
        retry_extraction_prompt_body=retry_extraction_prompt_body,
        llm=llm,
    )
    if extraction_result is None:
        # Extraction failed — mark chunk and move on
        chunk.extraction_status = "failed"
        await db_handler.safe_commit(db)
        ctx.record_paragraph_result(para_id, status="failed")
        return

    # ── Stage 2: Critique ─────────────────────────────────────────────
    critique_result = None
    if critique_enabled:
        critique_result = await _critique(
            ctx, para_id, paragraph_text, extraction_result,
            critique_prompt_body=critique_prompt_body,
            llm=llm,
            threshold=threshold,
        )

    # ── Stage 3: Retry loop ───────────────────────────────────────────
    retry_count = 0
    while (
        critique_enabled
        and max_retries > 0
        and retry_count < max_retries
        and critique_result
        and not critique_result.get("pass", False)
    ):
        retry_count += 1
        logger.info("[%s] [%s] Retry %s/%s", doc_id, para_id, retry_count, max_retries)
        try:
            feedback = critique_result.get("feedback", "")
            retry_output = ""
            async for text_chunk in stream_extract_facts(
                paragraph_text,
                critique_feedback=feedback,
                section_path=section_path,
                extraction_prompt_body=extraction_prompt_body,
                retry_extraction_prompt_body=retry_extraction_prompt_body,
                llm=llm,
            ):
                retry_output += text_chunk
            retry_result = parse_json_response(retry_output)
            if retry_result:
                extraction_result = retry_result
                critique_result = await critique_extraction(
                    paragraph_text,
                    extraction_result,
                    critique_prompt_body=critique_prompt_body,
                    llm=llm,
                )
        except Exception as retry_err:
            await record_paragraph_error(
                ctx, para_id, retry_err,
                context_label="retry", stage_label="extraction",
                error_details={"retry_count": retry_count},
            )
            break

    # ── Stage 4: Update chunk + persist facts ─────────────────────────
    facts = extraction_result.get("facts", [])
    summary = extraction_result.get("summary")

    chunk_critique_status = "skipped" if not critique_enabled else (
        "passed" if (critique_result and critique_result.get("pass")) else "failed"
    )

    # Update the shared chunk row
    chunk.extraction_status = "extracted"
    chunk.critique_status = chunk_critique_status
    if summary:
        chunk.summary = summary

    if facts:
        try:
            await db_handler.persist_facts(
                db, chunk, doc_uuid, facts, page_md, paragraph_text, page_number,
                sanitize_fn=_sanitize_fact_for_db,
                find_span_fn=_find_fact_span_in_markdown,
            )

            # Write embeddable units: chunk text + each fact
            try:
                await db_handler.write_embeddable_unit(
                    db, doc_uuid, "A", "chunk", chunk.id,
                    paragraph_text,
                    page_number=page_number,
                    paragraph_index=para_idx,
                    section_path=section_path,
                    metadata={"summary": summary},
                )
                for fact_obj in facts:
                    fact_text = (fact_obj.get("fact_text") or "").strip()
                    if fact_text:
                        await db_handler.write_embeddable_unit(
                            db, doc_uuid, "A", "fact", chunk.id,
                            fact_text,
                            page_number=page_number,
                            paragraph_index=para_idx,
                            section_path=section_path,
                        )
            except Exception as eu_err:
                logger.warning("[%s] [%s] embeddable_unit write (non-fatal): %s", doc_id, para_id, eu_err)

            await db_handler.safe_commit(db)
            logger.info("[%s] [%s] Persisted %s facts", doc_id, para_id, len(facts))

            status = (
                "passed" if (critique_result and critique_result.get("pass"))
                else "skipped" if not critique_enabled
                else "review"
            )
            ctx.record_paragraph_result(
                para_id, status=status, facts=facts,
                summary=summary, critique=critique_result,
            )
        except Exception as persist_err:
            await db_handler.safe_rollback(db)
            await record_paragraph_error(
                ctx, para_id, persist_err,
                context_label="persistence", stage_label="persistence",
            )
    else:
        await db_handler.safe_commit(db)
        ctx.record_paragraph_result(para_id, status="no_facts", summary=summary)


# ---------------------------------------------------------------------------
# Internal stage helpers
# ---------------------------------------------------------------------------

async def _extract(
    ctx: ChunkingRunContext,
    para_id: str,
    paragraph_text: str,
    section_path: str | None,
    *,
    extraction_prompt_body: str | None,
    retry_extraction_prompt_body: str | None,
    llm: Any,
) -> dict | None:
    """Run LLM extraction. Returns parsed result or None on total failure."""
    doc_id = ctx.document_id
    raw_output = ""
    try:
        await ctx.emit(
            "extraction_start",
            message=f"Extraction started for {para_id}.",
            user_message=f"Analyzing section {para_id}...",
            paragraph_id=para_id,
        )
        async for text_chunk in stream_extract_facts(
            paragraph_text,
            section_path=section_path,
            extraction_prompt_body=extraction_prompt_body,
            retry_extraction_prompt_body=retry_extraction_prompt_body,
            llm=llm,
        ):
            raw_output += text_chunk

        result = parse_json_response(raw_output)
        if not result or "summary" not in result:
            raise ValueError("Invalid extraction result: missing summary")

        n_facts = len(result.get("facts", []))
        await ctx.emit(
            "extraction_complete",
            message=f"Paragraph {para_id}: extraction complete, {n_facts} facts.",
            user_message=f"Found {n_facts} fact(s) in this section.",
            paragraph_id=para_id,
            extra={"summary": result.get("summary", ""), "facts": result.get("facts", [])},
        )
        return result
    except Exception as err:
        # Try best-effort parse
        result = parse_json_response_best_effort(raw_output)
        if result is not None:
            n_facts = len(result.get("facts", []))
            logger.warning("[%s] [%s] Best-effort extraction: %s; got %s facts", doc_id, para_id, err, n_facts)
            await ctx.emit(
                "extraction_complete",
                message=f"Paragraph {para_id}: extraction recovered, {n_facts} facts.",
                user_message=f"Found {n_facts} fact(s) (recovered).",
                paragraph_id=para_id,
                extra={"summary": result.get("summary", ""), "facts": result.get("facts", [])},
            )
            return result
        # Total failure
        await record_paragraph_error(
            ctx, para_id, err,
            context_label="extraction", stage_label="extraction",
            error_details={"raw_output": raw_output[:500]},
        )
        await ctx.send_status(
            f"Paragraph {para_id} failed: extraction error.",
            f"Could not process section {para_id}.",
        )
        return None


async def _critique(
    ctx: ChunkingRunContext,
    para_id: str,
    paragraph_text: str,
    extraction_result: dict,
    *,
    critique_prompt_body: str | None,
    llm: Any,
    threshold: float,
) -> dict | None:
    """Run critique. Returns parsed critique dict or None on failure."""
    doc_id = ctx.document_id
    try:
        await ctx.emit(
            "critique_start",
            message=f"Critique started for {para_id} (threshold={threshold}).",
            user_message=f"Reviewing extraction quality for section {para_id}...",
            paragraph_id=para_id,
        )
        cr = await critique_extraction(
            paragraph_text,
            extraction_result,
            critique_prompt_body=critique_prompt_body,
            llm=llm,
        )
        passed = cr and cr.get("pass", False)
        await ctx.emit(
            "critique_complete",
            message=f"Critique for {para_id}: {'PASS' if passed else 'FAIL'}.",
            user_message=f"Quality check: {'passed' if passed else 'needs review'}.",
            paragraph_id=para_id,
            extra={"critique_pass": passed, "feedback": (cr or {}).get("feedback", "")},
        )
        return cr
    except Exception as err:
        logger.warning("[%s] [%s] Critique failed (non-fatal): %s", doc_id, para_id, err, exc_info=True)
        await record_paragraph_error(
            ctx, para_id, err,
            context_label="critique", stage_label="critique",
        )
        return None
