"""
Separate worker process for processing chunking jobs.
This keeps the main API server responsive by offloading heavy processing.
"""
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import Document, DocumentPage, ChunkingJob, ChunkingResult, ChunkingEvent, EmbeddingJob
from app.services.chunking import split_paragraphs_from_markdown
from app.services.extraction import stream_extract_facts
from app.services.critique import stream_critique, critique_extraction, normalize_critique_result
from app.services.error_tracker import log_error, classify_error
from app.services.utils import parse_json_response, parse_json_response_best_effort
from app.config import CRITIQUE_RETRY_THRESHOLD

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [WORKER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _normalize_whitespace(s: str) -> str:
    """Collapse runs of whitespace to single space and strip."""
    return " ".join(s.split()) if s else ""


def _find_fact_span_in_markdown(
    page_md: str,
    fact_text: str,
    fallback_start: int | None,
    fallback_end: int | None,
) -> tuple[int | None, int | None]:
    """
    Return (start, end) in page_md that best matches fact_text.
    Verifies LLM-provided offsets; if the slice doesn't match fact_text, searches
    for fact_text in page_md (exact or flexible whitespace) so highlights align.
    """
    if not fact_text or not page_md:
        return fallback_start, fallback_end
    # Verify LLM slice if we have offsets
    if (
        fallback_start is not None
        and fallback_end is not None
        and 0 <= fallback_start < fallback_end <= len(page_md)
    ):
        slice_text = page_md[fallback_start:fallback_end]
        if _normalize_whitespace(slice_text) == _normalize_whitespace(fact_text):
            return fallback_start, fallback_end
    # Search with flexible whitespace (collapse \s+ in pattern)
    try:
        pattern = re.escape(fact_text).replace("\\ ", r"\\s+")
        m = re.search(pattern, page_md)
        if m:
            return m.start(), m.end()
    except re.error:
        pass
    # Exact substring
    idx = page_md.find(fact_text)
    if idx >= 0:
        return idx, idx + len(fact_text)
    return fallback_start, fallback_end


def _utc_now_naive():
    """Return naive UTC datetime for DB (TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Worker ID (unique per worker instance)
WORKER_ID = f"worker-{os.getpid()}-{_utc_now_naive().isoformat()}"


def _sanitize_fact_for_db(fact_data: dict) -> dict:
    """Return a copy of fact_data safe for PostgreSQL (String columns). Category scores are parsed into columns separately."""
    import math
    out = {}
    for k, v in fact_data.items():
        if k == "category_scores":
            continue  # Handled via category_scores_dict_to_columns -> individual columns
        if v is None:
            out[k] = None
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif k in ("is_verified", "is_eligibility_related", "is_pertinent_to_claims_or_members", "confidence") and v is not None:
            out[k] = str(v).lower() if isinstance(v, bool) else str(v)
        else:
            out[k] = v
    return out


def _classify_and_log_error(db, document_id, paragraph_id, err, context, stage_label, error_details=None):
    """Determine error_type from context, call classify_error, return (error_type, severity, stage) for log_error."""
    if context == "extraction":
        error_type = "json_parse_error" if isinstance(err, json.JSONDecodeError) else "llm_failure"
    elif context == "persistence":
        error_type = "persistence_error"
    elif context == "retry":
        error_type = "other"
    elif context == "critique":
        error_type = "other"
    else:
        error_type = "other"
    severity, stage = classify_error(error_type, err)
    return error_type, severity, stage


async def _safe_log_error(**kwargs):
    """Call log_error; never raise. Log if log_error fails (defensive)."""
    try:
        await log_error(**kwargs)
    except Exception as e:
        logger.error("log_error failed (unexpected): %s", e, exc_info=True)


async def _create_event_buffer_callback(document_id: str, db: AsyncSession):
    """Create an event callback that writes events to the database."""
    from app.models import ChunkingEvent
    from uuid import UUID
    
    doc_uuid = UUID(document_id)
    event_count = 0
    
    async def event_callback(event_type: str, data: dict):
        nonlocal event_count
        try:
            event = ChunkingEvent(
                document_id=doc_uuid,
                event_type=event_type,
                event_data=data
            )
            db.add(event)
            event_count += 1
            # Commit each progress/chunking event so Live Updates sees it right away
            await db.commit()
            logger.debug(f"[{document_id}] Event #{event_count}: {event_type} - {data.get('message', data.get('current_paragraph', 'no key'))}")
        except Exception as e:
            logger.error(f"[{document_id}] Failed to write event {event_type}: {e}", exc_info=True)
            await db.rollback()
        yield ""  # Yield empty for compatibility
    
    logger.info(f"[{document_id}] Event callback created, will write events to database")
    return event_callback


async def _run_chunking_loop(
    document_id: str,
    doc_uuid: UUID,
    pages: list,
    threshold: float,
    db: AsyncSession,
    event_callback=None,
    critique_enabled: bool = True,
    max_retries: int = 2,
    extraction_enabled: bool = True,
    extraction_prompt_body=None,
    retry_extraction_prompt_body=None,
    critique_prompt_body=None,
    llm=None,
    lexicon_snapshot=None,
):
    """
    Shared chunking loop logic. Runs the full paragraph processing loop.
    critique_enabled: if False, skip critique and retries (extraction only).
    max_retries: when critique enabled, max extraction retries on critique fail (0 = no retry).
    extraction_enabled: if False, only create hierarchical chunks (no LLM extraction/critique).
    extraction_prompt_body, retry_extraction_prompt_body, critique_prompt_body: optional prompt templates (from registry).
    llm: optional LLM provider instance (from llm_config).
    """
    results_paragraphs: dict = {}

    try:
        logger.info(f"[{document_id}] Starting chunking loop with threshold {threshold}, critique_enabled={critique_enabled}, max_retries={max_retries}")
        # Materialize page (page_number, markdown) before any await; use text_markdown as canonical, fallback to raw->md
        from app.services.page_to_markdown import raw_page_to_markdown
        page_data_list = []
        for page in pages:
            md = page.text_markdown if getattr(page, "text_markdown", None) and (page.text_markdown or "").strip() else raw_page_to_markdown(page.text or "")
            if not (md or "").strip():
                continue
            page_data_list.append((page.page_number, md))
        total_pages = len(page_data_list)
        total_paragraphs = sum(len(split_paragraphs_from_markdown(md)) for _, md in page_data_list)
        logger.info(f"[{document_id}] Found {total_pages} pages with markdown, {total_paragraphs} total paragraphs to process")

        async def _upsert(status: str = "in_progress"):
            try:
                # Check if document still exists
                try:
                    doc_check = await db.execute(select(Document).where(Document.id == doc_uuid))
                    if doc_check.scalar_one_or_none() is None:
                        logger.warning(f"[_upsert] Document {doc_uuid} no longer exists")
                        return False
                except Exception as check_err:
                    logger.error(f"[_upsert] Error checking document: {check_err}", exc_info=True)
                
                # Get error counts
                from app.models import ProcessingError
                error_counts = {"critical": 0, "warning": 0, "info": 0}
                try:
                    error_result = await db.execute(
                        select(ProcessingError).where(ProcessingError.document_id == doc_uuid)
                    )
                    errors = error_result.scalars().all()
                    for err in errors:
                        error_counts[err.severity] = error_counts.get(err.severity, 0) + 1
                except Exception as err_err:
                    logger.error(f"[_upsert] Error fetching error counts: {err_err}")
                
                # Upsert chunking result
                result_query = await db.execute(
                    select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid)
                )
                chunking_result = result_query.scalar_one_or_none()
                
                if not chunking_result:
                    chunking_result = ChunkingResult(
                        document_id=doc_uuid,
                        metadata_={},
                        results={}
                    )
                    db.add(chunking_result)
                
                # Update metadata
                metadata = chunking_result.metadata_ or {}
                metadata.update({
                    "status": status,
                    "total_paragraphs": total_paragraphs,
                    "completed_count": len(results_paragraphs),
                    "total_pages": total_pages,
                    "error_counts": error_counts,
                    "last_updated": _utc_now_naive().isoformat()
                })
                chunking_result.metadata_ = metadata
                chunking_result.results = results_paragraphs
                chunking_result.updated_at = _utc_now_naive()
                
                await db.flush()
                await db.commit()
                return True
            except Exception as e:
                await db.rollback()
                logger.error(f"[_upsert] Failed to upsert: {e}", exc_info=True)
                return False

        # Initial upsert
        await _upsert("in_progress")
        logger.info(f"[{document_id}] Initial progress state saved to database")

        # Path B: clear existing policy_paragraphs/lines for this document so we don't duplicate on re-run
        if not extraction_enabled:
            try:
                from sqlalchemy import delete
                from app.models import PolicyLine, PolicyParagraph
                await db.execute(delete(PolicyLine).where(PolicyLine.document_id == doc_uuid))
                await db.execute(delete(PolicyParagraph).where(PolicyParagraph.document_id == doc_uuid))
                await db.flush()
                logger.info(f"[{document_id}] Path B: cleared existing policy paragraphs/lines")
            except Exception as clear_err:
                logger.warning(f"[{document_id}] Path B clear existing policy data (non-fatal): {clear_err}")
        
        # Process each page (use materialized markdown)
        for page_num, (current_page_number, page_md) in enumerate(page_data_list, start=1):
            paragraphs = split_paragraphs_from_markdown(page_md)
            logger.info(f"[{document_id}] Processing page {current_page_number} ({page_num}/{total_pages}) with {len(paragraphs)} paragraphs")
            
            for para_idx, para_data in enumerate(paragraphs):
                paragraph_text = para_data["text"] if isinstance(para_data, dict) else para_data
                section_path = para_data.get("section_path") if isinstance(para_data, dict) else None
                para_id = f"{current_page_number}_{para_idx}"
                
                # Skip if already processed successfully
                if para_id in results_paragraphs:
                    existing = results_paragraphs[para_id]
                    if existing.get("status") == "passed" or existing.get("facts"):
                        continue
                
                async def _send_status(msg: str):
                    if event_callback:
                        try:
                            async for _ in event_callback("status_message", {"message": msg}):
                                pass
                        except Exception as e:
                            logger.error(f"Failed to send status_message: {e}", exc_info=True)

                async def _emit(ev: str, data: dict):
                    if event_callback:
                        try:
                            async for _ in event_callback(ev, data):
                                pass
                        except Exception as e:
                            logger.error(f"[{document_id}] Failed to emit {ev}: {e}", exc_info=True)

                completed_before = len(results_paragraphs)
                progress_pct = (completed_before / total_paragraphs * 100) if total_paragraphs > 0 else 0
                await _emit("paragraph_start", {
                    "paragraph_id": para_id,
                    "paragraph_text": paragraph_text[:2000] if len(paragraph_text) > 2000 else paragraph_text,
                    "page_number": current_page_number,
                    "total_pages": total_pages,
                    "total_paragraphs": total_paragraphs,
                    "completed_paragraphs": completed_before,
                    "progress_percent": progress_pct,
                    "current_paragraph": para_id,
                })
                await _send_status(f"Paragraph {para_id} (page {current_page_number}) started...")
                if not extraction_enabled:
                    # Path B (no LLM): persist hierarchical chunk only, mark as skipped
                    try:
                        from app.models import HierarchicalChunk
                        para_start = para_data.get("start_offset") if isinstance(para_data, dict) else None
                        chunk_query = await db.execute(
                            select(HierarchicalChunk).where(
                                HierarchicalChunk.document_id == doc_uuid,
                                HierarchicalChunk.page_number == current_page_number,
                                HierarchicalChunk.paragraph_index == para_idx,
                            )
                        )
                        chunk = chunk_query.scalar_one_or_none()
                        if not chunk:
                            chunk = HierarchicalChunk(
                                document_id=doc_uuid,
                                page_number=current_page_number,
                                paragraph_index=para_idx,
                                section_path=section_path,
                                text=paragraph_text,
                                text_length=len(paragraph_text),
                                start_offset_in_page=para_start,
                                extraction_status="skipped",
                                critique_status="skipped",
                            )
                            db.add(chunk)
                            await db.flush()
                        # Record paragraph result
                        results_paragraphs[para_id] = {
                            "paragraph_id": para_id,
                            "status": "skipped",
                            "facts": [],
                            "summary": None,
                        }
                        # Path B: build policy_paragraph + policy_lines and apply lexicon tags
                        try:
                            from app.services.policy_path_b import (
                                build_paragraph_and_lines,
                                apply_lexicon_to_lines,
                                get_phrase_to_tag_map,
                            )
                            para_obj, line_objs = await build_paragraph_and_lines(
                                db, doc_uuid, current_page_number, para_idx, section_path, paragraph_text
                            )
                            n_with_tags = 0
                            if lexicon_snapshot is not None and line_objs:
                                phrase_map = get_phrase_to_tag_map(lexicon_snapshot)
                                n_with_tags = await apply_lexicon_to_lines(line_objs, phrase_map)
                            logger.info(
                                f"[{document_id}] Path B: built 1 paragraph, {len(line_objs)} lines for {para_id}"
                                + (f" ({n_with_tags} with tags)" if n_with_tags else "")
                            )
                            # Commit policy data so a later event_callback rollback doesn't undo it
                            await db.commit()
                        except Exception as policy_err:
                            logger.warning(f"[{document_id}] [{para_id}] Path B policy build/tag (non-fatal): {policy_err}", exc_info=True)
                            await db.rollback()  # so _upsert and later steps can use the session
                        await _upsert("in_progress")
                        await _emit("paragraph_complete", {"paragraph_id": para_id, "status": "skipped", "facts": []})
                        continue
                    except Exception as e:
                        logger.error(f"[{document_id}] [{para_id}] Failed to persist hierarchical-only chunk: {e}", exc_info=True)
                        # Fall through to normal flow (may fail); keep going

                logger.info(f"[{document_id}] [{para_id}] Starting extraction (text length: {len(paragraph_text)} chars)")
                
                try:
                    # Stage 1: Extract facts
                    raw_extraction_output = ""
                    extraction_result = None
                    extraction_start = _utc_now_naive()
                    try:
                        await _emit("extraction_start", {"paragraph_id": para_id, "current_paragraph": para_id})
                        logger.info(f"[{document_id}] [{para_id}] Calling LLM for extraction (prompt ~{len(paragraph_text) + 500} chars)")
                        first_chunk_logged = False
                        async for chunk in stream_extract_facts(
                            paragraph_text,
                            section_path=section_path,
                            extraction_prompt_body=extraction_prompt_body,
                            retry_extraction_prompt_body=retry_extraction_prompt_body,
                            llm=llm,
                        ):
                            raw_extraction_output += chunk
                            if not first_chunk_logged and raw_extraction_output:
                                first_chunk_logged = True
                                t = (_utc_now_naive() - extraction_start).total_seconds()
                                logger.info(f"[{document_id}] [{para_id}] LLM first chunk received (after {t:.2f}s)")
                        
                        extraction_duration = (_utc_now_naive() - extraction_start).total_seconds()
                        logger.info(f"[{document_id}] [{para_id}] Extraction done in {extraction_duration:.2f}s, output len={len(raw_extraction_output)}")
                        
                        extraction_result = parse_json_response(raw_extraction_output)
                        if not extraction_result or "summary" not in extraction_result:
                            raise ValueError("Invalid extraction result: missing summary")
                        
                        n_facts = len(extraction_result.get("facts", []))
                        await _emit("extraction_complete", {
                            "paragraph_id": para_id,
                            "summary": extraction_result.get("summary", ""),
                            "facts": extraction_result.get("facts", []),
                        })
                        await _send_status(f"{n_facts} fact(s) found.")
                        logger.info(f"[{document_id}] [{para_id}] Extraction successful: {n_facts} facts found, summary: {extraction_result.get('summary', '')[:100]}")
                    except Exception as extract_err:
                        # Try best-effort parse so we can proceed with whatever facts we can get; log and continue if we get something
                        extraction_result = parse_json_response_best_effort(raw_extraction_output)
                        if extraction_result is not None:
                            n_facts = len(extraction_result.get("facts", []))
                            logger.warning(f"[{document_id}] [{para_id}] Proceeding with best-effort extraction after parse error: {extract_err}; got {n_facts} fact(s)")
                            await _emit("extraction_complete", {
                                "paragraph_id": para_id,
                                "summary": extraction_result.get("summary", ""),
                                "facts": extraction_result.get("facts", []),
                            })
                            await _safe_log_error(
                                db=db,
                                document_id=str(doc_uuid),
                                paragraph_id=para_id,
                                error_type="json_parse_error",
                                severity="warning",
                                error_message=str(extract_err),
                                error_details={"stage": "extraction", "raw_output": raw_extraction_output[:500], "best_effort": True},
                                stage="extraction"
                            )
                            await _send_status(f"{n_facts} fact(s) found (recovered from parse error).")
                        else:
                            error_type, severity, stage = _classify_and_log_error(
                                db, doc_uuid, para_id, extract_err, "extraction", "extraction"
                            )
                            await _safe_log_error(
                                db=db,
                                document_id=str(doc_uuid),
                                paragraph_id=para_id,
                                error_type=error_type,
                                severity=severity,
                                error_message=str(extract_err),
                                error_details={"stage": "extraction", "raw_output": raw_extraction_output[:500]},
                                stage=stage
                            )
                            results_paragraphs[para_id] = {
                                "paragraph_id": para_id,
                                "status": "failed",
                                "error": str(extract_err)
                            }
                            await _send_status(f"Paragraph {para_id} failed: extraction error.")
                            await _upsert("in_progress")
                            continue
                    
                    # Stage 2: Critique (skip if critique_enabled is False)
                    critique_result = None
                    if critique_enabled:
                        critique_start = _utc_now_naive()
                        try:
                            await _emit("critique_start", {"paragraph_id": para_id})
                            logger.info(f"[{document_id}] [{para_id}] Calling LLM for critique (threshold: {threshold})")
                            critique_result = await critique_extraction(
                                paragraph_text,
                                extraction_result,
                                critique_prompt_body=critique_prompt_body,
                                llm=llm,
                            )
                            critique_duration = (_utc_now_naive() - critique_start).total_seconds()

                            if critique_result and critique_result.get("pass"):
                                score = critique_result.get("score", 0)
                                await _emit("critique_complete", {
                                    "paragraph_id": para_id,
                                    "pass": True,
                                    "score": score,
                                    "category_assessment": critique_result.get("category_assessment") or {},
                                    "feedback": critique_result.get("feedback"),
                                    "issues": critique_result.get("issues") or [],
                                })
                                await _send_status("Critique passed.")
                                logger.info(f"[{document_id}] [{para_id}] Critique PASSED (score: {score:.3f}, duration: {critique_duration:.2f}s)")
                            else:
                                score = critique_result.get("score", 0) if critique_result else 0
                                feedback = (critique_result.get("feedback") or "")[:100] if critique_result else "No feedback"
                                await _emit("critique_complete", {
                                    "paragraph_id": para_id,
                                    "pass": False,
                                    "score": score,
                                    "category_assessment": (critique_result or {}).get("category_assessment") or {},
                                    "feedback": (critique_result or {}).get("feedback"),
                                    "issues": (critique_result or {}).get("issues") or [],
                                })
                                await _send_status("Critique failed.")
                                logger.info(f"[{document_id}] [{para_id}] Critique FAILED (score: {score:.3f}, duration: {critique_duration:.2f}s, feedback: {feedback})")
                        except Exception as critique_err:
                            error_type, severity, stage = _classify_and_log_error(
                                db, doc_uuid, para_id, critique_err, "critique", "critique"
                            )
                            await _safe_log_error(
                                db=db,
                                document_id=str(doc_uuid),
                                paragraph_id=para_id,
                                error_type=error_type,
                                severity=severity,
                                error_message=str(critique_err),
                                error_details={"stage": "critique"},
                                stage=stage
                            )
                            await _send_status("Critique failed (exception).")
                            # Continue with extraction result even if critique fails

                    # Stage 3: Retry if needed (only when critique enabled and max_retries > 0)
                    retry_count = 0
                    while (
                        critique_enabled
                        and max_retries > 0
                        and retry_count < max_retries
                        and critique_result
                        and not critique_result.get("pass", False)
                    ):
                        retry_count += 1
                        logger.info(f"[{document_id}] [{para_id}] Retry attempt {retry_count}/{max_retries}")
                        try:
                            feedback = critique_result.get("feedback", "")
                            retry_start = _utc_now_naive()
                            retry_extraction = stream_extract_facts(
                                paragraph_text,
                                critique_feedback=feedback,
                                section_path=section_path,
                                extraction_prompt_body=extraction_prompt_body,
                                retry_extraction_prompt_body=retry_extraction_prompt_body,
                                llm=llm,
                            )
                            retry_output = ""
                            retry_first_chunk = False
                            async for chunk in retry_extraction:
                                retry_output += chunk
                                if not retry_first_chunk and retry_output:
                                    retry_first_chunk = True
                                    t = (_utc_now_naive() - retry_start).total_seconds()
                                    logger.info(f"[{document_id}] [{para_id}] Retry {retry_count} LLM first chunk (after {t:.2f}s)")
                            
                            retry_duration = (_utc_now_naive() - retry_start).total_seconds()
                            retry_result = parse_json_response(retry_output)
                            if retry_result:
                                extraction_result = retry_result
                                n_facts_retry = len(extraction_result.get("facts", []))
                                logger.info(f"[{document_id}] [{para_id}] Retry {retry_count} extraction completed ({n_facts_retry} facts, {retry_duration:.2f}s)")
                                
                                critique_start = _utc_now_naive()
                                critique_result = await critique_extraction(
                                paragraph_text,
                                extraction_result,
                                critique_prompt_body=critique_prompt_body,
                                llm=llm,
                            )
                                critique_duration = (_utc_now_naive() - critique_start).total_seconds()
                                
                                if critique_result and critique_result.get("pass"):
                                    logger.info(f"[{document_id}] [{para_id}] Retry {retry_count} critique PASSED (score: {critique_result.get('score', 0):.3f})")
                                else:
                                    logger.info(f"[{document_id}] [{para_id}] Retry {retry_count} critique still FAILED (score: {(critique_result.get('score', 0) if critique_result else 0):.3f})")
                        except Exception as retry_err:
                            error_type, severity, stage = _classify_and_log_error(
                                db, doc_uuid, para_id, retry_err, "retry", "extraction"
                            )
                            await _safe_log_error(
                                db=db,
                                document_id=str(doc_uuid),
                                paragraph_id=para_id,
                                error_type=error_type,
                                severity=severity,
                                error_message=str(retry_err),
                                error_details={"stage": "retry", "retry_count": retry_count},
                                stage=stage
                            )
                            break
                    
                    # Stage 4: Persist facts
                    facts = extraction_result.get("facts", [])
                    if facts:
                        try:
                            logger.debug(f"[{document_id}] [{para_id}] Persisting {len(facts)} facts to database")
                            from app.models import HierarchicalChunk, ExtractedFact, category_scores_dict_to_columns
                            
                            # Get or create hierarchical chunk
                            chunk_query = await db.execute(
                                select(HierarchicalChunk).where(
                                    HierarchicalChunk.document_id == doc_uuid,
                                    HierarchicalChunk.page_number == current_page_number,
                                    HierarchicalChunk.paragraph_index == para_idx
                                )
                            )
                            chunk = chunk_query.scalar_one_or_none()
                            
                            if not chunk:
                                para_start = para_data.get("start_offset") if isinstance(para_data, dict) else None
                                chunk_critique_status = "skipped" if not critique_enabled else (
                                    "passed" if (critique_result and critique_result.get("pass")) else "failed"
                                )
                                chunk = HierarchicalChunk(
                                    document_id=str(doc_uuid),
                                    page_number=current_page_number,
                                    paragraph_index=para_idx,
                                    section_path=section_path,
                                    text=paragraph_text,
                                    text_length=len(paragraph_text),
                                    start_offset_in_page=para_start,
                                    summary=extraction_result.get("summary"),
                                    extraction_status="extracted",
                                    critique_status=chunk_critique_status,
                                )
                                db.add(chunk)
                                await db.flush()
                                logger.debug(f"[{document_id}] [{para_id}] Created new HierarchicalChunk {chunk.id}")
                            else:
                                if getattr(chunk, "start_offset_in_page", None) is None:
                                    para_start = para_data.get("start_offset") if isinstance(para_data, dict) else None
                                    if para_start is not None:
                                        chunk.start_offset_in_page = para_start
                                        await db.flush()
                                logger.debug(f"[{document_id}] [{para_id}] Using existing HierarchicalChunk {chunk.id}")
                            
                            # Delete existing facts for this chunk
                            from sqlalchemy import delete
                            delete_result = await db.execute(
                                delete(ExtractedFact).where(ExtractedFact.hierarchical_chunk_id == chunk.id)
                            )
                            deleted_count = delete_result.rowcount if hasattr(delete_result, 'rowcount') else 0
                            if deleted_count > 0:
                                logger.debug(f"[{document_id}] [{para_id}] Deleted {deleted_count} existing facts")
                            
                            # Insert new facts (sanitize strings; category scores -> individual columns)
                            # LLM source highlighting: source_start/source_end in paragraph -> page markdown offsets; verify/correct so highlight matches fact_text
                            for fact_idx, fact_data in enumerate(facts):
                                safe = _sanitize_fact_for_db(fact_data)
                                cat_cols = category_scores_dict_to_columns(safe.get("category_scores") or fact_data.get("category_scores"))
                                fact_text = (safe.get("fact_text") or "").strip()
                                fact_page_number = None
                                fact_start_offset = None
                                fact_end_offset = None
                                chunk_start = getattr(chunk, "start_offset_in_page", None)
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
                                # Ensure stored offsets span fact_text in page markdown (fixes wrong LLM source_start/source_end)
                                if fact_text:
                                    fact_start_offset, fact_end_offset = _find_fact_span_in_markdown(
                                        page_md, fact_text, fact_start_offset, fact_end_offset
                                    )
                                    if fact_start_offset is not None and fact_end_offset is not None and fact_page_number is None:
                                        fact_page_number = current_page_number
                                fact = ExtractedFact(
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
                                    **cat_cols
                                )
                                db.add(fact)
                            
                            await db.commit()
                            logger.info(f"[{document_id}] [{para_id}] Successfully persisted {len(facts)} facts to database")
                            
                            results_paragraphs[para_id] = {
                                "paragraph_id": para_id,
                                "summary": extraction_result.get("summary"),
                                "facts": facts,
                                "status": (
                                    "passed" if (critique_result and critique_result.get("pass"))
                                    else "skipped" if not critique_enabled
                                    else "review"
                                ),
                                "critique": critique_result
                            }
                        except Exception as persist_err:
                            await db.rollback()
                            error_type, severity, stage = _classify_and_log_error(
                                db, doc_uuid, para_id, persist_err, "persistence", "persistence"
                            )
                            await _safe_log_error(
                                db=db,
                                document_id=str(doc_uuid),
                                paragraph_id=para_id,
                                error_type=error_type,
                                severity=severity,
                                error_message=str(persist_err),
                                error_details={"stage": "persistence"},
                                stage=stage
                            )
                            results_paragraphs[para_id] = {
                                "paragraph_id": para_id,
                                "status": "failed",
                                "error": str(persist_err)
                            }
                    else:
                        results_paragraphs[para_id] = {
                            "paragraph_id": para_id,
                            "status": "no_facts",
                            "summary": extraction_result.get("summary")
                        }
                    
                    await _send_status(f"Paragraph {para_id} complete.")
                    completed_after = len(results_paragraphs)
                    progress_pct_after = (completed_after / total_paragraphs * 100) if total_paragraphs > 0 else 0
                    logger.info(f"[{document_id}] [{para_id}] Paragraph complete. Progress: {completed_after}/{total_paragraphs} ({progress_pct_after:.1f}%)")
                    
                    # One progress update per paragraph (key progress only)
                    if event_callback:
                        try:
                            async for _ in event_callback("progress_update", {
                                "current_paragraph": para_id,
                                "current_page": current_page_number,
                                "total_pages": total_pages,
                                "total_paragraphs": total_paragraphs,
                                "completed_paragraphs": completed_after,
                                "progress_percent": progress_pct_after
                            }):
                                pass
                        except Exception as progress_err:
                            logger.error(f"[{document_id}] Error sending progress_update event: {progress_err}", exc_info=True)
                    
                    # Update progress
                    await _upsert("in_progress")
                    
                except Exception as para_err:
                    logger.error(f"[{para_id}] Error processing paragraph: {para_err}", exc_info=True)
                    error_type, severity, stage = _classify_and_log_error(
                        db, doc_uuid, para_id, para_err, "other", "other"
                    )
                    await _safe_log_error(
                        db=db,
                        document_id=str(doc_uuid),
                        paragraph_id=para_id,
                        error_type=error_type,
                        severity=severity,
                        error_message=str(para_err),
                        error_details={"stage": "paragraph_processing"},
                        stage=stage
                    )
                    results_paragraphs[para_id] = {
                        "paragraph_id": para_id,
                        "status": "failed",
                        "error": str(para_err)
                    }
                    
                    await _send_status(f"Paragraph {para_id} failed: {str(para_err)[:80]}.")
                    
                    # Progress update on error too (so UI shows count)
                    if event_callback:
                        try:
                            completed_after = len(results_paragraphs)
                            progress_pct_after = (completed_after / total_paragraphs * 100) if total_paragraphs > 0 else 0
                            async for _ in event_callback("progress_update", {
                                "current_paragraph": para_id,
                                "current_page": current_page_number,
                                "total_pages": total_pages,
                                "total_paragraphs": total_paragraphs,
                                "completed_paragraphs": completed_after,
                                "progress_percent": progress_pct_after,
                                "error": str(para_err),
                            }):
                                pass
                        except Exception as progress_err:
                            logger.error(f"Error sending progress_update event: {progress_err}", exc_info=True)
                    
                    await _upsert("in_progress")
        
        # Send chunking_complete event
        n_done = len(results_paragraphs)
        logger.info(f"[{document_id}] Chunking loop complete: {n_done}/{total_paragraphs} paragraphs processed")
        
        if event_callback:
            try:
                async for _ in event_callback("status_message", {
                    "message": f"Chunking complete. {n_done} of {total_paragraphs} paragraphs processed."
                }):
                    pass
                async for _ in event_callback("chunking_complete", {
                    "total_paragraphs": total_paragraphs,
                    "completed_paragraphs": n_done
                }):
                    pass
            except Exception as complete_err:
                logger.error(f"[{document_id}] Error sending chunking_complete event: {complete_err}", exc_info=True)

        # Path B: extract lexicon candidates (phrases not in lexicon)
        if not extraction_enabled and lexicon_snapshot is not None:
            try:
                from app.services.policy_path_b import get_phrase_to_tag_map, extract_candidates_for_document
                phrase_map = get_phrase_to_tag_map(lexicon_snapshot)
                await extract_candidates_for_document(db, doc_uuid, run_id=None, phrase_map=phrase_map)
            except Exception as cand_err:
                logger.warning(f"[{document_id}] Path B candidate extraction (non-fatal): {cand_err}", exc_info=True)
        
        # Final upsert
        await _upsert("completed")
        logger.info(f"[{document_id}] Final state saved to database with status 'completed'")
        return True
        
    except Exception as e:
        logger.error(f"Error in chunking loop: {e}", exc_info=True)
        await _upsert("failed")
        return False


async def process_job(job: ChunkingJob, db: AsyncSession):
    """Process a single chunking job."""
    job_start_time = _utc_now_naive()
    try:
        logger.info(f"[JOB {job.id}] Starting processing for document {job.document_id} (threshold: {job.threshold})")
        
        # Update job status
        job.status = "processing"
        job.worker_id = WORKER_ID
        job.started_at = job_start_time
        await db.commit()
        logger.info(f"[JOB {job.id}] Job status updated to 'processing', assigned to worker {WORKER_ID}")
        
        # Get document
        doc_uuid = UUID(str(job.document_id))
        doc_result = await db.execute(select(Document).where(Document.id == doc_uuid))
        document = doc_result.scalar_one_or_none()
        
        if not document:
            logger.error("[JOB %s] Document not found: %s", job.id, job.document_id)
            job.status = "failed"
            job.error_message = f"Document {job.document_id} not found"
            job.completed_at = _utc_now_naive()
            await db.commit()
            return
        
        logger.info(f"[JOB {job.id}] Document found: {document.filename} (status: {document.status})")
        
        # Get pages
        pages_result = await db.execute(
            select(DocumentPage).where(DocumentPage.document_id == doc_uuid)
            .order_by(DocumentPage.page_number)
        )
        pages = pages_result.scalars().all()
        
        if not pages:
            logger.error("[JOB %s] No pages found for document %s", job.id, job.document_id)
            job.status = "failed"
            job.error_message = f"No pages found for document {job.document_id}"
            job.completed_at = _utc_now_naive()
            await db.commit()
            return
        
        logger.info(f"[JOB {job.id}] Found {len(pages)} pages for document")
        
        # Parse threshold (do not assume job.threshold exists or is valid)
        try:
            threshold = float(job.threshold) if job.threshold is not None else 0.6
        except (TypeError, ValueError) as e:
            logger.warning("[JOB %s] Invalid job.threshold %r, using 0.6: %s", job.id, getattr(job, "threshold", None), e)
            threshold = 0.6
        logger.info(f"[JOB {job.id}] Using critique threshold: {threshold}")
        
        # Create event callback (pass db session so it can write events)
        event_callback = await _create_event_buffer_callback(str(job.document_id), db)
        
        # Run mode from job (default: critique on, max_retries 2). No assumptions.
        try:
            critique_enabled = job.critique_enabled is None or (str(job.critique_enabled).lower() == "true")
        except Exception:
            critique_enabled = True
        try:
            max_retries = 2 if job.max_retries is None else max(0, int(job.max_retries))
        except (TypeError, ValueError):
            max_retries = 2

        # Extraction enabled: default True when missing (back-compat). Path B (generator_id B) never runs LLM.
        try:
            extraction_enabled = job.extraction_enabled is None or (str(job.extraction_enabled).lower() == "true")
        except Exception:
            extraction_enabled = True
        gen = (getattr(job, "generator_id", None) or "A").strip().upper() or "A"
        if gen == "B":
            extraction_enabled = False
            logger.info("[JOB %s] Path B: extraction_enabled=False (no LLM)", job.id)

        # Resolve prompt versions and LLM config from job (run-configured). No assumptions.
        extraction_prompt_body = None
        retry_extraction_prompt_body = None
        critique_prompt_body = None
        llm = None
        prompt_versions = getattr(job, "prompt_versions", None) or {}
        llm_config_version = getattr(job, "llm_config_version", None)
        try:
            if isinstance(prompt_versions, dict):
                from app.services.prompt_registry import get_prompt
                ext_ver = prompt_versions.get("extraction") or "v1"
                retry_ver = prompt_versions.get("extraction_retry") or "v1"
                crit_ver = prompt_versions.get("critique") or "v1"
                extraction_prompt_body = get_prompt("extraction", ext_ver)
                retry_extraction_prompt_body = get_prompt("extraction_retry", retry_ver)
                critique_prompt_body = get_prompt("critique", crit_ver)
        except Exception as e:
            logger.warning("[JOB %s] Prompt resolution failed, using in-code defaults: %s", job.id, e, exc_info=True)
        try:
            if llm_config_version:
                from app.services.llm_config import get_llm_config_resolved, get_llm_provider_from_config
                cfg = await get_llm_config_resolved(llm_config_version, db)
                if cfg:
                    llm = get_llm_provider_from_config(cfg)
                    logger.info(f"[JOB {job.id}] Using LLM config: {llm_config_version}")
        except Exception as e:
            logger.warning("[JOB %s] LLM config %r failed, falling back to default provider: %s", job.id, llm_config_version, e, exc_info=True)
        if not llm:
            try:
                from app.services.llm_provider import get_llm_provider
                llm = get_llm_provider()
            except Exception as e:
                logger.error("[JOB %s] get_llm_provider failed: %s", job.id, e, exc_info=True)
                job.status = "failed"
                job.error_message = f"LLM provider init failed: {e}"
                job.completed_at = _utc_now_naive()
                await db.commit()
                return

        # Path B: load lexicon once for tag application and candidate extraction
        lexicon_snapshot = None
        if gen == "B":
            try:
                from app.services.policy_lexicon_repo import load_lexicon_snapshot_db
                from app.services.policy_path_b import get_phrase_to_tag_map
                lexicon_snapshot = await load_lexicon_snapshot_db(db)
                n_p = len(getattr(lexicon_snapshot, "p_tags") or {})
                n_d = len(getattr(lexicon_snapshot, "d_tags") or {})
                n_j = len(getattr(lexicon_snapshot, "j_tags") or {})
                phrase_map = get_phrase_to_tag_map(lexicon_snapshot)
                n_phrases = len(phrase_map)
                if n_phrases == 0:
                    logger.warning(
                        "[JOB %s] Path B: lexicon has 0 phrases (entries: p=%s d=%s j=%s). "
                        "Populate policy_lexicon_entries with spec.phrases or spec.description; no p/d/j tags will be applied and candidates will not exclude existing tags.",
                        job.id, n_p, n_d, n_j,
                    )
                else:
                    logger.info(
                        "[JOB %s] Path B: loaded lexicon with %s phrases for tagging (entries: p=%s d=%s j=%s)",
                        job.id, n_phrases, n_p, n_d, n_j,
                    )
            except Exception as lex_err:
                logger.warning("[JOB %s] Path B: could not load lexicon (tags/candidates may be empty): %s", job.id, lex_err)

        # Run chunking loop
        logger.info(f"[JOB {job.id}] Starting chunking loop...")
        success = await _run_chunking_loop(
            str(job.document_id),
            doc_uuid,
            pages,
            threshold,
            db,
            event_callback=event_callback,
            critique_enabled=critique_enabled,
            max_retries=max_retries,
            extraction_enabled=extraction_enabled,
            extraction_prompt_body=extraction_prompt_body,
            retry_extraction_prompt_body=retry_extraction_prompt_body,
            critique_prompt_body=critique_prompt_body,
            llm=llm,
            lexicon_snapshot=lexicon_snapshot,
        )
        
        # Commit any remaining events
        await db.commit()
        
        # Update job status
        job_duration = (_utc_now_naive() - job_start_time).total_seconds()
        if success:
            job.status = "completed"
            job.completed_at = _utc_now_naive()
            logger.info(f"[JOB {job.id}] Job completed successfully in {job_duration:.2f}s")
            # Enqueue embedding job for this document+generator (if not already pending)
            try:
                gen = (getattr(job, "generator_id", None) or "A").strip().upper() or "A"
                if gen not in ("A", "B"):
                    gen = "A"
                if gen == "A":
                    from sqlalchemy import or_
                    where_gen = or_(EmbeddingJob.generator_id.is_(None), EmbeddingJob.generator_id == "A")
                else:
                    where_gen = (EmbeddingJob.generator_id == "B")
                existing = await db.execute(
                    select(EmbeddingJob).where(
                        EmbeddingJob.document_id == job.document_id,
                        where_gen,
                        EmbeddingJob.status == "pending",
                    ).limit(1)
                )
                if existing.scalar_one_or_none() is None:
                    # Preserve generator_id so A/B runs can embed independently.
                    embedding_job = EmbeddingJob(
                        document_id=job.document_id,
                        status="pending",
                        generator_id=getattr(job, "generator_id", None),
                    )
                    db.add(embedding_job)
                    logger.info(f"[JOB {job.id}] Enqueued embedding job for document {job.document_id}")
            except Exception as enq_err:
                logger.warning(f"[JOB {job.id}] Failed to enqueue embedding job: {enq_err}", exc_info=True)
        else:
            job.status = "failed"
            job.error_message = "Chunking loop returned False"
            logger.warning(f"[JOB {job.id}] Job failed after {job_duration:.2f}s: chunking loop returned False")
        
        await db.commit()
        logger.info(f"[JOB {job.id}] Final status: {job.status}")
        
    except Exception as e:
        job_duration = (_utc_now_naive() - job_start_time).total_seconds()
        job_id = getattr(job, "id", None)
        logger.error("[JOB %s] Error processing job after %.2fs: %s", job_id, job_duration, e, exc_info=True)
        try:
            await db.rollback()
        except Exception as rb_err:
            logger.error("[JOB %s] Rollback failed: %s", job_id, rb_err, exc_info=True)
        try:
            job.status = "failed"
            job.error_message = str(e)[:2000]
            job.completed_at = _utc_now_naive()
            await db.commit()
            logger.info("[JOB %s] Job marked as failed", job_id)
        except Exception as commit_err:
            logger.error("[JOB %s] Failed to persist job failure status: %s", job_id, commit_err, exc_info=True)


async def worker_loop():
    """Main worker loop - polls for pending jobs and processes them."""
    logger.info(f"Worker {WORKER_ID} starting...")
    # Ensure extracted_facts has category columns (same DB the worker uses)
    try:
        from app.migrations.category_scores_to_columns import migrate as migrate_category_columns
        await migrate_category_columns()
    except Exception as migrate_err:
        logger.warning(f"Startup migration (category_scores_to_columns) skipped or failed: {migrate_err}")
    poll_count = 0

    while True:
        try:
            async with AsyncSessionLocal() as db:
                # Find pending job
                result = await db.execute(
                    select(ChunkingJob)
                    .where(ChunkingJob.status == "pending")
                    .order_by(ChunkingJob.created_at)
                    .limit(1)
                )
                job = result.scalar_one_or_none()
                
                if job:
                    logger.info(f"Found pending job {job.id} for document {job.document_id}, processing...")
                    await process_job(job, db)
                    poll_count = 0  # Reset poll count after processing a job
                else:
                    poll_count += 1
                    if poll_count % 10 == 0:  # Log every 10th poll (every ~20 seconds)
                        logger.debug(f"No pending jobs, polling... (poll #{poll_count})")
                    # No jobs, wait a bit
                    await asyncio.sleep(2)
        
        except Exception as e:
            logger.error(f"Error in worker loop: {e}", exc_info=True)
            await asyncio.sleep(5)  # Wait longer on error


def main():
    """Entry point for worker process."""
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in worker: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
