"""
Chunking coordinator.

The single orchestration function that:
1. Materialises page data.
2. Constructs a :class:`ChunkingRunContext`.
3. Clears stale data (idempotent re-runs).
4. Iterates paragraphs, creating the shared HierarchicalChunk, then
   delegates to Path A (LLM) or Path B (deterministic) for enrichment.
5. Sends completion events and final upsert.

HierarchicalChunk creation is **shared** — both paths receive a persisted
chunk row and enrich it (A: extraction/facts, B: policy lines/tags).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.chunking import split_paragraphs_from_markdown
from app.worker.config import WorkerConfig
from app.worker.context import ChunkingRunContext
from app.worker.errors import record_paragraph_error
from app.worker import db as db_handler
from app.worker import path_a, path_b

logger = logging.getLogger(__name__)


async def run_chunking_loop(
    document_id: str,
    doc_uuid: UUID,
    job_id: str,
    pages: list,
    db: AsyncSession,
    *,
    threshold: float = 0.6,
    critique_enabled: bool = True,
    max_retries: int = 2,
    extraction_enabled: bool = True,
    extraction_prompt_body: str | None = None,
    retry_extraction_prompt_body: str | None = None,
    critique_prompt_body: str | None = None,
    llm: Any = None,
    lexicon_snapshot: Any | None = None,
    worker_cfg: WorkerConfig | None = None,
) -> bool:
    """Top-level chunking coordinator.

    Returns True on overall success, False on unrecoverable failure.
    """
    if worker_cfg is None:
        from app.worker.config import load_worker_config
        worker_cfg = load_worker_config()

    # --- Materialise pages ---
    from app.services.page_to_markdown import raw_page_to_markdown

    page_data_list: list[tuple[int, str]] = []
    for page in pages:
        md = (
            page.text_markdown
            if getattr(page, "text_markdown", None) and (page.text_markdown or "").strip()
            else raw_page_to_markdown(page.text or "")
        )
        if not (md or "").strip():
            continue
        page_data_list.append((page.page_number, md))

    total_pages = len(page_data_list)
    total_paragraphs = sum(
        len(split_paragraphs_from_markdown(md)) for _, md in page_data_list
    )

    logger.info(
        "[%s] Coordinator: %s pages, %s paragraphs, extraction_enabled=%s",
        document_id, total_pages, total_paragraphs, extraction_enabled,
    )

    # --- Build context ---
    ctx = ChunkingRunContext(
        db=db,
        document_id=document_id,
        doc_uuid=doc_uuid,
        job_id=str(job_id),
        total_paragraphs=total_paragraphs,
        total_pages=total_pages,
    )

    # --- Idempotent cleanup ---
    gen = "A" if extraction_enabled else "B"
    try:
        n_cleared = await db_handler.clear_embeddable_units(db, doc_uuid, generator_id=gen)
        if n_cleared:
            logger.info("[%s] Cleared %s stale embeddable_units (gen=%s)", document_id, n_cleared, gen)
        await db_handler.safe_commit(db)
    except Exception as clear_err:
        logger.warning("[%s] embeddable_units cleanup (non-fatal): %s", document_id, clear_err)
        await db_handler.safe_rollback(db)

    # Path-B specific: clear existing policy data so re-runs are idempotent
    if not extraction_enabled:
        await path_b.clear_policy_data(ctx)

    # --- Initial progress ---
    await ctx.upsert_progress("in_progress")

    # --- Emit job_start ---
    await ctx.emit(
        "job_start",
        message=f"Chunking started ({total_paragraphs} paragraphs across {total_pages} pages).",
        user_message=f"Starting document analysis: {total_paragraphs} sections found across {total_pages} pages.",
        extra={
            "total_paragraphs": total_paragraphs,
            "total_pages": total_pages,
            "extraction_enabled": extraction_enabled,
        },
    )

    # --- Main loop (with optional timeout) ---
    timeout = worker_cfg.job_timeout_seconds

    # Prepare path-B resources once (outside the paragraph loop)
    path_b_resources = None
    if not extraction_enabled:
        path_b_resources = path_b.prepare_resources(lexicon_snapshot)

    try:
        coro = _process_paragraphs(
            ctx, page_data_list,
            extraction_enabled=extraction_enabled,
            threshold=threshold,
            critique_enabled=critique_enabled,
            max_retries=max_retries,
            extraction_prompt_body=extraction_prompt_body,
            retry_extraction_prompt_body=retry_extraction_prompt_body,
            critique_prompt_body=critique_prompt_body,
            llm=llm,
            path_b_resources=path_b_resources,
        )
        if timeout:
            success = await asyncio.wait_for(coro, timeout=timeout)
        else:
            success = await coro
    except asyncio.TimeoutError:
        logger.error("[%s] Coordinator: job timed out after %ss", document_id, timeout)
        await ctx.emit(
            "job_timeout",
            message=f"Chunking timed out after {timeout}s.",
            user_message="Document analysis took too long and was stopped.",
        )
        await ctx.upsert_progress("failed")
        return False
    except Exception as exc:
        logger.error("[%s] Coordinator: unhandled error: %s", document_id, exc, exc_info=True)
        await ctx.emit(
            "job_failed",
            message=f"Chunking failed: {str(exc)[:200]}",
            user_message="Document analysis encountered an unexpected error.",
        )
        await ctx.upsert_progress("failed")
        return False

    # --- Path B post-processing: document-level tag aggregation ---
    if not extraction_enabled:
        await path_b.finalise(ctx, path_b_resources)

    # --- Completion events ---
    n_done = ctx.completed_count
    await ctx.emit(
        "chunking_complete",
        message=f"Chunking complete. {n_done}/{total_paragraphs} paragraphs processed.",
        user_message=f"Document analysis complete. {n_done} of {total_paragraphs} sections processed.",
        extra={
            "total_paragraphs": total_paragraphs,
            "completed_paragraphs": n_done,
        },
    )

    await ctx.upsert_progress("completed")
    logger.info("[%s] Coordinator: complete (%s/%s)", document_id, n_done, total_paragraphs)
    return success


# ---------------------------------------------------------------------------
# Shared paragraph loop
# ---------------------------------------------------------------------------

async def _process_paragraphs(
    ctx: ChunkingRunContext,
    page_data_list: list[tuple[int, str]],
    *,
    extraction_enabled: bool,
    threshold: float,
    critique_enabled: bool,
    max_retries: int,
    extraction_prompt_body: str | None,
    retry_extraction_prompt_body: str | None,
    critique_prompt_body: str | None,
    llm: Any,
    path_b_resources: Any | None,
) -> bool:
    """Iterate every paragraph across all pages.

    For each paragraph:
      1. Emit paragraph_start event.
      2. Create / fetch the shared HierarchicalChunk.
      3. Delegate to Path A or Path B for enrichment.
      4. Emit paragraph_complete / paragraph_failed.
    """
    doc_id = ctx.document_id
    doc_uuid = ctx.doc_uuid
    db = ctx.db

    for page_num, (current_page_number, page_md) in enumerate(page_data_list, start=1):
        paragraphs = split_paragraphs_from_markdown(page_md)
        logger.info(
            "[%s] Page %s (%s/%s): %s paragraphs",
            doc_id, current_page_number, page_num, ctx.total_pages, len(paragraphs),
        )

        for para_idx, para_data in enumerate(paragraphs):
            paragraph_text = para_data["text"] if isinstance(para_data, dict) else para_data
            section_path = para_data.get("section_path") if isinstance(para_data, dict) else None
            para_start_offset = para_data.get("start_offset") if isinstance(para_data, dict) else None
            para_id = f"{current_page_number}_{para_idx}"

            # Skip already done
            if para_id in ctx.results_paragraphs:
                existing = ctx.results_paragraphs[para_id]
                if existing.get("status") in ("passed", "skipped") or existing.get("facts"):
                    continue

            # --- paragraph_start event ---
            await ctx.emit(
                "paragraph_start",
                message=f"Starting paragraph {para_id}{' (Path B)' if not extraction_enabled else ''}.",
                user_message=f"Reading section {ctx.completed_count + 1} of {ctx.total_paragraphs}...",
                paragraph_id=para_id,
                extra={
                    "paragraph_text": paragraph_text[:2000],
                    "page_number": current_page_number,
                    "total_paragraphs": ctx.total_paragraphs,
                    "completed_paragraphs": ctx.completed_count,
                    "progress_percent": ctx.progress_percent,
                },
            )

            try:
                # ── SHARED: Create HierarchicalChunk ──────────────────────
                # Both paths get a persisted chunk row. Path A will update
                # extraction_status/critique_status/summary later.
                chunk = await db_handler.persist_chunk(
                    db, doc_uuid, current_page_number, para_idx, paragraph_text,
                    section_path=section_path,
                    start_offset_in_page=para_start_offset,
                    extraction_status="pending" if extraction_enabled else "skipped",
                    critique_status="pending" if extraction_enabled else "skipped",
                )

                # ── PATH-SPECIFIC ENRICHMENT ──────────────────────────────
                if extraction_enabled:
                    await path_a.process_paragraph(
                        ctx, chunk, para_id, paragraph_text, page_md,
                        section_path=section_path,
                        para_start_offset=para_start_offset,
                        page_number=current_page_number,
                        para_idx=para_idx,
                        threshold=threshold,
                        critique_enabled=critique_enabled,
                        max_retries=max_retries,
                        extraction_prompt_body=extraction_prompt_body,
                        retry_extraction_prompt_body=retry_extraction_prompt_body,
                        critique_prompt_body=critique_prompt_body,
                        llm=llm,
                    )
                else:
                    await path_b.process_paragraph(
                        ctx, chunk, para_id, paragraph_text,
                        section_path=section_path,
                        page_number=current_page_number,
                        para_idx=para_idx,
                        path_b_resources=path_b_resources,
                    )

                # ── paragraph complete ────────────────────────────────────
                await ctx.emit(
                    "paragraph_complete",
                    message=f"Paragraph {para_id} complete{' (Path B)' if not extraction_enabled else ''}.",
                    user_message=f"Section {ctx.completed_count} of {ctx.total_paragraphs} done.",
                    paragraph_id=para_id,
                )
                await ctx.upsert_progress("in_progress")

            except Exception as para_err:
                logger.error("[%s] Paragraph %s error: %s", doc_id, para_id, para_err, exc_info=True)
                await record_paragraph_error(ctx, para_id, para_err)
                await ctx.emit_progress(para_id, current_page_number, error=str(para_err))
                await ctx.upsert_progress("in_progress")

    return True
