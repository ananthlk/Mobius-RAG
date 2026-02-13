"""
Chunking worker error helpers.

Centralises the "classify -> log_error -> set results_paragraphs[para_id] = failed"
pattern so Path A, Path B, and any retry logic all use one function.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.error_tracker import classify_error, log_error
from app.worker.context import ChunkingRunContext

logger = logging.getLogger(__name__)


async def record_paragraph_error(
    ctx: ChunkingRunContext,
    para_id: str,
    error: Exception | str,
    *,
    context_label: str = "other",
    stage_label: str = "other",
    error_details: dict | None = None,
) -> None:
    """Classify, persist, and record a paragraph-level failure.

    * Calls :func:`classify_error` to get (severity, stage).
    * Calls :func:`log_error` to persist a ``ProcessingError`` row.
    * Stores ``{ status: "failed", error: ... }`` in ``ctx.results_paragraphs``.
    * Emits a ``paragraph_failed`` event with both technical and user messages.
    """
    import json

    err_str = str(error)

    # --- Classify ---
    if context_label == "extraction":
        error_type = "json_parse_error" if isinstance(error, json.JSONDecodeError) else "llm_failure"
    elif context_label == "persistence":
        error_type = "persistence_error"
    else:
        error_type = "other"

    severity, stage = classify_error(error_type, error)

    # --- Persist ProcessingError ---
    try:
        await log_error(
            db=ctx.db,
            document_id=ctx.document_id,
            paragraph_id=para_id,
            error_type=error_type,
            severity=severity,
            error_message=err_str,
            error_details=error_details or {"stage": stage_label},
            stage=stage,
        )
    except Exception as log_exc:
        logger.error("[errors] log_error failed: %s", log_exc, exc_info=True)

    # --- Record in context ---
    ctx.record_paragraph_result(para_id, status="failed", error=err_str[:500])

    # --- Emit event ---
    try:
        short = err_str[:80]
        await ctx.emit(
            "paragraph_failed",
            message=f"Paragraph {para_id} failed: {short}",
            user_message=f"Section {para_id} could not be processed.",
            paragraph_id=para_id,
            extra={"error_type": error_type, "severity": severity},
        )
    except Exception as emit_exc:
        logger.error("[errors] emit paragraph_failed: %s", emit_exc, exc_info=True)
