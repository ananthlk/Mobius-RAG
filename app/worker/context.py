"""
Chunking run context.

Shared state and convenience methods that both Path A and Path B use for
emitting events, updating progress, and sending status messages.

Every event includes *message* (technical) and *user_message* (user-facing,
suitable for chat / reasoning-style UI).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.worker.db import upsert_chunking_result, write_event

logger = logging.getLogger(__name__)


@dataclass
class ChunkingRunContext:
    """Holds run-scoped state for one document chunking pass."""

    db: AsyncSession
    document_id: str
    doc_uuid: UUID
    job_id: str

    total_paragraphs: int = 0
    total_pages: int = 0
    results_paragraphs: dict[str, Any] = field(default_factory=dict)

    # When True, path_b.finalise skips _trigger_lexicon_cleanup (the HTTP call
    # to LEXICON_MAINTENANCE_URL). Set for inline instant uploads where that
    # ~6s round-trip is unacceptable in the hot path. Candidate mining still
    # happens at corpus-promotion / batch re-process time.
    skip_lexicon_cleanup: bool = False

    # ------------------------------------------------------------------ emit
    async def emit(
        self,
        event_type: str,
        *,
        message: str = "",
        user_message: str = "",
        paragraph_id: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """Persist a ChunkingEvent with dual messages and optional extras.

        Per-paragraph events (``paragraph_start`` / ``paragraph_complete``)
        are SKIPPED by default — they fire 2× per paragraph on a 589k+ row
        hot table, dominating the per-paragraph commit budget. Cost
        analysis 2026-04-30 showed paragraph time was 0.7s on healthy
        days and 25-30s on contended days; per-paragraph chunking_events
        writes are a major contention source.

        Set ``EMIT_PARAGRAPH_EVENTS=1`` to re-enable for debugging.
        Job-level events (``job_start``, ``chunking_complete``,
        ``embedding_*``, ``status_message``) are always emitted.
        """
        import os as _os
        if event_type in ("paragraph_start", "paragraph_complete"):
            if _os.environ.get("EMIT_PARAGRAPH_EVENTS", "").strip() not in ("1", "true", "yes"):
                return
        data: dict[str, Any] = {
            "message": message,
            "user_message": user_message,
        }
        if paragraph_id is not None:
            data["paragraph_id"] = paragraph_id
        if extra:
            data.update(extra)
        await write_event(self.db, self.doc_uuid, event_type, data)
        logger.debug(
            "[%s] emit %s: %s", self.document_id, event_type, message[:120]
        )

    # -------------------------------------------------------------- send_status
    async def send_status(
        self,
        message: str,
        user_message: str | None = None,
    ) -> None:
        """Shorthand for emitting a ``status_message`` event."""
        await self.emit(
            "status_message",
            message=message,
            user_message=user_message or message,
        )

    # --------------------------------------------------------- upsert_progress
    async def upsert_progress(self, status: str = "in_progress") -> bool:
        """Persist current results_paragraphs into ChunkingResult."""
        return await upsert_chunking_result(
            self.db,
            self.doc_uuid,
            self.results_paragraphs,
            status=status,
            total_paragraphs=self.total_paragraphs,
            total_pages=self.total_pages,
        )

    # --------------------------------------------------- paragraph bookkeeping
    def record_paragraph_result(
        self,
        para_id: str,
        *,
        status: str,
        facts: list | None = None,
        summary: str | None = None,
        critique: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Store a paragraph's outcome in results_paragraphs (in-memory)."""
        entry: dict[str, Any] = {"paragraph_id": para_id, "status": status}
        if facts is not None:
            entry["facts"] = facts
        if summary is not None:
            entry["summary"] = summary
        if critique is not None:
            entry["critique"] = critique
        if error is not None:
            entry["error"] = error
        self.results_paragraphs[para_id] = entry

    # ----------------------------------------------------- progress helpers
    @property
    def completed_count(self) -> int:
        return len(self.results_paragraphs)

    @property
    def progress_percent(self) -> float:
        if self.total_paragraphs <= 0:
            return 0.0
        return self.completed_count / self.total_paragraphs * 100

    async def emit_progress(
        self,
        para_id: str,
        current_page: int,
        *,
        error: str | None = None,
    ) -> None:
        """Emit a ``progress_update`` event with current counts."""
        n = self.completed_count
        pct = self.progress_percent
        extra: dict[str, Any] = {
            "current_paragraph": para_id,
            "current_page": current_page,
            "total_pages": self.total_pages,
            "total_paragraphs": self.total_paragraphs,
            "completed_paragraphs": n,
            "progress_percent": pct,
        }
        if error:
            extra["error"] = error
        await self.emit(
            "progress_update",
            message=f"{n}/{self.total_paragraphs} paragraphs processed.",
            user_message=f"Processed {n} of {self.total_paragraphs} sections so far.",
            extra=extra,
        )
