"""Per-attempt fetch audit trail (Crawler).

``discovered_sources`` carries only the *latest* fetch verdict. This module
carries the history behind it: one append-only row per fetch attempt.

WHY IT MATTERS
    Latest-state-only means a bug that corrupts the verdict also erases the
    evidence that it did. The motivating class of defect is a transient 403
    on ``/robots.txt`` being recorded as a durable "disallow", which pins a
    host unreachable; with only ``last_fetch_status`` you cannot tell that
    from a host that is genuinely blocked, and you cannot prove a fix landed
    or catch it regressing. With the trail you can: the sequence
    ``[403, 200, 200]`` and the sequence ``[403, 403, 403]`` are different
    stories that collapse to the same single column today.

OWNERSHIP
    ``source_fetch_attempts`` is Crawler's, written only from here.
    ``discovered_sources`` is shared at COLUMN level under the DB seat's
    single-writer contract (ruled 2026-08-12): Crawler writes fetch columns,
    RAG writes discovery columns (incl. ``ingested``/``ingested_doc_id``),
    Sources writes ``curated_*``. Nothing in this module writes outside the
    Crawler columns.

The model lives here rather than in ``app/models.py`` deliberately —
``app/models.py`` is a shared file across all four RAG legs, and this table
has exactly one owner. Importing ``Base`` keeps it in the same metadata.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base


# Open vocabulary — mirrors the robots gate's tri-state plus an explicit
# error case. Not a DB enum (see migration note); validated here so a typo
# fails loudly at the call site instead of quietly polluting audit queries.
ROBOTS_DECISIONS = {"crawlable", "disallow_all", "unknown", "error"}


class SourceFetchAttempt(Base):
    """One fetch attempt against one discovered source. Append-only."""

    __tablename__ = "source_fetch_attempts"

    attempt_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    discovered_source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("discovered_sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    attempted_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # NULL = no HTTP response at all (timeout / DNS / connection refused).
    # Kept distinct from any 4xx on purpose — see module docstring.
    http_status = Column(Integer, nullable=True)
    bytes_downloaded = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)

    robots_decision = Column(Text, nullable=True)

    content_hash_before = Column(Text, nullable=True)
    content_hash_after = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)
    run_id = Column(Text, nullable=True)


async def record_fetch_attempt(
    db: AsyncSession,
    *,
    discovered_source_id: uuid.UUID,
    http_status: int | None = None,
    bytes_downloaded: int | None = None,
    latency_ms: int | None = None,
    robots_decision: str | None = None,
    content_hash_before: str | None = None,
    content_hash_after: str | None = None,
    error_message: str | None = None,
    run_id: str | None = None,
) -> SourceFetchAttempt:
    """Append one attempt row.

    Never updates an existing row — the trail is the point. Callers pass
    whatever they know; every field except the source id is optional, since
    a network-level failure legitimately has no status, no bytes and no hash.

    Raises ValueError on an unknown ``robots_decision`` so a typo surfaces at
    the call site rather than silently splitting audit aggregates.
    """
    if robots_decision is not None and robots_decision not in ROBOTS_DECISIONS:
        raise ValueError(
            f"robots_decision must be one of {sorted(ROBOTS_DECISIONS)}, "
            f"got {robots_decision!r}"
        )

    row = SourceFetchAttempt(
        discovered_source_id=discovered_source_id,
        attempted_at=datetime.utcnow(),
        http_status=http_status,
        bytes_downloaded=bytes_downloaded,
        latency_ms=latency_ms,
        robots_decision=robots_decision,
        content_hash_before=content_hash_before,
        content_hash_after=content_hash_after,
        error_message=error_message,
        run_id=run_id,
    )
    db.add(row)
    await db.flush()
    return row
