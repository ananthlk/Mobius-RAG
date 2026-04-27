"""Curator service — upsert + search + curate logic.

These functions are the SQL-touching layer. The HTTP routes
(``app.curator.routes``) call them; so does the freshness worker; so
will the future scraper-side push hook. Keeps SQL out of FastAPI
handlers.

Idempotency contract:
* ``upsert_source(url=...)`` is safe to call any number of times for
  the same URL. The first call inserts; subsequent calls UPDATE only
  fields actually present in the payload, leaving curation_status,
  curated_authority_level, etc. untouched (operator overrides win).
* The caller does not need to do exists-check first.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.curator.classifier import classify_url
from app.models import DiscoveredSource


# ── Upsert ───────────────────────────────────────────────────────────


async def upsert_source(
    db: AsyncSession,
    *,
    url: str,
    discovered_via: str | None = None,
    seed_url: str | None = None,
    depth_from_seed: int | None = None,
    scrape_job_id: str | None = None,
    fetch_status: int | None = None,
    content_type: str | None = None,
    content_length: int | None = None,
    content_hash: str | None = None,
    payer_hint: str | None = None,
    state_hint: str | None = None,
    program_hint: str | None = None,
    authority_hint: str | None = None,
) -> DiscoveredSource:
    """Insert or update a discovered_sources row.

    On first sight of a URL we run the URL classifier to fill in
    inferred_authority_level, payer, state, content_kind, extension.
    Caller-provided ``*_hint`` arguments override the classifier's
    inference (e.g., scraper knows the seed URL's payer better than
    the classifier does for off-domain links).

    On subsequent sightings we update liveness fields (last_seen_at,
    last_fetch_status, fetch_attempt_count) and the content
    fingerprint (content_hash + content_changed_at). Curation fields
    are NEVER overwritten by upsert — only by the explicit
    /sources/{id}/curate endpoint.
    """
    now = datetime.utcnow()
    result = await db.execute(
        select(DiscoveredSource).where(DiscoveredSource.url == url)
    )
    row = result.scalar_one_or_none()

    if row is None:
        # First sight — full classification.
        cls = classify_url(url)
        row = DiscoveredSource(
            url=url,
            host=cls["host"],
            path=cls["path"],
            payer=payer_hint or cls["payer"],
            state=state_hint or cls["state"],
            program=program_hint,
            inferred_authority_level=authority_hint or cls["inferred_authority_level"],
            content_kind=cls["content_kind"],
            extension=cls["extension"],
            first_seen_at=now,
            last_seen_at=now,
            discovered_via=discovered_via,
            seed_url=seed_url,
            depth_from_seed=depth_from_seed,
            scrape_job_id=scrape_job_id,
        )
        db.add(row)

    # Always-update fields (liveness)
    row.last_seen_at = now
    if fetch_status is not None:
        # Hash diff — only mark content_changed_at when we actually
        # fetched a body and computed a new hash.
        if content_hash and content_hash != row.content_hash:
            row.content_changed_at = now
        row.last_fetch_status = fetch_status
        row.last_fetch_at = now
        row.fetch_attempt_count = (row.fetch_attempt_count or 0) + 1
        if content_type:
            row.content_type = content_type
        if content_length is not None:
            row.content_length = content_length
        if content_hash:
            row.content_hash = content_hash

    await db.flush()
    return row


# ── Mark-ingested helper ─────────────────────────────────────────────


async def mark_ingested(
    db: AsyncSession,
    *,
    url: str,
    document_id: UUID,
) -> DiscoveredSource | None:
    """Link a discovered_sources row to a documents row after ingest.

    Called from the import-from-gcs / import-from-html paths so the
    registry knows which URLs are now in the indexed corpus.

    Returns the updated row, or None if the URL wasn't in the
    registry (which is fine — back-compat for pre-curator imports
    that don't pass a URL).
    """
    result = await db.execute(
        select(DiscoveredSource).where(DiscoveredSource.url == url)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None
    row.ingested = True
    row.ingested_doc_id = document_id
    row.ingested_at = datetime.utcnow()
    await db.flush()
    return row


# ── Search (used by ReAct lookup_authoritative_sources) ──────────────


async def search_sources(
    db: AsyncSession,
    *,
    payer: str | None = None,
    state: str | None = None,
    program: str | None = None,
    authority_level: str | None = None,
    topic: str | None = None,
    q: str | None = None,
    host: str | None = None,
    curation_status: str | None = None,
    ingested: bool | None = None,
    only_reachable: bool = True,
    limit: int = 50,
) -> Sequence[DiscoveredSource]:
    """Search the discovered_sources registry.

    Two retrieval modes, can be combined:

    * **Filter mode** — exact predicates on payer/state/program/
      authority_level/topic_tags. Returns rows that strictly match.

    * **Relevance mode** (Phase 13.5d) — when ``q`` is set, ranks
      results by Postgres ``ts_rank`` over the GIN-indexed
      ``search_vector`` column. The vector covers payer, state,
      program, host, path slugs, authority levels, topic_tags, and
      curation_notes — so a planner can pass ``q="dental plan
      transition"`` and surface URLs whose path contains those tokens
      WITHOUT needing topic_tags pre-populated.

    Combining both is the common case: ``payer="Sunshine Health"``
    narrows the candidate set, ``q="medicare preauth"`` ranks them.

    Default behavior (``only_reachable=True``) excludes 404/403 rows —
    ReAct should never recommend a URL it knows is broken.

    Ranking precedence:
      * If ``q`` set: ts_rank DESC, then ingested DESC, last_seen DESC
      * Else: canonical DESC, ingested DESC, last_seen DESC
    """
    conds: list = []
    if host:
        # Exact host match. Used by the Sources UI tree view to scope
        # the per-entity render. Cheap (host has its own index).
        conds.append(DiscoveredSource.host == host)
    if payer:
        conds.append(DiscoveredSource.payer == payer)
    if state:
        conds.append(DiscoveredSource.state == state)
    if program:
        conds.append(DiscoveredSource.program == program)
    if authority_level:
        # Match either curated override OR inferred — caller doesn't
        # care which one set the level.
        conds.append(or_(
            DiscoveredSource.curated_authority_level == authority_level,
            and_(
                DiscoveredSource.curated_authority_level.is_(None),
                DiscoveredSource.inferred_authority_level == authority_level,
            ),
        ))
    if curation_status:
        conds.append(DiscoveredSource.curation_status == curation_status)
    if ingested is not None:
        conds.append(DiscoveredSource.ingested == ingested)
    if topic:
        # JSONB array contains topic — uses the GIN index from the
        # migration. Only matches when topic_tags has been populated
        # (LLM categorization or manual curation). Use ``q=`` for
        # imperfect-match fallback.
        conds.append(DiscoveredSource.topic_tags.contains([topic]))
    if only_reachable:
        # Treat -1 (network error) and 4xx as unreachable. Allow
        # last_fetch_status IS NULL (never fetched yet) since those
        # might still be reachable.
        conds.append(or_(
            DiscoveredSource.last_fetch_status.is_(None),
            and_(
                DiscoveredSource.last_fetch_status >= 200,
                DiscoveredSource.last_fetch_status < 400,
            ),
        ))

    # Relevance mode: build ts_query, AND it into the WHERE, ORDER by rank
    if q and q.strip():
        # plainto_tsquery is permissive: handles arbitrary user text
        # without requiring boolean operators. Stop words dropped
        # automatically; common-stem matching via the english config.
        ts_query = func.plainto_tsquery("english", q.strip())
        # Use literal column reference since search_vector is a
        # generated column not declared in the SQLAlchemy model.
        from sqlalchemy import literal_column
        sv = literal_column("search_vector")
        conds.append(sv.op("@@")(ts_query))
        rank = func.ts_rank(sv, ts_query)
        order_clauses = [
            rank.desc(),
            DiscoveredSource.ingested.desc(),
            DiscoveredSource.last_seen_at.desc(),
        ]
    else:
        order_clauses = [
            (DiscoveredSource.curation_status == "canonical").desc(),
            DiscoveredSource.ingested.desc(),
            DiscoveredSource.last_seen_at.desc(),
        ]

    stmt = (
        select(DiscoveredSource)
        .where(and_(*conds) if conds else True)
        .order_by(*order_clauses)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


# ── Curate ───────────────────────────────────────────────────────────


_VALID_CURATION_STATUS = {"auto", "canonical", "noise", "stale", "needs_auth"}


async def curate_source(
    db: AsyncSession,
    *,
    source_id: UUID,
    curation_status: str | None = None,
    curated_authority_level: str | None = None,
    topic_tags: list[str] | None = None,
    notes: str | None = None,
    by: str | None = None,
) -> DiscoveredSource:
    """Apply human curation to a row. All args optional; only the
    fields actually passed are updated.

    Raises ValueError on bad curation_status — caller should turn it
    into a 400.
    """
    if curation_status is not None and curation_status not in _VALID_CURATION_STATUS:
        raise ValueError(
            f"curation_status must be one of {sorted(_VALID_CURATION_STATUS)}, got {curation_status!r}"
        )

    result = await db.execute(
        select(DiscoveredSource).where(DiscoveredSource.id == source_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise LookupError(f"discovered_sources id={source_id} not found")

    if curation_status is not None:
        row.curation_status = curation_status
    if curated_authority_level is not None:
        # Empty string means "clear override, fall back to inferred".
        row.curated_authority_level = curated_authority_level or None
    if topic_tags is not None:
        row.topic_tags = topic_tags
    if notes is not None:
        row.curation_notes = notes
    row.curated_by = by
    row.curated_at = datetime.utcnow()
    await db.flush()
    return row


# ── Stats helper for the /sources/stats endpoint ─────────────────────


async def stats(db: AsyncSession) -> dict[str, Any]:
    """Aggregate counts for an operator dashboard. Cheap query —
    counted at the DB. Five rows, runs in <50ms even on 100k+ records.
    """
    by_status = await db.execute(
        select(DiscoveredSource.curation_status, func.count())
        .group_by(DiscoveredSource.curation_status)
    )
    by_host = await db.execute(
        select(DiscoveredSource.host, func.count())
        .group_by(DiscoveredSource.host)
        .order_by(func.count().desc())
        .limit(20)
    )
    by_kind = await db.execute(
        select(DiscoveredSource.content_kind, func.count())
        .group_by(DiscoveredSource.content_kind)
    )
    ingested = await db.execute(
        select(DiscoveredSource.ingested, func.count())
        .group_by(DiscoveredSource.ingested)
    )
    return {
        "by_curation_status": {k: v for k, v in by_status.all()},
        "by_host":             {k: v for k, v in by_host.all()},
        "by_content_kind":     {k: v for k, v in by_kind.all()},
        "by_ingested":         {bool(k): v for k, v in ingested.all()},
    }
