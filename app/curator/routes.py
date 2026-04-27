"""FastAPI router for /sources/* — the curator's HTTP surface.

Wired into app/main.py via ``app.include_router(curator_router)``.
Three callers in production:

* **mobius-web-scraper** worker — POST /sources/upsert per URL it
  visits or links to. Both 200s and 4xx get persisted.
* **mobius-chat** ReAct loop — GET /sources/search for
  ``lookup_authoritative_sources`` tool; POST /sources/{id}/ingest
  for the on-demand ``ingest_url`` tool.
* **operator UI** — /sources/{id}/curate to flip status + add notes.

Auth: same admin-key pattern as the rest of /admin endpoints. Public
GETs (search, stats) are read-only and harmless; no auth there yet.
"""
from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.curator import service as curator_service
from app.database import AsyncSessionLocal
from app.models import DiscoveredSource

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sources", tags=["curator"])


# ── DB dependency (mirrors get_db in main.py) ────────────────────────


async def _get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


# ── Request/response models ──────────────────────────────────────────


class UpsertSourceBody(BaseModel):
    """Single source upsert. All fields except ``url`` are optional —
    sparse callers (e.g. sitemap parser that only knows URL exists)
    work fine. Hint fields override the URL-classifier inference.
    """
    url: str
    discovered_via: Optional[str] = Field(
        None, description="'sitemap' | 'bfs_link' | 'manual' | 'curator'"
    )
    seed_url: Optional[str] = None
    depth_from_seed: Optional[int] = None
    scrape_job_id: Optional[str] = None
    fetch_status: Optional[int] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    content_hash: Optional[str] = None
    payer_hint: Optional[str] = None
    state_hint: Optional[str] = None
    program_hint: Optional[str] = None
    authority_hint: Optional[str] = None


class CurateBody(BaseModel):
    curation_status: Optional[str] = Field(
        None, description="'auto' | 'canonical' | 'noise' | 'stale' | 'needs_auth'"
    )
    curated_authority_level: Optional[str] = None
    topic_tags: Optional[list[str]] = None
    notes: Optional[str] = None
    by: Optional[str] = None


class SourceOut(BaseModel):
    """Wire format for /sources/* responses. Mirrors the model but
    omits the FK plumbing that's not useful to callers.
    """
    id: UUID
    url: str
    host: str
    path: str
    payer: Optional[str]
    state: Optional[str]
    program: Optional[str]
    inferred_authority_level: Optional[str]
    curated_authority_level: Optional[str]
    effective_authority_level: Optional[str]  # computed: curated || inferred
    topic_tags: Optional[list[str]]
    content_kind: str
    extension: Optional[str]
    last_seen_at: str
    last_fetch_status: Optional[int]
    last_fetch_at: Optional[str]
    content_hash: Optional[str]
    content_changed_at: Optional[str]
    ingested: bool
    ingested_doc_id: Optional[UUID]
    discovered_via: Optional[str]
    curation_status: str

    @classmethod
    def from_row(cls, r: DiscoveredSource) -> "SourceOut":
        return cls(
            id=r.id,
            url=r.url,
            host=r.host,
            path=r.path,
            payer=r.payer,
            state=r.state,
            program=r.program,
            inferred_authority_level=r.inferred_authority_level,
            curated_authority_level=r.curated_authority_level,
            effective_authority_level=r.curated_authority_level or r.inferred_authority_level,
            topic_tags=r.topic_tags,
            content_kind=r.content_kind,
            extension=r.extension,
            last_seen_at=r.last_seen_at.isoformat() if r.last_seen_at else None,
            last_fetch_status=r.last_fetch_status,
            last_fetch_at=r.last_fetch_at.isoformat() if r.last_fetch_at else None,
            content_hash=r.content_hash,
            content_changed_at=r.content_changed_at.isoformat() if r.content_changed_at else None,
            ingested=r.ingested,
            ingested_doc_id=r.ingested_doc_id,
            discovered_via=r.discovered_via,
            curation_status=r.curation_status,
        )


# ── Routes ───────────────────────────────────────────────────────────


@router.post("/upsert", response_model=SourceOut)
async def upsert_endpoint(
    body: UpsertSourceBody,
    db: AsyncSession = Depends(_get_db),
):
    """Insert or update a discovered source by URL. Idempotent."""
    row = await curator_service.upsert_source(
        db,
        url=body.url,
        discovered_via=body.discovered_via,
        seed_url=body.seed_url,
        depth_from_seed=body.depth_from_seed,
        scrape_job_id=body.scrape_job_id,
        fetch_status=body.fetch_status,
        content_type=body.content_type,
        content_length=body.content_length,
        content_hash=body.content_hash,
        payer_hint=body.payer_hint,
        state_hint=body.state_hint,
        program_hint=body.program_hint,
        authority_hint=body.authority_hint,
    )
    await db.commit()
    return SourceOut.from_row(row)


class BulkUpsertBody(BaseModel):
    """Many-at-once upsert for the scraper to batch its writes.
    Single transaction commit — partial failures roll back as a unit
    so the scraper's view of "what got persisted" stays consistent.
    """
    sources: list[UpsertSourceBody] = Field(..., max_length=1000)


class BulkUpsertOut(BaseModel):
    inserted_or_updated: int
    failed: int


@router.post("/bulk_upsert", response_model=BulkUpsertOut)
async def bulk_upsert_endpoint(
    body: BulkUpsertBody,
    db: AsyncSession = Depends(_get_db),
):
    """Batch upsert. Used by curator-side bulk imports (e.g. seeding
    the table from sitemap_data_v0.json) and by the scraper at the end
    of a tree-scan run.
    """
    ok = 0
    fail = 0
    for src in body.sources:
        try:
            await curator_service.upsert_source(
                db,
                url=src.url,
                discovered_via=src.discovered_via,
                seed_url=src.seed_url,
                depth_from_seed=src.depth_from_seed,
                scrape_job_id=src.scrape_job_id,
                fetch_status=src.fetch_status,
                content_type=src.content_type,
                content_length=src.content_length,
                content_hash=src.content_hash,
                payer_hint=src.payer_hint,
                state_hint=src.state_hint,
                program_hint=src.program_hint,
                authority_hint=src.authority_hint,
            )
            ok += 1
        except Exception as exc:
            logger.exception("bulk_upsert: failed for url=%s: %s", src.url, exc)
            fail += 1
    await db.commit()
    return BulkUpsertOut(inserted_or_updated=ok, failed=fail)


@router.get("/search", response_model=list[SourceOut])
async def search_endpoint(
    payer: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    program: Optional[str] = Query(None),
    authority_level: Optional[str] = Query(None),
    topic: Optional[str] = Query(None, description="Exact match against topic_tags JSONB array"),
    host: Optional[str] = Query(None, description="Exact host match (e.g. www.samhsa.gov). Used by Sources UI tree view."),
    q: Optional[str] = Query(
        None,
        description=(
            "Free-text relevance query (Phase 13.5d). Ranked by Postgres "
            "ts_rank over the search_vector index covering payer/state/host/"
            "path-slugs/authority/notes. Use this when the planner has a "
            "topic keyword that may not exactly match a populated tag. "
            "Combinable with the exact filters above."
        ),
    ),
    curation_status: Optional[str] = Query(None),
    ingested: Optional[bool] = Query(None),
    only_reachable: bool = Query(True, description="Hide 404/403/etc."),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(_get_db),
):
    """Search the URL registry. Powers chat ReAct's
    ``lookup_authoritative_sources`` tool.
    """
    rows = await curator_service.search_sources(
        db,
        payer=payer,
        state=state,
        program=program,
        authority_level=authority_level,
        topic=topic,
        q=q,
        host=host,
        curation_status=curation_status,
        ingested=ingested,
        only_reachable=only_reachable,
        limit=limit,
    )
    return [SourceOut.from_row(r) for r in rows]


@router.post("/{source_id}/curate", response_model=SourceOut)
async def curate_endpoint(
    source_id: UUID,
    body: CurateBody,
    db: AsyncSession = Depends(_get_db),
):
    """Apply human curation. Used by the operator UI."""
    try:
        row = await curator_service.curate_source(
            db,
            source_id=source_id,
            curation_status=body.curation_status,
            curated_authority_level=body.curated_authority_level,
            topic_tags=body.topic_tags,
            notes=body.notes,
            by=body.by,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    await db.commit()
    return SourceOut.from_row(row)


@router.get("/stats")
async def stats_endpoint(db: AsyncSession = Depends(_get_db)):
    """Aggregate counts by status / host / kind / ingested.
    Cheap — single query plan, runs in tens of ms on 100k+ rows.
    """
    return await curator_service.stats(db)


@router.get("/corpus_by_host")
async def corpus_by_host_endpoint(db: AsyncSession = Depends(_get_db)):
    """Count documents in the corpus, grouped by host extracted from
    Document.file_path or source_metadata.source_url.

    Why this exists: the registry's ingested counter only fires when
    we have a discovered_sources row matching the URL. Many docs were
    imported BEFORE the curator existed (manual upload, pre-curator
    scrape) and don't have registry rows. The Sources UI needs to
    show "this entity has 700 docs in corpus" alongside "0 in registry"
    so operators don't see misleading 0% coverage.
    """
    from sqlalchemy import func, text
    # Use raw SQL: extract the host from file_path (URL or GCS path)
    # via regexp_substr. Cheap aggregate across <2000 docs at v1 scale.
    # Host source priority:
    #   1. file_path if it's a URL (HTML imports — file_path = source URL)
    #   2. source_metadata->>'source_url' (PDF imports — scraper put URL here)
    #   3. NULL otherwise (manual UI uploads have no URL provenance)
    #
    # ``published_at`` is computed (lives in publish_events /
    # rag_published_embeddings), not a column on documents. EXISTS
    # subquery against rag_published_embeddings — that's what gates
    # "queryable in chat" anyway.
    # Three-way URL-source priority for host extraction:
    #   1. file_path if it's a URL (HTML imports — file_path = source URL)
    #   2. source_metadata->>'source_url' (PDF imports going forward — set on import)
    #   3. discovered_sources.host where ingested_doc_id = d.id
    #      (recovers historic PDFs imported BEFORE we stored source_url
    #       in source_metadata; works as long as the curator has a
    #       row pointing at the doc, which mark_ingested guarantees
    #       for new imports + most existing scrape-driven docs)
    sql = text("""
        WITH docs AS (
            SELECT
                d.id,
                COALESCE(
                    CASE WHEN d.file_path LIKE 'http%'
                         THEN substring(d.file_path FROM '^https?://([^/]+)')
                    END,
                    substring(d.source_metadata->>'source_url' FROM '^https?://([^/]+)'),
                    (SELECT ds.host FROM discovered_sources ds
                     WHERE ds.ingested_doc_id = d.id LIMIT 1)
                ) AS host
            FROM documents d
        )
        SELECT
            host,
            COUNT(*) AS doc_count,
            SUM(
                CASE WHEN EXISTS (
                    SELECT 1 FROM rag_published_embeddings rpe
                    WHERE rpe.document_id = docs.id
                ) THEN 1 ELSE 0 END
            ) AS published_count
        FROM docs
        GROUP BY host
        ORDER BY doc_count DESC NULLS LAST
    """)
    result = await db.execute(sql)
    rows = result.all()
    return {
        "by_host": {
            (r.host or "(no_host)"): {
                "docs": r.doc_count,
                "published": r.published_count or 0,
            }
            for r in rows
        }
    }
