"""DB-backed batch loader for passenger_tables.py's resolve function.

Kept separate from passenger_tables.py deliberately -- that module's purity
(no DB, no I/O) is the whole reason it's unit-testable against 27 cases with
zero real rows. This file is the thin, async, DB-touching half: two batched
queries (never N+1 -- one IN over all breadcrumb table_ids, one over all
distinct (document_id, page_number) pairs needing the fallback), which then
hand plain dicts to the pure resolver.

Fail-open, same posture as the rest of the Table Capture program: a query
failure degrades to "no passenger tables for this turn", never a raised
exception that would take the answer down. Table content is additive to an
already-valid prose answer, never load-bearing for it -- unlike
_resolve_document_names (synthesis.py), which re-raises on a technical
failure because a missing document name is a real defect to chase. A missing
passenger table is not.
"""

from __future__ import annotations

import logging

from sqlalchemy import text as _sql
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retriever.passenger_tables import (
    PassengerTable,
    extract_breadcrumbs,
    resolve_passenger_tables,
)

logger = logging.getLogger(__name__)

_MAX_TECHNICAL_RETRIES = 1  # same standing principle as synthesis.py's _resolve_document_names


async def _fetch_tables_by_id(db: AsyncSession, table_ids: list[str]) -> dict[str, dict]:
    if not table_ids:
        return {}
    for attempt in range(_MAX_TECHNICAL_RETRIES + 1):
        try:
            result = await db.execute(
                _sql(
                    "SELECT id::text AS id, document_id::text AS document_id, page_number, "
                    "grid, anchor, coverage, caption, strategy, n_rows, n_cols, is_clean "
                    "FROM document_tables WHERE id = ANY(CAST(:ids AS uuid[]))"
                ),
                {"ids": table_ids},
            )
            rows = result.mappings().all()
            return {row["id"]: dict(row) for row in rows}
        except Exception as exc:
            if attempt < _MAX_TECHNICAL_RETRIES:
                logger.warning("passenger_tables_loader: breadcrumb fetch failed, retrying once: %s", exc)
                continue
            logger.warning(
                "passenger_tables_loader: breadcrumb fetch failed after retry, "
                "degrading to no passenger tables (fail-open): %s", exc,
            )
            return {}
    return {}


async def _fetch_tables_by_page(
    db: AsyncSession, pairs: list[tuple[str, int]]
) -> dict[tuple[str, int], list[dict]]:
    if not pairs:
        return {}
    doc_ids = [p[0] for p in pairs]
    page_numbers = [p[1] for p in pairs]
    for attempt in range(_MAX_TECHNICAL_RETRIES + 1):
        try:
            result = await db.execute(
                _sql(
                    "SELECT id::text AS id, document_id::text AS document_id, page_number, "
                    "grid, anchor, coverage, caption, strategy, n_rows, n_cols, is_clean "
                    "FROM document_tables "
                    "WHERE (document_id, page_number) IN ("
                    "  SELECT UNNEST(CAST(:doc_ids AS uuid[])), UNNEST(CAST(:pages AS int[]))"
                    ")"
                ),
                {"doc_ids": doc_ids, "pages": page_numbers},
            )
            rows = result.mappings().all()
            break
        except Exception as exc:
            if attempt < _MAX_TECHNICAL_RETRIES:
                logger.warning("passenger_tables_loader: page-proximity fetch failed, retrying once: %s", exc)
                continue
            logger.warning(
                "passenger_tables_loader: page-proximity fetch failed after retry, "
                "degrading to no passenger tables (fail-open): %s", exc,
            )
            return {}

    out: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        key = (row["document_id"], row["page_number"])
        out.setdefault(key, []).append(dict(row))
    return out


async def load_passenger_tables(db: AsyncSession, citations: list) -> list[PassengerTable]:
    """Batch-load and resolve passenger tables for a compiled citation list.
    One query per path (never per-citation), then delegates the actual
    breadcrumb/page-proximity/dedup logic to the pure resolver so that logic
    stays covered by passenger_tables.py's unit tests rather than duplicated
    here.
    """
    breadcrumb_table_ids: set[str] = set()
    fallback_pairs: set[tuple[str, int]] = set()

    for citation in citations:
        chunk_id = getattr(citation, "chunk_id", None)
        text = getattr(citation, "text", None)
        if not chunk_id or not text:
            continue

        crumbs = extract_breadcrumbs(text, chunk_id)
        if crumbs:
            breadcrumb_table_ids.update(c.table_id for c in crumbs)
            continue

        document_id = getattr(citation, "document_id", None)
        page_number = getattr(citation, "page_number", None)
        if document_id and page_number is not None:
            fallback_pairs.add((document_id, page_number))

    try:
        table_by_id = await _fetch_tables_by_id(db, list(breadcrumb_table_ids))
        tables_by_page = await _fetch_tables_by_page(db, list(fallback_pairs))
    except Exception as exc:
        # Belt-and-suspenders: the two helpers above already fail-open
        # internally, but a batching bug here must not either. Table content
        # is additive, never load-bearing (program-wide rule).
        logger.warning("passenger_tables_loader: unexpected failure, degrading to no passenger tables: %s", exc)
        return []

    return resolve_passenger_tables(
        citations,
        fetch_table=table_by_id.get,
        fetch_tables_for_page=lambda doc_id, page: tables_by_page.get((doc_id, page), []),
    )
