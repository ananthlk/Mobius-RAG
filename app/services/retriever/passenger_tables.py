"""Module 1 of the Table Capture program (Mobius/docs/TABLE_CAPTURE_PROGRAM.md,
stage 3) -- the passenger model. A retrieved prose chunk that carries a table
breadcrumb pulls its table row(s) into the result.

Two attachment paths, in priority order:

1. Breadcrumb (exact). Chunk text carries Sourcing's inline marker naming the
   table_id directly -- authoritative, no ambiguity.
2. Page-proximity (fallback). Ananth's go-ahead, 2026-08-19: a retrieved
   chunk with NO breadcrumb -- especially a numeric-only chunk
   (`text !~ '[A-Za-z]'`, the strongest signal the answer is tabular) -- joins
   `document_tables` on (document_id, page_number). This exists because a
   table isn't always the SOURCE of the chunk that references it in prose;
   sometimes the retrieved chunk IS a stray table cell (a row/column fragment
   that survived the min-substance guard) sitting on the same page as the
   real table, with no breadcrumb because it's not the excision's own
   breadcrumb chunk.

Dedup is per table_id ACROSS BOTH paths, non-negotiable: 14,446 numeric-only
chunks in the table-bearing documents collapse onto only 137 pages (measured
live, 2026-08-19). Without cross-path dedup, several cells retrieved off one
page would attach the same table dozens of times and blow the citation
budget on duplicates of a single table.

Contract (Sourcing writes / Retriever reads):
  breadcrumb in chunk text:  [Table: <caption> · →document_tables:<uuid>]
  document_tables.anchor  =  {chunk_id/section, page, bbox}

Pure module -- no DB, no I/O, no network, same posture as Sourcing's
capture_page_tables: unit-testable without a corpus, cannot take the answer
path down. Both `fetch_table` and `fetch_tables_for_page` are injected
callables; the DB-backed batched loader lives in passenger_tables_loader.py,
which does one query per path and hands this module plain dicts.

Fail-open throughout: a chunk with no breadcrumb and no page match, a
malformed breadcrumb, or a table_id/page lookup that comes back empty must
never break the answer -- it just carries no passenger table. Table content
is additive to an already-valid prose answer, never load-bearing for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# Sourcing's exact breadcrumb shape (S-4): "[Table: <caption> · →document_tables:<uuid>]"
# UUID captured loosely (hex + hyphens) rather than a strict UUID4 pattern --
# this only needs to round-trip whatever Sourcing writes, not validate it.
_BREADCRUMB_RE = re.compile(
    r"\[Table:\s*(?P<caption>[^\]·]+?)\s*·\s*→document_tables:(?P<table_id>[0-9a-fA-F-]{8,})\]"
)

# The fallback trigger's strongest signal (Ananth's framing, 2026-08-19): a
# chunk with no letters at all is very unlikely to be freestanding prose --
# it's a stray table cell/row. Not used to GATE the fallback (that's simply
# "no breadcrumb"), only to classify/telemetry-tag which fallback hits are
# the expected case vs an edge case worth watching.
_ALPHA_RE = re.compile(r"[A-Za-z]")


def is_numeric_only(text: str | None) -> bool:
    """True for a chunk with no alphabetic characters at all (blank/None is
    NOT numeric-only -- there's nothing to be numeric)."""
    if not text or not text.strip():
        return False
    return _ALPHA_RE.search(text) is None


@dataclass(frozen=True)
class TableBreadcrumb:
    caption: str
    table_id: str
    source_chunk_id: str


@dataclass(frozen=True)
class PassengerTable:
    table_id: str
    caption: str
    payload: dict  # document_tables row, opaque here -- shape is Sourcing/DB's, not Retriever's to assume
    cited_by_chunk_ids: tuple[str, ...]  # every citation that carried/matched this table, dedup-ordered
    matched_via: str  # "breadcrumb" | "page_proximity" -- first-seen path; used for gate-3 verification and Chat's confidence signal


def extract_breadcrumbs(text: str, chunk_id: str) -> list[TableBreadcrumb]:
    """Find every table breadcrumb in one chunk's text. Never raises -- a
    chunk with no breadcrumbs (the overwhelming majority, always, for any
    non-table-derived document) just returns []."""
    if not text:
        return []
    return [
        TableBreadcrumb(
            caption=m.group("caption").strip(),
            table_id=m.group("table_id"),
            source_chunk_id=chunk_id,
        )
        for m in _BREADCRUMB_RE.finditer(text)
    ]


def _record_hit(
    seen: dict[str, PassengerTable],
    order: list[str],
    table_id: str,
    caption: str,
    payload: dict,
    chunk_id: str,
    matched_via: str,
) -> None:
    """Add or extend one table's dedup entry. First path to see a table_id
    wins `matched_via` and `caption` -- breadcrumb always runs before
    page-proximity per citation, so a table reachable both ways is recorded
    as the more authoritative "breadcrumb", and later hits (either path)
    just extend cited_by_chunk_ids."""
    existing = seen.get(table_id)
    if existing is None:
        seen[table_id] = PassengerTable(
            table_id=table_id,
            caption=caption,
            payload=payload,
            cited_by_chunk_ids=(chunk_id,),
            matched_via=matched_via,
        )
        order.append(table_id)
        return

    if chunk_id not in existing.cited_by_chunk_ids:
        seen[table_id] = PassengerTable(
            table_id=existing.table_id,
            caption=existing.caption,
            payload=existing.payload,
            cited_by_chunk_ids=existing.cited_by_chunk_ids + (chunk_id,),
            matched_via=existing.matched_via,
        )


def resolve_passenger_tables(
    citations: list,  # CompiledCitation -- typed loosely to avoid a synthesis_contracts import until this is actually wired
    fetch_table: Callable[[str], dict | None],
    fetch_tables_for_page: Callable[[str, int], list[dict]] | None = None,
) -> list[PassengerTable]:
    """Scan compiled citations for breadcrumbs (path 1) and, for citations
    with none, page-proximity matches (path 2, Ananth's go-ahead 2026-08-19).
    Dedup per table_id ACROSS BOTH PATHS -- the contract's own word (stage 3
    gate is "attach, deduped per table") and load-bearing at scale: 14,446
    numeric-only chunks in the table-bearing docs collapse onto only 137
    pages, so without cross-path dedup a handful of retrieved cells from one
    page would attach that page's table dozens of times.

    `fetch_table`/`fetch_tables_for_page` are injected rather than hardcoded
    DB calls so this stays unit-testable without a live `document_tables`
    table -- same reasoning Sourcing gave for capture_page_tables being a
    pure function. `fetch_tables_for_page` is optional (None disables the
    fallback path entirely, e.g. for a caller that only wants path 1).

    Each dict `fetch_tables_for_page` returns must carry the table's own id
    under "id" or "table_id" -- rows missing both are skipped (fail-open,
    logged as a caller-side concern, not raised here).

    Fail-open throughout: a fetch that returns None/[] (unresolvable
    table_id, or no table on that page) or raises is dropped silently, not
    surfaced as an error -- the prose answer is already complete without it.
    """
    seen: dict[str, PassengerTable] = {}
    order: list[str] = []

    for citation in citations:
        chunk_id = getattr(citation, "chunk_id", None)
        text = getattr(citation, "text", None)
        if not chunk_id or not text:
            continue

        crumbs = extract_breadcrumbs(text, chunk_id)

        for crumb in crumbs:
            if crumb.table_id in seen:
                _record_hit(seen, order, crumb.table_id, crumb.caption, seen[crumb.table_id].payload, chunk_id, "breadcrumb")
                continue
            try:
                payload = fetch_table(crumb.table_id)
            except Exception:
                payload = None
            if payload is None:
                continue
            _record_hit(seen, order, crumb.table_id, crumb.caption, payload, chunk_id, "breadcrumb")

        if crumbs or fetch_tables_for_page is None:
            continue

        # Path 2: page-proximity fallback. Fires whenever a citation carried
        # no breadcrumb at all -- numeric-only chunks are the expected/
        # strongest-signal case (Ananth's framing), not an extra gate on top
        # of "no breadcrumb".
        document_id = getattr(citation, "document_id", None)
        page_number = getattr(citation, "page_number", None)
        if not document_id or page_number is None:
            continue

        try:
            rows = fetch_tables_for_page(document_id, page_number) or []
        except Exception:
            rows = []

        for row in rows:
            table_id = row.get("id") or row.get("table_id")
            if not table_id:
                continue
            caption = row.get("caption") or ""
            _record_hit(seen, order, str(table_id), caption, row, chunk_id, "page_proximity")

    return [seen[table_id] for table_id in order]
