"""Persist captured tables — the wiring half of table capture.

Sourcing's `capture_page_tables()` is pure and mints the row `id` client-side,
baking it into the breadcrumb it leaves in the page text. That id is the ONLY
link from a retrieved chunk back to its table, which makes persistence the place
where two silent failures live. Both are handled here, deliberately in one
function so no ingest path can implement them differently.

TRAP 1 — the id must be inserted EXPLICITLY.
  `document_tables.id` carries `DEFAULT gen_random_uuid()`. If we omit the id and
  let the default fire, every breadcrumb points at a row that does not exist.
  Nothing errors: capture fails open, Retriever fails open on an unresolvable id,
  the FK is satisfied and no uniqueness is violated. The system would simply never
  attach a table, forever, with every test passing. So the id is always supplied.

TRAP 2 — reingest must REPLACE, not accumulate.
  `uuid4` is fresh on every run and the applied schema has no
  UNIQUE (document_id, page_number, table_index). Re-extracting a page therefore
  inserts a SECOND complete set of rows while the rewritten page text carries only
  the new breadcrumbs — the old rows become unreachable and permanent. So each
  page's tables are deleted before its new ones are written, in the SAME
  transaction, making the write idempotent per page.
  (Asked DB to add table_index + UNIQUE so this is enforced by the database rather
  than by this function being the only writer. Until then, it is enforced here.)

SCOPE. Page-scoped, not document-scoped: a document is re-extracted page by page
and a page that captured nothing must still clear its old tables, or a table that
has genuinely disappeared would linger. `_replace_page_tables` is therefore called
for every page, including those with an empty list.

FAIL-OPEN, consistent with the rest of the path: a table that cannot be persisted
must not fail the document's ingest. Errors are logged and counted, never raised.
The counts are returned so the caller can record them rather than discover the
loss later — a write path with no reader is the defect class this repo keeps
re-learning.

PARTIAL REPLACEMENT IS POSSIBLE, AND IS THE DELIBERATE CHOICE.
  The delete runs first, so if some inserts then fail the page ends up with fewer
  tables than it had. That is not an oversight. The page TEXT is rewritten with
  fresh breadcrumbs in the same ingest, so the old rows are already unreachable —
  rolling the replacement back would leave every breadcrumb on the page
  unresolvable, which is strictly worse than leaving most of them resolvable.
  What matters is therefore not atomicity but VISIBILITY: `failed > 0` means
  breadcrumbs exist in the page text with no row behind them, and because both
  capture and Retriever fail open, nothing downstream will ever complain. The
  caller MUST read `failed` — gate 2 asserts it is zero.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Exactly the columns the applied schema (migration 050) carries, in order.
# `bbox` and `breadcrumb` are deliberately absent: bbox lives inside `anchor`,
# and the breadcrumb lives in the page text, not in this table.
_COLS = ("id", "document_id", "page_number", "grid", "anchor", "coverage",
         "caption", "strategy", "n_rows", "n_cols", "is_clean")
_JSONB = ("grid", "anchor")


def _params(table: dict, document_id: str) -> dict:
    """Map one captured table onto the document_tables columns as named binds."""
    out = {}
    for c in _COLS:
        if c == "document_id":
            out[c] = str(document_id)
        elif c in _JSONB:
            v = table.get(c)
            out[c] = json.dumps(v) if v is not None else None
        elif c == "coverage":
            # text[] in the applied schema, NOT jsonb — a list maps directly.
            out[c] = list(table.get("coverage") or [])
        else:
            out[c] = table.get(c)
    return out


async def persist_page_tables(db, document_id: str, page_number: int,
                              tables: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Replace `document_id`/`page_number`'s tables with `tables`. Never raises.

    `db` is the SQLAlchemy AsyncSession the ingest paths already hold. The caller
    owns the transaction and the commit.

    ORDERING REQUIREMENT: `document_tables` carries a composite FK to
    (document_id, page_number) on `document_pages`, so the pages must already be
    committed when this runs. Called after the page commit in every path.

    Returns {"written": n, "removed": n, "failed": n}.
    """
    tables = list(tables or [])
    stats = {"written": 0, "removed": 0, "failed": 0}
    try:
        res = await db.execute(
            text("DELETE FROM document_tables WHERE document_id = :doc "
                 "AND page_number = :pg"),
            {"doc": str(document_id), "pg": page_number})
        stats["removed"] = res.rowcount or 0

        if not tables:
            return stats

        cols = ",".join(_COLS)
        binds = ",".join(f":{c}" for c in _COLS)
        sql = text(f"INSERT INTO document_tables ({cols}) VALUES ({binds})")
        for t in tables:
            # SAVEPOINT per insert. Postgres aborts the ENTIRE transaction on an
            # integrity error, so a bare try/except here does not isolate a bad
            # table — it loses the good ones too AND leaves the caller's session
            # poisoned, which would fail the whole document's ingest. Verified
            # against the live schema: without this, a bad row followed by a good
            # one yielded written=0 failed=2 and an unusable session. That is the
            # precise opposite of fail-open, so the savepoint is load-bearing.
            try:
                async with db.begin_nested():
                    await db.execute(sql, _params(t, document_id))
                stats["written"] += 1
            except Exception as e:
                # Rolled back to the savepoint: this table is gone, the page's
                # other tables and the caller's transaction are intact.
                stats["failed"] += 1
                logger.warning("document_tables insert failed (doc=%s page=%s): %s",
                               document_id, page_number, e)
    except Exception as e:
        stats["failed"] += len(tables)
        logger.warning("document_tables persistence failed (doc=%s page=%s): %s",
                       document_id, page_number, e)
    return stats


async def persist_document_tables(db, document_id: str,
                                  pages: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Persist every page's tables for one document. Never raises.

    `pages` is the list `extract_text_from_gcs` returns. A page with no "tables"
    key was extracted with capture OFF and is left alone — clearing its tables
    would delete captured data simply because the flag was off for this run.
    A page with an EMPTY list was captured and genuinely has none, so it is
    cleared.
    """
    total = {"written": 0, "removed": 0, "failed": 0, "pages": 0}
    for p in pages or []:
        if "tables" not in p:
            continue                      # capture disabled for this page — not "no tables"
        s = await persist_page_tables(db, document_id, p.get("page_number"),
                                      p.get("tables") or [])
        for k in ("written", "removed", "failed"):
            total[k] += s[k]
        total["pages"] += 1
    if total["pages"]:
        logger.info("document_tables: doc=%s pages=%s written=%s removed=%s failed=%s",
                    document_id, total["pages"], total["written"], total["removed"],
                    total["failed"])
    return total
