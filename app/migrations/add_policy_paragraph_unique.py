"""Migration: dedupe ``policy_paragraphs`` and add a unique index.

Background — the chunking worker's ``clear_policy_for_document``
helper used to swallow exceptions and return as if the cleanup
succeeded. When the cleanup actually failed (deadlock, timeout),
the worker would proceed to insert paragraphs ON TOP of the
unwiped old set, producing N× duplicate rows. Across 11 oversize
docs we saw 5×–35× duplication; the worst was "Constrained
Budgets HIV": 43 unique paragraphs stored as 1,497 rows.

This migration:

1. Deletes duplicate ``policy_paragraphs`` rows, keeping the row
   with the most ``policy_lines`` per ``(document_id, page_number,
   order_index)`` tuple (ties broken by largest ``id``). The
   line-bearing copy is kept; cascade only removes line-less dupes.
2. Adds a UNIQUE index on the same key so future workers can no
   longer insert a duplicate at the DB level — the insert errors
   out instead of accumulating.

Performance design — ``policy_lines`` has 11.9M rows with no
single-column index on ``paragraph_id``, so a naive
``DELETE … NOT IN (keepers)`` + cascade is O(N²). Instead:

  a. Materialise keepers + losers into permanent tables (survive
     reconnects; idempotent drop-if-exists at the top).
  b. Delete ``policy_lines`` per affected document using the
     existing composite index
     ``idx_policy_lines_doc_page (document_id, …, paragraph_id, …)``
     so each per-doc scan is fast.
  c. Delete loser paragraphs — FK check is now instant (lines
     already gone for those paragraphs).

Idempotent: staging tables are always dropped at the start,
``CREATE UNIQUE INDEX IF NOT EXISTS`` is safe to re-run, and the
loser DELETE against an already-clean table deletes 0 rows.
"""
import asyncio
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg  # noqa: E402

from app.config import DATABASE_URL  # noqa: E402


def _parse_db_url(url: str) -> dict:
    m = re.match(
        r"postgresql(?:\+asyncpg)?://(?P<user>[^:]+):(?P<password>[^@]+)"
        r"@(?P<host>[^:/]+):(?P<port>\d+)/(?P<database>.+)",
        url,
    )
    if not m:
        raise ValueError(f"Cannot parse DATABASE_URL: {url!r}")
    return {
        "host": m.group("host"),
        "port": int(m.group("port")),
        "user": m.group("user"),
        "password": m.group("password"),
        "database": m.group("database"),
        "ssl": False,
    }


async def _connect() -> asyncpg.Connection:
    return await asyncpg.connect(**_parse_db_url(DATABASE_URL))


async def migrate() -> None:
    # ── Step 1: build staging tables (fresh connection) ──────────────
    conn = await _connect()
    try:
        before = await conn.fetchval("SELECT COUNT(*) FROM policy_paragraphs")
        print(f"  policy_paragraphs before: {before}")

        # Idempotent: drop any remnants from a prior failed run
        await conn.execute("DROP TABLE IF EXISTS _mig_pp_keepers")
        await conn.execute("DROP TABLE IF EXISTS _mig_pp_losers")

        # Keeper = the row with the most policy_lines (line-bearing copy).
        # Ties broken by largest id so the result is deterministic.
        print("  Building keepers table…")
        await conn.execute("""
            CREATE TABLE _mig_pp_keepers AS
            SELECT DISTINCT ON (pp.document_id, pp.page_number, pp.order_index)
                   pp.id, pp.document_id
            FROM policy_paragraphs pp
            LEFT JOIN (
                SELECT paragraph_id, count(*) AS n
                FROM policy_lines
                GROUP BY paragraph_id
            ) plc ON plc.paragraph_id = pp.id
            ORDER BY pp.document_id, pp.page_number, pp.order_index,
                     COALESCE(plc.n, 0) DESC, pp.id DESC
        """)
        kc = await conn.fetchval("SELECT COUNT(*) FROM _mig_pp_keepers")
        print(f"  Keepers: {kc}")
    finally:
        await conn.close()

    # ── Step 2: anti-join → losers (fresh connection) ────────────────
    conn = await _connect()
    try:
        await conn.execute("CREATE INDEX ON _mig_pp_keepers (id)")
        print("  Building losers table (anti-join)…")
        await conn.execute("""
            CREATE TABLE _mig_pp_losers AS
            SELECT pp.id, pp.document_id
            FROM policy_paragraphs pp
            LEFT JOIN _mig_pp_keepers k ON k.id = pp.id
            WHERE k.id IS NULL
        """)
        await conn.execute("CREATE INDEX ON _mig_pp_losers (id)")
        await conn.execute("CREATE INDEX ON _mig_pp_losers (document_id)")
        lc = await conn.fetchval("SELECT COUNT(*) FROM _mig_pp_losers")
        print(f"  Losers: {lc}")

        doc_ids = [r["document_id"] for r in await conn.fetch(
            "SELECT DISTINCT document_id FROM _mig_pp_losers ORDER BY document_id"
        )]
        print(f"  Affected documents: {len(doc_ids)}")
    finally:
        await conn.close()

    # ── Step 3: per-document policy_lines delete ─────────────────────
    # Uses existing idx_policy_lines_doc_page (document_id, …) so each
    # per-doc scan is an index range scan, not a full-table scan.
    total_lines = 0
    for i, doc_id in enumerate(doc_ids):
        conn = await _connect()
        try:
            r = await conn.execute("""
                DELETE FROM policy_lines pl
                USING _mig_pp_losers lsr
                WHERE lsr.id = pl.paragraph_id
                  AND pl.document_id = lsr.document_id
                  AND lsr.document_id = $1
            """, doc_id)
            n = int(r.split()[-1]) if r and r.split()[-1].isdigit() else 0
            total_lines += n
            if n > 0 or (i % 20 == 0):
                print(f"  [{i + 1}/{len(doc_ids)}] doc {doc_id}: {n} lines deleted")
        finally:
            await conn.close()

    print(f"  Total policy_lines deleted: {total_lines}")

    # ── Step 4: delete loser paragraphs + unique index ────────────────
    conn = await _connect()
    try:
        print("  Deleting loser paragraphs…")
        result = await conn.execute("""
            DELETE FROM policy_paragraphs
            WHERE id IN (SELECT id FROM _mig_pp_losers)
        """)
        deleted = int(result.split()[-1]) if result and result.split()[-1].isdigit() else 0
        after = await conn.fetchval("SELECT COUNT(*) FROM policy_paragraphs")
        print(f"  Deleted {deleted} duplicate policy_paragraphs rows")
        print(f"  policy_paragraphs after: {after}")

        print("  Adding unique index…")
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_policy_paragraphs_doc_page_order
            ON policy_paragraphs (document_id, page_number, order_index)
        """)
        print("  Ensured unique index: ux_policy_paragraphs_doc_page_order")

        await conn.execute("DROP TABLE IF EXISTS _mig_pp_losers")
        await conn.execute("DROP TABLE IF EXISTS _mig_pp_keepers")
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(migrate())
    print("\nMigration add_policy_paragraph_unique completed.")


if __name__ == "__main__":
    main()
