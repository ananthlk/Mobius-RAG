"""Migration: retype ``documents.effective_date`` / ``termination_date`` VARCHAR -> DATE.

**DRAFTED BY THE FACT STORE SEAT FOR DB REVIEW — NOT APPLIED.**
``documents`` is a shared corpus table and migrations are DB's governance. This
exists so DB reviews a concrete, verified change rather than a description.
Nothing here has been run against any database.

Why
---
``effective_date`` is becoming the **version-selection key** for the AHCA
reindex: 5,257 AHCA documents with ``distinct file_hash = row count`` (no
byte-identical duplicates), so dedup is near-duplicate version selection
ordered by date. Today the column is ``character varying`` and nothing
enforces a format at any layer — the write path
(``POST /documents/import-scraped-pages``) declares ``Optional[str]`` with no
validator. So ``"08/17/2026"`` would be accepted, stored, and sort **wrongly**
against ``2026-07-01``, silently, with no error anywhere. Two independent
layers of no-enforcement on the field the reindex turns on.

Verified safe as of 2026-08-17 (see ``verify()`` below, which re-checks at run
time rather than trusting this note):

===========================  ========  ==========  ====================
column                       non-null  non-ISO     casts cleanly to DATE
===========================  ========  ==========  ====================
documents.effective_date        5,261        0            5,261 / 5,261
documents.termination_date      9,630        0            9,630 / 9,630
===========================  ========  ==========  ====================

So this is a **pure type change with zero data cleanup attached**. That is not
a durable property — it holds precisely because no crawl has yet written a
non-ISO value. The first one converts a two-statement migration into a
data-cleanup project, which is why this is worth doing before the AHCA crawl
rather than after.

Properties
----------
* **Fails closed.** ``verify()`` runs first and *aborts* if any value would not
  cast. It never coerces, never nulls a bad row, never partially applies.
* **Idempotent.** Re-running when the columns are already ``date`` is a no-op.
* **Transactional.** Both columns move together or neither does.
* **Prints before/after** so the operator can confirm no rows were perturbed.

What this migration does NOT fix
--------------------------------
Recurrence. A ``NULL`` is not a malformed date, so typing the column never
forces anyone to supply one. The root cause was that RAG's Upload UI never sent
the field at all — fixed separately by Master RAG in ``19a9e25``, which added
the inputs. That fix closes the tap; this one makes the column honest. Both are
needed and neither substitutes for the other.

Rollback
--------
Use ``to_char``, **not** ``::text``::

    ALTER TABLE documents ALTER COLUMN effective_date   TYPE varchar
      USING to_char(effective_date, 'YYYY-MM-DD');
    ALTER TABLE documents ALTER COLUMN termination_date TYPE varchar
      USING to_char(termination_date, 'YYYY-MM-DD');

**This form is load-bearing, and the obvious one is wrong.** An earlier draft
used ``::text`` and claimed losslessness because "every ``date`` renders as
``YYYY-MM-DD``". That is false — ``date::text`` renders according to the
``DateStyle`` GUC, which is a session/server setting, not a property of the
column. Crawler caught it in review (§18) and I verified it directly:

======================  ================  ==========================
DateStyle               ``::text`` gives  outcome
======================  ================  ==========================
``ISO, MDY`` (current)  ``2024-09-01``    ok — but only by configuration
``SQL, MDY``            ``09/01/2024``    reintroduces the bug
``German, DMY``         ``01.09.2024``    reintroduces the bug
``Postgres, DMY``       ``01-09-2024``    reintroduces the bug
======================  ================  ==========================

So a rollback run in a session with a non-default ``DateStyle`` would write the
exact non-ISO strings this migration exists to eliminate — into a varchar
column with no enforcement. The undo would restore the original failure mode.
``to_char`` is explicit and ``DateStyle``-independent; verified ISO under all
four settings above.

Known consequence — month-precision dates
-----------------------------------------
A ``date`` column cannot hold month precision. Crawler measured (§12) that
roughly half the corpus dates itself in text as e.g. ``"September 2024"``.
After this migration those can only be stored as a **synthesised day**
(``-01``, an invention recorded as if it were source data) or as **NULL** (the
state this sprint exists to eliminate).

That is not an argument against the migration — the column should be a date.
But it *closes* the option of representing the precision we actually have, so
if a ``date_precision`` companion column is ever wanted, adding it alongside
this change is far cheaper than retrofitting it. Flagging for DB rather than
deciding it here.

Usage::

    python -m app.migrations.retype_document_dates_to_date          # verify + apply
    python -m app.migrations.retype_document_dates_to_date --check  # verify only
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg  # noqa: E402

from app.config import DATABASE_URL  # noqa: E402

COLUMNS = ("effective_date", "termination_date")
ISO = r"^\d{4}-\d{2}-\d{2}$"


async def _coltype(conn: asyncpg.Connection, column: str) -> str | None:
    return await conn.fetchval(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name='documents' AND column_name=$1",
        column,
    )


async def verify(conn: asyncpg.Connection) -> bool:
    """Report castability per column. Returns False if ANY value would fail.

    Deliberately re-measured at run time. The numbers in the docstring were
    true when this was drafted; the operator needs them true when it runs.
    """
    ok = True
    for col in COLUMNS:
        current = await _coltype(conn, col)
        if current == "date":
            print(f"  {col:18s} already DATE — nothing to do")
            continue
        non_null = await conn.fetchval(
            f"SELECT count(*) FROM documents WHERE {col} IS NOT NULL")
        bad = await conn.fetch(
            f"SELECT {col} AS v, count(*) n FROM documents "
            f"WHERE {col} IS NOT NULL AND {col} !~ '{ISO}' "
            f"GROUP BY 1 ORDER BY n DESC LIMIT 10")
        bad_total = sum(r["n"] for r in bad)
        print(f"  {col:18s} type={current} non_null={non_null} non_ISO={bad_total}")
        if bad:
            ok = False
            print(f"    ✗ would NOT cast — offending values:")
            for r in bad:
                print(f"        {r['v']!r} x{r['n']}")
    return ok


async def migrate(check_only: bool = False) -> int:
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        print("Pre-flight verification:")
        if not await verify(conn):
            print("\nABORTED — one or more values would not cast to DATE.")
            print("Fix or null the offending rows first; this migration will not coerce them.")
            return 1

        if check_only:
            print("\n--check: verification passed, no changes made.")
            return 0

        # Both columns move together — a half-applied retype would leave
        # version selection comparing a date against a varchar.
        async with conn.transaction():
            for col in COLUMNS:
                if await _coltype(conn, col) == "date":
                    continue
                await conn.execute(
                    f"ALTER TABLE documents ALTER COLUMN {col} TYPE date USING {col}::date")
                print(f"  Retyped documents.{col} -> DATE")

        print("\nPost-migration:")
        for col in COLUMNS:
            t = await _coltype(conn, col)
            n = await conn.fetchval(f"SELECT count(*) FROM documents WHERE {col} IS NOT NULL")
            print(f"  {col:18s} type={t} non_null={n}")
        return 0
    finally:
        await conn.close()


def main() -> None:
    check_only = "--check" in sys.argv
    rc = asyncio.run(migrate(check_only=check_only))
    if rc == 0 and not check_only:
        print("\nMigration retype_document_dates_to_date completed.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
