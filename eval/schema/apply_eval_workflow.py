"""Deploy the ratified eval-workflow schema (eval/schema/001_eval_workflow.sql).

IRREVERSIBLE STEP — creating the `eval` schema + tables against the RAG DB.
Ratified by Database (DB seat) 2026-08-12; execution holds for Ananth's direct go.

The DDL is fully idempotent (CREATE SCHEMA/TABLE/INDEX IF NOT EXISTS,
CREATE OR REPLACE FUNCTION, DROP+CREATE TRIGGER), so re-running is safe.
asyncpg's simple-query protocol executes the whole multi-statement file in one
execute() call (no parameters).

    python -m eval.schema.apply_eval_workflow          # apply
    python -m eval.schema.apply_eval_workflow --check  # verify tables exist, no writes
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

from eval.db import close_pool, execute, fetchrow

_DDL = pathlib.Path(__file__).with_name("001_eval_workflow.sql")
_EXPECTED = [
    "eval_valid_rulers",
    "eval_population_rules",
    "eval_bank_runs",
    "eval_bank_run_rows",
    "eval_computed_cells",
    "eval_published_priors",
]


async def _apply() -> None:
    sql = _DDL.read_text()
    await execute(sql)
    print(f"applied {_DDL.name} ({len(sql)} bytes)")
    await _check()


async def _check() -> None:
    missing = []
    for t in _EXPECTED:
        row = await fetchrow(
            "SELECT to_regclass($1) AS rel", f"eval.{t}"
        )
        present = row and row["rel"] is not None
        print(f"  eval.{t:24s} {'OK' if present else 'MISSING'}")
        if not present:
            missing.append(t)
    if missing:
        raise SystemExit(f"missing tables: {missing}")


async def _main() -> None:
    try:
        if "--check" in sys.argv:
            await _check()
        else:
            await _apply()
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
