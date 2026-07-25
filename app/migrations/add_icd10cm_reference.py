"""Migration: reference.icd10cm_reference — local ICD-10-CM code->description lookup.

Built 2026-07-23 as the local-data counterpart to the shape gate's HCPCS code
expansion (`app/services/retriever/shape/code_expand.py`). Problem: the only
ICD-10 lookup previously available anywhere in the fleet
(`mobius-skills/healthcare/app/clients/icd10.py`) is a LIVE NLM Clinical
Tables API call (15s timeout) — unusable inside the gate's real-time,
no-network, <500ms path.

Architecture (per DB agent review 2026-07-23): Postgres is the durable
SOURCE OF TRUTH, not a per-request dependency. The gate loads this table
ONCE into an in-memory dict at process/module startup (identical pattern to
`code_expand.py`'s `_load()` for the HCPCS CSV — just sourced from a table
via one SELECT instead of a file via `csv.DictReader`). Zero DB calls inside
`run_gate()`'s hot path; per-query behavior is unchanged. Postgres also lets
`mobius-skills/healthcare/app/clients/icd10.py` eventually query this table
instead of the live NLM API — a fleet-wide fix, not just this one cache.

Schema/role pattern mirrors `facts` (add_payor_fact_store.py): dedicated
schema + RW/RO group roles, not table grants on `public`.

Source: CDC's official FY2026 ICD-10-CM Code Descriptions file, staged at
`mobius-rag/data/reference/icd10cm_codes_2026.txt` (repo-tracked, not /tmp —
DB agent flagged the original draft's ephemeral path). Fixed-width format:
code padded to 8 chars, then description, CRLF line endings. 74,719 codes.

Idempotent — CREATE SCHEMA/TABLE IF NOT EXISTS, upsert on load, chunked
(~5k rows/batch per DB agent's resumability recommendation).
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL

SOURCE_TXT = project_root / "data" / "reference" / "icd10cm_codes_2026.txt"
BATCH_SIZE = 5000


def parse_source(path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            code = line[:8].strip().upper()
            desc = line[8:].strip()
            if code and desc:
                rows.append((code, desc))
    return rows


async def main() -> None:
    conn = await asyncpg.connect(DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS reference;")

        await conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='mobius_reference_rw') THEN
                    CREATE ROLE mobius_reference_rw NOLOGIN;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='mobius_reference_ro') THEN
                    CREATE ROLE mobius_reference_ro NOLOGIN;
                END IF;
            END
            $$;
            """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reference.icd10cm_reference (
                code TEXT PRIMARY KEY,
                description TEXT NOT NULL
            )
            """
        )

        await conn.execute("GRANT USAGE ON SCHEMA reference TO mobius_reference_rw, mobius_reference_ro;")
        await conn.execute(
            "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA reference TO mobius_reference_rw;"
        )
        await conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA reference TO mobius_reference_ro;")
        await conn.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA reference GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mobius_reference_rw;"
        )
        await conn.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA reference GRANT SELECT ON TABLES TO mobius_reference_ro;"
        )

        rows = parse_source(SOURCE_TXT)
        print(f"parsed {len(rows)} codes from {SOURCE_TXT}")

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            await conn.executemany(
                """
                INSERT INTO reference.icd10cm_reference (code, description)
                VALUES ($1, $2)
                ON CONFLICT (code) DO UPDATE SET description = EXCLUDED.description
                """,
                batch,
            )
            print(f"  loaded {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

        count = await conn.fetchval("SELECT count(*) FROM reference.icd10cm_reference")
        print(f"reference.icd10cm_reference now has {count} rows")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
