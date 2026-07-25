"""Migration: reference.hcpcs_level2_reference — full HCPCS Level II code->description lookup.

Extends the shape gate's code expansion beyond the 82-row FL behavioral-
health subset (`mobius-dbt/seeds/fl_bh_code_reference.csv`) to the complete
official HCPCS Level II code set (8,725 codes) — same architecture as
`add_icd10cm_reference.py`: Postgres as durable source of truth, loaded ONCE
into an in-memory dict at gate startup, zero per-query DB calls.

CPT-4 codes (5-digit numeric, AMA-copyrighted) are explicitly EXCLUDED —
the source file (CMS's July 2026 Alpha-Numeric HCPCS file) only contains
Level II codes (letter + 4 digits) in practice, but the parser also filters
defensively in case a future release ever bundles CPT rows, since CPT
decode was already scoped out as a known gap (no free translation source
exists) and using AMA-copyrighted text here would be a real problem, not
just a scope question.

Source: CMS's official July 2026 HCPCS Level II Alpha-Numeric file, staged
at `mobius-rag/data/reference/hcpcs_level2_2026.csv` (already parsed/filtered
from the raw fixed-width CMS file; code, short_description, long_description
columns).

Idempotent — CREATE SCHEMA/TABLE IF NOT EXISTS, upsert on load, chunked.
Reuses the `reference` schema + `mobius_reference_{rw,ro}` roles created by
`add_icd10cm_reference.py` (safe if run in either order — both use
IF NOT EXISTS / DO-block guards).
"""
import asyncio
import csv
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL

SOURCE_CSV = project_root / "data" / "reference" / "hcpcs_level2_2026.csv"
BATCH_SIZE = 5000
_LEVEL_II_CODE_RE = re.compile(r"^[A-Z]\d{4}$")


def parse_source(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip().upper()
            if not _LEVEL_II_CODE_RE.match(code):
                continue  # defensive: never load a CPT-shaped (5-digit numeric) code
            short_desc = (row.get("short_description") or "").strip()
            long_desc = (row.get("long_description") or "").strip()
            if code and (short_desc or long_desc):
                rows.append((code, short_desc, long_desc or short_desc))
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
            CREATE TABLE IF NOT EXISTS reference.hcpcs_level2_reference (
                code TEXT PRIMARY KEY,
                short_description TEXT NOT NULL,
                long_description TEXT NOT NULL
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

        rows = parse_source(SOURCE_CSV)
        print(f"parsed {len(rows)} Level II codes from {SOURCE_CSV}")

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            await conn.executemany(
                """
                INSERT INTO reference.hcpcs_level2_reference (code, short_description, long_description)
                VALUES ($1, $2, $3)
                ON CONFLICT (code) DO UPDATE SET
                    short_description = EXCLUDED.short_description,
                    long_description = EXCLUDED.long_description
                """,
                batch,
            )
            print(f"  loaded {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

        count = await conn.fetchval("SELECT count(*) FROM reference.hcpcs_level2_reference")
        print(f"reference.hcpcs_level2_reference now has {count} rows")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
