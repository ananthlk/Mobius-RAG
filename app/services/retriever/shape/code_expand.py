"""Literal procedure-code expansion — the pre-processing stage that feeds
decoded text into lexicon matching, per Retriever's ruling 2026-07-23.

Problem: "Does Sunshine Health cover H0019 for Medicaid patients?" carries
zero D-tag signal today — the lexicon has no entries for bare codes, only
descriptive phrases. Decoding H0019 -> "Behavioral health; long-term
residential..." before lexicon matching lets the EXISTING phrase-matching
pipeline do its job; this module does not touch `_classify()`, `gate.py`'s
decision branches, or any locked heuristic — it only produces better input
text for `expand_query_via_lexicon()`.

Data sources (per DB agent review 2026-07-23): Postgres is the durable
SOURCE OF TRUTH — `reference.hcpcs_level2_reference` (8,725 codes, official
CMS July 2026 Level II file) and `reference.icd10cm_reference` (74,719
codes, official CDC FY2026 file). See `app/migrations/add_hcpcs_level2_reference.py`
and `add_icd10cm_reference.py`.

Load pattern: LAZY, ASYNC, ONE-TIME per process — `ensure_loaded(db)` is
called by `run_gate()` before lexicon expansion; a module-level flag skips
the DB round-trip on every call after the first. This is NOT an import-time
load (an eager `asyncio.run()` at import risks "cannot be called from a
running event loop" inside pytest-asyncio/ASGI startup) — lazy-on-first-call
achieves the same "one DB query per process, zero per-request cost"
property without that fragility. `expand_query()` itself stays a plain
synchronous function operating on whatever's in memory (empty dicts before
the first load = safe no-op, never a crash).

CPT-4 explicitly OUT of scope: no free/local translation source exists
anywhere in the fleet (AMA-copyrighted); stubbed, not solved here.
"""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# code -> description, description (lowercased, for reverse match) -> code
_CODE_TO_DESC: dict[str, str] = {}
_DESC_TO_CODE: dict[str, str] = {}
_LOADED = False

_CODE_TOKEN_RE = re.compile(r"\b([A-Za-z]\d{4})\b")


async def ensure_loaded(db: AsyncSession) -> None:
    """Load both reference tables into memory, once per process. Cheap check
    on every call after the first (module-level bool), no DB round-trip."""
    global _LOADED
    if _LOADED:
        return

    hcpcs_rows = await db.execute(
        text("SELECT code, COALESCE(long_description, short_description) AS desc "
             "FROM reference.hcpcs_level2_reference")
    )
    for code, desc in hcpcs_rows:
        if code and desc:
            _CODE_TO_DESC[code] = desc
            _DESC_TO_CODE[desc.lower()] = code

    icd10_rows = await db.execute(text("SELECT code, description FROM reference.icd10cm_reference"))
    for code, desc in icd10_rows:
        if code and desc:
            _CODE_TO_DESC.setdefault(code, desc)  # HCPCS wins on any code collision (shouldn't happen — disjoint code shapes)
            _DESC_TO_CODE.setdefault(desc.lower(), code)

    _LOADED = True


def expand_query(query: str) -> str:
    """Return query + any decoded codes/matched descriptions appended.

    Bidirectional, both directions additive (never removes/replaces the
    original text — this return value is the "expansion text" fed to
    `expand_query_via_lexicon()`, never `GateResult.query`/`.normalized`):

      1. literal code in query  -> append its description
         "What about H0019?"   -> "What about H0019? Behavioral health;
                                    long-term residential..."
      2. known description phrase in query -> append its code
         "long-term residential behavioral health" -> "...health H0019"

    Word-boundary matching throughout (regex \\b for codes, substring-in-
    normalized-query for descriptions) — same safety principle as the
    lexicon's own `_match_entry`, avoiding partial-token false positives.

    Safe no-op if `ensure_loaded()` hasn't run yet (empty dicts) — never
    raises, just returns the query unchanged.
    """
    if not query or not _CODE_TO_DESC:
        return query

    additions: list[str] = []

    for token in _CODE_TOKEN_RE.findall(query.upper()):
        desc = _CODE_TO_DESC.get(token)
        if desc and desc not in additions:
            additions.append(desc)

    query_lower = query.lower()
    for desc_lower, code in _DESC_TO_CODE.items():
        if desc_lower in query_lower and code not in additions:
            additions.append(code)

    if not additions:
        return query
    return query + " " + " ".join(additions)
