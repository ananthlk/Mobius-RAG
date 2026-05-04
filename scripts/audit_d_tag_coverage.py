"""Diagnostic: audit phrase coverage for every active d-tag.

The Fail Fast gate refuses queries that don't match any d-tag in the
lexicon. We saw a real-world miss: "How do I get credentialed with
Centene" was refused even though ``credentialing.general`` IS an
active d-tag — because that entry's phrase list didn't contain the
bare word "credentialing" or the verb form "credentialed".

This script flags every d-tag entry whose phrase list looks
under-covered, so we can patch the lexicon. For each entry it checks:

  * Does the leaf name appear as a phrase (e.g. ``credentialing`` for
    ``credentialing.general``)?
  * Are common morphological variants present (-ing, -ed, -s, -tion)?
  * Total phrase count (entries with <3 phrases are suspicious).

Output:

  * Per-entry table: code | n_phrases | leaf_present | missing_variants
  * CSV: /tmp/d_tag_coverage_audit_<ts>.csv
  * Summary: count of entries flagged and proposed adds

Read-only — no DB writes. Suggested fix per entry is printed but not
applied.

Usage:
    python3 scripts/audit_d_tag_coverage.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
import urllib.parse


# Common morphological derivatives. If the leaf is "credentialing" we
# also want to see "credentialed", "credentialing", "credential" in the
# phrase list. Empirical: queries use mixed forms.
_VERB_BASES = {
    "credentialing": ["credentialed", "credential"],
    "billing": ["bill", "billed"],
    "claims": ["claim"],
    "appeals": ["appeal", "appealed"],
    "denial": ["denied", "deny"],
    "submission": ["submit", "submitted"],
    "verification": ["verify", "verified"],
    "enrollment": ["enroll", "enrolled"],
    "authorization": ["authorize", "authorized", "auth"],
    "training": ["trained", "train"],
    "reporting": ["report", "reported"],
    "referrals": ["referral", "refer"],
}


def get_database_url() -> str:
    if (url := os.environ.get("DATABASE_URL")):
        return url
    pw = os.popen(
        "gcloud secrets versions access latest --secret=db-password "
        "--project=mobius-os-dev 2>/dev/null"
    ).read().strip()
    if not pw:
        sys.exit("Set DATABASE_URL or have gcloud auth available.")
    pw_enc = urllib.parse.quote(pw)
    return f"postgresql://postgres:{pw_enc}@127.0.0.1:5433/mobius_rag"


def leaf_of(code: str) -> str:
    """Return the leaf of a dotted code as a phrase (underscores → spaces)."""
    leaf = (code or "").split(".")[-1]
    return leaf.replace("_", " ").strip().lower()


def expected_variants(leaf_phrase: str) -> list[str]:
    """Return morphological variants that should also appear in the phrase list.

    For multi-word leaves (e.g. "prior authorization"), we don't try to
    invent — return empty (those entries are usually well-covered).
    """
    if " " in leaf_phrase or not leaf_phrase:
        return []
    return _VERB_BASES.get(leaf_phrase, [])


async def main() -> None:
    try:
        import asyncpg  # type: ignore
    except ImportError:
        sys.exit("Run from rag venv:  source .venv/bin/activate")

    url = get_database_url().replace("+asyncpg", "")
    conn = await asyncpg.connect(url)

    rows = await conn.fetch(
        """
        SELECT code, spec
        FROM policy_lexicon_entries
        WHERE active = true AND kind = 'd'
        ORDER BY code
        """
    )

    print(f"Auditing {len(rows)} active d-tag entries\n")

    out_rows: list[dict] = []
    flagged = 0
    specs_by_code: dict[str, dict] = {}

    for row in rows:
        code = row["code"]
        spec = row["spec"] or {}
        if isinstance(spec, str):
            spec = json.loads(spec)
        specs_by_code[code] = spec
        strong = [p.strip().lower() for p in (spec.get("strong_phrases") or []) if p]
        aliases = [p.strip().lower() for p in (spec.get("aliases") or []) if p]
        all_phrases = set(strong) | set(aliases)

        leaf = leaf_of(code)
        leaf_present = leaf in all_phrases
        variants = expected_variants(leaf)
        missing = [v for v in variants if v not in all_phrases]

        # Flag rules:
        is_thin = len(all_phrases) < 3
        is_missing_leaf = not leaf_present
        is_missing_variants = bool(missing)
        flag = is_thin or is_missing_leaf or is_missing_variants
        if flag:
            flagged += 1

        out_rows.append({
            "code": code,
            "leaf": leaf,
            "n_phrases": len(all_phrases),
            "leaf_present": leaf_present,
            "missing_variants": ";".join(missing),
            "thin": is_thin,
            "needs_attention": flag,
        })

        if flag:
            issues = []
            if is_missing_leaf:
                issues.append(f"missing leaf={leaf!r}")
            if missing:
                issues.append(f"missing_variants={missing}")
            if is_thin:
                issues.append(f"thin (n={len(all_phrases)})")
            print(f"  ⚠️  d:{code:<55} {' | '.join(issues)}")

    # Per-flagged: print existing phrases so we can see what to add
    print(f"\n=== Detail on flagged entries ===")
    for r in out_rows:
        if not r["needs_attention"]:
            continue
        c = r["code"]
        spec = specs_by_code.get(c, {})
        existing_strong = list(spec.get("strong_phrases") or [])[:8]
        existing_aliases = list(spec.get("aliases") or [])[:8]
        print(f"\n  d:{c}  (n={r['n_phrases']})")
        print(f"    strong:    {existing_strong}")
        print(f"    aliases:   {existing_aliases[:6]}{'...' if len(existing_aliases) > 6 else ''}")
        proposed_adds: list[str] = []
        if not r["leaf_present"]:
            proposed_adds.append(r["leaf"])
        if r["missing_variants"]:
            proposed_adds.extend(r["missing_variants"].split(";"))
        if proposed_adds:
            print(f"    PROPOSE adding to aliases: {proposed_adds}")

    csv_path = f"/tmp/d_tag_coverage_audit_{int(time.time())}.csv"
    if out_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)

    print(f"\n=== SUMMARY ===")
    print(f"Total d-tags:      {len(rows)}")
    print(f"Flagged for fix:   {flagged}  ({flagged/max(len(rows),1):.0%})")
    print(f"CSV:               {csv_path}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
