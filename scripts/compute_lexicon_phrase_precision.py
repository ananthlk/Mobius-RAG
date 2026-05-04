"""Diagnostic: per-phrase document-frequency precision against lexicon tags.

For each phrase in every active lexicon entry, computes:

  df         = distinct docs whose search_vec matches the phrase
                (via ``@@ phraseto_tsquery('english', phrase)``)
  df_tagged  = of those docs, how many are also tagged with the entry's
                concept (``document_tags.{j,d,p}_tags ? code``)
  precision  = df_tagged / df

Then applies pruning rules to mark each phrase as KEEP / DROP_NOISY /
DROP_RARE / DROP_DUPE / KEEP_CANONICAL.

This is a READ-ONLY diagnostic — no DB writes. Outputs:

  * Per-phrase CSV: /tmp/lexicon_phrase_precision_<ts>.csv
  * Per-entry summary table on stdout
  * Aggregate counters

Why this matters: the BM25 lexicon expansion currently OR's together
*every* strong_phrase + alias of every matched entry. A query like
"Sunshine Health prior authorization behavioral health" can produce a
60+ phrase OR-tsquery, which is the bottleneck (Q1 took 25.9s on a
110-doc pool). Pruning low-precision high-volume phrases (e.g.,
``"authorization"`` alone — precision 0.49) collapses the expansion
without hurting recall — chunks already containing the canonical
phrase still match, but generic-noise matches stop.

Usage:
    python3 scripts/compute_lexicon_phrase_precision.py
    # Optional: limit to a single entry for fast iteration
    LEXICON_FILTER=d:utilization_management.prior_authorization \\
      python3 scripts/compute_lexicon_phrase_precision.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
import urllib.parse
from collections import Counter, defaultdict


# Pruning rules (applied per-phrase after precision is computed)
RULE_DROP_NOISY = {"min_df": 100, "max_precision": 0.60}
RULE_DROP_RARE = {"max_df": 5}
RULE_KEEP_CANONICAL = {"min_df": 100, "min_precision": 0.85}


def get_database_url() -> str:
    if (url := os.environ.get("DATABASE_URL")):
        return url
    # Build from gcloud secrets, mirroring scripts/probe_search_agent.py
    pw = os.popen(
        "gcloud secrets versions access latest --secret=db-password "
        "--project=mobius-os-dev 2>/dev/null"
    ).read().strip()
    if not pw:
        sys.exit("Set DATABASE_URL or have gcloud auth available.")
    pw_enc = urllib.parse.quote(pw)
    return (
        f"postgresql://postgres:{pw_enc}@127.0.0.1:5433/mobius_rag"
    )


async def main() -> None:
    try:
        import asyncpg  # type: ignore
    except ImportError:
        sys.exit("Run from rag venv:  source .venv/bin/activate")

    url = get_database_url().replace("+asyncpg", "")
    conn = await asyncpg.connect(url)
    only = os.environ.get("LEXICON_FILTER")

    # ── 1. Load active lexicon entries ────────────────────────────────
    if only and ":" in only:
        kind, code = only.split(":", 1)
        rows = await conn.fetch(
            """
            SELECT kind, code, spec
            FROM policy_lexicon_entries
            WHERE active = true AND kind = $1 AND code = $2
            ORDER BY kind, code
            """,
            kind, code,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT kind, code, spec
            FROM policy_lexicon_entries
            WHERE active = true
            ORDER BY kind, code
            """,
        )

    print(f"Loaded {len(rows)} active lexicon entries")
    if not rows:
        sys.exit(0)

    # ── 2. For each entry, fetch tagged-docs once + score each phrase ──
    out_rows: list[dict] = []
    started = time.time()

    for i, entry in enumerate(rows):
        kind = entry["kind"]
        code = entry["code"]
        full_code = f"{kind}:{code}"
        column = {"j": "j_tags", "d": "d_tags", "p": "p_tags"}.get(kind)
        if not column:
            continue

        spec = entry["spec"] or {}
        if isinstance(spec, str):
            spec = json.loads(spec)
        strong_phrases = list(spec.get("strong_phrases") or [])
        aliases = list(spec.get("aliases") or [])
        # Dedupe while preserving "is_strong" provenance
        phrase_to_kind: dict[str, str] = {}
        for p in strong_phrases:
            if p:
                phrase_to_kind[p.strip().lower()] = "strong"
        for p in aliases:
            if p:
                phrase_to_kind.setdefault(p.strip().lower(), "alias")

        # The tag's docs (one query per entry)
        tagged_docs_q = await conn.fetch(
            f"SELECT document_id FROM document_tags WHERE {column} ? $1",
            code,
        )
        tagged_docs = {r["document_id"] for r in tagged_docs_q}
        n_tagged = len(tagged_docs)

        # Per-phrase counts
        per_phrase: list[dict] = []
        for phrase, src in phrase_to_kind.items():
            try:
                # df + df_tagged in one query
                row = await conn.fetchrow(
                    """
                    SELECT
                      COUNT(DISTINCT document_id) AS df,
                      COUNT(DISTINCT document_id) FILTER (
                        WHERE document_id = ANY($2::uuid[])
                      ) AS df_tagged
                    FROM rag_published_embeddings
                    WHERE search_vec @@ phraseto_tsquery('english', $1)
                    """,
                    phrase, list(tagged_docs),
                )
                df = int(row["df"] or 0)
                df_tagged = int(row["df_tagged"] or 0)
            except Exception as exc:
                print(f"  WARN phrase={phrase!r}: {exc}")
                df = df_tagged = 0
            precision = (df_tagged / df) if df > 0 else 0.0
            per_phrase.append({
                "phrase": phrase,
                "src": src,
                "df": df,
                "df_tagged": df_tagged,
                "precision": precision,
            })

        # Mark substring-duplicates within this entry
        # (a phrase whose set of tagged docs is identical to a longer
        # phrase containing it is redundant — drop the shorter one).
        phrases_sorted = sorted(per_phrase, key=lambda p: -len(p["phrase"]))
        dropped_dupe: set[str] = set()
        for j, longer in enumerate(phrases_sorted):
            for shorter in phrases_sorted[j + 1:]:
                if (
                    shorter["phrase"] in longer["phrase"]
                    and shorter["df_tagged"] == longer["df_tagged"]
                    and shorter["df"] == longer["df"]
                    and shorter["phrase"] != longer["phrase"]
                ):
                    dropped_dupe.add(shorter["phrase"])

        # Pick canonical: highest df among (precision >= min) phrases
        canonical_phrase = None
        canonical_score = -1.0
        for p in per_phrase:
            if p["precision"] >= RULE_KEEP_CANONICAL["min_precision"] and \
               p["df"] >= RULE_KEEP_CANONICAL["min_df"]:
                if p["df"] > canonical_score:
                    canonical_score = p["df"]
                    canonical_phrase = p["phrase"]

        # Apply rules
        for p in per_phrase:
            verdict = "KEEP"
            if p["phrase"] in dropped_dupe:
                verdict = "DROP_DUPE"
            elif p["df"] <= RULE_DROP_RARE["max_df"]:
                verdict = "DROP_RARE"
            elif p["df"] >= RULE_DROP_NOISY["min_df"] and \
                 p["precision"] < RULE_DROP_NOISY["max_precision"]:
                verdict = "DROP_NOISY"
            if p["phrase"] == canonical_phrase:
                verdict = "KEEP_CANONICAL"
            p["verdict"] = verdict

        # Print compact per-entry summary
        sorted_phrases = sorted(per_phrase, key=lambda p: -p["df"])
        canonical_line = f"  canonical={canonical_phrase!r}" if canonical_phrase else "  canonical=NONE"
        print(f"\n[{i+1}/{len(rows)}] {full_code}  (n_tagged={n_tagged}, n_phrases={len(per_phrase)}){canonical_line}")
        for p in sorted_phrases:
            print(
                f"  {p['verdict']:<16} {p['phrase']!r:<40} "
                f"df={p['df']:>6} df_tagged={p['df_tagged']:>6} "
                f"prec={p['precision']:.2f}  src={p['src']}"
            )

        # Accumulate for CSV
        for p in per_phrase:
            out_rows.append({
                "tag_code": full_code,
                "phrase": p["phrase"],
                "src": p["src"],
                "df": p["df"],
                "df_tagged": p["df_tagged"],
                "precision": round(p["precision"], 4),
                "verdict": p["verdict"],
                "is_canonical": p["phrase"] == canonical_phrase,
            })

    elapsed = time.time() - started
    csv_path = f"/tmp/lexicon_phrase_precision_{int(time.time())}.csv"
    if out_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nCSV written: {csv_path} ({len(out_rows)} phrases across {len(rows)} entries) in {elapsed:.1f}s")

    # Aggregate verdict counts
    verdict_counts = Counter(r["verdict"] for r in out_rows)
    print("\n=== AGGREGATE ===")
    for v, n in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {v:<16} {n:>5}")
    total = sum(verdict_counts.values())
    if total:
        kept = verdict_counts["KEEP"] + verdict_counts["KEEP_CANONICAL"]
        dropped = total - kept
        print(f"\nKept: {kept}/{total} ({kept/total:.0%})  Dropped: {dropped}/{total} ({dropped/total:.0%})")

    # Entries that lost their canonical (no high-precision high-df phrase exists)
    n_no_canonical = sum(
        1 for entry_code, group in defaultdict(list).items()
        if not any(r["is_canonical"] for r in group)
    )
    entries_with_canonical = len({
        r["tag_code"] for r in out_rows if r["is_canonical"]
    })
    n_no_canonical = len(rows) - entries_with_canonical
    print(f"Entries with canonical: {entries_with_canonical}/{len(rows)}")
    print(f"Entries with NO canonical (none of their phrases met df>=100 + prec>=0.85): {n_no_canonical}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
