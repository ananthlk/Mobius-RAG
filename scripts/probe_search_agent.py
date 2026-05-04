"""Probe the corpus_search_agent against the 17-query test set.

Runs each test query against the LOCAL agent endpoint, prints a structured
trace per query, and writes a summary CSV for cross-query comparison.

Usage:
    # 1. Start the rag service locally (in a separate terminal):
    cd ~/Mobius/mobius-rag
    export DATABASE_URL='postgresql+asyncpg://...'   # dev pg
    export EMBEDDING_PROVIDER=vertex
    export VERTEX_PROJECT_ID=mobius-os-dev
    export VERTEX_LOCATION=us-central1
    gcloud auth application-default login    # once per machine
    uvicorn app.main:app --reload --port 8001

    # 2. From another terminal:
    python3 scripts/probe_search_agent.py
    # or against the deployed dev:
    BASE_URL=https://mobius-rag-ortabkknqa-uc.a.run.app python3 scripts/probe_search_agent.py

Output:
    - Per-query trace block on stdout
    - /tmp/probe_search_agent_<timestamp>.csv summary
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request


BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001")
ENDPOINT = f"{BASE_URL.rstrip('/')}/api/skills/v1/corpus_search_agent"


# ── 17-query test set, organized by what each tests ─────────────────────────

TESTS: list[tuple[str, str, str]] = [
    # (group, expected_query_type, query)

    # Group A — Selectivity principle validation
    ("A", "CONCEPTUAL", "Sunshine Health prior authorization behavioral health"),
    ("A", "CONCEPTUAL", "Sunshine Health prior authorization for behavioral health providers"),
    ("A", "MIXED", "providers in Sunshine Health"),

    # Group B — PRECISION_DOMINANT
    ("B", "PRECISION_DOMINANT", "What does FL.UM.51 say?"),
    ("B", "PRECISION_DOMINANT", "Show me CP.MP.98 and FL.UM.51 policies"),
    ("B", "PRECISION_DOMINANT", "H0019 fee schedule"),

    # Group C — Intersection vs single-tag
    ("C", "CONCEPTUAL", "What does AHCA say about behavioral health"),
    ("C", "MIXED", "What is AHCA"),
    ("C", "MIXED", "appeals process across payers"),

    # Group D — Edge cases
    ("D", "VAGUE", "tell me about behavioral health"),
    ("D", "CONCEPTUAL", "DME PA"),
    ("D", "CONCEPTUAL", "Centene Sunshine Health Wellcare prior authorization"),

    # Group E — Known coverage gaps
    ("E", "CONCEPTUAL", "What is Sunshine Health's behavioral health PA day count"),
    ("E", "CONCEPTUAL", "What's new in Centene credentialing for 2026"),

    # Group F — Real CMHC coordinator-style questions
    ("F", "CONCEPTUAL", "How do I get credentialed with Centene"),
    ("F", "CONCEPTUAL", "What's the timely filing deadline for AHCA"),
    ("F", "CONCEPTUAL", "Does Sunshine cover ABA therapy"),
]


def post(query: str, k: int = 10) -> dict:
    body = json.dumps({"query": query, "k": k}).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json", "X-Caller": "probe"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def fmt_terms(items: list[dict]) -> str:
    if not items:
        return "—"
    return ", ".join(
        f"{x['term']}({x['selectivity']:.2f})"
        for x in items[:6]
    ) + (f", +{len(items)-6}" if len(items) > 6 else "")


def fmt_strategy_row(o: dict) -> str:
    flag = "✓" if o.get("succeeded") else "✗"
    return (
        f"  {flag} {o['strategy']:<14} q={o['query_used']!r:<60} "
        f"chunks={o['n_chunks']:>3}  rerank={o['top_rerank']:.2f}  "
        f"{o['elapsed_ms']}ms  ({o['note']})"
    )


def main() -> None:
    print(f"Hitting {ENDPOINT}\n")
    rows = []
    for group, expected_type, q in TESTS:
        print("=" * 78)
        print(f"[{group}] expected={expected_type}  query={q!r}")
        print("=" * 78)
        t0 = time.time()
        r = post(q)
        elapsed = (time.time() - t0) * 1000

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            print()
            rows.append({
                "group": group, "expected": expected_type, "query": q,
                "got_type": "ERROR", "confidence": "ERROR",
                "pool_size": 0, "n_strategies": 0,
                "elapsed_ms": int(elapsed), "error": r["error"][:200],
            })
            continue

        prof = r.get("query_profile") or {}
        part = r.get("term_partition") or {}
        pool = r.get("candidate_pool") or {}
        strats = r.get("strategies_tried") or []
        hint = r.get("improvement_hint")
        ok = (prof.get("query_type") == expected_type)
        type_marker = "✓" if ok else "✗"

        print(f"  classify:    type={prof.get('query_type')} {type_marker} (expected {expected_type})  "
              f"coverage={prof.get('coverage'):.2f}  "
              f"tags={prof.get('tag_matches')}  literals={prof.get('literal_anchors')}")
        print(f"  partition:   REQUIRED=[{fmt_terms(part.get('required', []))}]")
        print(f"               BOOSTED =[{fmt_terms(part.get('boosted', []))}]")
        print(f"               DROP    =[{fmt_terms(part.get('dropped', []))}]")
        print(f"  pool:        size={pool.get('size')}  used={pool.get('used_for_search')}  "
              f"relaxed={pool.get('relaxed')}  intersect={pool.get('intersect_codes')}  "
              f"dropped_in_relax={pool.get('relaxed_dropped_codes')}")
        print(f"  strategies:")
        for o in strats:
            print(fmt_strategy_row(o))
        print(f"  confidence:  {r.get('confidence')}  total_ms={r.get('telemetry',{}).get('total_ms')}")
        if hint:
            print(f"  hint:        ({'will-help' if hint['would_reframing_help'] else 'wont-help'}) "
                  f"{hint['suggestion'][:160]}")
        else:
            print(f"  hint:        (none)")
        # Top 3 chunks
        chunks = r.get("chunks") or []
        if chunks:
            print(f"  top_chunks:")
            for c in chunks[:3]:
                doc = c.get("document_name") or c.get("document_filename") or "?"
                print(f"    - {doc[:48]:<48} p.{c.get('page_number')} arms={c.get('retrieval_arms')} "
                      f"rerank={c.get('rerank_score'):.2f}")
        print()

        rows.append({
            "group": group,
            "expected": expected_type,
            "query": q,
            "got_type": prof.get("query_type"),
            "type_match": ok,
            "coverage": prof.get("coverage"),
            "n_required": len(part.get("required") or []),
            "n_boosted": len(part.get("boosted") or []),
            "n_dropped": len(part.get("dropped") or []),
            "pool_size": pool.get("size"),
            "pool_relaxed": pool.get("relaxed"),
            "n_strategies": len(strats),
            "any_strategy_ok": any(s.get("succeeded") for s in strats),
            "first_strategy": strats[0]["strategy"] if strats else None,
            "first_strategy_ok": (strats[0]["succeeded"] if strats else None),
            "confidence": r.get("confidence"),
            "n_chunks_returned": len(chunks),
            "top_rerank": (chunks[0].get("rerank_score") if chunks else None),
            "elapsed_ms": int(elapsed),
            "hint": (hint["suggestion"][:120] if hint else None),
        })

    csv_path = f"/tmp/probe_search_agent_{int(time.time())}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary written: {csv_path}")
        # quick aggregates
        type_match_rate = sum(1 for r in rows if r.get("type_match")) / len(rows)
        any_ok_rate = sum(1 for r in rows if r.get("any_strategy_ok")) / len(rows)
        print(f"Aggregates: classifier_match={type_match_rate:.0%}  "
              f"any_strategy_succeeded={any_ok_rate:.0%}")


if __name__ == "__main__":
    main()
