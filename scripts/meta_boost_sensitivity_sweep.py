"""For every query in eval/queries_cmhc.yaml, forced through strategy a:
identify the pool candidate(s) that actually carry the must_facts (the
"gold" chunks), then sweep Filler a's meta_boost weight (currently 0.10,
_compute_rerank_score's fixed constant) to find:
  - if the gold chunk is OUTSIDE the current top-10: the minimum meta_boost
    weight at which it would enter the top-10 (a real full re-rank sweep,
    not a linear approximation -- every candidate's composite is
    recomputed and the whole pool re-sorted at each weight).
  - if the gold chunk is INSIDE the current top-10: the minimum meta_boost
    weight at which it would still be inside -- i.e. how far the weight
    could drop before it falls out.

Gold-chunk identification is a heuristic (must_facts are human-written,
not verbatim chunk text): extracts distinguishing tokens (number+unit
patterns like "180 days", quoted-looking multi-word phrases, and
capitalized proper nouns) from each must_fact and looks for candidates
whose text contains a strong majority of at least one fact's tokens.
Printed alongside each result so a bad heuristic match is visible, not
silently trusted.

Usage: DATABASE_URL=... .venv/bin/python scripts/meta_boost_sensitivity_sweep.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BANK_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries_cmhc.yaml"
WEIGHT_GRID = [round(x * 0.01, 2) for x in range(0, 101)]  # 0.00 .. 1.00 step 0.01
CURRENT_META_WEIGHT = 0.10
BM25_W, AUTH_W, COV_W, LEN_W = 0.55, 0.10, 0.10, 0.10
TOP_N = 10


def extract_fact_tokens(fact: str) -> list[str]:
    """Distinguishing substrings from a must_fact: number+unit ("180 days"),
    capitalized multi-word proper nouns ("Sunshine Health"), otherwise the
    3 longest words. Heuristic, not exhaustive."""
    tokens = []
    tokens += re.findall(r"\b\d+\s*(?:day|days|calendar day|calendar days|hour|hours)\b", fact, re.I)
    tokens += re.findall(r"\b(?:[A-Z][a-z]+\s){1,3}[A-Z][a-z]+\b", fact)
    if not tokens:
        words = sorted(set(re.findall(r"[a-zA-Z]{4,}", fact)), key=len, reverse=True)
        tokens = words[:3]
    return [t.lower() for t in tokens]


def find_gold_candidates(candidates, must_facts):
    """Candidates whose text contains tokens from at least one must_fact,
    ranked by how many distinct facts they match. Returns list of
    (candidate, matched_fact_texts)."""
    fact_tokens = [(f, extract_fact_tokens(f)) for f in (must_facts or [])]
    scored = []
    for c in candidates:
        text_lower = (c.text or "").lower()
        matched = []
        for fact, tokens in fact_tokens:
            if not tokens:
                continue
            hits = sum(1 for t in tokens if t in text_lower)
            if hits >= max(1, len(tokens) - 1):  # allow missing at most 1 token
                matched.append(fact)
        if matched:
            scored.append((c, matched))
    scored.sort(key=lambda pair: len(pair[1]), reverse=True)
    return scored


def composite_at_weight(cand, query, required_phrases, boosted_phrases, meta_w, precomputed):
    from app.services.retriever.fillers.filler_a import (
        _compute_authority_score, _compute_tag_coverage_score, _compute_length_score,
        _compute_meta_boost_score,
    )
    key = cand.chunk_id
    if key not in precomputed:
        bm25 = cand.bm25_score or 0.0
        auth = _compute_authority_score(cand.authority_level)
        cov = _compute_tag_coverage_score(cand.tags)
        length = _compute_length_score(cand.text)
        meta = _compute_meta_boost_score(cand.text, cand.tags, required_phrases, boosted_phrases)
        base = BM25_W * bm25 + AUTH_W * auth + COV_W * cov + LEN_W * length
        precomputed[key] = (base, meta)
    base, meta = precomputed[key]
    return base + meta_w * meta


def rank_at_weight(scored_candidates, query, required_phrases, boosted_phrases, meta_w, precomputed):
    scores = [
        (c, composite_at_weight(c, query, required_phrases, boosted_phrases, meta_w, precomputed))
        for c in scored_candidates
    ]
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return {c.chunk_id: i + 1 for i, (c, _) in enumerate(scores)}


async def analyze_query(db, q):
    from app.services.retriever.shape.gate import run_gate
    from app.services.retriever.shape.reformat import run_reformat
    from app.services.retriever.shape.structure import run_structure
    from app.services.retriever.shape.slots import run_slots
    from app.services.retriever.pool.public_adapter import PublicSourceAdapter
    from app.services.retriever.pool.pool import run_pool_for_query

    query_text = q["query"]
    must_facts = q.get("must_facts") or []
    gate = await run_gate(db, query_text)
    reformat = await run_reformat(db, gate)
    structure = run_structure(reformat, caller_mode="chat.default")
    rp = structure.resource_posture
    slots_result = run_slots(structure)
    adapter = PublicSourceAdapter(db)
    slot0 = slots_result.slots[0]
    pool_result = await run_pool_for_query(db, slot0.rewritten_query or query_text, gate, rp, adapter)
    scored = [c for c in pool_result.candidates if c.bm25_score is not None]

    gold = find_gold_candidates(scored, must_facts)
    if not gold:
        return {"id": q["id"], "query": query_text, "gold_found": False}

    precomputed = {}
    results = []
    for cand, matched_facts in gold[:3]:  # cap: report top-3 gold matches per query
        ranks_by_weight = {}
        for w in WEIGHT_GRID:
            ranks = rank_at_weight(scored, pool_result.query, pool_result.required_phrases, pool_result.boosted_phrases, w, precomputed)
            ranks_by_weight[w] = ranks[cand.chunk_id]

        current_rank = ranks_by_weight[CURRENT_META_WEIGHT]
        currently_in_top10 = current_rank <= TOP_N

        if currently_in_top10:
            # sweep DOWN from current weight to find the floor before it exits top-10
            floor_w = CURRENT_META_WEIGHT
            for w in sorted(WEIGHT_GRID):
                if w > CURRENT_META_WEIGHT:
                    break
                if ranks_by_weight[w] <= TOP_N:
                    floor_w = w
                    break
            direction = "already in top-10"
        else:
            # sweep UP from current weight to find the weight needed to enter top-10
            floor_w = None
            for w in sorted(WEIGHT_GRID):
                if w < CURRENT_META_WEIGHT:
                    continue
                if ranks_by_weight[w] <= TOP_N:
                    floor_w = w
                    break
            direction = "needs higher weight to enter top-10"

        results.append({
            "chunk_text": (cand.text or "")[:90],
            "matched_facts": matched_facts,
            "current_rank": current_rank,
            "currently_in_top10": currently_in_top10,
            "direction": direction,
            "crossover_weight": floor_w,
        })

    return {"id": q["id"], "query": query_text, "gold_found": True, "n_scored": len(scored), "gold_results": results}


async def main():
    from app.database import AsyncSessionLocal
    bank = yaml.safe_load(open(BANK_PATH, encoding="utf-8"))["queries"]
    async with AsyncSessionLocal() as db:
        for q in bank:
            try:
                result = await analyze_query(db, q)
            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                print(f"{q['id']}: ERROR {exc}")
                _tb.print_exc()
                continue
            print(f"\n{'='*80}\n{result['id']}: {result['query']}\n{'='*80}")
            if not result["gold_found"]:
                print("  no gold chunk identified (heuristic miss on must_facts) -- skipped")
                continue
            print(f"  ({result['n_scored']} scored candidates)")
            for r in result["gold_results"]:
                print(f"  matched facts: {r['matched_facts']}")
                print(f"  chunk: {r['chunk_text']!r}")
                print(f"  current rank (w=0.10): {r['current_rank']}  [{r['direction']}]")
                print(f"  crossover weight: {r['crossover_weight']}")
                print()


asyncio.run(main())
