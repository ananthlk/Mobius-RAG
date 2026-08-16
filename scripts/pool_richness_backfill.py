"""Backfill pool_size + bm25/vector top-10 mean/std for the 22-query bank,
WITHOUT re-running the expensive tripled-LLM-call eval pass -- Pool is
identical regardless of which filler is forced (fillers execute AFTER Pool
is built), and a/b/d/s fillers make zero LLM calls during retrieval, so
forcing 'a' here is cheap: real Gate->Reformat->Structure->Slots->Pool,
same pool everyone shares, no eval/judge calls at all.

Correlates against the ALREADY-COMPUTED a/b recall/synthesis_loss from the
completed bank-summary runs (job ids passed in), to test Ananth's
hypothesis: does raw pool richness/density predict final recall + how much
loss synthesis introduces?

Usage: .venv/bin/python scripts/pool_richness_backfill.py
"""
from __future__ import annotations
import asyncio, json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from app.database import AsyncSessionLocal  # noqa: E402
from app.services.retriever.orchestrator import run_retriever_partial_with_retry  # noqa: E402

BANK_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries_cmhc.yaml"
A_JOB = "33a2aef2ebf9"
B_JOB = "03bc108d622e"


def pool_stats(values):
    if not values:
        return {"n": 0, "mean": None, "std": None}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {"n": n, "mean": round(mean, 4), "std": round(var ** 0.5, 4)}


def get_bank_summaries(job_id):
    url = f"http://localhost:8000/admin/trace-explorer/run-bank/status?job_id={job_id}"
    with urllib.request.urlopen(url) as r:
        d = json.load(r)
    return {s["id"]: s for s in d["summaries"]}


async def main():
    bank = yaml.safe_load(open(BANK_PATH))["queries"]
    a_summaries = get_bank_summaries(A_JOB)
    b_summaries = get_bank_summaries(B_JOB)

    rows = []
    async with AsyncSessionLocal() as db:
        for q in bank:
            qid, query = q["id"], q["query"]
            result = await run_retriever_partial_with_retry(
                db, query, caller_mode="chat.default", forced_strategy="a",
            )
            all_cands = [c for p in result.pool for c in (p.candidates or [])]
            bm25 = sorted((c.bm25_score for c in all_cands if c.bm25_score is not None), reverse=True)
            vec = sorted((c.vector_similarity for c in all_cands if c.vector_similarity is not None), reverse=True)
            row = {
                "id": qid, "pool_size": len(all_cands),
                "bm25_top10": pool_stats(bm25[:10]), "vector_top10": pool_stats(vec[:10]),
                "a_recall": (a_summaries.get(qid) or {}).get("recall"),
                "a_recall_answer": (a_summaries.get(qid) or {}).get("recall_answer"),
                "b_recall": (b_summaries.get(qid) or {}).get("recall"),
                "b_recall_answer": (b_summaries.get(qid) or {}).get("recall_answer"),
            }
            rows.append(row)
            print(f"{qid}: pool_size={row['pool_size']} bm25_top10_mean={row['bm25_top10']['mean']} "
                  f"vector_top10_mean={row['vector_top10']['mean']} | a_recall={row['a_recall']} b_recall={row['b_recall']}")

    json.dump(rows, open("/tmp/pool_richness_backfill.json", "w"), indent=2)
    print(f"\nWrote {len(rows)} rows to /tmp/pool_richness_backfill.json")


if __name__ == "__main__":
    asyncio.run(main())
