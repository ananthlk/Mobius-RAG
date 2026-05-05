"""Quick timing harness for the 22 CMHC queries.

Streams wall-clock + BM25/rerank arm breakdown as each query completes.
Usage:
    python -m eval.time_cmhc --endpoint https://mobius-rag-1032922478554.us-central1.run.app/api/skills/v1/corpus_search_agent
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import yaml


QUERIES_FILE = Path(__file__).parent / "queries_cmhc.yaml"


async def run(endpoint: str, skip_synthesis: bool = True) -> None:
    with open(QUERIES_FILE) as f:
        bank = yaml.safe_load(f)

    queries = bank if isinstance(bank, list) else bank.get("queries", [])
    client = httpx.AsyncClient(timeout=120)

    totals: list[float] = []
    bm25_totals: list[float] = []
    rerank_totals: list[float] = []

    print(f"Running {len(queries)} queries against {endpoint}\n")

    for q in queries:
        qid = q.get("id", "?")
        text = q.get("query") or q.get("text", "")
        t0 = time.time()
        try:
            r = await client.post(endpoint, json={
                "query": text,
                "skip_synthesis": skip_synthesis,
            })
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            wall = (time.time() - t0) * 1000
            print(f"[{qid}] ERROR {wall:.0f}ms  {exc}")
            continue

        wall = (time.time() - t0) * 1000
        tel = data.get("telemetry") or {}
        agent_ms = tel.get("total_ms", 0)
        strats = data.get("strategies_tried") or []
        bm25_ms = sum(
            (s.get("telemetry") or {}).get("bm25_ms", 0) or 0
            for s in (strats if isinstance(strats, list) else [])
        )
        rerank_ms = sum(
            (s.get("telemetry") or {}).get("rerank_ms", 0) or 0
            for s in (strats if isinstance(strats, list) else [])
        )
        strategy = data.get("strategy_used") or tel.get("strategy_used", "?")
        gate = (data.get("gate") or {}).get("passed", True)
        gate_str = "✓" if gate else "✗"
        chunks = len(data.get("chunks") or [])

        totals.append(wall)
        bm25_totals.append(bm25_ms)
        rerank_totals.append(rerank_ms)

        print(
            f"[{qid}] gate={gate_str}  strategy={strategy}  chunks={chunks}"
            f"  wall={wall:.0f}ms  agent={agent_ms:.0f}ms"
        )

    await client.aclose()

    if totals:
        avg_wall = sum(totals) / len(totals)
        avg_bm25 = sum(bm25_totals) / len(bm25_totals)
        avg_rerank = sum(rerank_totals) / len(rerank_totals)
        print(f"\n{'='*60}")
        print(f"SUMMARY  n={len(totals)}")
        print(f"  avg_wall={avg_wall:.0f}ms")
        sorted_t = sorted(totals)
        print(f"  p50_wall={sorted_t[len(totals)//2]:.0f}ms  p90_wall={sorted_t[int(len(totals)*0.9)]:.0f}ms")
        print(f"  slow (>5s): {[round(t) for t in sorted_t if t > 5000]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="https://mobius-rag-1032922478554.us-central1.run.app/api/skills/v1/corpus_search_agent")
    parser.add_argument("--no-skip-synthesis", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.endpoint, skip_synthesis=not args.no_skip_synthesis))
