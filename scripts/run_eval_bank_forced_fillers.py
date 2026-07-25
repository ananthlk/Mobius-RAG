"""Run every query in eval/queries_cmhc.yaml through each of the 5 real
fillers (a/b/c/d/s), forced individually via RoutingContext.forced_strategy,
and dump a real artifact (raw chunks + latency, not just scores) for Eval
to grade precision/recall against must_facts/golden_answer/forbidden_facts.

Usage (from mobius-rag/):
    .venv/bin/python scripts/run_eval_bank_forced_fillers.py
"""

from __future__ import annotations

import asyncio
import sys
import json
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import AsyncSessionLocal  # noqa: E402
from app.services.retriever.shape.gate import run_gate  # noqa: E402
from app.services.retriever.shape.reformat import run_reformat  # noqa: E402
from app.services.retriever.shape.structure import run_structure  # noqa: E402
from app.services.retriever.shape.slots import run_slots  # noqa: E402
from app.services.retriever.pool.pool import run_pool_for_query  # noqa: E402
from app.services.retriever.pool.public_adapter import PublicSourceAdapter  # noqa: E402
from app.services.retriever.fillers.payer_context import (  # noqa: E402
    extract_payer_slug, resolve_payer_context,
)
from app.services.router.decision import RoutingContext, ResourcePosture as RouterResourcePosture  # noqa: E402
from app.services.router.router import route as router_route  # noqa: E402
from app.services.retriever.orchestrator import _build_pool_metadata, _run_fillers_simple  # noqa: E402

BANK_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries_cmhc.yaml"
STRATEGIES = ["a", "b", "c", "d", "s"]
OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "artifacts" / "forced_filler_bank_run.json"


async def prepare_query(db, query_text):
    """Gate through Pool ONCE per query -- these don't depend on which
    filler strategy gets forced, so re-running them per strategy (the
    original bug here) pays Gate's 1-8s and Pool's 3-6s five times over
    instead of once, which is what made the first attempt at this script
    take >11 real minutes to finish a single query."""
    gate_result = await run_gate(db, query_text)
    reformat_result = await run_reformat(db, gate_result)
    structure_result = run_structure(reformat_result, caller_mode=None)
    slots_result = run_slots(structure_result)
    payer_slug = extract_payer_slug(gate_result.j_codes)
    payer_context = await resolve_payer_context(db, payer_slug) if payer_slug else None

    adapter = PublicSourceAdapter(db)
    rq = structure_result.rewritten_queries[0] if structure_result.rewritten_queries else query_text
    pool_result = await run_pool_for_query(
        db, rq, gate_result, structure_result.resource_posture, adapter,
    )
    pool_metadata = _build_pool_metadata(slots_result.slots, [pool_result])
    return gate_result, structure_result, slots_result, payer_context, pool_result, pool_metadata


async def run_one(db, query_text, strategy, prepared):
    gate_result, structure_result, slots_result, payer_context, pool_result, pool_metadata = prepared

    ctx = RoutingContext(
        query=query_text,
        agent_id="retriever-eval-bank-run",
        forced_strategy=strategy,
        resource_posture=RouterResourcePosture(
            speed_budget=structure_result.resource_posture.speed_budget,
            confidence_bar=structure_result.resource_posture.confidence_bar,
            max_attempts_per_slot=structure_result.resource_posture.max_attempts,
            caller_mode="chat.default",
        ),
        slots=slots_result.slots,
        pool_metadata=pool_metadata,
        gate_j_codes=gate_result.j_codes,
        gate_d_codes=gate_result.d_codes,
        payer_crawlable=(payer_context.crawlable if payer_context else None),
    )
    t_router = time.monotonic()
    router_decision = await router_route(AsyncSessionLocal, ctx)
    router_ms = int((time.monotonic() - t_router) * 1000)

    t_fillers = time.monotonic()
    try:
        filled_shape = await _run_fillers_simple(
            db, slots_result.slots, [pool_result], router_decision,
            query_text, gate_result, payer_context,
        )
        fillers_ms = int((time.monotonic() - t_fillers) * 1000)
        chunks = [
            {
                "chunk_id": c.chunk_id, "text": c.text, "source_type": c.source_type,
                "original_score": c.original_score, "assignment_reason": c.assignment_reason,
                "url": getattr(c, "url", None), "document_id": c.document_id,
            }
            for slot in filled_shape.slots for c in slot.chunks
        ]
        occupancy = filled_shape.total_chunks_assigned
        error = None
    except Exception as exc:
        fillers_ms = int((time.monotonic() - t_fillers) * 1000)
        chunks = []
        occupancy = 0
        error = repr(exc)

    return {
        "strategy": strategy,
        "router_ms": router_ms,
        "fillers_ms": fillers_ms,
        "react_ms": router_ms + fillers_ms,
        "occupancy": occupancy,
        "chunks": chunks,
        "error": error,
    }


async def main():
    bank = yaml.safe_load(BANK_PATH.read_text())
    queries = bank["queries"]
    print(f"Loaded {len(queries)} queries from {BANK_PATH} (bank_version={bank.get('bank_version')})")

    results = []
    async with AsyncSessionLocal() as db:
        for i, q in enumerate(queries, 1):
            qid, query_text = q["id"], q["query"]
            print(f"\n[{i}/{len(queries)}] {qid}: {query_text!r}")
            t_prep = time.monotonic()
            prepared = await prepare_query(db, query_text)
            print(f"    (gate+pool prepared once: {int((time.monotonic() - t_prep) * 1000)}ms,"
                  f" pool_size={len(prepared[4].candidates)})")
            per_strategy = {}
            for strategy in STRATEGIES:
                t0 = time.monotonic()
                r = await run_one(db, query_text, strategy, prepared)
                per_strategy[strategy] = r
                elapsed = int((time.monotonic() - t0) * 1000)
                n_chunks = len(r["chunks"])
                err = f" ERROR={r['error']}" if r["error"] else ""
                print(f"    {strategy}: react_ms={r['react_ms']:6d} occ={r['occupancy']:2d} "
                      f"chunks={n_chunks}{err} (total call {elapsed}ms)")

            results.append({
                "id": qid,
                "query": query_text,
                "golden_answer": q.get("golden_answer"),
                "must_facts": q.get("must_facts", []),
                "bonus_facts": q.get("bonus_facts", []),
                "forbidden_facts": q.get("forbidden_facts", []),
                "expected": q.get("expected", {}),
                "per_strategy": per_strategy,
            })

            # Flush incrementally so a crash partway through doesn't lose everything.
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_text(json.dumps({
                "bank_version": bank.get("bank_version"),
                "bank_path": str(BANK_PATH),
                "strategies": STRATEGIES,
                "n_queries_total": len(queries),
                "n_queries_completed": len(results),
                "results": results,
            }, indent=2))

    print(f"\nDone. Artifact written to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
