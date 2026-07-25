"""Latency stats grouped by the strategy Router actually picked, across the
same 3 queries x 3 allocators x 3 repeats used in the determinism check.

Usage (from mobius-rag/):
    .venv/bin/python scripts/run_latency_by_strategy.py
"""

from __future__ import annotations

import asyncio
import sys
import statistics
from collections import defaultdict
from pathlib import Path

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
import time  # noqa: E402

QUERIES = [
    "What is the timely filing deadline for Sunshine Health FL Medicaid claims?",
    "Does Aetna Better Health of Florida require prior authorization for telehealth?",
    "What is Molina Healthcare's appeal deadline for denied claims in Florida?",
]
MODES = ["greedy", "optimizer", "bayesian"]

# rows: list of dicts {strategy_chain, query, mode, router_ms, fillers_ms, gate_ms, pool_ms, total_ms}
rows: list[dict] = []


def chain_key(routing_ladder) -> str:
    chains = sorted(tuple(v) for v in routing_ladder.per_slot.values())
    return "+".join("/".join(c) for c in chains) or "(none)"


async def main():
    async with AsyncSessionLocal() as db:
        for query in QUERIES:
            t0 = time.monotonic()
            gate_result = await run_gate(db, query)
            gate_ms = gate_result.gate_ms
            reformat_result = await run_reformat(db, gate_result)
            structure_result = run_structure(reformat_result, caller_mode=None)
            slots_result = run_slots(structure_result)
            payer_slug = extract_payer_slug(gate_result.j_codes)
            payer_context = await resolve_payer_context(db, payer_slug) if payer_slug else None

            adapter = PublicSourceAdapter(db)
            rq = structure_result.rewritten_queries[0] if structure_result.rewritten_queries else query
            t_pool = time.monotonic()
            pool_result = await run_pool_for_query(
                db, rq, gate_result, structure_result.resource_posture, adapter,
            )
            pool_ms = pool_result.pool_ms
            pool_metadata = _build_pool_metadata(slots_result.slots, [pool_result])

            for mode in MODES:
                for i in range(3):
                    ctx = RoutingContext(
                        query=query,
                        agent_id="retriever-latency-check",
                        mode_override=mode,
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
                    filled_shape = await _run_fillers_simple(
                        slots_result.slots, [pool_result], router_decision, query, gate_result,
                    )
                    fillers_ms = int((time.monotonic() - t_fillers) * 1000)

                    strategy = chain_key(router_decision.routing_ladder)
                    rows.append({
                        "query": query, "mode": mode, "strategy": strategy,
                        "gate_ms": gate_ms, "pool_ms": pool_ms,
                        "router_ms": router_ms, "fillers_ms": fillers_ms,
                        "react_ms": router_ms + fillers_ms,
                    })
                    print(f"{query[:35]!r:38s} mode={mode:9s} strategy={strategy:12s} "
                          f"router={router_ms:4d}ms fillers={fillers_ms:5d}ms react={router_ms+fillers_ms:5d}ms")

    print("\n" + "=" * 90)
    print("LATENCY STATS BY STRATEGY PICKED (react-loop = router_ms + fillers_ms; gate/pool excluded per Ananth's accounting note)")
    print("=" * 90)

    by_strategy = defaultdict(list)
    for r in rows:
        by_strategy[r["strategy"]].append(r)

    header = f"{'strategy':14s} {'n':>3s} {'router_ms':>18s} {'fillers_ms':>18s} {'react_ms (router+fillers)':>26s}"
    print(header)
    print("-" * len(header))
    for strategy, rs in sorted(by_strategy.items(), key=lambda kv: -len(kv[1])):
        def fmt_stats(key):
            vals = [r[key] for r in rs]
            return f"min{min(vals):>4d} avg{statistics.mean(vals):>6.0f} max{max(vals):>4d}"
        print(f"{strategy:14s} {len(rs):>3d} {fmt_stats('router_ms'):>18s} {fmt_stats('fillers_ms'):>18s} {fmt_stats('react_ms'):>26s}")

    print("\nPer-query x mode raw detail:")
    for r in rows:
        pass  # already printed above during the run

    print("\nGate/Pool (shared per query, not attributable to a strategy choice):")
    seen = set()
    for r in rows:
        if r["query"] not in seen:
            seen.add(r["query"])
            print(f"  {r['query'][:60]!r:63s} gate={r['gate_ms']:5d}ms pool={r['pool_ms']:5d}ms")


if __name__ == "__main__":
    asyncio.run(main())
