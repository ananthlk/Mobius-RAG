"""Run each query through the full pipeline 3x per Router allocator
(greedy/optimizer/bayesian), forced via mode_override instead of leaving it
to the A/B/C draw -- so every loop gets exercised deterministically instead
of only whichever one the draw happens to pick. Diffs output ignoring
latency/timing fields and the per-call decision_id (both expected to vary
run to run; everything else should not).

Usage (from mobius-rag/):
    .venv/bin/python scripts/run_determinism_check.py
"""

from __future__ import annotations

import asyncio
import sys
import json
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

QUERIES = [
    "What is the timely filing deadline for Sunshine Health FL Medicaid claims?",
    "Does Aetna Better Health of Florida require prior authorization for telehealth?",
    "What is Molina Healthcare's appeal deadline for denied claims in Florida?",
]

MODES = ["greedy", "optimizer", "bayesian"]

# Fields known to be timing/latency, and the per-call decision_id -- expected
# to vary run to run, not part of "the story."
IGNORE_KEYS = {
    "gate_ms", "reformat_ms", "slots_ms", "pool_ms", "router_ms", "fillers_ms",
    "total_ms", "probe_ms", "segment_ms", "latency_ms", "embed_ms", "vector_ms",
    "doc_narrow_ms", "tag_select_ms", "inherited_ms", "dedup_ms", "neighbor_ms",
    "latency_before_ms", "latency_after_ms", "latency_p50_ms", "latency_allowance_ms",
    "decision_id", "router_allocate_ms", "router_shadow_ms", "allocate_ms",
}


def strip_ignored(obj):
    if isinstance(obj, dict):
        return {k: strip_ignored(v) for k, v in obj.items() if k not in IGNORE_KEYS}
    if isinstance(obj, list):
        return [strip_ignored(x) for x in obj]
    return obj


def to_dict(o):
    if hasattr(o, "__dataclass_fields__"):
        return {k: to_dict(getattr(o, k)) for k in o.__dataclass_fields__}
    if isinstance(o, list):
        return [to_dict(x) for x in o]
    if isinstance(o, dict):
        return {k: to_dict(v) for k, v in o.items()}
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if hasattr(o, "value"):
        return o.value
    return str(o)


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list):
        for idx, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{idx}]"))
    else:
        out[prefix] = d
    return out


def diff_report(runs):
    base = json.dumps(runs[0], sort_keys=True)
    all_match = True
    lines = []
    for i, r in enumerate(runs[1:], start=2):
        cur = json.dumps(r, sort_keys=True)
        if cur != base:
            all_match = False
            f1, f2 = flatten(runs[0]), flatten(r)
            for k in sorted(set(f1) | set(f2)):
                if f1.get(k) != f2.get(k):
                    lines.append(f"    run1 vs run{i} :: {k}: {f1.get(k)!r} != {f2.get(k)!r}")
    return all_match, lines


async def main():
    async with AsyncSessionLocal() as db:
        for query in QUERIES:
            print(f"\n=== {query} ===")

            # Shape + Pool are shared/reused across mode loops below -- they
            # don't depend on the allocator, and re-running them per mode
            # would conflate their own determinism with Router's.
            gate_result = await run_gate(db, query)
            reformat_result = await run_reformat(db, gate_result)
            structure_result = run_structure(reformat_result, caller_mode=None)
            slots_result = run_slots(structure_result)
            payer_slug = extract_payer_slug(gate_result.j_codes)
            payer_context = await resolve_payer_context(db, payer_slug) if payer_slug else None

            adapter = PublicSourceAdapter(db)
            rq = structure_result.rewritten_queries[0] if structure_result.rewritten_queries else query
            pool_result = await run_pool_for_query(
                db, rq, gate_result, structure_result.resource_posture, adapter,
            )
            pool_metadata = _build_pool_metadata(slots_result.slots, [pool_result])

            for mode in MODES:
                print(f"  -- mode: {mode} --")
                runs = []
                for i in range(3):
                    ctx = RoutingContext(
                        query=query,
                        agent_id="retriever-determinism-check",
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
                    router_decision = await router_route(AsyncSessionLocal, ctx)
                    filled_shape = await _run_fillers_simple(
                        slots_result.slots, [pool_result], router_decision, query, gate_result,
                    )
                    d = strip_ignored({
                        "router_decision": to_dict(router_decision),
                        "filled_shape": to_dict(filled_shape),
                    })
                    runs.append(d)
                    print(f"    run {i+1} done")

                all_match, lines = diff_report(runs)
                if all_match:
                    print(f"    DETERMINISTIC (mode={mode}): all 3 runs identical")
                else:
                    print(f"    NON-DETERMINISTIC (mode={mode}):")
                    for line in lines:
                        print(line)


if __name__ == "__main__":
    asyncio.run(main())
