"""One-off diagnostic (Eval's exact ask, 2026-07-24): run cmhc002 through
run_retriever_partial DIRECTLY (not the calibration harness) and print the
FilledShape's per-slot chunks, to determine whether a/b/c/d are genuinely
empty in the integrated orchestrator path, or whether the calibration
harness was dropping real chunks before they reached the judge.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import AsyncSessionLocal
from app.services.retriever.orchestrator import run_retriever_partial

QUERY = "Does Sunshine Health require prior authorization for residential substance use treatment under code H0019?"


async def main():
    async with AsyncSessionLocal() as db:
        result = await run_retriever_partial(db, QUERY)

        print("=== POOL ===")
        for p in result.pool:
            print("  n_candidates:", len(p.candidates), "pool_ms:", p.pool_ms)

        print("=== ROUTER ===")
        if result.router_decision:
            rd = result.router_decision
            print("  per_slot ladder:", rd.routing_ladder.per_slot)
            print("  per_slot_status:", rd.routing_ladder.per_slot_status)
            print("  outcome:", rd.routing_ladder.outcome)
            print("  latency_allowance_ms:", getattr(rd.trace, "latency_allowance_ms", None))
        else:
            print("  router_decision is None")

        print("=== FILLED SHAPE ===")
        if result.filled_shape:
            print("  emit:", result.filled_shape.emit)
            for slot in result.filled_shape.slots:
                print(f"  slot={slot.slot_id} occupancy={slot.occupancy} under_filled={slot.under_filled}")
                for c in slot.chunks:
                    print("    chunk:", c.chunk_id, c.source_type, getattr(c, "assignment_reason", None), (c.text or "")[:70])
        else:
            print("  filled_shape is None")

        print("=== SYNTHESIS ===")
        if result.synthesis:
            print("  n_citations:", len(result.synthesis.citations))
            print("  telemetry:", result.synthesis.telemetry)
        else:
            print("  synthesis is None")

        print("=== TIMING ===")
        print("  gate_ms:", result.gate_ms, "reformat_ms:", result.reformat_ms, "slots_ms:", result.slots_ms,
              "pool_ms:", result.pool_ms, "router_ms:", result.router_ms, "fillers_ms:", result.fillers_ms,
              "synthesis_ms:", result.synthesis_ms, "total_ms:", result.total_ms)


asyncio.run(main())
