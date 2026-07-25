"""Integration-test Observer's real evaluate() against real live filler
results (real DB, real LLM/web calls where applicable) -- WITHOUT wiring it
into orchestrator.py's production loop (that stays gated per Eval).
Compares Observer's real verdict against the orchestrator's current
occupancy-only stopgap, to see where they'd actually diverge.

Usage (from mobius-rag/):
    .venv/bin/python scripts/integration_test_observer.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import AsyncSessionLocal  # noqa: E402
from app.services.retriever.shape.gate import run_gate  # noqa: E402
from app.services.retriever.shape.reformat import run_reformat  # noqa: E402
from app.services.retriever.shape.structure import run_structure  # noqa: E402
from app.services.retriever.shape.slots import run_slots, AnswerShapeResult  # noqa: E402
from app.services.retriever.pool.pool import run_pool_for_query  # noqa: E402
from app.services.retriever.pool.public_adapter import PublicSourceAdapter  # noqa: E402
from app.services.retriever.fillers.payer_context import (  # noqa: E402
    extract_payer_slug, resolve_payer_context,
)
from app.services.retriever.fillers.filler_a import fill_shape_bm25  # noqa: E402
from app.services.retriever.fillers.filler_b import fill_shape_vector  # noqa: E402
from app.services.retriever.fillers.filler_c import fill_shape_llm_retrieval  # noqa: E402
from app.services.retriever.fillers.filler_d import fill_shape_external  # noqa: E402
from app.services.retriever.fillers.filler_s import fill_shape_fact_store  # noqa: E402
from app.services.retriever.observer import evaluate  # noqa: E402

QUERIES = [
    "What is the timely filing deadline for Sunshine Health FL Medicaid claims?",
    "Does Sunshine Health require prior authorization for residential substance use treatment under code H0019?",
]
STRATEGIES = ["a", "b", "c", "d", "s"]


def _stopgap_verdict(occupancy: int, error) -> str:
    """The orchestrator's current live stopgap, for comparison only."""
    if error is not None:
        return "ERROR"
    return "SATISFIED" if occupancy > 0 else "WOULD_BENEFIT_or_EXHAUSTED(stopgap can't distinguish)"


async def run_one(db, query_text, strategy):
    gate = await run_gate(db, query_text)
    reformat = await run_reformat(db, gate)
    structure = run_structure(reformat, caller_mode=None)
    slots = run_slots(structure)
    payer_slug = extract_payer_slug(gate.j_codes)
    payer_context = await resolve_payer_context(db, payer_slug) if payer_slug else None
    adapter = PublicSourceAdapter(db)
    rq = structure.rewritten_queries[0] if structure.rewritten_queries else query_text
    pool = await run_pool_for_query(db, rq, gate, structure.resource_posture, adapter)
    shape = AnswerShapeResult(query=query_text, posture=None, slots=slots.slots, reason="", slots_ms=0)
    tag_matches = [*gate.d_codes, *gate.j_codes, *gate.p_codes]

    try:
        if strategy == "a":
            result = fill_shape_bm25(pool, shape)
        elif strategy == "b":
            result = fill_shape_vector(pool, shape)
        elif strategy == "c":
            result = await fill_shape_llm_retrieval(
                pool, shape, query_text, db=db, agent_id="observer-integration-test",
                tag_matches=tag_matches,
            )
        elif strategy == "d":
            result = await fill_shape_external(
                pool, shape, query_text, db=db, agent_id="observer-integration-test",
                tag_matches=tag_matches, payer_context=payer_context,
            )
        else:
            result = await fill_shape_fact_store(pool, shape, query_text, tag_matches=tag_matches)
        filled_slot = result.slots[0]
        error = None
    except Exception as exc:
        from app.services.retriever.fillers.contracts import FilledSlot
        filled_slot = FilledSlot(
            slot_id=slots.slots[0].slot_id, slot_semantics=slots.slots[0].slot_semantics,
            capacity=slots.slots[0].capacity, required=slots.slots[0].required,
            chunks=[], occupancy=0, under_filled=True, over_filled=False,
        )
        error = repr(exc)

    return filled_slot, error


async def main():
    divergences = []
    async with AsyncSessionLocal() as db:
        for query in QUERIES:
            print(f"\n=== {query!r} ===")
            for strategy in STRATEGIES:
                filled_slot, error = await run_one(db, query, strategy)
                stopgap = _stopgap_verdict(filled_slot.occupancy, error)

                if error is not None:
                    print(f"  {strategy}: filler raised ({error}) -- Observer not called (orchestrator would mark ERROR)")
                    continue

                verdict, reason = evaluate(strategy, filled_slot, attempt_number=1, max_attempts=3)
                flag = ""
                # Flag the interesting case: stopgap said SATISFIED (occupancy>0)
                # but Observer's real capacity-aware bar says otherwise.
                if filled_slot.occupancy > 0 and verdict != "SATISFIED":
                    flag = "  <-- DIVERGES from stopgap (stopgap=SATISFIED, Observer=" + verdict + ")"
                    divergences.append((query, strategy, filled_slot.occupancy, filled_slot.capacity, verdict, reason))
                print(f"  {strategy}: occupancy={filled_slot.occupancy}/{filled_slot.capacity} "
                      f"under_filled={filled_slot.under_filled} -> Observer=({verdict}, {reason!r}) "
                      f"[stopgap would say: {stopgap}]{flag}")

    print("\n" + "=" * 90)
    print(f"DIVERGENCES from the live stopgap: {len(divergences)}")
    for q, s, occ, cap, verdict, reason in divergences:
        print(f"  {q[:50]!r:53s} strategy={s} occ={occ}/{cap} -> {verdict} ({reason})")


if __name__ == "__main__":
    asyncio.run(main())
