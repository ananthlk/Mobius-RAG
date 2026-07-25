"""DEMO ONLY -- runs the real multi-turn continuation loop (same mechanism
as orchestrator.py's _run_fillers_simple) but swaps in Observer's REAL
evaluate() instead of the production stopgap, for ONE real query, and
prints full per-turn telemetry: Router's ContinuationDecision fields,
Observer's per-attempt verdicts/reasons, and the final emit.

Deliberately kept SEPARATE from orchestrator.py -- wiring Observer's real
logic into the actual live loop stays gated per Eval's build-gate
(Fillers+Synthesis+a committed calibration plan). This script proves the
mechanism works end-to-end without crossing that gate.

Usage (from mobius-rag/):
    .venv/bin/python scripts/demo_observer_live_loop.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
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
from app.services.retriever.fillers.contracts import FilledSlot  # noqa: E402
from app.services.retriever.observer import evaluate  # noqa: E402
from app.services.router.decision import RoutingContext, ResourcePosture as RouterResourcePosture  # noqa: E402
from app.services.router.router import route as router_route  # noqa: E402
from app.services.router.continuation import SlotTurnInput, decide_continuation  # noqa: E402
from app.services.retriever.orchestrator import (  # noqa: E402
    _build_pool_metadata, _IMPLEMENTED_FILLERS,
)

QUERY = "What is the timely filing deadline for Sunshine Health FL Medicaid claims?"
_MAX_TURNS = 5


async def try_strategy(db, strategy, slot, pr, raw_query, gate_result, payer_context):
    single_shape = AnswerShapeResult(query=raw_query, posture=None, slots=[slot], reason="", slots_ms=0)
    tag_matches = [*gate_result.d_codes, *gate_result.j_codes, *gate_result.p_codes]
    try:
        if strategy == "a":
            result = fill_shape_bm25(pr, single_shape)
        elif strategy == "b":
            result = fill_shape_vector(pr, single_shape)
        elif strategy == "c":
            result = await fill_shape_llm_retrieval(pr, single_shape, raw_query, db=db, agent_id="observer-demo", tag_matches=tag_matches)
        elif strategy == "d":
            result = await fill_shape_external(pr, single_shape, raw_query, db=db, agent_id="observer-demo", tag_matches=tag_matches, payer_context=payer_context)
        else:
            result = await fill_shape_fact_store(pr, single_shape, raw_query, tag_matches=tag_matches)
        return result.slots[0], None
    except Exception as exc:
        return FilledSlot(
            slot_id=slot.slot_id, slot_semantics=slot.slot_semantics,
            capacity=slot.capacity, required=slot.required,
            chunks=[], occupancy=0, under_filled=True, over_filled=False,
        ), repr(exc)


async def main():
    telemetry = {"turns": []}
    async with AsyncSessionLocal() as db:
        gate = await run_gate(db, QUERY)
        reformat = await run_reformat(db, gate)
        structure = run_structure(reformat, caller_mode=None)
        slots = run_slots(structure)
        payer_slug = extract_payer_slug(gate.j_codes)
        payer_context = await resolve_payer_context(db, payer_slug) if payer_slug else None
        # DEMO WIDENING: loosen the token-budget payload gate so more than
        # one strategy survives on the ladder -- the real query above got
        # gated down to just "s" at Structure's real chat.default budget
        # (3000), which correctly stopped after one turn but didn't
        # exercise the advance/ride-along mechanism. This override is
        # purely to make the multi-turn behavior visible; it's not
        # something production code does.
        structure.resource_posture.token_budget = 50_000

        adapter = PublicSourceAdapter(db)
        rq = structure.rewritten_queries[0] if structure.rewritten_queries else QUERY
        pool = await run_pool_for_query(db, rq, gate, structure.resource_posture, adapter)
        pool_metadata = _build_pool_metadata(slots.slots, [pool])

        ctx = RoutingContext(
            query=QUERY, agent_id="observer-demo",
            resource_posture=RouterResourcePosture(
                speed_budget=structure.resource_posture.speed_budget,
                confidence_bar=structure.resource_posture.confidence_bar,
                max_attempts_per_slot=structure.resource_posture.max_attempts,
                caller_mode="chat.default",
                token_budget=structure.resource_posture.token_budget,
                authority_requirement=structure.resource_posture.authority_requirement,
            ),
            slots=slots.slots, pool_metadata=pool_metadata,
            gate_j_codes=gate.j_codes, gate_d_codes=gate.d_codes,
            payer_crawlable=(payer_context.crawlable if payer_context else None),
        )
        router_decision = await router_route(AsyncSessionLocal, ctx)
        latency_allowance_ms = getattr(router_decision.trace, "latency_allowance_ms", 0.0) or 0.0

        # DEMO OVERRIDE: Router's natural allocator planned only ["s"] for
        # this query (it genuinely satisfies on its own -- confirmed in the
        # unmodified run). To actually exercise the advance mechanism, this
        # substitutes a hand-built ladder ["b", "a"] for direct_answer --
        # NOT fabricated data: b is independently verified (earlier this
        # session) to return 0/10 on this exact query (the junk-boilerplate
        # cluster poisoning the vector arm), and a independently verified to
        # return 10/10 real content on the same query/pool. Every filler
        # call, Observer verdict, and continuation decision below is real;
        # only the ladder's rung SEQUENCE is demo-constructed to guarantee
        # we see a real advance rather than a natural single-rung case.
        demo_ladder = {"direct_answer": ["b", "a"]}
        print(f"Query: {QUERY!r}")
        print(f"Natural ladder (unused for this demo): {router_decision.routing_ladder.per_slot}")
        print(f"DEMO ladder (hand-built to exercise advance, both rungs real/verified on this query): {demo_ladder}")
        print(f"latency_allowance_ms: {latency_allowance_ms}\n")

        pool_by_query = {pool.query: pool}
        state = {}
        for slot in slots.slots:
            chain = [s for s in demo_ladder.get(slot.slot_id, []) if s in _IMPLEMENTED_FILLERS]
            state[slot.slot_id] = {"slot": slot, "chain": chain, "cursor": 0, "filled_slot": None, "verdict": None, "reason": ""}

        t0 = time.monotonic()

        # Turn 0
        for slot_id, st in state.items():
            slot = st["slot"]
            pr = pool_by_query.get(slot.rewritten_query) or pool
            if not st["chain"]:
                st["filled_slot"] = FilledSlot(slot_id=slot.slot_id, slot_semantics=slot.slot_semantics, capacity=slot.capacity, required=slot.required, chunks=[], occupancy=0, under_filled=True, over_filled=False)
                st["verdict"], st["reason"] = "EXHAUSTED_ATTEMPTS", "no_implemented_strategy"
                continue
            strategy = st["chain"][0]
            filled_slot, error = await try_strategy(db, strategy, slot, pr, QUERY, gate, payer_context)
            st["cursor"] = 1
            st["filled_slot"] = filled_slot
            if error is not None:
                st["verdict"], st["reason"] = "ERROR", f"filler raised: {error}"
            else:
                st["verdict"], st["reason"] = evaluate(
                    strategy, filled_slot, attempt_number=1, max_attempts=len(st["chain"]),
                )
            print(f"TURN 0: slot={slot_id} strategy={strategy} occupancy={filled_slot.occupancy}/{filled_slot.capacity} -> ({st['verdict']}, {st['reason']!r})")

        turn_num = 0
        for turn_num in range(1, _MAX_TURNS + 1):
            turn_inputs = [
                SlotTurnInput(slot_id=sid, remaining_rungs=tuple(st["chain"][st["cursor"]:]), verdict=st["verdict"], reason=st["reason"], required=st["slot"].required)
                for sid, st in state.items()
            ]
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            decision = decide_continuation(turn_inputs, elapsed_ms, latency_allowance_ms)
            telemetry["turns"].append({
                "turn": turn_num, "elapsed_ms": elapsed_ms, "decision": decision.to_dict(),
            })
            print(f"\nCONTINUATION DECISION (before turn {turn_num}): new_turn={decision.new_turn} "
                  f"justified_by={decision.justified_by} ride_along={decision.ride_along} "
                  f"dropped={decision.dropped} stop_reason={decision.stop_reason!r} "
                  f"envelope_ms={decision.envelope_ms} budget_remaining_ms={decision.budget_remaining_ms}")
            if not decision.new_turn:
                break
            for slot_id, strategy in decision.turn_rungs.items():
                st = state[slot_id]
                slot = st["slot"]
                pr = pool_by_query.get(slot.rewritten_query) or pool
                filled_slot, error = await try_strategy(db, strategy, slot, pr, QUERY, gate, payer_context)
                st["cursor"] += 1
                st["filled_slot"] = filled_slot
                if error is not None:
                    st["verdict"], st["reason"] = "ERROR", f"filler raised: {error}"
                else:
                    st["verdict"], st["reason"] = evaluate(
                        strategy, filled_slot, attempt_number=st["cursor"], max_attempts=len(st["chain"]),
                    )
                ride_flag = " [RIDE_ALONG]" if slot_id in decision.ride_along else ""
                print(f"  TURN {turn_num}: slot={slot_id} strategy={strategy} occupancy={filled_slot.occupancy}/{filled_slot.capacity}"
                      f" -> ({st['verdict']}, {st['reason']!r}){ride_flag}")

        print("\n" + "=" * 90)
        print("FINAL RESULT")
        print("=" * 90)
        for slot_id, st in state.items():
            fs = st["filled_slot"]
            print(f"slot={slot_id} final_occupancy={fs.occupancy}/{fs.capacity} final_verdict={st['verdict']} reason={st['reason']!r}")
        print(f"\nTotal turns run: {turn_num}")
        print("\nFull telemetry (per-turn ContinuationDecision):")
        print(json.dumps(telemetry, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
