"""Observer calibration run: Arm 0 (single-rung, no loop) / Arm A (today's
real production stopgap-loop) / Arm B (real multi-turn loop wired to
Observer's actual evaluate()) against the 22-query bank, real LLM-judge
grading (eval/judge.py's adjudicate(), rubric mode).

Real per-arm mechanics:
  Arm 0: run ONLY the first ladder rung for each slot, no continuation loop.
  Arm A: today's actual _run_fillers_simple (stopgap verdict + continuation
         loop) -- the real production behavior, not a strawman.
  Arm B: same continuation-loop mechanism, but per-slot verdicts come from
         Observer's real evaluate() instead of the stopgap.

For each arm's result, compile_synthesis() produces the real citations
(document_name resolved, deduped, neighbor-complete) that get judged --
NOT raw FilledChunks, since Synthesis is real and shipped, and its output
is what a grading pass should actually look at.

Usage (from mobius-rag/):
    .venv/bin/python scripts/run_observer_calibration.py --limit 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

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
from app.services.retriever.fillers.contracts import FilledSlot, FilledShape  # noqa: E402
from app.services.retriever.fillers.filler_a import fill_shape_bm25  # noqa: E402
from app.services.retriever.fillers.filler_b import fill_shape_vector  # noqa: E402
from app.services.retriever.fillers.filler_c import fill_shape_llm_retrieval  # noqa: E402
from app.services.retriever.fillers.filler_d import fill_shape_external  # noqa: E402
from app.services.retriever.fillers.filler_s import fill_shape_fact_store  # noqa: E402
from app.services.retriever.observer import evaluate as observer_evaluate  # noqa: E402
from app.services.retriever.synthesis import compile_synthesis  # noqa: E402
from app.services.retriever.synthesis_contracts import SlotVerdict  # noqa: E402
from app.services.router.decision import RoutingContext, ResourcePosture as RouterResourcePosture  # noqa: E402
from app.services.router.router import route as router_route  # noqa: E402
from app.services.router.continuation import SlotTurnInput, decide_continuation  # noqa: E402
from app.services.retriever.orchestrator import _build_pool_metadata, _IMPLEMENTED_FILLERS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from judge import adjudicate  # noqa: E402

BANK_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries_cmhc.yaml"
OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "artifacts" / "observer_calibration_run.json"
_MAX_TURNS = 5


def _stopgap_verdict(occupancy, error, has_remaining):
    from app.services.router.continuation import (
        VERDICT_ERROR, VERDICT_SATISFIED, VERDICT_WOULD_BENEFIT, VERDICT_EXHAUSTED_ATTEMPTS,
    )
    if error is not None:
        return VERDICT_ERROR, f"filler raised: {error}"
    if occupancy > 0:
        return VERDICT_SATISFIED, "occupancy>0 (stopgap verdict)"
    if has_remaining:
        return VERDICT_WOULD_BENEFIT, "occupancy=0, rungs remain (stopgap verdict)"
    return VERDICT_EXHAUSTED_ATTEMPTS, "occupancy=0, chain exhausted"


async def try_strategy(db, strategy, slot, pr, raw_query, gate_result, payer_context):
    single_shape = AnswerShapeResult(query=raw_query, posture=None, slots=[slot], reason="", slots_ms=0)
    tag_matches = [*gate_result.d_codes, *gate_result.j_codes, *gate_result.p_codes]
    try:
        if strategy == "a":
            result = fill_shape_bm25(pr, single_shape)
        elif strategy == "b":
            result = fill_shape_vector(pr, single_shape)
        elif strategy == "c":
            result = await fill_shape_llm_retrieval(pr, single_shape, raw_query, db=db, agent_id="obs-cal", tag_matches=tag_matches)
        elif strategy == "d":
            result = await fill_shape_external(pr, single_shape, raw_query, db=db, agent_id="obs-cal", tag_matches=tag_matches, payer_context=payer_context)
        else:
            result = await fill_shape_fact_store(pr, single_shape, raw_query, tag_matches=tag_matches)
        return result.slots[0], None
    except Exception as exc:
        return FilledSlot(slot_id=slot.slot_id, slot_semantics=slot.slot_semantics, capacity=slot.capacity, required=slot.required, chunks=[], occupancy=0, under_filled=True, over_filled=False), repr(exc)


async def run_arm(db, arm, query, slots, pool_by_query, default_pool, ladder, raw_query, gate_result, payer_context, latency_allowance_ms):
    """arm in {"0", "A", "B"}. Returns (FilledShape, verdicts_dict, elapsed_ms,
    executed_per_slot) -- the last is Eval's per-strategy-chunks
    instrumentation ask (2026-07-24): {slot_id: [{strategy, occupancy,
    error}, ...]} in execution order, so the artifact shows what EACH rung
    actually returned, not just the final (DISCARD-model) winner."""
    t0 = time.monotonic()
    state = {}
    for slot in slots:
        chain = [s for s in ladder.get(slot.slot_id, []) if s in _IMPLEMENTED_FILLERS]
        state[slot.slot_id] = {
            "slot": slot, "chain": chain, "cursor": 0, "filled_slot": None, "verdict": None, "reason": "",
            # Eval's instrumentation ask (2026-07-24): per-strategy chunk
            # counts, not just the final (DISCARD-model) filled_slot -- so
            # the artifact can show "a returned 8, b returned 0" instead of
            # only the winning rung's occupancy.
            "executed": [],
        }

    def verdict_for(strategy, filled_slot, error, cursor, chain_len):
        if arm == "B":
            if error is not None:
                from app.services.router.continuation import VERDICT_ERROR
                return VERDICT_ERROR, f"filler raised: {error}"
            return observer_evaluate(strategy, filled_slot, attempt_number=cursor, max_attempts=chain_len)
        return _stopgap_verdict(filled_slot.occupancy, error, cursor < chain_len)

    for slot_id, st in state.items():
        slot, pr = st["slot"], pool_by_query.get(st["slot"].rewritten_query) or default_pool
        if not st["chain"] or pr is None:
            from app.services.router.continuation import VERDICT_EXHAUSTED_ATTEMPTS
            st["filled_slot"] = FilledSlot(slot_id=slot.slot_id, slot_semantics=slot.slot_semantics, capacity=slot.capacity, required=slot.required, chunks=[], occupancy=0, under_filled=True, over_filled=False)
            st["verdict"], st["reason"] = VERDICT_EXHAUSTED_ATTEMPTS, "no_implemented_strategy_or_pool"
            continue
        strategy = st["chain"][0]
        filled_slot, error = await try_strategy(db, strategy, slot, pr, raw_query, gate_result, payer_context)
        st["cursor"] = 1
        st["filled_slot"] = filled_slot
        st["executed"].append({"strategy": strategy, "occupancy": filled_slot.occupancy, "error": error})
        st["verdict"], st["reason"] = verdict_for(strategy, filled_slot, error, 1, len(st["chain"]))

    if arm != "0":
        for _ in range(_MAX_TURNS):
            turn_inputs = [
                SlotTurnInput(slot_id=sid, remaining_rungs=tuple(st["chain"][st["cursor"]:]), verdict=st["verdict"], reason=st["reason"], required=st["slot"].required)
                for sid, st in state.items()
            ]
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            decision = decide_continuation(turn_inputs, elapsed_ms, latency_allowance_ms)
            if not decision.new_turn:
                break
            for slot_id, strategy in decision.turn_rungs.items():
                st = state[slot_id]
                pr = pool_by_query.get(st["slot"].rewritten_query) or default_pool
                filled_slot, error = await try_strategy(db, strategy, st["slot"], pr, raw_query, gate_result, payer_context)
                st["cursor"] += 1
                st["filled_slot"] = filled_slot
                st["executed"].append({"strategy": strategy, "occupancy": filled_slot.occupancy, "error": error})
                st["verdict"], st["reason"] = verdict_for(strategy, filled_slot, error, st["cursor"], len(st["chain"]))

    filled_slots = [state[slot.slot_id]["filled_slot"] for slot in slots]
    verdicts = {sid: SlotVerdict(verdict=st["verdict"] or "", reason=st["reason"] or "", ride_along=False) for sid, st in state.items()}
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    executed_per_slot = {sid: st["executed"] for sid, st in state.items()}
    return FilledShape(slots=filled_slots, total_chunks_assigned=sum(s.occupancy for s in filled_slots), filling_strategy=f"calibration_arm_{arm}", emit={}), verdicts, elapsed_ms, executed_per_slot


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8, help="number of bank queries to run (pilot subset)")
    args = parser.parse_args()

    bank = yaml.safe_load(BANK_PATH.read_text())
    queries = bank["queries"][:args.limit]
    print(f"Running {len(queries)}/{len(bank['queries'])} bank queries, arms 0/A/B, real judge grading")

    results = []
    for i, q in enumerate(queries, 1):
        qid, query_text = q["id"], q["query"]
        print(f"\n[{i}/{len(queries)}] {qid}: {query_text!r}")

        # Fresh session PER QUERY (Eval's ask, 2026-07-24): a poisoned
        # connection from one query's DB error (e.g. the fact-store
        # document_id/UUID crash) must not cascade-abort the remaining
        # queries in the batch -- one shared session across all 22 queries
        # meant a single failure poisoned everything after it, which is
        # exactly what happened in the run that only completed 1/22.
        try:
            async with AsyncSessionLocal() as db:
                gate = await run_gate(db, query_text)
                reformat = await run_reformat(db, gate)
                structure = run_structure(reformat, caller_mode=None)
                slots = run_slots(structure)
                payer_slug = extract_payer_slug(gate.j_codes)
                payer_context = await resolve_payer_context(db, payer_slug) if payer_slug else None
                adapter = PublicSourceAdapter(db)
                rq = structure.rewritten_queries[0] if structure.rewritten_queries else query_text
                pool = await run_pool_for_query(db, rq, gate, structure.resource_posture, adapter)
                pool_metadata = _build_pool_metadata(slots.slots, [pool])

                ctx = RoutingContext(
                    query=query_text, agent_id="obs-calibration",
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
                ladder = router_decision.routing_ladder.per_slot
                pool_by_query = {pool.query: pool}

                # Eval's instrumentation ask (2026-07-24): dispatch_path,
                # per-slot depth_bucket, and per-rung skip_reason -- the
                # fields that would have caught the payload-gate collapse
                # on run 1 instead of three hypothesis rounds. Real values
                # off DecisionTrace, not recomputed/guessed.
                dispatch_path = router_decision.dispatch_path
                depth_bucket_by_slot = {st.slot_id: st.depth_bucket for st in router_decision.trace.slots}
                skip_trace_by_slot = {
                    st.slot_id: [
                        {"strategy": step.strategy_id, "action": step.action, "skip_reason": step.skip_reason}
                        for step in st.steps
                    ]
                    for st in router_decision.trace.slots
                }

                per_arm = {}
                for arm in ["0", "A", "B"]:
                    filled_shape, verdicts, elapsed_ms, executed_per_slot = await run_arm(
                        db, arm, query_text, slots.slots, pool_by_query, pool, ladder,
                        query_text, gate, payer_context, latency_allowance_ms,
                    )
                    try:
                        synthesis_result = await compile_synthesis(query_text, filled_shape, db=db, verdicts=verdicts)
                    except Exception as exc:
                        # Real bug, flagged to Synthesizer separately, 2026-07-24:
                        # Filler s's synthetic "fact_..." document_id isn't a real
                        # document row, but Synthesis's internal/external
                        # classification (document_id and not url) treats any
                        # non-None document_id as internal, triggering a
                        # neighbor-completion query against a non-UUID chunk_id.
                        # Roll back so the aborted Postgres transaction doesn't
                        # cascade-fail the rest of THIS query's arms -- this
                        # harness works around it rather than papering over the
                        # real defect, which stays reported, not silently patched
                        # here.
                        await db.rollback()
                        print(f"  arm {arm}: compile_synthesis FAILED ({exc!r}) -- skipping, rolled back")
                        per_arm[arm] = {
                            "elapsed_ms": elapsed_ms, "error": repr(exc),
                            "dispatch_path": dispatch_path, "depth_bucket": depth_bucket_by_slot,
                            "skip_trace": skip_trace_by_slot, "executed_per_slot": executed_per_slot,
                        }
                        continue
                    chunks = [
                        {"text": c.text, "document_name": c.document_name, "page_number": c.page_number, "is_neighbor": c.is_neighbor}
                        for c in synthesis_result.citations
                    ]
                    response = {
                        "chunks": chunks, "llm_answer": "", "strategy_used": ladder,
                        "confidence": None,
                    }
                    verdict, score, reasoning, model_used, judge_ms = await adjudicate(query_text, q, response)
                    per_arm[arm] = {
                        "elapsed_ms": elapsed_ms, "n_citations": len(chunks),
                        # Eval's ask (2026-07-24): raw compiled content
                        # alongside the score, not just the aggregate --
                        # needed to spot-check the judge's verdict against
                        # real content, same discipline as every artifact
                        # this session.
                        "chunks": chunks,
                        "judge_verdict": verdict, "judge_score": score, "judge_reasoning": reasoning,
                        "judge_model": model_used, "judge_ms": judge_ms,
                        "per_slot_verdicts": {sid: v.verdict for sid, v in verdicts.items()},
                        # Second instrumentation round (2026-07-24, post
                        # payload-gate fix) -- the fields that would have
                        # caught the collapse on run 1 immediately.
                        "dispatch_path": dispatch_path,
                        "depth_bucket": depth_bucket_by_slot,
                        "skip_trace": skip_trace_by_slot,
                        "executed_per_slot": executed_per_slot,
                    }
                    print(f"  arm {arm}: elapsed={elapsed_ms:6d}ms citations={len(chunks)} "
                          f"judge_verdict={verdict} score={score:.2f}")

                results.append({"id": qid, "query": query_text, "per_arm": per_arm})
        except Exception as exc:
            # Query-level failure (e.g. Gate/Pool/Router itself throwing,
            # not just one arm's compile_synthesis) -- log and move to the
            # next query with a genuinely fresh session, don't let it kill
            # the whole batch.
            print(f"  QUERY FAILED ({exc!r}) -- skipping entirely, fresh session next query")
            results.append({"id": qid, "query": query_text, "per_arm": {}, "query_error": repr(exc)})

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({"n_total_bank": len(bank["queries"]), "n_run": len(results), "results": results}, indent=2))

    print(f"\nDone. Artifact: {OUT_PATH}")
    print("\n=== AGGREGATE ===")
    for arm in ["0", "A", "B"]:
        clean = [r["per_arm"][arm] for r in results if "judge_score" in r["per_arm"][arm]]
        errored = len(results) - len(clean)
        if not clean:
            print(f"arm {arm}: no clean results ({errored} errored)")
            continue
        scores = [c["judge_score"] for c in clean]
        elapsed = [c["elapsed_ms"] for c in clean]
        print(f"arm {arm}: mean_score={sum(scores)/len(scores):.3f} mean_elapsed_ms={sum(elapsed)/len(elapsed):.0f} "
              f"(n={len(clean)}, {errored} errored)")


if __name__ == "__main__":
    asyncio.run(main())
