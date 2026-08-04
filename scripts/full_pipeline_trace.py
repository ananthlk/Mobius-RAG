"""Real, EXECUTED step-by-step trace of the pipeline for one query, forced to
one strategy -- Gate -> Reformat -> Structure -> Slots -> Pool -> Filler ->
Synthesis. Every stage is the REAL function, called directly, against a REAL
DB connection (local cloud-sql-proxy) -- not read from source, not guessed.
Prints full intermediate state after each stage, including the FULL pool
candidate list (not truncated) with keyword hits flagged.

Usage: DATABASE_URL=... .venv/bin/python scripts/full_pipeline_trace.py
"""
import asyncio, dataclasses, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUERY = "What is the timely filing deadline for Sunshine Health FL Medicaid claims?"
FORCED_STRATEGY = "a"
# Loose keyword flags for spotting the actual answer (day-count / timely
# filing table language) inside candidate/chunk text -- diagnostic only,
# not a correctness oracle.
ANSWER_KEYWORDS = re.compile(r"\b(\d{2,3}\s*(?:calendar\s+)?days?|timely filing|filing limit|filing deadline)\b", re.I)


def dump(obj, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            print(f"  {f.name}: {v!r}"[:2000])
    else:
        print(f"  {obj!r}"[:2000])


def flag(text: str) -> str:
    m = ANSWER_KEYWORDS.search(text or "")
    return f"  <<< KEYWORD HIT: {m.group(0)!r}" if m else ""


async def main():
    from app.database import AsyncSessionLocal
    from app.services.retriever.shape.gate import run_gate
    from app.services.retriever.shape.reformat import run_reformat
    from app.services.retriever.shape.structure import run_structure
    from app.services.retriever.shape.slots import run_slots
    from app.services.retriever.pool.public_adapter import PublicSourceAdapter
    from app.services.retriever.pool.pool import run_pool_for_query
    from app.services.retriever.fillers.filler_a import fill_shape_bm25
    from app.services.retriever.synthesis import compile_synthesis

    async with AsyncSessionLocal() as db:
        print(f"\nQUERY: {QUERY!r}\nFORCED STRATEGY: {FORCED_STRATEGY!r}")

        gate = await run_gate(db, QUERY)
        dump(gate, "STAGE 1: GATE (run_gate)")

        reformat = await run_reformat(db, gate)
        dump(reformat, "STAGE 2: REFORMAT (run_reformat)")

        structure = run_structure(reformat, caller_mode="chat.default")
        dump(structure, "STAGE 3: STRUCTURE (run_structure)")
        rp = structure.resource_posture
        print(f"\n  resource_posture detail: breadth={rp.breadth} confidence_bar={rp.confidence_bar} "
              f"max_attempts={rp.max_attempts} speed_budget={rp.speed_budget} token_budget={rp.token_budget}")

        slots_result = run_slots(structure)
        dump(slots_result, "STAGE 4: SLOTS (run_slots)")
        for s in slots_result.slots:
            print(f"  slot: id={s.slot_id} semantics={s.slot_semantics} capacity={s.capacity} "
                  f"required={s.required} rewritten_query={s.rewritten_query!r}")

        adapter = PublicSourceAdapter(db)
        slot = slots_result.slots[0]
        rewritten_query = slot.rewritten_query or QUERY
        pool_result = await run_pool_for_query(db, rewritten_query, gate, rp, adapter)
        print(f"\n{'='*70}\nSTAGE 5: POOL (run_pool_for_query) -- FULL candidate dump\n{'='*70}")
        print(f"  query={pool_result.query!r}")
        print(f"  fallback_triggered={pool_result.fallback_triggered}")
        print(f"  pool_ms={pool_result.pool_ms}")
        print(f"  candidates: {len(pool_result.candidates)}")
        ranked = sorted(pool_result.candidates, key=lambda c: (c.bm25_score or 0), reverse=True)
        hit_positions = []
        for i, c in enumerate(ranked, 1):
            hit = flag(c.text or "")
            if hit:
                hit_positions.append(i)
            print(f"    {i}. doc={c.document_id} bm25_score={c.bm25_score} vector_score={getattr(c,'vector_score',None)} "
                  f"source_type={c.source_type} authority_level={c.authority_level} "
                  f"text={(c.text or '')[:110]!r}{hit}")
        print(f"\n  >>> {len(hit_positions)}/{len(ranked)} candidates keyword-flagged as possibly answer-bearing "
              f"(rank positions: {hit_positions})")

        print(f"\n{'='*70}\nSTAGE 6: FILLER A (fill_shape_bm25)\n{'='*70}")
        filled = fill_shape_bm25(pool_result, slots_result)
        print(f"  total_chunks_assigned={filled.total_chunks_assigned}  filling_strategy={filled.filling_strategy}")
        for fs in filled.slots:
            print(f"  slot={fs.slot_id} occupancy={fs.occupancy} capacity={fs.capacity} under_filled={fs.under_filled}")
            for i, ch in enumerate(fs.chunks, 1):
                hit = flag(ch.text or "")
                print(f"    {i}. score={ch.original_score} doc={ch.document_id} source_type={ch.source_type} "
                      f"authority_level={ch.authority_level} text={(ch.text or '')[:110]!r}{hit}")

        print(f"\n{'='*70}\nSTAGE 7: SYNTHESIS (compile_synthesis, data_collection_mode=True -- forced-strategy posture)\n{'='*70}")
        synth = await compile_synthesis(
            QUERY, filled, db=db, token_budget=rp.token_budget, data_collection_mode=True,
        )
        print(f"  telemetry: {synth.telemetry!r}"[:1500])
        print(f"  citations: {len(synth.citations)}")
        for c in synth.citations:
            hit = flag(c.text or "")
            print(f"    [{c.index}] doc_name={c.document_name!r} source_type={c.source_type} "
                  f"chunk_id={c.chunk_id} text={(c.text or '')[:110]!r}{hit}")
        for s in synth.slots:
            print(f"  slot={s.slot_id} occupancy={s.occupancy}/{s.capacity} under_filled={s.under_filled} "
                  f"verdict={s.verdict!r} n_citations={len(s.citations)}")
        for slot_id, cd in synth.coverage_diagnostics.items():
            print(f"  coverage[{slot_id}]: verdict={cd.pool_verdict} reason={cd.reason!r} "
                  f"saturated_strategies={cd.saturated_strategies}")


asyncio.run(main())
