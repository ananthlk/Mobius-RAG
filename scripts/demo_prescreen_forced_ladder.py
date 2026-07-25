"""Forced-ladder demo, for the Chat/Eval/DB/TECH sign-off round: proves the
speculative search-prefetch actually hides latency in the FULL pipeline
path (Pool + Filler d), not just the isolated calls each of us verified
separately. CORRECTION (Web Search, 2026-07-24): this calls
fill_shape_external directly -- it does NOT go through Router's
forced_strategy/RoutingContext dispatch. "Forced" here just means "the
only filler under test," achieved by direct invocation, not by Router's
real forced-strategy mechanism. Same effect for what this measures (Filler
d unconditionally runs), different mechanism than the docstring originally
claimed -- fixing the claim, not the behavior.

Runs the SAME query twice, back to back, sharing nothing between runs
(fresh Gate/Pool each time so neither run gets an unfair cache advantage):
  COLD: Pool runs, THEN Filler d runs with no prescreen (today's baseline
        cost as it exists without this change).
  WARM: prescreen_search fires concurrently with Pool's own fetch (the
        real orchestrator.py wiring), then Filler d reuses the cached
        result instead of searching again.

MAGNITUDE CAVEAT (Web Search, 2026-07-24, confirmed via an independent
re-run): the percentage saved is NOT a stable guarantee -- Vertex
Grounding's search latency varies run-to-run (observed 5.4-9.7s across
different queries this session), so the savings scale with however much
of that variable search cost happens to fit inside Pool's window on a
given run. Two independent runs measured 20.2% and 53.2% reduction for the
identical mechanism -- report this as a RANGE with the mechanism explained
("search cost is hidden whenever Pool takes as long as or longer than
Vertex's response that run"), not a single fixed percentage, in the
sign-off writeup.

Usage (from mobius-rag/):
    .venv/bin/python scripts/demo_prescreen_forced_ladder.py
"""

from __future__ import annotations

import asyncio
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
from app.services.retriever.fillers.filler_d import (  # noqa: E402
    fill_shape_external, prescreen_search, should_prescreen_search,
)
from app.services.router.decision import RoutingContext, ResourcePosture as RouterResourcePosture  # noqa: E402
from app.services.router.router import route as router_route  # noqa: E402
from app.services.retriever.orchestrator import _build_pool_metadata  # noqa: E402

QUERY = "What is the timely filing deadline for Sunshine Health FL Medicaid claims?"


async def prepare(db):
    """Gate -> Reformat -> Structure -> Slots -> payer_context. Shared setup,
    NOT timed (identical cost in both variants, not what this demo measures)."""
    gate = await run_gate(db, QUERY)
    reformat = await run_reformat(db, gate)
    structure = run_structure(reformat, caller_mode=None)
    slots = run_slots(structure)
    payer_slug = extract_payer_slug(gate.j_codes)
    payer_context = await resolve_payer_context(db, payer_slug) if payer_slug else None
    tag_matches = [*gate.d_codes, *gate.j_codes, *gate.p_codes]
    return gate, reformat, structure, slots, payer_context, tag_matches


async def run_cold(db):
    gate, reformat, structure, slots, payer_context, tag_matches = await prepare(db)
    adapter = PublicSourceAdapter(db)
    rq = structure.rewritten_queries[0] if structure.rewritten_queries else QUERY

    t0 = time.monotonic()
    pool = await run_pool_for_query(db, rq, gate, structure.resource_posture, adapter)
    t_pool = time.monotonic()
    shape = AnswerShapeResult(query=QUERY, posture=None, slots=slots.slots, reason="", slots_ms=0)
    result = await fill_shape_external(
        pool, shape, QUERY, db=db, agent_id="demo-cold",
        tag_matches=tag_matches, payer_context=payer_context, prescreened=None,
    )
    t_fill = time.monotonic()

    return {
        "pool_ms": int((t_pool - t0) * 1000),
        "fill_ms": int((t_fill - t_pool) * 1000),
        "total_ms": int((t_fill - t0) * 1000),
        "occupancy": result.slots[0].occupancy,
    }


async def run_warm(db):
    gate, reformat, structure, slots, payer_context, tag_matches = await prepare(db)
    adapter = PublicSourceAdapter(db)
    rq = structure.rewritten_queries[0] if structure.rewritten_queries else QUERY

    t0 = time.monotonic()
    # Real orchestrator.py wiring: prescreen fires concurrently with Pool.
    prescreen_task = asyncio.ensure_future(
        prescreen_search(QUERY, tag_matches=tag_matches, payer_context=payer_context, n_search=5)
    )
    pool = await run_pool_for_query(db, rq, gate, structure.resource_posture, adapter)
    t_pool = time.monotonic()
    prescreened = await prescreen_task
    t_prescreen_ready = time.monotonic()
    shape = AnswerShapeResult(query=QUERY, posture=None, slots=slots.slots, reason="", slots_ms=0)
    result = await fill_shape_external(
        pool, shape, QUERY, db=db, agent_id="demo-warm",
        tag_matches=tag_matches, payer_context=payer_context, prescreened=prescreened,
    )
    t_fill = time.monotonic()

    return {
        "pool_ms": int((t_pool - t0) * 1000),
        "prescreen_extra_wait_ms": int((t_prescreen_ready - t_pool) * 1000),  # 0 if search finished before Pool did
        "fill_ms": int((t_fill - t_prescreen_ready) * 1000),
        "total_ms": int((t_fill - t0) * 1000),
        "occupancy": result.slots[0].occupancy,
    }


async def main():
    print(f"Query: {QUERY!r}")
    print(f"should_prescreen_search('any') = {should_prescreen_search('any')}\n")

    async with AsyncSessionLocal() as db:
        print("=== COLD (no prefetch, today's baseline without this change) ===")
        cold = await run_cold(db)
        print(cold)

    async with AsyncSessionLocal() as db:
        print("\n=== WARM (prescreen concurrent with Pool, real orchestrator.py wiring) ===")
        warm = await run_warm(db)
        print(warm)

    print("\n" + "=" * 90)
    print("COMPARISON")
    print("=" * 90)
    print(f"COLD total (pool + full d-fill incl. fresh search):  {cold['total_ms']:6d}ms")
    print(f"WARM total (pool || prescreen, then cached d-fill):  {warm['total_ms']:6d}ms")
    saved = cold["total_ms"] - warm["total_ms"]
    print(f"Latency hidden by concurrency: {saved}ms "
          f"({saved / cold['total_ms'] * 100:.1f}% of the cold total)")
    print(f"(WARM's prescreen_extra_wait_ms={warm['prescreen_extra_wait_ms']} -- "
          f"how much of the search cost did NOT fit inside Pool's window, if any)")
    print("\nCAVEAT (Web Search, confirmed via independent re-run): this percentage is NOT a "
          "stable guarantee -- Vertex Grounding's search latency varies run-to-run (observed "
          "5.4-9.7s across different queries), so savings scale with how much of that variable "
          "cost happens to fit inside Pool's window this run. Report as a range (observed so "
          "far: 20.2%-53.2% across two independent runs of this same script), not this one number.")


if __name__ == "__main__":
    asyncio.run(main())
