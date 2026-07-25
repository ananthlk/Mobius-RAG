"""Pool orchestrator (Step 2 entry point) -- pool-schematic-spec.md S3.

Runs once per rewritten_query: 3 strategies (concurrent) -> union -> dedup
-> neighbor-assembly. FAN_OUT concurrency (up to 4 rewritten_queries)
happens ONE LEVEL UP, via asyncio.gather over run_pool_for_query() calls --
not inside a single query's build (S1: prod embedding model
gemini-embedding-001 caps at 1 input/call, so concurrency across queries is
the lever, not batching within one).

Pure orchestration -- all actual retrieval logic lives in the SourceAdapter
(S3.0); this module never touches the DB directly and never branches on
which adapter it's talking to.
"""

from __future__ import annotations

import asyncio
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retriever.pool.contracts import PoolResult, SourceAdapter
from app.services.retriever.pool.dedup import dedup_candidates
from app.services.retriever.shape.contracts import GateResult, ResourcePosture

# Per-strategy width multipliers off ResourcePosture.breadth -- S3.4,
# build-time calibration, not yet Eval-tuned. Starting point per Ananth's
# own illustrative numbers (2026-07-23): tag_select ~20x, vector ~100x
# breadth -- NOT fixed ceilings, revisit once Eval has real numbers. Cost
# model: embed calls are the expensive resource, not k, so these stay
# generous rather than tight.
TAG_SELECT_WIDTH_MULTIPLIER = 20
VECTOR_WIDTH_MULTIPLIER = 100
INHERITED_WIDTH_MULTIPLIER = 20

# Real bug found verifying against live data (2026-07-23): the reused
# _expand_with_neighbors() caps its COMBINED (seeds+neighbors) output at
# _NEIGHBOR_TOTAL_CAP=50 (corpus_search.py:2547) -- it silently TRUNCATES
# the seed list itself once seeds alone exceed 50, not just "no room for
# neighbors." A live run against a real cmhc query produced 378 unioned
# candidates; passing all 378 as seeds returned a negative "kept" count
# (meta showed kept=-328) because 328 of the seeds themselves got dropped.
# That function was built for legacy's small post-rerank result set
# (~10-50 final hits), not Pool's wide over-fetched union. Per the
# reuse-verbatim decision (pool-schematic-spec.md S1: don't restructure
# _expand_with_neighbors itself, that's NUMBER-MOVING), the fix lives here:
# only send a bounded, per-arm-interleaved top-N to neighbor expansion,
# comfortably under the shared cap, and leave the rest of the wide union
# un-expanded (they still count for recall; they just don't carry sibling
# context). Multiplier is a v1 heuristic, not yet Eval-validated.
_NEIGHBOR_EXPANSION_CAP = 24


def _interleave_top_n(arms: list[list], n: int) -> list:
    """Round-robin across arms (each already ordered by its own metric --
    tag_select by coverage DESC, vector by similarity DESC, inherited by DB
    order) so no single arm crowds out the others in the neighbor-expansion
    seed set."""
    out: list = []
    idx = 0
    while len(out) < n and any(idx < len(a) for a in arms):
        for a in arms:
            if idx < len(a) and len(out) < n:
                out.append(a[idx])
        idx += 1
    return out


async def run_pool_for_query(
    db: AsyncSession,
    query: str,
    gate: GateResult,
    resource_posture: ResourcePosture,
    adapter: SourceAdapter,
) -> PoolResult:
    """Build one pool for one rewritten_query."""
    t_start = time.monotonic()

    breadth = resource_posture.breadth
    if breadth <= 0:
        # No-retrieval postures (CLARIFY/CLARIFY_REPHRASE/DECLINE) reach
        # here with an all-zero ResourcePosture -- same convention
        # Structure uses (shape/structure.py), not an error case.
        return PoolResult(query=query, fallback_triggered=True, pool_ms=0)

    tag_width = breadth * TAG_SELECT_WIDTH_MULTIPLIER
    vector_width = breadth * VECTOR_WIDTH_MULTIPLIER
    inherited_width = breadth * INHERITED_WIDTH_MULTIPLIER

    segment_ms: dict = {}
    # Strategies 1-3 run concurrently -- independent DB calls, no shared state.
    (tag_candidates, tag_ms), (vector_candidates, vector_ms, query_embedding), (inherited_candidates, inh_ms) = await asyncio.gather(
        adapter.tag_select(query, gate.d_codes, gate.j_codes, gate.p_codes, tag_width),
        adapter.vector_search(query, vector_width),
        adapter.inherited(query, gate.j_codes, inherited_width),
    )
    segment_ms.update(tag_ms)
    segment_ms.update(vector_ms)
    segment_ms.update(inh_ms)

    t_dedup = time.monotonic()
    unioned = dedup_candidates([*tag_candidates, *vector_candidates, *inherited_candidates])
    segment_ms["dedup_ms"] = int((time.monotonic() - t_dedup) * 1000)

    # Bound neighbor-expansion input -- see _NEIGHBOR_EXPANSION_CAP above.
    # Interleave per-arm on the DEDUPED per-arm lists (post-union some
    # entries may have been dropped as duplicates) so the seed set stays
    # representative rather than tag_select-heavy from dedup's list order.
    unioned_ids = {c.chunk_id for c in unioned}
    deduped_by_arm = [
        [c for c in tag_candidates if c.chunk_id in unioned_ids],
        [c for c in vector_candidates if c.chunk_id in unioned_ids],
        [c for c in inherited_candidates if c.chunk_id in unioned_ids],
    ]
    expansion_seeds = _interleave_top_n(deduped_by_arm, _NEIGHBOR_EXPANSION_CAP)
    expansion_seed_ids = {c.chunk_id for c in expansion_seeds}
    remainder = [c for c in unioned if c.chunk_id not in expansion_seed_ids]

    expanded, neighbor_ms = await adapter.neighbors(expansion_seeds)
    segment_ms.update(neighbor_ms)
    # _expand_with_neighbors only dedupes fresh siblings against the seeds
    # IT was given (the bounded 24, not the full union) -- a sibling that
    # happens to already be present in `remainder` as its own independent
    # match (from another arm) needs a second dedup pass here, caught live
    # (2026-07-23): one such collision on the cmhc smoke query.
    final_candidates = dedup_candidates(expanded + remainder)

    strategy_hint = "+".join(
        name for name, cands in (
            ("tag_select", tag_candidates),
            ("vector", vector_candidates),
            ("inherited", inherited_candidates),
        ) if cands
    )

    return PoolResult(
        query=query,
        candidates=final_candidates,
        segment_ms=segment_ms,
        strategy_hint=strategy_hint,
        fallback_triggered=not unioned,
        pool_ms=int((time.monotonic() - t_start) * 1000),
        query_embedding=query_embedding,
    )


async def run_pool_fanout(
    db: AsyncSession,
    rewritten_queries: list[str],
    gate: GateResult,
    resource_posture: ResourcePosture,
    adapter: SourceAdapter,
) -> list[PoolResult]:
    """FAN_OUT entry point -- up to MAX_FANOUT_THEMES=4 queries, each an
    independent run_pool_for_query() call, concurrent via asyncio.gather so
    the (up to 4) embed calls overlap rather than serialize."""
    return list(await asyncio.gather(*[
        run_pool_for_query(db, q, gate, resource_posture, adapter) for q in rewritten_queries
    ]))
