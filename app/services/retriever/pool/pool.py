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
import dataclasses
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retriever.pool.contracts import PoolResult, SourceAdapter
from app.services.retriever.pool.content_quality import filter_toc_noise
from app.services.retriever.pool.dedup import dedup_candidates
from app.services.retriever.shape.contracts import FanoutTheme, GateResult, ResourcePosture

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

    # BM25 phrase-input tightening (2026-07-29, Ananth's live-trace catch):
    # tag_select/vector_search/inherited previously fed the SQL bm25_score
    # expression the FULL raw gate.expansion_phrases -- every D/J/P-matched
    # phrase, undifferentiated by selectivity, and untiered by kind.
    # Verified live this causes real ranking failures: ts_rank_cd (cover-
    # density) rewards a chunk that repeats a REQUIRED-tier phrase many
    # times even in off-topic content (e.g. the payer name, matched 8x in a
    # chunk about something unrelated) over a chunk that states the actual
    # answer once, clearly. Two changes here: (1) restrict the phrase input
    # to REQUIRED+BOOSTED tier only (phrase_buckets already drops
    # low-selectivity DROP-tier noise) -- (2) D-codes ONLY, not J/P: J
    # (payer/jurisdiction) and P (procedural) axes are meant to be binary
    # presence filters (already enforced at the SQL WHERE-clause level in
    # tag_select/build_candidate_pool), not ranking signals -- letting a
    # payer-name match inflate bm25_score double-counts a filter that's
    # already guaranteed true for every candidate in the pool. D-only +
    # full-scope (D+J+P) phrase_buckets run concurrently (both independent
    # DB calls); tag_select/vector_search/inherited then use the D-only
    # result for their own bm25_score computation, while Filler A's
    # meta_boost still gets the full-scope required/boosted_phrases
    # unchanged (payer/jurisdiction presence is still meaningful there,
    # it's a discrete additive signal, not part of ts_rank_cd's density
    # math).
    (d_required, d_boosted), (required_phrases, boosted_phrases) = await asyncio.gather(
        adapter.phrase_buckets(gate.d_codes, [], [], gate.expansion_phrases),
        adapter.phrase_buckets(gate.d_codes, gate.j_codes, gate.p_codes, gate.expansion_phrases),
    )
    bm25_phrases = [p for p, _sel in d_required] + [p for p, _sel in d_boosted]

    # Strategies 1-3 run concurrently -- independent DB calls, no shared state.
    (
        (tag_candidates, tag_ms),
        (vector_candidates, vector_ms, query_embedding),
        (inherited_candidates, inh_ms),
    ) = await asyncio.gather(
        adapter.tag_select(query, bm25_phrases, gate.d_codes, gate.j_codes, gate.p_codes, tag_width),
        # j_codes added 2026-07-23 (Payor-Policy's live-trace report, real
        # correctness bug): vector_search had zero payer scoping, surfacing
        # e.g. Aetna/Molina content for a Sunshine-Health-specific query.
        # gate.j_codes (not bm25_phrases) -- cross-payer exclusion is a
        # tag-code check inside vector_search, unrelated to the bm25 tsquery
        # phrase-input tightening above.
        adapter.vector_search(query, bm25_phrases, gate.j_codes, vector_width),
        adapter.inherited(query, bm25_phrases, gate.j_codes, inherited_width),
    )
    segment_ms.update(tag_ms)
    segment_ms.update(vector_ms)
    segment_ms.update(inh_ms)

    t_dedup = time.monotonic()
    # Drop TOC/index leader-dot noise before dedup/ranking (2026-07-29,
    # Ananth's catch) -- see content_quality.py for why this content scores
    # artificially high on Filler A's own signals despite zero real
    # relevance; excluded here so no downstream weighting scheme has to
    # compensate for it.
    unioned = dedup_candidates(filter_toc_noise([*tag_candidates, *vector_candidates, *inherited_candidates]))
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

    # Real structural bug fix (2026-07-23, Retriever's live-trace report):
    # dedup is first-arm-wins on chunk_id collision (union order tag_select
    # -> vector -> inherited), so a chunk found by BOTH tag_select and
    # vector_search only keeps tag_select's provenance -- Filler b (filters
    # to source_arm=="vector") never sees it, even when vector_search
    # independently found the same chunk with a strong similarity score.
    # Confirmed live: a correct-answer chunk landed in the pool tagged
    # source_arm=tag_select while a standalone vector_search() call found the
    # identical chunk at rank 165/1000, similarity 0.823. Attaches a real,
    # comparable similarity to every final candidate (matches AND neighbors),
    # regardless of which arm's provenance survived the union -- same pattern
    # bm25_score already established.
    final_candidates, vsim_ms = await adapter.attach_vector_similarity(final_candidates, query_embedding)
    segment_ms.update(vsim_ms)

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
        required_phrases=required_phrases,
        boosted_phrases=boosted_phrases,
    )


async def run_pool_fanout(
    db: AsyncSession,
    rewritten_queries: list[str],
    gate: GateResult,
    resource_posture: ResourcePosture,
    adapter: SourceAdapter,
    fanout_themes: list[FanoutTheme] | None = None,
) -> list[PoolResult]:
    """FAN_OUT entry point -- up to MAX_FANOUT_THEMES=4 queries, each an
    independent run_pool_for_query() call, concurrent via asyncio.gather so
    the (up to 4) embed calls overlap rather than serialize.

    Real bug found + fixed live (2026-07-29, Ananth's catch): every slot's
    tag_select/inherited/phrase_buckets arms were built from the SAME
    original `gate` for every rewritten_query, regardless of theme --
    FanoutTheme.member_codes (the whole reason a theme was clustered out
    from the others) was computed and stored but never actually reached
    Pool. BM25/vector search still benefited from the rewritten query text,
    but the structured tag-based arms -- normally the highest-precision
    ones -- kept searching for the ORIGINAL topic's codes on every slot,
    including slots whose entire point was a DIFFERENT code. Confirmed live
    on cmhc001's FAN_OUT: the "appeal" sub-query's tag_select was still
    scoped to d:claims.timely_filing, never d:disputes.appeal, so only
    chunks that happened to mention "appeal" in free text within the
    timely-filing-tagged set were reachable, not the wider disputes.appeal
    corpus. Fixed by overriding d_codes per-slot from that slot's
    theme.member_codes when present (zipped 1:1 with rewritten_queries,
    same convention shape/slots.py already assumes) -- affects BOTH the
    explore_siblings mechanism (already live) and the newer corpus-grounded
    EXACT-contour decomposition equally, since both produce FanoutTheme.
    """
    if fanout_themes and len(fanout_themes) == len(rewritten_queries):
        gates = await asyncio.gather(*[
            _gate_for_theme(db, gate, theme) for theme in fanout_themes
        ])
    else:
        gates = [gate] * len(rewritten_queries)
    return list(await asyncio.gather(*[
        run_pool_for_query(db, q, g, resource_posture, adapter)
        for q, g in zip(rewritten_queries, gates)
    ]))


async def _gate_for_theme(db: AsyncSession, gate: GateResult, theme: FanoutTheme) -> GateResult:
    """Per-slot GateResult: d_codes replaced with this theme's own codes
    (member_codes already carries whatever combination -- union with the
    base topic, or a fully separate sibling code -- the theme's origin
    mechanism decided was right; Pool doesn't need to know which). j_codes/
    p_codes stay the base query's (payer/program/jurisdiction context is
    shared across every theme slot, only the topic axis differs).
    Falls back to the unmodified gate when a theme has no codes at all
    (the explore_siblings catch-all slot, by design -- its entire point is
    an un-corpus-scoped angle).

    Second real bug caught live (2026-07-29, Ananth's catch on the FIRST
    fix -- d_codes alone wasn't enough): PublicSourceAdapter.phrase_buckets
    only credits a code as required/boosted if that code's OWN lexicon
    phrases already appear in `expansion_phrases` -- and expansion_phrases
    is Gate's, computed once from the base query text, with zero awareness
    that a sub-query even exists. So d:disputes.appeal reached d_codes fine,
    but its phrases ("appeal", "reconsideration", ...) were never in the
    base query's "timely filing deadline..." expansion set -> phrase_buckets'
    `code_phrases = [p for p in phrases_by_code.get(code, []) if p in
    expansion_set]` came back EMPTY for the new code -> it contributed
    NOTHING to required/boosted, while the original codes' phrases (still
    present from the base query) kept dominating -- exactly the
    "claims.general still required" symptom. Fixed by fetching the new
    code's own phrases from the same lexicon snapshot phrase_buckets reads,
    and merging them into this slot's expansion_phrases too."""
    if not theme.member_codes:
        return gate
    new_codes = [c for c in theme.member_codes if c not in gate.d_codes]
    if not new_codes:
        return dataclasses.replace(gate, d_codes=theme.member_codes)
    from app.services.corpus_search_lexicon import _load_lexicon_snapshot
    snapshot = await _load_lexicon_snapshot(db)
    phrases_by_code = {e["full_code"]: e["phrases"] for e in snapshot}
    extra_phrases = [p for c in new_codes for p in phrases_by_code.get(c, [])]
    merged_expansion = list(dict.fromkeys([*gate.expansion_phrases, *extra_phrases]))
    return dataclasses.replace(
        gate, d_codes=theme.member_codes, expansion_phrases=merged_expansion,
    )
