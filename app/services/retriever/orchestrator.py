"""Retriever's top-level entry point — sequences Shape (Gate → Reformat →
Structure → Slots) → Pool → Router → Fillers → Synthesis → Contract →
Timing. This is the conductor role TECH asked about 2026-07-23: no single
sub-module naturally owns cross-module sequencing (Shape doesn't call Pool;
Pool doesn't call Router), so it belongs to Retriever directly, as the "one
clean answering contract" chat calls. Mirrors the legacy single entry point
(`corpus_search_agent() :3066 → _impl :3766`) — thin glue, no business logic
of its own, calling each module's public interface in order.

STATUS 2026-07-24: wires Shape → Pool → Router → Fillers → Synthesis, live.
Contract/Timing exist (contract.py, attempt_spans emit) but Contract's
build_contract() is called by whoever calls run_retriever_partial(), not
from inside this file (avoids a circular import -- contract.py imports
RetrieverPartialResult).

STATUS 2026-07-26: Fillers now call Observer's real per-strategy evaluate()
(_observer_verdict, replacing the old bare-occupancy _stopgap_verdict) --
Eval's calibration run this session (22-query forced bank + live router
baseline) is what this swap was gated on; Ananth directed the wire-in
directly.

CORRECTION 2026-07-23: this file previously stopped after Structure even
after Pool was built and closed — Pool's PoolResult.segment_ms had no live
caller anywhere (flagged in retriever-emit-telemetry-registry.md's `pool`
row, never actually fixed until now). Wiring it in is what actually makes
Pool's telemetry a real emit rather than an unreachable return value.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retriever.shape.contracts import (
    FanoutTheme,
    GateResult,
    ReformatPosture,
    ReformatResult,
    StructureResult,
)
from app.services.retriever.shape.gate import run_gate
from app.services.retriever.shape.narrate import narrate as narrate_gate
from app.services.retriever.shape.narrate import narrate_full as narrate_gate_full
from app.services.retriever.shape.reformat import run_reformat
from app.services.retriever.shape.reformat_narrate import narrate as narrate_reformat
from app.services.retriever.shape.reformat_narrate import narrate_full as narrate_reformat_full
from app.services.retriever.shape.structure import run_structure
from app.services.retriever.shape.slots import AnswerShapeResult, run_slots
from app.services.retriever.pool.contracts import PoolResult
from app.services.retriever.pool.pool import run_pool_fanout, run_pool_for_query
from app.services.retriever.pool.public_adapter import PublicSourceAdapter
from app.services.retriever.fillers.payer_context import (
    PayerContext,
    extract_payer_slug,
    resolve_payer_context,
)
from app.database import AsyncSessionLocal
from app.services.router.router import route as router_route
from app.services.router.decision import (
    ResourcePosture as RouterResourcePosture,
    RouterDecision,
    RoutingContext,
)
from app.services.router.allocation import SUPPLEMENT_ONLY_STRATEGIES
from app.services.retriever import observer
from app.services.retriever.fillers.contracts import FilledShape, FilledSlot
from app.services.retriever.fillers.filler_a import fill_shape_bm25
from app.services.retriever.fillers.filler_b import fill_shape_vector
from app.services.retriever.fillers.filler_c import fill_shape_llm_retrieval
from app.services.retriever.fillers.filler_d import (
    PrescreenedSearch,
    fill_shape_external,
    prescreen_search,
    should_prescreen_search,
)
from app.services.retriever.fillers.filler_s import fill_shape_fact_store
from app.services.retriever.synthesis import compile_synthesis
from app.services.retriever.synthesis_contracts import SlotVerdict, SynthesisResult
from app.services.router.continuation import (
    SlotTurnInput,
    VERDICT_ERROR,
    VERDICT_EXHAUSTED_ATTEMPTS,
    VERDICT_SATISFIED,
    VERDICT_WOULD_BENEFIT,
    decide_continuation,
)

# Which strategy letters have a real, callable filler today. f is still
# mid-design (no filler_f.py exists yet -- Sitemap's module is
# sitemap_links.py, a separate thing, not Router's real "f") -- a rung
# requesting "f" is a genuine gap, not a bug, until it lands.
_IMPLEMENTED_FILLERS: dict[str, str] = {"a": "pool", "b": "pool", "s": "external", "c": "external", "d": "external"}

logger = logging.getLogger(__name__)

# Per-slug TTL cache for payer-context resolution -- same in-process LRU/TTL
# pattern already established for query embeddings (corpus_search.py's
# _embed_with_cache). Resolution costs one live Payor Platform HTTP call
# (up to _PAYOR_PLATFORM_TIMEOUT_S); caching avoids paying that repeatedly
# for the same payer within a short window. Real_time callers skip
# resolution entirely on a cache miss (see _resolve_payer_context_cached)
# rather than block a latency-sensitive query on a cold external call --
# Router's crawl-gate already fails open on payer_context=None, so skipping
# degrades to today's behavior, never blocks.
_PAYER_CONTEXT_CACHE: dict[str, tuple[PayerContext, float]] = {}
_PAYER_CONTEXT_CACHE_TTL_S = 300.0
_PAYER_CONTEXT_CACHE_MAX = 256


async def _resolve_payer_context_cached(
    db: AsyncSession, slug: str, *, speed_budget: str,
) -> PayerContext | None:
    now = time.monotonic()
    cached = _PAYER_CONTEXT_CACHE.get(slug)
    if cached is not None and (now - cached[1]) < _PAYER_CONTEXT_CACHE_TTL_S:
        return cached[0]

    if speed_budget == "real_time":
        # Cold cache + real_time: don't pay the live-call latency. Falls
        # through to Router's existing fail-open gate (payer_crawlable=None).
        return None

    result = await resolve_payer_context(db, slug)
    if len(_PAYER_CONTEXT_CACHE) >= _PAYER_CONTEXT_CACHE_MAX:
        _PAYER_CONTEXT_CACHE.pop(next(iter(_PAYER_CONTEXT_CACHE)))  # evict oldest
    _PAYER_CONTEXT_CACHE[slug] = (result, now)
    return result


@dataclass
class RetrieverPartialResult:
    """What the chain has produced so far — PROVISIONAL, not a locked
    contract like GateResult/ReformatResult. Will be superseded once
    Structure (Step 1c) exists and defines the real Shape-output contract;
    this is a build-time stitch, not something downstream code should
    depend on long-term.
    """

    query: str = ""
    gate: GateResult | None = None
    reformat: ReformatResult | None = None
    structure: StructureResult | None = None
    slots: AnswerShapeResult | None = None
    pool: list[PoolResult] = field(default_factory=list)  # one per rewritten_query; empty for no-retrieval postures
    # Resolved ONCE here, post-Gate, concurrently with Pool's fetch (both
    # depend only on Gate's output) -- per Router/Filler-d/Filler-f's
    # agreed design 2026-07-23. Threaded to Router (RoutingContext.
    # payer_crawlable, plan-time crawl-gate) AND to whichever filler
    # executes at attempt time (avoids a second ~3s Payor Platform call).
    # None when Gate found no payor tag at all -- not an error case.
    payer_context: PayerContext | None = None
    router_decision: RouterDecision | None = None
    filled_shape: FilledShape | None = None
    # Synthesis (Step 5) wired in 2026-07-24 -- compile_synthesis() is real,
    # tested, 9/9 cross-agent signed off; this is the first live caller.
    # None whenever filled_shape is None (no-retrieval postures) -- nothing
    # for Synthesis to compile in that case, not a failure.
    synthesis: SynthesisResult | None = None

    gate_ms: int = 0
    reformat_ms: int = 0
    slots_ms: int = 0
    pool_ms: int = 0
    router_ms: int = 0
    fillers_ms: int = 0
    synthesis_ms: int = 0
    total_ms: int = 0

    # RESOLVED by UX 2026-07-23 (see docs/rag-agents/retriever-emit-telemetry-registry.md):
    # both narrate() outputs feed thinking_trace. shape_reformat (the structural
    # telemetry key: posture/fanout_themes/latency_ms) is backend/Diagnostics
    # only -- a DIFFERENT layer from the narrate() function, which is
    # user-facing by design, same as Gate's. Composition: Gate's narrate()
    # first (explains the contour), Reformat's narrate() second (explains the
    # resulting action) -- but ONLY for postures where Reformat adds real
    # information beyond Gate's own narration; see _include_reformat_narration().
    narrative: str = ""       # thinking_trace value -- Gate always, Reformat conditionally (see composition rule)
    narrative_full: str = ""  # combined, Diagnostics-only, NEVER persist (see narrate.py PHI note)

    pipeline_complete: bool = False  # False until Router/Fillers/Synthesis/Contract/Timing exist
    next_step: str = "Router (Reasoning + Strategy) — in build"


# Postures where Reformat's narrate() is included in thinking_trace alongside
# Gate's -- per UX's ruling 2026-07-23. PRECISE/DECLINE/CLARIFY_REPHRASE are
# excluded: Gate's own narration already fully explains those outcomes,
# Reformat's narrate() for them would be redundant, not additive.
_INCLUDE_REFORMAT_NARRATION = frozenset({
    ReformatPosture.FAN_OUT,
    ReformatPosture.RELY_ON_EXTERNAL,
    ReformatPosture.CLARIFY,
})


def _include_reformat_narration(posture: ReformatPosture) -> bool:
    return posture in _INCLUDE_REFORMAT_NARRATION


# How many of the pool's top candidates to scan for the diversity signal --
# large enough to catch a dominant duplicate cluster, small enough to stay
# cheap (this runs synchronously in the request path).
_DIVERSITY_TOP_K = 10


def _normalize_text_for_dedup(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def _build_pool_metadata(
    slots: list, pool_results: list[PoolResult],
) -> dict:
    """Per-slot corpus-depth signal (top_score_percentile/pool_size/
    distinct_content_topk, consumed by priors.compute_depth_bucket) -- slot
    fields themselves ride on RoutingContext.slots verbatim now (Router's
    real-seam fix, 182/182), so this no longer reconstructs
    slot_semantics/capacity/required/priority from a dict; that
    reconstruction was exactly the stale-comment gap Router closed by
    completing ctx.slots rather than deleting the words.

    Matched to slots by rewritten_query text, not index position -- correct
    for both PRECISE (1 slot, 1 pool result) and FAN_OUT (N slots, N pool
    results), same correlation discipline Slots uses for its own
    theme<->query mapping.

    top_score_percentile is a real approximation, not fabricated data: max
    candidate score in the slot's pool, clamped to [0,1] -- Router's own
    documented fallback path when historical percentile data isn't
    available yet (it isn't -- Eval's calibration loop is still
    bootstrapping). Only `vector`'s cosine similarity is genuinely
    [0,1]-scaled; `tag_select`'s `score` is a raw tag-coverage COUNT (e.g.
    3, 4), not a probability -- mixing it into this max and clamping
    silently mapped any coverage >= 1 straight to a fake "perfect" 1.0,
    with nothing to do with real retrieval confidence (found by Pool
    investigating Router's depth-bucket bug report, verified live
    2026-07-23: a genuine tag_select coverage=4 hit producing
    top_score_percentile=1.0 on a query with real signal). bm25_score
    (ts_rank_cd) is approximately bounded and kept clamped as a safety net,
    not because it's the source of this bug -- `tag_select`'s raw score is
    excluded from this calculation entirely instead of being rescaled,
    since there's no principled coverage-count -> [0,1] mapping to invent.

    distinct_content_topk (Router's ask, 2026-07-23: "ten copies of one
    chunk is evidence-of-one, not evidence-of-ten"): count of distinct
    normalized chunk texts among the top _DIVERSITY_TOP_K candidates.
    Deliberately groups by normalized TEXT, not `content_sha` -- verified
    live that this schema's `content_sha` is NOT a reliable text-dedup key
    (5 chunks with byte-identical text each carried a different
    content_sha), so a naive "distinct content_sha" check would report a
    100%-duplicate cluster as fully diverse and completely miss the bug.
    Same normalized-text fallback Pool's own dedup already uses when
    content_sha is unavailable.
    """
    pool_by_query = {pr.query: pr for pr in pool_results}
    default_pool = pool_results[0] if len(pool_results) == 1 else None

    metadata: dict = {}
    for slot in slots:
        pr = pool_by_query.get(slot.rewritten_query) or default_pool
        candidates = pr.candidates if pr else []
        scores = [
            c.score for c in candidates
            if c.score is not None and c.source_arm == "vector"
        ] + [
            c.bm25_score for c in candidates if c.bm25_score is not None
        ]
        top_score = max((min(max(s, 0.0), 1.0) for s in scores), default=0.0)
        top_k = candidates[:_DIVERSITY_TOP_K]
        distinct_content_topk = len({
            _normalize_text_for_dedup(c.text) for c in top_k
        })
        metadata[slot.slot_id] = {
            "top_score_percentile": top_score,
            "pool_size": len(candidates),
            "distinct_content_topk": distinct_content_topk,
        }
    return metadata


async def _run_router(
    query: str,
    slots: list,
    pool_results: list[PoolResult],
    resource_posture,
    gate_result: GateResult,
    payer_context: PayerContext | None,
    caller_mode: str | None,
    attempt: int = 0,
    retry_of_decision_id: str | None = None,
    forced_strategy: str | None = None,
    mode_override: str | None = None,
    reformat_result: ReformatResult | None = None,
) -> RouterDecision:
    pool_metadata = _build_pool_metadata(slots, pool_results)
    # attempt>0 marks a whole-loop technical-failure retry (2026-07-24,
    # Ananth's "ask once, try our best" principle) -- distinct agent_id so
    # Router's persisted rag_query_decisions row is distinguishable per
    # attempt, not a silent second observation of the same logical query
    # in Eval's calibration. No schema change: agent_id is already a plain
    # text column. retry_of_decision_id (Router's ask, 2026-07-24): the
    # attempt-0 decision_id, when it survived the failure (Router persisted
    # it before the pipeline died later, e.g. during Fillers) -- lands in
    # upstream_diagnostics so Router's persist_decision can stamp it into
    # feature_vector JSONB for EXACT pairing, not fuzzy (agent_id, query,
    # time-window) matching. None if the failure happened before Router
    # ever ran (genuinely nothing to pair against).
    ctx = RoutingContext(
        query=query,
        agent_id="retriever-orchestrator" if attempt == 0 else f"retriever-orchestrator-retry{attempt}",
        resource_posture=RouterResourcePosture(
            speed_budget=resource_posture.speed_budget,
            confidence_bar=resource_posture.confidence_bar,
            max_attempts_per_slot=resource_posture.max_attempts,
            caller_mode=caller_mode or "chat.default",
            # Straight pass-through, no name mapping on this side -- Router's
            # posture_dict bridge maps token_budget to the internal per-slot
            # key (2026-07-23, closing the 3-layer gap Structure caught).
            token_budget=resource_posture.token_budget,
            authority_requirement=resource_posture.authority_requirement,
        ),
        slots=slots,  # verbatim AnswerSlot objects -- authoritative for slot construction
        pool_metadata=pool_metadata,  # depth signal only, per Router's real-seam fix
        gate_j_codes=gate_result.j_codes,
        gate_d_codes=gate_result.d_codes,
        payer_crawlable=(payer_context.crawlable if payer_context else None),
        # Caller-forced strategy (2026-07-24): the offline calibration matrix
        # and the served /api/retriever endpoint's forced runs -- Router's
        # dispatch treats a non-None ctx.forced_strategy as the isolation
        # bypass (single-strategy ladder, no shadows), exactly the
        # uncontaminated path Eval's recall curves need. None = normal
        # dispatch (greedy/optimizer/bayesian or the throttle's own draw).
        forced_strategy=forced_strategy,
        # Caller-pinned executed allocator (greedy/optimizer/bayesian) for the
        # throttle comparison — dispatch precedence: calibration > forced >
        # mode_override > throttle draw. None = normal production dispatch.
        mode_override=mode_override,
        upstream_diagnostics={
            **({"retry_of_decision": retry_of_decision_id} if retry_of_decision_id else {}),
            **(
                {
                    "reformat_posture": reformat_result.posture.value,
                    "reformat_fanout_n": len(reformat_result.fanout_themes),
                }
                if reformat_result is not None
                else {}
            ),
        },
    )
    return await router_route(AsyncSessionLocal, ctx)


# Safety valve against a pathological infinite loop -- NOT a design decision.
# decide_continuation() already terminates on its own (no turn-eligible
# slots, or budget exhausted); this just guarantees termination even if a
# future bug in that logic or in a filler's verdict computation would
# otherwise spin forever.
_MAX_OBSERVER_TURNS = 5


def _observer_verdict(
    filled_slot: FilledSlot, error: str | None, strategy: str,
    attempt_number: int, max_attempts: int,
) -> tuple[str, str]:
    """Per-slot "would this benefit from another turn" verdict -- delegates
    to Observer's real per-strategy adequacy check (observer.py), each
    filler judged on its OWN bar (capacity-aware fill for a/b/s, decay-floor
    score check for d), not a shared bare-occupancy proxy.

    Supersedes the bare-occupancy `_stopgap_verdict` this function replaced
    (2026-07-26, Ananth's direct instruction to wire Observer in and verify
    against live queries). This module's own header said the swap was
    "gated on Eval's calibration run" -- that's now happened (22-query
    forced bank + live router calibration, this session), and Observer's
    own module header confirms a/b/c/s were already SIGNED OFF, not
    speculative.

    The only thing Observer's evaluate() doesn't itself judge is a filler
    raising an exception -- that's an orchestrator/infra concern (VERDICT_
    ERROR), not a per-strategy adequacy question, so it's handled here
    before ever calling into Observer. `has_remaining_rungs` is NOT passed
    through: Observer's EXHAUSTED_ATTEMPTS is purely budget-based
    (attempt_number>=max_attempts), by design (see observer.py's module
    docstring) -- whether the ladder has more rungs left is Router's
    decide_continuation() concern (SlotTurnInput.remaining_rungs), a
    separate layer that already receives this independently.
    """
    if error is not None:
        return VERDICT_ERROR, f"filler raised: {error}"
    return observer.evaluate(
        strategy, filled_slot,
        attempt_number=attempt_number, max_attempts=max_attempts,
    )


def _portfolio_stopgap_verdict(
    total_delivered: int, planned_total: int, any_error: str | None,
) -> tuple[str, str]:
    """v1 verdict for a portfolio-allocated slot (Router's correction,
    2026-07-24, blend-model-design.md §4): success is measured against
    what the ALLOCATOR PLANNED (sum of k_i), NOT slot.capacity -- a
    budget-bound portfolio that delivers 100% of its plan must not be
    marked under-filled just because slot.capacity (the ceiling that
    shaped the plan, not the success criterion) was never the real target.
    No multi-turn expansion for portfolio slots yet -- deferred to
    Synthesis's CoverageDiagnostic (Router/Synthesizer, in design) -- so
    this always resolves to a terminal verdict, never WOULD_BENEFIT.
    """
    if any_error is not None:
        return VERDICT_ERROR, f"filler raised: {any_error}"
    if total_delivered >= planned_total:
        return VERDICT_SATISFIED, f"portfolio delivered {total_delivered}/{planned_total} planned"
    return VERDICT_EXHAUSTED_ATTEMPTS, (
        f"portfolio under-delivered ({total_delivered}/{planned_total}) -- "
        "expansion deferred to CoverageDiagnostic"
    )


async def _run_fillers_simple(
    db: AsyncSession,
    slots: list,
    pool_results: list[PoolResult],
    router_decision: RouterDecision,
    raw_query: str,
    gate_result: GateResult,
    payer_context: PayerContext | None = None,
    prescreen_search_task: "asyncio.Task[PrescreenedSearch] | None" = None,
) -> FilledShape:
    """Real multi-turn loop, wired to Router's decide_continuation()
    (app/services/router/continuation.py, built + acked 2026-07-23): each
    slot runs its ladder's rungs one at a time; after every rung, this loop
    computes a per-slot (verdict, reason) -- via Observer's real per-filler
    criteria (_observer_verdict, wired 2026-07-26) -- and hands ALL slots'
    verdicts + the elapsed/remaining time
    budget to Router's aggregation, which decides ONE query-level "another
    turn, or done" call. This is the real seam Observer's design landed on:
    per-slot verdicts are this module's/Observer's concern, cross-slot
    aggregation is Router's, never mixed.

    RETAIN model (2026-07-24, replacing the earlier DISCARD stopgap --
    see docs/rag-agents/blend-model-design.md, signed off by Ananth/Router/
    Synthesizer/Eval): every executed rung's chunks are kept, not replaced.
    `state[slot_id]["retained_chunks"]` accumulates across every turn; the
    FINAL FilledSlot handed to Synthesis is built from that union, not from
    whichever rung happened to run last. Per-rung VERDICT computation still
    uses that individual rung's own occupancy (Observer's "per-rung
    sufficiency," not cumulative) -- retention only changes what Synthesis
    receives, not the continuation loop's own turn-by-turn logic. Synthesis
    is the layer that dedupes/reranks/fuses the retained union (its own
    dedup is already provenance-agnostic, no change needed there) -- this
    function deliberately does NOT dedupe before handing off, per the
    design doc's "one place responsible for this" call.

    Ride-along stamping (Eval's required addition, selection-bias
    segmentation): any slot Router's decision.ride_along marks gets
    ride_along=True recorded against its resulting observation, surfaced in
    this function's emit dict, so a future calibration consumer can
    segment "earned" SATISFIED observations from "rode along because a
    sibling was struggling" ones -- exactly the distinction Eval flagged as
    load-bearing, not decorative.

    Today's filler_a/b/s functions each loop over ALL slots in whatever
    AnswerShapeResult they're given internally (routing_ladders is
    documented "unused v1" in all three) -- they weren't built for
    per-slot ladder dispatch yet. Worked around here by constructing a
    single-slot AnswerShapeResult per call and taking just that one
    FilledSlot back out, rather than waiting on that redesign.
    """
    pool_by_query = {pr.query: pr for pr in pool_results}
    default_pool = pool_results[0] if len(pool_results) == 1 else None

    async def _try_strategy(strategy: str, slot, pr) -> tuple[FilledSlot, str | None]:
        single_shape = AnswerShapeResult(
            query=raw_query, posture=None, slots=[slot], reason="", slots_ms=0,
        )
        try:
            if strategy == "a":
                result = fill_shape_bm25(pr, single_shape)
            elif strategy == "b":
                result = fill_shape_vector(pr, single_shape)
            elif strategy == "c":
                result = await fill_shape_llm_retrieval(
                    pr, single_shape, raw_query,
                    db=db, agent_id="retriever-orchestrator",
                    tag_matches=[*gate_result.d_codes, *gate_result.j_codes, *gate_result.p_codes],
                )
            elif strategy == "d":
                # payer_context threaded through, NOT re-resolved here --
                # Router/Filler-d/Sitemap agreed design 2026-07-23 (avoids a
                # second ~3s Payor Platform call on top of the one already
                # paid for Router's plan-time crawl-gate).
                #
                # prescreen_search_task (2026-07-24, Web Search's fix): the
                # UNAWAITED task is what's threaded through, and it's only
                # awaited HERE, inside the "d" branch -- awaiting it earlier
                # (previously: right after Pool, before Router even ran)
                # stalled the entire pipeline on the full search cost for
                # every query that fired prescreen, including the majority
                # where Router never ends up picking "d" at all. Awaiting
                # the same Task object more than once (e.g. a second slot
                # or turn also landing on "d") is safe -- asyncio caches the
                # result after the first await, no re-execution.
                prescreened = await prescreen_search_task if prescreen_search_task else None
                result = await fill_shape_external(
                    pr, single_shape, raw_query,
                    db=db, agent_id="retriever-orchestrator",
                    tag_matches=[*gate_result.d_codes, *gate_result.j_codes, *gate_result.p_codes],
                    payer_context=payer_context,
                    prescreened=prescreened,
                )
            else:  # "s"
                result = await fill_shape_fact_store(
                    pr, single_shape, raw_query,
                    tag_matches=[*gate_result.d_codes, *gate_result.j_codes, *gate_result.p_codes],
                )
            return result.slots[0], None
        except Exception as exc:
            logger.warning(
                "orchestrator.fillers_simple slot=%s strategy=%s error=%r",
                slot.slot_id, strategy, exc,
            )
            return FilledSlot(
                slot_id=slot.slot_id, slot_semantics=slot.slot_semantics,
                capacity=slot.capacity, required=slot.required,
                chunks=[], occupancy=0, under_filled=True, over_filled=False,
            ), repr(exc)

    # Per-slot mutable state across turns: "remaining" is a real mutable
    # list (not a fixed chain + cursor) specifically so a not-yet-ready "d"
    # rung can be REORDERED behind an already-ready alternative without
    # losing its place in the ladder -- see _visible_remaining_rungs below.
    state: dict[str, dict] = {}
    for slot in slots:
        sequence = router_decision.routing_ladder.per_slot.get(slot.slot_id, [])
        chain = [s for s in sequence if s in _IMPLEMENTED_FILLERS]
        # Blend model (2026-07-24, blend-model-design.md) vs chain-mode
        # partial-fill (2026-07-26, Router's assign_chain_fills): the OLD
        # compat contract was "per_slot_portfolio non-empty == portfolio
        # mode" -- that broke the moment Router fixed the all-or-nothing
        # payload gate that was starving `d` (allocation.py/optimizer.py
        # now populate per_slot_portfolio for CHAIN allocators too, with
        # each rung's budget-scoped fill count). per_slot_portfolio's mere
        # presence can no longer discriminate "run everything concurrently,
        # no continuation" (true portfolio/blend) from "sequential chain,
        # but cap each rung's capacity by its assigned fill." The reliable
        # discriminator is the LADDER's own allocator label -- portfolio.py
        # explicitly sets ladder.allocator="portfolio"; greedy/optimizer/
        # bayesian never do -- a query-level property, not something to
        # infer per-slot. Verified live 2026-07-26 before this fix landed:
        # without it, every chain-mode query would have silently misrouted
        # through _run_portfolio_slot (bypassing Observer/continuation
        # entirely) on the next deploy after Router's change reached here.
        is_true_portfolio = router_decision.routing_ladder.allocator == "portfolio"
        fills = dict(router_decision.routing_ladder.per_slot_portfolio.get(slot.slot_id, {}))
        portfolio = fills if is_true_portfolio else {}
        state[slot.slot_id] = {
            "slot": slot, "remaining": chain,
            "filled_slot": None, "verdict": None, "reason": "", "ride_along": False,
            "d_ever_deferred": False,
            "executed": [],  # strategies actually run, in the order they actually ran (Router Sec11 boundary 2)
            # RETAIN model (2026-07-24): union of every executed rung's
            # chunks, not just the last one -- what Synthesis actually
            # fuses/dedupes from. Naive extend, no dedup here (Synthesis's
            # job, already provenance-agnostic).
            "retained_chunks": [],
            # Timing (Step 7, module-gates.md §7) -- per-attempt spans, one
            # entry per rung actually executed on this slot. Offsets are
            # relative to this call's t0 (monotonic), not wall-clock epoch --
            # same convention as pool/synthesis segment_ms, since only the
            # duration and relative ordering matter for the gate's
            # requirement, not absolute time.
            "attempt_spans": [],
            "portfolio": portfolio,
            # Chain-mode partial-fill assignment (Router, 2026-07-26): per-
            # strategy fill count from assign_chain_fills, used to cap each
            # rung's slot copy the same way portfolio execution already
            # does (replace(slot, capacity=fills[sid])) -- empty when the
            # allocator didn't compute one (e.g. forced/calibration paths),
            # in which case the rung runs at the slot's own full capacity,
            # unchanged from today's behavior. Empty for true-portfolio
            # slots too (that path uses `portfolio` above instead).
            "fills": {} if is_true_portfolio else fills,
            # Router's emit ask (2026-07-24): {strategy: {k_planned,
            # k_delivered}} -- the per-strategy fill-success signal Eval's
            # blend-era calibration needs, distinct from chain-era per-rung
            # verdicts. Empty for chain-mode slots.
            "portfolio_fill": {},
            # Data-collection posture (2026-07-24, Ananth's pullback +
            # Router's k0 requirement, blend-model-design.md): per EXECUTED
            # rung, {strategy, capacity, occupancy} -- capacity is the
            # requested fill depth (the slot's own capacity, = Eval's max-
            # depth-you'd-ever-fetch for forced calibration), occupancy is
            # what the strategy actually delivered. occupancy@capacity IS
            # the k0 of the resulting (strategy, depth_bucket) calibration
            # cell -- the forced-arm analog of portfolio_fill, recorded on
            # the chain/forced path (where portfolio_fill stays empty).
            # Without this, forced observations re-create the exact silent-
            # k0-rebase gap Eval's audit just closed.
            "fill_depth": [],
        }

    t0 = time.monotonic()
    latency_allowance_ms = getattr(router_decision.trace, "latency_allowance_ms", 0.0) or 0.0

    def _visible_remaining_rungs(state_entry: dict) -> list[str]:
        """View-only reorder (2026-07-24, Ananth via Web Search, spec'd and
        green-lit by Router in router-build-spec.md Sec11 -- planned-set-only,
        no unplanned substitutions): Router shouldn't commit to "d" as a
        slot's next attempt while its speculative prescreen isn't ready yet
        -- prioritize an already-ready strategy instead, since the search
        cost hasn't been paid down yet either way. Non-blocking readiness
        check (Task.done()), not a wait. "d" isn't removed from the ladder
        -- just moved behind the next alternative for THIS decision; a
        later turn re-checks readiness fresh (prescreen_search_task is one
        Task, self-reporting the same way throughout the query, so once
        it's actually done this reordering stops applying).

        Marks state_entry["d_ever_deferred"] whenever a reorder actually
        happens, so the caller can tell "d" never ran because readiness
        never cleared apart from "d" never ran because it wasn't planned/
        wasn't needed -- Router's required distinction (Sec11): this must
        surface as a distinct label, not fold into a generic
        failure/skip, or Eval's calibration cells for "d" absorb readiness
        noise as if it were performance noise.
        """
        remaining = state_entry["remaining"]
        if not remaining or remaining[0] != "d":
            return remaining
        if prescreen_search_task is None or prescreen_search_task.done():
            return remaining
        rest = remaining[1:]
        if not rest:
            return remaining  # "d" is the only option left -- nothing to defer behind
        state_entry["d_ever_deferred"] = True
        return [rest[0], "d"] + rest[1:]

    def _pick_and_consume(state_entry: dict) -> str:
        """Choose the rung to run THIS attempt from the readiness-reordered
        view, then remove exactly that one from the REAL remaining list
        (not necessarily index 0) -- so a deferred "d" stays present for a
        future turn instead of being silently dropped."""
        visible = _visible_remaining_rungs(state_entry)
        strategy = visible[0]
        state_entry["remaining"].remove(strategy)
        return strategy

    def _capped_slot(slot, state_entry: dict, strategy: str):
        """Chain-mode partial-fill (Router, 2026-07-26): if the allocator
        assigned this strategy a specific fill count (assign_chain_fills'
        budget-scoped assignment, e.g. `d` capped to 1 chunk instead of its
        full requested capacity), cap this rung's slot copy to that fill --
        same seam pattern portfolio execution already uses (replace(slot,
        capacity=k_i)). Falls back to the slot's own full capacity,
        unchanged, when no fill was assigned (forced/calibration paths,
        or an allocator that hasn't computed one)."""
        fill = state_entry["fills"].get(strategy)
        return replace(slot, capacity=fill) if fill is not None else slot

    async def _run_portfolio_slot(slot_id: str, st: dict) -> None:
        """Blend model (2026-07-24, Router-approved seam): execute EVERY
        strategy in a portfolio-allocated slot's {strategy: k_i}
        CONCURRENTLY in turn 0, each capped to its own allocated k_i via a
        capacity-adjusted slot copy -- matches Router's payload accounting
        (Σ k_i·tokens_i stays true because a filler literally cannot exceed
        its k_i) and latency model (MAX over contributors, each gated on
        its own p50). Retention unions everyone's chunks (already built);
        no multi-turn expansion for portfolio slots -- turn 0 IS the
        complete calibrated plan, deferred further only once Synthesis's
        CoverageDiagnostic exists to drive it honestly.
        """
        slot = st["slot"]
        portfolio = st["portfolio"]
        pr = pool_by_query.get(slot.rewritten_query) or default_pool
        if pr is None:
            st["filled_slot"] = FilledSlot(
                slot_id=slot.slot_id, slot_semantics=slot.slot_semantics,
                capacity=slot.capacity, required=slot.required,
                chunks=[], occupancy=0, under_filled=True, over_filled=False,
            )
            st["verdict"], st["reason"] = VERDICT_EXHAUSTED_ATTEMPTS, "no_pool_result_for_portfolio_slot"
            st["remaining"] = []
            return

        t_attempt_start = int((time.monotonic() - t0) * 1000)
        results = await asyncio.gather(*[
            _try_strategy(strategy, replace(slot, capacity=k_i), pr)
            for strategy, k_i in portfolio.items()
        ])
        t_attempt_end = int((time.monotonic() - t0) * 1000)

        any_error: str | None = None
        total_delivered = 0
        for (strategy, k_i), (filled_slot, error) in zip(portfolio.items(), results):
            st["attempt_spans"].append({
                "strategy": strategy, "attempt_number": 1,
                "t_attempt_start_ms": t_attempt_start, "t_attempt_end_ms": t_attempt_end,
            })
            st["executed"].append(strategy)
            st["retained_chunks"].extend(filled_slot.chunks)
            st["portfolio_fill"][strategy] = {
                "k_planned": k_i, "k_delivered": filled_slot.occupancy,
            }
            total_delivered += filled_slot.occupancy
            any_error = any_error or error

        st["filled_slot"] = FilledSlot(
            slot_id=slot.slot_id, slot_semantics=slot.slot_semantics,
            capacity=slot.capacity, required=slot.required,
            chunks=list(st["retained_chunks"]), occupancy=total_delivered,
            under_filled=total_delivered < slot.capacity, over_filled=False,
        )
        planned_total = sum(portfolio.values())
        st["verdict"], st["reason"] = _portfolio_stopgap_verdict(total_delivered, planned_total, any_error)
        st["remaining"] = []  # no expansion yet -- signals decide_continuation there's nothing left to try

    # Turn 0: run the first rung for every slot that has one -- or, for a
    # portfolio-allocated slot, run its whole portfolio concurrently.
    for slot_id, st in state.items():
        if st["portfolio"]:
            await _run_portfolio_slot(slot_id, st)
            continue
        slot = st["slot"]
        pr = pool_by_query.get(slot.rewritten_query) or default_pool
        if not st["remaining"] or pr is None:
            st["filled_slot"] = FilledSlot(
                slot_id=slot.slot_id, slot_semantics=slot.slot_semantics,
                capacity=slot.capacity, required=slot.required,
                chunks=[], occupancy=0, under_filled=True, over_filled=False,
            )
            st["verdict"], st["reason"] = VERDICT_EXHAUSTED_ATTEMPTS, "no_implemented_strategy_or_pool"
            continue
        strategy = _pick_and_consume(st)
        t_attempt_start = int((time.monotonic() - t0) * 1000)
        filled_slot, error = await _try_strategy(strategy, _capped_slot(slot, st, strategy), pr)
        t_attempt_end = int((time.monotonic() - t0) * 1000)
        st["attempt_spans"].append({
            "strategy": strategy, "attempt_number": 1,
            "t_attempt_start_ms": t_attempt_start, "t_attempt_end_ms": t_attempt_end,
        })
        st["executed"].append(strategy)
        st["filled_slot"] = filled_slot
        st["retained_chunks"].extend(filled_slot.chunks)
        st["fill_depth"].append({
            # Router's k0 requirement (2026-07-26): capacity reflects what
            # was ACTUALLY requested of this rung -- filled_slot.capacity,
            # not the slot's blanket capacity -- so a capped chain-mode
            # fill (e.g. `d` scoped to 1 via assign_chain_fills) reports
            # its real scoped occupancy@capacity, exactly like the
            # portfolio path's portfolio_fill already does.
            "strategy": strategy, "capacity": filled_slot.capacity, "occupancy": filled_slot.occupancy,
        })
        st["verdict"], st["reason"] = _observer_verdict(
            filled_slot, error, strategy,
            attempt_number=len(st["executed"]),
            max_attempts=router_decision.resource_posture.max_attempts_per_slot,
        )

    # Multi-turn loop: ask Router's aggregation whether another turn is
    # justified, given every slot's current verdict + the time budget.
    for _ in range(_MAX_OBSERVER_TURNS):
        turn_inputs = [
            SlotTurnInput(
                slot_id=slot_id,
                # Readiness-reordered VIEW, not the raw remaining list --
                # so decide_continuation's remaining_rungs[0] (what it
                # actually plans to run) is never a not-yet-ready "d".
                remaining_rungs=tuple(_visible_remaining_rungs(st)),
                verdict=st["verdict"], reason=st["reason"],
                required=st["slot"].required,
            )
            for slot_id, st in state.items()
        ]
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        decision = decide_continuation(turn_inputs, elapsed_ms, latency_allowance_ms)
        if not decision.new_turn:
            break

        for slot_id, strategy in decision.turn_rungs.items():
            st = state[slot_id]
            slot = st["slot"]
            pr = pool_by_query.get(slot.rewritten_query) or default_pool
            t_attempt_start = int((time.monotonic() - t0) * 1000)
            filled_slot, error = await _try_strategy(strategy, _capped_slot(slot, st, strategy), pr)
            t_attempt_end = int((time.monotonic() - t0) * 1000)
            st["attempt_spans"].append({
                "strategy": strategy, "attempt_number": len(st["attempt_spans"]) + 1,
                "t_attempt_start_ms": t_attempt_start, "t_attempt_end_ms": t_attempt_end,
            })
            st["remaining"].remove(strategy)  # not necessarily index 0 -- "d" may still sit ahead, deferred
            st["executed"].append(strategy)
            st["filled_slot"] = filled_slot  # this rung's own result, for verdict computation only
            st["retained_chunks"].extend(filled_slot.chunks)  # RETAIN model -- unioned, not replaced
            st["fill_depth"].append({
                "strategy": strategy, "capacity": slot.capacity, "occupancy": filled_slot.occupancy,
            })
            st["verdict"], st["reason"] = _observer_verdict(
                filled_slot, error, strategy,
                attempt_number=len(st["executed"]),
                max_attempts=router_decision.resource_posture.max_attempts_per_slot,
            )
            if slot_id in decision.ride_along:
                st["ride_along"] = True

    # Clean up the prescreen task if "d" was never tried on any slot/turn --
    # the common case, since s/a/b usually satisfy first (Router picked "d"
    # 0/9 times in this session's natural-allocator sampling). Left
    # unawaited, an eventually-failing search would raise "Task exception
    # was never retrieved"; left pending (rare -- cancelled tasks resolve
    # fast), garbage collection would warn "Task was destroyed but it is
    # pending." Cancelling first is a no-op if it already finished.
    if prescreen_search_task is not None and not prescreen_search_task.cancelled():
        prescreen_search_task.cancel()
        try:
            await prescreen_search_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # A real search failure on a task nothing else consumed --
            # TECH's P2 finding, 2026-07-24: swallowing CancelledError here
            # is correct (that's this cancel() call's own effect), but a
            # genuine search error going unlogged made every unconsumed
            # prescreen failure invisible.
            logger.info(
                "orchestrator.fillers_simple: prescreen_search_task failed after "
                "cancellation, never consumed by any slot -- %r", exc,
            )

    # RETAIN model (2026-07-24): the FINAL FilledSlot handed to Synthesis is
    # built from the union of every executed rung's chunks, not whichever
    # rung ran last (the old DISCARD behavior). A slot that never executed
    # any rung (no_implemented_strategy_or_pool) falls back to the empty
    # FilledSlot already built for it above -- retained_chunks stays empty
    # in that case, so `if retained` correctly selects the fallback.
    filled_slots = []
    for slot in slots:
        st = state[slot.slot_id]
        retained = st["retained_chunks"]
        if retained:
            occupancy = len(retained)
            filled_slots.append(FilledSlot(
                slot_id=slot.slot_id, slot_semantics=slot.slot_semantics,
                capacity=slot.capacity, required=slot.required,
                chunks=retained, occupancy=occupancy,
                under_filled=occupancy < slot.capacity, over_filled=False,
            ))
        else:
            filled_slots.append(st["filled_slot"])
    total_assigned = sum(s.occupancy for s in filled_slots)
    return FilledShape(
        slots=filled_slots,
        total_chunks_assigned=total_assigned,
        filling_strategy="multi_turn_continuation_retain_model",
        emit={
            "slots_filled": len([s for s in filled_slots if s.occupancy > 0]),
            "under_filled": len([s for s in filled_slots if s.under_filled]),
            "final_verdicts": {sid: st["verdict"] for sid, st in state.items()},
            # Synthesis wiring (2026-07-24): reason strings alongside the
            # verdicts so compile_synthesis's SlotVerdict.reason isn't left
            # empty -- Eval's non-negotiable ask was the verdict/reason pair
            # passing through byte-for-byte, not just the verdict.
            "final_reasons": {sid: st["reason"] for sid, st in state.items()},
            "ride_along_slots": [sid for sid, st in state.items() if st["ride_along"]],
            # Router (router-build-spec.md Sec11) boundary 2: executed order,
            # emitted as executed -- what actually ran, in the order it ran.
            "executed_order": {sid: st["executed"] for sid, st in state.items()},
            # Router Sec11 boundary 3 (Eval-required distinction): "d" was
            # deferred for readiness at least once AND never got a turn to
            # actually run this query -- a DISTINCT label, deliberately NOT
            # folded into final_verdicts/EXHAUSTED_ATTEMPTS, so Eval's
            # calibration cells for "d" don't absorb readiness noise as if
            # it were the strategy's own performance.
            "prescreen_not_ready_deferred_slots": [
                sid for sid, st in state.items()
                if st["d_ever_deferred"] and "d" in st["remaining"]
            ],
            # Timing (Step 7) -- the concrete gap module-gates.md §7 called
            # out by name: "escalation loop must emit per-attempt timing
            # (t_attempt_start, t_attempt_end) before shape can be signed
            # off." One entry per rung actually executed, per slot, in
            # execution order.
            "attempt_spans": {sid: st["attempt_spans"] for sid, st in state.items()},
            # Router's emit ask (2026-07-24, blend-model-design.md §4):
            # {slot_id: {strategy: {k_planned, k_delivered}}} -- the
            # per-strategy fill-success signal Eval's blend-era calibration
            # needs (delivered-vs-planned), and the join key for their
            # reward-attribution ("did this strategy's chunks survive
            # rerank AND get cited"). Empty dict for chain-mode slots.
            "portfolio_fill": {sid: st["portfolio_fill"] for sid, st in state.items()},
            # Data-collection posture (2026-07-24): per-slot list of
            # {strategy, capacity, occupancy} for every executed rung on the
            # chain/forced path -- occupancy@capacity is each resulting
            # calibration cell's k0 (Router's requirement). For a FORCED
            # single-strategy run (the offline calibration matrix), each
            # slot's list has exactly one entry: the forced strategy's true
            # fill depth. Populated on the chain/forced path; empty for
            # portfolio slots (which use portfolio_fill instead).
            "fill_depth": {sid: st["fill_depth"] for sid, st in state.items()},
        },
    )


async def run_retriever_partial(
    db: AsyncSession, query: str, caller_mode: str | None = None, attempt: int = 0,
    retry_of_decision_id: str | None = None, token_budget_for_retrieval: int | None = None,
    forced_strategy: str | None = None, mode_override: str | None = None,
    force_fanout_queries: list[str] | None = None,
) -> RetrieverPartialResult:
    """Sequence Gate → Reformat → Structure → Slots → Pool. Stops there —
    Router onward doesn't exist yet. This function's own scope will shrink
    over time as real modules absorb what it currently does (today:
    sequencing, narrative-stitching, and picking single-query vs FAN_OUT
    Pool dispatch; no other business logic lives here or ever should).
    `caller_mode` passes straight through to Structure's ResourcePosture
    lookup, untouched by Gate/Reformat -- unrecognized values fall back to
    chat.default cleanly (Structure's own behavior, not this function's),
    which is exactly what absorbs the known 3-way caller_mode vocabulary
    mismatch without crashing.

    `token_budget_for_retrieval` (2026-07-24, Ananth's direct correction):
    Chat's real per-request context-window math, straight through to
    Structure -- replaces Structure's static per-mode guess when supplied.
    None (legacy callers) falls through to the static table, no breaking
    change. This is the actual fix for the payload-gate collapse this
    session root-caused (Structure's guessed 3000 vs a real ~10k+ from
    Chat) -- Router's per-chunk estimate fix alone wasn't the full story.

    `attempt` (2026-07-24): 0 for a normal call, >0 when
    run_retriever_partial_with_retry is retrying after a technical failure
    -- threaded to _run_router so the persisted decision row is
    distinguishable per attempt. This function itself has no retry logic;
    it's a single real pass, same as always. The retry loop lives one
    level up, wrapping this whole function.

    `force_fanout_queries` (2026-07-29, debug/calibration override, same
    class as `forced_strategy`): when set, overrides Reformat's own
    posture/rewritten_queries decision with FAN_OUT + these exact queries,
    regardless of what Gate's contour actually was. Exists to test whether
    manually decomposing a compound question into targeted sub-queries
    (each aimed at one specific fact) recovers what a single EXACT-contour
    passthrough query misses -- EXACT's "pass through unchanged" behavior
    has no notion of "this question needs multiple distinct facts,"
    confirmed live on a real compound billing question. Reuses Pool's
    existing FAN_OUT dispatch (run_pool_fanout) unchanged -- that multi-
    query machinery already exists and works; this override just feeds it
    from a different (manual, not lexicon-cluster) source. None (default)
    is a complete no-op, byte-identical to today's behavior.
    """
    t0 = time.monotonic()

    gate_result = await run_gate(db, query)
    reformat_result = await run_reformat(db, gate_result)
    if force_fanout_queries:
        # run_slots' FAN_OUT branch is entirely driven by fanout_themes (one
        # theme per rewritten_query, same order -- see slots.py's
        # theme_to_query zip) -- an empty list there means "no fanout_themes
        # provided (unexpected)", zero slots built, breadth=0, no_retrieval
        # downstream even though Pool itself gets real candidates for every
        # forced query (confirmed live: pool_candidate_counts=[517,517,517]
        # yet 0 final chunks). One flat-score placeholder theme per query.
        fake_themes = [
            FanoutTheme(theme_label=f"debug override {i + 1}", score=1.0)
            for i in range(len(force_fanout_queries))
        ]
        reformat_result = ReformatResult(
            query=reformat_result.query,
            posture=ReformatPosture.FAN_OUT,
            rewritten_queries=list(force_fanout_queries),
            fanout_themes=fake_themes,
            reason=(
                f"DEBUG override: forced FAN_OUT with {len(force_fanout_queries)} "
                f"explicit queries (was {reformat_result.posture.value}: {reformat_result.reason!r})"
            ),
            reformat_ms=reformat_result.reformat_ms,
        )
    structure_result = run_structure(
        reformat_result, caller_mode=caller_mode, token_budget_for_retrieval=token_budget_for_retrieval,
    )
    slots_result = run_slots(structure_result)

    # Payer context (site_domain/display_name/crawlable) only depends on
    # Gate's j_codes, same as Pool -- run both concurrently so this costs
    # zero added wall-clock in the common case (Router/Filler-d/Filler-f
    # agreed design 2026-07-23). extract_payer_slug is sync/cheap; the real
    # cost is resolve_payer_context's live Payor Platform call, gathered
    # alongside Pool's fetch below.
    payer_slug = extract_payer_slug(gate_result.j_codes)
    payer_context_task = (
        asyncio.ensure_future(_resolve_payer_context_cached(
            db, payer_slug, speed_budget=structure_result.resource_posture.speed_budget,
        ))
        if payer_slug else None
    )

    # Speculative prefetch of Filler d's cheap SEARCH step only (not
    # fetch+synthesize), fired concurrently with Pool's build -- 2026-07-24,
    # Web Search built search-only + gate helper, this wires the dispatch
    # side (their scope stopped at the API surface, orchestration is mine).
    # Gated on authority_requirement != "citable_required": for
    # citable_required queries d is already excluded by Router's authority
    # gate, so prefetching its search would be pure waste; this only fires
    # for the queries where d could actually get picked. Waits on
    # payer_context_task internally (for site: restriction) but still runs
    # fully concurrent with Pool below, same as payer_context itself.
    # FAN_OUT scoped OUT for v1 (not decided how to dispatch N speculative
    # searches per theme slot) -- single-query PRECISE/EXACT path only.
    async def _prescreen_task() -> PrescreenedSearch:
        payer_context = await payer_context_task if payer_context_task else None
        tag_matches = [*gate_result.d_codes, *gate_result.j_codes, *gate_result.p_codes]
        return await prescreen_search(
            query, tag_matches=tag_matches, payer_context=payer_context, n_search=5,
        )

    prescreen_search_task = (
        asyncio.ensure_future(_prescreen_task())
        if (
            reformat_result.posture != ReformatPosture.FAN_OUT
            and should_prescreen_search(structure_result.resource_posture.authority_requirement)
        )
        else None
    )

    pool_results: list[PoolResult] = []
    if structure_result.resource_posture.breadth > 0:
        adapter = PublicSourceAdapter(db)
        if reformat_result.posture == ReformatPosture.FAN_OUT:
            pool_results = await run_pool_fanout(
                db, structure_result.rewritten_queries, gate_result,
                structure_result.resource_posture, adapter,
                fanout_themes=structure_result.fanout_themes,
            )
        else:
            rq = structure_result.rewritten_queries[0] if structure_result.rewritten_queries else query
            pool_results = [await run_pool_for_query(
                db, rq, gate_result, structure_result.resource_posture, adapter,
            )]
    # else: no-retrieval postures (CLARIFY/CLARIFY_REPHRASE/DECLINE) reach
    # here with an all-zero ResourcePosture -- same convention Pool itself
    # uses, not an error case. pool_results stays [].

    payer_context = await payer_context_task if payer_context_task else None
    # NOT awaited here (2026-07-24 fix, Web Search's catch): _run_router
    # never needs prescreen_search_task -- awaiting it unconditionally
    # before Router runs stalled the ENTIRE pipeline (Router + every other
    # filler) on the full search cost for any query that fires prescreen,
    # even the majority where Router doesn't end up picking "d" at all.
    # The task itself (not its result) is threaded through to
    # _run_fillers_simple, which only awaits it inside the "d" branch --
    # i.e. only paid if Router actually assigns a slot to "d".

    router_decision: RouterDecision | None = None
    filled_shape: FilledShape | None = None
    synthesis_result: SynthesisResult | None = None
    router_ms = 0
    fillers_ms = 0
    synthesis_ms = 0
    if slots_result.slots and pool_results:
        t_router = time.monotonic()
        router_decision = await _run_router(
            query, slots_result.slots, pool_results,
            structure_result.resource_posture, gate_result, payer_context, caller_mode,
            attempt, retry_of_decision_id, forced_strategy, mode_override,
            reformat_result=reformat_result,
        )
        router_ms = int((time.monotonic() - t_router) * 1000)

        t_fillers = time.monotonic()
        try:
            filled_shape = await _run_fillers_simple(
                db, slots_result.slots, pool_results, router_decision, query, gate_result,
                payer_context, prescreen_search_task,
            )
        except Exception as exc:
            # Router already ran and persisted a decision row before this
            # failed -- stamp its decision_id on the exception (dynamic
            # attribute, no new exception type needed) so
            # run_retriever_partial_with_retry can pass it as
            # retry_of_decision_id on the next attempt (Router's ask,
            # 2026-07-24: exact pairing via feature_vector JSONB instead of
            # fuzzy agent_id/query/time-window matching).
            #
            # LOAD-BEARING INVARIANT (Eval, 2026-07-24) -- protect this if
            # this block or router_decision's construction ever gets
            # refactored: an attempt-0 rag_query_decisions row exists IFF
            # retry_of_decision is non-None on its retry partner. Holds by
            # construction today because decision_id is created AT persist
            # time inside _run_router -- if Router persisted a row,
            # router_decision.decision_id is already real by the time we'd
            # reach this except block; if Router (or anything before it)
            # died first, router_decision is still None and this line
            # correctly yields None too. If a future change ever lets a row
            # persist WITHOUT its id reaching here, Eval's calibration
            # silently double-counts an orphaned, un-excludable row -- this
            # exact line is where that guarantee is made or broken.
            exc.retriever_partial_decision_id = getattr(router_decision, "decision_id", None)
            raise
        fillers_ms = int((time.monotonic() - t_fillers) * 1000)

        # Synthesis (Step 5), wired in 2026-07-24 -- first live caller of
        # compile_synthesis(). Same exception-stamping treatment as Fillers
        # above: if this raises (compile_synthesis's own 3-layer retry
        # already exhausted before it re-raises -- see synthesis_contracts.py's
        # SynthesisTelemetry note), the whole-loop retry is the real
        # recovery mechanism, and Eval's decision_id pairing invariant must
        # hold here too, not just for Fillers' own failures.
        t_synthesis = time.monotonic()
        try:
            verdicts = {
                sid: SlotVerdict(
                    verdict=v or "",
                    reason=filled_shape.emit.get("final_reasons", {}).get(sid, ""),
                    ride_along=sid in filled_shape.emit.get("ride_along_slots", []),
                )
                for sid, v in filled_shape.emit.get("final_verdicts", {}).items()
            }
            synthesis_result = await compile_synthesis(
                query, filled_shape, db=db, verdicts=verdicts,
                # Real containment fix (2026-07-24, a live ~473K-char
                # prompt incident): neighbor completion has no budget
                # awareness of its own -- pass Structure's real
                # (now caller-supplied) token_budget through so Synthesis
                # can trim back to it before this ever reaches Contract/Chat.
                token_budget=structure_result.resource_posture.token_budget or None,
                # Router's seam (2026-07-24, blend-model-design.md §4):
                # the ex-ante per-slot retrieval spend, reused as SPLIT
                # WEIGHTS for the synthesis-input budget once the fusion
                # rewrite consumes it -- not the synthesis-input budget
                # itself. Real RouterDecision field, not invented here.
                per_slot_payload_tokens=(
                    dict(router_decision.routing_ladder.per_slot_payload_tokens)
                    if router_decision else None
                ),
                # Data-collection posture (2026-07-24, Synthesizer's explicit
                # flag -- NOT chunk-shape inference, which would wrongly fire
                # in ordinary blend-mode whenever one filler happens to
                # satisfy a slot alone). True IFF this is a forced-family
                # dispatch (calibration / caller-forced / the 1-in-5
                # data-collection throttle -- all set dispatch_path=="forced",
                # dispatch.py). In that mode Synthesis skips MMR's real
                # redundancy-drop/budget-cutoff so the forced strategy's true
                # top-X reaches Eval's grader UNCONTAMINATED (MMR still runs
                # as a log-only probe). Never fires for a normal
                # greedy/optimizer/bayesian query.
                data_collection_mode=(
                    router_decision.dispatch_path == "forced" if router_decision else False
                ),
            )
        except Exception as exc:
            exc.retriever_partial_decision_id = getattr(router_decision, "decision_id", None)
            raise
        synthesis_ms = int((time.monotonic() - t_synthesis) * 1000)
    elif prescreen_search_task is not None:
        # No-retrieval posture (0 slots/pool) but a prescreen was still
        # fired (its gating only checks posture/authority_requirement, not
        # breadth) -- nothing will ever await it in this path, so clean it
        # up explicitly rather than leave an unretrieved task exception.
        prescreen_search_task.cancel()
        try:
            await prescreen_search_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.info(
                "orchestrator: prescreen_search_task failed after cancellation "
                "on a no-slots path, never consumed -- %r", exc,
            )
    # else: no-retrieval postures (0 slots) or a no-retrieval pool (0
    # results, breadth==0) -- nothing for Router/Fillers to do, same
    # convention as Pool's own breadth==0 skip above.

    total_ms = int((time.monotonic() - t0) * 1000)
    pool_ms = sum(pr.pool_ms for pr in pool_results)

    if _include_reformat_narration(reformat_result.posture):
        narrative = f"{narrate_gate(gate_result)}\n\n{narrate_reformat(gate_result, reformat_result)}"
    else:
        narrative = narrate_gate(gate_result)
    narrative_full = (
        f"--- Shape: Gate ---\n{narrate_gate_full(gate_result)}\n\n"
        f"--- Shape: Reformat ---\n{narrate_reformat_full(gate_result, reformat_result)}"
    )

    return RetrieverPartialResult(
        query=query,
        gate=gate_result,
        reformat=reformat_result,
        structure=structure_result,
        slots=slots_result,
        pool=pool_results,
        payer_context=payer_context,
        router_decision=router_decision,
        filled_shape=filled_shape,
        synthesis=synthesis_result,
        gate_ms=gate_result.gate_ms,
        reformat_ms=reformat_result.reformat_ms,
        slots_ms=slots_result.slots_ms,
        pool_ms=pool_ms,
        router_ms=router_ms,
        fillers_ms=fillers_ms,
        synthesis_ms=synthesis_ms,
        total_ms=total_ms,
        narrative=narrative,
        narrative_full=narrative_full,
        pipeline_complete=synthesis_result is not None,
        next_step=(
            "Contract/Timing built (see contract.py) but not yet called from"
            " here -- build_contract() lives outside orchestrator.py to avoid"
            " a circular import (contract.py imports RetrieverPartialResult)."
            " Fillers now use Observer's real per-strategy evaluate()"
            " (_observer_verdict, wired 2026-07-26)."
            if synthesis_result is not None
            else "Router/Fillers/Synthesis — no slots or no pool result for this posture"
        ),
    )


async def run_retriever_partial_with_retry(
    db: AsyncSession, query: str, caller_mode: str | None = None, max_retries: int = 1,
    token_budget_for_retrieval: int | None = None, forced_strategy: str | None = None,
    mode_override: str | None = None, force_fanout_queries: list[str] | None = None,
) -> RetrieverPartialResult:
    """Whole-loop retry on TECHNICAL failure (Ananth, 2026-07-24): "ask
    once, we try our best to get first-pass resolution." If ANY unhandled
    exception reaches this level -- a dropped DB connection mid-query (the
    exact real failure mode seen live this session), an infra timeout,
    anything that would otherwise surface as a broken response -- retry
    the WHOLE pipeline once before giving up. This is deliberately NOT the
    same thing as a low-confidence result: Observer/Router already handle
    "the answer is weak" honestly (WOULD_BENEFIT, partial_infeasible,
    EXHAUSTED_ATTEMPTS) as a real, gradeable outcome, not a failure to
    paper over. This wrapper only catches the case where the pipeline
    itself broke, not the case where it ran fine and produced a weak
    answer.

    ONE retry, not a loop -- `max_retries=1` by default, matching "ask
    once" literally: we try our best twice, not indefinitely. Each
    attempt's Router decision is persisted under a distinct agent_id
    (`_run_router`'s `attempt` param) so Eval's calibration can tell a
    technical-failure retry apart from an independent second query for
    the same text, rather than silently double-counting it.

    Real cost, accepted deliberately: a retry re-pays Gate+Pool's full
    latency (often 8-12s) on top of whatever the failed attempt already
    spent. Worth it against returning nothing; still worth knowing before
    reading a total_ms number without this context.

    Exact retry pairing for Eval's calibration (Router's ask, 2026-07-24):
    if a failed attempt's Router decision was already persisted before the
    pipeline died later (e.g. during Fillers), the failing exception
    carries `retriever_partial_decision_id` (stamped in
    run_retriever_partial's own except block) -- threaded here into the
    next attempt as `retry_of_decision_id`, landing in Router's
    feature_vector JSONB so Eval can pair rows exactly instead of falling
    back to fuzzy (agent_id, query, time-window) matching. None if the
    failure happened before Router ever ran -- genuinely nothing to pair.
    """
    last_exc: Exception | None = None
    retry_of_decision_id: str | None = None
    for attempt in range(max_retries + 1):
        try:
            return await run_retriever_partial(
                db, query, caller_mode=caller_mode, attempt=attempt,
                retry_of_decision_id=retry_of_decision_id,
                token_budget_for_retrieval=token_budget_for_retrieval,
                forced_strategy=forced_strategy, mode_override=mode_override,
                force_fanout_queries=force_fanout_queries,
            )
        except Exception as exc:
            last_exc = exc
            retry_of_decision_id = getattr(exc, "retriever_partial_decision_id", None)
            logger.warning(
                "run_retriever_partial_with_retry: query_len=%d attempt=%d failed (%r) "
                "decision_id=%s -- %s",
                len(query), attempt, exc, retry_of_decision_id,
                "retrying" if attempt < max_retries else "giving up, re-raising",
            )
    raise last_exc
