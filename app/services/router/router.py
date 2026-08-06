"""Router main orchestrator — forced bypass or dual-allocator shadow A/B.

Lifecycle (runs ONCE upfront, before Fillers acts; never re-invoked):
  1. DISPATCH: forced (calibration/override), else production
  2. PLAN (production): BOTH allocators compute a RoutingLadder —
       greedy    (allocation.py, sequential-fallback heuristic)
       optimizer (optimizer.py, exact constrained solve)
     The A/B draw picks which ladder EXECUTES; the other is the SHADOW plan
     (computed, logged, never run). Spec addendum 2026-07-23.
  3. PERSIST: exactly one rag_query_decisions row (ONE-WRITER), carrying
     executed_ladder + shadow_ladder + confidence_bar (schema widening with
     DB/Eval).
  4. RETURN: RouterDecision with the executed ladder/trace + shadow
     ladder/trace. Narration renders on demand (emit-only, never persisted).
"""

from __future__ import annotations

import logging
import time

from app.services.router.dispatch import dispatch
from app.services.router.allocation import AnswerSlot, RoutingLadder, allocate_strategies
from app.services.router.optimizer import optimize_allocation
from app.services.router.bayesian_optimizer import optimize_allocation_bayesian
from app.services.router.priors import compute_depth_bucket, load_priors
from app.services.router.persist import persist_decision
from app.services.router.decision import RouterDecision, RoutingContext, ResourcePosture
from app.services.router.tracing import DecisionTrace

logger = logging.getLogger(__name__)


def _core_slot_id(routing_ladder: RoutingLadder, pool_metadata: dict) -> str | None:
    if not routing_ladder.per_slot:
        return None
    def prio(sid: str) -> int:
        return pool_metadata.get(sid, {}).get("priority", 1)
    return min(routing_ladder.per_slot.keys(), key=prio)


def _slots_from_pool(ctx: RoutingContext, resource_posture: ResourcePosture) -> list[AnswerSlot]:
    return [
        AnswerSlot(
            slot_id=sid,
            slot_semantics=meta.get("slot_semantics", "direct_answer"),
            capacity=int(meta.get("capacity", 5)),
            rewritten_query=meta.get("rewritten_query", ""),
            required=bool(meta.get("required", True)),
            priority=meta.get("priority", 1),
        )
        for sid, meta in ctx.pool_metadata.items()
    ]


def _make_trace(mode: str, role: str, dd, bundle, resource_posture) -> DecisionTrace:
    return DecisionTrace(
        mode=mode,
        role=role,
        mode_reason=dd.reason,  # carries the weights + draw arithmetic
        draw=dd.draw,
        priors_version=bundle.version,
        priors_source=bundle.source,
        caller_mode=resource_posture.caller_mode,
    )


async def route(db_session_factory, ctx: RoutingContext) -> RouterDecision:
    """Main Router entry point. See module docstring."""
    start = time.monotonic()
    resource_posture = ctx.resource_posture or ResourcePosture()
    bundle = load_priors()
    policy = bundle.exploration_policy

    dd = dispatch(
        is_calibration=ctx.is_calibration,
        forced_strategy=ctx.forced_strategy,
        allocator_weights=policy.get("allocator_weights"),
        query_key=ctx.query,
        allocator_override=ctx.allocator_override,
        # data-collection throttle (Ananth's pullback): policy-driven,
        # file-swappable — phase/fraction/arms all come from Eval's file
        phase=str(policy.get("phase", "bootstrap")),
        forced_fraction=float(policy.get("forced_fraction", 0.0) or 0.0),
        forced_arm_weights=policy.get("forced_arm_weights"),
        has_payor_context=bool(ctx.gate_j_codes),
    )

    posture_dict = {
        "speed_budget": resource_posture.speed_budget,
        "confidence_bar": resource_posture.confidence_bar,
        "accuracy_bar": resource_posture.accuracy_bar,
        "max_attempts_per_slot": resource_posture.max_attempts_per_slot,
        "tolerance_bands": resource_posture.tolerance_bands,
        "caller_mode": resource_posture.caller_mode,
        # query-level Gate signal → tag-gated eligibility (s); fail-closed
        "gate_j_codes": ctx.gate_j_codes,
        # Gate's d-codes → sitemap_links helper gating (topic-match precondition)
        "gate_d_codes": ctx.gate_d_codes,
        # tri-state crawlability → crawl-gated eligibility (d); fail-open on None
        "payer_crawlable": ctx.payer_crawlable,
        # per-slot payload allowance: Structure's token_budget maps 1:1 to the
        # internal per-slot key (None → resolve_constraints applies default)
        "token_allowance_per_slot": resource_posture.token_budget,
        # caller-declared citability → authority-gated eligibility (d)
        "authority_requirement": resource_posture.authority_requirement,
    }
    per_slot_depth = {
        sid: compute_depth_bucket(meta) for sid, meta in ctx.pool_metadata.items()
    }

    shadow_ladders: list[RoutingLadder] = []
    shadow_traces: list[DecisionTrace] = []

    if dd.path == "forced":
        # dd.forced_strategy covers BOTH caller-forced and the data-collection
        # throttle's drawn arm (ctx.forced_strategy is None on throttle hits)
        strategy_id = dd.forced_strategy or ctx.forced_strategy or "a"
        slot_ids = list(ctx.pool_metadata.keys()) or ["slot_0"]
        executed_ladder = RoutingLadder(
            per_slot={sid: [strategy_id] for sid in slot_ids},
            outcome="forced",
            feasible=True,
        )
        executed_trace = _make_trace("forced", "executed", dd, bundle, resource_posture)
        executed_trace.outcome = "forced"
        executed_trace.feasible = True
        # THROTTLE-forced keeps counterfactual shadows ("what would each
        # allocator have planned" beside the forced observation — the return
        # ticket to the blend); calibration/caller-forced stay isolated
        # (dd.shadow_allocators empty there).
        if dd.shadow_allocators:
            slots = list(ctx.slots) if ctx.slots else _slots_from_pool(ctx, resource_posture)
            from app.services.router.portfolio import allocate_portfolio
            shadow_fns = {
                "greedy": allocate_strategies,
                "optimizer": optimize_allocation,
                "bayesian": optimize_allocation_bayesian,
                "portfolio": allocate_portfolio,
            }
            for shadow_name in dd.shadow_allocators:
                try:
                    s_trace = _make_trace(shadow_name, "shadow", dd, bundle, resource_posture)
                    s_ladder = shadow_fns[shadow_name](
                        slots, ctx.pool_metadata, posture_dict, bundle=bundle, trace=s_trace
                    )
                    s_ladder.allocator = shadow_name
                    shadow_ladders.append(s_ladder)
                    shadow_traces.append(s_trace)
                except Exception:
                    logger.warning("router.shadow %s failed on throttle-forced query "
                                   "— degraded to absent", shadow_name, exc_info=True)
    else:
        # REAL SEAM: prefer verbatim AnswerShapeResult.slots from the
        # orchestrator; fall back to reconstructing from pool_metadata for
        # callers that predate the Slots integration.
        slots = list(ctx.slots) if ctx.slots else _slots_from_pool(ctx, resource_posture)
        from app.services.router.portfolio import allocate_portfolio
        allocators = {
            "greedy": allocate_strategies,
            "optimizer": optimize_allocation,
            "bayesian": optimize_allocation_bayesian,
            # blend model (signed 2026-07-24): shadow-only until Eval shifts
            # weights AND Retriever's retention wiring can execute a portfolio
            "portfolio": allocate_portfolio,
        }
        t_alloc = time.monotonic()
        executed_trace = _make_trace(dd.path, "executed", dd, bundle, resource_posture)
        executed_ladder = allocators[dd.path](
            slots, ctx.pool_metadata, posture_dict, bundle=bundle, trace=executed_trace
        )
        executed_ladder.allocator = dd.path
        allocate_ms = int((time.monotonic() - t_alloc) * 1000)

        # Shadow plans are ADVISORY-ONLY (§6b: plan-diagnostics, never outcome
        # attribution): a shadow failure must never block the production path —
        # that shadow degrades to absent with a warning (TECH §11).
        t_shadow = time.monotonic()
        for shadow_name in dd.shadow_allocators:
            try:
                s_trace = _make_trace(shadow_name, "shadow", dd, bundle, resource_posture)
                s_ladder = allocators[shadow_name](
                    slots, ctx.pool_metadata, posture_dict, bundle=bundle, trace=s_trace
                )
                s_ladder.allocator = shadow_name
                shadow_ladders.append(s_ladder)
                shadow_traces.append(s_trace)
            except Exception as exc:
                logger.warning("router.shadow_failed allocator=%s err=%s "
                               "(production path unaffected)", shadow_name, exc)
        shadow_ms = int((time.monotonic() - t_shadow) * 1000)

        if executed_ladder.outcome != "all_slots_cleared":
            # §2a: infeasibility is an EXPLICIT outcome, never silently absorbed.
            # Router surfaces it; Chat/Synthesis decide honest-no-answer vs
            # ask-for-relaxation. This is logged as a distinct signal, and the
            # outcome + per-slot statuses ride on the RouterDecision, the trace,
            # and the persisted executed_ladder for the orchestrator to act on.
            logger.warning(
                "router.outcome decision outcome=%s statuses=%s reason=%s",
                executed_ladder.outcome,
                executed_ladder.per_slot_status,
                executed_ladder.infeasibility_reason,
            )
        for s_ladder in shadow_ladders:
            # Shadow-diff telemetry (ids and numbers only — never query text)
            chains_differ = s_ladder.per_slot != executed_ladder.per_slot
            logger.info(
                "router.shadow_diff executed=%s shadow=%s chains_differ=%s "
                "dconf=%+.4f dcost=%+.1f dlatency_ms=%+d",
                executed_ladder.allocator, s_ladder.allocator, chains_differ,
                s_ladder.aggregate_confidence_estimate
                - executed_ladder.aggregate_confidence_estimate,
                s_ladder.total_estimated_cost - executed_ladder.total_estimated_cost,
                s_ladder.total_estimated_ms - executed_ladder.total_estimated_ms,
            )

    core_sid = _core_slot_id(executed_ladder, ctx.pool_metadata)
    core_chain = executed_ladder.per_slot.get(core_sid, []) if core_sid else []
    total_ms = int((time.monotonic() - start) * 1000)
    if dd.path == "forced":
        allocate_ms = 0
        shadow_ms = 0

    t_persist = time.monotonic()
    decision_id = await persist_decision(
        db_session_factory,
        agent_id=ctx.agent_id or "router-4c",
        query=ctx.query,
        is_calibration=ctx.is_calibration,
        is_prod=not ctx.is_calibration,
        eval_run_id=None,
        depth_bucket=per_slot_depth.get(core_sid, 4) if core_sid else 4,
        strategy_chosen=core_chain[0] if core_chain else "s",
        strategy_sequence=core_chain,
        executed_ladder={
            "allocator": executed_ladder.allocator or dd.path,
            "per_slot": executed_ladder.per_slot,
            "per_slot_status": executed_ladder.per_slot_status,
            "per_slot_lb": executed_ladder.per_slot_lb,
            "per_slot_terminal": executed_ladder.per_slot_terminal,
            "terminal_action": executed_ladder.terminal_action,
            "per_slot_helpers": executed_ladder.per_slot_helpers,
            "helpers": executed_ladder.helpers,
            "outcome": executed_ladder.outcome,
            "confidence": executed_ladder.aggregate_confidence_estimate,
            "cost": executed_ladder.total_estimated_cost,
            "latency_ms": executed_ladder.total_estimated_ms,
            "payload_tokens": executed_ladder.total_payload_tokens,
            "per_slot_payload_tokens": executed_ladder.per_slot_payload_tokens,
            "per_slot_portfolio": executed_ladder.per_slot_portfolio or None,
            "feasible": executed_ladder.feasible,
        },
        shadow_ladder=(
            # §6b: plan-diagnostics only. List of ALL untaken allocators' plans
            # (3-way A/B/C: two shadows per production query).
            {
                "plans": [
                    {
                        "allocator": s.allocator,
                        "per_slot": s.per_slot,
                        "per_slot_status": s.per_slot_status,
                        "per_slot_lb": s.per_slot_lb,
                        "outcome": s.outcome,
                        "confidence": s.aggregate_confidence_estimate,
                        "cost": s.total_estimated_cost,
                        "latency_ms": s.total_estimated_ms,
                        "payload_tokens": s.total_payload_tokens,
                        "per_slot_portfolio": s.per_slot_portfolio or None,
                        "feasible": s.feasible,
                    }
                    for s in shadow_ladders
                ]
            }
            if shadow_ladders else None
        ),
        confidence_bar=resource_posture.confidence_bar,
        feature_vector={
            "dispatch_mode": dd.path,
            # row hygiene (Eval): throttle-forced (unconditioned calibration
            # observation, shadows kept) vs calibration/caller-forced
            # (isolation) vs None (production draw)
            "bypass_kind": dd.bypass_kind,
            "shadow_allocators": [s.allocator for s in shadow_ladders],
            "allocator_weights": dd.weights,
            "draw": dd.draw,
            "per_slot_depth_buckets": per_slot_depth,
            # Eval-architect 2026-08-05 (via Retriever): the RAW pool signals
            # behind per_slot_depth_buckets (top_score_percentile, pool_size,
            # distinct_content_topk) were computed, consumed locally by
            # compute_depth_bucket, then discarded — never serialized. Needed
            # to stratify Wilson/Bayesian CIs by actual pool composition
            # instead of pooling all queries as identically-distributed.
            # Additive only: mirrors per_slot_depth_buckets's own pattern,
            # same dict, same trace path, no behavior change.
            "per_slot_pool_metadata": ctx.pool_metadata,
            "gate_j_codes": ctx.gate_j_codes,
            "gate_d_codes": ctx.gate_d_codes,
            "payer_crawlable": ctx.payer_crawlable,
            "router_allocate_ms": allocate_ms,
            "router_shadow_ms": shadow_ms,
            # Whole-loop technical retry (Retriever's "ask once, try our
            # best"): the retry attempt's row back-references the interrupted
            # attempt-0 decision so Eval pairs EXACTLY and excludes attempt-0
            # (ERROR-class infra cut). Exact pairing is required — fuzzy
            # time-window pairing can false-exclude real observations on
            # identical-text queries (Eval-ratified). Absent (None) on
            # non-retry rows.
            "retry_of_decision": ctx.upstream_diagnostics.get("retry_of_decision"),
        },
        strategy_scores=dict(executed_ladder.per_slot_confidence),
        priors_version=bundle.version,
        confidence=executed_ladder.aggregate_confidence_estimate,
        accuracy_estimate=executed_ladder.aggregate_accuracy_estimate,
        cost=executed_ladder.total_estimated_cost,
        total_ms=total_ms,
        leaf_key=core_sid or "unknown",
        gate_contour=ctx.upstream_diagnostics.get("gate_contour"),
        gate_underspecified_kind=ctx.upstream_diagnostics.get("gate_underspecified_kind"),
        reformat_posture=ctx.upstream_diagnostics.get("reformat_posture"),
        reformat_fanout_n=ctx.upstream_diagnostics.get("reformat_fanout_n"),
    )
    persist_ms = int((time.monotonic() - t_persist) * 1000)

    # CALIBRATION-DEDUP INVARIANT (Eval, on record 2026-07-23): attempt-0 row
    # persisted ⟺ its decision_id reaches a subsequent technical retry as
    # retry_of_decision. KNOWN NARROW CRACK, one-directional: a crash between
    # the persist above and the caller storing the returned decision_id
    # (these few lines + the return) leaves a persisted row whose id never
    # reaches the retry → un-excludable orphan that double-counts. Reverse
    # direction is benign (persist is fire-and-forget: a failed persist with
    # a returned id → exclusion of a nonexistent row = no-op). Mitigation is
    # Eval-side MONITORING (orphan-audit query that ALERTS, never excludes) —
    # not code: a function that died here has nothing to return. Do not
    # widen this window: keep persist as the LAST side effect before return.
    logger.info(
        "router.route decision_id=%s mode=%s feasible=%s "
        "router_allocate_ms=%d router_shadow_ms=%d router_persist_ms=%d total_ms=%d",
        decision_id, dd.path, executed_ladder.feasible,
        allocate_ms, shadow_ms, persist_ms, total_ms,
    )

    return RouterDecision(
        routing_ladder=executed_ladder,
        decision_id=decision_id,
        resource_posture=resource_posture,
        feature_context={
            "priors_version": bundle.version,
            "priors_source": bundle.source,
            "confidence_estimate": executed_ladder.aggregate_confidence_estimate,
            "feasible": executed_ladder.feasible,
            "dispatch_path": dd.path,
            "shadow_allocators": [s.allocator for s in shadow_ladders],
            # Retriever 2026-08-05 (post-deploy trace catch): feature_vector
            # above only reaches persist_decision's DB write — this dict is
            # what actually returns to the caller and flows into contract.py's
            # routing_keys (API/trace-explorer/bank summaries). The earlier
            # per_slot_pool_metadata fix landed the data in the DB but left it
            # invisible everywhere else — same fields, same source, mirrored
            # here so external consumers actually see it.
            "per_slot_depth_buckets": per_slot_depth,
            "per_slot_pool_metadata": ctx.pool_metadata,
        },
        dispatch_path=dd.path,
        reason=dd.reason,
        trace=executed_trace,
        shadow_ladders=shadow_ladders,
        shadow_traces=shadow_traces,
    )
