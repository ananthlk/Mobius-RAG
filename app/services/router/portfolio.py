"""Portfolio allocator — the blend-model's ex-ante retrieval allocation.

Blend-model-design.md §3 (Ananth-signed 2026-07-24): choose per-slot {k_i} —
chunks to FETCH per strategy — maximizing

    P(covered | {k_i}) = 1 − Π_i (1 − q_i)^{k_i}

subject to   Σ_i k_i·tokens_i ≤ token_budget(retrieval)   and   k_i ≤ cap_i,

where q_i is the per-chunk hit probability from the Eval-RATIFIED capacity
transform of existing (strategy × depth) cells:

    q_i = 1 − (1 − r_i)^(1/k₀)          (r_i = cell recall_lift, k₀ = the
                                         capacity the cell was calibrated at)

LB track (ratified, exact for SAMPLING uncertainty): the transform is
strictly monotone in r, so quantiles commute — LB_q_i = 1−(1−LB_r_i)^(1/k₀)
with the existing Wilson/Beta bound unchanged, and the portfolio LB composes
as 1−Π(1−LB_q_i)^{k_i} (same conservative product composition as chain_lb).
TRUST WINDOW (Eval's binding correction): the MODEL is approximate
(uniform-q / independent chunks / single-hit) — trust k ∈ [k₀/2, 2k₀];
allocations are capped inside the window by construction (cap_i ≤ 2k₀ is the
caller's concern; slot capacity today is ≤ k₀ so v1 stays in-window).

SOLVER: in log space the objective is linear — maximize Σ k_i·v_i with
v_i = −ln(1−q_i) — a bounded knapsack. Solved EXACTLY by DP over the token
budget quantized at gcd(tokens_i) (5 strategies × ≤160 budget states × cap
≤10 ≈ 10⁴ ops; no greedy-by-density boundary suboptimality to document).

Execution model: contributing strategies run in PARALLEL (blend default) —
worst-case latency = MAX over contributing strategies' p50; a strategy whose
p50 exceeds the latency allowance cannot contribute (same gate as chains).
All four eligibility dimensions apply unchanged (slot_semantics, tag-gate,
crawl-gate, authority-gate). The SUPPLEMENT_ONLY gate is NOT applied here:
in a portfolio s contributes alongside corpus strategies — the tension the
blend dissolves (design doc §6). Payload is Σ k_i·tokens_i BY CONSTRUCTION
(the SUM model — retention semantics; no post-hoc gate needed).

Deployment: registered as the FOURTH allocator ("portfolio"), shadow-only at
bootstrap (absent from allocator_weights ⇒ never executed) — portfolio plans
accumulate in persisted shadow data on every query; Eval shifts traffic via
the weights file when ready (code-free cutover per design doc §3).

PHI rule (Router standard): slot ids, strategy ids, numbers only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Optional

from app.services.router.allocation import (
    AnswerSlot,
    OUTCOME_ALL_CLEARED,
    OUTCOME_NO_SLOTS,
    PAYLOAD_TOKENS_PER_CHUNK,
    RoutingLadder,
    STATUS_CLEARED,
    STATUS_NO_VIABLE,
    STATUS_OPTIONAL,
    STATUS_UNDER_CONFIDENT,
    STRATEGY_PRIORITY_ORDER,
    _PAYLOAD_TOKENS_UNKNOWN_STRATEGY,
    decision_helpers,
    decision_terminal_action,
    eligible_strategies,
    ladder_outcome,
    lookup_with_fallback,
    resolve_constraints,
    slot_helper_plan,
    slot_status,
    slot_terminal_action,
    strategy_authority_eligible,
    strategy_crawl_eligible,
    strategy_tag_eligible,
)
from app.services.router.priors import (
    PriorsBundle,
    StrategyProfile,
    compute_depth_bucket,
    load_priors,
    wilson_lower_bound,
)
from app.services.router.tracing import DecisionTrace, SlotTrace, StrategyStep

# k₀ is PER-CELL (Eval's audit, 2026-07-24): a/b/d calibrate at occupancy
# ~10 but c/s rarely exceed 1 — a global constant would misrebase c/s. Each
# StrategyProfile carries its own k0 (priors.K0_NOMINAL default; Eval's
# empirical writer emits real per-cell values alongside n). AUDIT VERDICT:
# today's cells are ALL seeds (hand-set, never measured at any capacity), so
# the transform is INERT-UNTIL-REAL-CALIBRATION — on seeds it produces
# regularized-prior bounds on designed numbers, fine as shadow diagnostics,
# not to be trusted as guarantees until cells carry (recall_lift, n, k0)
# from a real run.
from app.services.router.priors import K0_NOMINAL  # noqa: E402 (re-export for tests)

_R_CLAMP = 0.999  # r=1.0 would give q=1, ln(0) → -inf; clamp for stability


def q_from_recall(r: float, k0: int = K0_NOMINAL) -> float:
    """Per-chunk hit probability from a cell's recall_lift (ratified transform)."""
    r = max(0.0, min(_R_CLAMP, r))
    return 1.0 - (1.0 - r) ** (1.0 / k0)


def coverage(qs_ks: list[tuple[float, int]]) -> float:
    """P(covered) = 1 − Π (1−q_i)^{k_i}."""
    miss = reduce(lambda acc, qk: acc * (1.0 - qk[0]) ** qk[1], qs_ks, 1.0)
    return 1.0 - miss


@dataclass
class PortfolioPlan:
    """One slot's ex-ante allocation: {strategy: k} + both confidence tracks."""
    slot_id: str
    allocation: dict[str, int] = field(default_factory=dict)   # strategy -> k_i (>0 only)
    mean_coverage: float = 0.0     # from mean-track q
    lb_coverage: float = 0.0       # from LB-track q (THE enforced quantity)
    payload_tokens: int = 0        # Σ k_i·tokens_i — SUM model, by construction
    latency_ms: int = 0            # MAX over contributing strategies (parallel)
    budget_tokens: int = 0
    k0: int = K0_NOMINAL


def _gcd_all(values: list[int]) -> int:
    return reduce(math.gcd, values) if values else 1


def _solve_knapsack(
    candidates: list[tuple[str, float, float, int, int]],  # (sid, v_mean, v_lb, tokens, cap)
    budget_tokens: int,
) -> dict[str, int]:
    """Exact bounded knapsack: maximize Σ k_i·v_lb_i s.t. Σ k_i·tokens_i ≤ budget.

    DP over the budget quantized at gcd(token costs). Objective uses the LB
    track (the enforced quantity); mean track is recomputed on the result.
    """
    if not candidates or budget_tokens <= 0:
        return {}
    unit = _gcd_all([tokens for _, _, _, tokens, _ in candidates])
    n_units = budget_tokens // unit
    if n_units <= 0:
        return {}
    # dp[u] = (best value, allocation dict) at budget u*unit
    best_val = [0.0] * (n_units + 1)
    best_alloc: list[dict[str, int]] = [{} for _ in range(n_units + 1)]
    for sid, _v_mean, v_lb, tokens, cap in candidates:
        if v_lb <= 0.0:
            continue
        cost_units = tokens // unit
        # bounded item: iterate copies (caps are small, ≤ slot capacity)
        for _copy in range(cap):
            # standard 0/1 pass per copy (descending to avoid reuse of this copy)
            for u in range(n_units, cost_units - 1, -1):
                cand = best_val[u - cost_units] + v_lb
                if cand > best_val[u] + 1e-12:
                    alloc = dict(best_alloc[u - cost_units])
                    if alloc.get(sid, 0) < cap:
                        alloc[sid] = alloc.get(sid, 0) + 1
                        best_val[u] = cand
                        best_alloc[u] = alloc
    # best over all budget levels (monotone, but be safe)
    u_best = max(range(n_units + 1), key=lambda u: best_val[u])
    return best_alloc[u_best]


def assign_chain_fills(
    members: list[tuple[str, StrategyProfile, int]],  # (sid, profile, capacity)
    budget_tokens: int,
    floor: int = 1,
) -> dict[str, int]:
    """Per-rung fill assignment for CHAIN allocators (Ananth's partial-fill
    directive, 2026-07-24: "I'll take 3 of d" — scope a rung down to fit the
    budget rather than skip-or-nothing).

    Two passes: (1) DIVERSITY FLOOR — each member gets `floor` chunks if
    affordable, cheapest-first (the §4-ratified "≥1 from each viable
    strategy" principle applied at retrieval planning, so d is present even
    when seed values under-rank it); (2) remaining budget by VALUE via the
    exact knapsack (LB-track q per token). Fills are SEED-APPORTIONED
    planning during bootstrap — the transform's inert ruling holds: fills
    steer APPORTIONMENT only; reported chain confidence stays on the
    UNTRANSFORMED cell values until cells activate with real (r, n, k0).

    Members whose fill lands at 0 (floor unaffordable) are the caller's to
    drop from the chain, with a trace reason. Σ fills·tokens ≤ budget BY
    CONSTRUCTION — under the retention model (chain rungs union, payload is
    additive) this IS the SUM-model budget enforcement.
    """
    fills: dict[str, int] = {}
    remaining = max(0, int(budget_tokens))
    # pass 1: floor, cheapest per-chunk first (maximizes members seated)
    for sid, prof, cap in sorted(
            members, key=lambda m: PAYLOAD_TOKENS_PER_CHUNK.get(
                m[0], _PAYLOAD_TOKENS_UNKNOWN_STRATEGY)):
        tokens = PAYLOAD_TOKENS_PER_CHUNK.get(sid, _PAYLOAD_TOKENS_UNKNOWN_STRATEGY)
        seat = min(floor, cap, max(0, remaining // tokens))
        if seat > 0:
            fills[sid] = seat
            remaining -= seat * tokens
    # pass 2: remainder by value (exact knapsack over residual caps)
    candidates = []
    for sid, prof, cap in members:
        if sid not in fills:
            continue
        tokens = PAYLOAD_TOKENS_PER_CHUNK.get(sid, _PAYLOAD_TOKENS_UNKNOWN_STRATEGY)
        residual_cap = min(cap, 2 * prof.k0) - fills[sid]  # slot cap + trust window
        lb_r = wilson_lower_bound(prof.recall_lift, prof.n)
        q_lb = q_from_recall(lb_r, prof.k0)
        v_lb = -math.log(1.0 - q_lb) if q_lb < 1.0 else float("inf")
        if residual_cap > 0 and v_lb > 0:
            candidates.append((sid, v_lb, v_lb, tokens, residual_cap))
    for sid, extra in _solve_knapsack(candidates, remaining).items():
        fills[sid] = fills.get(sid, 0) + extra
    return fills


def allocate_portfolio(
    slots: list[AnswerSlot],
    pool_metadata: dict[str, dict],
    resource_posture: dict[str, Any],
    bundle: Optional[PriorsBundle] = None,
    trace: Optional[DecisionTrace] = None,
) -> RoutingLadder:
    """Portfolio allocation rendered into the RoutingLadder shape.

    Same signature as the chain allocators, so it drops into the existing
    dispatch/shadow/persist machinery unchanged. per_slot carries the
    contributing strategies (ordered by allocated k desc, then priority
    order — SCHEDULING hint, not fallback priority); per_slot_portfolio
    carries the actual {strategy: k_i}.
    """
    ladder = RoutingLadder(allocator="portfolio")
    bundle = bundle if bundle is not None else load_priors()
    trace = trace if trace is not None else DecisionTrace()
    confidence_level = float(bundle.exploration_policy.get("confidence_level", 0.95))

    c = resolve_constraints(resource_posture)
    ladder.tolerance_pct = c["tolerance_pct"]
    ladder.adjusted_confidence_bar = c["adjusted_bar"]

    trace.caller_mode = resource_posture.get("caller_mode", "")
    trace.tolerance_pct = c["tolerance_pct"]
    trace.confidence_bar = c["confidence_bar"]
    trace.adjusted_confidence_bar = c["adjusted_bar"]
    trace.speed_budget_ms = int(c["speed_budget_ms"])
    trace.latency_allowance_ms = c["latency_allowance_ms"]
    trace.priors_version = bundle.version
    trace.priors_source = bundle.source
    trace.confidence_level = confidence_level

    if not slots:
        ladder.outcome = OUTCOME_NO_SLOTS
        ladder.feasible = False
        ladder.infeasibility_reason = "no slots to allocate"
        trace.outcome = ladder.outcome
        trace.feasible = False
        trace.infeasibility_reason = ladder.infeasibility_reason
        return ladder

    budget = int(c["token_allowance_per_slot"])

    for slot in sorted(slots, key=lambda s: s.priority):
        meta = pool_metadata.get(slot.slot_id, {})
        depth = compute_depth_bucket(meta)
        st = SlotTrace(
            slot_id=slot.slot_id, priority=slot.priority,
            required=slot.required, slot_semantics=slot.slot_semantics,
            pool_size=meta.get("pool_size"),
            top_score_percentile=meta.get("top_score_percentile"),
            distinct_content_topk=meta.get("distinct_content_topk"),
            depth_bucket=depth, phase="portfolio_solve",
        )
        trace.slots.append(st)

        # ---- candidate assembly: all four eligibility gates, unchanged ----
        candidates: list[tuple[str, float, float, int, int]] = []
        profiles: dict[str, tuple[StrategyProfile, str]] = {}
        allowed = eligible_strategies(slot.slot_semantics)
        cap = max(0, int(slot.capacity))
        for sid in STRATEGY_PRIORITY_ORDER:
            if sid not in allowed:
                st.steps.append(StrategyStep(
                    sid, "skipped", skip_reason=f"ineligible_for_{slot.slot_semantics}"))
                continue
            if not strategy_tag_eligible(sid, c["j_codes"]):
                st.steps.append(StrategyStep(
                    sid, "skipped", skip_reason="tag_gated_no_payor_j_code"))
                continue
            if not strategy_crawl_eligible(sid, c["payer_crawlable"]):
                st.steps.append(StrategyStep(
                    sid, "skipped", skip_reason="crawl_gated_payer_not_crawlable"))
                continue
            prof, source = lookup_with_fallback(depth, sid, bundle)
            if prof is None:
                st.steps.append(StrategyStep(sid, "skipped", skip_reason="no_prior"))
                continue
            if not strategy_authority_eligible(sid, c["authority_requirement"],
                                               slot.required, prof.authority):
                st.steps.append(StrategyStep(
                    sid, "skipped", skip_reason="authority_gated_non_citable",
                    prior_source=source))
                continue
            if prof.recall_lift <= 0.0:
                st.steps.append(StrategyStep(
                    sid, "skipped", skip_reason="zero_or_negative_recall_lift",
                    prior_source=source, recall_lift=prof.recall_lift))
                continue
            if prof.latency_p50_ms > c["latency_allowance_ms"]:
                # parallel model: a contributor's OWN p50 must fit the allowance
                st.steps.append(StrategyStep(
                    sid, "skipped", skip_reason="over_latency_allowance",
                    prior_source=source, recall_lift=prof.recall_lift,
                    latency_p50_ms=prof.latency_p50_ms))
                continue
            tokens = PAYLOAD_TOKENS_PER_CHUNK.get(sid, _PAYLOAD_TOKENS_UNKNOWN_STRATEGY)
            q_mean = q_from_recall(prof.recall_lift, prof.k0)  # per-cell k₀
            lb_r = wilson_lower_bound(prof.recall_lift, prof.n, confidence_level)
            q_lb = q_from_recall(lb_r, prof.k0)
            v_mean = -math.log(1.0 - q_mean)
            v_lb = -math.log(1.0 - q_lb) if q_lb < 1.0 else float("inf")
            # TRUST-WINDOW CAP, literally by construction (Eval 2026-07-24):
            # the transform is trusted only for k ≤ 2·k₀ — cap each
            # strategy's k there so a low-k₀ cell (c/s empirical ≈1) can
            # never be allocated 10× outside the window. No-op for k₀=10
            # cells (2k₀=20 > slot capacity today).
            eff_cap = min(cap, 2 * prof.k0)
            candidates.append((sid, v_mean, v_lb, tokens, eff_cap))
            profiles[sid] = (prof, source)

        # optional slots: cheapest single chunk only (Eval's optional-cheap
        # ruling carries over — telemetry-only slots never spend real budget)
        if not slot.required and candidates:
            cheapest = min(candidates, key=lambda cand: (cand[3], -cand[2]))
            alloc = {cheapest[0]: 1}
        else:
            alloc = _solve_knapsack(candidates, budget)

        # ---- render the slot result ----
        qk_lb = []
        qk_mean = []
        payload = 0
        latency = 0
        for sid, k in alloc.items():
            prof, source = profiles[sid]
            q_mean = q_from_recall(prof.recall_lift, prof.k0)
            lb_r = wilson_lower_bound(prof.recall_lift, prof.n, confidence_level)
            qk_mean.append((q_mean, k))
            qk_lb.append((q_from_recall(lb_r, prof.k0), k))
            payload += k * PAYLOAD_TOKENS_PER_CHUNK.get(sid, _PAYLOAD_TOKENS_UNKNOWN_STRATEGY)
            latency = max(latency, prof.latency_p50_ms)
            st.steps.append(StrategyStep(
                sid, "added", prior_source=source,
                recall_lift=prof.recall_lift, n=prof.n,
                lb_lift=q_from_recall(lb_r, prof.k0),  # per-chunk LB q (portfolio semantics)
                latency_p50_ms=prof.latency_p50_ms,
                cost=prof.cost, accuracy_estimate=prof.accuracy_estimate))
            ladder.total_estimated_cost += prof.cost  # cost per contributing strategy

        mean_cov = coverage(qk_mean)
        lb_cov = coverage(qk_lb)
        # scheduling-hint order: k desc, then static priority (NOT fallback priority)
        order = sorted(alloc, key=lambda sid: (-alloc[sid],
                                               STRATEGY_PRIORITY_ORDER.index(sid)))
        status = slot_status(order, lb_cov, c["adjusted_bar"], required=slot.required)

        sid_ = slot.slot_id
        ladder.per_slot[sid_] = order
        ladder.per_slot_portfolio[sid_] = dict(alloc)
        ladder.per_slot_confidence[sid_] = mean_cov
        ladder.per_slot_lb[sid_] = lb_cov
        ladder.per_slot_status[sid_] = status
        ladder.per_slot_terminal[sid_] = slot_terminal_action(status)
        ladder.per_slot_helpers[sid_] = slot_helper_plan(status, c["j_codes"], c["d_codes"])
        ladder.per_slot_accuracy[sid_] = (
            sum(profiles[s][0].accuracy_estimate * k for s, k in alloc.items())
            / max(1, sum(alloc.values()))
        ) if alloc else 0.0
        ladder.per_slot_latency_ms[sid_] = latency
        ladder.per_slot_payload_tokens[sid_] = payload

        st.payload_tokens_worst_case = payload
        st.final_chain = order
        st.final_confidence = mean_cov
        st.final_lb = lb_cov
        st.final_latency_ms = latency
        st.bar_cleared = lb_cov >= c["adjusted_bar"]
        st.status = status
        st.terminal_action = ladder.per_slot_terminal[sid_]
        st.stop_reason = "portfolio_solved" if alloc else "no_viable_candidates"

    ladder.total_estimated_ms = max(ladder.per_slot_latency_ms.values(), default=0)
    ladder.total_payload_tokens = sum(ladder.per_slot_payload_tokens.values())
    n = max(1, len(ladder.per_slot))
    ladder.aggregate_confidence_estimate = sum(ladder.per_slot_confidence.values()) / n
    ladder.aggregate_accuracy_estimate = sum(ladder.per_slot_accuracy.values()) / n

    parts = " + ".join(f"{v:.4f}" for v in ladder.per_slot_lb.values())
    trace.aggregate_confidence = ladder.aggregate_confidence_estimate
    trace.aggregate_arithmetic = (
        f"per-slot portfolio LBs [{parts}] vs bar {c['adjusted_bar']:.4f} "
        f"(coverage = 1−Π(1−q)^k; mean is telemetry only)"
    )

    ladder.outcome = ladder_outcome(ladder.per_slot_status)
    ladder.terminal_action = decision_terminal_action(ladder.per_slot_terminal)
    ladder.helpers = decision_helpers(ladder.per_slot_helpers)
    ladder.feasible = ladder.outcome == OUTCOME_ALL_CLEARED
    if not ladder.feasible:
        failed = {s: st for s, st in ladder.per_slot_status.items()
                  if st in (STATUS_UNDER_CONFIDENT, STATUS_NO_VIABLE)}
        if failed:
            ladder.infeasibility_reason = (
                f"required slots below LB bar {c['adjusted_bar']:.3f} "
                f"at level {confidence_level:.2f}: {failed}"
            )
    trace.outcome = ladder.outcome
    trace.feasible = ladder.feasible
    trace.infeasibility_reason = ladder.infeasibility_reason
    return ladder
