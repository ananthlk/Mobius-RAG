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

# SAME-STRATEGY DIMINISHING RETURNS (2026-08-07, Ananth's directive after
# reviewing the portfolio-vs-greedy/bayesian bank comparison: "is 1st chunk
# from every strategy better than 3 from a and 2 from c" -- the model's
# original flat-value-per-copy assumption said no, since it valued every
# chunk of a strategy identically regardless of how many it had already
# taken. PROVISIONAL constant, not yet Eval-calibrated against a real
# empirical recall@k curve (that data exists in the forced single-arm sweep
# rows from tonight's greedy validation but hasn't been mined into a decay
# shape yet -- this is a placeholder pending that analysis, chosen to be
# directionally correct rather than precisely fit). Each additional chunk
# from the SAME strategy is modeled as DECAY_SAME_STRATEGY× the per-chunk
# hit probability of the previous one (q_copy_n = q_i * decay^(n-1)) --
# chunk 1 unaffected, chunk 2 discounted, etc. This makes a FRESH strategy's
# first chunk compete fairly against a THIRD chunk of an already-included
# strategy, instead of the old model where every copy looked identical.
DECAY_SAME_STRATEGY = 0.7

# COST-GATED STRATEGIES (2026-08-07, Ananth's directive: "stop c or have
# extra penalties and use it if and only when we really need it"). These
# strategies carry a REAL marginal dollar cost per attempt (an LLM call,
# unlike a/b/d/s which are ~free) that priors.cost_per_attempt has never
# actually been populated with (confirmed: every seed cell hardcodes
# cost_per_attempt=0, including c's -- so the knapsack has been treating a
# genuinely costly strategy as free). Rather than guess at a real dollar
# figure and fold it into the value function as a soft penalty (which still
# lets `c` win purely on token-efficiency grounds, exactly what happened in
# tonight's bank run), this is a hard GATE: allocate_portfolio first solves
# using only the FREE strategies, and only lets a cost-gated strategy
# compete for the leftover/full budget if the free-only allocation doesn't
# clear the slot's confidence bar on its own. "Use only when really needed"
# implemented literally, not as a tunable weight.
_COST_GATED_STRATEGIES = frozenset({"c"})

# PER-STRATEGY TURN FLOOR (2026-08-07, Ananth's directives, in sequence:
# "let's restrict c until the 3rd turn... if we get to rag for the third
# time lets allow c", then "lets d in on r2 and c on r3"). A HARD exclusion
# from the candidate pool entirely below this turn -- not a soft penalty.
# Rationale is DIFFERENT per strategy even though the mechanism is shared:
#   c: turn 3 -- c is the one strategy with a real (if currently
#      unmeasured) marginal DOLLAR cost per call. On top of this turn
#      floor, c ALSO stays subject to the separate coverage-bar gate below
#      once turn-unlocked (being turn 3+ doesn't force c in, just allows it
#      to compete if the free strategies genuinely aren't enough).
#   d: turn 2 -- d has ~zero dollar cost but real high LATENCY (p50
#      ~9.7s, live-verified vastly exceeding chat.default's ~2.3s
#      allowance). Turn-unlocking d is a deliberate override of the
#      standalone latency gate below (see its call site) -- by turn 2 the
#      caller is already committing to another round-trip regardless, so
#      paying d's real latency is an accepted, conscious tradeoff, not
#      something the same rigid real_time allowance should keep blocking.
#      No coverage-bar condition for d (unlike c) -- turn 2 unlocks it
#      outright, per Ananth's ask.
_STRATEGY_MIN_TURN: dict[str, int] = {"d": 2, "c": 3}

# COST-GATE FORCE TURN (2026-08-07, Ananth: "we will never use c in
# thinking, that feels off... could we force a bit of c in turn 3 of
# thinking too"). Real gap: under chat.thinking's bigger latency/token
# budget, the FREE strategies alone (s/a/b/d) already clear the confidence
# bar every time -- so c's "only if genuinely needed" bar-gate (just above)
# correctly never opens, meaning c NEVER gets exercised under thinking
# mode at all, turn 3 included. Turn 3 is meant to be a real escalation
# checkpoint, not just "eligible but still gated the same as turn 1-2 would
# be if the bypass applied there." At this turn, c competes in the FULL
# knapsack unconditionally -- no free-only bar-check first. This is exactly
# what chat.default's turn 3 already does DE FACTO today (free-only s/a/b
# there happens to fall short of the bar every time, so the gate always
# opens) -- making it an EXPLICIT, deliberate rule here makes that
# consistent for thinking mode too, instead of accidentally depending on
# whether the free strategies happen to be enough.
_COST_GATE_FORCE_TURN = 3

# TOKEN-COST RANKING OVERRIDE (2026-08-07, Ananth's directive: "the token
# preference is too punitive... let's make the penalty 0% and see what the
# raw answer is, then optimize"). PAYLOAD_TOKENS_PER_CHUNK's real d=500
# (vs a/b/c=250) is a MEASURED value (allocation.py's own comment: "a/b/c/d
# now measured") -- web content genuinely comes back larger, so this is NOT
# a modeling penalty to just delete; other consumers of that constant (real
# context-window budget accounting) need the true number. This override is
# scoped to portfolio's OWN knapsack RANKING decision only: for VALUE/TOKEN
# comparison purposes, treat d's per-chunk cost as if it were the same as
# a/b/c (250, i.e. 0% extra penalty) so d can compete for selection purely
# on its recall_lift merits. The REAL token cost (PAYLOAD_TOKENS_PER_CHUNK's
# 500) still governs how much of the actual budget gets consumed once d is
# selected -- this only changes which candidate LOOKS most valuable per
# token during the solve, not what gets charged against the budget
# afterward. Experimental: empty dict = no override (real costs everywhere,
# the pre-2026-08-07 behavior); set d's override key to try other levels
# between 250 (0% penalty, current) and 500 (full real penalty) once we've
# seen the raw, unconstrained answer this call is asking for.
_TOKEN_COST_RANKING_OVERRIDE: dict[str, int] = {"d": 250}


def _portfolio_tokens(sid: str) -> int:
    """PAYLOAD_TOKENS_PER_CHUNK, with _TOKEN_COST_RANKING_OVERRIDE applied --
    used ONLY inside allocate_portfolio (not assign_chain_fills, which stays
    on the real, unoverridden constant -- chain-allocator fill-scoping is
    out of scope for this directive)."""
    return _TOKEN_COST_RANKING_OVERRIDE.get(
        sid, PAYLOAD_TOKENS_PER_CHUNK.get(sid, _PAYLOAD_TOKENS_UNKNOWN_STRATEGY))


def q_from_recall(r: float, k0: int = K0_NOMINAL) -> float:
    """Per-chunk hit probability from a cell's recall_lift (ratified transform)."""
    r = max(0.0, min(_R_CLAMP, r))
    return 1.0 - (1.0 - r) ** (1.0 / k0)


def coverage(qs_ks: list[tuple[float, int]]) -> float:
    """P(covered) = 1 − Π (1−q_i)^{k_i}."""
    miss = reduce(lambda acc, qk: acc * (1.0 - qk[0]) ** qk[1], qs_ks, 1.0)
    return 1.0 - miss


def decayed_coverage(qs_ks: list[tuple[float, int]], decay: float = 1.0) -> float:
    """P(covered) under the SAME-STRATEGY diminishing-returns model: the
    n-th chunk (0-indexed) from a strategy contributes q_i*decay**n instead
    of the flat q_i every copy used in `coverage()` above. decay=1.0
    reduces to exactly `coverage()` (verified equal in tests). Must be used
    consistently with whatever decay `_solve_knapsack` was called with —
    reporting flat coverage() on a decayed allocation overstates it."""
    miss = 1.0
    for q, k in qs_ks:
        for n in range(k):
            miss *= 1.0 - min(_R_CLAMP, q * (decay ** n))
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
    *,
    decay: float = 1.0,
) -> dict[str, int]:
    """Exact bounded knapsack: maximize Σ k_i·v_lb_i s.t. Σ k_i·tokens_i ≤ budget.

    DP over the budget quantized at gcd(token costs). Objective uses the LB
    track (the enforced quantity); mean track is recomputed on the result.

    `decay` (2026-08-07, Ananth's directive, DECAY_SAME_STRATEGY at the
    default call site): when < 1.0, the n-th copy (0-indexed) of a strategy
    is worth v_lb converted back from a decayed q (q * decay**n) instead of
    the flat v_lb every previous copy used -- makes repeated chunks from the
    SAME strategy worth progressively less, so a fresh strategy's first
    chunk can compete fairly against a third/fourth chunk of an
    already-included one instead of the old flat-value model where every
    copy looked identical. Default 1.0 (no decay) preserves the exact prior
    behavior for assign_chain_fills' call site below, which this parameter
    intentionally does NOT change -- chain-allocator fill-scoping is a
    separate, already-ratified mechanism, not in scope for this directive.
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
        # Recover q_lb from v_lb (v_lb = -ln(1-q_lb)) so each copy's decayed
        # q can be converted back to a per-copy value -- keeps the decayed
        # allocation's REPORTED coverage mathematically consistent with
        # 1-Π(1-q_copy), not just a value-space fudge (see the render loop's
        # matching decay application below for the same reason).
        q_lb = 1.0 - math.exp(-v_lb) if v_lb < float("inf") else 1.0
        # bounded item: iterate copies (caps are small, ≤ slot capacity)
        for _copy in range(cap):
            q_copy = min(_R_CLAMP, q_lb * (decay ** _copy))
            v_copy = -math.log(1.0 - q_copy) if q_copy < 1.0 else float("inf")
            if v_copy <= 0.0:
                break  # decayed to worthless; higher copies only decay further
            # standard 0/1 pass per copy (descending to avoid reuse of this copy)
            for u in range(n_units, cost_units - 1, -1):
                cand = best_val[u - cost_units] + v_copy
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
            _min_turn = _STRATEGY_MIN_TURN.get(sid)
            # REVERTED the THINKING+ANY BYPASS (2026-08-19, Ananth, live-trace
            # latency work: "hold c/d for 1st round anyways ... will make 1st
            # round fast"). The bypass (2026-08-07) let chat.thinking +
            # authority=any skip the turn floor entirely, so d -- priced at a
            # 3000ms p50 prior, the slowest strategy in the roster -- could
            # land on round 1 even though round 1 is exactly the round every
            # caller is waiting on synchronously. The floor is now absolute:
            # d/c never contribute before their _STRATEGY_MIN_TURN round,
            # regardless of caller_mode or authority_requirement. A caller
            # that wants d's recall lift still gets it -- just starting round
            # 2, once round 1's a/b/s answer is already in hand.
            _turn_unlocked = _min_turn is None or c["call_number"] >= _min_turn
            if _min_turn is not None and not _turn_unlocked:
                st.steps.append(StrategyStep(
                    sid, "skipped",
                    skip_reason=f"turn_gated (call_number={c['call_number']} < "
                                f"{_min_turn})"))
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
            # Turn-unlocking a strategy in _STRATEGY_MIN_TURN (d, specifically)
            # IS the deliberate override of this latency gate -- by the turn
            # it unlocks at, the caller has already accepted a slower
            # round-trip. Strategies with no turn floor (a/b/s) always go
            # through this check normally; it's never silently skipped for
            # them.
            _latency_gate_applies = sid not in _STRATEGY_MIN_TURN
            if _latency_gate_applies and prof.latency_p50_ms > c["latency_allowance_ms"]:
                # parallel model: a contributor's OWN p50 must fit the allowance
                st.steps.append(StrategyStep(
                    sid, "skipped", skip_reason="over_latency_allowance",
                    prior_source=source, recall_lift=prof.recall_lift,
                    latency_p50_ms=prof.latency_p50_ms))
                continue
            tokens = _portfolio_tokens(sid)
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
        cost_gate_used = False
        if not slot.required and candidates:
            cheapest = min(candidates, key=lambda cand: (cand[3], -cand[2]))
            alloc = {cheapest[0]: 1}
        else:
            # COST GATE (Ananth's directive): solve free-only first; only
            # let cost-gated strategies (c) compete if free-only can't clear
            # this slot's own bar. "Use c if and only when we really need
            # it" implemented as a hard gate, not a value-space penalty that
            # c could still out-token-efficiency its way past.
            free_candidates = [cand for cand in candidates if cand[0] not in _COST_GATED_STRATEGIES]
            gated_present = len(free_candidates) < len(candidates)
            _force_cost_gate_open = c["call_number"] >= _COST_GATE_FORCE_TURN
            if gated_present and _force_cost_gate_open:
                # Turn 3+: c competes for real, no "only if needed" check --
                # see _COST_GATE_FORCE_TURN's docstring.
                alloc = _solve_knapsack(candidates, budget, decay=DECAY_SAME_STRATEGY)
                cost_gate_used = True
            else:
                alloc = _solve_knapsack(free_candidates, budget, decay=DECAY_SAME_STRATEGY)
            if gated_present and not _force_cost_gate_open:
                free_qk = [
                    (q_from_recall(
                        wilson_lower_bound(profiles[sid][0].recall_lift, profiles[sid][0].n, confidence_level),
                        profiles[sid][0].k0), k)
                    for sid, k in alloc.items()
                ]
                # GATE CHECK uses FLAT (undecayed) coverage deliberately —
                # `adjusted_bar` was calibrated against the flat model (every
                # allocator up to now, and this same function's own FINAL
                # reported lb_cov below, compares against it that way).
                # Checking the gate against DECAYED coverage would silently
                # stack two conservatisms (decay lowers the estimate, then
                # the lowered estimate gets compared to a bar that assumes
                # it wasn't lowered) -- confirmed live: with the decayed
                # check, c triggered on 4/4 real test queries, defeating
                # "use c only when really needed." Flat coverage for the
                # GATE DECISION only; the final rendered lb_cov (below) still
                # uses decay, since that's an honest accounting of what the
                # chosen allocation actually delivers, not a threshold check.
                free_lb_cov = coverage(free_qk)
                if free_lb_cov < c["adjusted_bar"]:
                    alloc = _solve_knapsack(candidates, budget, decay=DECAY_SAME_STRATEGY)
                    cost_gate_used = True
                else:
                    for cand in candidates:
                        if cand[0] in _COST_GATED_STRATEGIES:
                            st.steps.append(StrategyStep(
                                cand[0], "skipped",
                                skip_reason=f"cost_gate_not_needed (free-only coverage "
                                            f"{free_lb_cov:.3f} already clears bar {c['adjusted_bar']:.3f})"))

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
            payload += k * _portfolio_tokens(sid)
            latency = max(latency, prof.latency_p50_ms)
            st.steps.append(StrategyStep(
                sid, "added", prior_source=source,
                recall_lift=prof.recall_lift, n=prof.n,
                lb_lift=q_from_recall(lb_r, prof.k0),  # per-chunk LB q (portfolio semantics)
                latency_p50_ms=prof.latency_p50_ms,
                cost=prof.cost, accuracy_estimate=prof.accuracy_estimate))
            ladder.total_estimated_cost += prof.cost  # cost per contributing strategy

        # decayed_coverage, not coverage() -- must match whatever decay
        # _solve_knapsack used, or this overstates the real allocation's
        # coverage (a single-chunk optional-slot alloc is unaffected: decay
        # only bites at k>1). See DECAY_SAME_STRATEGY's docstring.
        mean_cov = decayed_coverage(qk_mean, DECAY_SAME_STRATEGY)
        lb_cov = decayed_coverage(qk_lb, DECAY_SAME_STRATEGY)
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
