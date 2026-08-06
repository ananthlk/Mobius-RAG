"""OPTIMIZER allocator — the real constrained-optimization solve (spec §1).

One of Router's two production allocators (2026-07-23 addendum, dual-build):
  greedy    — allocation.py: deterministic sequential-fallback heuristic
  optimizer — THIS module: exact joint solve over all N slots

Both are computed for EVERY production query; an A/B split picks which ladder
actually executes — the other is logged as a shadow plan (router.py). This
replaces the rejected parallel-execution exploration idea: no extra retrieval
attempts, just two allocation strategies compared on identical inputs.

THE SOLVE
  Decision variable per slot: which SUBSET of viable strategies forms its
  fallback chain (chain success probability is commutative, so a chain is a
  subset; execution order is fixed cheap/fast-first for expected latency).

  Per-slot feasibility: |S| <= max_attempts and sum(latency) <= per-slot
  allowance (parallel-slot model — same constraint greedy uses). Every slot
  with at least one viable strategy must receive a non-empty chain (Structure
  created the slot because part of the query needs answering — the optimizer
  may not silently drop it to save cost).

  Objective (lexicographic — v3, per §2a addendum + uncertainty co-design):
    1. MAXIMIZE the slot's LOWER-BOUND confidence (Wilson LB composition over
       the subset at the policy confidence_level) within the per-slot time
       allowance and max_attempts. The LB is the enforced §2a quantity — two
       equal-mean options with different spread stop tying: tighter wins.
    2. tie-break: maximize MEAN confidence ("spend the budget you have" — a
       rung whose LB rounds to zero but raises the mean is still worth taking
       when nothing better fits; preserves the cmhc013 fix).
    3. tie-break: minimize cost, then max slot latency.
  The confidence bar is a per-slot FEASIBILITY FLOOR (drives status/outcome),
  never a stopping target.
  (History: v1 minimized cost among bar-clearing options — satisficer bug,
  cmhc013, picked {a} .80 over {s,a} .92 to save 100ms nobody needed. v2
  maximized the mean. v3 maximizes the LB per the uncertainty workstream.)

  Method: per slot, enumerate all subsets of viable strategies (<= 2^6 = 64),
  filter by attempts/latency, Pareto-prune, then select per slot the maximum-
  confidence option (ties broken by cost, then latency). Because the objective
  (sum of slot confidences) is separable and all constraints are per-slot, the
  joint solve factorizes into exact per-slot argmax — no approximation. The
  _solve() boundary is kept so a future GLOBAL constraint (e.g. a query-wide
  cost cap) can restore a cross-slot DP without touching callers.

WHERE OPTIMIZER DIFFERS FROM GREEDY
  Greedy stops at the bar (satisficer by design — cheap, known-safe). The
  optimizer fills the remaining time/attempt budget with the best available
  contingency rungs, maximizing the chance the slot resolves within the same
  wall-clock allowance. Extra rungs are contingency, not guaranteed spend —
  Fillers walks the chain only until Observer clears the bar at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

from app.services.router.allocation import (
    AnswerSlot,
    AUTHORITY_ANY,
    RoutingLadder,
    STRATEGY_PRIORITY_ORDER,
    chain_expected_accuracy,
    chain_lb,
    chain_payload_tokens,
    chain_success_probability,
    decision_helpers,
    DEFAULT_TOKEN_ALLOWANCE_PER_SLOT,
    PAYLOAD_TOKENS_PER_CHUNK,
    _PAYLOAD_TOKENS_UNKNOWN_STRATEGY as _PAYLOAD_TOKENS_UNKNOWN,
    decision_terminal_action,
    eligible_strategies,
    ladder_outcome,
    lookup_with_fallback,
    resolve_constraints,
    rung_payload_tokens,
    slot_helper_plan,
    warn_if_estimate_driven_skip,
    slot_status,
    slot_terminal_action,
    SUPPLEMENT_ONLY_STRATEGIES,
    strategy_authority_eligible,
    strategy_crawl_eligible,
    strategy_tag_eligible,
    OUTCOME_ALL_CLEARED,
    OUTCOME_NO_SLOTS,
    STATUS_CLEARED,
    STATUS_NO_VIABLE,
    STATUS_UNDER_CONFIDENT,
)
from app.services.router.priors import (
    PriorsBundle,
    StrategyProfile,
    compute_depth_bucket,
    load_priors,
    wilson_lower_bound,
)
from app.services.router.tracing import DecisionTrace, SlotTrace, StrategyStep

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Option:
    """One candidate chain (subset) for a slot."""
    chain: tuple[str, ...]           # in execution order (cheap/fast-first)
    confidence: float                # MEAN chain confidence (telemetry / tie-break 2)
    lb: float                        # LOWER-BOUND chain confidence (primary objective)
    cost: float
    latency_ms: int                  # sequential sum (worst case)


def _viable_strategies(
    slot: AnswerSlot, depth_bucket: int, latency_allowance_ms: float,
    bundle: PriorsBundle, st: SlotTrace, j_codes: list,
    payer_crawlable=None,
    token_allowance: int = DEFAULT_TOKEN_ALLOWANCE_PER_SLOT,
    authority_requirement: str = AUTHORITY_ANY,
) -> list[tuple[str, StrategyProfile, str]]:
    allowed = eligible_strategies(slot.slot_semantics)
    out = []
    for sid in STRATEGY_PRIORITY_ORDER:
        if sid not in allowed:
            st.steps.append(StrategyStep(
                sid, "skipped",
                skip_reason=f"ineligible_for_{slot.slot_semantics}"))
            continue
        if not strategy_tag_eligible(sid, j_codes):
            st.steps.append(StrategyStep(
                sid, "skipped", skip_reason="tag_gated_no_payor_j_code"))
            continue
        if not strategy_crawl_eligible(sid, payer_crawlable):
            st.steps.append(StrategyStep(
                sid, "skipped", skip_reason="crawl_gated_payer_not_crawlable"))
            continue
        prof, source = lookup_with_fallback(depth_bucket, sid, bundle)
        if prof is None:
            st.steps.append(StrategyStep(sid, "skipped", skip_reason="no_prior"))
            continue
        if not strategy_authority_eligible(sid, authority_requirement, slot.required,
                                           prof.authority):
            st.steps.append(StrategyStep(
                sid, "skipped", skip_reason="authority_gated_non_citable",
                prior_source=source))
            continue
        if prof.recall_lift <= 0.0:
            st.steps.append(StrategyStep(sid, "skipped",
                                         skip_reason="zero_or_negative_recall_lift",
                                         prior_source=source, recall_lift=prof.recall_lift))
            continue
        if prof.latency_p50_ms > latency_allowance_ms:
            st.steps.append(StrategyStep(sid, "skipped",
                                         skip_reason="over_latency_allowance",
                                         prior_source=source, recall_lift=prof.recall_lift,
                                         latency_p50_ms=prof.latency_p50_ms))
            continue
        # PARTIAL-FILL model (Ananth 2026-07-24): viable if ONE chunk is
        # affordable; actual fills assigned jointly post-solve (SUM model —
        # retention live). Old all-or-nothing gate = d-starvation cause #5.
        if PAYLOAD_TOKENS_PER_CHUNK.get(sid, _PAYLOAD_TOKENS_UNKNOWN) > token_allowance:
            warn_if_estimate_driven_skip(sid, slot.capacity, token_allowance)
            st.steps.append(StrategyStep(sid, "skipped",
                                         skip_reason="payload_over_token_allowance",
                                         prior_source=source, recall_lift=prof.recall_lift))
            continue
        out.append((sid, prof, source))
    return out


def _enumerate_options(
    viable: list[tuple[str, StrategyProfile, str]],
    max_attempts: int, latency_allowance_ms: float,
    confidence_level: float,
    lb_fn=wilson_lower_bound,
) -> list[_Option]:
    """All feasible non-empty subsets with both confidence tracks computed."""
    order = {sid: i for i, sid in enumerate(STRATEGY_PRIORITY_ORDER)}
    options: list[_Option] = []
    max_size = min(len(viable), max_attempts)
    for size in range(1, max_size + 1):
        for combo in combinations(viable, size):
            latency = sum(p.latency_p50_ms for _, p, _ in combo)
            if latency > latency_allowance_ms:
                continue
            chain = tuple(sorted((sid for sid, _, _ in combo), key=order.__getitem__))
            options.append(_Option(
                chain=chain,
                confidence=chain_success_probability([p.recall_lift for _, p, _ in combo]),
                lb=chain_lb([p for _, p, _ in combo], confidence_level, lb_fn=lb_fn),
                cost=sum(p.cost for _, p, _ in combo),
                latency_ms=latency,
            ))
    return options


def _solve(per_slot_options: list[list[_Option]],
           required_flags: list[bool]) -> list[_Option]:
    """Per-slot selection with §2a semantics:

    REQUIRED slots — LB MAXIMIZER (v3): maximize the lower-bound chain
    confidence; ties broken by higher mean (spend the budget), then lower
    cost, then lower latency.

    OPTIONAL slots — CHEAPEST-FIRST (Eval's ruling 2026-07-23): (cost, latency)
    ascending, LB as final tie-break. A telemetry-only slot must never pay
    wall-clock for a confidence number nobody gates on — best-LB picks were
    making the optional slot the query's latency max (e.g. 'd' @3000ms vs a
    2100ms required chain). Both allocators now agree optional = cheap, so the
    A/B compares only the required-slot logic, which is what matters.

    Separable objective + per-slot constraints => per-slot argmax IS the exact
    joint optimum. Per-slot feasibility (LB vs bar) is judged by the caller."""
    choices: list[_Option] = []
    for options, required in zip(per_slot_options, required_flags):
        if required:
            choices.append(max(
                options,
                key=lambda o: (round(o.lb, 6), round(o.confidence, 6), -o.cost, -o.latency_ms),
            ))
        else:
            choices.append(min(
                options,
                key=lambda o: (o.cost, o.latency_ms, -round(o.lb, 6)),
            ))
    return choices


def _record_chain_steps(st: SlotTrace, option: _Option,
                        profiles: dict[str, tuple[StrategyProfile, str]],
                        confidence_level: float,
                        lb_fn=wilson_lower_bound) -> None:
    conf = 0.0
    lb = 0.0
    lat = 0
    for sid in option.chain:
        prof, source = profiles[sid]
        lb_lift = lb_fn(prof.recall_lift, prof.n, confidence_level)
        conf_before, lb_before, lat_before = conf, lb, lat
        conf = conf + (1 - conf) * prof.recall_lift
        lb = lb + (1 - lb) * lb_lift
        lat += prof.latency_p50_ms
        st.steps.append(StrategyStep(
            strategy_id=sid, action="added", prior_source=source,
            recall_lift=prof.recall_lift, n=prof.n, lb_lift=lb_lift,
            latency_p50_ms=prof.latency_p50_ms,
            cost=prof.cost, accuracy_estimate=prof.accuracy_estimate,
            conf_before=conf_before, added_term=(1 - conf_before) * prof.recall_lift,
            conf_after=conf, lb_before=lb_before, lb_after=lb,
            latency_before_ms=lat_before, latency_after_ms=lat,
        ))


def optimize_allocation(
    slots: list[AnswerSlot],
    pool_metadata: dict[str, dict],
    resource_posture: dict[str, Any],
    bundle: Optional[PriorsBundle] = None,
    trace: Optional[DecisionTrace] = None,
    lb_fn=wilson_lower_bound,
    allocator_name: str = "optimizer",
) -> RoutingLadder:
    """Exact per-slot LB-maximizing allocation (see module docstring).

    `lb_fn` is the per-strategy lower-bound function: Wilson by default;
    bayesian_optimizer.py passes beta_lower_bound — the SOLE difference
    between the two optimizer allocators, by design (isolates the bound
    comparison in the A/B/C data)."""
    ladder = RoutingLadder(allocator=allocator_name)
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

    ordered = sorted(slots, key=lambda s: s.priority)
    slot_traces: list[SlotTrace] = []
    per_slot_options: list[list[_Option]] = []
    per_slot_profiles: list[dict[str, tuple[StrategyProfile, str]]] = []
    option_stats = []

    for slot in ordered:
        meta = pool_metadata.get(slot.slot_id, {})
        depth = compute_depth_bucket(meta)
        st = SlotTrace(
            slot_id=slot.slot_id, priority=slot.priority,
            required=slot.required,
            slot_semantics=slot.slot_semantics,
            pool_size=meta.get("pool_size"),
            top_score_percentile=meta.get("top_score_percentile"),
            distinct_content_topk=meta.get("distinct_content_topk"),
            depth_bucket=depth, phase="optimizer_solve",
        )
        slot_traces.append(st)
        trace.slots.append(st)

        viable = _viable_strategies(slot, depth, c["latency_allowance_ms"], bundle, st,
                                    c["j_codes"], c["payer_crawlable"],
                                    token_allowance=c["token_allowance_per_slot"],
                                    authority_requirement=c["authority_requirement"])
        profiles = {sid: (prof, source) for sid, prof, source in viable}
        per_slot_profiles.append(profiles)
        # Optional slots: single cheap attempt, not a bar-chase (see allocation.py
        # OPTIONAL_SLOT_MAX_ATTEMPTS rationale) — enumerate size-1 subsets only.
        # QUERY-level attempts budget (real AnswerSlot has no per-slot field)
        effective_attempts = int(c["max_attempts"]) if slot.required else 1
        options = _enumerate_options(viable, effective_attempts,
                                     c["latency_allowance_ms"], confidence_level,
                                     lb_fn=lb_fn)
        # SUPPLEMENT GATE (shared rule, see allocation.SUPPLEMENT_ONLY_STRATEGIES):
        # a required slot may not be served by a sole supplement-only rung
        # (e.g. chain ("s",)) while any other option exists — the optimizer's
        # cost tie-break otherwise hands one-shot ladders to the free bare-
        # fact strategy at tied LBs, same collapse as greedy's.
        if slot.required:
            non_sole_supplement = [
                o for o in options
                if not (len(o.chain) == 1 and o.chain[0] in SUPPLEMENT_ONLY_STRATEGIES)
            ]
            if non_sole_supplement and len(non_sole_supplement) < len(options):
                st.steps.append(StrategyStep(
                    next(iter(SUPPLEMENT_ONLY_STRATEGIES)), "skipped",
                    skip_reason="supplement_only_not_sole_rung"))
                options = non_sole_supplement
        option_stats.append((len(viable), len(options)))
        if not options:
            # slot has nothing viable: explicit empty placeholder -> NO_VIABLE_STRATEGY
            options = [_Option(chain=(), confidence=0.0, lb=0.0, cost=0.0, latency_ms=0)]
        per_slot_options.append(options)

    choices = _solve(per_slot_options, [s.required for s in ordered])

    trace.mode_reason = (trace.mode_reason + " | " if trace.mode_reason else "") + (
        "solver: " + ", ".join(
            f"{s.slot_id}: {nv} viable → {no} subset options"
            for s, (nv, no) in zip(ordered, option_stats)
        )
    )

    slot_accs = []
    for slot, st, option, profiles in zip(ordered, slot_traces, choices, per_slot_profiles):
        _record_chain_steps(st, option, profiles, confidence_level, lb_fn=lb_fn)
        chain = list(option.chain)
        chain_profiles = [profiles[sid][0] for sid in chain]
        acc = chain_expected_accuracy(chain_profiles)
        status = slot_status(chain, option.lb, c["adjusted_bar"], required=slot.required)

        ladder.per_slot[slot.slot_id] = chain
        ladder.per_slot_confidence[slot.slot_id] = option.confidence
        ladder.per_slot_lb[slot.slot_id] = option.lb
        ladder.per_slot_status[slot.slot_id] = status
        ladder.per_slot_terminal[slot.slot_id] = slot_terminal_action(status)
        ladder.per_slot_helpers[slot.slot_id] = slot_helper_plan(status, c["j_codes"], c["d_codes"])
        ladder.per_slot_accuracy[slot.slot_id] = acc
        ladder.per_slot_latency_ms[slot.slot_id] = option.latency_ms
        # PARTIAL-FILL assignment on the chosen subset (SUM model)
        from app.services.router.portfolio import assign_chain_fills
        members = [(s_, profiles[s_][0], slot.capacity) for s_ in chain]
        fills = assign_chain_fills(members, c["token_allowance_per_slot"])
        kept = [s_ for s_ in chain if fills.get(s_, 0) >= 1]
        if kept != chain:
            for s_ in chain:
                if s_ not in kept:
                    st.steps.append(StrategyStep(
                        s_, "skipped",
                        skip_reason="budget_fill_zero_after_assignment"))
            chain = kept
            ladder.per_slot[slot.slot_id] = chain
        if fills:
            ladder.per_slot_portfolio[slot.slot_id] = {
                s_: fills[s_] for s_ in chain}
        ladder.per_slot_payload_tokens[slot.slot_id] = sum(
            fills.get(s_, 0) * PAYLOAD_TOKENS_PER_CHUNK.get(
                s_, _PAYLOAD_TOKENS_UNKNOWN) for s_ in chain)
        ladder.total_estimated_cost += option.cost
        slot_accs.append(acc)

        st.payload_tokens_worst_case = ladder.per_slot_payload_tokens[slot.slot_id]
        st.final_chain = chain
        st.final_confidence = option.confidence
        st.final_lb = option.lb
        st.final_latency_ms = option.latency_ms
        st.bar_cleared = option.lb >= c["adjusted_bar"]
        st.status = status
        st.terminal_action = slot_terminal_action(status)
        st.stop_reason = "solver_selected" if chain else "no_viable_candidates"

    ladder.total_estimated_ms = max(ladder.per_slot_latency_ms.values(), default=0)
    ladder.total_payload_tokens = sum(ladder.per_slot_payload_tokens.values())
    ladder.aggregate_confidence_estimate = (
        sum(ladder.per_slot_confidence.values()) / len(choices)
    )
    ladder.aggregate_accuracy_estimate = sum(slot_accs) / len(slot_accs)

    parts = " + ".join(f"{ladder.per_slot_lb[s.slot_id]:.4f}" for s in ordered)
    trace.aggregate_confidence = ladder.aggregate_confidence_estimate
    trace.aggregate_arithmetic = (
        f"per-slot LBs [{parts}] vs bar {c['adjusted_bar']:.4f} (mean is telemetry only)"
    )

    ladder.outcome = ladder_outcome(ladder.per_slot_status)
    ladder.terminal_action = decision_terminal_action(ladder.per_slot_terminal)
    ladder.helpers = decision_helpers(ladder.per_slot_helpers)
    ladder.feasible = ladder.outcome == OUTCOME_ALL_CLEARED
    if not ladder.feasible:
        # only REQUIRED slots gate — optional slots never appear here
        failed = {sid: status for sid, status in ladder.per_slot_status.items()
                  if status in (STATUS_UNDER_CONFIDENT, STATUS_NO_VIABLE)}
        ladder.infeasibility_reason = (
            f"required slots below LB bar {c['adjusted_bar']:.3f} at level "
            f"{confidence_level:.2f}: {failed}"
        )

    trace.outcome = ladder.outcome
    trace.helpers = list(ladder.helpers)
    trace.feasible = ladder.feasible
    trace.infeasibility_reason = ladder.infeasibility_reason
    return ladder
