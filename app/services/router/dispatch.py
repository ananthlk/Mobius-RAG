"""Router dispatch — forced bypass, or three-way allocator A/B/C.

  FORCED     — calibration or caller-forced strategy: single strategy,
               max_attempts=1, no optimization (isolation measurement).
  PRODUCTION — ALL selectable allocators compute a RoutingLadder every query:
      greedy    (allocation.py)          — sequential-fallback heuristic
      optimizer (optimizer.py)           — exact solve, Wilson-LB bound
      bayesian  (bayesian_optimizer.py)  — exact solve, Beta-quantile bound
    ONE ladder executes (chosen by a deterministic weighted draw over the
    Eval-owned `allocator_weights` policy); the untaken allocators' plans are
    logged as SHADOW plans — plan-diagnostics only per spec §6b (attribution
    of outcomes is population-level via this split, never per-query
    counterfactuals).

Selection is a DETERMINISTIC sha256 draw over the query text mapped onto the
cumulative weight buckets (fixed allocator order: greedy, optimizer,
bayesian). Same query -> same executed allocator: replayable. Default weights
= equal thirds during bootstrap (all three accumulate comparable executed
samples); legacy `ab_split_optimizer` maps to a two-way split for back-compat.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Optional

# Fixed order — determines the cumulative bucket layout for the draw.
# "portfolio" (blend-model design doc §3, Ananth-signed 2026-07-24) rides as
# SHADOW-ONLY at bootstrap: it is in the order (so it shadow-computes and
# persists on every query) but absent from the default weights (so it never
# executes until Eval shifts allocator_weights in the priors file —
# code-free cutover; execution also needs Retriever's retention wiring).
ALLOCATOR_ORDER = ("greedy", "optimizer", "bayesian", "portfolio")


# DATA-COLLECTION THROTTLE (Ananth's pullback, 2026-07-24): forced-strategy
# arms the throttle may draw. {a,b,c,d,s} only — e is a terminal dispatch
# outcome and f is retired (both Eval-ratified boundaries); s is drawn only
# for payor-tagged queries (its cells mean P(success | payor tag)). The
# PROD arm set is CONFIGURABLE via the policy file (forced_arms) so
# excluding e.g. c from live traffic is a config flip, not a rebuild.
THROTTLE_ARM_ROSTER = ("a", "b", "c", "d", "s")


@dataclass
class DispatchDecision:
    """Which path handles this query — and, in production, which allocator executes."""
    path: Literal["forced", "greedy", "optimizer", "bayesian"]
    shadow_allocators: list[str] = field(default_factory=list)  # production: the untaken ones
    bypass_kind: Optional[Literal["calibration", "forced_strategy",
                                  "data_collection_throttle"]] = None
    forced_strategy: str | None = None
    max_attempts: int | None = 1   # 1 for forced (isolation); None otherwise (budget-driven)
    weights: dict[str, float] | None = None
    draw: float | None = None
    reason: str = ""


def stable_draw(key: str) -> float:
    """Deterministic draw in [0,1) from a string key (sha256-based).

    Same query -> same draw -> same executed allocator: replayable, and immune
    to PYTHONHASHSEED (never uses built-in hash())."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _pick_allocator(weights: dict[str, float], draw: float) -> str:
    """Map the draw onto cumulative weight buckets in fixed ALLOCATOR_ORDER."""
    active = [(name, weights.get(name, 0.0)) for name in ALLOCATOR_ORDER
              if weights.get(name, 0.0) > 0.0]
    if not active:
        return "greedy"
    total = sum(w for _, w in active)
    cum = 0.0
    for name, w in active:
        cum += w / total
        if draw < cum:
            return name
    return active[-1][0]


def _pick_throttle_arm(arm_weights: dict[str, float], draw: float) -> str:
    """Cumulative-bucket pick over forced arms (fixed THROTTLE_ARM_ROSTER order)."""
    active = [(a, arm_weights.get(a, 0.0)) for a in THROTTLE_ARM_ROSTER
              if arm_weights.get(a, 0.0) > 0.0]
    total = sum(w for _, w in active)
    cum = 0.0
    for arm, w in active:
        cum += w / total
        if draw < cum:
            return arm
    return active[-1][0]


def dispatch(
    *,
    is_calibration: bool,
    forced_strategy: str | None,
    allocator_weights: dict[str, float] | None = None,
    query_key: str = "",
    mode_override: str | None = None,
    phase: str = "bootstrap",
    forced_fraction: float = 0.0,
    forced_arm_weights: dict[str, float] | None = None,
    has_payor_context: bool = False,
) -> DispatchDecision:
    """Route to forced bypass, or pick the EXECUTED allocator (others = shadow).

    Precedence: calibration > forced_strategy > mode_override >
    data-collection throttle > weighted draw.

    THROTTLE (phase == "data_collection", Ananth's 1-in-5): a deterministic
    salted draw sends `forced_fraction` of traffic to a forced-strategy arm
    (stateless stable_draw — round-robin needs a counter and breaks under
    concurrent instances; epsilon is exploitation-biased, collection wants
    coverage). Throttle-forced decisions KEEP shadow allocators (all four,
    incl. portfolio) — the "what would the blend have planned" counterfactual
    beside each forced observation is the return ticket to the blend. This
    differs deliberately from calibration/caller-forced (isolation, no
    shadows) — distinguished by bypass_kind for Eval's row hygiene.
    """
    if is_calibration:
        return DispatchDecision(
            path="forced",
            bypass_kind="calibration",
            forced_strategy=forced_strategy,
            max_attempts=1,
            reason="Eval calibration: isolation mode, single strategy, no optimization",
        )

    if forced_strategy is not None:
        return DispatchDecision(
            path="forced",
            bypass_kind="forced_strategy",
            forced_strategy=forced_strategy,
            max_attempts=1,
            reason=f"caller forced strategy '{forced_strategy}': isolation mode",
        )

    # default: equal thirds over the CHAIN allocators; portfolio weight 0
    # (shadow-only) until Eval's file says otherwise
    weights = allocator_weights or {"greedy": 1 / 3, "optimizer": 1 / 3, "bayesian": 1 / 3}

    if mode_override in ALLOCATOR_ORDER:
        others = [a for a in ALLOCATOR_ORDER if a != mode_override]
        return DispatchDecision(
            path=mode_override,  # type: ignore[arg-type]
            shadow_allocators=others,
            max_attempts=None,
            reason=f"caller pinned executed allocator '{mode_override}' "
                   f"(shadows: {', '.join(others)})",
        )

    if phase == "data_collection" and forced_fraction > 0.0:
        # independent salted draw — decorrelated from the allocator draw
        throttle_draw = stable_draw(f"{query_key}#throttle")
        if throttle_draw < forced_fraction:
            arms = dict(forced_arm_weights or
                        {a: 1.0 for a in THROTTLE_ARM_ROSTER})
            if not has_payor_context:
                arms.pop("s", None)  # s cells mean P(success | payor tag)
            arms = {a: w for a, w in arms.items()
                    if a in THROTTLE_ARM_ROSTER and w > 0.0}
            if arms:
                arm_draw = stable_draw(f"{query_key}#arm")
                arm = _pick_throttle_arm(arms, arm_draw)
                return DispatchDecision(
                    path="forced",
                    bypass_kind="data_collection_throttle",
                    forced_strategy=arm,
                    max_attempts=1,
                    # counterfactual shadows KEPT (unlike isolation-forced)
                    shadow_allocators=list(ALLOCATOR_ORDER),
                    draw=throttle_draw,
                    reason=(f"data-collection throttle: draw {throttle_draw:.4f} < "
                            f"{forced_fraction:.2f} → forced '{arm}' "
                            f"(arm draw {arm_draw:.4f}; shadows kept for counterfactual)"),
                )

    draw = stable_draw(query_key)
    executed = _pick_allocator(weights, draw)
    others = [a for a in ALLOCATOR_ORDER if a != executed]
    weights_str = ", ".join(f"{k}={weights.get(k, 0.0):.2f}" for k in ALLOCATOR_ORDER)
    return DispatchDecision(
        path=executed,  # type: ignore[arg-type]
        shadow_allocators=others,
        max_attempts=None,
        weights=weights,
        draw=draw,
        reason=(f"A/B/C draw {draw:.4f} over weights ({weights_str}) → "
                f"{executed} executes, {', '.join(others)} shadow"),
    )
