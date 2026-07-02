"""Router — picks one retrieval strategy for a query, given caller preferences.

The router is the load-bearing component of the retrieval portfolio.
Each strategy (a-d) has a known precision/recall/speed profile; each
caller has different needs ("I want it fast and accuracy isn't critical"
vs "I want it definitive and I'll wait"). The router matches them.

Architecture (locked 2026-05-01)
--------------------------------

1. **Caller passes preferences** — explicitly, or as a named ``caller_mode``
   that resolves to a preset. Individual fields override preset defaults.

2. **Two-stage routing**
     Stage 1 — Fail-Fast gate (PHI / jailbreak / no-d-tag → refuse)
     Stage 2 — Score the surviving strategies, pick top-2 (primary + fallback)

3. **Conditional priors per query class** — a strategy that's strong on
   PRECISION_DOMINANT queries shouldn't have its prior bleed into VAGUE
   queries. The bandit (later) updates per-class cells, not global rows.

4. **Multiplicative quality enforced via ReAct re-route** — the score
   function is additive for ranking, but if the chosen strategy returns
   ``confidence=low``, the agent runs the fallback. ReAct does the
   "accuracy needs coverage, coverage needs accuracy" enforcement, not
   the math.

5. **Telemetry frozen on day 1** — every routing decision logs the full
   schema (prefs + scores + priors_version + outcome) so the bandit can
   train off historical data without schema migrations.

Bandit update rules and reward attribution are deferred. The architecture
is bandit-ready; the priors table is mutable; the schema captures the
needed signal. We log first, learn later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal


logger = logging.getLogger(__name__)


# ============================================================================
# Public types
# ============================================================================

AnswerShape = Literal["essay", "structured", "binary", "any"]
SpeedBudget = Literal["real_time", "interactive", "background", "none"]
StrategyId = Literal["a", "b", "c", "d", "e"]
RoutingMethod = Literal["deterministic", "bandit", "override", "fail_fast"]


@dataclass
class RoutePreferences:
    """Caller's wish-list. All fields optional; missing fields fall back
    to the preset implied by ``caller_mode``, which itself defaults to
    ``chat.default`` if unset.
    """
    answer_shape: AnswerShape | None = None
    accuracy_need: float | None = None        # 0..1
    recall_demand: float | None = None        # 0..1, was "miss_cost"
    speed_budget: SpeedBudget | None = None
    cost_budget: float | None = None          # USD per query; not enforced in v1
    caller_mode: str | None = None             # path: "chat.default", "auth_agent", ...


@dataclass
class StrategyPrior:
    """Static profile of a strategy on a given query class.

    These are the bandit's parameters. v1 ships hand-set values; the
    bandit updates the cell that matched the query.

    v1.2 adds ``accuracy_std`` and ``recall_std`` — the run-to-run
    standard deviation observed on this cell. Filled by the
    ``derive_priors`` script from ``--repeats N`` matrix data. The
    scorer uses these to compute a confidence-adjusted lower bound
    when the caller's ``accuracy_need`` is high (risk-averse). At
    ``accuracy_need=0.5`` the std is ignored (raw mean used); at
    ``accuracy_need=1.0`` we use ``mean − 1·std`` (one-σ haircut).
    """
    shape_capabilities: list[AnswerShape]
    accuracy: float           # mean accuracy across observed cells (0..1)
    recall_capacity: float    # 0..1, was "coverage"
    speed: float              # 0..1, higher = faster
    cost_per_call: float = 0.0  # USD; only c, d are non-zero
    accuracy_std: float = 0.0   # run-to-run std on accuracy (default 0 — pre-v1.2)
    recall_std: float = 0.0     # run-to-run std on recall_capacity


@dataclass
class RouteDecision:
    """Output of ``Router.decide()`` — what should run, and why."""
    strategy: StrategyId
    fallback: StrategyId | None
    routing_method: RoutingMethod
    query_class: str
    scores: dict[str, float] = field(default_factory=dict)  # strategy_id -> score
    prefs_resolved: dict[str, Any] = field(default_factory=dict)
    priors_version: str = "v1"
    fail_fast_reason: str | None = None     # populated when strategy == "e"


# ============================================================================
# Caller-mode presets
# ============================================================================
#
# Hand-tuned starting values. Operators can edit in code (v1) or move to
# a config table later. Each preset fully specifies a RoutePreferences;
# caller-supplied fields override on a per-field basis.

CALLER_MODE_PRESETS: dict[str, RoutePreferences] = {
    # --- Chat surface ---
    "chat.copilot": RoutePreferences(
        answer_shape="structured",
        accuracy_need=0.70,
        recall_demand=0.70,
        speed_budget="real_time",
    ),
    "chat.default": RoutePreferences(
        answer_shape="essay",
        accuracy_need=0.85,
        recall_demand=0.95,
        speed_budget="real_time",
    ),
    "chat.thinking": RoutePreferences(
        answer_shape="essay",
        accuracy_need=0.95,
        recall_demand=0.95,
        speed_budget="interactive",
    ),
    # --- Programmatic callers ---
    "auth_agent": RoutePreferences(
        answer_shape="binary",
        accuracy_need=1.00,
        recall_demand=0.95,
        speed_budget="interactive",
    ),
    "research": RoutePreferences(
        answer_shape="essay",
        accuracy_need=0.95,
        recall_demand=1.00,
        speed_budget="none",      # triggers fanout when implemented
    ),
    "batch": RoutePreferences(
        answer_shape="structured",
        accuracy_need=0.90,
        recall_demand=0.80,
        speed_budget="background",
    ),
}

DEFAULT_CALLER_MODE = "chat.default"


def resolve_preferences(prefs: RoutePreferences | None) -> RoutePreferences:
    """Merge caller's input onto the named-preset defaults.

    Resolution order (highest precedence first):
      1. Fields the caller set explicitly
      2. Preset implied by ``caller_mode``
      3. ``chat.default`` preset
    """
    if prefs is None:
        prefs = RoutePreferences()

    mode = prefs.caller_mode or DEFAULT_CALLER_MODE
    base = CALLER_MODE_PRESETS.get(mode) or CALLER_MODE_PRESETS[DEFAULT_CALLER_MODE]

    return RoutePreferences(
        caller_mode=mode,
        answer_shape=prefs.answer_shape or base.answer_shape,
        accuracy_need=prefs.accuracy_need if prefs.accuracy_need is not None else base.accuracy_need,
        recall_demand=prefs.recall_demand if prefs.recall_demand is not None else base.recall_demand,
        speed_budget=prefs.speed_budget or base.speed_budget,
        cost_budget=prefs.cost_budget if prefs.cost_budget is not None else base.cost_budget,
    )


# ============================================================================
# Strategy priors — conditional on query class
# ============================================================================
#
# Query classes coarsely group queries by their retrieval-meaningful
# features. The bandit later updates the (strategy, query_class) cell;
# global priors don't bleed across classes.

QueryClass = Literal[
    "literal_anchor",     # has an HCPCS / policy-ID literal
    "tight_pool",         # cascade L1/L2 produced ≤500 docs and tags matched
    "wide_pool",          # cascade L3/L4 only — broad domain, no tight pool
    "conceptual",         # query_type=CONCEPTUAL — explanation/definition, not lookup
    "exploratory",        # exploratory phrasing detected (tell me / overview / across)
    "vague",              # VAGUE classification (no tags, no literals)
]

PRIORS_VERSION = "v2.1.2026-07-01-canonical-blend"  # +canonical/factual prior blend for b/d niche (run 41b5c5e7)

# v1.2 update — derived from N=5 strategy×query verdict matrix
# (eval/calibration/strategy_matrix_n5_20260503-183648.json) with
# Pythagorean correction subtracting the empirically-measured judge
# noise floor (σ_judge ≈ 0.183, derived from 10 retrieval-deterministic
# cells in the post-fix N=3 a-only matrix
# eval/calibration/strategy_matrix_a_only_n3_20260503-202103.json).
#
# Method:
#   • accuracy        = pooled mean of judge_score across 5 repeats per
#                       cell, then averaged within each (qclass, strategy)
#   • accuracy_std    = √max(0, σ_total² − σ_judge²)  ← the noise floor
#                       we subtract is the LLM judge's own variance, so
#                       what's left is genuine strategy variance
#   • recall_capacity = unchanged from v1.1 (we don't yet have an
#                       independent recall measurement; keeping the
#                       hand-set values that were already tuned)
#
# Bonus from the SQL fix that landed alongside this patch (BM25
# candidate CTE now does ORDER BY ts_rank_cd DESC, id ASC inside the
# LIMIT cap — was non-deterministic before): retrieval for (a) is now
# fully deterministic on 10/10 of the eval queries (was 6/10). cmhc006
# and cmhc010 went from 4 unique top docs across runs to 1 each, AND
# the deterministic top-K is actually a Sunshine doc instead of
# random tangential chunks (judge_score for cmhc006/a went from 0.90
# σ=0.22 to 1.00 σ=0). The accuracy/std values below were measured
# BEFORE this fix landed, so they are pessimistic for (a) on
# tight_pool — we'll re-measure once the bandit has accumulated
# post-fix evidence.
#
# What this changes in routing scores (chat.default mode):
#   • literal_anchor: (a) accuracy 0.50, std 0.330 → eff under k=0.7
#     drops to 0.27. (b) accuracy 0.53, std 0 → eff 0.53. (b) wins
#     outright. Fixes cmhc002 + cmhc008.
#   • tight_pool: (a) 0.67 std 0.278 → eff 0.48; (b) 0.50 std 0.479 →
#     eff 0.16. (a) wins, with confidence.
#   • wide_pool: (a) 0.74 std 0.212 → eff 0.59; (b) 0.73 std 0.201 →
#     eff 0.59. Tie — bandit picks based on cost/speed.
#   • vague: all four drop sharply (means 0.0). Cmhc009 routes to (e)
#     fail-fast.
_BASE_PRIORS: dict[StrategyId, dict[QueryClass, StrategyPrior]] = {
    "a": {
        "literal_anchor": StrategyPrior(['structured'], accuracy=0.329, accuracy_std=0.229, recall_capacity=1, speed=0.07),  # calib n=7
        "tight_pool": StrategyPrior(['structured'], accuracy=0.5, accuracy_std=0.126, recall_capacity=1, speed=0.72),  # calib n=8
        "wide_pool": StrategyPrior(['structured'], accuracy=0.667, accuracy_std=0.105, recall_capacity=1, speed=0.62),  # calib n=6
        "conceptual": StrategyPrior(['structured'], accuracy=0.405, accuracy_std=0.194, recall_capacity=0.905, speed=0.94),  # calib n=21
        "exploratory": StrategyPrior(['structured'], accuracy=0.55, accuracy_std=0.2, recall_capacity=0.35, speed=0.85),  # (no calib data — kept)
        "vague": StrategyPrior(['structured'], accuracy=0.0, accuracy_std=0.0, recall_capacity=0, speed=0.95),  # calib n=4
    },
    "b": {
        "literal_anchor": StrategyPrior(['structured', 'essay'], accuracy=0.474, accuracy_std=0.264, recall_capacity=1, speed=0.12),  # calib n=7
        "tight_pool": StrategyPrior(['structured', 'essay'], accuracy=0.334, accuracy_std=0.357, recall_capacity=0.5, speed=0.3),  # calib n=8
        "wide_pool": StrategyPrior(['structured', 'essay'], accuracy=0.3, accuracy_std=0.298, recall_capacity=0.6, speed=0.3),  # calib n=5
        "conceptual": StrategyPrior(['structured', 'essay'], accuracy=0.317, accuracy_std=0.401, recall_capacity=0.476, speed=0.23),  # calib n=21
        "exploratory": StrategyPrior(['structured', 'essay'], accuracy=0.75, accuracy_std=0.2, recall_capacity=0.85, speed=0.85),  # (no calib data — kept)
        "vague": StrategyPrior(['structured', 'essay'], accuracy=0.0, accuracy_std=0.0, recall_capacity=0, speed=0.95),  # calib n=4
    },
    "c": {
        "literal_anchor": StrategyPrior(['essay', 'structured'], accuracy=0.0, accuracy_std=0.0, recall_capacity=0.714, speed=0.05, cost_per_call=0.02),  # calib n=7
        "tight_pool": StrategyPrior(['essay', 'structured'], accuracy=0.107, accuracy_std=0.142, recall_capacity=0.857, speed=0.12, cost_per_call=0.02),  # calib n=7
        "wide_pool": StrategyPrior(['essay', 'structured'], accuracy=0.0, accuracy_std=0.0, recall_capacity=0.5, speed=0.09, cost_per_call=0.02),  # calib n=6
        "conceptual": StrategyPrior(['essay', 'structured'], accuracy=0.233, accuracy_std=0.278, recall_capacity=0.55, speed=0.09, cost_per_call=0.02),  # calib n=20
        "exploratory": StrategyPrior(['essay', 'structured'], accuracy=0.6, accuracy_std=0.3, recall_capacity=0.85, speed=0.4, cost_per_call=0.02),  # (no calib data — kept)
        "vague": StrategyPrior(['essay', 'structured'], accuracy=0.0, accuracy_std=0.0, recall_capacity=0, speed=0.95, cost_per_call=0.02),  # calib n=4
    },
    "d": {
        "literal_anchor": StrategyPrior(['essay', 'structured'], accuracy=0.22, accuracy_std=0.228, recall_capacity=1, speed=0.08, cost_per_call=0.03),  # calib n=5
        "tight_pool": StrategyPrior(['essay', 'structured'], accuracy=0.719, accuracy_std=0.318, recall_capacity=1, speed=0.12, cost_per_call=0.03),  # calib n=8
        "wide_pool": StrategyPrior(['essay', 'structured'], accuracy=0.25, accuracy_std=0.175, recall_capacity=1, speed=0.19, cost_per_call=0.03),  # calib n=6
        "conceptual": StrategyPrior(['essay', 'structured'], accuracy=0.436, accuracy_std=0.339, recall_capacity=0.905, speed=0.17, cost_per_call=0.03),  # calib n=21
        "exploratory": StrategyPrior(['essay', 'structured'], accuracy=0.5, accuracy_std=0.2, recall_capacity=0.5, speed=0.65, cost_per_call=0.03),  # (no calib data — kept)
        "vague": StrategyPrior(['essay', 'structured'], accuracy=0.0, accuracy_std=0.0, recall_capacity=0, speed=0.95, cost_per_call=0.03),  # calib n=4
    },
}


# ============================================================================
# Canonical-vs-factual prior blend
# ============================================================================
#
# The 6 pool/anchor classes above split queries by SHAPE, not by whether the
# answer lives in a canonical policy theme vs a precise fact snippet. That
# axis is orthogonal, and it's where strategy (b) lives: on the run-41b5c5e7
# baseline, (b) recall is BIMODAL — ~0.51 on canonical queries (a clean
# J-payor × D-topic tag pair resolving to a moderate ~100-500 doc pool, e.g.
# the Sunshine-Health prior-authorization cluster cmhc002/006/011/018) but
# ~0.03 on factual queries. The single pool-class prior averages these to
# ~0.3, which is simultaneously too LOW to ever let (b) win its niche and too
# HIGH for factual queries — so (b) is never picked at all.
#
# Fix: measure a CANONICAL prior profile (below) and blend it into the
# pool-class prior by a continuous canonicality weight w∈[0,1]. w=0 leaves the
# factual (pool-class) prior untouched; w=1 uses the canonical profile. This
# lifts (b) AND (d) — both are canonical-theme tools — exactly on the class
# where they beat (a), without touching factual routing (where (b) stays ~0).
# Numbers are measured recall/answer_rate/latency from the canonical cells of
# run 41b5c5e7 (fixed-retrieval baseline, locked judge).
_CANONICAL_PRIORS: dict[StrategyId, StrategyPrior] = {
    "a": StrategyPrior(['structured'],            accuracy=0.467, accuracy_std=0.125, recall_capacity=1.00, speed=0.98),                     # calib n=10
    "b": StrategyPrior(['structured', 'essay'],   accuracy=0.508, accuracy_std=0.377, recall_capacity=0.70, speed=0.75),                     # calib n=10 — the niche
    "c": StrategyPrior(['essay', 'structured'],   accuracy=0.033, accuracy_std=0.100, recall_capacity=0.40, speed=0.12, cost_per_call=0.02), # calib n=10
    "d": StrategyPrior(['essay', 'structured'],   accuracy=0.600, accuracy_std=0.367, recall_capacity=1.00, speed=0.48, cost_per_call=0.03), # calib n=10
}


def _canonicality(profile_features: dict[str, Any]) -> float:
    """Continuous [0,1] weight — how much this query looks like a *canonical
    policy* question (answer spread across a coherent theme) vs a *factual*
    lookup. Gated on a J×D tag pair (payer × topic = a real policy exists),
    peaked on a moderate pool (~100-500 docs = one coherent theme), and off
    for VAGUE queries (no signal). Codes/anchors do NOT force factual —
    cmhc002 has HCPCS H0019 yet is canonical (the PA policy is the answer)."""
    if not (profile_features.get("has_j_tag") and profile_features.get("has_d_tag")):
        return 0.0
    if profile_features.get("query_type") == "VAGUE":
        return 0.0
    pool = int(profile_features.get("pool_size", 0) or 0)
    if pool <= 0:
        return 0.0
    # Soft pool membership: plateau at [100,500], linear taper to 0 at 50 / 900.
    if 100 <= pool <= 500:
        pool_w = 1.0
    elif pool < 100:
        pool_w = max(0.0, (pool - 50) / 50.0)
    else:
        pool_w = max(0.0, (900 - pool) / 400.0)
    return pool_w


def _blend_prior(base: StrategyPrior, canon: StrategyPrior, w: float) -> StrategyPrior:
    """Linear interpolation of the numeric prior fields toward the canonical
    profile by weight w. w=0 → base (factual/pool-class) prior unchanged."""
    if w <= 0.0:
        return base
    return StrategyPrior(
        shape_capabilities=base.shape_capabilities,
        accuracy=(1 - w) * base.accuracy + w * canon.accuracy,
        recall_capacity=(1 - w) * base.recall_capacity + w * canon.recall_capacity,
        speed=(1 - w) * base.speed + w * canon.speed,
        cost_per_call=base.cost_per_call,
        accuracy_std=(1 - w) * base.accuracy_std + w * canon.accuracy_std,
        recall_std=(1 - w) * base.recall_std + w * canon.recall_std,
    )


# ============================================================================
# Query-class derivation
# ============================================================================
#
# We don't import QueryProfile here to keep the router decoupled — caller
# passes a small dict with the features we need. Keeps router unit-testable.

def derive_query_class(profile_features: dict[str, Any]) -> QueryClass:
    """Coarse query-class label from classify_query features + cascade pool.

    Inputs (any may be missing):
      * query_type:   PRECISION_DOMINANT | CONCEPTUAL | MIXED | VAGUE
      * has_literal:  bool
      * has_d_tag:    bool
      * pool_size:    int  (from cascade builder; 0 if not run yet)
      * is_exploratory: bool
    """
    qt = profile_features.get("query_type")
    pool = int(profile_features.get("pool_size", 0) or 0)
    if profile_features.get("has_literal"):
        return "literal_anchor"
    if profile_features.get("is_exploratory"):
        return "exploratory"
    if qt == "VAGUE":
        return "vague"
    # Conceptual queries ask HOW/WHY/WHAT-IS rather than looking up a specific
    # fact. (c) LLM parametric prior beats (d) external search for these because
    # the answer is stable policy knowledge, not a live lookup. Route to the
    # dedicated conceptual class so the priors reflect this asymmetry.
    # MIXED and PRECISION_DOMINANT still fall through to pool-size routing.
    if qt == "CONCEPTUAL":
        return "conceptual"
    if 0 < pool <= 500:
        return "tight_pool"
    if pool > 500:
        return "wide_pool"
    # Fallback when no pool info: tag-presence is the proxy
    return "tight_pool" if profile_features.get("has_d_tag") else "vague"


# ============================================================================
# Speed-budget weighting
# ============================================================================
#
# Speed contributes to the score in proportion to how budget-constrained
# we are. ``none`` zeroes out the speed term — caller said latency is no
# constraint, so we shouldn't penalize slow strategies. ``real_time``
# weighs speed heavily.

_SPEED_WEIGHT: dict[SpeedBudget, float] = {
    "real_time":   1.0,
    "interactive": 0.4,
    "background":  0.1,
    "none":        0.0,
}


# ============================================================================
# Scoring + decision
# ============================================================================

# Shape-match weight in the score. Soft: a strategy capable of producing
# the requested shape gets the bonus; one that isn't, doesn't. We don't
# hard-exclude on shape — sometimes the chat planner can render any shape.
#
# v1.2.2 update — bumped from 0.20 → 0.30 so essay-capable strategies
# (b, c, d) earn a bigger edge when chat.default asks for ``essay``.
# The N=5 matrix showed (b) consistently outperforms (a) on
# wide_pool conceptual queries (cmhc003/004) but (a) was still
# winning routing because shape contributed only 0.10. With the bump,
# (b) wins wide_pool routing while (a) still wins tight_pool/literal
# (where (b)'s bimodal σ takes it out of contention).
_SHAPE_MATCH_WEIGHT = 0.30


def _shape_match(prior: StrategyPrior, requested: AnswerShape | None) -> float:
    if requested is None or requested == "any":
        return 1.0
    # v1.2.2: penalty for missing the requested shape tightened 0.5 → 0.4.
    # When the caller specifically asks for essay (chat.default,
    # chat.thinking, research), a structured-only strategy is a worse
    # match than the previous 0.5 floor implied.
    return 1.0 if requested in prior.shape_capabilities else 0.4


def _score_strategy(prior: StrategyPrior, prefs: RoutePreferences) -> float:
    """Linear-ish score. ReAct enforces the multiplicative-quality rule
    at the executor level — if the chosen strategy returns low confidence,
    the agent re-routes to the fallback. The score function only ranks.

    v1.2 confidence-adjusted scoring: high-stakes callers (accuracy_need
    near 1.0) trust the LOWER end of each strategy's observed accuracy
    distribution, not the mean. ``risk_k`` interpolates linearly from 0
    at ``accuracy_need=0.5`` to 1.0 at ``accuracy_need=1.0``. This
    naturally penalises high-variance strategies (e.g. (d) external
    Google: SERP composition shifts between runs → wide accuracy_std)
    when the caller can't tolerate noise. Same logic for recall.
    """
    speed_w = _SPEED_WEIGHT.get(prefs.speed_budget or "real_time", 1.0)
    accuracy_need = prefs.accuracy_need or 0.5
    recall_demand = prefs.recall_demand or 0.5

    # 0 at need=0.5, 1.0 at need=1.0; clamped at 0 below 0.5 (no risk-seeking).
    risk_k = max(0.0, (accuracy_need - 0.5) * 2.0)
    accuracy_eff = max(0.0, prior.accuracy        - risk_k * prior.accuracy_std)
    recall_eff   = max(0.0, prior.recall_capacity - risk_k * prior.recall_std)

    return (
        accuracy_eff * accuracy_need
        + recall_eff * recall_demand
        + prior.speed * speed_w
        + _shape_match(prior, prefs.answer_shape) * _SHAPE_MATCH_WEIGHT
    )


# Strategies whose effective recall is below this threshold are
# withdrawn from competition entirely — they admit they have nothing
# to offer for this query.
_WITHDRAW_RECALL_THRESHOLD = 0.05


def decide(
    profile_features: dict[str, Any],
    prefs: RoutePreferences | None = None,
    *,
    fail_fast_reason: str | None = None,
    self_assessments: dict[str, tuple[float, str]] | None = None,
    prior_strategies_tried: list[str] | None = None,
) -> RouteDecision:
    """Pick a strategy for this query.

    ``profile_features`` — features from ``classify_query``.
    ``prefs`` — caller's wish-list (or None for defaults).
    ``fail_fast_reason`` — short-circuits to strategy ``e``.
    ``self_assessments`` — optional ``{strategy_id: (est_recall, reason)}``
        produced by the agent before calling. Strategies whose estimate
        is below the withdrawal threshold are dropped. When the static
        prior is preferred (e.g. for c, d which always have recall),
        omit them from this dict.
    ``prior_strategies_tried`` — strategies already executed in this
        thread (e.g. ["a", "b"]). They are excluded from scoring so the
        bandit picks the next best arm on re-invocation. When all corpus
        strategies (a/b/c) are exhausted, the router naturally escalates
        to (d) external.

    Returns ``RouteDecision`` with primary + fallback. The agent
    executes primary and may run fallback if confidence is low and
    speed budget allows (ReAct re-route handled outside the router).
    """
    resolved = resolve_preferences(prefs)
    prefs_dump = {
        "answer_shape": resolved.answer_shape,
        "accuracy_need": resolved.accuracy_need,
        "recall_demand": resolved.recall_demand,
        "speed_budget": resolved.speed_budget,
        "cost_budget": resolved.cost_budget,
        "caller_mode": resolved.caller_mode,
    }

    if fail_fast_reason:
        return RouteDecision(
            strategy="e",
            fallback=None,
            routing_method="fail_fast",
            query_class=derive_query_class(profile_features),
            scores={},
            prefs_resolved=prefs_dump,
            priors_version=PRIORS_VERSION,
            fail_fast_reason=fail_fast_reason,
        )

    qclass = derive_query_class(profile_features)
    # Canonicality weight: blends the pool-class prior toward the canonical
    # profile (lifts b/d on J×D-pair moderate-pool policy queries). 0 for
    # factual queries → no change to existing routing.
    w_canon = _canonicality(profile_features)
    self_assessments = self_assessments or {}
    excluded_strategies: set[str] = set(prior_strategies_tried or [])

    scores: dict[str, float] = {}
    score_breakdown: dict[str, dict[str, Any]] = {}
    withdrawn: list[str] = []
    assessment_log: dict[str, dict[str, Any]] = {}

    speed_w = _SPEED_WEIGHT.get(resolved.speed_budget or "real_time", 1.0)
    accuracy_need = resolved.accuracy_need or 0.5
    recall_demand = resolved.recall_demand or 0.5

    for sid in ("a", "b", "c", "d"):
        prior = _blend_prior(_BASE_PRIORS[sid][qclass], _CANONICAL_PRIORS[sid], w_canon)

        # Exclude strategies already tried in this thread — re-invocation
        # path. Scored zero so they sort last and never win.
        if sid in excluded_strategies:
            withdrawn.append(sid)
            scores[sid] = 0.0
            score_breakdown[sid] = {
                "withdrawn": True,
                "withdraw_reason": "already_tried_in_thread",
            }
            continue

        # Per-query recall: use self-assessment if provided; otherwise
        # fall back to the static prior. (a) and (b) always self-assess
        # (corpus coverage check); (c) and (d) intentionally don't —
        # their recall is "the world" by definition.
        if sid in self_assessments:
            est_recall, reason = self_assessments[sid]
            assessment_log[sid] = {
                "est_recall": round(est_recall, 3),
                "static_recall": prior.recall_capacity,
                "reason": reason,
            }
        else:
            est_recall = prior.recall_capacity
            assessment_log[sid] = {
                "est_recall": round(est_recall, 3),
                "static_recall": prior.recall_capacity,
                "reason": "static_prior",
            }

        # Withdrawal — strategy admits it has nothing for this query.
        if est_recall < _WITHDRAW_RECALL_THRESHOLD:
            withdrawn.append(sid)
            scores[sid] = 0.0
            score_breakdown[sid] = {
                "withdrawn": True,
                "withdraw_reason": (
                    f"est_recall={est_recall:.3f} < threshold "
                    f"{_WITHDRAW_RECALL_THRESHOLD:.2f}"
                ),
                "total": 0.0,
            }
            continue

        # Score using estimated_recall in place of the static recall_capacity.
        # Each term is exposed so the trace UI can show "this strategy
        # got 0.81 from accuracy * 0.85 + 0.78 from recall * 0.95 + …".
        accuracy_term = prior.accuracy * accuracy_need
        recall_term = est_recall * recall_demand
        speed_term = prior.speed * speed_w
        shape_match = _shape_match(prior, resolved.answer_shape)
        shape_term = shape_match * _SHAPE_MATCH_WEIGHT
        total = accuracy_term + recall_term + speed_term + shape_term

        # Payer-specificity-aware adjustment (v1.2.4 — 2026-05-03):
        # When the query has NO PAYER-specific j-tag (j:state.* and
        # j:program.* don't count — they're jurisdiction, not payer),
        # the answer is "general knowledge" shape — standardised codes,
        # universal procedures, training-data territory. (a)/(b) lose
        # because the cascade pool widens (no payer narrowing) and
        # corpus retrieval becomes diffuse; (c) wins because its LLM
        # prior IS "the world's policy" when no specific payer applies.
        # Boosts only when:
        #   • not has_j_payor_tag (no specific payer named)
        #   • AND has_d_tag (some topical anchor — not fail-fast vague)
        # Empirical anchor: cmhc012 (ABA HCPCS — query mentions
        # "FL Medicaid" which matches j:state.florida but no payer).
        # (c) was the only strategy returning the right CPT codes via
        # LLM prior + clinical-policy citation; (a)/(b)/(d) abstained
        # or returned generic content.
        adj = 0.0
        adj_reason = None
        # No-payer routing: when the query has a domain tag but no specific
        # payer, the answer is general/public-knowledge shape. (c) wins
        # because its LLM prior IS global policy when no payer narrows it.
        # Exception: when pool already landed at AHCA scope (L3_AHCA_D /
        # L4_AHCA), the domain pool IS meaningful narrowing — AHCA
        # substitutes for the absent payer (v1.2.6). Skip the haircut and
        # let the AHCA-substitution block below do the routing instead.
        if (
            profile_features.get("has_d_tag")
            and not profile_features.get("has_j_payor_tag")
            and not profile_features.get("has_ahca_pool")
        ):
            if sid == "c":
                adj = +0.40
                adj_reason = "no_payer_general_knowledge_boost"
            elif sid == "a":
                adj = -0.30
                adj_reason = "no_payer_a_recall_haircut"

        # AHCA domain substitutes absent payer (v1.2.6 — 2026-05-05):
        # When the pool cascade landed at AHCA scope and the query has a
        # domain tag but no payer tag, AHCA is the effective payer scope.
        # (a) BM25 precision WITHIN AHCA∩D is the right first attempt —
        # it finds whatever payer-adjacent content the corpus has.
        # (b) wide-themes is redundant: the pool is already domain-narrowed.
        if (
            profile_features.get("has_ahca_pool")
            and not profile_features.get("has_j_payor_tag")
            and profile_features.get("has_d_tag")
        ):
            if sid == "a":
                adj += +0.20
                adj_reason = (adj_reason or "") + "+ahca_domain_substitutes_payer"
            elif sid == "b":
                adj -= 0.20
                adj_reason = (adj_reason or "") + "-ahca_b_redundant_with_domain_pool"

        # Zero-cooc routing (v1.2.5 — 2026-05-05):
        # When _estimate_internal_recall found a content token with ZERO
        # corpus presence (e.g. "molina" when Molina docs aren't indexed),
        # no amount of BM25 or vector recall can help — the corpus simply
        # doesn't have this entity. Route hard to (d) external.
        # Condition: has_zero_cooc_term AND has_d_tag (domain is valid) AND
        # NOT has_literal (literal-anchor miss is a different failure mode —
        # let the literal-anchor hard-withdraw handle it via est_recall=0).
        if (
            profile_features.get("has_zero_cooc_term")
            and profile_features.get("has_d_tag")
            and not profile_features.get("has_literal")
        ):
            if sid == "d":
                adj += +0.60
                adj_reason = (adj_reason or "") + "+zero_cooc_entity_not_in_corpus"
            elif sid in ("a", "b"):
                adj += -0.40
                adj_reason = (adj_reason or "") + "-zero_cooc_internal_strategy_penalised"
        total += adj

        scores[sid] = round(total, 4)
        score_breakdown[sid] = {
            "withdrawn": False,
            "accuracy": {
                "prior": prior.accuracy,
                "weight": round(accuracy_need, 3),
                "contrib": round(accuracy_term, 4),
            },
            "recall": {
                "est": round(est_recall, 3),
                "static": prior.recall_capacity,
                "weight": round(recall_demand, 3),
                "contrib": round(recall_term, 4),
            },
            "speed": {
                "prior": prior.speed,
                "weight": round(speed_w, 3),
                "contrib": round(speed_term, 4),
            },
            "shape": {
                "match": round(shape_match, 3),
                "weight": _SHAPE_MATCH_WEIGHT,
                "contrib": round(shape_term, 4),
            },
            "cost_per_call": prior.cost_per_call,
            "total": round(total, 4),
            "adj": round(adj, 4),
            "adj_reason": adj_reason,
        }

    # Pick highest-scoring non-withdrawn strategy.
    ranked = sorted(
        ((s, sc) for s, sc in scores.items() if s not in withdrawn),
        key=lambda kv: -kv[1],
    )
    if ranked:
        primary = ranked[0][0]
        fallback = ranked[1][0] if len(ranked) > 1 else None
    else:
        # All strategies withdrew — extremely degenerate case. Fall back
        # to (e) refuse with a synthetic reason so the agent surfaces the
        # situation honestly rather than picking a zero-recall winner.
        return RouteDecision(
            strategy="e",
            fallback=None,
            routing_method="fail_fast",
            query_class=qclass,
            scores=scores,
            prefs_resolved=prefs_dump,
            priors_version=PRIORS_VERSION,
            fail_fast_reason="all_strategies_withdrew",
        )

    logger.info(
        "[router] decide qclass=%s prefs=%s scores=%s withdrawn=%s primary=%s fallback=%s",
        qclass, prefs_dump, scores, withdrawn, primary, fallback,
    )

    decision = RouteDecision(
        strategy=primary,
        fallback=fallback,
        routing_method="deterministic",
        query_class=qclass,
        scores=scores,
        prefs_resolved=prefs_dump,
        priors_version=PRIORS_VERSION,
    )
    # Stash assessments + withdrawn + per-strategy breakdown for telemetry.
    setattr(decision, "self_assessments", assessment_log)
    setattr(decision, "withdrawn", withdrawn)
    setattr(decision, "score_breakdown", score_breakdown)
    return decision


async def persist_decision(
    db_session_factory,
    *,
    agent_id: str,
    query: str,
    profile_features: dict[str, Any],
    decision: RouteDecision,
    response_dump: dict[str, Any],
) -> None:
    """Append-only write of one routing decision + immediate outcome.

    Called by the agent at the end of every request. Errors are logged
    and swallowed — telemetry persistence MUST NOT break the user's
    request. ``db_session_factory`` is the AsyncSessionLocal callable
    so we can open an independent session (the agent's session may
    already be in use elsewhere).
    """
    from sqlalchemy import text as _t

    try:
        async with db_session_factory() as db:
            await db.execute(_t("""
                INSERT INTO rag_routing_decisions (
                  agent_id, query,
                  query_type, query_class, coverage,
                  has_d_tag, has_literal, is_exploratory,
                  tag_matches, literal_anchors, untagged_meaningful,
                  caller_mode, prefs_received, prefs_resolved,
                  routing_method, scores, self_assessments, withdrawn,
                  strategy_chosen, strategy_executed, fallback_strategy,
                  priors_version, fail_fast_reason,
                  confidence, n_chunks, top_rerank, total_ms,
                  per_strategy_telemetry
                ) VALUES (
                  :agent_id, :query,
                  :query_type, :query_class, :coverage,
                  :has_d_tag, :has_literal, :is_exploratory,
                  :tag_matches, :literal_anchors, :untagged_meaningful,
                  :caller_mode, :prefs_received, :prefs_resolved,
                  :routing_method, :scores, :self_assessments, :withdrawn,
                  :strategy_chosen, :strategy_executed, :fallback_strategy,
                  :priors_version, :fail_fast_reason,
                  :confidence, :n_chunks, :top_rerank, :total_ms,
                  :per_strategy_telemetry
                )
            """), {
                "agent_id": agent_id,
                "query": query,
                "query_type": profile_features.get("query_type"),
                "query_class": decision.query_class,
                "coverage": profile_features.get("coverage"),
                "has_d_tag": profile_features.get("has_d_tag"),
                "has_literal": profile_features.get("has_literal"),
                "is_exploratory": profile_features.get("is_exploratory"),
                "tag_matches": _json(profile_features.get("tag_matches")),
                "literal_anchors": _json(profile_features.get("literal_anchors")),
                "untagged_meaningful": _json(profile_features.get("untagged_meaningful_tokens")),
                "caller_mode": (decision.prefs_resolved or {}).get("caller_mode"),
                "prefs_received": _json(response_dump.get("prefs_received")),
                "prefs_resolved": _json(decision.prefs_resolved),
                "routing_method": decision.routing_method,
                "scores": _json(decision.scores),
                "self_assessments": _json(getattr(decision, "self_assessments", {})),
                "withdrawn": _json(getattr(decision, "withdrawn", [])),
                "strategy_chosen": decision.strategy,
                "strategy_executed": response_dump.get("strategy_executed", decision.strategy),
                "fallback_strategy": decision.fallback,
                "priors_version": decision.priors_version,
                "fail_fast_reason": decision.fail_fast_reason,
                "confidence": response_dump.get("confidence"),
                "n_chunks": response_dump.get("n_chunks"),
                "top_rerank": response_dump.get("top_rerank"),
                "total_ms": response_dump.get("total_ms"),
                "per_strategy_telemetry": _json(response_dump.get("per_strategy_telemetry")),
            })
            await db.commit()
    except Exception as exc:
        logger.warning("persist_decision failed (non-fatal): %s", exc)


def _json(v: Any) -> str | None:
    """JSON-encode a value for JSONB column. None passes through."""
    if v is None:
        return None
    import json as _j
    try:
        return _j.dumps(v)
    except Exception:
        return None


def decide_override(
    forced_strategy: StrategyId,
    profile_features: dict[str, Any],
    prefs: RoutePreferences | None = None,
) -> RouteDecision:
    """Explicit-override path. Used by ``request.mode = "explore" |
    "validate"`` for testing/diagnostics. No fallback (override is a
    forcing tool — caller doesn't want the router second-guessing).
    """
    resolved = resolve_preferences(prefs)
    return RouteDecision(
        strategy=forced_strategy,
        fallback=None,
        routing_method="override",
        query_class=derive_query_class(profile_features),
        scores={},
        prefs_resolved={
            "answer_shape": resolved.answer_shape,
            "accuracy_need": resolved.accuracy_need,
            "recall_demand": resolved.recall_demand,
            "speed_budget": resolved.speed_budget,
            "cost_budget": resolved.cost_budget,
            "caller_mode": resolved.caller_mode,
        },
        priors_version=PRIORS_VERSION,
    )
