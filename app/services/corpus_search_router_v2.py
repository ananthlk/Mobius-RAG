"""Router v2 — action-tree with multi-invoke on impure leaves + linear tweaks.

Drop-in replacement for corpus_search_router.decide(). Activate via:
  ROUTER_VERSION=v2 env var (checked in corpus_search_agent.py)

Changes from v1 (build order per EVAL spec, docs/rag-retrieval-learning-architecture.md §8):
  #1 — Multi-invoke on impure leaves: when the gap between the top-2 strategy
       scores is below IMPURE_GAP_THRESHOLD, sets RouteDecision.invoke_all so the
       agent runs both strategies concurrently, unions chunks, and synthesises once.
       Expected +0.057 on the 22-query CMHC bank (GA forensic, run 53f3aefe).
  #2 — Linear tweaks: neuter corpus_depth (near-1 on almost every query →
       over-pumped a, suppressed d); add tag_coverage (0 matched d-tags →
       downweight a). Fixes cmhc016/021 structural leaks.
  [#3 s top-split, #4 f floor replaces c-terminal, #5 tree+bandit — queued]

RouteDecision.invoke_all is a new field added to corpus_search_router.RouteDecision
(with default None) so v1 callers are unaffected.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.corpus_search_router import (
    RouteDecision,
    RoutePreferences,
    StrategyId,
    resolve_preferences,
    derive_query_class,
    _SPEED_WEIGHT,
    _WITHDRAW_RECALL_THRESHOLD,
)


logger = logging.getLogger(__name__)

PRIORS_VERSION = "v2.6.2026-07-15-union-cap"

# ---------------------------------------------------------------------------
# Impurity threshold for multi-invoke
# When the absolute gap between the top-2 strategy scores is below this value
# AND both strategies score above MIN_INVOKE_SCORE, the agent runs both and
# unions the results. Value calibrated against the 110-cell GA forensic.
# ---------------------------------------------------------------------------
IMPURE_GAP_THRESHOLD = 0.08
MIN_INVOKE_SCORE = 0.30  # both strategies must be plausible, not just close

# ---------------------------------------------------------------------------
# v2 linear scoring weights
# Tweaks vs v1 (_LINEAR_BASE / _LINEAR_WEIGHTS in corpus_search_router.py):
#   - corpus_depth weight for "a": 0.20 → 0.05 (neutered — was near-1 everywhere)
#   - tag_coverage weight for "a": new +0.15 (rewards a when topic tags exist)
# All other weights identical to v1.
# ---------------------------------------------------------------------------
_LINEAR_BASE_V2: dict[str, float] = {
    "a": 0.40,
    "b": 0.20,
    "c": 0.05,
    "d": 0.20,
}

_LINEAR_WEIGHTS_V2: dict[str, dict[str, float]] = {
    "a": {
        "exclusivity":     0.30,
        "literal":         0.25,
        # v2.2: replaced binary corpus_depth (0.05) with depth_x_excl (corpus_depth × exclusivity).
        # Binary was always 1 for any payer with >20 docs — told us nothing about whether
        # the corpus has DEPTH ON THIS SPECIFIC TOPIC. depth_x_excl collapses to ~0 for
        # broad-pool queries (exclusivity ≈ 0.1) while staying at 1.0 for narrow-pool ones.
        # Fixes cmhc013: pool=276 → depth_x_excl=0.181 → a=0.463, gap to b=0.054 → multi-invoke.
        "depth_x_excl":    0.05,
        "tag_coverage":    0.15,   # gated by exclusivity (v2.1): 0 when pool > 250 docs
        "thematic_policy": -0.10,
        "wide_pool":       -0.15,
        # v2.3: inheritance REMOVED from a's weights. Inherited docs expand pool_size,
        # reducing exclusivity (already penalises a via that feature). Keeping +0.05 here
        # double-counted the effect and over-scored a for AHCA-inherited-corpus queries
        # (cmhc013: live a=0.513 vs predicted 0.463, gap to b=0.104 > 0.08 → no multi-invoke).
        # With weight=0: a=0.463, gap=0.054 < 0.08 → invoke_all=['a','b'] fires.
    },
    "b": {
        "thematic_policy":  0.40,
        "corpus_depth":     0.20,  # b: binary depth fine — wide retrieval benefits from any depth
        "literal":         -0.20,
        "exclusivity":      0.05,
    },
    "c": {},
    "d": {
        "crawlability":    0.40,
        "wide_pool":       0.25,
        "inheritance":    -0.25,
        "thematic_policy": -0.20,
        "corpus_depth":   -0.15,
        "literal":        -0.05,
    },
}


def _compute_linear_features_v2(profile_features: dict[str, Any]) -> dict[str, float]:
    """Compute normalized [0..1] feature values for v2 linear scoring.

    Extends v1 with tag_coverage: rewards strategy-a when the query matches
    NARROW topic tags (small corpus pool → the corpus has specific depth here).
    A broad tag (dental.general, 600-doc pool) is NOT an a-signal — it only
    says the payer has docs on the topic, not that a can find the specific answer.
    Gate: exclusivity < 0.20 (pool > 250 docs) → tag_coverage = 0.
    Fixes cmhc013 regression where a broad dental tag over-scored a vs b.
    """
    pool_size = max(1, int(profile_features.get("pool_size") or 500))
    tag_matches = profile_features.get("tag_matches") or []
    n_d_tags = sum(1 for t in tag_matches if str(t).startswith("d:"))
    exclusivity = min(1.0, 50.0 / pool_size)
    corpus_depth_binary = float(
        bool(profile_features.get("has_j_payor_tag", False))
        and (pool_size >= 20 or bool(profile_features.get("has_inherited_docs", False)))
    )
    # depth_x_excl: continuous composite replacing binary corpus_depth in a's weights.
    # Binary corpus_depth is near-1 for any payer with documents (not discriminative).
    # Multiplying by exclusivity scales the bonus to how SPECIFIC the topic match is:
    # pool=25 (narrow topic) → 1.0; pool=276 (broad dental) → 0.18; pool=600 → 0.08.
    depth_x_excl = corpus_depth_binary * exclusivity
    # tag_coverage: only reward a for NARROW tags (exclusivity ≥ 0.20 = pool ≤ 250 docs).
    tag_coverage = min(1.0, n_d_tags / 3.0) if exclusivity >= 0.20 else 0.0
    return {
        "exclusivity":     exclusivity,
        "literal":         float(bool(profile_features.get("has_literal", False))),
        "corpus_depth":    corpus_depth_binary,   # kept for b's weight lookup (b uses binary)
        "depth_x_excl":   depth_x_excl,           # a uses this composite instead
        "thematic_policy": float(bool(profile_features.get("thematic_policy", False))),
        "wide_pool":       float(pool_size > 500),
        "inheritance":     float(bool(profile_features.get("has_inherited_docs", False))),
        "crawlability":    float(profile_features.get("crawlability", 0.3)),
        "tag_coverage":    tag_coverage,
    }


def _linear_score_v2(sid: str, feats: dict[str, float]) -> float:
    base = _LINEAR_BASE_V2.get(sid, 0.0)
    weights = _LINEAR_WEIGHTS_V2.get(sid, {})
    return base + sum(w * feats.get(feat, 0.0) for feat, w in weights.items())


def decide(
    profile_features: dict[str, Any],
    prefs: RoutePreferences | None = None,
    *,
    fail_fast_reason: str | None = None,
    self_assessments: dict[str, tuple[float, str]] | None = None,
    prior_strategies_tried: list[str] | None = None,
) -> RouteDecision:
    """Pick a strategy (or strategy set) for this query.

    Identical call signature to corpus_search_router.decide(). Returns a
    RouteDecision; when invoke_all is set, the agent should run all listed
    strategies concurrently and union the results.

    Multi-invoke fires only on first-invocation (no prior_strategies_tried).
    On escalation re-invocations the router falls back to single next-best
    strategy so the ReAct loop still terminates.
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
    self_assessments = self_assessments or {}
    excluded_strategies: set[str] = set(prior_strategies_tried or [])

    scores: dict[str, float] = {}
    score_breakdown: dict[str, dict[str, Any]] = {}
    withdrawn: list[str] = []
    assessment_log: dict[str, dict[str, Any]] = {}

    feats = _compute_linear_features_v2(profile_features)

    for sid in ("a", "b", "c", "d"):
        if sid in excluded_strategies:
            withdrawn.append(sid)
            scores[sid] = 0.0
            score_breakdown[sid] = {
                "withdrawn": True, "withdraw_reason": "already_tried_in_thread",
            }
            continue

        if sid in self_assessments:
            est_recall, reason = self_assessments[sid]
            assessment_log[sid] = {
                "est_recall": round(est_recall, 3),
                "reason": reason,
            }
        else:
            est_recall = 1.0  # conservative: assume retrievable until proven otherwise
            assessment_log[sid] = {"est_recall": est_recall, "reason": "static_prior"}

        if est_recall < _WITHDRAW_RECALL_THRESHOLD:
            withdrawn.append(sid)
            scores[sid] = 0.0
            score_breakdown[sid] = {
                "withdrawn": True,
                "withdraw_reason": f"est_recall={est_recall:.3f} < {_WITHDRAW_RECALL_THRESHOLD}",
            }
            continue

        total = _linear_score_v2(sid, feats)
        scores[sid] = round(total, 4)
        score_breakdown[sid] = {
            "withdrawn": False,
            "linear_score_v2": round(total, 4),
            "features": feats,
        }

    ranked = sorted(
        ((s, sc) for s, sc in scores.items() if s not in withdrawn),
        key=lambda kv: -kv[1],
    )

    if not ranked:
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

    primary = ranked[0][0]
    fallback = ranked[1][0] if len(ranked) > 1 else None

    # ── Multi-invoke on impure leaves ────────────────────────────────────────
    # Fire only on first-pass (no prior_strategies_tried) so escalation
    # re-invocations stay single-strategy and the ReAct loop terminates.
    invoke_all: list[str] | None = None
    if not prior_strategies_tried and len(ranked) >= 2:
        top_score = ranked[0][1]
        second_score = ranked[1][1]
        gap = top_score - second_score
        if gap < IMPURE_GAP_THRESHOLD and top_score >= MIN_INVOKE_SCORE and second_score >= MIN_INVOKE_SCORE:
            invoke_all = [ranked[0][0], ranked[1][0]]
            logger.info(
                "[router_v2] multi-invoke gap=%.4f strategies=%s top=%.4f second=%.4f",
                gap, invoke_all, top_score, second_score,
            )

    logger.info(
        "[router_v2] decide qclass=%s scores=%s withdrawn=%s primary=%s fallback=%s invoke_all=%s",
        qclass, scores, withdrawn, primary, fallback, invoke_all,
    )

    decision = RouteDecision(
        strategy=primary,
        fallback=fallback,
        routing_method="deterministic",
        query_class=qclass,
        scores=scores,
        prefs_resolved=prefs_dump,
        priors_version=PRIORS_VERSION,
        invoke_all=invoke_all,
    )
    setattr(decision, "self_assessments", assessment_log)
    setattr(decision, "withdrawn", withdrawn)
    setattr(decision, "score_breakdown", score_breakdown)
    return decision
