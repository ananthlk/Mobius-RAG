"""Priors infrastructure — corpus-depth bucketing + file-loaded strategy profiles.

PRIORS SOURCE OF TRUTH (in order):
  1. YAML file (primary): `eval/priors_bootstrap.yaml` — Eval-owned artifact.
     Eval's calibration loop rewrites this file (or a successor) with empirical
     priors in Week 2-3. Router picks up changes WITHOUT a code change or redeploy
     (mtime-based cache invalidation on every lookup path).
     Override the path with env var ROUTER_PRIORS_PATH.
  2. Hardcoded fallback (last resort): _HARDCODED_FALLBACK below — used ONLY if
     the file is missing or unparseable, with a warning log. Never the primary.

Corpus-depth bucketing (Pool signal → bucket 0-4):
  Bucket 0: tight        (top_score_percentile >= 0.90 AND pool_size < 50)
  Bucket 1: tight-mod    (top_score_percentile >= 0.75 AND pool_size < 200)
  Bucket 2: moderate     (top_score_percentile >= 0.50 AND pool_size < 500)
  Bucket 3: broad-mod    (top_score_percentile >= 0.25 AND pool_size < 5000)
  Bucket 4: broad        (everything else, incl. missing metadata)

Sanitization: recall_lift and accuracy_estimate clamped to [0, 1]; latency and
cost clamped to >= 0. Malformed per-strategy entries are dropped (not crashed on).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# DELIBERATE ARCHITECTURAL BOUNDARY (Eval-ratified 2026-07-23, NOT an oversight):
# the priors universe is exactly these SIX retrieval strategies. Fillers `e`
# (Fast Exit) and `q` (Clarify) are PERMANENTLY EXCLUDED — they are terminal
# dispatch outcomes of upstream verdicts (q executes Reformat's CLARIFY
# posture; e executes Router's §2a infeasibility outcome), not retrieval
# attempts with a recall/cost/latency tradeoff. The chain math 1-Π(1-lift)
# assumes each rung independently attempts the slot's need; e/q represent NOT
# attempting it — modeling them here would be a category error. Their
# calibration question ("did Gate/Reformat judge unanswerability correctly")
# belongs to the Shape layer, not this table. Do not "complete" this roster.
#
# `f` RETIRED (Eval-ratified 2026-07-23, third phantom-lift instance): the
# seeded "f" cell ("external fallback similar to d") was placeholder data
# authored before the real Sitemap module's shape settled — that module emits
# unscored suggested_links[] (never a scored FilledChunk), so it structurally
# CANNOT produce a recall_lift contribution; legacy StrategyId never had "f"
# either. Different reason than e/q (output-shape incompatibility with the LB
# chain math, not terminal-dispatch nature), same boundary treatment. Seeded
# YAML cells are archived-for-provenance in Eval's file; the parser ignores
# them (STRATEGY_IDS filter below). Reinstate only with a real, SCORED
# strategy design + kickoff.
STRATEGY_IDS = ("a", "b", "c", "d", "s")
DEPTH_BUCKETS = (0, 1, 2, 3, 4)

# Per-strategy defaults used when the qclass fallback table (success-rate only)
# needs latency/cost to build a full profile.
_DEFAULT_STRATEGY_LATENCY_MS = {"a": 500, "b": 1500, "c": 2000, "d": 3000, "s": 100}
_DEFAULT_STRATEGY_COST = {"a": 1.0, "b": 1.0, "c": 2.0, "d": 3.0, "s": 0.0}


SEED_PSEUDO_COUNT = 8  # hand-set seed cells count as n=8 observations (locked w/ Eval 2026-07-23)

# Capacity a cell's recall_lift was observed at (the capacity transform's k₀,
# blend-model doc §3). PER-CELL, not global — Eval's audit (2026-07-24):
# a/b/d calibrate at occupancy ~10, but c and s rarely exceed 1; a global 10
# would badly misrebase c/s (treating 1-chunk recall as if 10 produced it).
# Today's cells are ALL seeds (no explicit n in the file → pseudo-n=8,
# hand-set values never measured at ANY capacity), so this default is a
# CONVENTION on designed numbers — the transform is INERT-until-real-
# calibration: it activates per cell only when Eval's empirical writer emits
# (recall_lift, n, k0) together from a real run.
K0_NOMINAL = 10


@dataclass(frozen=True)
class StrategyProfile:
    """Profile for a strategy at a given corpus depth.

    recall_lift/accuracy_estimate are MEANS; n is the sample count behind them
    (SEED_PSEUDO_COUNT for hand-set seeds; real N once Eval's empirical writer
    emits it — mandatory per the uncertainty co-design). Allocators enforce the
    Wilson LOWER BOUND of recall_lift, never the raw mean."""
    recall_lift: float        # [0, 1] — P(this strategy satisfies the slot), MEAN
    latency_p50_ms: int       # milliseconds
    cost: float               # relative cost units
    accuracy_estimate: float  # [0, 1], MEAN
    n: int = SEED_PSEUDO_COUNT  # sample count behind the estimates
    k0: int = K0_NOMINAL      # capacity recall_lift was observed at (per-cell;
                              # mandatory companion to n from the empirical writer)
    authority: float = 1.0    # [0, 1] — P(strategy's evidence is citable to a
                              # payor), MEAN. Default 1.0 = fully authoritative
                              # (no data yet ⇒ no new exclusions). Feeds
                              # allocation.strategy_authority_eligible as an
                              # ADDITIONAL gate alongside the legacy hardcoded
                              # NON_CITABLE_STRATEGIES set — additive only, so
                              # an unpopulated file changes nothing (Eval's
                              # 2026-08-05 per-strategy-prior proposal).


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz's method,
    Numerical Recipes betacf). Deterministic, stdlib-only."""
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) — CDF of Beta(a, b) at x. Stdlib-only (math.lgamma)."""
    import math

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


# Jeffreys smoothing for the Beta prior: alpha0 = p*n + 0.5, beta0 = (1-p)*n + 0.5.
# NEEDED, not cosmetic: several seed cells sit at recall_lift exactly 0 (c at
# depth_2/depth_4) where a raw Beta(0, n) is degenerate. Jeffreys is the
# standard reference prior for a binomial proportion. Flagged to Eval for
# ratification with the rest of the Bayesian-allocator modeling choices.
JEFFREYS_SMOOTHING = 0.5


def beta_lower_bound(p_hat: float, n: int, confidence_level: float = 0.95) -> float:
    """One-sided lower bound from the Beta posterior quantile.

    Beta(alpha, beta) with alpha = p*n + 0.5, beta = (1-p)*n + 0.5 (Jeffreys
    smoothing); returns the (1 - confidence_level) quantile via bisection on
    the regularized incomplete beta. The Bayesian-allocator counterpart of
    wilson_lower_bound — same signature so allocators can swap bound functions.
    """
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, p_hat))
    level = max(0.5, min(0.9999, confidence_level))
    a = p * n + JEFFREYS_SMOOTHING
    b = (1.0 - p) * n + JEFFREYS_SMOOTHING
    target = 1.0 - level  # lower-tail mass
    lo, hi = 0.0, 1.0
    for _ in range(80):  # ~1e-24 interval; plenty for 1e-9 comparisons
        mid = (lo + hi) / 2.0
        if regularized_incomplete_beta(a, b, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def wilson_lower_bound(p_hat: float, n: int, confidence_level: float = 0.95) -> float:
    """One-sided Wilson score lower bound for a [0,1] proportion.

    Chosen over Jeffreys/Beta (locked w/ Eval): closed-form, stdlib-only,
    well-behaved at p_hat=0/1 boundaries where several seed cells sit.
    z = Phi^-1(confidence_level) (one-sided). Returns 0.0 for n<=0."""
    from statistics import NormalDist

    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, p_hat))
    z = NormalDist().inv_cdf(max(0.5, min(0.9999, confidence_level)))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * ((p * (1.0 - p) / n + z2 / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - margin) / denom)


# A/B policy defaults (2026-07-23 addendum: dual-build shadow A/B — both
# allocators compute a ladder every query; ab_split_optimizer is the fraction
# of queries where the OPTIMIZER's ladder executes, greedy's otherwise; the
# untaken plan is logged as a shadow decision). Eval overrides via an
# `exploration_policy:` block in the priors YAML — same swap-without-redeploy
# contract as the priors themselves.
DEFAULT_EXPLORATION_POLICY = {
    # Fallback ONLY — priors_bootstrap.yaml's exploration_policy.allocator_weights
    # is what's actually live and always wins when present (Eval-owned,
    # code-free cutover). This equal-thirds default was the original
    # "serve all three, compare via executed outcomes" bootstrap philosophy
    # (Ananth's third-allocator directive) -- SUPERSEDED 2026-08-06 by
    # validate-then-serve (greedy pinned to 1.0 in the YAML after clearing
    # a forced eval-bank matrix; optimizer/bayesian must clear the same bar
    # before their weight rises above 0). Left as-is here only as a safe
    # bootstrap-time default for a fresh deploy with no YAML override yet --
    # not a statement of current policy.
    "allocator_weights": {"greedy": 1 / 3, "optimizer": 1 / 3, "bayesian": 1 / 3},
    "confidence_level": 0.95,  # one-sided level for the LB (locked w/ Eval)
    "phase": "bootstrap",
}


@dataclass
class PriorsBundle:
    """A loaded, sanitized set of priors + provenance."""
    by_depth: dict[str, dict[int, StrategyProfile]]        # strategy -> depth -> profile
    qclass_fallback: dict[str, dict[str, float]] = field(default_factory=dict)  # strategy -> qclass -> rate
    exploration_policy: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_EXPLORATION_POLICY))
    version: str = "unknown"
    source: str = "unknown"  # "file" | "hardcoded"


# ---------------------------------------------------------------------------
# Hardcoded fallback (LAST RESORT ONLY — the YAML file is the source of truth;
# these values are a snapshot and may lag the file)
# ---------------------------------------------------------------------------
_HARDCODED_FALLBACK: dict[str, dict[int, StrategyProfile]] = {
    "a": {
        0: StrategyProfile(0.100, 500, 1, 0.165),
        1: StrategyProfile(0.350, 500, 1, 0.263),
        2: StrategyProfile(0.543, 500, 1, 0.500),
        3: StrategyProfile(0.800, 500, 1, 0.425),
        4: StrategyProfile(0.315, 500, 1, 0.313),
    },
    "b": {
        0: StrategyProfile(0.100, 1500, 1, 0.225),
        1: StrategyProfile(0.175, 1500, 1, 0.206),
        2: StrategyProfile(0.286, 1500, 1, 0.225),
        3: StrategyProfile(0.480, 1500, 1, 0.284),
        4: StrategyProfile(0.765, 1500, 1, 0.450),
    },
    "c": {
        0: StrategyProfile(0.071, 2000, 2, 0.180),
        1: StrategyProfile(0.300, 2000, 2, 0.151),
        2: StrategyProfile(0.330, 2000, 2, 0.000),
        3: StrategyProfile(0.400, 2000, 2, 0.091),
        4: StrategyProfile(0.765, 2000, 2, 0.000),
    },
    "d": {
        0: StrategyProfile(0.150, 3000, 3, 0.350),
        1: StrategyProfile(0.330, 3000, 3, 0.400),
        2: StrategyProfile(0.450, 3000, 3, 0.450),
        3: StrategyProfile(0.550, 3000, 3, 0.500),
        4: StrategyProfile(0.700, 3000, 3, 0.550),
    },
    "s": {
        0: StrategyProfile(0.100, 100, 0, 0.400),
        1: StrategyProfile(0.350, 100, 0, 0.450),
        2: StrategyProfile(0.500, 100, 0, 0.500),
        3: StrategyProfile(0.600, 100, 0, 0.550),
        4: StrategyProfile(0.750, 100, 0, 0.600),
    },
}

# Backwards-compatible alias (old name). Points at the fallback snapshot only.
PRIORS_SEED_DEFAULTS = _HARDCODED_FALLBACK


# ---------------------------------------------------------------------------
# File loading (primary source)
# ---------------------------------------------------------------------------

def default_priors_path() -> Path:
    """Resolve the priors YAML path: ROUTER_PRIORS_PATH env var, else eval/priors_bootstrap.yaml."""
    env = os.environ.get("ROUTER_PRIORS_PATH")
    if env:
        return Path(env)
    # .../mobius-rag/app/services/router/priors.py -> .../mobius-rag/eval/priors_bootstrap.yaml
    return Path(__file__).resolve().parents[3] / "eval" / "priors_bootstrap.yaml"


def _clamp01(v: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _nonneg(v: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(v))
    except (TypeError, ValueError):
        return default


def _parse_priors_yaml(raw: dict[str, Any], version: str) -> PriorsBundle:
    """Parse + sanitize the priors YAML structure into a PriorsBundle.

    Expected shape (Eval's priors_bootstrap.yaml):
      seed_priors:
        depth_0: {a: {recall_lift, latency_p50_ms, cost_per_attempt, accuracy_estimate, ...}, ...}
        ...
      fallback_qclass_priors:
        a: {tight_pool: 0.5, ...}
    """
    by_depth: dict[str, dict[int, StrategyProfile]] = {}
    seed = raw.get("seed_priors") or {}
    for depth_key, strategies in seed.items():
        try:
            depth = int(str(depth_key).rsplit("_", 1)[-1])
        except ValueError:
            logger.warning("priors: skipping unrecognized depth key %r", depth_key)
            continue
        if depth not in DEPTH_BUCKETS or not isinstance(strategies, dict):
            continue
        for strategy_id, vals in strategies.items():
            if str(strategy_id) not in STRATEGY_IDS:
                # retired (f) or stray cells: archived-for-provenance in the
                # file, never loaded into the live universe
                continue
            if not isinstance(vals, dict):
                logger.warning("priors: skipping malformed entry %s/%s", depth_key, strategy_id)
                continue
            # FAIL-CLOSED VALIDATION (Eval-ratified 2026-07-24): explicit `n`
            # = a measurement claim; a measurement without its capacity is
            # malformed (silent k₀ rebase on the confidence path). Seeds have
            # no `n` key → harmless defaults. Guards against a writer bug or
            # a hand-edit adding n without k0 — the cell is REJECTED, falling
            # back like any malformed entry, never silently rebased.
            if "n" in vals and "k0" not in vals:
                logger.warning(
                    "priors: REJECTING %s/%s — explicit n without k0 "
                    "(empirical cells must carry the (recall_lift, n, k0) "
                    "triple; missing capacity would silently rebase the "
                    "capacity transform)", depth_key, strategy_id)
                continue
            try:
                n = max(1, int(vals.get("n", SEED_PSEUDO_COUNT)))
            except (TypeError, ValueError):
                n = SEED_PSEUDO_COUNT
            try:
                k0 = max(1, int(vals.get("k0", K0_NOMINAL)))
            except (TypeError, ValueError):
                k0 = K0_NOMINAL
            profile = StrategyProfile(
                recall_lift=_clamp01(vals.get("recall_lift")),
                latency_p50_ms=int(_nonneg(vals.get("latency_p50_ms"), 1000)),
                cost=_nonneg(vals.get("cost_per_attempt", vals.get("cost")), 1.0),
                accuracy_estimate=_clamp01(vals.get("accuracy_estimate")),
                n=n,
                k0=k0,
                authority=_clamp01(vals.get("authority", 1.0)),
            )
            by_depth.setdefault(str(strategy_id), {})[depth] = profile

    qclass: dict[str, dict[str, float]] = {}
    for strategy_id, classes in (raw.get("fallback_qclass_priors") or {}).items():
        if isinstance(classes, dict):
            qclass[str(strategy_id)] = {
                str(k): _clamp01(v) for k, v in classes.items()
            }

    if not by_depth:
        raise ValueError("priors file parsed to an empty seed_priors table")

    policy = dict(DEFAULT_EXPLORATION_POLICY)
    raw_policy = raw.get("exploration_policy")
    if isinstance(raw_policy, dict):
        raw_weights = raw_policy.get("allocator_weights")
        if isinstance(raw_weights, dict):
            weights = {
                str(k): _nonneg(v, 0.0)
                for k, v in raw_weights.items()
                # "portfolio" accepted so Eval's file can shift blend traffic
                # (code-free cutover, design doc §3); absent ⇒ shadow-only
                if str(k) in ("greedy", "optimizer", "bayesian", "portfolio")
            }
            total = sum(weights.values())
            if total > 0:
                policy["allocator_weights"] = {k: v / total for k, v in weights.items()}
        elif "ab_split_optimizer" in raw_policy:
            # legacy two-way knob: maps to optimizer-vs-greedy, no bayesian traffic
            split = _clamp01(raw_policy.get("ab_split_optimizer"), default=0.5)
            policy["allocator_weights"] = {"greedy": 1.0 - split, "optimizer": split}
        try:
            cl = float(raw_policy.get("confidence_level", policy["confidence_level"]))
            policy["confidence_level"] = max(0.5, min(0.9999, cl))
        except (TypeError, ValueError):
            pass
        policy["phase"] = str(raw_policy.get("phase", policy["phase"]))
        # DATA-COLLECTION THROTTLE knobs (Ananth's pullback, 2026-07-24):
        # forced_fraction = share of traffic sent to a forced-strategy arm
        # (prod: 1-in-5 = 0.2); forced_arms = list (uniform) or dict
        # (weighted, Eval steers toward under-sampled cells). All
        # file-swappable — activating collection is an Eval file edit.
        try:
            ff = float(raw_policy.get("forced_fraction", 0.0))
            policy["forced_fraction"] = max(0.0, min(1.0, ff))
        except (TypeError, ValueError):
            policy["forced_fraction"] = 0.0
        raw_arms = raw_policy.get("forced_arms")
        if isinstance(raw_arms, list):
            policy["forced_arm_weights"] = {
                str(a): 1.0 for a in raw_arms if str(a) in STRATEGY_IDS}
        elif isinstance(raw_arms, dict):
            policy["forced_arm_weights"] = {
                str(a): _nonneg(w, 0.0) for a, w in raw_arms.items()
                if str(a) in STRATEGY_IDS and _nonneg(w, 0.0) > 0.0}

    return PriorsBundle(by_depth=by_depth, qclass_fallback=qclass,
                        exploration_policy=policy, version=version, source="file")


def _hardcoded_bundle() -> PriorsBundle:
    return PriorsBundle(
        by_depth=_HARDCODED_FALLBACK,
        qclass_fallback={},
        version="hardcoded-fallback-2026-07-23",
        source="hardcoded",
    )


# mtime-keyed cache: (resolved path, mtime) -> PriorsBundle
_cache: dict[str, Any] = {"key": None, "bundle": None}


def load_priors(path: Optional[Path] = None, force_reload: bool = False) -> PriorsBundle:
    """Load priors from YAML (primary) with hardcoded fallback (last resort).

    Cache is invalidated when the file's mtime changes, so Eval rewriting the
    file is picked up on the next query without a redeploy.
    """
    import yaml  # local import: keep module importable even if PyYAML missing

    p = Path(path) if path is not None else default_priors_path()
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        logger.warning("priors: file %s missing — using HARDCODED FALLBACK (last resort)", p)
        return _hardcoded_bundle()

    cache_key = (str(p), mtime)
    if not force_reload and _cache["key"] == cache_key and _cache["bundle"] is not None:
        return _cache["bundle"]

    try:
        with open(p) as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError("priors file is not a mapping")
        bundle = _parse_priors_yaml(raw, version=f"file:{p.name}@{mtime}")
    except Exception as exc:
        logger.warning("priors: failed to load %s (%s) — using HARDCODED FALLBACK", p, exc)
        return _hardcoded_bundle()

    _cache["key"] = cache_key
    _cache["bundle"] = bundle
    logger.info("priors: loaded %s (%d strategies)", bundle.version, len(bundle.by_depth))
    return bundle


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def compute_depth_bucket(pool_metadata: dict) -> int:
    """Discretize Pool's corpus-depth signal into a bucket [0-4].

    DIVERSITY DEMOTION (junk-poisoning defense, 2026-07-23 — Filler b's
    verified finding): a high top score dominated by REPEATED content (e.g.
    23-char boilerplate at 0.797 cosine across the whole top-10) must not
    read as "tight, findable" — that misclassification selects the wrong
    priors row AND suppresses d-escalation for exactly the hollow-corpus
    queries that need it. Pool now supplies `distinct_content_topk` (count of
    distinct normalized-TEXT chunks in the top 10 — text, not content_sha,
    which is unreliable as a text-dedup key in this schema):
      distinct <= 2  -> bucket forced to >= 3 (top is essentially one or two
                        contents; the findability signal is untrustworthy)
      distinct <= 4  -> bucket forced to >= 2
      absent (None)  -> unchanged (fail-open compat until Pool supplies it)
    Thresholds flagged to Eval for ratification alongside the §6c formula."""
    top_score_pct = pool_metadata.get("top_score_percentile", 0.0) or 0.0
    pool_size = pool_metadata.get("pool_size", 0) or 0

    if top_score_pct >= 0.90 and pool_size < 50:
        bucket = 0
    elif top_score_pct >= 0.75 and pool_size < 200:
        bucket = 1
    elif top_score_pct >= 0.50 and pool_size < 500:
        bucket = 2
    elif top_score_pct >= 0.25 and pool_size < 5000:
        bucket = 3
    else:
        bucket = 4

    distinct = pool_metadata.get("distinct_content_topk")
    if distinct is not None:
        if distinct <= 2:
            bucket = max(bucket, 3)
        elif distinct <= 4:
            bucket = max(bucket, 2)
    return bucket


def lookup_priors(
    depth_bucket: int,
    strategy_id: str,
    bundle: Optional[PriorsBundle] = None,
) -> Optional[StrategyProfile]:
    """Look up (depth_bucket, strategy) profile from the loaded bundle."""
    b = bundle if bundle is not None else load_priors()
    return b.by_depth.get(strategy_id, {}).get(depth_bucket)


def lookup_priors_qclass_fallback(
    query_class: str,
    strategy_id: str,
    bundle: Optional[PriorsBundle] = None,
) -> Optional[StrategyProfile]:
    """Fallback: qclass-based priors (bootstrap path, no depth_bucket data yet).

    Uses the file's fallback_qclass_priors success rate as recall_lift with
    per-strategy default latency/cost. If the (strategy, qclass) cell is absent,
    fall back to the strategy's depth-2 (moderate) profile.
    """
    b = bundle if bundle is not None else load_priors()
    rate = b.qclass_fallback.get(strategy_id, {}).get(query_class)
    if rate is not None:
        return StrategyProfile(
            recall_lift=_clamp01(rate),
            latency_p50_ms=_DEFAULT_STRATEGY_LATENCY_MS.get(strategy_id, 1000),
            cost=_DEFAULT_STRATEGY_COST.get(strategy_id, 1.0),
            accuracy_estimate=0.0,  # unknown at qclass granularity
        )
    return b.by_depth.get(strategy_id, {}).get(2)


def get_priors_version(bundle: Optional[PriorsBundle] = None) -> str:
    b = bundle if bundle is not None else load_priors()
    return b.version
