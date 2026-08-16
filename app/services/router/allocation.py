"""GREEDY allocation mode — sequential fallback chains, per-slot LB enforcement.

One of Router's two production allocators (dispatch.py picks the executor,
the other runs as shadow):
  greedy    — THIS module: static-order sequential fallback chains
  optimizer — optimizer.py: exact per-slot subset solve

KNOWN LIMITATION OF GREEDY: the fixed chain order means strategy `b` only
executes when `a` failed — greedy outcomes are selection-biased samples;
calibration treats them as a secondary signal vs the A/B shadow comparison.

EXECUTION MODEL (architect-approved, weakest-link):
- Slots run in PARALLEL: query latency = MAX over slots, not sum.
- Within a slot, the fallback chain runs SEQUENTIALLY (Observer fails an
  attempt -> orchestrator advances to the next rung). Worst-case slot latency
  is the SUM of the chain's strategy latencies.
- Time budget applies PER SLOT (each slot's chain must fit the wall-clock
  budget), NOT deducted globally across slots. (Spec §3 amended 2026-07-23.)

CONFIDENCE MODEL (expected value over the fallback chain):
  P(slot success | chain [s1..sn]) = 1 - PRODUCT(1 - p_i)
  Two parallel tracks:
    MEAN track — p_i = recall_lift (TELEMETRY ONLY)
    LB track   — p_i = wilson_lower_bound(recall_lift, n, confidence_level)
  The LB track is THE enforced quantity (§2a addendum). Composing per-rung
  one-sided LBs multiplicatively is conservative (joint coverage > nominal
  level under independence) — documented v1 approximation; Monte Carlo over
  Beta posteriors deferred.

PER-QUESTION GATE (§2a addendum 2026-07-23, corrected: per QUESTION, not
mechanically per slot — SUPERSEDES the aggregate-mean gate):
  Every REQUIRED slot (required=True — the user's actual question or a genuine
  FAN_OUT sub-question) must independently satisfy LB(chain) >= adjusted bar.
  NO cross-slot compensation — the former Phase-2 top-up is REMOVED.
  OPTIONAL slots (required=False — e.g. RELY_ON_EXTERNAL's external_context,
  CLARIFY_REPHRASE's best_guess: supplementary/fallback BY DESIGN) are still
  filled and traced, but carry status OPTIONAL — achieved confidence reported,
  no pass/fail verdict, and they never gate the outcome.
  Aggregate mean is telemetry only.
  Required-slot status: CLEARED | UNDER_CONFIDENT (best-effort chain, bar
  unreachable, binding constraint recorded) | NO_VIABLE_STRATEGY.
  Ladder outcome (from required slots only): all_slots_cleared |
  partial_infeasible | no_slots — the explicit infeasibility signal (never
  silently swallowed; Chat/Synthesis decide honest-no-answer vs
  ask-for-relaxation downstream).

TOLERANCE BANDS: ±15% real-time (chat.default, real_time), ±25% background
  (chat.thinking, batch, background), else tolerance_bands.max_pct -> 0.25.

ACCURACY: expected accuracy reported as telemetry, NOT a feasibility gate yet
  (seed accuracies are fact-check-scale; gating awaits empirical calibration).

V1 SIMPLIFICATIONS (documented): strategy independence (P(b|!a)=P(b),
  slightly optimistic); static chain order s,a,b,c,d,f (cheap/fast first);
  conservative LB composition (above).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.services.router.priors import (
    PriorsBundle,
    StrategyProfile,
    compute_depth_bucket,
    load_priors,
    lookup_priors,
    wilson_lower_bound,
)
from app.services.router.tracing import DecisionTrace, SlotTrace, StrategyStep

logger = logging.getLogger(__name__)

# Cheap/fast first. Commutative for success probability; minimizes expected latency.
# Fillers `e`/`q` are DELIBERATELY absent (Eval-ratified 2026-07-23): terminal
# dispatch outcomes of upstream verdicts, never allocator candidates — see the
# boundary note on priors.STRATEGY_IDS. Do not add them here. `f` is
# RETIRED (Eval 2026-07-23): unscored-output shape can't contribute recall
# lift — see the retirement note on STRATEGY_IDS.
STRATEGY_PRIORITY_ORDER = ["s", "a", "b", "c", "d"]

# Strategy eligibility by slot ROLE (Ananth's catch 2026-07-23: an
# external_context slot is candidate-filtered to source_type web/external —
# internal-only strategies (s=field short-circuit, a=BM25 pool, b=vector pool,
# c=LLM->validate-against-own-corpus) cannot produce a matching candidate for
# it, so their priors are answering a meaningless question for that role).
# external_context -> d ONLY (Eval's default ruling; c excluded unless someone
# argues corpus-validation framing on external claims).
# Unknown semantics -> full set (don't over-restrict on unknown).
_ALL_STRATEGIES = frozenset(STRATEGY_PRIORITY_ORDER)
STRATEGY_ELIGIBILITY: dict[str, frozenset] = {
    "direct_answer": _ALL_STRATEGIES,
    "thematic_exploration": _ALL_STRATEGIES,
    "best_guess": _ALL_STRATEGIES,
    "external_context": frozenset({"d"}),
}


def eligible_strategies(slot_semantics: str) -> frozenset:
    """Strategies allowed to serve a slot of this role."""
    return STRATEGY_ELIGIBILITY.get(slot_semantics, _ALL_STRATEGIES)


# SECOND eligibility dimension — TAG-GATING (Eval-ratified 2026-07-23):
# strategy `s` calls the Payor Fact Store and is gated on a payor
# jurisdiction code; on non-payor queries it is a guaranteed clean miss.
# Signal source: GateResult.j_codes (shape/contracts.py — codes carry no kind
# prefix, e.g. "payor.sunshine_health"), threaded to Router via the
# query-level posture context (gate_j_codes). FAIL CLOSED: absent/empty
# j_codes → tag-gated strategies ineligible — never plan a payor call without
# evidence of payor-ness. Consequence for calibration: s's priors cells mean
# P(success | payor tag present) — clean by construction.
TAG_GATED_STRATEGIES: dict[str, str] = {"s": "payor."}  # strategy -> required j_code prefix


def _normalize_gate_code(code: str) -> str:
    """Strip an optional kind prefix ("j:", "d:", "p:") from a Gate code.

    LIVE-EVIDENCE FIX 2026-07-23: contracts.py claims "codes carry no kind
    prefix" but the lexicon's full_code INCLUDES it (legacy corpus_search.py
    strips "j:" before use; Retriever's first live run showed
    d_codes=["d:claims.timely_filing", ...]). Without normalization the payor
    predicate failed closed on every REAL payor query — s and the sitemap
    helper silently never planned. Accept both forms; never trust the comment
    over the wire format again."""
    c = str(code)
    return c[2:] if len(c) > 2 and c[1] == ":" and c[0] in ("j", "d", "p") else c


def has_payor_code(j_codes: Optional[list]) -> bool:
    """Shared payor-identity predicate (s tag-gate + sitemap helper gate)."""
    return bool(j_codes) and any(
        _normalize_gate_code(c).startswith("payor.") for c in j_codes
    )


def strategy_tag_eligible(strategy_id: str, j_codes: Optional[list]) -> bool:
    """True if the strategy's tag gate (if any) is satisfied by the query's j_codes."""
    prefix = TAG_GATED_STRATEGIES.get(strategy_id)
    if prefix is None:
        return True
    return bool(j_codes) and any(
        _normalize_gate_code(c).startswith(prefix) for c in j_codes
    )


# THIRD eligibility dimension — CRAWLABILITY. BUILT 2026-07-23, DISABLED
# 2026-07-24 (Ananth via Retriever) — kept as dormant machinery, not deleted,
# because the call is explicitly "for now."
#
# ORIGINAL premise: strategy `d` is near-guaranteed weak against a payor
# whose site is affirmatively NON-crawlable, gated on fillers/payer_context.py's
# tri-state verdict (`payer_crawlable: True|False|None`), fail-OPEN on None.
#
# WHY DISABLED (Ananth's correction, live-trace-driven, 2026-07-24): the
# premise conflates two different questions. `payer_crawlable` measures
# whether OUR fetcher can reach the payor's OWN domain directly — but d is a
# general web search (Vertex+DDG; verified in filler_d.py: only USES a
# `site:domain` operator when a site_domain happens to be present, otherwise
# searches broadly) that can surface THIRD-PARTY sources — cached pages,
# provider bulletins, law-firm summaries — discussing that payor even when
# the payor's own site blocks direct scraping. If anything, a genuinely
# non-crawlable payor site is exactly when general web search matters MOST,
# not when it should be excluded. Confirmed on a live query (Sunshine Health,
# payer_crawlable=False) where the gate was silently starving d, currently
# the strongest-recall arm per Eval's forced-arm calibration (0.67 marginal).
#
# CONSEQUENCE FOR EVAL: the priors comment this replaced said d's SINGLE
# cell per depth means "P(success | not affirmatively non-crawlable)" — that
# population definition is now WRONG going forward: d executes on
# crawlable=False queries too, broadening the population the cell describes.
# Flagged to Eval 2026-07-24; the seed cell's semantics need re-stating (not
# a value change — do-not-fold still stands on VALUES; this is a MEANING
# correction), and any future empirical d cell must not silently inherit the
# old population assumption.
#
# REVERSIBILITY: re-enable by restoring `frozenset({"d"})`; strategy_crawl_eligible
# and the whole eligibility-check call site are untouched, so this is a
# one-line flip either direction, not a design fork.
CRAWL_GATED_STRATEGIES = frozenset()


def strategy_crawl_eligible(strategy_id: str, payer_crawlable) -> bool:
    """False only when the strategy is crawl-gated AND crawlable is
    affirmatively False. True/None keep the strategy eligible (fail open)."""
    if strategy_id not in CRAWL_GATED_STRATEGIES:
        return True
    return payer_crawlable is not False

_REAL_TIME_MODES = {"chat.default", "real_time"}
_BACKGROUND_MODES = {"chat.thinking", "batch", "background"}

# UX-signed tolerance bands (2026-07-23): named constants, not magic numbers.
TOLERANCE_REAL_TIME = 0.15
TOLERANCE_BACKGROUND = 0.25

_SPEED_BUDGET_MS = {
    "real_time": 2000,
    "interactive": 5000,
    "background": 60000,
    "none": 10**9,
}

# CALLER-MODE-SCOPED latency allowance override (Ananth 2026-08-05, via
# Retriever): d's real measured latency (9732ms attempt_ms) needs the
# CUMULATIVE chain latency through s+a+b+c+d (13832ms at today's seed
# values — the chain tries cheap-fast-first and d is last) to clear the
# per-rung budget check; the generic interactive formula (5000*1.25=6250ms)
# structurally excludes d in every mode, confirmed via the real 3-mode
# allocator run for cmhc001 (all 3 modes skip d on over_latency_allowance).
# Scoped NARROWLY to chat.thinking, per explicit instruction — real_time
# modes (chat.copilot/chat.default) stay tight. NOT a bump to the shared
# "interactive" speed_budget_ms value: that string is also used by
# auth_agent (Structure's caller_mode table), which must stay untouched.
# This map is checked BEFORE the speed_budget*tolerance formula in
# resolve_constraints and, when present, IS the final allowance (no
# further multiplier) — simplest way to hit a specific cumulative-latency
# target without touching the shared formula's other callers. Reversible:
# delete the chat.thinking entry, falls back to the standard formula.
#
# EXPERIMENT (2026-08-15, Ananth's ask): chat.default's real_time 2000ms
# budget was found to be the actual binding constraint behind its bank
# recall (0.6265) landing nearly identical to chat.copilot's (0.6264),
# despite default's target breadth (Structure's _BASE_K*recall_demand)
# being nominally equal to chat.thinking's -- default never got enough
# time to act on that breadth. Testing an intermediate allowance (6000ms,
# 3x real_time, well under thinking's 16000ms) to see where recall/
# latency land between copilot/default's ~0.63 and thinking's ~0.85.
# Paired with a matching _TOKEN_BUDGET bump in structure.py so the extra
# chunks this buys aren't immediately trimmed back out by Synthesis.
CALLER_MODE_LATENCY_ALLOWANCE_OVERRIDE_MS = {
    "chat.thinking": 16000,  # clears cumulative-through-d (13832ms) with ~16% headroom
    "chat.default": 6000,  # EXPERIMENT: intermediate step between real_time (2000) and thinking's 16000
}

# Optional (required=False) slots are supplementary by design: they get ONE
# cheap attempt, not a bar-chase — chasing a bar they never gate on burns real
# Fillers latency/compute for telemetry-only value (Eval's efficiency catch,
# 2026-07-23: an optional slot's 4-rung chain doubled worst-case query latency).
OPTIONAL_SLOT_MAX_ATTEMPTS = 1

# Per-slot statuses / ladder outcomes (§2a addendum, per-QUESTION gating)
STATUS_CLEARED = "CLEARED"
STATUS_UNDER_CONFIDENT = "UNDER_CONFIDENT"
STATUS_NO_VIABLE = "NO_VIABLE_STRATEGY"
STATUS_OPTIONAL = "OPTIONAL"     # required=False slot: reported, never gated
OUTCOME_ALL_CLEARED = "all_slots_cleared"
OUTCOME_PARTIAL_INFEASIBLE = "partial_infeasible"
OUTCOME_NO_SLOTS = "no_slots"

# TERMINAL LEG (Ananth's rule, 2026-07-23): e/q have NO priors and never
# compete in the chain math (ratified boundary above) — but Router ATTACHES
# them as the verdict-driven FINAL LEG of the plan. Rule-based, deterministic,
# zero priors: a required slot ending UNDER_CONFIDENT gets the clarify
# terminal (asking the user is the one move that can rescue an
# under-confident answer); NO_VIABLE_STRATEGY gets the fast-exit terminal
# (nothing to retrieve, nothing to ask about). CLEARED/OPTIONAL get none.
#
# NAMING (Retriever/Eval directive 2026-07-23, before Contract/Synthesis
# formalize the field): values are self-disambiguating strings, NOT the bare
# filler ids "q"/"e" — those collided semantically with Shape's own
# DECLINE/CLARIFY_REPHRASE stage (different mechanism: Shape fires pre-Fillers
# on query ambiguity; this fires post-allocation on weak evidence). Mapping to
# fillers: clarify_low_confidence → dispatches Filler q; fast_exit_no_viable →
# dispatches Filler e. ("no_viable_strategy" was rejected as the terminal
# value: it would shadow the STATUS_NO_VIABLE status string one field over.)
TERMINAL_CLARIFY = "clarify_low_confidence"
TERMINAL_FAST_EXIT = "fast_exit_no_viable"

# HELPER LAYER (Ananth, 2026-07-23): the recall loop (a/b/c/d/s, scored,
# priors-driven) is one machine; HELPERS are the "what do you do when the
# recall loop fails" layer — non-recall user-pain-solvers, ZERO priors, never
# in the chain math (the ratified boundary holds):
#   clarify_low_confidence — ask the user for the missing detail (Filler q)
#   sitemap_links          — hand the user the payer's real pages that likely
#                            contain the answer we couldn't confidently
#                            extract (Sitemap module: discovered_sources
#                            links-only, no fetch — lightest possible aid).
#                            GATED on payor identity (it queries
#                            discovered_sources BY PAYER — without a payor
#                            j_code there is nothing to look up).
# fast_exit remains the honest terminal formatter when NO helper applies.
HELPER_SITEMAP = "sitemap_links"


def slot_helper_plan(status: str, j_codes: Optional[list],
                     d_codes: Optional[list] = None) -> list[str]:
    """Verdict → ordered helper plan for one slot (pure rules, no priors).

    UNDER_CONFIDENT: clarify first (asking can rescue the answer), plus
    sitemap links when the aid can actually produce something.
    NO_VIABLE: sitemap links if producible; otherwise nothing — fast_exit
    terminal handles it. CLEARED / OPTIONAL: no helpers.

    sitemap_links PRECONDITIONS (from Sitemap's real lookup contract,
    2026-07-23): payer identity (payor j_code) AND at least one d-code —
    lookup_sitemap_links() requires BOTH a payer_display_name and a matching
    d:-tag topic; missing either deterministically returns []. Router's gate
    is the cheap pre-filter (payor + any-d-code present); the keyword-table
    match happens inside Sitemap, so the ORCHESTRATOR still reconciles at
    execution: an empty suggested_links[] drops the helper from the final
    verdict (pairing invariant, build-spec)."""
    sitemap_producible = has_payor_code(j_codes) and bool(d_codes)
    if status == STATUS_UNDER_CONFIDENT:
        return [TERMINAL_CLARIFY] + ([HELPER_SITEMAP] if sitemap_producible else [])
    if status == STATUS_NO_VIABLE:
        return [HELPER_SITEMAP] if sitemap_producible else []
    return []


def slot_terminal_action(status: str) -> Optional[str]:
    """Verdict → terminal leg for one slot (no priors — pure rule)."""
    if status == STATUS_UNDER_CONFIDENT:
        return TERMINAL_CLARIFY
    if status == STATUS_NO_VIABLE:
        return TERMINAL_FAST_EXIT
    return None  # CLEARED and OPTIONAL: normal synthesis, no terminal leg


def decision_helpers(per_slot_helpers: dict[str, list]) -> list:
    """Decision-level helper plan: canonical-ordered union (clarify, then sitemap)."""
    seen = set()
    for plan in per_slot_helpers.values():
        seen.update(plan)
    canon = [TERMINAL_CLARIFY, HELPER_SITEMAP]
    return [h for h in canon if h in seen]


def decision_terminal_action(per_slot_terminal: dict[str, Optional[str]]) -> Optional[str]:
    """Decision-level terminal: clarify wins over fast-exit (if ANY slot has
    something worth asking about, ask; only exit fast when nothing does)."""
    actions = set(a for a in per_slot_terminal.values() if a)
    if TERMINAL_CLARIFY in actions:
        return TERMINAL_CLARIFY
    if TERMINAL_FAST_EXIT in actions:
        return TERMINAL_FAST_EXIT
    return None


# THE REAL SLOT CONTRACT — imported from Slots (Step 1d), never re-declared.
# P0 seam bug fixed 2026-07-23 (Retriever's fleet inspection): this module
# previously defined its own AnswerSlot stand-in carrying two fields the real
# contract does not have (`query_class`, `max_attempts`), so every test passed
# against the fake while real AnswerShapeResult.slots would have raised
# AttributeError on first contact. Reconciliation:
#   - max_attempts is QUERY-level (ResourcePosture.max_attempts) — it now
#     flows through resolve_constraints, not per-slot. (Future gap, noted not
#     built: FAN_OUT may one day want a per-slot split of the query budget —
#     that needs a Slots-side field first.)
#   - query_class existed nowhere in the Shape chain (legacy holdover from the
#     pre-refactor classifier) — the qclass priors-fallback tier is DORMANT
#     until the new pipeline provides an equivalent signal; the fallback is
#     now the depth-2 default profile directly.
from app.services.retriever.shape.slots import AnswerSlot, AnswerShapeResult  # noqa: F401


@dataclass
class RoutingLadder:
    """Output of an allocator: plan per slot + estimates + per-slot verdicts.

    Produced by BOTH allocators — identical shape, so executed vs shadow plans
    are directly comparable. `per_slot_lb`/`per_slot_status`/`outcome` carry
    the §2a per-slot enforcement result; `feasible` is outcome ==
    all_slots_cleared (kept for continuity)."""
    per_slot: dict[str, list[str]] = field(default_factory=dict)
    per_slot_confidence: dict[str, float] = field(default_factory=dict)  # MEAN (telemetry)
    per_slot_lb: dict[str, float] = field(default_factory=dict)          # ENFORCED
    per_slot_status: dict[str, str] = field(default_factory=dict)        # CLEARED | UNDER_CONFIDENT | NO_VIABLE_STRATEGY | OPTIONAL
    per_slot_terminal: dict[str, Optional[str]] = field(default_factory=dict)  # verdict-driven final leg
    terminal_action: Optional[str] = None    # decision-level (clarify wins over fast-exit)
    per_slot_helpers: dict[str, list] = field(default_factory=dict)  # recall-failure helper plan per slot
    helpers: list = field(default_factory=list)  # decision-level ordered-unique helper plan
    per_slot_accuracy: dict[str, float] = field(default_factory=dict)
    per_slot_latency_ms: dict[str, int] = field(default_factory=dict)
    per_slot_payload_tokens: dict[str, int] = field(default_factory=dict)  # worst case (max rung)
    total_payload_tokens: int = 0         # sum over slots (worst-case synthesis input)
    # blend model (design doc §8): portfolio allocators fill {strategy: k_i};
    # chain allocators leave this empty — per_slot stays the compat view
    per_slot_portfolio: dict[str, dict[str, int]] = field(default_factory=dict)
    allocator: str = ""                   # "greedy" | "optimizer" | "" (forced)
    outcome: str = ""                     # all_slots_cleared | partial_infeasible | no_slots
    total_estimated_ms: int = 0           # MAX over slots (parallel-slot model)
    total_estimated_cost: float = 0.0     # worst-case: sum of all rungs
    aggregate_confidence_estimate: float = 0.0  # mean-track mean — TELEMETRY ONLY
    aggregate_accuracy_estimate: float = 0.0
    adjusted_confidence_bar: float = 0.0
    tolerance_pct: float = 0.25
    feasible: bool = False
    infeasibility_reason: str = ""


def chain_success_probability(lifts: list[float]) -> float:
    """P(success) of independent rungs: 1 - PRODUCT(1 - p_i)."""
    fail = 1.0
    for p in lifts:
        fail *= (1.0 - max(0.0, min(1.0, p)))
    return 1.0 - fail


def chain_expected_accuracy(profiles: list[StrategyProfile]) -> float:
    """Expected accuracy of whichever rung succeeds, probability-weighted (mean track)."""
    prefix_fail = 1.0
    weighted = 0.0
    total_p = 0.0
    for prof in profiles:
        p = max(0.0, min(1.0, prof.recall_lift))
        weighted += prefix_fail * p * prof.accuracy_estimate
        total_p += prefix_fail * p
        prefix_fail *= (1.0 - p)
    return (weighted / total_p) if total_p > 0 else 0.0


def resolve_tolerance_pct(resource_posture: dict[str, Any]) -> float:
    """Caller-mode-dependent tolerance: ±15% real-time, ±25% background/default."""
    mode = resource_posture.get("caller_mode")
    if mode in _REAL_TIME_MODES:
        return TOLERANCE_REAL_TIME
    if mode in _BACKGROUND_MODES:
        return TOLERANCE_BACKGROUND
    bands = resource_posture.get("tolerance_bands") or {}
    try:
        return float(bands.get("max_pct", TOLERANCE_BACKGROUND))
    except (TypeError, ValueError):
        return TOLERANCE_BACKGROUND


def _speed_budget_to_ms(speed_budget: str) -> int:
    return _SPEED_BUDGET_MS.get(speed_budget, 5000)


# ---------------------------------------------------------------------------
# AUTHORITY GATE — caller-declared citability (Ananth 2026-07-23: "the
# challenge with d is it is not authoritative, not quotable for healthcare...
# accurate from a website but not citable to a payor. Why can't the caller
# tell us if this is important — for an appeal this is important, but for a
# chat call?").
#
# Bifurcation is CALLER-DECLARED, not Router-guessed: posture field
# `authority_requirement`:
#   "any"              (default) — d competes on recall merits. FAIL-OPEN:
#                       zero behavior change until a caller declares.
#   "citable_required" — non-citable strategies are ineligible for REQUIRED
#                       slots (evidence-bearing). Optional/external_context
#                       slots KEEP d — web context is still useful context
#                       even when it can't serve as citable evidence, and
#                       optional slots never gate the outcome (§2a).
# Seam status: Router-side knob built first (concrete target); caller-side
# declaration being engaged with Chat, then Retriever/Structure for the
# ResourcePosture threading (same path token_budget is taking).
# ---------------------------------------------------------------------------
NON_CITABLE_STRATEGIES = frozenset({"d"})  # live web: accurate ≠ citable to a payor

# Per-strategy authority PRIOR threshold (Eval's 2026-08-05 proposal): a
# strategy is ALSO ineligible under citable_required if its measured
# `authority` prior sits below this bar. ADDITIVE to NON_CITABLE_STRATEGIES
# above, never a replacement — an unpopulated priors file (authority
# defaults to 1.0) changes nothing, so this activates only once Eval folds
# real per-strategy authority measurements.
# EVAL-RATIFIED 2026-08-05: 0.6 (was a 0.5 seed placeholder). c's measured
# authority band is 0.48-0.52 — at 0.5 with `>=`, c would have PASSED
# (0.50 >= 0.50) and wrongly survived citable_required slots. 0.6 excludes
# c cleanly and sits clear above its whole band, so n=1 wobble can't flip
# it. Caller-gated (only bites under citable_required) — does not preempt
# Ananth's pending "any"-default A/B ruling, just makes the existing
# caller-declared path correct for c once real values land.
AUTHORITY_CITABLE_THRESHOLD = 0.6

# SUPPLEMENT-ONLY strategies (Eval's category-error finding, cmhc002 collapse
# 2026-07-23): s (Payor Fact Store) returns ONE certified fact — a code, not
# answer content. It supplements corpus retrieval; it must never SUBSTITUTE
# for it. Rule (both allocators): s may not be the SOLE planned rung of a
# REQUIRED slot while any other strategy is viable — a one-shot ladder
# betting everything on a bare fact-store code is how 19/22 bank queries
# returned 0 chunks. s in multi-rung chains, or as the only viable option,
# is unchanged. NOTE: the priors file has no slot dimension (cells are
# depth × strategy), so this is the ONLY expressible home for the rule.
SUPPLEMENT_ONLY_STRATEGIES = frozenset({"s"})
AUTHORITY_ANY = "any"
AUTHORITY_CITABLE_REQUIRED = "citable_required"


def strategy_authority_eligible(strategy_id: str, authority_requirement: str,
                                slot_required: bool,
                                authority: float = 1.0) -> bool:
    """False when a non-citable strategy would fill an evidence-bearing
    (required) slot under a caller-declared citability requirement.

    Two OR'd checks: the legacy hardcoded classification (NON_CITABLE_
    STRATEGIES) and the per-strategy authority prior vs threshold. `authority`
    defaults to 1.0 (fully authoritative) so callers that don't pass a real
    prior — or profiles with no measured authority yet — see no new
    exclusions beyond the legacy set."""
    if authority_requirement != AUTHORITY_CITABLE_REQUIRED:
        return True
    if not slot_required:
        return True  # context, not evidence — d stays useful here
    if strategy_id in NON_CITABLE_STRATEGIES:
        return False
    return authority >= AUTHORITY_CITABLE_THRESHOLD


# ---------------------------------------------------------------------------
# Payload/token budget (Ananth 2026-07-23: "keep track of tokens you can
# send — if it extracts a whole provider manual that's no use"). Router plans
# must be token-feasible BY CONSTRUCTION, not just time-feasible.
#
# Chain payload is NOT additive UNDER CURRENT LIVE SEMANTICS: only the
# winning rung's chunks fill the slot (advance-on-empty discards failed
# rungs' output), so the gate is PER RUNG — worst-case slot payload for a
# rung = slot.capacity × per-chunk tokens — and the slot's reported worst
# case is the MAX over chosen rungs.
#
# PLANNED FLIP (Ananth via Retriever, 2026-07-23): the ruling is RETAIN —
# superseded rungs' outputs will be kept for a future Synthesis step to
# combine across rungs. When the retention mechanism ships (gated on
# Synthesis kickoff — retention without a consumer is pure token liability),
# this model changes MAX→SUM and the continuation decision gains a
# remaining-token-budget axis. Do NOT flip early: SUM under today's discard
# behavior would overstate payload and cause false payload skips. The MAX
# model here is correct exactly as long as the orchestrator discards.
#
# Per-chunk worst-case tokens by strategy (≈4 chars/token):
#   d: VERIFIED — filler_d.py caps passages at _MAX_PASSAGE_CHARS=2000 ≈ 500.
#   a/b/c: MEASURED (Retriever, real corpus, 2026-07-24): length(text)
#     p50=34 chars, avg≈214, p95=919 ≈ 230 tokens → 250 with margin.
#     HISTORY — the original 1000 "conservative guess" was ~4× real p95 and
#     DETERMINISTICALLY gated a/b/c off every capacity-10 slot (10,000
#     demanded vs token_budget 3000, and vs the 8000 default too): the true
#     mechanism behind 19/22 empty bank queries. Eval's estimate-warning
#     materialized exactly as predicted; the WARNING fired into unmonitored
#     logs. Lesson: an estimate in a skip gate must be validated against a
#     REAL capacity × budget combination before shipping, not just marked.
#   s: ESTIMATE — fact-store rows are compact structured facts.
# Flagged to Eval: these belong in the priors file per (strategy) cell,
# same swap-without-redeploy contract as recall_lift.
# ---------------------------------------------------------------------------
PAYLOAD_TOKENS_PER_CHUNK = {"a": 250, "b": 250, "c": 250, "d": 500, "s": 150}
_PAYLOAD_TOKENS_UNKNOWN_STRATEGY = 500  # fail-conservative for unlisted ids
# Eval's guard (2026-07-23): an ESTIMATE driving a real skip gate can silently
# mis-skip. Skips based on UNMEASURED per-chunk values log a WARNING.
# a/b/c/d now measured; only s remains an estimate.
ESTIMATED_PAYLOAD_STRATEGIES = frozenset({"s"})
DEFAULT_TOKEN_ALLOWANCE_PER_SLOT = 8000  # generous: no behavior change at defaults


def warn_if_estimate_driven_skip(strategy_id: str, capacity: int,
                                 token_allowance: int) -> None:
    """Loud, not silent: payload skips keyed on unmeasured estimates."""
    if strategy_id in ESTIMATED_PAYLOAD_STRATEGIES:
        logger.warning(
            "router payload gate skipped '%s' (capacity=%d, allowance=%d) based "
            "on an UNMEASURED per-chunk estimate — measure corpus chunk sizes "
            "before trusting this gate at tight allowances",
            strategy_id, capacity, token_allowance,
        )


def rung_payload_tokens(strategy_id: str, capacity: int) -> int:
    """Worst-case tokens this rung delivers if it wins the slot."""
    per_chunk = PAYLOAD_TOKENS_PER_CHUNK.get(strategy_id, _PAYLOAD_TOKENS_UNKNOWN_STRATEGY)
    return max(0, int(capacity)) * per_chunk


def chain_payload_tokens(chain: list[str], capacity: int) -> int:
    """Worst-case slot payload: MAX over rungs (single winner fills the slot)."""
    return max((rung_payload_tokens(sid, capacity) for sid in chain), default=0)


def resolve_constraints(resource_posture: dict[str, Any]) -> dict[str, Any]:
    """Shared constraint resolution for greedy + optimizer modes.

    `gate_j_codes` (query-level Gate signal, same precedent as caller_mode)
    rides the posture context and drives tag-gated strategy eligibility."""
    tolerance_pct = resolve_tolerance_pct(resource_posture)
    speed_budget_ms = _speed_budget_to_ms(resource_posture.get("speed_budget", "interactive"))
    confidence_bar = float(resource_posture.get("confidence_bar", 0.85))
    allocator_override_ms = CALLER_MODE_LATENCY_ALLOWANCE_OVERRIDE_MS.get(
        resource_posture.get("caller_mode"))
    latency_allowance_ms = (
        float(allocator_override_ms) if allocator_override_ms is not None
        else speed_budget_ms * (1.0 + tolerance_pct))
    return {
        "tolerance_pct": tolerance_pct,
        "speed_budget_ms": speed_budget_ms,
        "confidence_bar": confidence_bar,
        "adjusted_bar": confidence_bar * (1.0 - tolerance_pct),
        "latency_allowance_ms": latency_allowance_ms,
        "j_codes": resource_posture.get("gate_j_codes") or [],
        "d_codes": resource_posture.get("gate_d_codes") or [],
        # SOFT SAFETY CEILING only (Ananth via Structure, 2026-07-23):
        # attempts are DERIVED from the time budget (the chain grows while it
        # fits latency_allowance_ms — that IS time_budget ÷ per-strategy
        # latency); max_attempts is a "never exceed N regardless of speed"
        # cap, min()'d on top — never the plan itself.
        "max_attempts": int(resource_posture.get("max_attempts_per_slot", 6)),
        # tri-state payor crawlability (fillers/payer_context.py verdict);
        # gates strategy d — fail-open on None (see CRAWL_GATED_STRATEGIES)
        "payer_crawlable": resource_posture.get("payer_crawlable"),
        # per-slot payload budget (tokens the slot may deliver downstream);
        # generous default = accounting + guardrail, not a behavior change.
        # None-tolerant: the route() bridge passes ResourcePosture.token_budget
        # verbatim, which defaults to None
        "token_allowance_per_slot": int(
            resource_posture.get("token_allowance_per_slot")
            or DEFAULT_TOKEN_ALLOWANCE_PER_SLOT),
        # caller-declared citability (appeal vs casual chat); fail-open default
        "authority_requirement": str(resource_posture.get(
            "authority_requirement", AUTHORITY_ANY)),
        # Which call number this is in Chat's retry loop (2026-08-07,
        # portfolio's cost gate). None/missing fails CLOSED to 1 (first
        # call) -- see RoutingContext.call_number's docstring.
        "call_number": int(resource_posture.get("call_number") or 1),
        # caller_mode passthrough (2026-08-07, portfolio's turn-floor
        # bypass: thinking mode + authority=any waives the c/d turn floors
        # from round 1). Already on the raw posture dict elsewhere in this
        # module; just wasn't in THIS returned dict yet.
        "caller_mode": resource_posture.get("caller_mode"),
    }


def lookup_with_fallback(
    depth_bucket: int, strategy_id: str, bundle: PriorsBundle,
) -> tuple[Optional[StrategyProfile], str]:
    """Priors lookup with tier attribution for tracing.

    Fallback tier is the depth-2 (moderate) default profile. The qclass tier
    is DORMANT — Eval-ratified 2026-07-23: no replacement signal will be
    built (the tier's bootstrap purpose is already served by the file's full
    depth-bucket coverage); the fallback_qclass_priors table awaits a future
    cleanup pass. Reactivation would need a real upstream signal + Eval
    sign-off on tier semantics."""
    prof = lookup_priors(depth_bucket, strategy_id, bundle)
    if prof is not None:
        return prof, "depth_bucket"
    prof = bundle.by_depth.get(strategy_id, {}).get(2)
    if prof is not None:
        return prof, "default_depth2_fallback"
    return None, "none"


def chain_lb(profiles: list[StrategyProfile], confidence_level: float,
             lb_fn=wilson_lower_bound) -> float:
    """Chain lower-bound confidence: compose per-rung lower bounds.

    THE shared LB helper — single implementation used by ALL allocators
    (greedy chain state + both optimizers' subset enumeration). `lb_fn`
    defaults to Wilson; the Bayesian optimizer passes beta_lower_bound —
    the ONLY thing that differs between the two optimizers, so any A/B/C
    divergence is attributable to the bound alone. Conservative v1
    composition (see module docstring)."""
    return chain_success_probability([
        lb_fn(p.recall_lift, p.n, confidence_level) for p in profiles
    ])


def slot_status(chain: list[str], lb: float, adjusted_bar: float,
                required: bool = True) -> str:
    """§2a per-QUESTION verdict. Optional slots report, never pass/fail."""
    if not required:
        return STATUS_OPTIONAL
    if not chain:
        return STATUS_NO_VIABLE
    return STATUS_CLEARED if lb >= adjusted_bar else STATUS_UNDER_CONFIDENT


def ladder_outcome(statuses: dict[str, str]) -> str:
    """§2a decision-level outcome — gated on REQUIRED slots only.

    A query with slots but zero required slots is vacuously cleared (nothing
    the user asked is being gated); an empty slot map is no_slots."""
    if not statuses:
        return OUTCOME_NO_SLOTS
    required_statuses = [s for s in statuses.values() if s != STATUS_OPTIONAL]
    if all(s == STATUS_CLEARED for s in required_statuses):
        return OUTCOME_ALL_CLEARED
    return OUTCOME_PARTIAL_INFEASIBLE


@dataclass
class _SlotState:
    slot: AnswerSlot
    depth_bucket: int
    trace: SlotTrace
    confidence_level: float
    max_attempts: int = 6  # SOFT ceiling; attempts derive from the time budget
    payer_crawlable: object = None  # True | False | None (tri-state)
    token_allowance: int = DEFAULT_TOKEN_ALLOWANCE_PER_SLOT
    authority_requirement: str = AUTHORITY_ANY
    j_codes: list = field(default_factory=list)
    chain: list[str] = field(default_factory=list)
    profiles: list[StrategyProfile] = field(default_factory=list)
    latency_ms: int = 0
    _skips_recorded: set = field(default_factory=set)

    def _lb(self, prof: StrategyProfile) -> float:
        return wilson_lower_bound(prof.recall_lift, prof.n, self.confidence_level)

    @property
    def confidence(self) -> float:
        """Mean-track chain confidence (telemetry)."""
        return chain_success_probability([p.recall_lift for p in self.profiles])

    @property
    def lb(self) -> float:
        """Lower-bound-track chain confidence (THE enforced quantity)."""
        return chain_lb(self.profiles, self.confidence_level)  # shared helper

    @property
    def accuracy(self) -> float:
        return chain_expected_accuracy(self.profiles)


def _record_skip(state: _SlotState, strategy_id: str, reason: str,
                 prof: Optional[StrategyProfile], source: str) -> None:
    key = (strategy_id, reason)
    if key in state._skips_recorded:
        return
    state._skips_recorded.add(key)
    step = StrategyStep(strategy_id=strategy_id, action="skipped", skip_reason=reason)
    if prof is not None:
        step.prior_source = source
        step.recall_lift = prof.recall_lift
        step.n = prof.n
        step.lb_lift = state._lb(prof)
        step.latency_p50_ms = prof.latency_p50_ms
        step.cost = prof.cost
        step.accuracy_estimate = prof.accuracy_estimate
    state.trace.steps.append(step)


def _effective_max_attempts(slot: AnswerSlot, query_max_attempts: int) -> int:
    """Query-level attempts budget; optional slots capped at one cheap attempt."""
    if slot.required:
        return query_max_attempts
    return min(query_max_attempts, OPTIONAL_SLOT_MAX_ATTEMPTS)


def _next_viable_strategy(
    state: _SlotState, latency_allowance_ms: float, bundle: PriorsBundle,
) -> Optional[tuple[str, StrategyProfile, str]]:
    """Next rung selection; records skips.

    BEST-LB-FIRST (Ananth 2026-08-05, confidence-density over latency) —
    REQUIRED slots only: every rung picks the highest per-rung Wilson LB
    among viable strategies, using the priors already loaded for this
    depth_bucket — not a static priority order. Supersedes the prior
    cheap-fast-first default for required slots (was: first eligible
    strategy in STRATEGY_PRIORITY_ORDER wins, with a LAST-ATTEMPT-only
    best-LB exception — see git history). Priority order now only breaks
    exact LB ties. REAL TRADEOFF, accepted: a high-LB but slow strategy
    (e.g. d, ~9.7s) can now be tried before a fast-but-lower-LB one (e.g.
    a, ~0.5s) — some queries get noticeably slower in exchange for closer
    oracle-matching. Live-verified motivation: the old fixed order made
    citable_required a structural no-op in common cases (chain never went
    deep enough to reach where it would exclude anything) and missed the
    true best strategy on real bucket-3 data where it wasn't cheapest.

    OPTIONAL slots are UNCHANGED and orthogonal to this directive: they
    still take the first viable rung in cheap-fast-first order and stop
    (Eval's efficiency ruling — telemetry-only slots never pay wall-clock
    for confidence). See the `if not state.slot.required` early-return
    inside the loop below.
    SUPPLEMENT_ONLY gate still applies on the sole/first rung (below)."""
    effective_max = _effective_max_attempts(state.slot, state.max_attempts)
    if len(state.chain) >= effective_max:
        return None
    last_attempt = state.slot.required and (effective_max - len(state.chain) == 1)
    viable: list[tuple[str, StrategyProfile, str]] = []
    allowed = eligible_strategies(state.slot.slot_semantics)
    for strategy_id in STRATEGY_PRIORITY_ORDER:
        if strategy_id in state.chain:
            continue
        if strategy_id not in allowed:
            _record_skip(state, strategy_id,
                         f"ineligible_for_{state.slot.slot_semantics}", None, "")
            continue
        if not strategy_tag_eligible(strategy_id, state.j_codes):
            _record_skip(state, strategy_id, "tag_gated_no_payor_j_code", None, "")
            continue
        if not strategy_crawl_eligible(strategy_id, state.payer_crawlable):
            _record_skip(state, strategy_id, "crawl_gated_payer_not_crawlable", None, "")
            continue
        prof, source = lookup_with_fallback(
            state.depth_bucket, strategy_id, bundle
        )
        if prof is None:
            _record_skip(state, strategy_id, "no_prior", None, source)
            continue
        if not strategy_authority_eligible(strategy_id, state.authority_requirement,
                                           state.slot.required, prof.authority):
            _record_skip(state, strategy_id, "authority_gated_non_citable", prof, source)
            continue
        if prof.recall_lift <= 0.0:
            _record_skip(state, strategy_id, "zero_or_negative_recall_lift", prof, source)
            continue
        if state.latency_ms + prof.latency_p50_ms > latency_allowance_ms:
            _record_skip(state, strategy_id, "over_latency_allowance", prof, source)
            continue
        # PARTIAL-FILL model (Ananth 2026-07-24: "take 3 of d" beats no d):
        # a rung is payload-viable if even ONE chunk is affordable — the
        # actual per-rung fill is assigned jointly post-chain
        # (portfolio.assign_chain_fills: diversity floor + value knapsack,
        # Σ fills·tokens ≤ budget by construction — the SUM model, live now
        # that the orchestrator RETAINS across rungs). The old all-or-nothing
        # gate (capacity × per-chunk > budget → skip) was cause #5 of the
        # live d-starvation: it excluded d entirely at capacity 10 × budget
        # 3000 when d at fill 6 fits exactly.
        per_chunk = PAYLOAD_TOKENS_PER_CHUNK.get(
            strategy_id, _PAYLOAD_TOKENS_UNKNOWN_STRATEGY)
        if per_chunk > state.token_allowance:
            warn_if_estimate_driven_skip(strategy_id, state.slot.capacity,
                                         state.token_allowance)
            _record_skip(state, strategy_id, "payload_over_token_allowance", prof, source)
            continue
        viable.append((strategy_id, prof, source))
        if not state.slot.required:
            # OPTIONAL slots keep cheap-fast-first, UNCHANGED (Eval's
            # ruling, orthogonal to the best-LB-first directive above):
            # telemetry-only slots never pay wall-clock for confidence,
            # so they take the first viable rung and stop — no LB
            # comparison, no SUPPLEMENT_ONLY gate (matches the pre-existing
            # behavior exactly, which never reached that block either).
            return strategy_id, prof, source
    if not viable:
        return None
    # SUPPLEMENT GATE on the sole/first rung: at tied LBs (e.g. depth_1:
    # s=a=.35) the tie-break would still hand the shot to s — a bare
    # fact-store code for a content question. If this is the slot's FIRST
    # rung (empty chain), supplement-only strategies are excluded while
    # anything else is viable (see SUPPLEMENT_ONLY_STRATEGIES).
    if not state.chain:
        non_supplement = [c for c in viable if c[0] not in SUPPLEMENT_ONLY_STRATEGIES]
        if non_supplement:
            for sid, prof, source in viable:
                if sid in SUPPLEMENT_ONLY_STRATEGIES:
                    _record_skip(state, sid, "supplement_only_not_sole_rung", prof, source)
            viable = non_supplement
    best = max(viable, key=lambda c: (state._lb(c[1]), -STRATEGY_PRIORITY_ORDER.index(c[0])))
    for sid, prof, source in viable:
        if sid != best[0]:
            _record_skip(state, sid, f"single_shot_lower_lb_than_{best[0]}", prof, source)
    return best


def _binding_constraint(state: _SlotState) -> str:
    """Why the chain stopped short of the bar — recorded on UNDER_CONFIDENT slots."""
    if not state.slot.required:
        return "optional_capped"
    if len(state.chain) >= state.max_attempts:
        return "attempts_exhausted"
    if any(r == "over_latency_allowance" for (_, r) in state._skips_recorded):
        return "budget_exhausted"
    if any(r == "payload_over_token_allowance" for (_, r) in state._skips_recorded):
        return "payload_budget_exhausted"
    return "strategies_exhausted"


def _add_strategy(state: _SlotState, strategy_id: str, prof: StrategyProfile,
                  source: str, ladder: RoutingLadder) -> None:
    conf_before = state.confidence
    lb_before = state.lb
    lat_before = state.latency_ms
    state.chain.append(strategy_id)
    state.profiles.append(prof)
    state.latency_ms += prof.latency_p50_ms
    ladder.total_estimated_cost += prof.cost
    state.trace.steps.append(StrategyStep(
        strategy_id=strategy_id, action="added", prior_source=source,
        recall_lift=prof.recall_lift, n=prof.n, lb_lift=state._lb(prof),
        latency_p50_ms=prof.latency_p50_ms,
        cost=prof.cost, accuracy_estimate=prof.accuracy_estimate,
        conf_before=conf_before,
        added_term=(1.0 - conf_before) * prof.recall_lift,
        conf_after=state.confidence,
        lb_before=lb_before, lb_after=state.lb,
        latency_before_ms=lat_before, latency_after_ms=state.latency_ms,
    ))


def allocate_strategies(
    slots: list[AnswerSlot],
    pool_metadata: dict[str, dict],
    resource_posture: dict[str, Any],
    bundle: Optional[PriorsBundle] = None,
    trace: Optional[DecisionTrace] = None,
) -> RoutingLadder:
    """Greedy sequential-fallback allocation, per-slot LB gate (see module docstring)."""
    ladder = RoutingLadder(allocator="greedy")
    bundle = bundle if bundle is not None else load_priors()
    trace = trace if trace is not None else DecisionTrace()
    confidence_level = float(bundle.exploration_policy.get("confidence_level", 0.95))

    c = resolve_constraints(resource_posture)
    tolerance_pct = c["tolerance_pct"]
    adjusted_bar = c["adjusted_bar"]
    latency_allowance_ms = c["latency_allowance_ms"]

    ladder.tolerance_pct = tolerance_pct
    ladder.adjusted_confidence_bar = adjusted_bar

    trace.caller_mode = resource_posture.get("caller_mode", "")
    trace.tolerance_pct = tolerance_pct
    trace.confidence_bar = c["confidence_bar"]
    trace.adjusted_confidence_bar = adjusted_bar
    trace.speed_budget_ms = int(c["speed_budget_ms"])
    trace.latency_allowance_ms = latency_allowance_ms
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

    states = []
    for s in sorted(slots, key=lambda s: s.priority):
        meta = pool_metadata.get(s.slot_id, {})
        st = _SlotState(
            slot=s,
            depth_bucket=compute_depth_bucket(meta),
            confidence_level=confidence_level,
            max_attempts=c["max_attempts"],
            payer_crawlable=c["payer_crawlable"],
            token_allowance=c["token_allowance_per_slot"],
            authority_requirement=c["authority_requirement"],
            j_codes=c["j_codes"],
            trace=SlotTrace(
                slot_id=s.slot_id, priority=s.priority,
                required=s.required,
                slot_semantics=s.slot_semantics,
                pool_size=meta.get("pool_size"),
                top_score_percentile=meta.get("top_score_percentile"),
                distinct_content_topk=meta.get("distinct_content_topk"),
                depth_bucket=compute_depth_bucket(meta),
                phase="per_slot_lb",
            ),
        )
        states.append(st)
        trace.slots.append(st.trace)

    # Single phase: each slot independently chases the LB bar. No top-up —
    # cross-slot compensation is prohibited by §2a.
    for st in states:
        while st.lb < adjusted_bar:
            nxt = _next_viable_strategy(st, latency_allowance_ms, bundle)
            if nxt is None:
                break
            _add_strategy(st, nxt[0], nxt[1], nxt[2], ladder)

    # PARTIAL-FILL assignment (post-chain, per slot): jointly scope each
    # rung's fill to the token budget — diversity floor seats every member,
    # remainder by value. Fills are the SUM-model budget enforcement
    # (retention live: every executed rung's chunks union). Members whose
    # floor is unaffordable drop from the chain with a trace reason.
    from app.services.router.portfolio import assign_chain_fills
    for st in states:
        if not st.chain:
            continue
        members = [(sid_, prof_, st.slot.capacity)
                   for sid_, prof_ in zip(st.chain, st.profiles)]
        fills = assign_chain_fills(members, c["token_allowance_per_slot"])
        dropped = [sid_ for sid_ in st.chain if fills.get(sid_, 0) < 1]
        if dropped:
            keep = [(sid_, prof_) for sid_, prof_ in zip(st.chain, st.profiles)
                    if sid_ not in dropped]
            for sid_ in dropped:
                _record_skip(state=st, strategy_id=sid_,
                             reason="budget_fill_zero_after_assignment",
                             prof=None, source="")
            st.chain = [sid_ for sid_, _ in keep]
            st.profiles = [prof_ for _, prof_ in keep]
            st.latency_ms = sum(p.latency_p50_ms for p in st.profiles)
        st.fills = {sid_: fills[sid_] for sid_ in st.chain}

    # Assemble ladder + per-slot verdicts (per-QUESTION: required slots gated,
    # optional slots reported)
    for st in states:
        sid = st.slot.slot_id
        status = slot_status(st.chain, st.lb, adjusted_bar, required=st.slot.required)
        ladder.per_slot[sid] = st.chain
        ladder.per_slot_confidence[sid] = st.confidence
        ladder.per_slot_lb[sid] = st.lb
        ladder.per_slot_status[sid] = status
        ladder.per_slot_terminal[sid] = slot_terminal_action(status)
        ladder.per_slot_helpers[sid] = slot_helper_plan(status, c["j_codes"], c["d_codes"])
        ladder.per_slot_accuracy[sid] = st.accuracy
        ladder.per_slot_latency_ms[sid] = st.latency_ms
        fills = getattr(st, "fills", {}) or {}
        if fills:
            ladder.per_slot_portfolio[sid] = dict(fills)
        # SUM model: worst case = every rung runs and its fill is retained
        ladder.per_slot_payload_tokens[sid] = sum(
            k * PAYLOAD_TOKENS_PER_CHUNK.get(s_, _PAYLOAD_TOKENS_UNKNOWN_STRATEGY)
            for s_, k in fills.items())

        st.trace.payload_tokens_worst_case = ladder.per_slot_payload_tokens[sid]
        st.trace.final_chain = list(st.chain)
        st.trace.final_confidence = st.confidence
        st.trace.final_lb = st.lb
        st.trace.final_latency_ms = st.latency_ms
        st.trace.bar_cleared = st.lb >= adjusted_bar
        st.trace.status = status
        st.trace.terminal_action = ladder.per_slot_terminal[sid]
        st.trace.stop_reason = (
            "bar_cleared" if st.lb >= adjusted_bar else _binding_constraint(st)
        )

    ladder.total_estimated_ms = max(ladder.per_slot_latency_ms.values(), default=0)
    ladder.total_payload_tokens = sum(ladder.per_slot_payload_tokens.values())
    ladder.aggregate_confidence_estimate = (
        sum(st.confidence for st in states) / len(states)
    )
    ladder.aggregate_accuracy_estimate = sum(st.accuracy for st in states) / len(states)

    parts = " + ".join(f"{st.lb:.4f}" for st in states)
    trace.aggregate_confidence = ladder.aggregate_confidence_estimate
    trace.aggregate_arithmetic = (
        f"per-slot LBs [{parts}] vs bar {adjusted_bar:.4f} (mean is telemetry only)"
    )

    ladder.outcome = ladder_outcome(ladder.per_slot_status)
    ladder.terminal_action = decision_terminal_action(ladder.per_slot_terminal)
    ladder.helpers = decision_helpers(ladder.per_slot_helpers)
    ladder.feasible = ladder.outcome == OUTCOME_ALL_CLEARED
    if not ladder.feasible:
        # only REQUIRED slots can fail the query — optional ones never gate
        failed = {
            st.slot.slot_id: f"{ladder.per_slot_status[st.slot.slot_id]} ({st.trace.stop_reason})"
            for st in states
            if ladder.per_slot_status[st.slot.slot_id] in
            (STATUS_UNDER_CONFIDENT, STATUS_NO_VIABLE)
        }
        ladder.infeasibility_reason = (
            f"required slots below LB bar {adjusted_bar:.3f} at level {confidence_level:.2f}: {failed}"
        )

    trace.outcome = ladder.outcome
    trace.helpers = list(ladder.helpers)
    trace.feasible = ladder.feasible
    trace.infeasibility_reason = ladder.infeasibility_reason
    return ladder
