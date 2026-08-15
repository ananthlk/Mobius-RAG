"""Structure — Step 1c of the shape module (Gate → Reformat → Structure).

Emits the Shape→Pool contract: `rewritten_queries[]` (passthrough from
Reformat), `posture` (carried forward), and the new `ResourcePosture`
(breadth/confidence_bar/speed_budget/token_budget/max_attempts) — how much
retrieval effort this specific query gets. Legacy `answer_shape` (essay/
structured/binary/any) is an answer-FORM hint for Synthesis/Chat and lives
on the original request only — Structure never sees or re-emits it.

Pure compute, zero DB calls (confirmed by DB sign-off). See
docs/rag-agents/shape-structure-schematic-spec.md for the full design
record and cross-agent sign-off (UX/Chat/Eval/DB/TECH, 2026-07-23).

Does not touch Gate's classification or Reformat's posture/fan-out logic.
Does not do Pool's actual corpus search.
"""

from __future__ import annotations

import time

from app.services.retriever.shape.contracts import (
    ReformatPosture,
    ReformatResult,
    ResourcePosture,
    StructureResult,
)

# Mirrored by hand from corpus_search_router.py's CALLER_MODE_PRESETS
# (accuracy_need / recall_demand / speed_budget only — answer_shape is out
# of scope here). NOT imported: Router is downstream of Pool, not a Shape
# dependency, and importing it would reach backward across a layer boundary
# that doesn't exist yet. Kept in sync by hand until/unless TECH proposes a
# shared source of truth. Verified live 2026-07-23, corpus_search_router.py:118-159.
#
# INTERIM 2026-08-11 (Eval-RAG, Ananth's directive to cut slack now rather
# than wait for the full calibration pass): chat.copilot/chat.default/
# chat.thinking dropped from 0.70/0.85/0.95 to 0.40/0.45/0.55 -- anchored
# to the cold-start achievable-gate ceiling (~0.46, the most a strong full
# 3-strategy portfolio retrieval reaches right NOW, at seed n under the
# uncalibrated 0.7 same-strategy decay constant; verified live, reconciles
# to a real reported gate of 0.4633). Real root cause diagnosed same
# session: 5/5 real production queries -- all single-topic, non-multi-part
# -- exhausted the full 3-round call_number retry budget before this
# change, confirming the OLD bars sat above what the gate could ever
# produce, not that retrieval itself was weak. Set just below that ceiling
# so a strong single-round retrieval clears with margin (stops needless
# escalation) while a genuinely weak one (gate ~0.20-0.25) still escalates
# (bar isn't dead, just reachable). Mode ordering preserved (copilot <
# default < thinking).
#
# NOT the calibrated value -- do not treat as final. Real fix = per-
# strategy same-strategy-decay fit from forced-arm recall@k data (the 0.7
# decay constant is itself an uncalibrated guess, confirmed the dominant
# suppressor) + bars re-anchored to production reference-free faithfulness
# grading, both queued as Eval-RAG's next-session resume with real budget.
# As production n accumulates the achievable gate rises toward ~0.59 even
# under the current decay, and a properly-fit (likely milder) decay lifts
# it further -- these interim bars will likely need to go UP once that
# lands, not down. auth_agent/research/batch left untouched -- Eval-RAG's
# interim scoped explicitly to copilot/default/thinking only.
_ACCURACY_NEED = {
    "chat.copilot": 0.40,
    "chat.default": 0.45,
    "chat.thinking": 0.55,
    "auth_agent": 1.00,
    "research": 0.95,
    "batch": 0.90,
}
_RECALL_DEMAND = {
    "chat.copilot": 0.70,
    "chat.default": 0.95,
    "chat.thinking": 0.95,
    "auth_agent": 0.95,
    "research": 1.00,
    "batch": 0.80,
}
_SPEED_BUDGET = {
    "chat.copilot": "real_time",
    "chat.default": "real_time",
    "chat.thinking": "interactive",
    "auth_agent": "interactive",
    "research": "none",
    "batch": "background",
}
# Per-slot token ceiling — first-cut hand-set values, grounded in Filler d's
# live measurement (a full d-slot at capacity=5 measured ~2,021 tokens, not
# guessed) and the codebase's existing per-passage cap (corpus_search_
# strategy_d.py: _MAX_PASSAGE_CHARS=2000 chars × _MAX_FETCH=5, unbounded in
# total until now). Tracks speed_budget's urgency tier, NOT accuracy_need —
# auth_agent is tentative (accuracy_need=1.00 there is about precision on a
# binary answer_shape, not evidence volume; flagged open in the schematic
# spec addendum §11, not resolved here). See docs/rag-agents/
# shape-structure-schematic-spec.md §11.
_TOKEN_BUDGET = {
    "chat.copilot": 2000,
    "chat.default": 3000,
    # Bumped 6000 -> 16000 (2026-08-15, Ananth's call): a live comparison
    # against Anthropic's own contextual-retrieval benchmarking (their
    # cookbook -- top_k=20 @ ~800 tokens/chunk, their most performant
    # tested config) put chat.thinking well below what Anthropic's own
    # data showed working best, while chat.default/research already sit
    # in-band with industry norms (72Technologies' 2000-4000 typical
    # chat-turn range; research already ~matches Anthropic's ~16-18k
    # figure). This ONLY affects Synthesis's per-slot MMR/trim ceiling
    # (_resolve_resource_posture below, token_budget field) -- NOT
    # retrieval breadth/pool width (driven by recall_demand/_BASE_K) or
    # the speed_budget/latency allowance, so raising it lets more already-
    # retrieved content survive synthesis for this mode without changing
    # how much gets retrieved or how long retrieval itself takes.
    "chat.thinking": 16000,
    "auth_agent": 2500,
    "research": 20000,
    "batch": 12000,
}
# Mirrors corpus_search_router.py's DEFAULT_CALLER_MODE. Also the safe
# landing spot for any caller_mode value Structure doesn't recognize —
# notably the corpus_search skill's assembly_strategy values ("score" /
# "canonical_first" / "balanced") that get sent into the caller_mode field
# today (Chat sign-off finding, part of the 3-way vocabulary bug). Falling
# through to the default here mirrors what nearly all live traffic already
# gets today (Chat confirmed most turns send no caller_mode at all), so
# unrecognized input degrades to today's status quo, not a crash or a
# guess.
DEFAULT_CALLER_MODE = "chat.default"

# Anchors verified live 2026-07-23 (schematic spec §5) — not guessed:
# corpus_search.py:81-83 / corpus_search_agent.py:2915,2924 (k=10 default),
# corpus_search_agent.py:4441 (strategy-b floor = max(k, 15)).
_BASE_K = 10
_FAN_OUT_BASE_K = 15
# chat.default IS DEFAULT_CALLER_MODE — its recall_demand is the baseline
# the two K constants above were measured against.
_BASELINE_RECALL_DEMAND = _RECALL_DEMAND[DEFAULT_CALLER_MODE]

# CORRECTED 2026-07-23 (Router caught, with live evidence): the original
# per-posture values (PRECISE=1, FAN_OUT=2, RELY_ON_EXTERNAL=1) were meant
# as a soft safety ceiling, subordinate to speed_budget's time-derived
# attempt count (§10 addendum) — but min(time_derived, 1) == 1 ALWAYS, so a
# ceiling of 1 IS the plan, not a rarely-binding backstop. This directly
# amputated the fallback chain in production (19/22 bank queries produced
# 0-occupancy answers, blocking Eval's Observer calibration) since PRECISE
# dominates real_time traffic. Router independently verified a uniform
# ceiling of 6 never actually binds for real_time (their allowance —
# 2000ms x 1.15 tolerance = 2300ms — naturally settles fallback chains
# around ~3 strategies / 2100ms on its own); the ceiling exists only to
# stop pathological runaway, never to be the real constraint. Posture-based
# variation removed entirely — it was working against the "time is the
# real constraint" directive, not serving it. One uniform value, matching
# what Router's own allocator already treats as generous-but-safe.
_MAX_ATTEMPTS_CEILING = 6

# Postures that don't reach retrieval this turn. UX sign-off 2026-07-23:
# resolve to an explicit all-zero ResourcePosture, not None — simpler for
# Diagnostics/Chat to consume, no null-checking on the far side.
_NO_RETRIEVAL_POSTURES = (
    ReformatPosture.CLARIFY,
    ReformatPosture.CLARIFY_REPHRASE,
    ReformatPosture.DECLINE,
)

_ZERO_RESOURCE_POSTURE = ResourcePosture(
    breadth=0, confidence_bar=0.0, speed_budget="none", token_budget=0, max_attempts=0, authority_requirement="any"
)

# Caller-declared, valid values only — anything else degrades to "any"
# (fail-open, matches Router's own AUTHORITY_ANY default) rather than
# propagating a typo'd/unrecognized value into eligibility filtering.
_VALID_AUTHORITY_REQUIREMENTS = frozenset({"any", "citable_required"})


def run_structure(
    reformat: ReformatResult, caller_mode: str | None = None, authority_requirement: str | None = None,
    token_budget_for_retrieval: int | None = None,
) -> StructureResult:
    """`token_budget_for_retrieval` (2026-07-24, Ananth's direct correction:
    "RAG does not have to guess"): Chat's real, request-level context-window
    math (context_window - system_prompt - conversation_history -
    answer_generation_reserve), NOT Structure's static _TOKEN_BUDGET table --
    same caller-declares pattern as authority_requirement. Chat/Eval's joint
    ruling: request-varying (conversation history dominates: a thread's
    first turn vs. turn 20 differ by thousands of tokens), a hard CAPACITY
    ceiling for Router's allocator (not a calibration-cell dimension --
    Eval: it's an allocation input, not a strategy-quality axis, and adding
    it as a prior cell would fragment already-sparse calibration data).
    Optional and falls through to _TOKEN_BUDGET's static table when omitted
    (legacy call sites) -- no breaking change. This is exactly the
    replacement for the guessed 3000 that caused Router's payload gate to
    reject a/b/c on every real_time query (see shape-structure-schematic-spec.md
    §11 and the 2026-07-24 fleet-wide root-cause writeup)."""
    t0 = time.monotonic()
    mode = caller_mode if caller_mode in _ACCURACY_NEED else DEFAULT_CALLER_MODE
    authority = authority_requirement if authority_requirement in _VALID_AUTHORITY_REQUIREMENTS else "any"
    resource_posture = _resolve_resource_posture(reformat.posture, mode, authority, token_budget_for_retrieval)
    result = StructureResult(
        query=reformat.query,
        rewritten_queries=list(reformat.rewritten_queries),
        posture=reformat.posture,
        fanout_themes=list(reformat.fanout_themes),  # passthrough for Shape:Slots (Step 1d) prioritization
        resource_posture=resource_posture,
        reason=_reason(reformat.posture, mode, caller_mode),
    )
    result.structure_ms = int((time.monotonic() - t0) * 1000)
    return result


def _resolve_resource_posture(
    posture: ReformatPosture, mode: str, authority_requirement: str,
    token_budget_for_retrieval: int | None = None,
) -> ResourcePosture:
    if posture in _NO_RETRIEVAL_POSTURES:
        return _ZERO_RESOURCE_POSTURE

    confidence_bar = _ACCURACY_NEED[mode]
    speed_budget = _SPEED_BUDGET[mode]
    # Caller-declared (Chat's real context-window math) wins when present;
    # the static table is a fallback for callers that haven't wired this
    # through yet, not the source of truth going forward.
    token_budget = token_budget_for_retrieval if token_budget_for_retrieval is not None else _TOKEN_BUDGET[mode]
    recall_demand = _RECALL_DEMAND[mode]
    max_attempts = _MAX_ATTEMPTS_CEILING

    if posture == ReformatPosture.PRECISE:
        breadth = round(_BASE_K * recall_demand / _BASELINE_RECALL_DEMAND)
    elif posture == ReformatPosture.FAN_OUT:
        breadth = round(_FAN_OUT_BASE_K * recall_demand / _BASELINE_RECALL_DEMAND)
    else:
        # RELY_ON_EXTERNAL — Eval sign-off 2026-07-23: leave as a
        # placeholder. Real unit (external result count from Router
        # strategy c/d, not corpus chunk count) is undecided and deferred
        # downstream — not invented here.
        breadth = _BASE_K

    return ResourcePosture(
        breadth=breadth,
        confidence_bar=confidence_bar,
        speed_budget=speed_budget,
        token_budget=token_budget,
        max_attempts=max_attempts,
        authority_requirement=authority_requirement,
    )


def _reason(posture: ReformatPosture, resolved_mode: str, raw_caller_mode: str | None) -> str:
    if posture in _NO_RETRIEVAL_POSTURES:
        return f"{posture.value} — no retrieval this turn, all-zero ResourcePosture"
    if raw_caller_mode is not None and raw_caller_mode != resolved_mode:
        return (
            f"{posture.value} + caller_mode={raw_caller_mode!r} unrecognized, "
            f"fell back to {resolved_mode} — resourced via v1 lookup table"
        )
    return f"{posture.value} + caller_mode={resolved_mode} — resourced via v1 lookup table"
