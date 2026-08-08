"""Router decision types for constrained optimizer design.

Router's output:
  - RoutingLadder: strategy sequences per slot (what Fillers will walk)
  - Feature context for bandit (priors_version, confidence, etc.)

Router's input:
  - N slots with priority, query_class, etc.
  - Pool's corpus-depth signals
  - Resource posture (time/confidence/accuracy budgets, tolerance bands)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from app.services.router.allocation import RoutingLadder


@dataclass
class ResourcePosture:
    """Caller's constraints (from Structure or explicit API).

    THIS dataclass is the wire contract at the route() boundary; router.py
    bridges it to the raw posture dict that resolve_constraints() consumes
    internally. A field that exists only on the dict side is unreachable
    from the live orchestrator path — every caller-facing knob must be
    declared HERE and carried across the bridge (gap caught by Structure
    2026-07-23: token/authority knobs were dict-only)."""
    speed_budget: Literal["real_time", "interactive", "background", "none"] = "interactive"
    confidence_bar: float = 0.85  # [0, 1]
    accuracy_bar: float = 0.80  # [0, 1]
    max_attempts_per_slot: int = 3
    caller_mode: str = "chat.default"  # chat.default, chat.thinking, batch, etc.
    tolerance_bands: dict[str, float] = field(default_factory=lambda: {"min_pct": -0.15, "max_pct": 0.25})
    # PER-SLOT payload allowance in tokens (Structure's field name, per the
    # original Retriever relay: a hard per-slot cap for fillers/planning).
    # None → Router's generous default (DEFAULT_TOKEN_ALLOWANCE_PER_SLOT).
    # If Structure ever redefines this as a QUERY-level total, the router.py
    # bridge is the single translation point to change.
    token_budget: Optional[int] = None
    # Caller-declared citability (Ananth's bifurcation): "any" | "citable_required"
    authority_requirement: str = "any"


@dataclass
class RouterDecision:
    """Complete output of Router's reasoning phase.

    Frozen for Fillers + Observer + Orchestrator:
      - RoutingLadder defines the per-slot strategy plan (sequential chain in
        greedy mode; parallel candidate set in optimization mode — see
        routing_ladder.execution_mode/gating)
      - trace is the full structured decision trace (replayable-by-hand;
        Diagnostics emit). Prose narration renders from it on demand via
        router_narrate.narrate() and is NEVER persisted (PHI rule).
    """
    routing_ladder: RoutingLadder            # the EXECUTED plan
    decision_id: str
    resource_posture: ResourcePosture
    feature_context: dict[str, Any] = field(default_factory=dict)
    dispatch_path: Literal["forced", "greedy", "optimizer", "bayesian"] = "greedy"
    reason: str = ""
    trace: Any = None                        # tracing.DecisionTrace (executed)
    shadow_ladders: list = field(default_factory=list)  # RoutingLadder — untaken plans, never run
    shadow_traces: list = field(default_factory=list)   # tracing.DecisionTrace per shadow


@dataclass
class RoutingContext:
    """Context for Router to make its decision."""
    query: str
    agent_id: str
    is_calibration: bool = False
    forced_strategy: Optional[str] = None
    allocator_override: Optional[str] = None  # pins executed allocator: "greedy" | "optimizer" | "bayesian"
    resource_posture: Optional[ResourcePosture] = None
    # REAL SEAM (completed 2026-07-23 after Retriever caught the comment/code
    # mismatch): the orchestrator passes AnswerShapeResult.slots verbatim here
    # — real AnswerSlot objects, no shim reconstruction. When empty, route()
    # falls back to building slots from pool_metadata (compat path).
    # pool_metadata is STILL required either way: it carries the per-slot
    # corpus-depth signals (top_score_percentile, pool_size) that slots
    # themselves don't hold.
    slots: list = field(default_factory=list)
    pool_metadata: dict[str, Any] = field(default_factory=dict)  # slot_id → pool signal
    # Gate's jurisdiction codes (GateResult.j_codes, no kind prefix — e.g.
    # "payor.sunshine_health"). Drives tag-gated strategy eligibility (s).
    # FAIL CLOSED: empty → tag-gated strategies are never planned.
    gate_j_codes: list = field(default_factory=list)
    # Gate's domain codes (GateResult.d_codes, e.g. "claims.timely_filing").
    # Gates the sitemap_links HELPER: Sitemap's lookup requires a d:-tag topic
    # match in addition to payer identity — without any d-code the lookup
    # deterministically returns [].
    gate_d_codes: list = field(default_factory=list)
    # Tri-state payor crawlability verdict (fillers/payer_context.py):
    # True | False | None. Gates strategy d — only affirmative False
    # disqualifies (fail-open on None; d is the general web strategy).
    payer_crawlable: object = None
    upstream_diagnostics: dict[str, Any] = field(default_factory=dict)  # from Shape/Gate/Reformat
    # Caller-supplied identifier for cross-DB join back to chat_turns
    # (2026-08-06, Chat Master's grading-callback gap: the legacy
    # corpus_search_agent path threaded this as correlation_id=caller_id so
    # Chat could PATCH /observe/decisions/{correlation_id}/grade after
    # producing its own synthesis-skipped answer -- the new pipeline never
    # had anywhere for a caller to supply it). None for calibration/eval
    # runs and any caller that doesn't need the post-hoc grading callback.
    correlation_id: Optional[str] = None
    # Which call number THIS is, within Chat's own retry/round loop for the
    # SAME underlying user question (2026-08-07, Ananth's directive: "let's
    # restrict c until the 3rd turn"). None/1 (unknown or genuinely first
    # call) fails CLOSED -- treated as turn 1, c restricted -- rather than
    # silently allowing c whenever a caller doesn't send this. Chat's own
    # react_loop already tracks this as its round counter; this just exposes
    # it to the allocator that can act on it (currently: portfolio's cost
    # gate; chain allocators don't read it, out of scope for this directive).
    call_number: Optional[int] = None
