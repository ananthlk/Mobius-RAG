"""Cross-slot continuation decision — "one more turn, or done?"

The Router piece of the refined Observer scope (Ananth via Retriever,
2026-07-23): each filler answers a strategy-specific "would this slot benefit
from another turn?" with its own bar; THIS module aggregates all N slots'
(verdict, reason) pairs + the time budget into ONE query-level decision.
Pure function, no I/O — the call site is Retriever's orchestrator.py loop
(same ownership seam as allocate_strategies).

Interface (locked with Retriever 2026-07-23): per slot the orchestrator
passes the pre-sliced REMAINDER of the planned chain (already
_IMPLEMENTED_FILLERS-filtered, already past the current rung) — that
filtering is deliberately the orchestrator's concern, not this module's.

Verdict semantics in aggregation (acked by Retriever + Eval):
  WOULD_BENEFIT      — turn-eligible; JUSTIFIES a turn if the slot is required
  SATISFIED          — terminal; never runs again (the verdict's job is to say
                       whether another turn helps — SATISFIED means it said no)
  EXHAUSTED_ATTEMPTS — terminal, permanently: the strategy chain had its shot
  EXHAUSTED_BUDGET   — conditionally re-eligible: stopped by the CLOCK, not by
                       its own limits — rides along on a sibling's turn if its
                       next rung fits inside the justifying envelope. NEVER
                       justifies a turn itself (the clock only moves forward).
  ERROR              — infra failure: the rung is marked failed and the slot
                       advances to its NEXT rung; ride-along-eligible only.
                       Deliberate consequence (conservative, avoids burning
                       budget on infra loops): a single-required-slot query
                       that ERRORs gets no new turn — flagged as an open
                       corner, revisit if it bites in practice.

RIDE-ALONG SELECTION-BIAS FLAG (Eval's required addition, 2026-07-23):
ride-along turns only happen because a required sibling was struggling —
ride-along SATISFIED observations are sampled from a non-representative
subset. The `participation` map marks every participating slot "justifying"
or "ride_along"; the orchestrator MUST stamp `ride_along: true` on any
observation emitted from a ride_along participant so Eval's priors
derivation can segment/down-weight them instead of silently drifting the
(depth_bucket, strategy) cell. The earlier budget-cut rung stays fully
excluded from calibration regardless of what the ride-along rung does.

Token budget: deliberately NOT an input — every planned rung already passed
the payload gate at plan time, and under advance-on-empty a later rung
REPLACES the failed one's contribution, so a new turn cannot exceed the
plan-time worst case. RETENTION RULING (Ananth via Retriever, 2026-07-23):
the answer is RETAIN — superseded rungs' outputs will be kept for Synthesis
to combine across rungs. When retention ships (gated on Synthesis kickoff),
this function GAINS a remaining-token-budget input alongside the latency
budget, and plan-time accounting flips MAX→SUM. Correct as-is exactly as
long as the orchestrator's live behavior is discard.

PHI rule (Router standard): inputs and outputs carry slot ids, strategy ids,
verdicts, and numbers only — never query or chunk text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.services.router.priors import (
    PriorsBundle,
    _DEFAULT_STRATEGY_LATENCY_MS,
    load_priors,
)

VERDICT_WOULD_BENEFIT = "WOULD_BENEFIT"
VERDICT_SATISFIED = "SATISFIED"
VERDICT_EXHAUSTED_ATTEMPTS = "EXHAUSTED_ATTEMPTS"
VERDICT_EXHAUSTED_BUDGET = "EXHAUSTED_BUDGET"
VERDICT_ERROR = "ERROR"

# verdicts that leave a slot eligible to participate in a turn at all
_JUSTIFY_CAPABLE = frozenset({VERDICT_WOULD_BENEFIT})
_RIDE_ONLY = frozenset({VERDICT_EXHAUSTED_BUDGET, VERDICT_ERROR})
_TERMINAL = frozenset({VERDICT_SATISFIED, VERDICT_EXHAUSTED_ATTEMPTS})

PARTICIPATION_JUSTIFYING = "justifying"
PARTICIPATION_RIDE_ALONG = "ride_along"


@dataclass(frozen=True)
class SlotTurnInput:
    """One slot's state at the turn boundary, as the orchestrator sees it."""
    slot_id: str
    remaining_rungs: tuple[str, ...]  # pre-sliced, pre-filtered (orchestrator's concern)
    verdict: str                      # one of the VERDICT_* constants
    reason: str = ""                  # filler's own reason text — passed through verbatim
    required: bool = True


@dataclass
class ContinuationDecision:
    """Reason-rich, never a bare bool. Reasons flow to telemetry verbatim —
    Eval's calibration exclusions key off the ORIGINAL verdict enum."""
    new_turn: bool = False
    justified_by: list[str] = field(default_factory=list)   # required slots that earn the turn
    ride_along: list[str] = field(default_factory=list)     # free riders inside the envelope
    dropped: dict[str, str] = field(default_factory=dict)   # eligible but excluded: slot_id -> reason
    # per participating slot: "justifying" | "ride_along". CONTRACT: the
    # orchestrator stamps ride_along=true on any observation emitted from a
    # ride_along participant (Eval's selection-bias segmentation).
    participation: dict[str, str] = field(default_factory=dict)
    turn_rungs: dict[str, str] = field(default_factory=dict)  # slot_id -> strategy to run this turn
    envelope_ms: int = 0              # MAX next-rung latency over participants (parallel model)
    budget_remaining_ms: int = 0
    stop_reason: str = ""             # set when new_turn=False
    per_slot_verdicts: dict[str, tuple[str, str]] = field(default_factory=dict)  # verbatim (verdict, reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_turn": self.new_turn,
            "justified_by": self.justified_by,
            "ride_along": self.ride_along,
            "dropped": self.dropped,
            "participation": self.participation,
            "turn_rungs": self.turn_rungs,
            "envelope_ms": self.envelope_ms,
            "budget_remaining_ms": self.budget_remaining_ms,
            "stop_reason": self.stop_reason,
            "per_slot_verdicts": {k: list(v) for k, v in self.per_slot_verdicts.items()},
        }


def _next_rung_latency_ms(strategy_id: str, bundle: PriorsBundle) -> int:
    """Latency of a strategy's next attempt. The priors file carries the same
    latency_p50 per strategy across all depth buckets (verified against the
    live file), so a depth-independent lookup is faithful to current data;
    depth-2 is the canonical row, per-strategy defaults the fallback."""
    prof = bundle.by_depth.get(strategy_id, {}).get(2)
    if prof is not None:
        return prof.latency_p50_ms
    return _DEFAULT_STRATEGY_LATENCY_MS.get(strategy_id, 1000)


def decide_continuation(
    slots: list[SlotTurnInput],
    elapsed_ms: int,
    latency_allowance_ms: float,
    bundle: Optional[PriorsBundle] = None,
) -> ContinuationDecision:
    """Aggregate per-slot verdicts + time budget into ONE turn/done decision.

    Rule (acked design):
      1. A turn happens iff ≥1 REQUIRED slot is justify-capable
         (WOULD_BENEFIT + rungs remain + its next rung fits the remaining
         allowance). Optional slots NEVER justify (§2a: optional never gates).
      2. Envelope = MAX next-rung latency over justifiers (parallel model).
      3. Any other eligible slot rides free iff its next rung fits INSIDE the
         envelope; a rung that would BECOME the new max is dropped, with
         reason — "free" means free.
    """
    bundle = bundle if bundle is not None else load_priors()
    d = ContinuationDecision(
        budget_remaining_ms=max(0, int(latency_allowance_ms) - int(elapsed_ms)),
        per_slot_verdicts={s.slot_id: (s.verdict, s.reason) for s in slots},
    )
    remaining = d.budget_remaining_ms

    justifiers: list[tuple[SlotTurnInput, str, int]] = []   # (slot, rung, latency)
    riders: list[tuple[SlotTurnInput, str, int]] = []

    for s in slots:
        if s.verdict in _TERMINAL:
            continue  # SATISFIED / EXHAUSTED_ATTEMPTS: never participate
        if not s.remaining_rungs:
            d.dropped[s.slot_id] = "no_remaining_rungs"
            continue
        rung = s.remaining_rungs[0]
        lat = _next_rung_latency_ms(rung, bundle)
        if s.verdict in _JUSTIFY_CAPABLE and s.required:
            if lat <= remaining:
                justifiers.append((s, rung, lat))
            else:
                d.dropped[s.slot_id] = "next_rung_over_remaining_budget"
        elif s.verdict in _JUSTIFY_CAPABLE or s.verdict in _RIDE_ONLY:
            # optional WOULD_BENEFIT, or budget-cut / errored slots: ride-only
            riders.append((s, rung, lat))
        else:
            d.dropped[s.slot_id] = f"unknown_verdict_{s.verdict}"

    if not justifiers:
        d.new_turn = False
        if any(s.verdict in _JUSTIFY_CAPABLE and s.required for s in slots):
            d.stop_reason = "required_slots_over_budget"
        elif riders:
            d.stop_reason = "only_ride_eligible_slots_no_justifier"
            for s, _, _ in riders:
                d.dropped.setdefault(s.slot_id, "no_justifying_sibling")
        else:
            d.stop_reason = "no_turn_eligible_slots"
        return d

    d.new_turn = True
    d.envelope_ms = max(lat for _, _, lat in justifiers)
    for s, rung, _ in justifiers:
        d.justified_by.append(s.slot_id)
        d.participation[s.slot_id] = PARTICIPATION_JUSTIFYING
        d.turn_rungs[s.slot_id] = rung
    for s, rung, lat in riders:
        if lat <= d.envelope_ms:
            d.ride_along.append(s.slot_id)
            d.participation[s.slot_id] = PARTICIPATION_RIDE_ALONG
            d.turn_rungs[s.slot_id] = rung
        else:
            d.dropped[s.slot_id] = "would_extend_envelope"
    return d


def narrate_continuation(d: ContinuationDecision) -> str:
    """Prose rendering, Router narrate() standard: everything, in order, with
    numbers. Computed on demand, never persisted (PHI rule — though this
    trace carries no text, the standard is uniform)."""
    lines = []
    verdict_bits = ", ".join(
        f"{sid}={v}" + (f" ({r})" if r else "")
        for sid, (v, r) in d.per_slot_verdicts.items()
    )
    lines.append(f"CONTINUATION — verdicts: {verdict_bits}. "
                 f"Budget remaining {d.budget_remaining_ms}ms.")
    if d.new_turn:
        lines.append(
            f"NEW TURN — justified by {d.justified_by} "
            f"(envelope {d.envelope_ms}ms, parallel-max)."
        )
        if d.ride_along:
            lines.append(
                f"  Ride-along (free, inside envelope): {d.ride_along} — "
                "observations from these slots carry ride_along=true "
                "(selection-bias segmentation, per Eval)."
            )
        lines.append("  Turn rungs: "
                     + ", ".join(f"{sid}→'{r}'" for sid, r in d.turn_rungs.items()) + ".")
    else:
        lines.append(f"DONE — {d.stop_reason}.")
    for sid, reason in d.dropped.items():
        lines.append(f"  Dropped {sid}: {reason}.")
    return "\n".join(lines)
