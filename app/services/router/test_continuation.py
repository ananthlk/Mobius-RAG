"""Continuation-decision tests — the acked cross-slot aggregation design.

All latencies from the frozen fixture via conftest's ROUTER_PRIORS_PATH
(latency_p50 is per-strategy constant across depths: s=100, a=500, b=1500,
c=2000, d=3000).
"""

import pytest

from app.services.router.continuation import (
    ContinuationDecision,
    SlotTurnInput,
    VERDICT_ERROR,
    VERDICT_EXHAUSTED_ATTEMPTS,
    VERDICT_EXHAUSTED_BUDGET,
    VERDICT_SATISFIED,
    VERDICT_WOULD_BENEFIT,
    decide_continuation,
    narrate_continuation,
)


def slot(sid="slot_0", rungs=("b",), verdict=VERDICT_WOULD_BENEFIT,
         reason="", required=True):
    return SlotTurnInput(slot_id=sid, remaining_rungs=tuple(rungs),
                         verdict=verdict, reason=reason, required=required)


class TestJustification:
    def test_required_would_benefit_justifies_turn(self):
        d = decide_continuation([slot(rungs=("b", "c"))], elapsed_ms=1000,
                                latency_allowance_ms=5750)
        assert d.new_turn is True
        assert d.justified_by == ["slot_0"]
        assert d.turn_rungs == {"slot_0": "b"}
        assert d.envelope_ms == 1500  # b's p50
        assert d.participation["slot_0"] == "justifying"

    def test_optional_never_justifies_alone(self):
        d = decide_continuation(
            [slot(required=False, rungs=("d",))], elapsed_ms=0,
            latency_allowance_ms=5750)
        assert d.new_turn is False
        assert d.stop_reason == "only_ride_eligible_slots_no_justifier"
        assert d.dropped["slot_0"] == "no_justifying_sibling"

    def test_over_budget_required_slot_cannot_justify(self):
        d = decide_continuation([slot(rungs=("d",))], elapsed_ms=4000,
                                latency_allowance_ms=5750)  # 1750 left < 3000
        assert d.new_turn is False
        assert d.stop_reason == "required_slots_over_budget"
        assert d.dropped["slot_0"] == "next_rung_over_remaining_budget"

    def test_no_eligible_slots(self):
        d = decide_continuation(
            [slot(verdict=VERDICT_SATISFIED),
             slot(sid="s2", verdict=VERDICT_EXHAUSTED_ATTEMPTS)],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.new_turn is False
        assert d.stop_reason == "no_turn_eligible_slots"


class TestRideAlong:
    def test_optional_rides_inside_envelope(self):
        d = decide_continuation(
            [slot(rungs=("d",)),                              # envelope 3000
             slot(sid="opt", required=False, rungs=("b",))],  # 1500 fits inside
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.new_turn and d.justified_by == ["slot_0"]
        assert d.ride_along == ["opt"]
        assert d.participation["opt"] == "ride_along"
        assert d.turn_rungs["opt"] == "b"

    def test_rider_dropped_if_it_would_extend_envelope(self):
        """'free' means free: a rung that would BECOME the new max is dropped."""
        d = decide_continuation(
            [slot(rungs=("b",)),                              # envelope 1500
             slot(sid="opt", required=False, rungs=("d",))],  # 3000 would extend
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.new_turn and d.ride_along == []
        assert d.dropped["opt"] == "would_extend_envelope"

    def test_exhausted_budget_rides_but_never_justifies(self):
        # alone: no turn
        alone = decide_continuation(
            [slot(verdict=VERDICT_EXHAUSTED_BUDGET, rungs=("s",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert alone.new_turn is False
        # with a justifying sibling whose envelope covers it: rides
        d = decide_continuation(
            [slot(rungs=("d",)),
             slot(sid="cut", verdict=VERDICT_EXHAUSTED_BUDGET, rungs=("s",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.new_turn and d.ride_along == ["cut"]

    def test_exhausted_attempts_never_rides_even_with_sibling(self):
        d = decide_continuation(
            [slot(rungs=("d",)),
             slot(sid="done", verdict=VERDICT_EXHAUSTED_ATTEMPTS, rungs=("s",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.new_turn and "done" not in d.ride_along
        assert "done" not in d.participation

    def test_satisfied_never_reruns(self):
        d = decide_continuation(
            [slot(rungs=("d",)),
             slot(sid="ok", verdict=VERDICT_SATISFIED, rungs=("s",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.new_turn and "ok" not in d.participation

    def test_error_rides_but_does_not_justify(self):
        """Documented corner: a single-required-slot query that ERRORs gets no
        new turn (conservative: no budget burned on infra loops)."""
        alone = decide_continuation(
            [slot(verdict=VERDICT_ERROR, rungs=("b",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert alone.new_turn is False
        d = decide_continuation(
            [slot(rungs=("d",)),
             slot(sid="err", verdict=VERDICT_ERROR, rungs=("b",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.new_turn and d.ride_along == ["err"]

    def test_empty_remainder_ineligible_regardless_of_verdict(self):
        d = decide_continuation(
            [slot(rungs=()), slot(sid="s2", rungs=("b",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.justified_by == ["s2"]
        assert d.dropped["slot_0"] == "no_remaining_rungs"


class TestBudgetMath:
    def test_budget_remaining_computed(self):
        d = decide_continuation([slot(rungs=("s",))], elapsed_ms=2000,
                                latency_allowance_ms=5750)
        assert d.budget_remaining_ms == 3750

    def test_envelope_is_max_over_justifiers(self):
        d = decide_continuation(
            [slot(sid="s1", rungs=("b",)), slot(sid="s2", rungs=("d",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert sorted(d.justified_by) == ["s1", "s2"]
        assert d.envelope_ms == 3000

    def test_one_justifier_over_budget_other_fits(self):
        d = decide_continuation(
            [slot(sid="s1", rungs=("s",)), slot(sid="s2", rungs=("d",))],
            elapsed_ms=4000, latency_allowance_ms=5750)  # 1750 left
        assert d.new_turn and d.justified_by == ["s1"]
        assert d.dropped["s2"] == "next_rung_over_remaining_budget"


class TestTelemetryContract:
    def test_verdicts_flow_verbatim_never_collapsed(self):
        """Eval's calibration exclusions key off the ORIGINAL enum — reasons
        must survive aggregation untouched."""
        d = decide_continuation(
            [slot(rungs=("b",), reason="below own bar 0.6<0.8"),
             slot(sid="cut", verdict=VERDICT_EXHAUSTED_BUDGET, rungs=("s",),
                  reason="clock cut at rung 2")],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.per_slot_verdicts["slot_0"] == (VERDICT_WOULD_BENEFIT, "below own bar 0.6<0.8")
        assert d.per_slot_verdicts["cut"] == (VERDICT_EXHAUSTED_BUDGET, "clock cut at rung 2")
        payload = d.to_dict()
        assert payload["per_slot_verdicts"]["cut"] == [VERDICT_EXHAUSTED_BUDGET, "clock cut at rung 2"]

    def test_participation_map_supports_ride_along_stamping(self):
        """Eval's selection-bias addition: the orchestrator stamps
        ride_along=true from this map — it must be complete and disjoint."""
        d = decide_continuation(
            [slot(rungs=("d",)),
             slot(sid="opt", required=False, rungs=("s",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        assert d.participation == {"slot_0": "justifying", "opt": "ride_along"}
        assert set(d.turn_rungs) == set(d.participation)

    def test_narration_mentions_ride_along_flag_contract(self):
        d = decide_continuation(
            [slot(rungs=("d",)),
             slot(sid="opt", required=False, rungs=("s",))],
            elapsed_ms=0, latency_allowance_ms=5750)
        text = narrate_continuation(d)
        assert "NEW TURN" in text and "ride_along=true" in text
        assert "opt" in text and "3000ms" in text

    def test_narration_done_path(self):
        d = decide_continuation(
            [slot(verdict=VERDICT_SATISFIED)],
            elapsed_ms=100, latency_allowance_ms=5750)
        text = narrate_continuation(d)
        assert "DONE — no_turn_eligible_slots" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
