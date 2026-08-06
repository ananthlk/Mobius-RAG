"""Allocation tests — fallback-chain math + scenario/edge-case coverage.

All expected values are hand-derived from eval/priors_bootstrap.yaml:
  depth_0: s .100/100ms  a .100/500  b .100/1500  c .071/2000  d .150/3000  f .120/2500
  depth_1: s .350/100    a .350/500  b .175/1500  c .300/2000  d .330/3000  f .280/2500
  depth_2: s .500/100    a .543/500  b .286/1500  c .330/2000  d .450/3000  f .380/2500
  depth_3: s .600/100    a .800/500  b .480/1500  c .400/2000  d .550/3000  f .480/2500
Chain order is static: s, a, b, c, d, f.
"""

import pytest

from app.services.router.allocation import (
    AnswerSlot,
    allocate_strategies,
    chain_expected_accuracy,
    chain_success_probability,
    resolve_tolerance_pct,
)
from app.services.router.priors import (
    PriorsBundle,
    StrategyProfile,
    load_priors,
)


# Pool metadata presets per depth bucket (see compute_depth_bucket thresholds)
POOL_DEPTH_0 = {"top_score_percentile": 0.95, "pool_size": 30}
POOL_DEPTH_1 = {"top_score_percentile": 0.80, "pool_size": 150}
POOL_DEPTH_2 = {"top_score_percentile": 0.60, "pool_size": 400}
POOL_DEPTH_3 = {"top_score_percentile": 0.30, "pool_size": 2000}


def make_slot(slot_id="slot_0", priority=0, required=True,
              slot_semantics="direct_answer"):
    """Constructs the REAL AnswerSlot (shape/slots.py) — no stand-in fields.
    max_attempts is QUERY-level: set posture(max_attempts_per_slot=N)."""
    return AnswerSlot(
        slot_id=slot_id,
        slot_semantics=slot_semantics,
        capacity=5,
        rewritten_query="q",
        required=required,
        priority=priority,
    )


def posture(**overrides):
    base = {
        "speed_budget": "interactive",
        "confidence_bar": 0.85,
        "accuracy_bar": 0.80,
        "max_attempts_per_slot": 6,
        # tests simulate PAYOR queries so tag-gated 's' stays eligible
        "gate_j_codes": ["payor.sunshine_health"],
        # d-code present → sitemap_links helper producible
        "gate_d_codes": ["claims.timely_filing"],
        "tolerance_bands": {"max_pct": 0.25},
        "caller_mode": "chat.default",
    }
    base.update(overrides)
    return base


class TestChainMath:
    """Dedicated tests for the expected-value formula with known inputs/outputs."""

    def test_empty_chain_is_zero(self):
        assert chain_success_probability([]) == 0.0

    def test_single_rung_equals_lift(self):
        assert chain_success_probability([0.5]) == pytest.approx(0.5)

    def test_two_rungs(self):
        # P = p1 + (1-p1)*p2
        assert chain_success_probability([0.5, 0.5]) == pytest.approx(0.75)
        assert chain_success_probability([0.45, 0.44]) == pytest.approx(0.692)

    def test_three_rungs(self):
        # 1 - (0.5 * 0.6 * 0.7) = 0.79
        assert chain_success_probability([0.5, 0.4, 0.3]) == pytest.approx(0.79)

    def test_order_is_commutative_for_success_probability(self):
        assert chain_success_probability([0.2, 0.7]) == pytest.approx(
            chain_success_probability([0.7, 0.2])
        )

    def test_clamping_out_of_range_lifts(self):
        assert chain_success_probability([1.5]) == pytest.approx(1.0)
        assert chain_success_probability([-0.2, 0.5]) == pytest.approx(0.5)

    def test_expected_accuracy_probability_weighted(self):
        # rung1: p=.5 acc=.8 ; rung2: p=.5 acc=.4
        # weighted = .5*.8 + .5*.5*.4 = .5 ; total_p = .5 + .25 = .75 ; E = 2/3
        profiles = [
            StrategyProfile(0.5, 100, 0, 0.8),
            StrategyProfile(0.5, 500, 1, 0.4),
        ]
        assert chain_expected_accuracy(profiles) == pytest.approx(2 / 3)

    def test_expected_accuracy_empty_or_zero_chain(self):
        assert chain_expected_accuracy([]) == 0.0
        assert chain_expected_accuracy([StrategyProfile(0.0, 100, 0, 0.9)]) == 0.0


class TestToleranceBands:
    """Caller-mode-dependent bands actually apply (not just documented)."""

    def test_real_time_modes_get_15_pct(self):
        assert resolve_tolerance_pct({"caller_mode": "chat.default"}) == 0.15
        assert resolve_tolerance_pct({"caller_mode": "real_time"}) == 0.15

    def test_background_modes_get_25_pct(self):
        assert resolve_tolerance_pct({"caller_mode": "batch"}) == 0.25
        assert resolve_tolerance_pct({"caller_mode": "chat.thinking"}) == 0.25
        assert resolve_tolerance_pct({"caller_mode": "background"}) == 0.25

    def test_unknown_mode_uses_explicit_bands_then_default(self):
        assert resolve_tolerance_pct({"caller_mode": "??", "tolerance_bands": {"max_pct": 0.10}}) == 0.10
        assert resolve_tolerance_pct({}) == 0.25

    def test_bands_change_adjusted_bar_and_allowance(self):
        """Caller mode drives tolerance: bar .85 → adjusted .7225 (real-time)
        vs .6375 (background); allowance 5750ms vs 6250ms. (Under LB enforcement
        both modes yield the same depth_1 chain here, so the band's effect is
        asserted on the bars/tolerances directly.)"""
        slots = [make_slot()]
        pool = {"slot_0": POOL_DEPTH_1}

        ladder_rt = allocate_strategies(slots, pool, posture(caller_mode="chat.default"))
        ladder_bg = allocate_strategies(slots, pool, posture(caller_mode="batch"))

        assert ladder_rt.tolerance_pct == 0.15
        assert ladder_bg.tolerance_pct == 0.25
        assert ladder_rt.adjusted_confidence_bar == pytest.approx(0.85 * 0.85)
        assert ladder_bg.adjusted_confidence_bar == pytest.approx(0.85 * 0.75)


class TestChatThinkingLatencyOverride:
    """Ananth 2026-08-05 via Retriever: d's real 9732ms attempt_ms prior needs
    the cumulative chain latency through s+a+b+c+d (13832ms at seed values)
    to clear the per-rung budget check — the generic interactive formula
    (6250ms) structurally excludes d everywhere. Scoped NARROWLY to
    chat.thinking; real_time modes must stay untouched."""

    def test_thinking_allowance_is_overridden(self):
        from app.services.router.allocation import resolve_constraints
        c = resolve_constraints({"caller_mode": "chat.thinking", "speed_budget": "interactive"})
        assert c["latency_allowance_ms"] == 16000

    def test_real_time_modes_unaffected(self):
        from app.services.router.allocation import resolve_constraints
        c_copilot = resolve_constraints({"caller_mode": "chat.copilot", "speed_budget": "real_time"})
        c_default = resolve_constraints({"caller_mode": "chat.default", "speed_budget": "real_time"})
        assert c_copilot["latency_allowance_ms"] == pytest.approx(2000 * 1.25)  # unknown-mode default band
        assert c_default["latency_allowance_ms"] == pytest.approx(2000 * 1.15)

    def test_other_interactive_modes_unaffected(self):
        """auth_agent also uses speed_budget='interactive' — must not
        inherit chat.thinking's override just because it shares the string."""
        from app.services.router.allocation import resolve_constraints
        c = resolve_constraints({"caller_mode": "auth_agent", "speed_budget": "interactive"})
        assert c["latency_allowance_ms"] == pytest.approx(5000 * 1.25)

    def test_d_becomes_reachable_under_chat_thinking(self):
        """End-to-end: d actually appears in the chain once the chain grows
        long enough to need it (seed data never clears the bar earlier)."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(caller_mode="chat.thinking", speed_budget="interactive",
                    confidence_bar=0.99, max_attempts_per_slot=6),
        )
        assert "d" in ladder.per_slot["slot_0"]

    def test_d_still_excluded_under_real_time_modes(self):
        for mode, sb in (("chat.copilot", "real_time"), ("chat.default", "real_time")):
            ladder = allocate_strategies(
                [make_slot()], {"slot_0": POOL_DEPTH_2},
                posture(caller_mode=mode, speed_budget=sb,
                        confidence_bar=0.99, max_attempts_per_slot=6),
            )
            assert "d" not in ladder.per_slot["slot_0"], mode


class TestAllocationScenarios:
    def test_single_slot_depth2_deterministic_chain_lb_enforced(self):
        """RE-DERIVED 2026-08-05: BEST-LB-FIRST (Ananth, confidence-density
        over latency) replaced cheap-fast-first — every rung now picks the
        highest per-rung Wilson LB, not the next strategy in a static
        priority order. This test pins to the FROZEN machinery snapshot
        (conftest.py's autouse fixture, not the live eval file) — d's
        latency there is 3000ms, unrelated to the live file's real
        9732ms value from Retriever's n=22 measurement.

        depth_2 (frozen), chat.default, bar .85 → adjusted .7225.
        Wilson LBs at n=8 (lb95): a .2815 > s .2486 > d .2122 > c .1327 > b .1065
        — best-LB order a,s,d,c,b, not s,a,b,c,d. Chain: a(500)→s(100)→
        d(3000, cum 3600 fits)→c(2000, cum 5600 fits); b(1500) would push
        7100ms over the 5750ms allowance, so it's skipped — chain stops at
        4 rungs, UNDER_CONFIDENT (budget_exhausted). LB=.6311 (higher than
        the old [s,a,b,c] chain's .5816 — d's LB, even gated by its own
        latency budget, contributes more per rung than b's did)."""
        ladder = allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture())
        assert ladder.per_slot["slot_0"] == ["a", "s", "d", "c"]
        assert ladder.per_slot_lb["slot_0"] == pytest.approx(0.6311, abs=1e-3)
        assert ladder.per_slot_confidence["slot_0"] == pytest.approx(0.9158, abs=1e-3)
        assert ladder.per_slot_status["slot_0"] == "UNDER_CONFIDENT"
        assert ladder.per_slot_latency_ms["slot_0"] == 5600
        assert ladder.outcome == "partial_infeasible"
        assert ladder.feasible is False

    def test_parallel_total_latency_is_max_not_sum(self):
        """RE-DERIVED 2026-08-05 (best-LB-first, frozen priors): two identical
        depth_2 slots → each chain [a,s,d,c] (best-LB order a>s>d>c>b) @5600ms
        (d fits at 3000ms in the frozen snapshot; b would push over the batch
        allowance 6250ms); total = 5600, not 11200."""
        slots = [make_slot("s1", priority=0), make_slot("s2", priority=1)]
        pool = {"s1": POOL_DEPTH_2, "s2": POOL_DEPTH_2}
        ladder = allocate_strategies(slots, pool, posture(caller_mode="batch", confidence_bar=0.9))
        assert ladder.per_slot["s1"] == ["a", "s", "d", "c"]
        assert ladder.per_slot["s2"] == ["a", "s", "d", "c"]
        assert ladder.total_estimated_ms == 5600  # MAX over slots, not sum

    def test_per_slot_budget_not_globally_deducted(self):
        """RE-DERIVED 2026-08-05 (best-LB-first): Parallel model: each slot
        gets the full wall-clock allowance.

        real_time (2000ms, ±15% → 2300 per slot), depth_2, unreachable bar:
        each slot independently fits best-LB order [a,s,b] = 2100ms (d's
        3000ms latency alone exceeds the 2300ms allowance, so it's gated
        out entirely regardless of its LB rank). Under the old (wrong)
        global-deduction model, the second slot would have been starved.
        """
        slots = [make_slot("s1", priority=0), make_slot("s2", priority=1)]
        pool = {"s1": POOL_DEPTH_2, "s2": POOL_DEPTH_2}
        ladder = allocate_strategies(
            slots, pool, posture(speed_budget="real_time", confidence_bar=0.99)
        )
        assert ladder.per_slot["s1"] == ["a", "s", "b"]
        assert ladder.per_slot["s2"] == ["a", "s", "b"]  # NOT starved
        assert ladder.per_slot_latency_ms["s1"] == 2100
        assert ladder.feasible is False
        assert "LB bar" in ladder.infeasibility_reason

    def test_all_strategies_fail_to_clear_bar(self):
        """depth_0: all 5 rungs (f retired) → mean .4243 (telemetry),
        LB ~.103 « adjusted bar .525 → UNDER_CONFIDENT / partial_infeasible."""
        ladder = allocate_strategies(
            [make_slot()],
            {"slot_0": POOL_DEPTH_0},
            posture(caller_mode="batch", speed_budget="background", confidence_bar=0.70),
        )
        assert len(ladder.per_slot["slot_0"]) == 5  # exhausted every option (f retired)
        assert ladder.aggregate_confidence_estimate == pytest.approx(0.4243, abs=0.002)
        assert ladder.per_slot_lb["slot_0"] < 0.20  # LB far below the mean
        assert ladder.per_slot_status["slot_0"] == "UNDER_CONFIDENT"
        assert ladder.outcome == "partial_infeasible"
        assert ladder.feasible is False
        assert "LB bar" in ladder.infeasibility_reason

    def test_zero_slot_query_is_flagged_not_crashed(self):
        ladder = allocate_strategies([], {}, posture())
        assert ladder.feasible is False
        assert ladder.infeasibility_reason == "no slots to allocate"
        assert ladder.per_slot == {}
        assert ladder.total_estimated_ms == 0

    def test_missing_pool_metadata_defaults_to_broad_bucket(self):
        """RE-DERIVED 2026-08-05 (best-LB-first): slot with no pool signal →
        bucket 4, still allocates without crashing.
        depth_4 LBs: b .4746, c .4746 (exact tie — b wins on priority-order
        tie-break), s .4602, d .4136, a .1236 — best-LB order is b,c,s,d,a,
        not s,a,.... Chain [b,c] lb=.7240 ≥ adjusted .525 → CLEARED."""
        ladder = allocate_strategies(
            [make_slot()], {}, posture(caller_mode="batch", confidence_bar=0.7)
        )
        assert ladder.per_slot["slot_0"] == ["b", "c"]
        assert ladder.per_slot_status["slot_0"] == "CLEARED"
        assert ladder.feasible is True

    def test_max_attempts_respected(self):
        """max_attempts=1 caps the chain at one rung even when bar unreachable.

        LAST-ATTEMPT RULE (cmhc002 collapse fix): the single rung is the
        BEST-LB viable strategy ('a', lb .2815), not the cheapest-first 's'
        (lb .2486) — this test previously pinned ['s'], i.e. the exact
        degenerate collapse that produced 0-occupancy on 19/22 bank queries."""
        ladder = allocate_strategies(
            [make_slot()],
            {"slot_0": POOL_DEPTH_2},
            posture(confidence_bar=0.99, max_attempts_per_slot=1),
        )
        assert ladder.per_slot["slot_0"] == ["a"]
        assert len(ladder.per_slot["slot_0"]) == 1  # cap still respected
        assert ladder.feasible is False

    def test_no_cross_slot_compensation(self):
        """§2a INVERSION of the old top-up test: a strong slot must NOT be
        extended to drag the mean over the bar for a weak slot.

        RE-DERIVED 2026-08-05 (best-LB-first): weak (depth_0): best-LB order
        d,s,a,b,c (d's LB .0383 edges out the .0195 3-way tie) — exhausts
        every rung anyway, LB .1034 « .525 → UNDER_CONFIDENT. strong
        (depth_3): best-LB order a,s — [a,s] LB .6698 ≥ .525 → CLEARED and
        STOPS THERE — no compensation rungs. Outcome is partial_infeasible
        even though the aggregate MEAN would have cleared. (max_attempts is
        QUERY-level now — the real AnswerSlot has no per-slot attempts
        field.)
        """
        slots = [
            make_slot("weak", priority=0),
            make_slot("strong", priority=1),
        ]
        pool = {"weak": POOL_DEPTH_0, "strong": POOL_DEPTH_3}
        ladder = allocate_strategies(
            slots, pool,
            posture(caller_mode="batch", speed_budget="background", confidence_bar=0.70),
        )
        assert len(ladder.per_slot["weak"]) == 5  # exhausted everything (f retired)
        assert ladder.per_slot_status["weak"] == "UNDER_CONFIDENT"
        assert ladder.per_slot["strong"] == ["a", "s"]  # stopped at ITS OWN bar
        assert ladder.per_slot_status["strong"] == "CLEARED"
        assert ladder.per_slot_lb["strong"] == pytest.approx(0.6698, abs=1e-3)
        assert ladder.outcome == "partial_infeasible"
        assert ladder.feasible is False
        # the weak slot's verdict names its binding constraint
        assert "weak" in ladder.infeasibility_reason

    def test_budget_exhausted_on_every_slot_simultaneously(self):
        """RE-DERIVED 2026-08-05 (best-LB-first): all slots capped by the
        per-slot time allowance at the same time.

        real_time chat.default → 2300ms allowance; depth_0's d (LB .0383)
        is always latency-gated out (its own 3000ms exceeds the 2300ms
        allowance from any starting point). s/a/b are EXACTLY tied at
        .0195 (same recall_lift=.1, same n) — but on the FIRST rung the
        SUPPLEMENT_ONLY gate excludes 's' while non-supplement candidates
        are viable, so a/b tie-break between themselves (a wins, earlier
        priority index) → chain starts [a]. Round 2 (chain non-empty, gate
        lifted): s re-enters, ties b, s wins its own tie-break → [a,s].
        Round 3: only b/c remain, b wins → [a,s,b]=2100ms; c (2000ms) would
        overflow too → capped at 3 rungs on every slot, bar unreachable →
        infeasible.
        """
        slots = [make_slot(f"s{i}", priority=i) for i in range(3)]
        pool = {f"s{i}": POOL_DEPTH_0 for i in range(3)}
        ladder = allocate_strategies(
            slots, pool, posture(speed_budget="real_time", confidence_bar=0.85)
        )
        for i in range(3):
            assert ladder.per_slot[f"s{i}"] == ["a", "s", "b"]
            assert ladder.per_slot_latency_ms[f"s{i}"] == 2100
        assert ladder.feasible is False

    def test_zero_and_negative_recall_lift_skipped_not_crashed(self):
        """Malformed priors (zero/negative lift) are skipped, never allocated."""
        bundle = PriorsBundle(
            by_depth={
                "a": {2: StrategyProfile(0.0, 500, 1, 0.5)},     # zero lift → skip
                "b": {2: StrategyProfile(-0.3, 1500, 1, 0.5)},   # clamped later, but guard anyway
                "d": {2: StrategyProfile(0.9, 3000, 3, 0.5)},    # only viable rung
            },
            version="test", source="test",
        )
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(caller_mode="batch", confidence_bar=0.7),
            bundle=bundle,
        )
        assert ladder.per_slot["slot_0"] == ["d"]
        assert ladder.feasible is True

    def test_priority_order_governs_topup_sequence(self):
        """Top-up extends higher-priority (lower number) slots first."""
        # Both slots identical depth_2, but bar high enough that phase-1 fails
        # per-slot within 2 attempts and top-up must pick someone: priority wins.
        slots = [
            make_slot("low_prio", priority=5),
            make_slot("high_prio", priority=0),
        ]
        pool = {"low_prio": POOL_DEPTH_2, "high_prio": POOL_DEPTH_2}
        ladder = allocate_strategies(
            slots, pool, posture(caller_mode="batch", speed_budget="background",
                                 confidence_bar=0.999, max_attempts_per_slot=3)
        )
        # Both end fully extended (3 attempts each) chasing an unreachable bar;
        # assertion here is ordering-stability: high_prio allocated ≥ low_prio.
        assert len(ladder.per_slot["high_prio"]) >= len(ladder.per_slot["low_prio"])


class TestPerQuestionEnforcement:
    """§2a correction: gate per QUESTION (required=True), not per slot."""

    def test_under_confident_optional_slot_does_not_flip_outcome(self):
        """EVAL'S DEMANDED TEST: a required slot that clears + an optional
        (required=False, e.g. RELY_ON_EXTERNAL external_context) slot that is
        hopeless (depth_0) → outcome must be all_slots_cleared, NOT
        partial_infeasible. The optional slot is still filled, LB-computed,
        and traced — just never gated."""
        slots = [
            make_slot("question", priority=0),
            make_slot("external_context", priority=1, required=False),
        ]
        pool = {"question": POOL_DEPTH_3, "external_context": POOL_DEPTH_0}
        ladder = allocate_strategies(
            slots, pool,
            posture(caller_mode="batch", speed_budget="background", confidence_bar=0.70),
        )
        assert ladder.per_slot_status["question"] == "CLEARED"
        assert ladder.per_slot_status["external_context"] == "OPTIONAL"
        # optional slot IS filled and reported…
        assert len(ladder.per_slot["external_context"]) >= 1
        assert ladder.per_slot_lb["external_context"] < 0.20  # hopeless, and visible
        # …but never gates:
        assert ladder.outcome == "all_slots_cleared"
        assert ladder.feasible is True
        assert ladder.infeasibility_reason == ""

    def test_under_confident_required_slot_still_gates(self):
        """Inverse control: same shape but the weak slot is required → partial_infeasible."""
        slots = [
            make_slot("question", priority=0),
            make_slot("hard_question", priority=1),  # required=True
        ]
        pool = {"question": POOL_DEPTH_3, "hard_question": POOL_DEPTH_0}
        ladder = allocate_strategies(
            slots, pool,
            posture(caller_mode="batch", speed_budget="background", confidence_bar=0.70),
        )
        assert ladder.outcome == "partial_infeasible"
        assert "hard_question" in ladder.infeasibility_reason
        assert "question" not in ladder.infeasibility_reason.replace("hard_question", "")

    def test_optional_slot_capped_at_one_cheap_attempt(self):
        """Eval's efficiency catch: optional slots must NOT chase the bar —
        one attempt only (greedy static order → 's', the 100ms cache rung),
        stop_reason 'optional_capped'. A telemetry-only slot may not burn
        Fillers latency on rungs it never gates with."""
        slots = [
            make_slot("question", priority=0),
            make_slot("external_context", priority=1, required=False),
        ]
        pool = {"question": POOL_DEPTH_3, "external_context": POOL_DEPTH_2}
        ladder = allocate_strategies(
            slots, pool,
            posture(caller_mode="batch", speed_budget="background", confidence_bar=0.70),
        )
        assert ladder.per_slot["external_context"] == ["s"]  # exactly one cheap rung
        assert ladder.per_slot_latency_ms["external_context"] == 100
        assert ladder.per_slot_status["external_context"] == "OPTIONAL"
        # required slot still chases normally
        assert len(ladder.per_slot["question"]) >= 2

    def test_optional_cap_does_not_inflate_query_latency(self):
        """The cmhc010 shape: without the cap the optional slot's chain doubled
        worst-case latency; with it, total latency tracks the required slot."""
        slots = [
            make_slot("question", priority=0),
            make_slot("best_guess", priority=1, required=False),
        ]
        pool = {"question": POOL_DEPTH_3, "best_guess": POOL_DEPTH_2}
        ladder = allocate_strategies(
            slots, pool,
            posture(caller_mode="batch", speed_budget="background", confidence_bar=0.70),
        )
        assert ladder.total_estimated_ms == ladder.per_slot_latency_ms["question"]

    def test_all_optional_slots_is_vacuously_cleared(self):
        """Zero required slots (only supplementary ones) → nothing the user
        asked is gated → all_slots_cleared even if every optional is weak."""
        slots = [make_slot("ctx", required=False)]
        ladder = allocate_strategies(
            slots, {"ctx": POOL_DEPTH_0},
            posture(caller_mode="batch", confidence_bar=0.70),
        )
        assert ladder.per_slot_status["ctx"] == "OPTIONAL"
        assert ladder.outcome == "all_slots_cleared"
        assert ladder.feasible is True


class TestStrategyEligibilityBySemantics:
    """Ananth's catch: strategy eligibility must match the slot's ROLE.
    external_context slots are candidate-filtered to web/external — internal
    strategies (s,a,b,c) cannot serve them; their priors are meaningless there."""

    def test_external_context_slot_only_considers_d(self):
        slots = [make_slot("ext", slot_semantics="external_context")]
        from app.services.router.tracing import DecisionTrace
        trace = DecisionTrace()
        ladder = allocate_strategies(
            slots, {"ext": POOL_DEPTH_2},
            posture(caller_mode="batch", speed_budget="background", confidence_bar=0.70),
            trace=trace,
        )
        assert ladder.per_slot["ext"] == ["d"]  # ONLY eligible strategy
        # ineligible rungs recorded as skips with the semantic reason
        skips = {s.strategy_id: s.skip_reason
                 for s in trace.slots[0].steps if s.action == "skipped"}
        for sid in ("s", "a", "b", "c"):
            assert skips.get(sid) == "ineligible_for_external_context"

    def test_direct_answer_slot_unrestricted(self):
        """RE-DERIVED 2026-08-05 (best-LB-first): same {s,a,b,c} set as
        before (unrestricted eligibility, unchanged), reached via best-LB
        order a>s>d>c>b instead of the old fixed s,a,b,c,d — see
        test_single_slot_depth2_deterministic_chain_lb_enforced for the
        full derivation of this exact scenario (same posture/pool)."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2}, posture(),
        )
        assert ladder.per_slot["slot_0"] == ["a", "s", "d", "c"]

    def test_unknown_semantics_defaults_to_full_set(self):
        slots = [make_slot("x", slot_semantics="future_new_role")]
        ladder = allocate_strategies(
            slots, {"x": POOL_DEPTH_2},
            posture(caller_mode="batch", confidence_bar=0.7),
        )
        assert len(ladder.per_slot["x"]) >= 2  # not over-restricted

    def test_optional_external_context_gets_d_not_bogus_s(self):
        """The cmhc004 shape post-fix: optional external slot's single cheap
        rung is chosen among ELIGIBLE strategies — 'd' is the only one, so the
        bogus internal 's' pick is gone even though 's' is cheaper."""
        slots = [
            make_slot("question", priority=0),
            make_slot("external_context", priority=1, required=False,
                      slot_semantics="external_context"),
        ]
        pool = {"question": POOL_DEPTH_3, "external_context": POOL_DEPTH_0}
        ladder = allocate_strategies(
            slots, pool,
            posture(caller_mode="batch", speed_budget="background", confidence_bar=0.70),
        )
        assert ladder.per_slot["external_context"] == ["d"]
        assert ladder.per_slot_status["external_context"] == "OPTIONAL"
        assert ladder.outcome == "all_slots_cleared"  # still never gates


class TestTagGating:
    """Eval-ratified second eligibility dimension: strategy s (Payor Fact
    Store) is gated on a payor j_code from GateResult.j_codes. FAIL CLOSED:
    no payor code → s never planned (its clean-miss would waste a call and
    inflate planned confidence; its calibration cells stay P(success|tag))."""

    def test_strategy_tag_eligible_unit(self):
        from app.services.router.allocation import strategy_tag_eligible
        assert strategy_tag_eligible("s", ["payor.sunshine_health"]) is True
        assert strategy_tag_eligible("s", ["payor.aetna", "florida"]) is True
        assert strategy_tag_eligible("s", ["florida.medicaid"]) is False  # j-code, not payor.*
        assert strategy_tag_eligible("s", []) is False    # fail closed
        assert strategy_tag_eligible("s", None) is False  # fail closed
        assert strategy_tag_eligible("a", []) is True     # untagged strategies unaffected

    def test_payor_query_plans_s(self):
        """RE-DERIVED 2026-08-05 (best-LB-first): s is still PLANNED (tag
        gate passes, payor code present) but no longer necessarily FIRST —
        at depth_2 a's LB (.2815) beats s's (.2486), so best-LB order puts
        a first. s still appears in the chain (2nd rung here)."""
        ladder = allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture())
        assert "s" in ladder.per_slot["slot_0"]  # helper posture carries payor code

    def test_non_payor_query_never_plans_s(self):
        from app.services.router.tracing import DecisionTrace
        trace = DecisionTrace()
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(gate_j_codes=[]),  # no payor signal → fail closed
            trace=trace,
        )
        assert "s" not in ladder.per_slot["slot_0"]
        assert ladder.per_slot["slot_0"][0] == "a"  # chain starts at next cheap rung
        skips = {s.strategy_id: s.skip_reason
                 for s in trace.slots[0].steps if s.action == "skipped"}
        assert skips.get("s") == "tag_gated_no_payor_j_code"

    def test_non_payor_planned_confidence_not_inflated(self):
        """The core of the bug: without the gate, s's seed lift (.5) inflated
        every chain's planned confidence with a contribution that could never
        materialize on non-payor queries."""
        with_s = allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture())
        without_s = allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2},
                                        posture(gate_j_codes=[]))
        assert "s" in with_s.per_slot["slot_0"]
        assert "s" not in without_s.per_slot["slot_0"]
        # both are honest ladders for their own query type — the point is the
        # non-payor plan no longer carries s's phantom lift in its math
        assert without_s.per_slot_lb["slot_0"] != with_s.per_slot_lb["slot_0"]

    def test_optimizer_gate_matches_greedy(self):
        from app.services.router.optimizer import optimize_allocation
        ladder = optimize_allocation(
            [make_slot()], {"slot_0": POOL_DEPTH_2}, posture(gate_j_codes=[]),
        )
        assert "s" not in ladder.per_slot["slot_0"]

    def test_bayesian_inherits_gate(self):
        from app.services.router.bayesian_optimizer import optimize_allocation_bayesian
        ladder = optimize_allocation_bayesian(
            [make_slot()], {"slot_0": POOL_DEPTH_2}, posture(gate_j_codes=[]),
        )
        assert "s" not in ladder.per_slot["slot_0"]


class TestCrawlGating:
    """Third eligibility dimension: DISABLED 2026-07-24 (Ananth via Retriever
    — payer_crawlable measures OUR fetcher against the payor's OWN domain,
    but d is a general web search that surfaces third-party sources
    regardless; the gate had the premise backwards). Dormant, not deleted —
    CRAWL_GATED_STRATEGIES=frozenset() today; these tests pin the CURRENT
    (disabled) behavior AND keep strategy_crawl_eligible's mechanics honest
    so a future re-enable (restore frozenset({"d"})) is a one-line flip with
    working tests already in place, not a rebuild."""

    def test_strategy_crawl_eligible_unit_disabled_state(self):
        from app.services.router.allocation import strategy_crawl_eligible
        # nothing is crawl-gated today — d included, regardless of verdict
        assert strategy_crawl_eligible("d", True) is True
        assert strategy_crawl_eligible("d", None) is True
        assert strategy_crawl_eligible("d", False) is True   # the flip: was False
        assert strategy_crawl_eligible("a", False) is True

    def test_non_crawlable_payor_still_plans_d(self):
        """THE regression this whole change targets: a payor whose own site
        is non-crawlable no longer starves d — general web search still runs."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(payer_crawlable=False, caller_mode="batch",
                    speed_budget="background", confidence_bar=0.99),
        )
        assert "d" in ladder.per_slot["slot_0"]

    def test_unknown_crawlability_keeps_d_eligible(self):
        """None (no payer / no verdict) → d stays in play — unaffected by
        this change either way (None always fell through to eligible)."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(caller_mode="batch", speed_budget="background",
                    confidence_bar=0.99),  # payer_crawlable absent → None
        )
        assert "d" in ladder.per_slot["slot_0"]

    def test_non_crawlable_external_context_slot_no_longer_dead_ends(self):
        """The flipped sharp edge: external_context is d-ONLY (semantics
        gate, untouched) — previously an affirmatively non-crawlable payor
        left this slot with NO viable strategy (NO_VIABLE + fast-exit);
        now d serves it like any other payor, since the crawl gate no
        longer excludes d here either. This is the intended fix, not a
        regression: those slots used to dead-end for no good reason."""
        slots = [
            make_slot("question", priority=0),
            make_slot("ext", priority=1, slot_semantics="external_context"),
        ]
        pool = {"question": POOL_DEPTH_3, "ext": POOL_DEPTH_2}
        ladder = allocate_strategies(
            slots, pool,
            posture(payer_crawlable=False, caller_mode="batch",
                    speed_budget="background", confidence_bar=0.70),
        )
        assert ladder.per_slot["ext"] == ["d"]
        # required=True by default (make_slot) — d attempts but this
        # bar/depth combo doesn't clear it; the point is d RAN, not that it
        # won. Previously this was NO_VIABLE_STRATEGY (d never attempted).
        assert ladder.per_slot_status["ext"] == "UNDER_CONFIDENT"

    def test_regression_guard_reenable_is_a_one_line_flip(self):
        """If CRAWL_GATED_STRATEGIES is ever restored to {"d"}, the mechanics
        (strategy_crawl_eligible's fail-open-on-None / fail-closed-on-False
        logic) must still be correct — this test exercises that logic
        directly against a hypothetical re-enable so nobody has to
        rediscover the semantics from scratch."""
        from app.services.router.allocation import strategy_crawl_eligible
        hypothetical_gated = frozenset({"d"})

        def _eligible_if_gated(strategy_id, payer_crawlable):
            if strategy_id not in hypothetical_gated:
                return True
            return payer_crawlable is not False

        assert _eligible_if_gated("d", True) is True
        assert _eligible_if_gated("d", None) is True
        assert _eligible_if_gated("d", False) is False
        # today's real function agrees only because the set is empty —
        # the moment it's restored, real and hypothetical converge
        assert strategy_crawl_eligible("d", False) is True  # disabled today

    def test_optimizer_and_bayesian_also_stop_gating_d(self):
        """Both optimizers share strategy_crawl_eligible — the disable
        applies identically across all allocators, no per-allocator drift."""
        from app.services.router.optimizer import optimize_allocation
        from app.services.router.bayesian_optimizer import optimize_allocation_bayesian
        for fn in (optimize_allocation, optimize_allocation_bayesian):
            ladder = fn(
                [make_slot()], {"slot_0": POOL_DEPTH_2},
                posture(payer_crawlable=False, caller_mode="batch",
                        speed_budget="background", confidence_bar=0.99),
            )
            assert "d" in ladder.per_slot["slot_0"], fn.__name__


class TestTerminalLeg:
    """Ananth's rule: e/q have NO priors and never compete in the chain math,
    but Router attaches them as the verdict-driven FINAL LEG of the plan."""

    def test_under_confident_required_slot_gets_clarify_terminal(self):
        """depth_2 @ bar .85 ends UNDER_CONFIDENT → terminal q attached."""
        ladder = allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture())
        assert ladder.per_slot_status["slot_0"] == "UNDER_CONFIDENT"
        assert ladder.per_slot_terminal["slot_0"] == "clarify_low_confidence"
        assert ladder.terminal_action == "clarify_low_confidence"
        # the chain itself NEVER contains e/q — they are the leg AFTER it
        assert not set(ladder.per_slot["slot_0"]) & {"e", "q"}

    def test_cleared_slot_gets_no_terminal(self):
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_1}, posture(confidence_bar=0.30),
        )
        assert ladder.per_slot_status["slot_0"] == "CLEARED"
        assert ladder.per_slot_terminal["slot_0"] is None
        assert ladder.terminal_action is None

    def test_no_viable_slot_gets_fast_exit_terminal(self):
        """Empty-bundle slot (no priors at all) → NO_VIABLE → terminal e."""
        from app.services.router.priors import PriorsBundle
        empty = PriorsBundle(by_depth={}, version="test", source="test")
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2}, posture(), bundle=empty,
        )
        assert ladder.per_slot_status["slot_0"] == "NO_VIABLE_STRATEGY"
        assert ladder.per_slot_terminal["slot_0"] == "fast_exit_no_viable"
        assert ladder.terminal_action == "fast_exit_no_viable"

    def test_clarify_wins_over_fast_exit_at_decision_level(self):
        """Mixed failures: one UNDER_CONFIDENT (q) + one NO_VIABLE (e) →
        decision terminal = q (if anything is worth asking about, ask)."""
        from app.services.router.priors import PriorsBundle, StrategyProfile
        bundle = PriorsBundle(
            by_depth={"a": {2: StrategyProfile(0.4, 500, 1, 0.5)}},  # depth_2 only
            version="test", source="test",
        )
        slots = [
            make_slot("askable", priority=0),        # depth_2 → chain [a], under-confident
            # required external_context: only 'd' eligible, and the bundle has
            # no 'd' cells → genuinely NO_VIABLE (the qclass fallback can't
            # rescue an ineligible strategy)
            make_slot("hopeless", priority=1, slot_semantics="external_context"),
        ]
        pool = {"askable": POOL_DEPTH_2, "hopeless": POOL_DEPTH_0}
        ladder = allocate_strategies(slots, pool, posture(caller_mode="batch"), bundle=bundle)
        assert ladder.per_slot_terminal["askable"] == "clarify_low_confidence"
        assert ladder.per_slot_terminal["hopeless"] == "fast_exit_no_viable"
        assert ladder.terminal_action == "clarify_low_confidence"

    def test_optional_slot_never_gets_terminal(self):
        slots = [
            make_slot("question", priority=0),
            make_slot("ctx", priority=1, required=False),
        ]
        pool = {"question": POOL_DEPTH_3, "ctx": POOL_DEPTH_0}
        ladder = allocate_strategies(
            slots, pool,
            posture(caller_mode="batch", speed_budget="background", confidence_bar=0.70),
        )
        assert ladder.per_slot_terminal["ctx"] is None  # OPTIONAL: no terminal leg
        assert ladder.terminal_action is None           # cleared required slot → none

    def test_terminal_persisted_and_narrated(self):
        """Terminal rides the trace and the narration."""
        from app.services.router.tracing import DecisionTrace
        from app.services.router.router_narrate import narrate
        trace = DecisionTrace(mode="greedy")
        allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture(), trace=trace)
        assert trace.slots[0].terminal_action == "clarify_low_confidence"
        text = narrate(trace)
        assert "Terminal leg: clarify_low_confidence" in text


@pytest.mark.live_priors
class TestFileBundleIsDefault:
    def test_default_bundle_comes_from_yaml_file(self):
        bundle = load_priors(force_reload=True)
        assert bundle.source == "file"
        assert bundle.version.startswith("file:priors_bootstrap.yaml@")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestHelperLayer:
    """Ananth's helper layer: when the recall loop fails, helpers solve the
    user's pain — clarify (ask) and sitemap_links (point at real payer pages).
    Zero priors, never in chain math; sitemap gated on payor identity."""

    def test_under_confident_payor_query_gets_clarify_plus_sitemap(self):
        ladder = allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture())
        assert ladder.per_slot_status["slot_0"] == "UNDER_CONFIDENT"
        assert ladder.per_slot_helpers["slot_0"] == ["clarify_low_confidence", "sitemap_links"]
        assert ladder.helpers == ["clarify_low_confidence", "sitemap_links"]

    def test_under_confident_non_payor_gets_clarify_only(self):
        """No payor identity → sitemap has nothing to look up in discovered_sources."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2}, posture(gate_j_codes=[]),
        )
        assert ladder.per_slot_helpers["slot_0"] == ["clarify_low_confidence"]
        assert "sitemap_links" not in ladder.helpers

    def test_payor_without_d_codes_gets_clarify_only(self):
        """Sitemap's lookup needs a d:-tag topic match too — payer identity
        alone deterministically returns [] (their real contract), so Router
        doesn't plan an aid that can't produce."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2}, posture(gate_d_codes=[]),
        )
        assert ladder.per_slot_helpers["slot_0"] == ["clarify_low_confidence"]
        assert "sitemap_links" not in ladder.helpers

    def test_low_confidence_payor_query_gets_clarify_and_sitemap(self):
        """UPDATED 2026-07-24 (crawl gate disabled): d now attempts even on
        an affirmatively non-crawlable payor and doesn't clear the bar here
        → UNDER_CONFIDENT, not NO_VIABLE_STRATEGY. Helper set reflects that:
        clarify (low confidence) AND sitemap (still a useful pointer) both
        fire — previously d never even ran, so only fast_exit+sitemap did."""
        slots = [make_slot("ext", slot_semantics="external_context")]
        ladder = allocate_strategies(
            slots, {"ext": POOL_DEPTH_2},
            posture(payer_crawlable=False, caller_mode="batch",
                    speed_budget="background", confidence_bar=0.70),
        )
        assert ladder.per_slot["ext"] == ["d"]
        assert ladder.per_slot_status["ext"] == "UNDER_CONFIDENT"
        assert ladder.per_slot_helpers["ext"] == ["clarify_low_confidence", "sitemap_links"]
        assert ladder.per_slot_terminal["ext"] == "clarify_low_confidence"

    def test_cleared_slot_gets_no_helpers(self):
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_1}, posture(confidence_bar=0.30),
        )
        assert ladder.per_slot_helpers["slot_0"] == []
        assert ladder.helpers == []

    def test_helpers_never_in_recall_chain(self):
        """The boundary holds: helper names never appear as chain rungs."""
        ladder = allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture())
        assert not set(ladder.per_slot["slot_0"]) & {"clarify_low_confidence",
                                                     "sitemap_links", "e", "q", "f"}

    def test_helpers_persisted_and_narrated(self):
        from app.services.router.tracing import DecisionTrace
        from app.services.router.router_narrate import narrate
        trace = DecisionTrace(mode="greedy")
        allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture(), trace=trace)
        assert trace.helpers == ["clarify_low_confidence", "sitemap_links"]
        text = narrate(trace)
        assert "HELPERS (recall loop fell short" in text
        assert "sitemap_links" in text


class TestGateCodePrefixNormalization:
    """Live-evidence regression guard: Gate codes arrive WITH kind prefixes
    ("j:payor.x", "d:claims.y") despite contracts.py's no-prefix comment —
    proven by Retriever's first live run and the legacy strip in
    corpus_search.py. The predicates must accept both forms."""

    def test_prefixed_and_bare_payor_codes_both_gate_s_open(self):
        from app.services.router.allocation import has_payor_code, strategy_tag_eligible
        assert has_payor_code(["j:payor.sunshine_health"]) is True   # live wire form
        assert has_payor_code(["payor.sunshine_health"]) is True     # documented form
        assert has_payor_code(["j:florida.medicaid"]) is False       # j-code, not payor
        assert strategy_tag_eligible("s", ["j:payor.sunshine_health"]) is True

    def test_live_run_shape_plans_s(self):
        """The exact regression: prefixed j_codes (real wire format) must not
        silently exclude s from a genuine payor query's chain.

        RE-DERIVED 2026-08-05 (best-LB-first): the assertion is about s
        being PLANNED at all (the actual regression this guards), not
        about s being first — best-LB order no longer guarantees that."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(gate_j_codes=["j:payor.sunshine_health"],
                    gate_d_codes=["d:claims.timely_filing"]),
        )
        assert "s" in ladder.per_slot["slot_0"]
        assert "sitemap_links" in ladder.helpers  # helper gate normalized too


class TestTokenPayloadBudget:
    """Ananth 2026-07-23: 'keep track of tokens you can send — extracting a
    whole provider manual is no use.' Per-rung gate: capacity × per-chunk
    tokens ≤ token_allowance_per_slot. Payload is NOT additive along a chain
    (single winning rung fills the slot) — worst case = MAX over rungs."""

    def test_defaults_do_not_change_behavior(self):
        """RE-DERIVED 2026-08-05 (best-LB-first): at the generous default
        allowance (8000), same {s,a,b,c}... now {a,s,d,c} member SET
        reached via best-LB order (d out-competes b's LB at depth_2)."""
        ladder = allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture())
        assert ladder.per_slot["slot_0"] == ["a", "s", "d", "c"]

    def test_payload_accounting_math(self):
        """RE-DERIVED 2026-08-05 (best-LB-first): PARTIAL-FILL + SUM model
        (retention live): capacity-5 chain [a,s,d,c] at default budget 8000
        — floor seats all, value knapsack fills all to cap: {a:5,s:5,d:5,c:5}
        → Σ = 750(s) + 1250(a) + 2500(d, 500/chunk) + 1250(c) = 5750."""
        ladder = allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture())
        assert ladder.per_slot["slot_0"] == ["a", "s", "d", "c"]
        assert ladder.per_slot_portfolio["slot_0"] == {"a": 5, "s": 5, "d": 5, "c": 5}
        assert ladder.per_slot_payload_tokens["slot_0"] == 5750  # SUM of fills
        assert ladder.total_payload_tokens == 5750

    def test_production_shape_regression_capacity10_budget3000(self):
        """THE cmhc002-empties regression (2026-07-24): real slots are
        capacity=10 and Structure sends token_budget=3000. Under the old
        1000-token guess, a/b/c demanded 10,000 — deterministically gated on
        EVERY query (and on the 8000 default too), collapsing all ladders to
        ['s']. With measured 250/chunk: a/b/c=2500 ≤ 3000 → real retrieval
        chains return.

        RE-DERIVED 2026-08-05 (best-LB-first): the SPECIFIC member set is
        no longer {s,a,b,c} — d now out-competes b on LB at depth_2, so
        the chain is [a,s,d,c] at both budgets (d joins as a PARTIAL fill,
        Ananth's "partial-d beats no-d" — its full 10x500=5000 fill still
        doesn't fit 3000, but one chunk does). The regression this guards
        (capacity-10 doesn't collapse to a bare ['s']) still holds — check
        that, not a specific strategy-b guarantee that no longer applies
        under best-LB-first."""
        slot10 = AnswerSlot(slot_id="slot_0", slot_semantics="direct_answer",
                            capacity=10, rewritten_query="q", required=True,
                            priority=0)
        for budget in (3000, 8000):
            ladder = allocate_strategies(
                [slot10], {"slot_0": POOL_DEPTH_2},
                posture(token_allowance_per_slot=budget))
            chain = ladder.per_slot["slot_0"]
            assert "a" in chain, (budget, chain)
            assert chain != ["s"], "capacity-10 collapse regressed"
            assert len(chain) > 1, "real retrieval chain, not a bare fallback"

    def test_partial_fill_brings_d_back_at_live_shape(self):
        """ANANTH'S DIRECTIVE (partial-d beats no-d): capacity=10 ×
        budget=3000 — the exact live shape where the old all-or-nothing gate
        starved d (0/22). Now d joins at a scoped fill, budget-tight."""
        slot10 = AnswerSlot(slot_id="slot_0", slot_semantics="direct_answer",
                            capacity=10, rewritten_query="q", required=True,
                            priority=0)
        ladder = allocate_strategies(
            [slot10], {"slot_0": POOL_DEPTH_2},
            posture(speed_budget="background", token_allowance_per_slot=3000),
        )
        fills = ladder.per_slot_portfolio["slot_0"]
        assert "d" in ladder.per_slot["slot_0"] and fills["d"] >= 1
        assert ladder.per_slot_payload_tokens["slot_0"] <= 3000  # by construction

    def test_skip_reason_recorded_in_trace(self):
        """Payload skip now = can't afford even ONE chunk: allowance 400
        admits s(150)/a(250)/b/c but not d(500); b/c seat no floor after
        s+a consume 400 → dropped with the assignment reason."""
        from app.services.router.tracing import DecisionTrace
        slot10 = AnswerSlot(slot_id="slot_0", slot_semantics="direct_answer",
                            capacity=10, rewritten_query="q", required=True,
                            priority=0)
        trace = DecisionTrace(mode="greedy")
        ladder = allocate_strategies(
            [slot10], {"slot_0": POOL_DEPTH_2},
            posture(speed_budget="background", token_allowance_per_slot=400),
            trace=trace,
        )
        reasons = {s.strategy_id: s.skip_reason for s in trace.slots[0].steps
                   if s.action == "skipped"}
        assert reasons.get("d") == "payload_over_token_allowance"  # 500 > 400
        assert ladder.per_slot["slot_0"] == ["a", "s"]  # RE-DERIVED 2026-08-05 (best-LB-first): reordered, b/c fill-dropped
        assert reasons.get("b") == "budget_fill_zero_after_assignment"
        assert trace.slots[0].payload_tokens_worst_case == 400  # 150+250

    def test_binding_constraint_payload_budget_exhausted(self):
        """When ONLY the payload budget blocks further rungs, the stop reason
        says so (allowance below every strategy's payload → empty chain)."""
        from app.services.router.tracing import DecisionTrace
        trace = DecisionTrace(mode="greedy")
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(token_allowance_per_slot=100), trace=trace,
        )
        assert ladder.per_slot["slot_0"] == []
        assert ladder.per_slot_status["slot_0"] == "NO_VIABLE_STRATEGY"
        assert trace.slots[0].stop_reason == "payload_budget_exhausted"

    def test_optimizer_and_bayesian_respect_gate(self):
        from app.services.router.optimizer import optimize_allocation
        from app.services.router.bayesian_optimizer import optimize_allocation_bayesian
        slot10 = AnswerSlot(slot_id="slot_0", slot_semantics="direct_answer",
                            capacity=10, rewritten_query="q", required=True,
                            priority=0)
        for fn in (optimize_allocation, optimize_allocation_bayesian):
            ladder = fn(
                [slot10], {"slot_0": POOL_DEPTH_2},
                posture(speed_budget="background", token_allowance_per_slot=3000),
            )
            chain = ladder.per_slot["slot_0"]
            assert "d" in chain, f"{fn.__name__} still starves d: {chain}"
            assert ladder.per_slot_payload_tokens["slot_0"] <= 3000
            assert ladder.total_payload_tokens == ladder.per_slot_payload_tokens["slot_0"]

    def test_narration_shows_payload(self):
        from app.services.router.tracing import DecisionTrace
        from app.services.router.router_narrate import narrate
        trace = DecisionTrace(mode="greedy")
        allocate_strategies([make_slot()], {"slot_0": POOL_DEPTH_2}, posture(),
                            trace=trace)
        # RE-DERIVED 2026-08-05 (best-LB-first): chain [a,s,d,c] not
        # [s,a,b,c] — d's 500/chunk (vs b's 250) shifts the sum to 5750.
        assert "worst-case payload 5750 tokens" in narrate(trace)

    def test_estimate_driven_skip_logs_warning(self, caplog):
        """Eval's guard: a skip keyed on an UNMEASURED per-chunk estimate must
        be loud (WARNING), never silent. Post-measurement (2026-07-24) only
        s remains an estimate: allowance 700 gates s (750, warns) AND a/b/c
        (1250, measured — silent)."""
        import logging
        with caplog.at_level(logging.WARNING, logger="app.services.router.allocation"):
            allocate_strategies(
                [make_slot()], {"slot_0": POOL_DEPTH_2},
                posture(token_allowance_per_slot=100),  # under every per-chunk cost
            )
        warned = [r for r in caplog.records if "UNMEASURED" in r.getMessage()]
        assert any("'s'" in w.getMessage() for w in warned)   # estimate → loud
        assert not any("'a'" in w.getMessage() for w in warned)  # measured → silent


class TestAuthorityGate:
    """Ananth 2026-07-23: caller-declared citability bifurcation — 'for an
    appeal this is important, but for a chat call?' d (live web) is accurate
    but NOT citable to a payor; the CALLER declares whether that matters."""

    def test_default_fail_open_d_still_competes(self):
        """No declaration → 'any' → d plans exactly as before (zero change)."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(caller_mode="batch", confidence_bar=0.99,
                    speed_budget="background"),  # 75000ms allowance fits full chain
        )
        assert "d" in ladder.per_slot["slot_0"]

    def test_citable_required_gates_d_off_required_slot(self):
        from app.services.router.tracing import DecisionTrace
        trace = DecisionTrace(mode="greedy")
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(caller_mode="batch", confidence_bar=0.99,
                    speed_budget="background",
                    authority_requirement="citable_required"),
            trace=trace,
        )
        assert "d" not in ladder.per_slot["slot_0"]
        reasons = {s.strategy_id: s.skip_reason for s in trace.slots[0].steps
                   if s.action == "skipped"}
        assert reasons.get("d") == "authority_gated_non_citable"

    def test_optional_external_context_keeps_d_under_citable_required(self):
        """Web context is still useful CONTEXT even when it can't be evidence:
        the gate applies to required (evidence-bearing) slots only."""
        slots = [make_slot("core", priority=0),
                 make_slot("ext", priority=1, required=False,
                           slot_semantics="external_context")]
        pool = {"core": POOL_DEPTH_2, "ext": POOL_DEPTH_2}
        ladder = allocate_strategies(
            slots, pool,
            posture(authority_requirement="citable_required"),
        )
        assert ladder.per_slot["ext"] == ["d"]  # external slot keeps its only strategy
        assert "d" not in ladder.per_slot["core"]

    def test_optimizer_and_bayesian_respect_authority_gate(self):
        from app.services.router.optimizer import optimize_allocation
        from app.services.router.bayesian_optimizer import optimize_allocation_bayesian
        for fn in (optimize_allocation, optimize_allocation_bayesian):
            ladder = fn(
                [make_slot()], {"slot_0": POOL_DEPTH_2},
                posture(caller_mode="batch", confidence_bar=0.99,
                        speed_budget="background",
                        authority_requirement="citable_required"),
            )
            assert "d" not in ladder.per_slot["slot_0"], fn.__name__


class TestAuthorityPriorThreshold:
    """Eval's 2026-08-05 proposal: the gate should ALSO read a per-strategy
    authority PRIOR (continuous, threshold-gated), additive to the legacy
    hardcoded NON_CITABLE_STRATEGIES set — not a replacement. Unpopulated
    file (authority defaults to 1.0) must reproduce today's behavior
    exactly; only a real sub-threshold measurement adds a new exclusion."""

    def test_default_authority_unpopulated_changes_nothing(self):
        from app.services.router.allocation import strategy_authority_eligible
        # No prior passed → default 1.0 → above threshold → eligible,
        # same as pre-change behavior for every strategy except d.
        assert strategy_authority_eligible("c", "citable_required", True) is True
        assert strategy_authority_eligible("a", "citable_required", True) is True

    def test_legacy_set_still_excludes_d_regardless_of_authority_value(self):
        from app.services.router.allocation import strategy_authority_eligible
        # Even a HIGH authority value doesn't rescue d — the legacy
        # hardcoded classification is a floor, not overridden by the prior.
        assert strategy_authority_eligible("d", "citable_required", True,
                                           authority=0.99) is False

    def test_low_authority_prior_excludes_a_non_legacy_strategy(self):
        from app.services.router.allocation import strategy_authority_eligible
        assert strategy_authority_eligible("c", "citable_required", True,
                                           authority=0.3) is False
        assert strategy_authority_eligible("c", "citable_required", True,
                                           authority=0.7) is True

    def test_threshold_only_bites_under_citable_required(self):
        from app.services.router.allocation import strategy_authority_eligible
        # "any" (default/no declaration) fail-opens regardless of authority.
        assert strategy_authority_eligible("c", "any", True, authority=0.0) is True

    def test_integration_low_authority_strategy_excluded_from_ladder(
        self, tmp_path, monkeypatch
    ):
        """End-to-end: a strategy with a real sub-threshold `authority` cell
        in the priors file is excluded from a citable_required REQUIRED slot
        via allocate_strategies — proving the mechanism works, not just the
        unit function."""
        import textwrap
        yaml_path = tmp_path / "low_authority.yaml"
        yaml_path.write_text(textwrap.dedent("""
            seed_priors:
              depth_2:
                a:
                  recall_lift: 0.9
                  latency_p50_ms: 500
                  cost_per_attempt: 1
                  accuracy_estimate: 0.5
                  authority: 0.95
                c:
                  recall_lift: 0.9
                  latency_p50_ms: 500
                  cost_per_attempt: 1
                  accuracy_estimate: 0.5
                  authority: 0.2
        """))
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_path))
        from app.services.router.tracing import DecisionTrace
        trace = DecisionTrace(mode="greedy")
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(caller_mode="batch", confidence_bar=0.99,
                    speed_budget="background",
                    authority_requirement="citable_required"),
            trace=trace,
        )
        assert "c" not in ladder.per_slot["slot_0"]
        assert "a" in ladder.per_slot["slot_0"]
        reasons = {s.strategy_id: s.skip_reason for s in trace.slots[0].steps
                   if s.action == "skipped"}
        assert reasons.get("c") == "authority_gated_non_citable"


class TestExecutionReorderingContract:
    """Ananth via Web Search (2026-07-23): the execution loop may check live
    readiness (e.g. d's prescreen_search_task.done()) and run a DIFFERENT
    planned rung first when d isn't ready. This is PLAN-PRESERVING by
    construction: every §2a-enforced quantity is invariant under permutation
    of the chain — the ladder is a rung SET with a preferred order, not a
    schedule. This test is the contract: if any enforced quantity ever
    becomes order-dependent, this fails and the reordering permission must
    be renegotiated with the orchestrator."""

    def test_enforced_quantities_invariant_under_chain_permutation(self):
        """BOTH bound tracks guarded (Ananth's catch: Wilson AND Bayesian must
        hold — the optimizers differ ONLY in lb_fn, and chain_lb composes
        either through the same commutative product 1−Π(1−lb_i), so the
        reordering permission must be proven per-bound, not assumed)."""
        from itertools import permutations
        from app.services.router.allocation import chain_lb
        from app.services.router.priors import (
            StrategyProfile, beta_lower_bound, wilson_lower_bound,
        )
        profiles = [
            StrategyProfile(0.500, 100, 0, 0.5),   # s
            StrategyProfile(0.543, 500, 1, 0.5),   # a
            StrategyProfile(0.450, 3000, 3, 0.45), # d
        ]
        for lb_fn in (wilson_lower_bound, beta_lower_bound):
            base_lb = chain_lb(profiles, 0.95, lb_fn=lb_fn)
            base_mean = chain_success_probability([p.recall_lift for p in profiles])
            base_latency = sum(p.latency_p50_ms for p in profiles)
            base_cost = sum(p.cost for p in profiles)
            for perm in permutations(profiles):
                perm = list(perm)
                assert chain_lb(perm, 0.95, lb_fn=lb_fn) == \
                    pytest.approx(base_lb, abs=1e-12), lb_fn.__name__
                assert chain_success_probability(
                    [p.recall_lift for p in perm]) == pytest.approx(base_mean, abs=1e-12)
                assert sum(p.latency_p50_ms for p in perm) == base_latency
                assert sum(p.cost for p in perm) == base_cost


class TestLastAttemptRule:
    """cmhc002 collapse fix (Retriever's live trace, 2026-07-23): greedy's
    static cheap-first order degenerates at max_attempts=1 to 'always plan
    the cheapest alone' — planned ['s'] for content queries → 0 occupancy on
    19/22 bank queries. Rule: on a REQUIRED slot's FINAL available attempt,
    pick the strongest viable rung (max per-rung LB), matching the optimizer."""

    CMHC002_POOL = {"top_score_percentile": 0.83, "pool_size": 493}  # real, depth 2

    def test_cap1_picks_best_lb_not_cheapest(self):
        """The exact reproduction: greedy now matches optimizer's ['a']."""
        from app.services.router.optimizer import optimize_allocation
        posture_1 = posture(speed_budget="real_time", max_attempts_per_slot=1)
        g = allocate_strategies([make_slot()], {"slot_0": self.CMHC002_POOL}, posture_1)
        o = optimize_allocation([make_slot()], {"slot_0": self.CMHC002_POOL}, posture_1)
        assert g.per_slot["slot_0"] == ["a"]  # lb .2815 beats s's .2486
        assert g.per_slot["slot_0"] == o.per_slot["slot_0"]  # allocators agree

    def test_not_chosen_rungs_recorded_with_reason(self):
        from app.services.router.tracing import DecisionTrace
        trace = DecisionTrace(mode="greedy")
        allocate_strategies([make_slot()], {"slot_0": self.CMHC002_POOL},
                            posture(speed_budget="real_time", max_attempts_per_slot=1),
                            trace=trace)
        reasons = {s.strategy_id: s.skip_reason for s in trace.slots[0].steps
                   if s.action == "skipped"}
        # s is excluded by the supplement gate (fires before LB comparison);
        # other viable-but-weaker rungs get the LB reason
        assert reasons.get("s") == "supplement_only_not_sole_rung"
        assert reasons.get("b") == "single_shot_lower_lb_than_a"

    def test_optional_slot_keeps_cheapest_first(self):
        """Eval's optional-cheap ruling survives: optional slots are capped at
        1 attempt but must NOT best-LB-select — telemetry-only slots never
        pay wall-clock for confidence."""
        slots = [make_slot("core", priority=0),
                 make_slot("opt", priority=1, required=False)]
        pool = {"core": self.CMHC002_POOL, "opt": self.CMHC002_POOL}
        ladder = allocate_strategies(slots, pool, posture())
        assert ladder.per_slot["opt"] == ["s"]  # cheapest, NOT best-LB 'a'

    def test_longer_chain_final_rung_also_best_lb(self):
        """RE-DERIVED 2026-08-05: the rule now generalizes to EVERY rung, not
        just the final one (Ananth's confidence-density directive superseded
        the old cheap-first-except-last split) — cap 2 → both rungs are
        best-LB: 'a' (lb .2815, non-supplement winner — s is gated off the
        FIRST rung by SUPPLEMENT_ONLY while a is viable) then 's' (re-enters
        once the gate lifts, ties nothing left at this point, wins round 2)."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": self.CMHC002_POOL},
            posture(speed_budget="real_time", max_attempts_per_slot=2,
                    confidence_bar=0.99))
        assert ladder.per_slot["slot_0"] == ["a", "s"]


class TestSupplementGate:
    """Eval's category-error finding (cmhc002): s returns ONE certified fact —
    a code, not answer content. It supplements corpus retrieval, never
    substitutes: s may not be the SOLE planned rung of a required slot while
    anything else is viable. Critically closes the depth_1 TIED-lift edge
    (s=a=.35) where the last-attempt LB rule alone still handed the single
    shot to s via the tie-break."""

    DEPTH1_POOL = {"top_score_percentile": 0.80, "pool_size": 150}

    def test_tied_lb_single_shot_goes_to_corpus_not_s(self):
        from app.services.router.tracing import DecisionTrace
        trace = DecisionTrace(mode="greedy")
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": self.DEPTH1_POOL},
            posture(speed_budget="real_time", max_attempts_per_slot=1),
            trace=trace)
        assert ladder.per_slot["slot_0"] == ["a"]
        reasons = {s.strategy_id: s.skip_reason for s in trace.slots[0].steps
                   if s.action == "skipped"}
        assert reasons.get("s") == "supplement_only_not_sole_rung"

    def test_all_three_allocators_refuse_sole_s(self):
        from app.services.router.optimizer import optimize_allocation
        from app.services.router.bayesian_optimizer import optimize_allocation_bayesian
        p = posture(speed_budget="real_time", max_attempts_per_slot=1)
        for fn in (allocate_strategies, optimize_allocation, optimize_allocation_bayesian):
            chain = fn([make_slot()], {"slot_0": self.DEPTH1_POOL}, p).per_slot["slot_0"]
            assert chain != ["s"], fn.__name__
            assert len(chain) == 1  # cap still respected

    def test_s_still_leads_multi_rung_chains(self):
        """RE-DERIVED 2026-08-05 (best-LB-first): the supplement role is now
        STRONGER, not weaker — s is deferred to round 2, not the leader.
        At depth_1, s/a are LB-tied (.1452 each); the SUPPLEMENT_ONLY gate
        excludes s from the FIRST rung specifically because a (non-
        supplement) is viable, so a wins round 1. s re-enters once the
        gate lifts (chain non-empty) and gets added next. s still appears
        in the chain — supplementing, exactly its designed role — just
        never as the sole/first answer to the question."""
        ladder = allocate_strategies([make_slot()], {"slot_0": self.DEPTH1_POOL},
                                     posture())
        assert ladder.per_slot["slot_0"][0] == "a"
        assert ladder.per_slot["slot_0"][1] == "s"
        assert len(ladder.per_slot["slot_0"]) > 1

    def test_s_kept_when_nothing_else_viable(self):
        """Sole-s allowed as last resort: allowance fits only s (140ms —
        a/b/c/d all over-latency) → ['s'] beats an empty ladder."""
        ladder = allocate_strategies(
            [make_slot()], {"slot_0": self.DEPTH1_POOL},
            posture(speed_budget="real_time", max_attempts_per_slot=1,
                    caller_mode="custom_tight",
                    tolerance_bands={"max_pct": -0.93}))
        assert ladder.per_slot["slot_0"] == ["s"]
