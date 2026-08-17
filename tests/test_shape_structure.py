"""Tests for Step 1c SHAPE structure (app/services/retriever/shape/structure.py).

Pure unit tests only — Structure is pure compute, zero DB calls (confirmed
by DB sign-off, docs/rag-agents/shape-structure-simulation-tracker.md), so
unlike test_shape_gate.py / test_shape_reformat.py there's no DB-backed
integration layer here. Landed alongside the module per standing process —
TECH caught a "bank drafted, unit tests forgotten" gap on both Gate and
Reformat; not repeating that here.
"""

from __future__ import annotations

from app.services.retriever.shape.contracts import FanoutTheme, ReformatPosture, ReformatResult
from app.services.retriever.shape.structure import (
    DEFAULT_CALLER_MODE,
    _ACCURACY_NEED,
    _MAX_ATTEMPTS_CEILING,
    _MAX_ATTEMPTS_CEILING_OVERRIDE,
    _SPEED_BUDGET,
    _TOKEN_BUDGET,
    run_structure,
)


def _reformat(posture: ReformatPosture, query: str = "q", rewritten_queries=None) -> ReformatResult:
    return ReformatResult(
        query=query,
        posture=posture,
        rewritten_queries=rewritten_queries or [query],
    )


class TestNoRetrievalPostures:
    """CLARIFY / CLARIFY_REPHRASE / DECLINE — all-zero ResourcePosture, not None."""

    def test_clarify_gives_all_zero(self):
        result = run_structure(_reformat(ReformatPosture.CLARIFY))
        rp = result.resource_posture
        assert (rp.breadth, rp.confidence_bar, rp.max_attempts, rp.speed_budget, rp.token_budget) == (
            0,
            0.0,
            0,
            "none",
            0,
        )

    def test_clarify_rephrase_gives_all_zero(self):
        result = run_structure(_reformat(ReformatPosture.CLARIFY_REPHRASE))
        assert result.resource_posture.breadth == 0

    def test_decline_gives_all_zero(self):
        result = run_structure(_reformat(ReformatPosture.DECLINE))
        assert result.resource_posture.max_attempts == 0

    def test_no_retrieval_reason_is_explicit(self):
        result = run_structure(_reformat(ReformatPosture.DECLINE))
        assert "no retrieval" in result.reason


class TestPassthrough:
    def test_rewritten_queries_pass_through_unchanged(self):
        reformat = _reformat(ReformatPosture.PRECISE, rewritten_queries=["a", "b", "c"])
        result = run_structure(reformat)
        assert result.rewritten_queries == ["a", "b", "c"]

    def test_fanout_themes_pass_through_for_slots(self):
        # Regression test — Retriever caught 2026-07-23: StructureResult had
        # the field but run_structure() never populated it, blocking
        # Shape:Slots (Step 1d) from starting its build.
        themes = [FanoutTheme(theme_label="income eligibility", score=0.8)]
        reformat = ReformatResult(query="q", posture=ReformatPosture.FAN_OUT, fanout_themes=themes)
        result = run_structure(reformat)
        assert result.fanout_themes == themes

    def test_fanout_themes_empty_by_default_for_non_fan_out(self):
        reformat = _reformat(ReformatPosture.PRECISE)
        result = run_structure(reformat)
        assert result.fanout_themes == []

    def test_posture_carried_forward(self):
        reformat = _reformat(ReformatPosture.FAN_OUT)
        result = run_structure(reformat)
        assert result.posture == ReformatPosture.FAN_OUT

    def test_query_carried_forward(self):
        reformat = _reformat(ReformatPosture.PRECISE, query="how do I confirm eligibility")
        result = run_structure(reformat)
        assert result.query == "how do I confirm eligibility"


class TestCallerModeResolution:
    def test_known_caller_mode_used_directly(self):
        result = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode="chat.thinking")
        assert result.resource_posture.confidence_bar == _ACCURACY_NEED["chat.thinking"]

    def test_missing_caller_mode_falls_back_to_default(self):
        result = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode=None)
        assert result.resource_posture.confidence_bar == _ACCURACY_NEED[DEFAULT_CALLER_MODE]

    def test_unrecognized_caller_mode_falls_back_to_default_not_crash(self):
        # Regression test for the 3-way caller_mode vocabulary bug (schematic
        # spec §5 / simulation tracker): the one live path that sends
        # caller_mode today actually sends assembly_strategy values
        # ("score" / "canonical_first" / "balanced"), which match neither
        # CALLER_MODE_PRESETS nor _get_escalation_budget()'s vocabulary.
        # Structure must degrade to the default cell, not raise or silently
        # mis-key.
        result = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode="canonical_first")
        assert result.resource_posture.confidence_bar == _ACCURACY_NEED[DEFAULT_CALLER_MODE]
        assert "unrecognized" in result.reason
        assert "canonical_first" in result.reason


class TestResourcePostureByPosture:
    def test_precise_breadth_at_default_mode_matches_base_k(self):
        # chat.default IS DEFAULT_CALLER_MODE and the anchor recall_demand
        # the base K constants were measured against — breadth should come
        # out to exactly the base k=10, not a rounded-off approximation.
        result = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode="chat.default")
        assert result.resource_posture.breadth == 10

    def test_fan_out_breadth_at_default_mode_matches_base_fan_out_k(self):
        result = run_structure(_reformat(ReformatPosture.FAN_OUT), caller_mode="chat.default")
        assert result.resource_posture.breadth == 15

    def test_fan_out_breadth_wider_than_precise_same_mode(self):
        precise = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode="chat.thinking")
        fan_out = run_structure(_reformat(ReformatPosture.FAN_OUT), caller_mode="chat.thinking")
        assert fan_out.resource_posture.breadth > precise.resource_posture.breadth

    def test_max_attempts_is_uniform_ceiling_across_postures(self):
        # CORRECTED 2026-07-23 (Router caught): per-posture values were
        # amputating the fallback chain (min(time_derived, 1) == 1 always,
        # making the "soft ceiling" the actual plan). One uniform, generous
        # ceiling now — time (speed_budget) is the real constraint. Uses
        # chat.default deliberately, NOT chat.thinking -- chat.thinking has
        # its own real, live-evidence-backed override (see
        # test_max_attempts_chat_thinking_override below); this test is
        # about posture-uniformity for the flat-ceiling modes, unrelated.
        for posture in (ReformatPosture.PRECISE, ReformatPosture.FAN_OUT, ReformatPosture.RELY_ON_EXTERNAL):
            result = run_structure(_reformat(posture), caller_mode="chat.default")
            assert result.resource_posture.max_attempts == _MAX_ATTEMPTS_CEILING

    def test_max_attempts_identical_across_caller_modes_except_chat_thinking(self):
        # v1 (2026-07-23) explicitly did NOT modulate max_attempts by
        # caller_mode at all. SUPERSEDED 2026-08-15 (Ananth's call, live
        # evidence): chat.thinking's own generous latency_allowance_ms made
        # the flat ceiling=6 the actual binding constraint (confirmed live:
        # routing_keys.per_slot_verdict=="EXHAUSTED_ATTEMPTS" on a turn with
        # real time budget still unspent) -- raised to 10 for that mode
        # only. Every OTHER mode keeps the original v1 invariant: same
        # posture must give the same (flat) max_attempts regardless of mode.
        modes = list(_ACCURACY_NEED)
        results = {m: run_structure(_reformat(ReformatPosture.FAN_OUT), caller_mode=m) for m in modes}
        for mode, result in results.items():
            expected = _MAX_ATTEMPTS_CEILING_OVERRIDE.get(mode, _MAX_ATTEMPTS_CEILING)
            assert result.resource_posture.max_attempts == expected, mode
        assert results["chat.thinking"].resource_posture.max_attempts == 10
        non_thinking = {m: r.resource_posture.max_attempts for m, r in results.items() if m != "chat.thinking"}
        assert set(non_thinking.values()) == {_MAX_ATTEMPTS_CEILING}

    def test_speed_budget_reused_as_is_from_caller_mode(self):
        for mode, expected in _SPEED_BUDGET.items():
            result = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode=mode)
            assert result.resource_posture.speed_budget == expected

    def test_higher_recall_demand_mode_gives_wider_breadth(self):
        # research has the highest recall_demand (1.00); chat.copilot the
        # lowest (0.70) — breadth should reflect that ordering.
        research = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode="research")
        copilot = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode="chat.copilot")
        assert research.resource_posture.breadth > copilot.resource_posture.breadth

    def test_rely_on_external_confidence_and_speed_still_meaningful(self):
        # Eval sign-off: only breadth is left as a placeholder for
        # RELY_ON_EXTERNAL — confidence_bar/speed_budget still resolve for real.
        result = run_structure(_reformat(ReformatPosture.RELY_ON_EXTERNAL), caller_mode="auth_agent")
        assert result.resource_posture.confidence_bar == _ACCURACY_NEED["auth_agent"]
        assert result.resource_posture.speed_budget == _SPEED_BUDGET["auth_agent"]

    def test_token_budget_matches_caller_mode_table(self):
        # New field, 2026-07-23 (Ananth via Retriever) — per-slot ceiling on
        # evidence volume a Filler may hand back, grounded in Filler d's real
        # measurement (~2,021 tokens for a full d-slot). See schematic spec §11.
        for mode, expected in _TOKEN_BUDGET.items():
            result = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode=mode)
            assert result.resource_posture.token_budget == expected

    def test_token_budget_tighter_for_real_time_than_background(self):
        real_time = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode="chat.default")
        background = run_structure(_reformat(ReformatPosture.PRECISE), caller_mode="batch")
        assert real_time.resource_posture.token_budget < background.resource_posture.token_budget

    def test_token_budget_zero_for_no_retrieval_postures(self):
        result = run_structure(_reformat(ReformatPosture.DECLINE))
        assert result.resource_posture.token_budget == 0


class TestAuthorityRequirement:
    """Caller-DECLARED, not computed — Structure only threads it through
    (matches Router's allocation.py: AUTHORITY_ANY="any" default,
    AUTHORITY_CITABLE_REQUIRED="citable_required")."""

    def test_defaults_to_any_when_not_supplied(self):
        result = run_structure(_reformat(ReformatPosture.PRECISE))
        assert result.resource_posture.authority_requirement == "any"

    def test_citable_required_passes_through(self):
        result = run_structure(_reformat(ReformatPosture.PRECISE), authority_requirement="citable_required")
        assert result.resource_posture.authority_requirement == "citable_required"

    def test_unrecognized_value_falls_back_to_any_not_crash(self):
        result = run_structure(_reformat(ReformatPosture.PRECISE), authority_requirement="bogus_value")
        assert result.resource_posture.authority_requirement == "any"

    def test_no_retrieval_postures_default_to_any(self):
        result = run_structure(_reformat(ReformatPosture.DECLINE), authority_requirement="citable_required")
        assert result.resource_posture.authority_requirement == "any"


class TestTokenBudgetForRetrieval:
    """Caller-DECLARED (Chat's real context-window math), not guessed
    (2026-07-24, Ananth's correction: "RAG does not have to guess") --
    same pattern as authority_requirement. Falls through to the static
    _TOKEN_BUDGET table only when the caller omits it."""

    def test_defaults_to_static_table_when_not_supplied(self):
        result = run_structure(_reformat(ReformatPosture.PRECISE))
        # chat.default table value -- bumped 3000->6000 2026-08-16 (Ananth's
        # ask), paired with a matching latency-allowance bump; verified via
        # a real 22-query bank sweep (recall 0.6265->0.7022, ~0ms latency
        # cost) before deploy. Test was left stale until now.
        assert result.resource_posture.token_budget == 6000

    def test_caller_supplied_value_overrides_table(self):
        result = run_structure(_reformat(ReformatPosture.PRECISE), token_budget_for_retrieval=10_000)
        assert result.resource_posture.token_budget == 10_000

    def test_no_retrieval_postures_ignore_caller_value(self):
        result = run_structure(_reformat(ReformatPosture.DECLINE), token_budget_for_retrieval=10_000)
        assert result.resource_posture.token_budget == 0


class TestTiming:
    def test_structure_ms_recorded(self):
        result = run_structure(_reformat(ReformatPosture.PRECISE))
        assert result.structure_ms >= 0
