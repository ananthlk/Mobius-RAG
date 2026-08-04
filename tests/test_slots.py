"""Tests for Shape:Slots (Step 1d) — slot derivation logic.

Pure compute, no DB. Characterization tests: same input → byte-identical output
before/after refactors (SAFE-tagged changes).
"""

import pytest

from app.services.retriever.shape.contracts import (
    FanoutTheme,
    ReformatPosture,
    ResourcePosture,
    StructureResult,
)
from app.services.retriever.shape.slots import (
    AnswerShapeResult,
    AnswerSlot,
    run_slots,
)


class TestSlotsDeriveFromPosture:
    """Verify slot count and semantics per posture."""

    def test_precise_emits_one_direct_answer_slot(self):
        """PRECISE posture → 1 direct_answer slot."""
        structure = StructureResult(
            query="What are FL Medicaid eligibility requirements?",
            posture=ReformatPosture.PRECISE,
            resource_posture=ResourcePosture(
                breadth=5, confidence_bar=0.85, max_attempts=1, speed_budget="real_time"
            ),
            reason="EXACT contour",
        )

        result = run_slots(structure)

        assert len(result.slots) == 1
        slot = result.slots[0]
        assert slot.slot_id == "direct_answer"
        assert slot.slot_semantics == "direct_answer"
        assert slot.capacity == 5
        assert slot.required is True
        assert slot.priority == 0

    def test_fanout_emits_one_slot_per_theme(self):
        """FAN_OUT posture → N slots, one per theme, sorted by score, rewritten_query populated."""
        themes = [
            FanoutTheme(
                theme_label="Traditional Medicaid",
                member_codes=["trad"],
                score=0.85,
            ),
            FanoutTheme(
                theme_label="Managed Care",
                member_codes=["managed"],
                score=0.79,
            ),
            FanoutTheme(
                theme_label="Expansion",
                member_codes=["expand"],
                score=0.72,
            ),
        ]
        rewritten_queries = [
            "What is traditional Medicaid?",
            "What is managed care Medicaid?",
            "What are Medicaid expansion programs?",
        ]
        structure = StructureResult(
            query="What are the different types of Medicaid?",
            rewritten_queries=rewritten_queries,
            posture=ReformatPosture.FAN_OUT,
            fanout_themes=themes,
            resource_posture=ResourcePosture(
                breadth=9, confidence_bar=0.80, max_attempts=2, speed_budget="interactive"
            ),
            reason="explore_siblings",
        )

        result = run_slots(structure)

        assert len(result.slots) == 3
        # Sorted by theme.score descending: 0.85 → 0.79 → 0.72
        assert result.slots[0].slot_id == "fanout_0"
        assert result.slots[0].priority == 0  # Highest score = priority 0
        assert result.slots[0].slot_semantics == "thematic_exploration"
        assert result.slots[0].capacity == 3  # breadth=9 / 3 themes
        assert result.slots[0].rewritten_query == "What is traditional Medicaid?"

        assert result.slots[1].slot_id == "fanout_1"
        assert result.slots[1].priority == 1
        assert result.slots[1].capacity == 3
        assert result.slots[1].rewritten_query == "What is managed care Medicaid?"

        assert result.slots[2].slot_id == "fanout_2"
        assert result.slots[2].priority == 2
        assert result.slots[2].capacity == 3
        assert result.slots[2].rewritten_query == "What are Medicaid expansion programs?"

    def test_clarify_emits_no_slots(self):
        """CLARIFY posture → 0 slots (Chat asks clarifying questions)."""
        structure = StructureResult(
            query="What do I need?",
            posture=ReformatPosture.CLARIFY,
            resource_posture=ResourcePosture(
                breadth=0, confidence_bar=0.0, max_attempts=0, speed_budget="none"
            ),
            reason="missing domain and jurisdiction",
        )

        result = run_slots(structure)

        assert len(result.slots) == 0

    def test_rely_on_external_emits_one_optional_slot(self):
        """RELY_ON_EXTERNAL posture → 1 external_context slot (optional, fallback)."""
        structure = StructureResult(
            query="What's the latest FDA approval for [rare drug]?",
            posture=ReformatPosture.RELY_ON_EXTERNAL,
            resource_posture=ResourcePosture(
                breadth=5, confidence_bar=0.70, max_attempts=1, speed_budget="interactive"
            ),
            reason="Corpus gap: bleeding-edge pharma not covered",
        )

        result = run_slots(structure)

        assert len(result.slots) == 1
        slot = result.slots[0]
        assert slot.slot_id == "external_context"
        assert slot.slot_semantics == "external_context"
        assert slot.capacity == 5
        assert slot.required is False  # Optional fallback
        assert slot.priority == 1  # Lower than default

    def test_decline_emits_no_slots(self):
        """DECLINE posture → 0 slots (out of scope)."""
        structure = StructureResult(
            query="How do I become a doctor?",
            posture=ReformatPosture.DECLINE,
            resource_posture=ResourcePosture(
                breadth=0, confidence_bar=0.0, max_attempts=0, speed_budget="none"
            ),
            reason="Out of scope (not healthcare operations)",
        )

        result = run_slots(structure)

        assert len(result.slots) == 0

    def test_clarify_rephrase_emits_one_optimistic_slot(self):
        """CLARIFY_REPHRASE posture → 1 optimistic slot (best guess, Synthesis handles uncertainty)."""
        structure = StructureResult(
            query="How do I eligibility?",
            posture=ReformatPosture.CLARIFY_REPHRASE,
            resource_posture=ResourcePosture(
                breadth=5, confidence_bar=0.5, max_attempts=1, speed_budget="real_time"
            ),
        )

        result = run_slots(structure)

        assert len(result.slots) == 1
        slot = result.slots[0]
        assert slot.slot_id == "best_guess"
        assert slot.slot_semantics == "direct_answer"
        assert slot.capacity == 5
        assert slot.required is False  # Optional — may fail if guess is wrong
        assert "uncertainty" in result.reason.lower() or "best_guess" in result.reason.lower()


class TestCapacityDerivation:
    """Verify capacity derives from resource_posture.breadth."""

    def test_capacity_equals_breadth_for_single_slot(self):
        """Single-slot postures: capacity = breadth."""
        for posture in [ReformatPosture.PRECISE, ReformatPosture.RELY_ON_EXTERNAL]:
            structure = StructureResult(
                query="test",
                posture=posture,
                resource_posture=ResourcePosture(
                    breadth=10,
                    confidence_bar=0.5,
                    max_attempts=1,
                    speed_budget="interactive",
                ),
            )
            result = run_slots(structure)
            if len(result.slots) > 0:
                assert result.slots[0].capacity == 10

    def test_capacity_distributed_across_fanout_themes(self):
        """FAN_OUT: capacity_per_theme = breadth / num_themes."""
        themes = [
            FanoutTheme(theme_label="A", score=0.8),
            FanoutTheme(theme_label="B", score=0.7),
            FanoutTheme(theme_label="C", score=0.6),
        ]
        structure = StructureResult(
            query="test",
            posture=ReformatPosture.FAN_OUT,
            fanout_themes=themes,
            resource_posture=ResourcePosture(
                breadth=15,
                confidence_bar=0.5,
                max_attempts=2,
                speed_budget="interactive",
            ),
        )

        result = run_slots(structure)

        expected_capacity_per_theme = 15 // 3  # 5
        for slot in result.slots:
            assert slot.capacity == expected_capacity_per_theme


class TestCharacterizationTest:
    """Ensure same input → byte-identical output (SAFE changes)."""

    def test_deterministic_output(self):
        """Same StructureResult produces identical AnswerShapeResult."""
        structure = StructureResult(
            query="What are FL Medicaid eligibility requirements?",
            rewritten_queries=["FL", "Medicaid", "eligibility"],
            posture=ReformatPosture.PRECISE,
            resource_posture=ResourcePosture(
                breadth=5, confidence_bar=0.85, max_attempts=1, speed_budget="real_time"
            ),
            reason="EXACT contour",
        )

        # Run twice, should be identical.
        result1 = run_slots(structure)
        result2 = run_slots(structure)

        assert result1.query == result2.query
        assert result1.posture == result2.posture
        assert len(result1.slots) == len(result2.slots)
        for s1, s2 in zip(result1.slots, result2.slots):
            assert s1.slot_id == s2.slot_id
            assert s1.slot_semantics == s2.slot_semantics
            assert s1.capacity == s2.capacity
            assert s1.required == s2.required
            assert s1.priority == s2.priority


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
