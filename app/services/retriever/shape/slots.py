"""Slots — Step 1d of the shape module (Gate → Reformat → Structure → Slots).

Emits the answer structure that Fillers (Step 3) will fill with Pool chunks.

Input: StructureResult (now carries fanout_themes for FAN_OUT postures).
Output: AnswerShapeResult (slots[], each with semantics/capacity/priority).

Pure compute, zero DB calls. See docs/rag-agents/shape-slots-module-spec-v1.md
for the full design record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.services.retriever.shape.contracts import (
    ReformatPosture,
    StructureResult,
)


@dataclass
class AnswerSlot:
    """One slot in the answer structure."""

    slot_id: str  # "direct_answer" | "fanout_0" | "external_context" | etc.
    slot_semantics: str  # "direct_answer" | "thematic_exploration" | "external_context"
    capacity: int  # target chunk count for this slot
    rewritten_query: str = ""  # FAN_OUT only: the query this slot corresponds to
    required: bool = True  # must this slot be filled?
    priority: int = 0  # ranking hint for Fillers (0=highest)


@dataclass
class AnswerShapeResult:
    """Everything Slots decided about one StructureResult (Step 1d output).

    The real slot model that Fillers consumes.
    """

    query: str = ""
    posture: ReformatPosture = ReformatPosture.CLARIFY_REPHRASE
    slots: list[AnswerSlot] = field(default_factory=list)
    reason: str = ""
    slots_ms: int = 0


def run_slots(structure_result: StructureResult) -> AnswerShapeResult:
    """Derive answer slots from Structure's output.

    Args:
        structure_result: Everything Structure decided (posture, breadth, etc.)

    Returns:
        AnswerShapeResult with the real slot model for Fillers.
    """
    t0 = time.time()

    posture = structure_result.posture
    breadth = structure_result.resource_posture.breadth
    reason = ""

    slots: list[AnswerSlot] = []

    if posture == ReformatPosture.PRECISE:
        # One clear query with complete coverage.
        slots = [
            AnswerSlot(
                slot_id="direct_answer",
                slot_semantics="direct_answer",
                capacity=breadth,
                required=True,
                priority=0,
            )
        ]
        reason = "PRECISE: one direct answer slot"

    elif posture == ReformatPosture.FAN_OUT:
        # Multiple thematic angles. One slot per theme, capped at MAX_FANOUT_THEMES.
        fanout_themes = structure_result.fanout_themes
        rewritten_queries = structure_result.rewritten_queries
        if not fanout_themes:
            # Fallback: empty theme list shouldn't happen, but handle gracefully.
            reason = "FAN_OUT: no fanout_themes provided (unexpected)"
        else:
            # Build index mapping: fanout_themes[i] ↔ rewritten_queries[i] (from Reformat, same order)
            # This assumes Reformat produces themes and queries in 1:1 correspondence.
            theme_to_query = {id(theme): query for theme, query in zip(fanout_themes, rewritten_queries)}

            # Sort by score (descending) to set priorities.
            sorted_themes = sorted(fanout_themes, key=lambda t: t.score, reverse=True)
            capacity_per_theme = max(1, breadth // len(sorted_themes))

            for i, theme in enumerate(sorted_themes):
                # Look up the rewritten_query for this theme.
                rewritten_query = theme_to_query.get(id(theme), "")

                slots.append(
                    AnswerSlot(
                        slot_id=f"fanout_{i}",
                        slot_semantics="thematic_exploration",
                        capacity=capacity_per_theme,
                        rewritten_query=rewritten_query,
                        required=True,
                        priority=i,  # Higher theme score = lower priority number
                    )
                )
            reason = f"FAN_OUT: {len(slots)} thematic slots (scored, sorted by theme.score)"

    elif posture == ReformatPosture.CLARIFY:
        # Query needs clarification. No answer slots — Chat asks questions instead.
        slots = []
        reason = "CLARIFY: no answer slots; Chat will ask clarifying questions"

    elif posture == ReformatPosture.RELY_ON_EXTERNAL:
        # Corpus incomplete. One fallback slot for Router strategies c/d.
        slots = [
            AnswerSlot(
                slot_id="external_context",
                slot_semantics="external_context",
                capacity=breadth,
                required=False,  # Optional — fallback only
                priority=1,  # Lower than default answers
            )
        ]
        reason = "RELY_ON_EXTERNAL: one fallback slot for Router c/d"

    elif posture == ReformatPosture.DECLINE:
        # Out of scope. No answer slots.
        slots = []
        reason = "DECLINE: out of scope; Chat declines gracefully"

    elif posture == ReformatPosture.CLARIFY_REPHRASE:
        # Make best guess: emit 1 optimistic slot. Synthesis will handle
        # uncertainty gracefully: "we think you meant X, but if not please
        # clarify" + confidence signal to Chat.
        slots = [
            AnswerSlot(
                slot_id="best_guess",
                slot_semantics="direct_answer",
                capacity=breadth,
                required=False,  # Optional — may fail if guess is wrong
                priority=0,
            )
        ]
        reason = "CLARIFY_REPHRASE: best guess (1 optimistic slot, Synthesis handles uncertainty)"

    else:
        # Unknown posture — shouldn't happen.
        reason = f"UNKNOWN posture {posture}"

    slots_ms = int((time.time() - t0) * 1000)

    return AnswerShapeResult(
        query=structure_result.query,
        posture=posture,
        slots=slots,
        reason=reason,
        slots_ms=slots_ms,
    )
