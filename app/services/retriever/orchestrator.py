"""Retriever's top-level entry point — sequences Shape (Gate → Reformat →
Structure) and, once built, Pool → Router → Fillers → Synthesis → Contract →
Timing. This is the conductor role TECH asked about 2026-07-23: no single
sub-module naturally owns cross-module sequencing (Shape doesn't call Pool;
Pool doesn't call Router), so it belongs to Retriever directly, as the "one
clean answering contract" chat calls. Mirrors the legacy single entry point
(`corpus_search_agent() :3066 → _impl :3766`) — thin glue, no business logic
of its own, calling each module's public interface in order.

STATUS 2026-07-23: PARTIAL PIPELINE. Wires Shape:Gate (signed off, closed) +
Shape:Reformat (built, DB/TECH sign-off pending — used as-is here, including
its known FAN_OUT latency limitation, not worked around). Structure (Step 1c)
and everything from Pool onward are NOT YET BUILT — this stops after Reformat
and returns what exists, clearly marked as partial, so Gate+Reformat can be
exercised end-to-end right now rather than waiting for the full chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retriever.shape.contracts import GateResult, ReformatPosture, ReformatResult
from app.services.retriever.shape.gate import run_gate
from app.services.retriever.shape.narrate import narrate as narrate_gate
from app.services.retriever.shape.narrate import narrate_full as narrate_gate_full
from app.services.retriever.shape.reformat import run_reformat
from app.services.retriever.shape.reformat_narrate import narrate as narrate_reformat
from app.services.retriever.shape.reformat_narrate import narrate_full as narrate_reformat_full


@dataclass
class RetrieverPartialResult:
    """What the chain has produced so far — PROVISIONAL, not a locked
    contract like GateResult/ReformatResult. Will be superseded once
    Structure (Step 1c) exists and defines the real Shape-output contract;
    this is a build-time stitch, not something downstream code should
    depend on long-term.
    """

    query: str = ""
    gate: GateResult | None = None
    reformat: ReformatResult | None = None

    gate_ms: int = 0
    reformat_ms: int = 0
    total_ms: int = 0

    # RESOLVED by UX 2026-07-23 (see docs/rag-agents/retriever-emit-telemetry-registry.md):
    # both narrate() outputs feed thinking_trace. shape_reformat (the structural
    # telemetry key: posture/fanout_themes/latency_ms) is backend/Diagnostics
    # only -- a DIFFERENT layer from the narrate() function, which is
    # user-facing by design, same as Gate's. Composition: Gate's narrate()
    # first (explains the contour), Reformat's narrate() second (explains the
    # resulting action) -- but ONLY for postures where Reformat adds real
    # information beyond Gate's own narration; see _include_reformat_narration().
    narrative: str = ""       # thinking_trace value -- Gate always, Reformat conditionally (see composition rule)
    narrative_full: str = ""  # combined, Diagnostics-only, NEVER persist (see narrate.py PHI note)

    pipeline_complete: bool = False  # False until Structure/Pool/.../Timing exist
    next_step: str = "Structure (Step 1c) — not yet built"


# Postures where Reformat's narrate() is included in thinking_trace alongside
# Gate's -- per UX's ruling 2026-07-23. PRECISE/DECLINE/CLARIFY_REPHRASE are
# excluded: Gate's own narration already fully explains those outcomes,
# Reformat's narrate() for them would be redundant, not additive.
_INCLUDE_REFORMAT_NARRATION = frozenset({
    ReformatPosture.FAN_OUT,
    ReformatPosture.RELY_ON_EXTERNAL,
    ReformatPosture.CLARIFY,
})


def _include_reformat_narration(posture: ReformatPosture) -> bool:
    return posture in _INCLUDE_REFORMAT_NARRATION


async def run_retriever_partial(db: AsyncSession, query: str) -> RetrieverPartialResult:
    """Sequence Gate → Reformat. Stops there — Structure onward doesn't
    exist yet. This function's own scope will shrink over time as real
    modules absorb what it currently does (today: nothing but sequencing
    and narrative-stitching; no business logic lives here or ever should).
    """
    t0 = time.monotonic()

    gate_result = await run_gate(db, query)
    reformat_result = await run_reformat(db, gate_result)

    total_ms = int((time.monotonic() - t0) * 1000)

    if _include_reformat_narration(reformat_result.posture):
        narrative = f"{narrate_gate(gate_result)}\n\n{narrate_reformat(gate_result, reformat_result)}"
    else:
        narrative = narrate_gate(gate_result)
    narrative_full = (
        f"--- Shape: Gate ---\n{narrate_gate_full(gate_result)}\n\n"
        f"--- Shape: Reformat ---\n{narrate_reformat_full(gate_result, reformat_result)}"
    )

    return RetrieverPartialResult(
        query=query,
        gate=gate_result,
        reformat=reformat_result,
        gate_ms=gate_result.gate_ms,
        reformat_ms=reformat_result.reformat_ms,
        total_ms=total_ms,
        narrative=narrative,
        narrative_full=narrative_full,
        pipeline_complete=False,
        next_step="Structure (Step 1c) — not yet built",
    )
