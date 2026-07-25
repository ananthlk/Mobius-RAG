"""Independent verification that Filler c makes a REAL LLM call, not a
mock -- prints the actual chunk content/citation output for inspection,
not just latency/occupancy numbers.

Usage (from mobius-rag/):
    .venv/bin/python scripts/verify_filler_c_real_output.py
"""

from __future__ import annotations

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import AsyncSessionLocal  # noqa: E402
from app.services.retriever.shape.gate import run_gate  # noqa: E402
from app.services.retriever.shape.reformat import run_reformat  # noqa: E402
from app.services.retriever.shape.structure import run_structure  # noqa: E402
from app.services.retriever.shape.slots import run_slots  # noqa: E402
from app.services.retriever.pool.pool import run_pool_for_query  # noqa: E402
from app.services.retriever.pool.public_adapter import PublicSourceAdapter  # noqa: E402
from app.services.retriever.fillers.filler_c import fill_shape_llm_retrieval  # noqa: E402
from app.services.retriever.shape.slots import AnswerShapeResult  # noqa: E402

QUERY = "What is the timely filing deadline for Sunshine Health FL Medicaid claims?"


async def main():
    async with AsyncSessionLocal() as db:
        gate_result = await run_gate(db, QUERY)
        reformat_result = await run_reformat(db, gate_result)
        structure_result = run_structure(reformat_result, caller_mode=None)
        slots_result = run_slots(structure_result)

        adapter = PublicSourceAdapter(db)
        rq = structure_result.rewritten_queries[0] if structure_result.rewritten_queries else QUERY
        pool_result = await run_pool_for_query(
            db, rq, gate_result, structure_result.resource_posture, adapter,
        )

        shape = AnswerShapeResult(
            query=QUERY, posture=None, slots=slots_result.slots, reason="", slots_ms=0,
        )
        tag_matches = [*gate_result.d_codes, *gate_result.j_codes, *gate_result.p_codes]

        print(f"Query: {QUERY!r}")
        print(f"tag_matches passed to filler_c: {tag_matches}")
        print("Calling fill_shape_llm_retrieval() -- real DB, real LLM path, no mocking...\n")

        result = await fill_shape_llm_retrieval(
            pool_result, shape, QUERY,
            db=db, agent_id="verify-real-output",
            tag_matches=tag_matches,
        )

        print("=== RAW RESULT ===")
        for slot in result.slots:
            print(f"\nslot_id={slot.slot_id} occupancy={slot.occupancy} under_filled={slot.under_filled}")
            for chunk in slot.chunks:
                print(f"  chunk_id={chunk.chunk_id}")
                print(f"  document_id={chunk.document_id!r} url={chunk.url!r}")
                print(f"  source_type={chunk.source_type} assignment_reason={chunk.assignment_reason}")
                print(f"  original_score={chunk.original_score}")
                print(f"  text={chunk.text!r}")
                print(f"  tags={chunk.tags}")

        print("\n=== EMIT (should show real model call metadata if not mocked) ===")
        print(json.dumps(result.emit, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
