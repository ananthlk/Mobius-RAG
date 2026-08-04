"""One-off diagnostic: trace a single query through the real pipeline to see
exactly where a/b/c/d fillers go empty (Eval's finding, 2026-07-24: only
Filler s returns anything in the integrated orchestrator loop for 19/22
calibration queries). Not a permanent script -- ad hoc trace only.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import AsyncSessionLocal
from app.services.retriever.shape.gate import run_gate
from app.services.retriever.shape.reformat import run_reformat
from app.services.retriever.shape.structure import run_structure
from app.services.retriever.shape.slots import run_slots
from app.services.retriever.pool.pool import run_pool_for_query
from app.services.retriever.pool.public_adapter import PublicSourceAdapter

QUERY = "Does Sunshine Health require prior authorization for residential substance use treatment under code H0019?"


async def main():
    async with AsyncSessionLocal() as db:
        gate = await run_gate(db, QUERY)
        print("GATE contour:", gate.contour, "d_codes:", gate.d_codes, "j_codes:", gate.j_codes)

        reformat = await run_reformat(db, gate)
        print("REFORMAT posture:", reformat.posture, "rewritten_queries:", reformat.rewritten_queries)

        structure = run_structure(reformat, caller_mode=None)
        print("STRUCTURE posture:", structure.resource_posture)

        slots = run_slots(structure)
        print("SLOTS:", [(s.slot_id, s.slot_semantics, s.capacity, s.rewritten_query) for s in slots.slots])

        adapter = PublicSourceAdapter(db)
        rq = structure.rewritten_queries[0] if structure.rewritten_queries else QUERY
        pool = await run_pool_for_query(db, rq, gate, structure.resource_posture, adapter)
        print("POOL: n_candidates=", len(pool.candidates), "pool_ms=", pool.pool_ms)
        for c in pool.candidates[:10]:
            print("  cand:", c.chunk_id, c.source_arm, round(c.score or 0, 3) if c.score else c.score,
                  round(c.bm25_score or 0, 3) if c.bm25_score else c.bm25_score, (c.text or "")[:60])


asyncio.run(main())
