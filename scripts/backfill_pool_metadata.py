"""Backfill pool_metadata (top_score_percentile/pool_size/distinct_content_topk)
into the existing forced_filler_bank_run.json artifact, by re-running ONLY
Gate+Pool (fast, no external calls) for the same 22 queries -- NOT re-running
the 5 forced filler calls, which already exist in the artifact and are the
expensive part (c/d's real LLM/web calls). Cheap merge, not a re-run.

Usage (from mobius-rag/):
    .venv/bin/python scripts/backfill_pool_metadata.py
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
from app.services.retriever.orchestrator import _build_pool_metadata  # noqa: E402

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "eval" / "artifacts" / "forced_filler_bank_run.json"


async def main():
    artifact = json.loads(ARTIFACT_PATH.read_text())

    async with AsyncSessionLocal() as db:
        for i, r in enumerate(artifact["results"], 1):
            query_text = r["query"]
            print(f"[{i}/{len(artifact['results'])}] {r['id']}: {query_text!r}")

            gate_result = await run_gate(db, query_text)
            reformat_result = await run_reformat(db, gate_result)
            structure_result = run_structure(reformat_result, caller_mode=None)
            slots_result = run_slots(structure_result)
            adapter = PublicSourceAdapter(db)
            rq = structure_result.rewritten_queries[0] if structure_result.rewritten_queries else query_text
            pool_result = await run_pool_for_query(
                db, rq, gate_result, structure_result.resource_posture, adapter,
            )
            pool_metadata = _build_pool_metadata(slots_result.slots, [pool_result])

            # Single "direct_answer"-style slot dominates this bank -- take
            # the first slot's metadata as the query-level depth signal
            # (matches how compute_depth_bucket is actually invoked per slot
            # today; this bank's queries are all PRECISE/single-slot).
            slot_meta = next(iter(pool_metadata.values()), {})
            r["pool_metadata"] = slot_meta
            print(f"    pool_size={slot_meta.get('pool_size')} "
                  f"top_score_percentile={slot_meta.get('top_score_percentile')} "
                  f"distinct_content_topk={slot_meta.get('distinct_content_topk')}")

            ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2))

    print(f"\nDone. Backfilled pool_metadata into {ARTIFACT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
