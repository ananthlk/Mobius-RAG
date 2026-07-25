"""Backfill search_result_count (Filler d's raw n_hits, before fetch/
synthesize) for the same 22 queries -- NOT a/b/c/s, which don't need this
and are already correct in the main artifact. This is the one piece of
Eval's ask that genuinely requires a re-run (the original run didn't
capture Filler d's emit dict), scoped to the minimum re-run needed.

Writes to a SEPARATE file (not forced_filler_bank_run.json) because the
pool_metadata backfill (backfill_pool_metadata.py) may be running
concurrently against the main artifact -- both scripts read-modify-write
the whole JSON, so running them against the same file at the same time
would race and one script's writes would clobber the other's. Merge both
into the main artifact once both finish.

Usage (from mobius-rag/):
    .venv/bin/python scripts/backfill_d_search_hits.py
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
from app.services.retriever.shape.slots import run_slots, AnswerShapeResult  # noqa: E402
from app.services.retriever.pool.pool import run_pool_for_query  # noqa: E402
from app.services.retriever.pool.public_adapter import PublicSourceAdapter  # noqa: E402
from app.services.retriever.fillers.payer_context import (  # noqa: E402
    extract_payer_slug, resolve_payer_context,
)
from app.services.retriever.fillers.filler_d import fill_shape_external  # noqa: E402

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "eval" / "artifacts" / "forced_filler_bank_run.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "artifacts" / "d_search_hits_backfill.json"


async def main():
    artifact = json.loads(ARTIFACT_PATH.read_text())
    out: dict = {}

    async with AsyncSessionLocal() as db:
        for i, r in enumerate(artifact["results"], 1):
            query_text = r["query"]
            print(f"[{i}/{len(artifact['results'])}] {r['id']}: {query_text!r}")

            gate_result = await run_gate(db, query_text)
            reformat_result = await run_reformat(db, gate_result)
            structure_result = run_structure(reformat_result, caller_mode=None)
            slots_result = run_slots(structure_result)
            payer_slug = extract_payer_slug(gate_result.j_codes)
            payer_context = await resolve_payer_context(db, payer_slug) if payer_slug else None
            adapter = PublicSourceAdapter(db)
            rq = structure_result.rewritten_queries[0] if structure_result.rewritten_queries else query_text
            pool_result = await run_pool_for_query(
                db, rq, gate_result, structure_result.resource_posture, adapter,
            )
            shape = AnswerShapeResult(
                query=query_text, posture=None, slots=slots_result.slots, reason="", slots_ms=0,
            )
            tag_matches = [*gate_result.d_codes, *gate_result.j_codes, *gate_result.p_codes]

            try:
                result = await fill_shape_external(
                    pool_result, shape, query_text, db=db, agent_id="retriever-backfill-d",
                    tag_matches=tag_matches, payer_context=payer_context,
                )
                n_hits = result.emit.get("n_hits")
                search_returned_zero = (n_hits or 0) == 0
                error = None
            except Exception as exc:
                n_hits = None
                search_returned_zero = None
                error = repr(exc)

            out[r["id"]] = {
                "search_result_count": n_hits,
                "search_returned_zero": search_returned_zero,
                "search_hits_backfill_error": error,
            }

            print(f"    n_hits={n_hits} search_returned_zero={search_returned_zero} error={error}")
            OUT_PATH.write_text(json.dumps(out, indent=2))

    print(f"\nDone. Wrote {OUT_PATH} -- merge into {ARTIFACT_PATH} once both backfills finish.")


if __name__ == "__main__":
    asyncio.run(main())
