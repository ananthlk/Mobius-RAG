"""One-off debug: reproduce the STAGE 7 chunk-count reconciliation mismatch
(expected_chunks_out=21 actual_chunks_out=20) with per-chunk visibility into
_complete_neighbors' internals -- to find exactly which chunk vanishes and
why, per Ananth's hypothesis that a neighbor-expanded chunk collides with
(duplicates) another already-selected chunk and isn't being counted as such.

Throwaway diagnostic script -- imports private synthesis.py helpers directly.
"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUERY = "What is the timely filing deadline for Sunshine Health FL Medicaid claims?"


async def main():
    from app.database import AsyncSessionLocal
    from app.services.retriever.shape.gate import run_gate
    from app.services.retriever.shape.reformat import run_reformat
    from app.services.retriever.shape.structure import run_structure
    from app.services.retriever.shape.slots import run_slots
    from app.services.retriever.pool.public_adapter import PublicSourceAdapter
    from app.services.retriever.pool.pool import run_pool_for_query
    from app.services.retriever.fillers.filler_a import fill_shape_bm25
    from app.services.retriever.synthesis import _complete_neighbors, _dedup_cross_slot

    async with AsyncSessionLocal() as db:
        gate = await run_gate(db, QUERY)
        reformat = await run_reformat(db, gate)
        structure = run_structure(reformat, caller_mode="chat.default")
        rp = structure.resource_posture
        slots_result = run_slots(structure)
        adapter = PublicSourceAdapter(db)
        slot0 = slots_result.slots[0]
        pool_result = await run_pool_for_query(db, slot0.rewritten_query or QUERY, gate, rp, adapter)
        filled = fill_shape_bm25(pool_result, slots_result)

        slot_items = [(slot, chunk) for slot in filled.slots for chunk in slot.chunks]
        print(f"INPUT slot_items: {len(slot_items)}")
        for slot, c in slot_items:
            print(f"  seed chunk_id={c.chunk_id} doc={c.document_id} is_neighbor={c.is_neighbor} "
                  f"content_sha={c.content_sha!r} score={c.original_score} text={(c.text or '')[:60]!r}")

        out, added, skipped = await _complete_neighbors(db, slot_items)
        print(f"\n_complete_neighbors: added={added} skipped={skipped} out_len={len(out)} "
              f"(expected {len(slot_items)}+{added}={len(slot_items)+added})")

        # Check for duplicate chunk_ids WITHIN `out` (pre-dedup) directly.
        from collections import Counter
        id_counts = Counter(c.chunk_id for _, c in out)
        dupes = {cid: n for cid, n in id_counts.items() if n > 1}
        print(f"\nduplicate chunk_ids in `out` pre-dedup: {dupes}")

        sha_counts = Counter((c.content_sha or "").strip() for _, c in out if (c.content_sha or "").strip())
        sha_dupes = {sha: n for sha, n in sha_counts.items() if n > 1}
        print(f"duplicate content_sha in `out` pre-dedup: {sha_dupes}")

        # Print every neighbor chunk added, flagging any whose chunk_id or
        # content_sha matches something already in slot_items (original 10).
        original_ids = {c.chunk_id for _, c in slot_items}
        original_shas = {(c.content_sha or "").strip() for _, c in slot_items if (c.content_sha or "").strip()}
        print(f"\nNEIGHBOR chunks added (all {added}):")
        for slot, c in out:
            if (slot, c) in slot_items or c.chunk_id in original_ids and not c.is_neighbor:
                continue
            if not c.is_neighbor:
                continue
            flag = ""
            if c.chunk_id in original_ids:
                flag += " <<< CHUNK_ID MATCHES AN ORIGINAL SEED"
            sha = (c.content_sha or "").strip()
            if sha and sha in original_shas:
                flag += " <<< CONTENT_SHA MATCHES AN ORIGINAL SEED"
            print(f"  chunk_id={c.chunk_id} doc={c.document_id} content_sha={sha!r} "
                  f"score={c.original_score} text={(c.text or '')[:60]!r}{flag}")

        deduped, removed = _dedup_cross_slot(out)
        print(f"\n_dedup_cross_slot: removed={removed} final_len={len(deduped)}")
        print(f"\nRECONCILIATION: chunks_in={len(slot_items)} + neighbors_added={added} "
              f"- duplicates_removed={removed} = expected {len(slot_items)+added-removed}, "
              f"actual deduped len={len(deduped)}")


asyncio.run(main())
