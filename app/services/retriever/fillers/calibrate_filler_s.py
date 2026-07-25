"""Real calibration run for Filler s against the LIVE staging/dev Payor Fact Store.

Not a mock -- this hits https://mobius-payor-ortabkknqa-uc.a.run.app for real,
using the actual fill_shape_fact_store() function, no fakes. Per the fleet's
standing artifact-validation requirement (a filler's calibration was
fabricated once before and caught -- never repeat that), this script and its
JSON output are the real artifact backing the sign-off claim. The checked-in
copy of a real run's output lives in the parent monorepo at
docs/rag-agents/filler-s-payor-calibration-results.json (mobius-rag is a
separate git submodule, so this script writes locally and the docs/ copy is
updated by hand after each re-run).

CAVEAT (honest, not hidden): tag_matches below are hand-picked stand-ins for
what Gate would actually produce for these queries -- there's no live
Gate/Shape run wired up yet to generate real tag_matches. The gate-condition
LOGIC (go/no-go decisions) is validated for real against this filler's own
code; the exact hit/miss outcome for a given production query also depends on
Gate's real tag extraction, which is outside this filler's control and not
exercised here. Re-run against real Gate output once Gate/Fillers integration
exists, don't treat these hit-rate numbers as final.

Run: python -m app.services.retriever.fillers.calibrate_filler_s
"""
import asyncio
import json

import httpx

from app.services.retriever.fillers.filler_s import fill_shape_fact_store
from app.services.retriever.pool.contracts import PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult, AnswerSlot


def shape_with_direct_answer():
    return AnswerShapeResult(
        slots=[
            AnswerSlot(
                slot_id="direct_answer",
                slot_semantics="direct_answer",
                capacity=1,
                required=True,
                priority=0,
            ),
        ]
    )


EMPTY_POOL = PoolResult(query="", candidates=[], pool_ms=0)

# (case_id, raw_query, tag_matches, expected_gate_pass, note)
CASES = [
    (
        "tp_sunshine_phone",
        "What is the phone number for Sunshine Health?",
        ["j:payor.sunshine_health", "d:contact"],
        True,
        "Known-payer, non-conceptual, factual predicate -- gate should pass, real hit expected.",
    ),
    (
        "tp_aetna_priorauth_url",
        "What is the prior authorization URL for Aetna?",
        ["j:payor.aetna", "d:prior_auth"],
        True,
        "Known-payer, non-conceptual -- gate should pass, real hit expected.",
    ),
    (
        "tp_ahca_portal",
        "What portal do I use for AHCA Florida Medicaid?",
        ["j:payor.ahca", "d:portal"],
        True,
        "Known-payer, non-conceptual -- gate should pass, real hit expected.",
    ),
    (
        "tn_conceptual_with_payer",
        "Explain the philosophy behind Sunshine Health's prior authorization process",
        ["j:payor.sunshine_health", "d:prior_auth"],
        False,
        "Payer tag present but conceptual marker ('explain', 'philosophy') -- client gate must reject, ZERO HTTP calls.",
    ),
    (
        "tn_no_payer_tag",
        "What is the credentialing process for behavioral health providers?",
        ["d:credentialing", "p:process"],
        False,
        "No j:payor.* tag at all -- client gate must reject, ZERO HTTP calls.",
    ),
    (
        "known_bug_repro_unstored_payer",
        "What is the phone number for Humana?",
        ["j:payor.humana", "d:contact"],
        True,
        "Payer tag present, non-conceptual -- client gate PASSES (it only checks tag presence, not whether the payer is actually stored). This is the documented, PARKED over-fire bug ([[project-payor-fact-store]]) -- expected to reproduce live, not a new bug introduced by Filler s.",
    ),
]


async def run():
    results = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for case_id, query, tags, expected_gate_pass, note in CASES:
            filled = await fill_shape_fact_store(
                EMPTY_POOL,
                shape_with_direct_answer(),
                query,
                tag_matches=tags,
                http_client=client,
            )
            slot = filled.slots[0]
            per_slot = filled.emit["per_slot_details"][0]
            gate_passed = per_slot["gate_passed"]
            hit = per_slot["hit"]
            served_text = slot.chunks[0].text if slot.chunks else None
            served_score = slot.chunks[0].original_score if slot.chunks else None

            results.append({
                "case_id": case_id,
                "query": query,
                "tag_matches": tags,
                "expected_gate_pass": expected_gate_pass,
                "actual_gate_pass": gate_passed,
                "gate_match_expected": gate_passed == expected_gate_pass,
                "hit": hit,
                "served_text": served_text,
                "served_score": served_score,
                "fact_store_ms": per_slot["fact_store_ms"],
                "note": note,
            })

    return results


if __name__ == "__main__":
    results = asyncio.run(run())
    print(json.dumps(results, indent=2))
    all_gate_match = all(r["gate_match_expected"] for r in results)
    print(f"\n{'='*60}")
    print(f"ALL GATE PREDICTIONS MATCHED: {all_gate_match}")
    with open("filler_s_calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
