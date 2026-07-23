"""Run the shape gate against eval/queries_gate_contours.yaml — asserts
expected_contour match (pass/fail), unlike run_gate_on_cmhc.py which just
reports distribution. This bank exists specifically to exercise contour
DIVERSITY (cmhc structurally can't hit 4 of 6 contours).

Usage (from mobius-rag/):
    .venv/bin/python scripts/run_gate_on_contour_bank.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import AsyncSessionLocal  # noqa: E402
from app.services.retriever.shape.gate import run_gate  # noqa: E402
from app.services.retriever.shape.narrate import narrate, narrate_full  # noqa: E402

BANK_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries_gate_contours.yaml"


async def main() -> None:
    bank = yaml.safe_load(BANK_PATH.read_text())
    queries = bank["queries"]

    print(f"bank: {bank.get('bank_version')} — {len(queries)} queries\n")

    passed = 0
    failed = []
    xfailed = []
    gate_times = []
    contour_counts: dict[str, int] = {}
    total_scored = 0

    async with AsyncSessionLocal() as db:
        for q in queries:
            qid = q["id"]
            text = q["query"]
            expected = q["expected_contour"]
            is_xfail = q.get("xfail", False)

            r = await run_gate(db, text)
            actual = r.contour.value
            contour_counts[actual] = contour_counts.get(actual, 0) + 1
            gate_times.append(r.gate_ms)

            if is_xfail:
                # Documented known gap — report separately, don't count as pass/fail.
                still_matches_expected = actual == q.get("actual_contour", expected)
                xfailed.append((qid, text, expected, actual, still_matches_expected))
                status = "XFAIL (as documented)" if still_matches_expected else "XFAIL-DRIFTED (gap changed!)"
            else:
                total_scored += 1
                ok = actual == expected
                passed += ok
                status = "PASS" if ok else "FAIL"
                if not ok:
                    failed.append((qid, text, expected, actual))

            print(f"[{qid}] {status}  expected={expected}  actual={actual}  gate_ms={r.gate_ms}")
            print(f"  query: {text!r}")
            print(f"  short: {narrate(r)}")
            print()

    print("=" * 70)
    print(f"RESULT: {passed}/{total_scored} passed (scored cases only)")
    if failed:
        print("\nFAILURES:")
        for qid, text, expected, actual in failed:
            print(f"  [{qid}] {text!r} — expected={expected}, got={actual}")
    if xfailed:
        print("\nXFAIL entries (documented known gaps, not counted above):")
        for qid, text, expected, actual, matches in xfailed:
            drift = "" if matches else "  ⚠️  GAP BEHAVIOR CHANGED — investigate"
            print(f"  [{qid}] {text!r} — documented gap: expected={expected}, got={actual}{drift}")
    print("\ncontour distribution:", contour_counts)
    if gate_times:
        p50 = sorted(gate_times)[len(gate_times) // 2]
        p95 = sorted(gate_times)[int(len(gate_times) * 0.95)]
        print(f"gate_ms p50={p50} p95={p95} max={max(gate_times)}")


if __name__ == "__main__":
    asyncio.run(main())
