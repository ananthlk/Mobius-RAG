"""Run the shape gate (Step 1a) against the cmhc 22-query eval bank.

Ad-hoc verification script — not part of the pytest suite. Prints one row
per query: contour, matched J/P/D codes, corpus probe counts, and timing.
This is the first checkpoint before building reformat/structure: every
query in the bank is in-corpus, so we expect EXACT or VICINITY on all of
them, never UNCLEAR/CORPUS_GAP, and gate_ms comfortably under 500ms.

Usage (from mobius-rag/):
    .venv/bin/python scripts/run_gate_on_cmhc.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import AsyncSessionLocal  # noqa: E402
from app.services.retriever.shape.gate import run_gate  # noqa: E402

BANK_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries_cmhc.yaml"


async def main() -> None:
    bank = yaml.safe_load(BANK_PATH.read_text())
    queries = bank["queries"]

    print(f"bank: {bank.get('bank_version')} — {len(queries)} queries\n")

    contour_counts: dict[str, int] = {}
    gate_times: list[int] = []

    async with AsyncSessionLocal() as db:
        for q in queries:
            qid = q["id"]
            text = q["query"]
            expected_strategy = q.get("expected", {}).get("strategy", "?")

            result = await run_gate(db, text)
            contour_counts[result.contour.value] = contour_counts.get(result.contour.value, 0) + 1
            gate_times.append(result.gate_ms)

            print(f"[{qid}] expected_strategy={expected_strategy}  contour={result.contour.value}")
            print(f"  query: {text}")
            print(
                f"  d={result.d_codes} j={result.j_codes} p={result.p_codes} "
                f"missing={result.missing_kinds}"
            )
            print(
                f"  probe: union={result.probe.union_docs} intersection={result.probe.intersection_docs} "
                f"(d={result.probe.d_docs} j={result.probe.j_docs} p={result.probe.p_docs}) "
                f"probe_ms={result.probe.probe_ms}"
            )
            print(f"  reason: {result.reason}")
            print(f"  gate_ms={result.gate_ms}\n")

    print("--- summary ---")
    for contour, count in sorted(contour_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {contour}: {count}")
    p50 = sorted(gate_times)[len(gate_times) // 2]
    p95 = sorted(gate_times)[int(len(gate_times) * 0.95)]
    print(f"gate_ms p50={p50} p95={p95} max={max(gate_times)}")


if __name__ == "__main__":
    asyncio.run(main())
