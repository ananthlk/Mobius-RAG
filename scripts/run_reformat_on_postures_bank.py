"""Run Gate then Reformat against eval/queries_reformat_postures.yaml, using
REAL Gate output (not the bank's hand-authored gate_contour) — the bank's
gate_contour/gate_codes/fanout_codes fields document EXPECTED Gate behavior
for readability, but this runner classifies live and reports where real
Gate output diverges from the bank's assumption, same discipline Gate's own
contour-bank runner uses.

Usage (from mobius-rag/):
    .venv/bin/python scripts/run_reformat_on_postures_bank.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import AsyncSessionLocal  # noqa: E402
from app.services.retriever.shape.gate import run_gate  # noqa: E402
from app.services.retriever.shape.reformat import run_reformat  # noqa: E402

BANK_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries_reformat_postures.yaml"


async def main() -> None:
    # The bank file has a trailing markdown section (Eval's §6 scoring
    # criteria, with **bold** etc.) after a "---" line that isn't valid
    # YAML at all — only the text before the first "---" is machine-readable.
    raw = BANK_PATH.read_text()
    yaml_part = raw.split("\n---\n", 1)[0]
    bank = yaml.safe_load(yaml_part)
    queries = bank["queries"]
    print(f"bank: {bank.get('bank_version')} — {len(queries)} queries\n")

    passed = 0
    gate_mismatches = []
    posture_mismatches = []
    results = []

    async with AsyncSessionLocal() as db:
        for q in queries:
            qid = q["id"]
            text = q["query"]
            expected_gate_contour = q.get("gate_contour")
            expected_posture = q["expected_posture"]

            t0 = time.monotonic()
            gr = await run_gate(db, text)
            rr = await run_reformat(db, gr)
            total_ms = int((time.monotonic() - t0) * 1000)

            gate_ok = gr.contour.value == expected_gate_contour
            posture_ok = rr.posture.value == expected_posture

            status = "PASS" if (gate_ok and posture_ok) else "FAIL"
            if status == "PASS":
                passed += 1
            if not gate_ok:
                gate_mismatches.append((qid, text, expected_gate_contour, gr.contour.value))
            if not posture_ok:
                posture_mismatches.append((qid, text, expected_posture, rr.posture.value))

            n_rewritten = len(rr.rewritten_queries)
            print(
                f"[{status}] {qid}: gate={gr.contour.value}"
                f"{'(' + gr.underspecified_kind + ')' if gr.underspecified_kind else ''}"
                f" (expected {expected_gate_contour}) -> posture={rr.posture.value}"
                f" (expected {expected_posture}) | n_rewritten={n_rewritten}"
                f" | total_ms={total_ms}"
            )
            if rr.posture.value == "fan_out":
                for t in rr.fanout_themes:
                    tag = "CATCHALL" if t.is_catchall else f"n={len(t.member_codes)} prev={t.prevalence_docs}"
                    print(f"         theme: {t.theme_label!r} [{tag}]")
            results.append((qid, gr, rr, total_ms))

    print(f"\n{passed}/{len(queries)} fully matched (gate contour AND posture)")
    if gate_mismatches:
        print(f"\n{len(gate_mismatches)} GATE CONTOUR mismatches (bank's assumption vs live):")
        for qid, text, exp, act in gate_mismatches:
            print(f"  {qid}: {text!r} — bank expected gate={exp}, live gate={act}")
    if posture_mismatches:
        print(f"\n{len(posture_mismatches)} POSTURE mismatches:")
        for qid, text, exp, act in posture_mismatches:
            print(f"  {qid}: {text!r} — expected posture={exp}, got posture={act}")


if __name__ == "__main__":
    asyncio.run(main())
