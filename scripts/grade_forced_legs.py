"""Grade each FORCED leg's already-retrieved chunks with the SAME judge
(eval/judge.py adjudicate, chunk-only: llm_answer="") used by the observer
calibration run. Purpose: establish per-leg ground truth on the 22-query
bank so any integrated-arm score below the best single leg is provably an
integration loss, not a corpus/recall problem.

Reuses the chunks already stored in eval/artifacts/forced_filler_bank_run.json
(occupancy up to 10 per leg) -- no re-retrieval. Deterministic legs (a/b/s)
are stable; c/d chunks are a valid prior sample.

Usage: .venv/bin/python scripts/grade_forced_legs.py --legs a,d
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import yaml  # noqa: E402
from judge import adjudicate  # noqa: E402

FORCED = ROOT / "eval" / "artifacts" / "forced_filler_bank_run.json"
BANK = ROOT / "eval" / "queries_cmhc.yaml"
OUT = ROOT / "eval" / "artifacts" / "forced_leg_grades.json"


async def grade_one(query_text, q, chunks):
    for attempt in range(4):
        try:
            response = {"chunks": chunks, "llm_answer": "", "strategy_used": None, "confidence": None}
            verdict, score, reasoning, model, ms = await adjudicate(query_text, q, response)
            return {"verdict": verdict, "score": score, "reasoning": reasoning, "model": model}
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                return {"verdict": "ERROR", "score": None, "reasoning": repr(exc), "model": None}
            await asyncio.sleep(4 * (attempt + 1))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="a,d")
    args = ap.parse_args()
    legs = args.legs.split(",")

    forced = json.loads(FORCED.read_text())
    bank = yaml.safe_load(BANK.read_text())
    bank_by_id = {q["id"]: q for q in bank["queries"]}

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for r in forced["results"]:
        qid = r["id"]
        q = bank_by_id.get(qid)
        if q is None:
            continue
        query_text = r["query"]
        out.setdefault(qid, {})
        for leg in legs:
            ps = r["per_strategy"].get(leg, {})
            chunks = ps.get("chunks") or []
            if ps.get("error") is not None or not chunks:
                out[qid][leg] = {"verdict": "NO_CHUNKS", "score": 0.0, "n": len(chunks)}
                print(f"{qid} {leg}: NO_CHUNKS (n={len(chunks)})")
                continue
            g = await grade_one(query_text, q, chunks)
            g["n"] = len(chunks)
            out[qid][leg] = g
            print(f"{qid} {leg}: {g['verdict']:>16} {g['score']} n={len(chunks)}")
        OUT.write_text(json.dumps(out, indent=2))
    # aggregate
    print("\n=== per-leg mean score ===")
    for leg in legs:
        scores = [v[leg]["score"] for v in out.values() if v.get(leg) and v[leg].get("score") is not None]
        m = sum(scores) / len(scores) if scores else 0
        print(f"  leg {leg}: mean={m:.3f}  (n={len(scores)})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
