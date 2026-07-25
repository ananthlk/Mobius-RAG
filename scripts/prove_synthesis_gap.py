"""Prove the synthesis-gap thesis empirically. For a few queries with HIGH
chunk-recall but LOW chunk-only rubric score, synthesize a real answer from
the retrieved leg-d chunks (simulating Chat, which the offline loop omits),
then grade the SAME chunks two ways:
  (1) chunk-only  (llm_answer="")   -- what the calibration harness does
  (2) grounding   (llm_answer=<synthesized answer>) -- true end-to-end

If (2) >> (1), the low calibration score is the missing synthesis step,
not a retrieval or integration failure.
"""
from __future__ import annotations
import asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import yaml  # noqa: E402
from judge import adjudicate  # noqa: E402
from app.services import llm_manager_client  # noqa: E402

FORCED = ROOT / "eval" / "artifacts" / "forced_filler_bank_run.json"
BANK = ROOT / "eval" / "queries_cmhc.yaml"
QIDS = ["cmhc008", "cmhc010", "cmhc012", "cmhc005", "cmhc022"]

SYNTH_SYSTEM = (
    "You are a claims/payer support assistant. Answer the user's question "
    "using ONLY the provided source passages. Be direct and specific: state "
    "the codes, day-counts, and yes/no determinations the passages support. "
    "If the passages do not answer it, say so plainly. Do not invent facts."
)


async def synth(query, chunks):
    body = "\n\n".join(f"[{i+1}] {c.get('text','')}" for i, c in enumerate(chunks[:12]))
    user = f"Question: {query}\n\nSource passages:\n{body}\n\nAnswer:"
    raw, _ = await llm_manager_client.generate(
        system=SYNTH_SYSTEM, user=user, stage="rag_eval_adjudicate", max_tokens=2048
    )
    return raw.strip()


async def main():
    forced = {r["id"]: r for r in json.loads(FORCED.read_text())["results"]}
    bank = {q["id"]: q for q in yaml.safe_load(BANK.read_text())["queries"]}
    print(f"{'qid':9} | {'chunk-only':>10} | {'grounding':>10} | delta")
    for qid in QIDS:
        r, q = forced.get(qid), bank.get(qid)
        if not r or not q:
            print(f"{qid}: missing"); continue
        chunks = (r["per_strategy"].get("d") or {}).get("chunks") or []
        if not chunks:
            print(f"{qid}: no d chunks"); continue
        qt = r["query"]
        v0, s0, _, _, _ = await adjudicate(qt, q, {"chunks": chunks, "llm_answer": "", "confidence": None})
        ans = await synth(qt, chunks)
        v1, s1, r1, _, _ = await adjudicate(qt, q, {"chunks": chunks, "llm_answer": ans, "confidence": None})
        print(f"{qid:9} | {v0[:9]:>10} {s0:.2f} | {v1[:9]:>10} {s1:.2f} | +{s1-s0:.2f}")
        print(f"    synth answer: {ans[:180].replace(chr(10),' ')}")


if __name__ == "__main__":
    asyncio.run(main())
