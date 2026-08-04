"""Re-grade a SINGLE query and patch it into an existing kcurve results.json
(for recovering a query that errored on a transient network blip, e.g. cmhc012).
Reuses the same synth + grade logic as kcurve_bank.py. Merges in place.

Usage: KCURVE_JSON=/path/results.json .venv/bin/python scripts/kcurve_patch_one.py <qid>
"""
import asyncio, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.services.fact_checker import check_facts          # noqa: E402
from app.services import llm_manager_client                # noqa: E402

LOCKED_STAGE = "rag_eval_adjudicate"
KS = [1, 3, 5, 10]
QID = sys.argv[1] if len(sys.argv) > 1 else "cmhc012"
OUT_JSON = Path(os.environ["KCURVE_JSON"])
SYNTH_SYSTEM = (
    "You are a claims/payer support assistant. Answer the question using ONLY "
    "the provided source passages — state the codes, day-counts, and yes/no "
    "determinations they support. If the passages don't answer it, say so. "
    "Do not invent facts.")


async def _gen(system, user):
    for attempt in range(4):
        try:
            raw, meta = await llm_manager_client.generate(system=system, user=user, stage=LOCKED_STAGE, max_tokens=2048)
            return raw.strip(), (meta or {}).get("model") or "unknown"
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            await asyncio.sleep(3 * (attempt + 1))


async def synth(query, chunks):
    body = "\n\n".join(f"[{i+1}] {c.get('text','')}" for i, c in enumerate(chunks))
    draft, model = await _gen(SYNTH_SYSTEM, f"Question: {query}\n\nPassages:\n{body}\n\nAnswer:")
    crit = await check_facts(query=query, must_facts=[], chunks=chunks, answer=draft, stage=LOCKED_STAGE)
    halluc = list(crit.hallucinated_claims or [])
    if not halluc:
        return draft, draft, [], model
    halluc_txt = "\n".join(f"- {h}" for h in halluc)
    final, _ = await _gen(SYNTH_SYSTEM,
        f"Question: {query}\n\nPassages:\n{body}\n\nDraft answer:\n{draft}\n\n"
        f"A grounding check verified these specific claims are NOT supported by ANY passage:\n{halluc_txt}\n\n"
        f"Rewrite the answer: DELETE each of those claims exactly. Do NOT add anything. Keep everything else. Output the corrected answer only:")
    return final, draft, halluc, model


async def grade_k(query, must_facts, chunks_k):
    final, draft, pruned, smodel = await synth(query, chunks_k)
    ra, rb_draft, rb, rc = await asyncio.gather(
        check_facts(query=query, must_facts=must_facts, chunks=chunks_k, answer=None, stage=LOCKED_STAGE),
        check_facts(query=query, must_facts=must_facts, chunks=chunks_k, answer=draft, stage=LOCKED_STAGE),
        check_facts(query=query, must_facts=must_facts, chunks=chunks_k, answer=final, stage=LOCKED_STAGE),
        check_facts(query=query, must_facts=[], chunks=chunks_k, answer=final, stage=LOCKED_STAGE))
    facts = []
    for v in (ra.verdicts if ra else []):
        in_answer = bool(rb and any(getattr(bv, 'fact', '') == getattr(v, 'fact', '') and bv.support >= 1.0 for bv in rb.verdicts))
        facts.append({"fact": getattr(v, 'fact', '?'), "in_chunk": v.support, "in_answer": in_answer})
    return {"k": len(chunks_k), "draft": draft, "pruned": pruned, "final": final, "synth_model": smodel, "facts": facts,
            "mode_a_recall": round(ra.coverage, 3) if ra else None,
            "mode_b_draft": round(rb_draft.coverage, 3) if rb_draft else None,
            "mode_b_answercov": round(rb.coverage, 3) if rb else None,
            "critique_damage": (round((rb_draft.coverage or 0) - (rb.coverage or 0), 3) if (rb_draft and rb) else None),
            "mode_b_honesty": round(rb.score, 3) if rb else None,
            "mode_c_ground": None if (rc is None or rc.error) else round(rc.score, 3),
            "mode_c_excluded": bool(rc is None or rc.error), "judge": (ra.model if ra else "?")}


async def main():
    forced = {r["id"]: r for r in json.load(open(ROOT / "eval/artifacts/forced_filler_bank_run.json"))["results"]}
    fq = forced[QID]
    state = json.loads(OUT_JSON.read_text())
    qrec = next(q for q in state["queries"] if q["qid"] == QID)
    qrec["status"] = "running"; qrec.pop("error", None)
    for leg_rec in qrec["legs"]:
        ps = fq["per_strategy"].get(leg_rec["leg"])
        leg_rec["ks"] = []
        if leg_rec["empty"]:
            continue
        all_chunks = ps["chunks"]
        eff = sorted({min(k, len(all_chunks)) for k in KS})
        leg_rec["eff_ks"] = eff
        for k in eff:
            leg_rec["ks"].append(await grade_k(qrec["question"], qrec["must_facts"], all_chunks[:k]))
            print(f"{QID} {leg_rec['leg']}@{k} ok", flush=True)
    qrec["status"] = "done"
    state["done"] = sum(1 for q in state["queries"] if q["status"] == "done")
    state["status"] = "done" if state["done"] == state["total"] else "running"
    state["updated_at"] = time.time()
    OUT_JSON.write_text(json.dumps(state, indent=2))
    print(f"PATCHED {QID} → done={state['done']}/{state['total']}", flush=True)


asyncio.run(main())
