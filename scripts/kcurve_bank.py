"""Full-bank recall@K · 3-adjudicator calibration, streamed to a live JSON the
web dashboard polls. Generalizes scripts/inspect_one_query.py from one query to
the whole 22-query forced bank.

For each (query, leg, K∈{1,3,5,10}) — with the top-K discipline: synth + all
judges see ONLY the top-K chunks — we run:
  synth: draft → fact-check(grounding) → prune → final   (Chat-style composer)
  (a) chunk-recall     check_facts(must_facts, chunks_k, answer=None)
  (b·draft) answer-cov check_facts(must_facts, chunks_k, answer=DRAFT)   ← pre-critique
  (b·final) answer-cov check_facts(must_facts, chunks_k, answer=FINAL)   ← post-critique
  (c) groundedness     check_facts([],         chunks_k, answer=FINAL)   ← reference-free
critique_damage = b·draft − b·final = correct facts the critique wrongly deleted.

Judge LOCKED to rag_eval_adjudicate (gemini-2.5-pro). Queries run concurrently
(semaphore) so the whole bank finishes in ~20 min; state is rewritten atomically
after every K so the dashboard fills in live.

Usage: KCURVE_JSON=/path/results.json .venv/bin/python scripts/kcurve_bank.py [conc]
"""
import asyncio, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.services.fact_checker import check_facts          # noqa: E402
from app.services import llm_manager_client                # noqa: E402

LOCKED_STAGE = "rag_eval_adjudicate"
KS = [1, 3, 5, 10]
CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT_JSON = Path(os.environ.get("KCURVE_JSON", "/tmp/kcurve_bank.json"))
LEGS = ["a", "b", "c", "d", "s"]
LEG_LABEL = {"a": "our vector index", "b": "internal arm b", "c": "internal arm c",
             "d": "web (Google)", "s": "payor fact-store"}

SYNTH_SYSTEM = (
    "You are a claims/payer support assistant. Answer the question using ONLY "
    "the provided source passages — state the codes, day-counts, and yes/no "
    "determinations they support. If the passages don't answer it, say so. "
    "Do not invent facts."
)

_write_lock = asyncio.Lock()
_state = None


async def write_state():
    async with _write_lock:
        _state["updated_at"] = time.time()
        tmp = OUT_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(_state, indent=2))
        tmp.replace(OUT_JSON)


async def _gen(system, user):
    raw, meta = await llm_manager_client.generate(
        system=system, user=user, stage=LOCKED_STAGE, max_tokens=2048)
    return raw.strip(), (meta or {}).get("model") or "unknown"


async def synth(query, chunks):
    """draft → fact-check(grounding) → prune → final. Sees EXACTLY `chunks`."""
    body = "\n\n".join(f"[{i+1}] {c.get('text','')}" for i, c in enumerate(chunks))
    draft, model = await _gen(SYNTH_SYSTEM, f"Question: {query}\n\nPassages:\n{body}\n\nAnswer:")
    crit = await check_facts(query=query, must_facts=[], chunks=chunks, answer=draft, stage=LOCKED_STAGE)
    halluc = list(crit.hallucinated_claims or [])
    if not halluc:
        return draft, draft, [], model
    halluc_txt = "\n".join(f"- {h}" for h in halluc)
    final, _ = await _gen(
        SYNTH_SYSTEM,
        f"Question: {query}\n\nPassages:\n{body}\n\nDraft answer:\n{draft}\n\n"
        f"A grounding check verified these specific claims are NOT supported by "
        f"ANY passage:\n{halluc_txt}\n\nRewrite the answer: DELETE each of those "
        f"claims exactly. Do NOT add anything. Keep everything else. Output the "
        f"corrected answer only:")
    return final, draft, halluc, model


async def grade_k(query, must_facts, chunks_k):
    final, draft, pruned, smodel = await synth(query, chunks_k)
    ra, rb_draft, rb, rc = await asyncio.gather(
        check_facts(query=query, must_facts=must_facts, chunks=chunks_k, answer=None, stage=LOCKED_STAGE),
        check_facts(query=query, must_facts=must_facts, chunks=chunks_k, answer=draft, stage=LOCKED_STAGE),
        check_facts(query=query, must_facts=must_facts, chunks=chunks_k, answer=final, stage=LOCKED_STAGE),
        check_facts(query=query, must_facts=[], chunks=chunks_k, answer=final, stage=LOCKED_STAGE),
    )
    facts = []
    for v in (ra.verdicts if ra else []):
        in_answer = bool(rb and any(getattr(bv, 'fact', '') == getattr(v, 'fact', '') and bv.support >= 1.0 for bv in rb.verdicts))
        facts.append({"fact": getattr(v, 'fact', '?'), "in_chunk": v.support, "in_answer": in_answer})
    return {
        "k": len(chunks_k),
        "draft": draft, "pruned": pruned, "final": final, "synth_model": smodel,
        "facts": facts,
        "mode_a_recall": round(ra.coverage, 3) if ra else None,
        "mode_b_draft": round(rb_draft.coverage, 3) if rb_draft else None,
        "mode_b_answercov": round(rb.coverage, 3) if rb else None,
        "critique_damage": (round((rb_draft.coverage or 0) - (rb.coverage or 0), 3)
                            if (rb_draft and rb) else None),
        "mode_b_honesty": round(rb.score, 3) if rb else None,
        "mode_c_ground": None if (rc is None or rc.error) else round(rc.score, 3),
        "mode_c_excluded": bool(rc is None or rc.error),
        "judge": (ra.model if ra else "?"),
    }


async def run_query(qrec, forced_q):
    qrec["status"] = "running"; await write_state()
    for leg_rec in qrec["legs"]:
        ps = forced_q["per_strategy"].get(leg_rec["leg"])
        if leg_rec["empty"]:
            continue
        all_chunks = ps["chunks"]
        eff_ks = sorted({min(k, len(all_chunks)) for k in KS})
        leg_rec["eff_ks"] = eff_ks
        for k in eff_ks:
            row = await grade_k(qrec["question"], qrec["must_facts"], all_chunks[:k])
            leg_rec["ks"].append(row)
            await write_state()
    qrec["status"] = "done"
    _state["done"] += 1
    await write_state()
    print(f"[{_state['done']}/{_state['total']}] {qrec['qid']} done", flush=True)


async def main():
    global _state
    forced = json.load(open(ROOT / "eval/artifacts/forced_filler_bank_run.json"))["results"]
    _state = {
        "bank": "queries_cmhc (forced) · 22q",
        "judge": "rag_eval_adjudicate → locked gemini-2.5-pro",
        "status": "running", "total": len(forced), "done": 0,
        "started_at": time.time(), "updated_at": time.time(),
        "queries": [],
    }
    forced_by_id = {}
    for fq in forced:
        forced_by_id[fq["id"]] = fq
        _state["queries"].append({
            "qid": fq["id"], "question": fq["query"],
            "must_facts": fq.get("must_facts", []), "status": "pending",
            "legs": [{
                "leg": leg, "label": LEG_LABEL[leg],
                "occupancy": (fq["per_strategy"].get(leg) or {}).get("occupancy", 0),
                "chunks_available": len((fq["per_strategy"].get(leg) or {}).get("chunks") or []),
                "empty": not ((fq["per_strategy"].get(leg) or {}).get("chunks")),
                "eff_ks": [], "ks": [],
            } for leg in LEGS],
        })
    await write_state()

    sem = asyncio.Semaphore(CONC)

    async def guarded(qrec):
        async with sem:
            try:
                await run_query(qrec, forced_by_id[qrec["qid"]])
            except Exception as e:  # noqa: BLE001
                qrec["status"] = "error"; qrec["error"] = str(e)[:200]
                await write_state()
                print(f"ERROR {qrec['qid']}: {e}", flush=True)

    await asyncio.gather(*(guarded(q) for q in _state["queries"]))
    _state["status"] = "done"; await write_state()
    print(f"BANK COMPLETE — {_state['done']}/{_state['total']} → {OUT_JSON}", flush=True)


asyncio.run(main())
