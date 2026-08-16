"""Router-model K-curve run: replicate the EXACT forced-arm analysis (recall@K
K=1,3,5,10 · 3 adjudicators · draft→critique→final · char-adjusted · overlap)
but on the LIVE router's actually-fetched chunks, with the 3 allocators
(greedy/optimizer/bayesian) as the "legs".

Per (query, allocator): POST the live endpoint with mode_override → take the
router's dispatched chunks → sweep K over that chunk set with the identical
grade_k (same as kcurve_bank.py). Emits router_kcurve.json (same schema as
results.json so index.html renders it) + router_chars.json (per-chunk lengths).

Usage:
  SERVICE_URL=... KCURVE_JSON=.../router_kcurve.json CHARS_JSON=.../router_chars.json \
  CHAT_INTERNAL_LLM_URL=... MOBIUS_SKILL_LLM_INTERNAL_KEY=... VERTEX_PROJECT_ID=... \
  .venv/bin/python scripts/router_kcurve_run.py [--one QID]
"""
import asyncio, json, os, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.services.fact_checker import check_facts          # noqa: E402
from app.services import llm_manager_client                # noqa: E402

LOCKED_STAGE = "rag_eval_adjudicate"
KS = [1, 3, 5, 10]
MODES = ["greedy", "optimizer", "bayesian"]
MODE_LABEL = {"greedy": "greedy (seq-fallback)", "optimizer": "optimizer (Wilson-LB)", "bayesian": "bayesian (Beta-LB)"}
SERVICE_URL = os.environ.get("SERVICE_URL", "https://mobius-rag-1032922478554.us-central1.run.app").rstrip("/")
OUT_JSON = Path(os.environ.get("KCURVE_JSON", "/tmp/router_kcurve.json"))
CHARS_JSON = Path(os.environ.get("CHARS_JSON", "/tmp/router_chars.json"))
ONE = (sys.argv[sys.argv.index("--one") + 1] if "--one" in sys.argv else None)

SYNTH_SYSTEM = (
    "You are a claims/payer support assistant. Answer the question using ONLY "
    "the provided source passages — state the codes, day-counts, and yes/no "
    "determinations they support. If the passages don't answer it, say so. "
    "Do not invent facts.")


def _post(query, mode):
    body = json.dumps({"query": query, "mode_override": mode, "caller_mode": "chat.default"}).encode()
    last = None
    for attempt in range(4):
        req = urllib.request.Request(f"{SERVICE_URL}/api/retriever/answer", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(3 * (attempt + 1))
    raise last


def get_chunks(resp):
    acc = []
    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("text"), str) and ("chunk_id" in o or "index" in o):
                acc.append({"chunk_id": o.get("chunk_id"), "text": o.get("text")})
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(resp.get("contract", resp))
    seen, out = set(), []
    for c in acc:
        k = c.get("chunk_id") or (c.get("text") or "")[:80]
        if k not in seen:
            seen.add(k); out.append(c)
    return out


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


_state = None
_chars = {}


def write_state():
    _state["updated_at"] = time.time()
    tmp = OUT_JSON.with_suffix(".tmp"); tmp.write_text(json.dumps(_state, indent=2)); tmp.replace(OUT_JSON)
    CHARS_JSON.write_text(json.dumps(_chars))


async def main():
    global _state
    forced = json.load(open(ROOT / "eval/artifacts/forced_filler_bank_run.json"))["results"]
    if ONE:
        forced = [q for q in forced if q["id"] == ONE]
    _state = {"bank": "router allocators · live GCP", "judge": "rag_eval_adjudicate → locked gemini-2.5-pro",
              "status": "running", "total": len(forced), "done": 0, "updated_at": time.time(), "queries": []}
    for fq in forced:
        _state["queries"].append({"qid": fq["id"], "question": fq["query"],
                                  "must_facts": fq.get("must_facts", []), "status": "pending",
                                  "legs": [{"leg": m, "label": MODE_LABEL[m], "occupancy": 0,
                                            "chunks_available": 0, "empty": True, "eff_ks": [], "ks": []} for m in MODES]})
        _chars[fq["id"]] = {}
    write_state()

    for i, fq in enumerate(forced):
        qrec = _state["queries"][i]; qrec["status"] = "running"; write_state()
        for li, mode in enumerate(MODES):
            leg = qrec["legs"][li]
            try:
                resp = _post(fq["query"], mode)
            except Exception as e:  # noqa: BLE001
                leg["error"] = str(e)[:160]; write_state()
                print(f"{fq['id']} {mode} POST ERROR: {e}", flush=True); continue
            chunks = get_chunks(resp)
            contract = resp.get("contract", {}) or {}
            rk = contract.get("routing_keys", {}) or {}
            leg["occupancy"] = len(chunks); leg["chunks_available"] = len(chunks); leg["empty"] = not chunks
            leg["latency_total_ms"] = (resp.get("latency_ms") or {}).get("total_ms")
            leg["planned_ladder"] = rk.get("routing_ladder_per_slot")
            leg["executed_order"] = rk.get("executed_order")
            leg["contract_status"] = contract.get("status")
            _chars[fq["id"]][mode] = [len(c.get("text") or "") for c in chunks]
            if chunks:
                eff = sorted({min(k, len(chunks)) for k in KS})
                leg["eff_ks"] = eff
                for k in eff:
                    leg["ks"].append(await grade_k(fq["query"], fq.get("must_facts", []), chunks[:k]))
                    write_state()
            r0 = leg["ks"][-1] if leg["ks"] else {}
            print(f"{fq['id']} {mode}: nchunks={len(chunks)} recall={r0.get('mode_a_recall')} "
                  f"ran={rk.get('executed_order')} ms={leg['latency_total_ms']}", flush=True)
        qrec["status"] = "done"; _state["done"] += 1; write_state()
    _state["status"] = "done"; write_state()
    print(f"ROUTER K-CURVE COMPLETE {_state['done']}/{_state['total']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
