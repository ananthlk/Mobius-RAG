"""Router-model throttle run: drive the LIVE deployed retriever with each of the
3 allocators (greedy / optimizer / bayesian) per query, capture the actually-
dispatched chunks + real latency, and grade with the SAME locked-judge harness
as the forced-arm calibration. Streams to router_results.json for the dashboard.

Unlike the forced arms (per-K sweep), each router model produces ONE retrieval
set + one synthesized answer per query — graded as a single point:
  (a) chunk-recall   = must_facts vs the router's retrieved chunks
  (b) answer-cov     = must_facts vs the router's OWN synthesized answer (real prod output)
  (c) groundedness   = is that answer traceable to the chunks (no golden)
Plus: real total latency (ms) from GCP, and which strategies the ladder chose.

Usage:
  SERVICE_URL=https://mobius-rag-...run.app  KCURVE_JSON=.../router_results.json \
  CHAT_INTERNAL_LLM_URL=... MOBIUS_SKILL_LLM_INTERNAL_KEY=... VERTEX_PROJECT_ID=... \
  .venv/bin/python scripts/router_throttle_run.py [--dump] [--one QID]
"""
import asyncio, json, os, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.services.fact_checker import check_facts          # noqa: E402

LOCKED_STAGE = "rag_eval_adjudicate"
MODES = ["greedy", "optimizer", "bayesian"]
SERVICE_URL = os.environ.get("SERVICE_URL", "https://mobius-rag-1032922478554.us-central1.run.app").rstrip("/")
ADMIN_KEY = os.environ.get("RAG_ADMIN_API_KEY", "")
OUT_JSON = Path(os.environ.get("KCURVE_JSON", "/tmp/router_results.json"))
DUMP = "--dump" in sys.argv
ONE = (sys.argv[sys.argv.index("--one") + 1] if "--one" in sys.argv else None)


def _post(query, mode):
    body = json.dumps({"query": query, "mode_override": mode, "caller_mode": "chat.default"}).encode()
    last = None
    for attempt in range(4):  # Cloud Run min=max=1 recycles → transient conn drops
        req = urllib.request.Request(f"{SERVICE_URL}/api/retriever/answer", data=body,
                                     headers={"Content-Type": "application/json",
                                              **({"X-API-Key": ADMIN_KEY} if ADMIN_KEY else {})})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                resp = json.loads(r.read())
            resp["_wall_ms"] = int((time.monotonic() - t0) * 1000)
            return resp
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    raise last


def collect_chunks(obj, acc):
    """Recursively find chunk-like dicts ({text, chunk_id}) anywhere in the contract."""
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str) and ("chunk_id" in obj or "index" in obj):
            acc.append({"chunk_id": obj.get("chunk_id"), "text": obj.get("text")})
        for v in obj.values():
            collect_chunks(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            collect_chunks(v, acc)


def get_chunks(resp):
    acc = []
    collect_chunks(resp.get("contract", resp), acc)
    # dedup by chunk_id/text, preserve order
    seen = set(); out = []
    for c in acc:
        k = c.get("chunk_id") or c.get("text", "")[:80]
        if k in seen:
            continue
        seen.add(k); out.append(c)
    return out


def get_answer(resp):
    c = resp.get("contract", {}) or {}
    for k in ("answer_text", "answer_markdown", "answer", "markdown", "synthesis", "body", "text", "final_answer"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # nested synthesis object
    syn = c.get("synthesis") or {}
    if isinstance(syn, dict):
        for k in ("answer_markdown", "answer", "markdown", "text"):
            if isinstance(syn.get(k), str) and syn[k].strip():
                return syn[k]
    return ""


_lock = asyncio.Lock()
_state = None


async def write_state():
    async with _lock:
        _state["updated_at"] = time.time()
        tmp = OUT_JSON.with_suffix(".tmp"); tmp.write_text(json.dumps(_state, indent=2)); tmp.replace(OUT_JSON)


async def grade(query, must_facts, chunks, answer):
    ra, rb, rc = await asyncio.gather(
        check_facts(query=query, must_facts=must_facts, chunks=chunks, answer=None, stage=LOCKED_STAGE),
        check_facts(query=query, must_facts=must_facts, chunks=chunks, answer=answer, stage=LOCKED_STAGE),
        check_facts(query=query, must_facts=[], chunks=chunks, answer=answer, stage=LOCKED_STAGE))
    facts = [{"fact": getattr(v, "fact", "?"), "in_chunk": v.support} for v in (ra.verdicts if ra else [])]
    return {
        "recall": round(ra.coverage, 3) if ra else None,
        "answer_cov": round(rb.coverage, 3) if rb else None,
        "ground": None if (rc is None or rc.error) else round(rc.score, 3),
        "facts": facts,
    }


async def main():
    global _state
    forced = json.load(open(ROOT / "eval/artifacts/forced_filler_bank_run.json"))["results"]
    if ONE:
        forced = [q for q in forced if q["id"] == ONE]
    _state = {"kind": "router-throttle", "service": SERVICE_URL,
              "judge": "rag_eval_adjudicate → locked gemini-2.5-pro",
              "modes": MODES, "status": "running", "total": len(forced), "done": 0,
              "updated_at": time.time(), "queries": []}
    for fq in forced:
        _state["queries"].append({"qid": fq["id"], "question": fq["query"],
                                  "must_facts": fq.get("must_facts", []), "status": "pending",
                                  "runs": {}})
    await write_state()

    for i, fq in enumerate(forced):
        qrec = _state["queries"][i]; qrec["status"] = "running"; await write_state()
        for mode in MODES:
            try:
                resp = _post(fq["query"], mode)
            except Exception as e:  # noqa: BLE001
                qrec["runs"][mode] = {"error": str(e)[:200]}; await write_state()
                print(f"{fq['id']} {mode} POST ERROR: {e}", flush=True); continue
            if DUMP:
                print(f"=== {fq['id']} {mode} contract keys ===", flush=True)
                print(json.dumps(list((resp.get('contract') or {}).keys()), indent=2), flush=True)
                print("chunks found:", len(get_chunks(resp)), " answer_len:", len(get_answer(resp)), flush=True)
                print("latency:", resp.get("latency_ms"), " dispatch:", resp.get("dispatch_path"),
                      " strategies:", resp.get("strategies_per_slot"), flush=True)
            chunks = get_chunks(resp); answer = get_answer(resp)
            g = await grade(fq["query"], fq.get("must_facts", []), chunks, answer)
            contract = resp.get("contract", {}) or {}
            rk = contract.get("routing_keys", {}) or {}
            qrec["runs"][mode] = {
                **g,
                "n_chunks": len(chunks),
                "answer": answer,
                "contract_status": contract.get("status"),
                "planned_ladder": rk.get("routing_ladder_per_slot"),
                "executed_order": rk.get("executed_order"),
                "routing_verdict": rk.get("routing_verdict"),
                "latency_total_ms": (resp.get("latency_ms") or {}).get("total_ms"),
                "latency_router_ms": (resp.get("latency_ms") or {}).get("router_ms"),
                "wall_ms": resp.get("_wall_ms"),
                "dispatch_path": resp.get("dispatch_path"),
            }
            await write_state()
            print(f"{fq['id']} {mode}: recall={g['recall']} ans={g['answer_cov']} ground={g['ground']} "
                  f"nchunks={len(chunks)} total_ms={(resp.get('latency_ms') or {}).get('total_ms')} "
                  f"dispatch={resp.get('dispatch_path')}", flush=True)
        qrec["status"] = "done"; _state["done"] += 1; await write_state()
    _state["status"] = "done"; await write_state()
    print(f"ROUTER THROTTLE COMPLETE {_state['done']}/{_state['total']} → {OUT_JSON}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
