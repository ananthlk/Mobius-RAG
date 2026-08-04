"""Forced full-trace runner (Ananth, 2026-07-27): run EACH strategy (a/b/c/d/s)
individually via forced_strategy against the LIVE deployed retriever, and
capture the ENTIRE contract response -- every chunk (with authority_level!),
every routing_keys sub-field (executed_order, fill_depth, attempt_spans,
portfolio_fill, observer_final_verdicts/reasons, routing_verdict,
prescreen_not_ready_deferred_slots), latency breakdown, status -- nothing
summarized or dropped. Writes one JSON file the dashboard renders in full.

Usage:
  SERVICE_URL=https://mobius-rag-...run.app \
  .venv/bin/python scripts/forced_strategy_full_trace.py [--query "..."] [--out /path/results.json]
"""
import json, os, sys, time, urllib.request

SERVICE_URL = os.environ.get("SERVICE_URL", "https://mobius-rag-ortabkknqa-uc.a.run.app").rstrip("/")
STRATEGIES = ["a", "b", "c", "d", "s"]
DEFAULT_QUERY = "What is the timely filing deadline for Sunshine Health FL Medicaid claims?"

OUT = "/tmp/mobius_router_dashboard/forced_full_trace.json"
if "--out" in sys.argv:
    OUT = sys.argv[sys.argv.index("--out") + 1]
QUERY = DEFAULT_QUERY
if "--query" in sys.argv:
    QUERY = sys.argv[sys.argv.index("--query") + 1]


def _post(query, forced_strategy):
    body = json.dumps({
        "query": query, "forced_strategy": forced_strategy, "caller_mode": "chat.default",
    }).encode()
    req = urllib.request.Request(
        f"{SERVICE_URL}/api/retriever/answer", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            resp["_wall_ms"] = int((time.monotonic() - t0) * 1000)
            return resp
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    return {"_error": str(last)}


def main():
    state = {"kind": "forced-strategy-full-trace", "query": QUERY,
              "service": SERVICE_URL, "status": "running",
              "started_at": time.time(), "updated_at": time.time(), "strategies": {}}

    def flush():
        state["updated_at"] = time.time()
        with open(OUT, "w") as f:
            json.dump(state, f, indent=2)

    flush()
    for s in STRATEGIES:
        print(f"=== forcing strategy '{s}' ===", flush=True)
        resp = _post(QUERY, s)
        state["strategies"][s] = resp
        flush()
        c = resp.get("contract", {}) or {}
        rk = c.get("routing_keys", {}) or {}
        print(f"  status={c.get('status')} chunks={len(c.get('chunks', []))} "
              f"executed={rk.get('executed_order')} wall_ms={resp.get('_wall_ms')}", flush=True)
    state["status"] = "done"
    flush()
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
