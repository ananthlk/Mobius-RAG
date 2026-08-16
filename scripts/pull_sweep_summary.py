#!/usr/bin/env python3
"""Pulls one job's bank-run summary stats via the trace-explorer admin API
and appends/updates it in eval/artifacts/mode_strategy_sweep.json -- the
running results array for the a/b/c/d x copilot/default/thinking sweep."""
import json
import os
import subprocess
import sys
import urllib.request

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "artifacts", "mode_strategy_sweep.json")
BASE_URL = "https://mobius-rag-ortabkknqa-uc.a.run.app"


def get_admin_key():
    return subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest",
         "--secret=rag-admin-api-key", "--project=mobius-os-dev"],
        capture_output=True, text=True,
    ).stdout.strip()


def avg(vals):
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def pull_tokens(job_id, qids, key):
    """synth_input_tokens/synth_output_tokens only live on the FULL
    per-query trace (eval.*), not the bank summary -- one extra API call
    per query to get them (Ananth's ask, 2026-08-05: bring cost/tokens into
    the matrix so we can see what drives it, not just recall)."""
    in_toks, out_toks = [], []
    for qid in qids:
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/admin/trace-explorer/run-bank/result?job_id={job_id}&qid={qid}",
                headers={"X-Admin-Key": key},
            )
            r = json.load(urllib.request.urlopen(req))
            ev = r.get("eval") or {}
            in_toks.append(ev.get("synth_input_tokens"))
            out_toks.append(ev.get("synth_output_tokens"))
        except Exception:
            in_toks.append(None)
            out_toks.append(None)
    return avg(in_toks), avg(out_toks)


def pull(job_id, strategy, mode, key):
    req = urllib.request.Request(
        f"{BASE_URL}/admin/trace-explorer/run-bank/status?job_id={job_id}",
        headers={"X-Admin-Key": key},
    )
    d = json.load(urllib.request.urlopen(req))
    s = d.get("summaries", [])
    qids = [x["id"] for x in s if x.get("status") == "ok"]
    avg_synth_input_tokens, avg_synth_output_tokens = pull_tokens(job_id, qids, key)
    # K range extended to max(10, chunks_out) -- Eval's M1 catch, 2026-08-05:
    # capping at 10 made a deep-serving query's headline recall legitimately
    # exceed recall_at_k[10] whenever chunks_out>10, which reads as a broken
    # reconciliation when it's really just an untested rank-11+ tail.
    max_k = max([10] + [x.get("chunks_out") or 0 for x in s])
    recall_at_k_avg = {}
    for k in range(1, max_k + 1):
        pts = [(x.get("recall_at_k") or {}).get(str(k)) for x in s]
        recall_at_k_avg[str(k)] = avg(pts)
    return {
        "job_id": job_id, "strategy": strategy, "caller_mode": mode,
        "status": d.get("status"), "done": d.get("done"), "total": d.get("total"),
        "avg_recall": avg([x.get("recall") for x in s]),
        "avg_recall_answer": avg([x.get("recall_answer") for x in s]),
        "avg_authority_score": avg([x.get("authority_score") for x in s]),
        "avg_chunks_out": avg([x.get("chunks_out") for x in s]),
        "avg_retrieval_ms": avg([x.get("retrieval_ms") for x in s]),
        "avg_wall_ms": avg([x.get("wall_ms") for x in s]),
        "avg_signal_ratio": avg([x.get("signal_ratio") for x in s]),
        "avg_signal_density": avg([x.get("signal_density") for x in s]),
        "avg_fact_redundancy": avg([x.get("avg_fact_redundancy") for x in s]),
        "avg_pool_size": avg([x.get("pool_size") for x in s]),
        "avg_pool_bm25_top10_mean": avg([x.get("pool_bm25_top10_mean") for x in s]),
        "avg_pool_vector_top10_mean": avg([x.get("pool_vector_top10_mean") for x in s]),
        "recall_at_k_avg": recall_at_k_avg,
        "n_contradicted_total": sum(x.get("n_contradicted") or 0 for x in s),
        "n_hallucinated_total": sum(x.get("n_hallucinated_claims") or 0 for x in s),
        "avg_synth_input_tokens": avg_synth_input_tokens,
        "avg_synth_output_tokens": avg_synth_output_tokens,
    }


def main():
    job_id, strategy, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    key = get_admin_key()
    entry = pull(job_id, strategy, mode, key)

    results = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    results = [r for r in results if r["job_id"] != job_id]  # replace if re-pulled
    results.append(entry)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
