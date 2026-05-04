"""Derive variance-aware priors from a N-repeat strategy matrix.

Reads ``eval/calibration/strategy_matrix_n*.json`` (rows include
``judge_score`` per (qid, strategy, repeat)) and produces a priors
table keyed by (qclass, strategy) with mean/std for accuracy.

Output is a Python snippet ready to paste into
``app/services/corpus_search_router.py``'s ``_BASE_PRIORS``.
"""
from __future__ import annotations
import argparse, collections, json, statistics
from pathlib import Path

QCLASSES = ("literal_anchor", "tight_pool", "wide_pool", "exploratory", "vague")

def qclass(features: dict) -> str:
    anc = features.get("literal_anchors") or []
    qt   = (features.get("query_type") or "").upper()
    pool = features.get("pool_size") or 0
    if anc: return "literal_anchor"
    if qt == "VAGUE": return "vague"
    if 0 < pool <= 500: return "tight_pool"
    if pool > 500: return "wide_pool"
    return "vague"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    args = ap.parse_args()
    rows = json.loads(Path(args.matrix).read_text())

    # qmap: qid → qclass (use strategy a's features which always carries pool)
    qmap = {}
    for r in rows:
        if r["strategy_forced"] == "a" and r["repeat"] == 0:
            qmap[r["qid"]] = qclass(r["features"])

    # Per-cell scores: (qid, strategy) → list[score]
    cell_scores: dict[tuple[str,str], list[float]] = collections.defaultdict(list)
    for r in rows:
        cell_scores[(r["qid"], r["strategy_forced"])].append(float(r["judge_score"] or 0.0))

    print(f"\nPER-CELL  (qid, strategy) → N samples, mean ± std")
    print("-" * 90)
    print(f"{'qid':10} {'qclass':16} {'s':3} {'n':>3}  {'scores':30} {'mean':>5} {'std':>5}")
    for (qid, s) in sorted(cell_scores):
        scores = cell_scores[(qid, s)]
        m = statistics.mean(scores)
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        scores_str = "[" + " ".join(f"{x:.2f}" for x in scores) + "]"
        print(f"{qid:10} {qmap.get(qid,'?'):16} {s:3} {len(scores):>3}  {scores_str:30} {m:.3f} {sd:.3f}")

    # Aggregate by (qclass, strategy): pool all repeats from all queries in that class
    by = collections.defaultdict(list)
    for (qid, s), scores in cell_scores.items():
        by[(qmap[qid], s)].extend(scores)

    print(f"\nAGGREGATE  (qclass, strategy) → pooled mean / std")
    print("-" * 70)
    print(f"{'qclass':16} {'s':3} {'n':>3}  {'mean':>5} {'std':>5}")
    aggregate: dict[str, dict[str, dict[str, float]]] = collections.defaultdict(dict)
    for q in QCLASSES:
        for s in ("a","b","c","d"):
            scores = by.get((q, s), [])
            if not scores:
                continue
            m = statistics.mean(scores)
            sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f"{q:16} {s:3} {len(scores):>3}  {m:.3f} {sd:.3f}")
            aggregate[s][q] = {"accuracy": round(m, 3), "accuracy_std": round(sd, 3)}

    print(f"\n# Patch — paste into _BASE_PRIORS (only accuracy/accuracy_std fields shown)")
    for s in ("a","b","c","d"):
        print(f'    "{s}": {{')
        for q in QCLASSES:
            cell = aggregate[s].get(q)
            if cell:
                print(f'        "{q}": StrategyPrior([...], accuracy={cell["accuracy"]}, accuracy_std={cell["accuracy_std"]}, recall_capacity=..., speed=...),')
        print(f'    }},')

if __name__ == "__main__":
    raise SystemExit(main())
