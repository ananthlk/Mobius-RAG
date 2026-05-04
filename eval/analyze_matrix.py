"""Analyze the strategy×query verdict matrix and produce per-strategy
precision/recall, sliced priors, and the disagreement list.

Reads ``eval/calibration/strategy_matrix_*.json`` (produced by
``eval.run_matrix``) and emits:

  1. Per-strategy aggregate — TP/FP/TN/FN/MISS counts + P/R/F1/U +
     honest-abstain and over-abstain rates.
  2. Per-query attribution — which state each cell landed in, oracle flag.
  3. Sliced priors — per (feature × strategy) F1 and U, the input the
     router-tuner consumes to fit weights.
  4. Disagreement list — queries where the current router's pick is
     dominated by another strategy.

Outcome classification (per cell ``(query, strategy)``):

  ``TP``   verdict==correct OR score>=0.7 with verdict in {correct,partial}
  ``WP``   verdict==partial AND 0.3<=score<0.7  (weak partial — counts in TP)
  ``FP``   verdict==wrong OR (answered AND score<0.3)
  ``HA``   verdict==unable_to_verify AND score==0.5  (becomes TN or FN)
  ``MISS`` empty result, score==0, or agent_error
  ``HA → TN`` when oracle says no strategy could answer (honest abstain)
  ``HA → FN`` when oracle says some strategy COULD answer (over-abstain)

Single-knob utility:  ``U = α·TP − β·FP + γ·TN − δ·FN``

Defaults:  α=1.0  β=2.0  γ=0.5  δ=1.0   (override via CLI flags)
Oracle threshold: a query is ``answerable`` iff max_s(score) >= 0.7.

Usage::

    python -m eval.analyze_matrix \\
        --matrix eval/calibration/strategy_matrix_<ts>.json \\
        --out    eval/calibration/priors_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import collections
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Knobs:
    alpha: float = 1.0   # TP weight (correct answer)
    beta:  float = 2.0   # FP weight (wrong answer — punished extra under risk-aversion)
    gamma: float = 0.5   # TN weight (honest abstain)
    delta: float = 1.0   # FN weight (over-abstain)
    oracle_threshold: float = 0.7   # query is answerable iff some strategy reached this


# ----- Outcome classification -----------------------------------------------

def _classify_cell(verdict: str, score: float) -> str:
    """Return one of {TP, WP, FP, HA, MISS}.

    HA is a placeholder; it becomes TN/FN after the oracle check.
    """
    v = (verdict or "").lower()
    s = float(score or 0.0)
    if v == "agent_error":
        return "MISS"
    if v == "correct":
        return "TP"
    if v == "partial":
        return "TP" if s >= 0.7 else "WP"
    if v == "wrong":
        return "FP"
    if v == "unable_to_verify":
        if s == 0.0:
            return "MISS"
        # 0.5 honest-abstain band
        return "HA"
    # Fallback for unknown verdict labels
    if s >= 0.7:
        return "TP"
    if s == 0.0:
        return "MISS"
    return "WP" if s >= 0.3 else "FP"


def _promote_ha(state: str, *, oracle_answerable: bool) -> str:
    if state != "HA":
        return state
    return "FN" if oracle_answerable else "TN"


# ----- Per-strategy aggregate -----------------------------------------------

@dataclass
class StratStats:
    n: int = 0
    tp_score_sum: float = 0.0  # partial-credit sum of TP/WP scores
    tp_count: int = 0
    wp_count: int = 0
    fp_count: int = 0
    tn_count: int = 0
    fn_count: int = 0
    miss_count: int = 0
    high_conf_wrong: int = 0   # confidently wrong cells (FP + conf=high)

    @property
    def precision(self) -> float | None:
        denom = self.tp_score_sum + self.fp_count
        return None if denom == 0 else self.tp_score_sum / denom

    def recall(self, n_answerable: int) -> float | None:
        denom = self.tp_score_sum + self.fn_count + self.miss_count
        return None if denom == 0 else self.tp_score_sum / denom

    def f1(self, n_answerable: int) -> float | None:
        p = self.precision; r = self.recall(n_answerable)
        if p is None or r is None or (p + r) == 0:
            return None
        return 2.0 * p * r / (p + r)

    def utility(self, k: Knobs) -> float:
        return (k.alpha * self.tp_score_sum
                - k.beta  * self.fp_count
                + k.gamma * self.tn_count
                - k.delta * self.fn_count)

    def honest_abstain_rate(self, n_unanswerable: int) -> float | None:
        return None if n_unanswerable == 0 else self.tn_count / n_unanswerable

    def over_abstain_rate(self, n_answerable: int) -> float | None:
        return None if n_answerable == 0 else self.fn_count / n_answerable


# ----- Feature-key derivation for slicing ----------------------------------

def _feature_key(features: dict[str, Any]) -> tuple[str, ...]:
    """Coarse buckets for slicing. Keep a small handful so n-per-slice isn't 1."""
    anc = "anc>=1" if (features.get("literal_anchors") or []) else "anc=0"
    qt = (features.get("query_type") or "?").upper()[:7]   # CONCEPT/PRECIS/MIXED
    pool = features.get("pool_size") or 0
    if pool == 0:
        pool_b = "pool=NA"
    elif pool < 50:
        pool_b = "pool<50"
    elif pool < 300:
        pool_b = "pool<300"
    else:
        pool_b = "pool>=300"
    pers = (features.get("persona") or "?")[:5]
    return (anc, qt, pool_b, pers)


# ----- Driver ---------------------------------------------------------------

def analyze(matrix_path: Path, knobs: Knobs, out_path: Path | None) -> dict:
    rows = json.loads(matrix_path.read_text())
    by_qid: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        by_qid[r["qid"]][r["strategy_forced"]] = r

    # ── Oracle per query ────────────────────────────────────────────────
    oracle: dict[str, dict[str, Any]] = {}
    for qid, cells in by_qid.items():
        scores = {s: float(c.get("judge_score") or 0.0) for s, c in cells.items()}
        best = max(scores, key=scores.get) if scores else None
        oracle[qid] = {
            "best_strategy": best,
            "best_score": scores[best] if best else 0.0,
            "answerable": (max(scores.values()) if scores else 0.0) >= knobs.oracle_threshold,
            "scores": scores,
        }
    n_answerable   = sum(1 for o in oracle.values() if o["answerable"])
    n_unanswerable = len(oracle) - n_answerable

    # ── Per-cell classification (with HA promotion) ──────────────────────
    classified: list[dict[str, Any]] = []
    strats = sorted({s for cells in by_qid.values() for s in cells})
    per_strat: dict[str, StratStats] = {s: StratStats() for s in strats}
    sliced: dict[tuple, dict[str, StratStats]] = collections.defaultdict(
        lambda: {s: StratStats() for s in strats}
    )

    for qid, cells in by_qid.items():
        ans = oracle[qid]["answerable"]
        slice_key = _feature_key(cells[strats[0]]["features"])
        for s in strats:
            c = cells.get(s)
            if not c:
                continue
            v, sc = c.get("judge_verdict", ""), float(c.get("judge_score") or 0.0)
            raw_state = _classify_cell(v, sc)
            state = _promote_ha(raw_state, oracle_answerable=ans)
            conf  = c.get("agent_confidence")
            classified.append({
                "qid": qid, "strategy": s, "verdict": v, "score": sc,
                "raw_state": raw_state, "state": state, "conf": conf,
                "oracle_answerable": ans,
            })

            ss = per_strat[s]; sliced_ss = sliced[slice_key][s]
            ss.n += 1; sliced_ss.n += 1
            if state in ("TP", "WP"):
                ss.tp_score_sum += sc; sliced_ss.tp_score_sum += sc
                if state == "TP": ss.tp_count += 1; sliced_ss.tp_count += 1
                else:             ss.wp_count += 1; sliced_ss.wp_count += 1
            elif state == "FP":
                ss.fp_count += 1; sliced_ss.fp_count += 1
                if conf == "high":
                    ss.high_conf_wrong += 1; sliced_ss.high_conf_wrong += 1
            elif state == "TN":
                ss.tn_count += 1; sliced_ss.tn_count += 1
            elif state == "FN":
                ss.fn_count += 1; sliced_ss.fn_count += 1
            elif state == "MISS":
                ss.miss_count += 1; sliced_ss.miss_count += 1

    # ── Disagreement list ─────────────────────────────────────────────────
    # Define router_pick = "a" (current behaviour for these queries was a-heavy;
    # we'll later swap in the actual chosen strategy from logged routing).
    disagreements: list[dict[str, Any]] = []
    for qid, o in oracle.items():
        # Per-cell utility (single query); pick the one with the highest utility
        # under our knobs.
        cells = by_qid[qid]
        cell_u: dict[str, float] = {}
        for s, c in cells.items():
            v, sc = c.get("judge_verdict",""), float(c.get("judge_score") or 0.0)
            raw  = _classify_cell(v, sc)
            st   = _promote_ha(raw, oracle_answerable=o["answerable"])
            u = 0.0
            if st in ("TP","WP"): u =  knobs.alpha * sc
            elif st == "FP":      u = -knobs.beta
            elif st == "TN":      u =  knobs.gamma
            elif st == "FN":      u = -knobs.delta
            cell_u[s] = u
        best_s = max(cell_u, key=cell_u.get)
        best_u = cell_u[best_s]
        # Router currently mostly picks 'a' (until we wire actual logged choice in)
        router_pick = "a"
        gap = best_u - cell_u.get(router_pick, 0.0)
        if gap > 0:
            disagreements.append({
                "qid": qid, "router_pick": router_pick, "oracle_pick": best_s,
                "gap": round(gap, 3),
                "router_utility": round(cell_u.get(router_pick, 0.0), 3),
                "oracle_utility": round(best_u, 3),
                "all_utilities": {k: round(v, 3) for k, v in cell_u.items()},
                "answerable": o["answerable"],
            })
    disagreements.sort(key=lambda d: -d["gap"])

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\nKnobs: α={knobs.alpha} β={knobs.beta} γ={knobs.gamma} δ={knobs.delta} oracle_threshold={knobs.oracle_threshold}")
    print(f"Queries: {len(oracle)}  answerable={n_answerable}  unanswerable={n_unanswerable}\n")

    print("=" * 102)
    print(f"{'strat':5} {'TP':>5}({'partial$':>8}) {'FP':>4} {'TN':>4} {'FN':>4} {'MISS':>4}   {'P':>6} {'R':>6} {'F1':>6} {'U':>7}   {'HA%':>5} {'OA%':>5}  {'hcW':>4}")
    print("-" * 102)
    for s in strats:
        st = per_strat[s]
        p  = st.precision
        r  = st.recall(n_answerable)
        f1 = st.f1(n_answerable)
        u  = st.utility(knobs)
        ha = st.honest_abstain_rate(n_unanswerable)
        oa = st.over_abstain_rate(n_answerable)
        def fmt(x):
            return "  -  " if x is None else f"{x:.3f}"
        print(f"{s:5} {st.tp_count:5}({st.tp_score_sum:8.2f}) "
              f"{st.fp_count:4} {st.tn_count:4} {st.fn_count:4} {st.miss_count:4}   "
              f"{fmt(p):>6} {fmt(r):>6} {fmt(f1):>6} {u:7.2f}   "
              f"{fmt(ha):>5} {fmt(oa):>5}  {st.high_conf_wrong:4}")
    print("=" * 102)

    # Sliced priors
    print(f"\nSLICED PRIORS  ({len(sliced)} feature combos)")
    print("=" * 102)
    print(f"{'slice':40}  n  " + "  ".join(f"{s:>5}_F1 {s:>5}_U" for s in strats))
    print("-" * 102)
    for key in sorted(sliced):
        line = f"{','.join(key):40}"
        # n is per-slice queries; same across strategies
        slice_n = max(sliced[key][s].n for s in strats) if strats else 0
        line += f"  {slice_n:1}  "
        for s in strats:
            ss = sliced[key][s]
            f1 = ss.f1(slice_n) or 0.0
            u  = ss.utility(knobs)
            line += f"  {f1:7.3f} {u:7.2f}"
        print(line)
    print("=" * 102)

    # Disagreements
    print(f"\nDISAGREEMENTS  (router='a' vs oracle pick under utility U)")
    print(f"  {'qid':9}  {'router':6}  {'oracle':6}  {'gap':>5}  {'router_U':>8}  {'oracle_U':>8}  per-strat utilities")
    for d in disagreements:
        utl = "  ".join(f"{k}={v:+.2f}" for k, v in d["all_utilities"].items())
        print(f"  {d['qid']:9}  {d['router_pick']:6}  {d['oracle_pick']:6}  {d['gap']:+.2f}  {d['router_utility']:+8.2f}  {d['oracle_utility']:+8.2f}  {utl}")

    out = {
        "knobs":          asdict(knobs),
        "n_queries":      len(oracle),
        "n_answerable":   n_answerable,
        "n_unanswerable": n_unanswerable,
        "per_strategy":   {s: {
            "n": per_strat[s].n,
            "tp_count": per_strat[s].tp_count,
            "wp_count": per_strat[s].wp_count,
            "fp_count": per_strat[s].fp_count,
            "tn_count": per_strat[s].tn_count,
            "fn_count": per_strat[s].fn_count,
            "miss_count": per_strat[s].miss_count,
            "high_conf_wrong": per_strat[s].high_conf_wrong,
            "tp_score_sum":  round(per_strat[s].tp_score_sum, 3),
            "precision":  per_strat[s].precision,
            "recall":     per_strat[s].recall(n_answerable),
            "f1":         per_strat[s].f1(n_answerable),
            "utility":    round(per_strat[s].utility(knobs), 3),
            "honest_abstain_rate": per_strat[s].honest_abstain_rate(n_unanswerable),
            "over_abstain_rate":   per_strat[s].over_abstain_rate(n_answerable),
        } for s in strats},
        "sliced": [
            {"slice": list(key), "n": max(sliced[key][s].n for s in strats),
             "per_strategy": {s: {
                 "f1":       sliced[key][s].f1(sliced[key][s].n),
                 "utility":  round(sliced[key][s].utility(knobs), 3),
                 "tp_count": sliced[key][s].tp_count,
                 "fp_count": sliced[key][s].fp_count,
                 "tn_count": sliced[key][s].tn_count,
                 "fn_count": sliced[key][s].fn_count,
                 "miss_count": sliced[key][s].miss_count,
             } for s in strats}}
            for key in sorted(sliced)
        ],
        "disagreements":  disagreements,
        "classified":     classified,
        "oracle":         oracle,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved → {out_path}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta",  type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--oracle-threshold", type=float, default=0.7)
    args = p.parse_args()

    knobs = Knobs(args.alpha, args.beta, args.gamma, args.delta, args.oracle_threshold)
    out = Path(args.out) if args.out else None
    analyze(Path(args.matrix), knobs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
