"""Oracle Matrix — per-query comparison of a real bank run (e.g. greedy,
optimizer, bayesian, or any forced/auto dispatch) against the per-query
oracle (best single arm among a/b/c/d), using Eval's ship-gate SCORE
formula. Built 2026-08-06 from what was originally a one-off manual
analysis (the "greedy vs oracle" matrix) -- generalized so any bank run
can get this comparison, not just that specific eval.

SCORE = authority * clamp(recall_answer - w_c*contradiction_rate -
                           w_h*hallucination_rate, 0, 1)
contradiction_rate = min(n_contradicted / n_facts_total, 1)
hallucination_rate = min(n_hallucinated / n_facts_total, 1)

w_c=1.0/w_h=0.5 are Eval's defaults from the ship-gate pilots -- NOT a
fully ratified constant (Eval flagged w_c as a real domain-judgment
question, still open). Callers may override; the response always echoes
the weights used so nothing downstream mistakes them for fixed truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def score(
    recall_answer: Optional[float], authority: Optional[float],
    n_contradicted: Optional[int], n_hallucinated: Optional[int],
    n_facts_total: Optional[int], w_c: float = 1.0, w_h: float = 0.5,
) -> Optional[float]:
    if recall_answer is None or authority is None or not n_facts_total:
        return None
    contra_r = min((n_contradicted or 0) / n_facts_total, 1)
    halluc_r = min((n_hallucinated or 0) / n_facts_total, 1)
    q_adj = max(0.0, min(1.0, recall_answer - w_c * contra_r - w_h * halluc_r))
    return round(authority * q_adj, 4)


@dataclass
class OracleMatrixRow:
    query_id: str
    query: str
    test_chain: list[str]
    test_score: Optional[float]
    arm_scores: dict[str, Optional[float]]
    oracle_strategy: Optional[str]
    oracle_score: Optional[float]
    verdict: str  # BEAT | TIE | MISS | NO_DATA


def compute_oracle_matrix(
    test_rows: list[dict], arm_rows_by_strategy: dict[str, dict[str, dict]],
    w_c: float = 1.0, w_h: float = 0.5, tie_tolerance: float = 0.01,
) -> dict:
    """test_rows: one dict per query from the run being evaluated --
    {query_id, query, chain (list[str], the ACTUAL executed strategies,
    not planned), recall_answer, authority, n_contradicted, n_hallucinated,
    n_facts_total}.
    arm_rows_by_strategy: {strategy_id: {query_id: {recall_answer,
    authority, n_contradicted, n_hallucinated, n_facts_total}}} -- one
    single-arm forced run per strategy, SAME caller_mode as the test run
    (mixing modes would compare apples to oranges -- caller's job to pass
    matching-mode arm data)."""
    rows: list[OracleMatrixRow] = []
    beat = tie = miss = no_data = 0
    for t in test_rows:
        qid = t["query_id"]
        test_score = score(
            t.get("recall_answer"), t.get("authority"),
            t.get("n_contradicted"), t.get("n_hallucinated"),
            t.get("n_facts_total"), w_c, w_h,
        )
        arm_scores: dict[str, Optional[float]] = {}
        for strat, by_q in arm_rows_by_strategy.items():
            r = by_q.get(qid)
            if not r:
                continue
            arm_scores[strat] = score(
                r.get("recall_answer"), r.get("authority"),
                r.get("n_contradicted"), r.get("n_hallucinated"),
                r.get("n_facts_total") or t.get("n_facts_total"), w_c, w_h,
            )
        valid = {k: v for k, v in arm_scores.items() if v is not None}
        oracle_strategy = max(valid, key=valid.get) if valid else None
        oracle_score = valid.get(oracle_strategy) if oracle_strategy else None

        if test_score is None or oracle_score is None:
            verdict = "NO_DATA"
            no_data += 1
        else:
            gap = test_score - oracle_score
            if abs(gap) < tie_tolerance:
                verdict = "TIE"
                tie += 1
            elif gap > 0:
                verdict = "BEAT"
                beat += 1
            else:
                verdict = "MISS"
                miss += 1

        rows.append(OracleMatrixRow(
            query_id=qid, query=t.get("query", ""), test_chain=t.get("chain") or [],
            test_score=test_score, arm_scores=arm_scores,
            oracle_strategy=oracle_strategy, oracle_score=oracle_score, verdict=verdict,
        ))

    return {
        "weights": {"w_c": w_c, "w_h": w_h},
        "summary": {"beat": beat, "tie": tie, "miss": miss, "no_data": no_data},
        "rows": [
            {
                "query_id": r.query_id, "query": r.query, "test_chain": r.test_chain,
                "test_score": r.test_score, "arm_scores": r.arm_scores,
                "oracle_strategy": r.oracle_strategy, "oracle_score": r.oracle_score,
                "verdict": r.verdict,
            }
            for r in rows
        ],
    }
