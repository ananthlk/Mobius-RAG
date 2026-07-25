"""ONE-WRITER enforcement: persist_decision() is the ONLY function that INSERTs rag_query_decisions.

Router writes exactly ONE row per query (24 columns after the 2026-07-23
dual-build addendum), capturing:
  - Dispatch decision (forced / greedy / optimizer)
  - `executed_ladder` (the plan Fillers actually walks — drives real outcomes)
  - `shadow_ladder` (the untaken allocator's plan — comparison-only, no outcome)
  - `confidence_bar` (the actual bar for this query — calibration must judge
    success against THIS, never a hardcoded threshold)
  - Estimated metrics (confidence, accuracy, latency, cost)
  - Deferred columns from upstream modules (gate_contour, reformat_posture, etc.)

TECH gate (f): exactly one function may INSERT into rag_query_decisions.
Both calibration (Eval) and production paths call this same function.

Schema status: the 24-column DDL is DB's to write/run as ONE combined
migration (design approved; executed_ladder/shadow_ladder/confidence_bar
widening communicated to DB + Eval 2026-07-23).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


async def persist_decision(
    db_session_factory,
    *,
    agent_id: str,
    query: str,
    is_calibration: bool = False,
    is_prod: bool = True,
    eval_run_id: Optional[str] = None,
    # Router outputs
    depth_bucket: int,
    strategy_chosen: str,
    strategy_sequence: list[str],
    executed_ladder: Optional[dict[str, Any]] = None,
    shadow_ladder: Optional[dict[str, Any]] = None,
    confidence_bar: Optional[float] = None,
    feature_vector: Optional[dict[str, Any]] = None,
    strategy_scores: Optional[dict[str, float]] = None,
    priors_version: str = "seed-2026-07-23",
    confidence: float = 0.0,
    accuracy_estimate: float = 0.0,
    cost: float = 0.0,
    total_ms: int = 0,
    leaf_key: str = "unknown",
    # Deferred columns from upstream modules
    gate_contour: Optional[str] = None,
    gate_underspecified_kind: Optional[str] = None,
    reformat_posture: Optional[str] = None,
    reformat_fanout_n: Optional[int] = None,
    # Optional context
    decision_id: Optional[str] = None,
) -> str:
    """Persist ONE Router decision to rag_query_decisions (21-column schema).

    CRITICAL: This is the ONLY site that writes to rag_query_decisions post-refactor.
    Called by:
      - prod: orchestrator.py on every production query (is_prod=True, is_calibration=False)
      - eval: calibrate.py on every eval run (is_calibration=True, eval_run_id set)
      - forced: orchestrator.py when caller forces a strategy (forced_strategy set, max_attempts=1)

    Errors are logged + swallowed (telemetry MUST NOT break the user's request).

    Args:
        db_session_factory: AsyncSessionLocal callable
        agent_id: identifier for the calling agent (e.g., 'router-4c')
        query: the user's query text
        is_calibration: True iff Eval is measuring in isolation (priors collection)
        is_prod: True for prod, False for eval/test
        eval_run_id: non-null only for eval paths
        depth_bucket: [0, 1, 2, 3, 4] from Pool's corpus-depth signal
        strategy_chosen: which strategy was actually executed (may differ from ladder if escalated)
        strategy_sequence: full ordered sequence Router produced (JSON list)
        feature_vector: context dict for bandit (None for s-rows = short-circuit)
        strategy_scores: per-strategy confidence scores (dict strategy_id → score)
        priors_version: tag for bandit tracking (default: seed version)
        confidence: aggregate confidence achieved (for calibration)
        accuracy_estimate: predicted accuracy
        cost: estimated cost (relative units)
        total_ms: query latency (MAX of all slot latencies, parallel model)
        leaf_key: deterministic key for audit trail (usually strategy_chosen or dominant slot)
        gate_contour: from Shape:Gate (deferred column)
        gate_underspecified_kind: from Shape:Gate (deferred column)
        reformat_posture: from Shape:Reformat (deferred column)
        reformat_fanout_n: from Shape:Reformat (deferred column)
        decision_id: pre-generated UUID (caller may set for API response timing)

    Returns:
        decision_id (UUID for this decision row)

    Frozen bandit contract (do NOT change):
        - 21-column schema (Router 11 + deferred 4 + shared 6)
        - depth_bucket + strategy_sequence + priors_version as calibration inputs
        - feature_vector non-null on non-s rows; s-rows NULL emergent
        - ONE-WRITER: only this function writes (both calibration + production paths)
    """
    from sqlalchemy import text as _sql

    decision_id = decision_id or str(uuid.uuid4())

    # Validate basic invariants
    if not isinstance(strategy_sequence, list):
        logger.warning(
            "persist_decision: strategy_sequence is not a list. decision_id=%s",
            decision_id,
        )
        strategy_sequence = [strategy_chosen] if strategy_chosen else []

    try:
        async with db_session_factory() as db:
            await db.execute(
                _sql(
                    """
                    INSERT INTO rag_query_decisions (
                        id, agent_id, query,
                        is_calibration, is_prod, eval_run_id,
                        depth_bucket, strategy_chosen, strategy_sequence,
                        executed_ladder, shadow_ladder, confidence_bar,
                        gate_contour, gate_underspecified_kind, reformat_posture, reformat_fanout_n,
                        feature_vector, strategy_scores, priors_version,
                        confidence, accuracy_estimate, cost,
                        total_ms, leaf_key
                    ) VALUES (
                        :decision_id, :agent_id, :query,
                        :is_calibration, :is_prod, :eval_run_id,
                        :depth_bucket, :strategy_chosen, :strategy_sequence,
                        :executed_ladder, :shadow_ladder, :confidence_bar,
                        :gate_contour, :gate_underspecified_kind, :reformat_posture, :reformat_fanout_n,
                        :feature_vector, :strategy_scores, :priors_version,
                        :confidence, :accuracy_estimate, :cost,
                        :total_ms, :leaf_key
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                {
                    "decision_id": decision_id,
                    "agent_id": agent_id,
                    "query": query,
                    "is_calibration": is_calibration,
                    "is_prod": is_prod,
                    "eval_run_id": eval_run_id,
                    "depth_bucket": depth_bucket,
                    "strategy_chosen": strategy_chosen,
                    "strategy_sequence": _json(strategy_sequence),
                    "executed_ladder": _json(executed_ladder),
                    "shadow_ladder": _json(shadow_ladder),
                    "confidence_bar": confidence_bar,
                    "gate_contour": gate_contour,
                    "gate_underspecified_kind": gate_underspecified_kind,
                    "reformat_posture": reformat_posture,
                    "reformat_fanout_n": reformat_fanout_n,
                    "feature_vector": _json(feature_vector),
                    "strategy_scores": _json(strategy_scores or {}),
                    "priors_version": priors_version,
                    "confidence": confidence,
                    "accuracy_estimate": accuracy_estimate,
                    "cost": cost,
                    "total_ms": total_ms,
                    "leaf_key": leaf_key,
                },
            )
            await db.commit()
            logger.info(
                "persist_decision: persisted %s (is_calibration=%s is_prod=%s eval_run_id=%s)",
                decision_id,
                is_calibration,
                is_prod,
                eval_run_id,
            )
    except Exception as exc:
        logger.warning(
            "persist_decision failed (non-fatal): %s decision_id=%s",
            exc,
            decision_id,
        )
        # Telemetry must not block the user's request
        # Return the ID anyway so caller can proceed

    return decision_id


def _json(v: Any) -> Optional[str]:
    """JSON-encode for JSONB/JSON column. None passes through."""
    if v is None:
        return None
    try:
        return json.dumps(v)
    except Exception as exc:
        logger.warning("_json encoding failed: %s value=%s", exc, v)
        return None
