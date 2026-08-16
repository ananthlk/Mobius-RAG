"""§7 review→publish gate — the APPROVABILITY predicate + sha-diff (Eval-RAG lane).

Division of labor (see project_eval_workflow_tooling):
  - Eval-architect owns priors_store.publish_cell — the WRITE-side guard that
    enforces "only the human-signed sha can land" (fail-closed, append-only).
  - Eval-RAG (here) owns "which cells are approvable + what the reviewer sees":
    the approvability predicate, the ack/override gates, and the field-by-field
    sha-diff vs the incumbent published cell.

This module is deliberately PURE: it consumes one dict from
`priors_store.fetch_computed_cells(...)` and returns a verdict dict. No DB, no
async, no UI — so it is surface-agnostic (whatever renders the gate calls this)
and trivially testable. The human reads the verdict, clears any gates, and signs
the returned `sha`; publish_cell then enforces that exact sha.

Severity model (ruled 2026-08-12): exactly ONE hard block; everything else is
surface-and-acknowledge.
  reconciliation_ok=false  -> HARD BLOCK, never approvable, no override
  n < N_FLOOR_OVERRIDE     -> approvable ONLY with explicit override + reason
  N_FLOOR_OVERRIDE<=n<N_FLOOR_CLEAN -> approvable with mandatory thin-n ack
  population warnings       -> approvable with ack (one per warning)
  fact_checker_version != incumbent -> comparability CAUTION ack (not a block)
"""
from __future__ import annotations

from typing import Any, Optional

# Publishability floors on the cell's sample size `n` (Eval-RAG's call, tunable).
# Below OVERRIDE: CIs too wide to trust a published prior over an incumbent, but
# legitimate bootstrapping (a new strategy's first real data) must remain
# possible -> override-with-reason, not an absolute block.
N_FLOOR_OVERRIDE = 20
N_FLOOR_CLEAN = 30

# Appliable fields whose change constitutes a real sha-diff (mirrors the
# cell_sha256 field set; latency is curated/out-of-sha and NOT diffed here).
_APPLIABLE_FIELDS = ("recall_lift", "accuracy_estimate", "authority", "k0")

# Rounding precision of the canonical (written == hashed) value.
_DP = 4

# Gate severities.
HARD_BLOCK = "hard_block"   # not approvable at all
OVERRIDE = "override"       # approvable only with explicit override + reason
ACK = "ack"                # approvable once the reviewer acknowledges


def _round(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    r = round(float(v), _DP)
    return 0.0 if r == 0.0 else r  # normalize -0.0 -> 0.0 (matches sha canonical)


def _sha_diff(cell: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Field-by-field diff of appliable values vs the incumbent published cell.

    Returns None when this is the FIRST publish for the slot (no incumbent).
    Otherwise reports which appliable fields changed (at canonical 4dp), whether
    the sha changed, and whether it's an identical no-op re-publish.
    """
    pub = cell.get("currently_published")
    if not pub:
        return {"is_first_publish": True, "sha_changed": True, "changed_fields": {}, "no_op": False}

    new_ap = cell.get("appliable") or {}
    old_ap = pub.get("appliable") or {}
    changed: dict[str, Any] = {}
    for f in _APPLIABLE_FIELDS:
        nv, ov = _round(new_ap.get(f)), _round(old_ap.get(f))
        if nv != ov:
            delta = None if (nv is None or ov is None) else round(nv - ov, _DP)
            changed[f] = {"from": ov, "to": nv, "delta": delta}

    sha_changed = cell.get("sha256") != pub.get("sha256")
    return {
        "is_first_publish": False,
        "sha_changed": sha_changed,
        "changed_fields": changed,
        # sha unchanged AND no appliable field moved => publishing this is a no-op.
        "no_op": (not sha_changed) and (not changed),
    }


def evaluate_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Return the §7 approvability verdict for one fetched computed cell.

    Verdict shape:
      {
        "cell_id", "slot": "depth_3/b/default",
        "approvable": bool,                 # False iff a hard block fired
        "hard_block": {code,message}|None,  # the single absolute
        "gates": [ {code, severity, message, requires_reason} ],
        "sha": "<the sha the reviewer signs>" | None (None iff not approvable),
        "sha_diff": {...} | None,
        "reconciliation": {"ok": bool, "delta": float|None},
        "verdict": "hard_block"|"clean"|"needs_ack"|"needs_override",
      }
    """
    cid = cell.get("cell_id")
    slot = f"{cell.get('bucket')}/{cell.get('strategy')}/{cell.get('caller_mode')}"
    recon_ok = bool(cell.get("reconciliation_ok"))
    recon_delta = cell.get("reconciliation_delta")  # may be None until v1.1
    n = cell.get("n")

    hard_block: Optional[dict[str, Any]] = None
    gates: list[dict[str, Any]] = []

    # (1) reconciliation_ok=false -> the one absolute. A non-reconciling cell is
    # corrupt (the population-coupling invariant rl*ae==mean(answer_recall) is
    # violated): never approvable, no override path.
    if not recon_ok:
        hard_block = {
            "code": "reconciliation_failed",
            "message": (
                "reconciliation_ok=false: recall_lift*accuracy_estimate does not "
                "reconcile to mean(answer_recall). Cell is corrupt — not publishable. "
                "Investigate the population-coupling (recall_lift & accuracy_estimate "
                "must be computed over the identical recall-not-None population)."
            ),
        }

    # (2)/(3) sample-size floors.
    if n is None:
        hard_block = hard_block or {
            "code": "n_missing",
            "message": "cell has no sample size n — cannot assess publishability.",
        }
    elif n < N_FLOOR_OVERRIDE:
        gates.append({
            "code": "thin_n_override",
            "severity": OVERRIDE,
            "requires_reason": True,
            "message": (
                f"n={n} < {N_FLOOR_OVERRIDE}: sample too thin for confident publish "
                f"over an incumbent. Approvable only with an explicit override and a "
                f"captured reason (e.g. bootstrapping a new strategy's first data)."
            ),
        })
    elif n < N_FLOOR_CLEAN:
        gates.append({
            "code": "thin_n_ack",
            "severity": ACK,
            "requires_reason": False,
            "message": f"n={n} in [{N_FLOOR_OVERRIDE},{N_FLOOR_CLEAN}): thin sample — acknowledge before publishing.",
        })

    # (4) population warnings -> one ack each (clamp-bite, high-failure-fraction,
    # authority_n << n, etc. — the compute layer emits these strings).
    for w in (cell.get("warnings") or []):
        gates.append({
            "code": "population_warning",
            "severity": ACK,
            "requires_reason": False,
            "message": str(w),
        })

    # (5) cross-ruler comparability -> caution, NOT a block. The sha-diff vs the
    # incumbent is not apples-to-apples if graded under a different ruler version.
    pub = cell.get("currently_published")
    if pub and pub.get("fact_checker_version") and cell.get("fact_checker_version"):
        if pub["fact_checker_version"] != cell["fact_checker_version"]:
            gates.append({
                "code": "ruler_version_mismatch",
                "severity": ACK,
                "requires_reason": False,
                "message": (
                    f"incumbent graded under {pub['fact_checker_version']}, this cell "
                    f"under {cell['fact_checker_version']}: the sha-diff is cross-ruler, "
                    f"not apples-to-apples. Acknowledge before replacing."
                ),
            })

    diff = _sha_diff(cell)

    # No-op re-publish (identical sha, nothing moved) -> surface as an ack so the
    # reviewer doesn't publish a redundant audit row unknowingly.
    if diff and diff.get("no_op"):
        gates.append({
            "code": "no_op_republish",
            "severity": ACK,
            "requires_reason": False,
            "message": "identical to the currently-published cell (same sha, no field moved) — publishing is a redundant no-op.",
        })

    approvable = hard_block is None
    if not approvable:
        verdict = "hard_block"
    elif any(g["severity"] == OVERRIDE for g in gates):
        verdict = "needs_override"
    elif gates:
        verdict = "needs_ack"
    else:
        verdict = "clean"

    return {
        "cell_id": cid,
        "slot": slot,
        "approvable": approvable,
        "hard_block": hard_block,
        "gates": gates,
        "sha": cell.get("sha256") if approvable else None,
        "sha_diff": diff,
        "reconciliation": {"ok": recon_ok, "delta": recon_delta},
        "verdict": verdict,
    }


def evaluate_run(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate every cell in a fetched bank-run and add a roll-up summary.

    Returns {"cells": [verdict,...], "summary": {counts by verdict, blocked slots}}.
    Convenience for a gate UI that renders the whole run at once.
    """
    verdicts = [evaluate_cell(c) for c in cells]
    summary = {
        "total": len(verdicts),
        "clean": sum(1 for v in verdicts if v["verdict"] == "clean"),
        "needs_ack": sum(1 for v in verdicts if v["verdict"] == "needs_ack"),
        "needs_override": sum(1 for v in verdicts if v["verdict"] == "needs_override"),
        "hard_block": sum(1 for v in verdicts if v["verdict"] == "hard_block"),
        "blocked_slots": [v["slot"] for v in verdicts if v["verdict"] == "hard_block"],
    }
    return {"cells": verdicts, "summary": summary}
