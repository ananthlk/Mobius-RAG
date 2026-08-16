"""Priors Lab — repeatable computation of Router priors cells from real
per-query bank-run data, with the population/reconciliation rules Eval
ratified live (2026-08-05, this session's depth_3 fold) encoded once instead
of re-derived by hand every time:

  - recall_lift is a MULTIPLIED-into-the-chain field (allocation.py:397
    composes it with accuracy_estimate) -- it must be computed over the FULL
    attempt population, failures counted as 0. So must accuracy_estimate,
    for the same reason (a strategy's own zero-output failures must not be
    conditioned away from a field that gets multiplied downstream).
  - authority is a standalone GATE field (strategy_authority_eligible),
    never multiplied -- it is correctly computed CONDITIONAL on the
    strategy having produced content to judge (zero-output rows excluded).
    Its n is documented separately when it differs from the cell's n.
  - k0 (nominal fill capacity) is the n-weighted mean of the REAL observed
    capacity per row, not a guessed constant -- rows at different caller
    modes can have genuinely different capacity (chat.copilot=7 vs
    chat.default/thinking=10 confirmed live 2026-08-05).
  - bucket cutoffs are derived from the REAL pool_size distribution of the
    data being folded, not the hardcoded depth_bucket cutoffs in priors.py
    (50/200/500/5000) -- those were set without reference to any real
    corpus and left 4 of 5 buckets permanently empty on the CMHC bank
    (every query >= 425, structurally never below bucket 3). This module
    computes cutoffs live from whatever population is handed to it.

Computation (compute_priors_table) is read-only against bank-run data and
never touches eval/priors_bootstrap.yaml. Applying a computed cell TO that
file (apply_cell_to_yaml_text) is a separate, explicit, human-approved step
-- Phase 1 (2026-08-05, Ananth's directive): local-file-only, no live Cloud
Run write path yet (the deployed container's filesystem is ephemeral and
would silently revert on the next unrelated deploy -- see module comment at
the call site). A durable, live-write path is planned for later, once
priors_bootstrap.yaml itself moves to persistent (GCS-backed) storage the
way bank-run results already are.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PriorsLabRow:
    """One (query, caller_mode, strategy) observation, matching the fields
    already threaded through sweep_per_query_rows.json + the per-query
    fillers_ms telemetry."""

    query_id: str
    caller_mode: str
    strategy: str
    pool_size: Optional[int]
    recall: Optional[float]  # chunk-level recall (what recall_lift measures)
    recall_answer: Optional[float]  # answer-level recall (None on zero-output)
    authority: Optional[float]  # None on zero-output (nothing to judge)
    capacity: Optional[int] = None  # real fill_depth capacity for this row
    fillers_ms: Optional[float] = None  # real per-strategy filler latency


@dataclass
class PriorsCell:
    bucket: str
    strategy: str
    caller_mode: Optional[str]  # None when pooled across modes
    n: int
    recall_lift: Optional[float]
    accuracy_estimate: Optional[float]
    authority: Optional[float]
    authority_n: int
    k0: Optional[int]
    latency_p50_ms: Optional[int]
    reconciliation_ok: bool  # recall_lift * accuracy_estimate ~= mean(answer_recall over n, zeros for None)
    reconciliation_delta: Optional[float] = None  # abs(recall_lift*accuracy_estimate - mean_answer_recall); display-enrichment, gates nothing (the bool gates). None when reconciliation can't be assessed (recall_lift 0/None).
    warnings: list[str] = field(default_factory=list)

    def to_priors_dict(self) -> dict:
        """Same key shape as a eval/priors_bootstrap.yaml strategy entry --
        cost_per_attempt intentionally omitted (never derived from real
        calibration anywhere in this file yet; don't fabricate it here
        either)."""
        return {
            "recall_lift": self.recall_lift,
            "accuracy_estimate": self.accuracy_estimate,
            "authority": self.authority,
            "n": self.n,
            "k0": self.k0,
            "latency_p50_ms": self.latency_p50_ms,
        }


def derive_bucket_cutoffs(pool_sizes: list[int], n_buckets: int = 3) -> list[int]:
    """Quantile cutoffs from the REAL observed pool_size distribution.
    n_buckets=3 -> 2 cutoffs (tertiles), splitting into that many
    roughly-equal-count bins. Deduplicates identical quantile values (a
    thin/lumpy distribution can otherwise emit a degenerate zero-width
    bucket)."""
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")
    sizes = sorted(s for s in pool_sizes if s is not None)
    if not sizes:
        return []
    cuts = []
    for i in range(1, n_buckets):
        idx = min(len(sizes) - 1, (len(sizes) * i) // n_buckets)
        cuts.append(sizes[idx])
    # dedupe while preserving order (identical quantiles collapse buckets)
    out: list[int] = []
    for c in cuts:
        if not out or c != out[-1]:
            out.append(c)
    return out


def assign_bucket(pool_size: Optional[int], cutoffs: list[int]) -> str:
    """cutoffs=[906, 1202] -> 'b0 (<906)', 'b1 (906-1201)', 'b2 (>=1202)'.
    A missing pool_size goes to its own bucket rather than silently
    joining b0 -- matches priors.py's compute_depth_bucket's own
    "missing metadata -> broadest bucket" fail-open convention."""
    if pool_size is None:
        return "unknown (no pool_size)"
    if not cutoffs:
        return "b0 (all)"
    for i, cut in enumerate(cutoffs):
        if pool_size < cut:
            lo = "" if i == 0 else str(cutoffs[i - 1])
            return f"b{i} (<{cut})" if i == 0 else f"b{i} ({lo}-{cut - 1})"
    return f"b{len(cutoffs)} (>={cutoffs[-1]})"


def _mean(vals: list[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def compute_cell(
    rows: list[PriorsLabRow], bucket: str, strategy: str, caller_mode: Optional[str] = None,
) -> PriorsCell:
    """Fold one (bucket, strategy[, caller_mode]) population into a priors
    cell, per Eval's ratified rules (see module docstring)."""
    n = len(rows)
    warnings: list[str] = []

    recall_lift = _mean([r.recall for r in rows])

    # accuracy_estimate: multiplicative field -- zero-output rows (recall_answer
    # is None) count as 0 in this mean, same population as recall_lift.
    answer_recall_zeroed = [0.0 if r.recall_answer is None else r.recall_answer for r in rows]
    mean_answer_recall = _mean(answer_recall_zeroed)
    accuracy_estimate = None
    reconciliation_ok = False
    reconciliation_delta = None
    if mean_answer_recall is not None and recall_lift:
        accuracy_estimate = min(1.0, max(0.0, mean_answer_recall / recall_lift))
        # Gate on the RAW delta (unchanged 1e-6 threshold, no behavior shift);
        # store a ROUNDED copy for display. When the clamp doesn't bite, the raw
        # product == mean_answer_recall to float precision (~1e-16); when it bites
        # (mean_answer_recall > recall_lift), the delta is the real gap. Rounding
        # the stored value must NOT feed the bool — a borderline raw delta just
        # under 1e-6 could round up to 1e-6 and wrongly flip reconciliation_ok.
        raw_delta = abs(recall_lift * accuracy_estimate - mean_answer_recall)
        reconciliation_delta = round(raw_delta, 6)
        reconciliation_ok = raw_delta < 1e-6
    elif mean_answer_recall is not None and not recall_lift:
        warnings.append("recall_lift is 0 or missing -- accuracy_estimate undefined (would divide by zero)")

    zero_output_rows = [r for r in rows if r.recall_answer is None]
    if zero_output_rows:
        warnings.append(
            f"{len(zero_output_rows)}/{n} rows are zero-output (recall_answer=None) -- "
            f"counted as 0 in accuracy_estimate's numerator, EXCLUDED from authority's mean below."
        )

    # authority: gate field, conditional on the strategy having produced
    # content to judge -- correctly excludes zero-output rows.
    authority_rows = [r.authority for r in rows if r.authority is not None]
    authority = _mean(authority_rows)
    authority_n = len(authority_rows)
    if authority_n != n and authority_n > 0:
        warnings.append(f"authority is n={authority_n}, not the cell's n={n} -- documented, not a bug.")

    # k0: n-weighted mean of REAL observed capacity, not a guessed constant.
    capacities = [r.capacity for r in rows if r.capacity is not None]
    k0 = round(sum(capacities) / len(capacities)) if capacities else None
    if capacities and len(capacities) != n:
        warnings.append(f"k0 computed from {len(capacities)}/{n} rows with known capacity.")

    # latency: median of REAL per-strategy filler-only latency (fillers_ms),
    # never the total pipeline wall_ms.
    latencies = [r.fillers_ms for r in rows if r.fillers_ms is not None]
    latency_p50_ms = round(statistics.median(latencies)) if latencies else None

    return PriorsCell(
        bucket=bucket, strategy=strategy, caller_mode=caller_mode, n=n,
        recall_lift=round(recall_lift, 4) if recall_lift is not None else None,
        accuracy_estimate=round(accuracy_estimate, 4) if accuracy_estimate is not None else None,
        authority=round(authority, 4) if authority is not None else None,
        authority_n=authority_n,
        k0=k0, latency_p50_ms=latency_p50_ms,
        reconciliation_ok=reconciliation_ok, reconciliation_delta=reconciliation_delta,
        warnings=warnings,
    )


def compute_priors_table(
    rows: list[PriorsLabRow], n_buckets: int = 3, split_by_mode: bool = False,
    cutoffs: Optional[list[int]] = None,
) -> dict:
    """Full pipeline: derive cutoffs (or use given ones) from this row set's
    real pool_size distribution, bucket every row, fold into cells grouped
    by (bucket[, caller_mode], strategy). Returns a JSON-able dict:
    {cutoffs, buckets: [{bucket, caller_mode, cells: {strategy: PriorsCell}}]}."""
    pool_sizes = [r.pool_size for r in rows if r.pool_size is not None]
    resolved_cutoffs = cutoffs if cutoffs is not None else derive_bucket_cutoffs(pool_sizes, n_buckets)

    groups: dict[tuple, list[PriorsLabRow]] = {}
    for r in rows:
        b = assign_bucket(r.pool_size, resolved_cutoffs)
        key = (b, r.caller_mode if split_by_mode else None)
        groups.setdefault(key, []).append(r)

    buckets_out = []
    for (bucket, mode), grouped_rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        by_strategy: dict[str, list[PriorsLabRow]] = {}
        for r in grouped_rows:
            by_strategy.setdefault(r.strategy, []).append(r)
        cells = {
            s: compute_cell(strategy_rows, bucket, s, mode)
            for s, strategy_rows in sorted(by_strategy.items())
        }
        buckets_out.append({
            "bucket": bucket,
            "caller_mode": mode,
            "n_rows": len(grouped_rows),
            "cells": {s: c.to_priors_dict() for s, c in cells.items()},
            "warnings": {s: c.warnings for s, c in cells.items() if c.warnings},
            "reconciliation_ok": {s: c.reconciliation_ok for s, c in cells.items()},
        })

    return {"cutoffs": resolved_cutoffs, "split_by_mode": split_by_mode, "buckets": buckets_out}


# ---------------------------------------------------------------------------
# Apply (Phase 1, local-file-only -- Ananth's directive 2026-08-05): patch a
# SINGLE computed cell into eval/priors_bootstrap.yaml's real depth_N/
# strategy block, in place, preserving every comment and every other line
# byte-for-byte. Only recall_lift/accuracy_estimate/authority/n/k0 are
# touched by default -- latency_p50_ms carries hand-curated methodology
# comments (see depth_3/d's caveat) that a blind overwrite would destroy, so
# it's opt-in via include_latency and only applied when that line carries no
# existing inline comment (refuses rather than silently deleting one).
#
# This only makes sense for a bucket that maps 1:1 onto priors.py's real
# compute_depth_bucket bins (depth_bucket=0..4) -- the auto-derived quantile
# buckets compute_priors_table produces for exploration (b0/b1/b2 from
# THIS data's own distribution) are NOT the same cutoffs Router's live
# lookup uses, and there is no mode dimension in the live schema at all
# (compute_priors_table's split_by_mode output can't be applied directly).
# Landing an exploratory finding into production is a deliberate, separate
# decision -- this function only ever writes what it's explicitly told to.
# ---------------------------------------------------------------------------

# The applied identity of a cell = exactly the fields apply_cell_to_yaml_text can
# WRITE. latency_p50_ms is deliberately EXCLUDED: it is computed-and-surfaced but
# HUMAN-CURATED on write (apply refuses to overwrite its methodology comments), so
# it never lands from the cell and is NOT part of the applied identity. Governance
# class: "curated / out-of-sha-governance". Do NOT add it back — the sha must change
# iff a LANDED value changed, else §102 recompute-and-compare fires false drift on a
# field the file-write doesn't touch. (Ruled by Eval-RAG 2026-08-12; supersedes the
# proposal-doc's aspirational 6-field set.) A latency-drift canary, if wanted, belongs
# as a separate run-level check on bank_run metadata, never smuggled into this sha.
_APPLIABLE_FIELDS = ("recall_lift", "accuracy_estimate", "authority", "n", "k0")
_SHA_FLOAT_FIELDS = frozenset({"recall_lift", "accuracy_estimate", "authority"})
_SHA_INT_FIELDS = frozenset({"depth_bucket", "n", "k0"})


class PriorsApplyError(Exception):
    pass


def _sha_value_token(key: str, value) -> str:
    """Canonical byte token for one field (Eval-RAG serialization contract,
    2026-08-12): None -> literal null; float fields -> fixed 4dp with -0.0
    normalized to 0.0; int fields -> bare integer; strings -> JSON-quoted.
    Fixed 4dp (not json.dumps float repr) makes the bytes deterministic across
    processes/versions -- json.dumps(round(x,4)) loses trailing zeros and hits
    float-repr edge cases, defeating §102 recompute-and-compare."""
    if value is None:
        return "null"
    if key in _SHA_FLOAT_FIELDS:
        v = float(value) + 0.0        # -0.0 + 0.0 -> +0.0
        if v == 0:
            v = 0.0                   # belt-and-suspenders: no -0.0000 token
        return f"{v:.4f}"
    if key in _SHA_INT_FIELDS:
        return str(int(value))
    return json.dumps(value)          # strategy + any string: proper JSON quoting


def cell_sha256(depth_bucket: int, strategy: str, cell: dict) -> str:
    """Canonical hash over exactly the fields apply can write (the applied
    identity), deterministic across processes/versions.

    WRITTEN == HASHED invariant (Eval-RAG, 2026-08-12): the value hashed here
    MUST be the same round-to-4dp value apply_cell_to_yaml_text writes to the
    file. compute_cell rounds every appliable float to 4dp once (round(v,4));
    that single canonical value feeds BOTH the file-write and this hash, so a
    published file value always reconciles to its own sha. Do not round
    differently in either path or the spine self-contradicts."""
    fields = {
        "depth_bucket": depth_bucket, "strategy": strategy,
        **{k: cell.get(k) for k in _APPLIABLE_FIELDS},
    }
    parts = [f"{json.dumps(k)}:{_sha_value_token(k, fields[k])}" for k in sorted(fields)]
    blob = "{" + ",".join(parts) + "}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _find_block_span(lines: list[str], header_pattern: str, indent: str) -> tuple[int, int]:
    """Find [start, end) line range of a `<indent><header_pattern>:` block --
    from the header line to the next line at the SAME indent (a sibling key)
    or a shallower indent (dedent out of the block), whichever comes first.
    Raises if the header isn't found exactly once."""
    header_re = re.compile(rf"^{indent}{header_pattern}:\s*(#.*)?$")
    matches = [i for i, l in enumerate(lines) if header_re.match(l)]
    if len(matches) == 0:
        raise PriorsApplyError(f"block '{header_pattern}' not found at indent {len(indent)}")
    if len(matches) > 1:
        raise PriorsApplyError(f"block '{header_pattern}' found {len(matches)} times -- ambiguous, refusing")
    start = matches[0]
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].rstrip("\n")
        if stripped.strip() == "":
            continue
        this_indent = len(stripped) - len(stripped.lstrip(" "))
        if this_indent <= len(indent):
            end = i
            break
    return start, end


def apply_cell_to_yaml_text(
    yaml_text: str, depth_bucket: int, strategy: str, cell: dict, include_latency: bool = False,
) -> tuple[str, dict]:
    """Patch ONE strategy cell within ONE depth_N block, in place. Returns
    (new_yaml_text, applied_fields). Raises PriorsApplyError on anything
    ambiguous or unsafe -- never guesses, never partially applies."""
    import yaml as _yaml

    lines = yaml_text.splitlines(keepends=True)
    depth_start, depth_end = _find_block_span(lines, rf"depth_{depth_bucket}", "  ")
    depth_block = lines[depth_start:depth_end]
    local_strat_start, local_strat_end = _find_block_span(
        depth_block, re.escape(strategy), "    ",
    )
    strat_start = depth_start + local_strat_start
    strat_end = depth_start + local_strat_end

    fields_to_apply = list(_APPLIABLE_FIELDS) + (["latency_p50_ms"] if include_latency else [])
    applied = {}
    block = lines[strat_start:strat_end]
    for field_name in fields_to_apply:
        if cell.get(field_name) is None:
            continue
        value = cell[field_name]
        key_re = re.compile(rf"^(\s+){re.escape(field_name)}:\s*[^\s#]+(\s*#.*)?\s*$")
        matched_idx = None
        for i, l in enumerate(block):
            m = key_re.match(l)
            if m:
                if matched_idx is not None:
                    raise PriorsApplyError(f"key '{field_name}' appears twice in {strategy}'s block -- refusing")
                matched_idx = i
        if matched_idx is None:
            raise PriorsApplyError(f"key '{field_name}' not found in {strategy}'s block -- refusing (would need to invent a new line, not doing that silently)")
        m = key_re.match(block[matched_idx])
        indent_str, trailing_comment = m.group(1), m.group(2)
        if trailing_comment and field_name == "latency_p50_ms":
            raise PriorsApplyError(
                f"latency_p50_ms carries an inline comment -- refusing to overwrite it silently "
                f"(pass a pre-merged comment or apply that field by hand)"
            )
        formatted = json.dumps(value) if not isinstance(value, (int, float)) else str(value)
        new_line = f"{indent_str}{field_name}: {formatted}{trailing_comment or ''}\n"
        applied[field_name] = value
        block[matched_idx] = new_line
    lines[strat_start:strat_end] = block
    new_text = "".join(lines)

    # Round-trip safety check: the new text must parse, and the target cell
    # must read back exactly what was applied -- refuse to write anything
    # that doesn't verify.
    parsed = _yaml.safe_load(new_text)
    readback = ((parsed.get("seed_priors") or {}).get(f"depth_{depth_bucket}") or {}).get(strategy) or {}
    for field_name, value in applied.items():
        if readback.get(field_name) != value:
            raise PriorsApplyError(
                f"round-trip verification failed for {field_name}: wrote {value!r}, read back {readback.get(field_name)!r} -- refusing to save"
            )
    return new_text, applied
