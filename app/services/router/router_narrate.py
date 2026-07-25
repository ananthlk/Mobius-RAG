"""Human-readable narration of a Router decision, rendered from the trace.

Same principle as Gate/Reformat narrate(): state everything transparently, in
order, with the actual numbers and arithmetic — no pick-and-summarize. Shows
BOTH confidence tracks: the mean (telemetry) and the Wilson lower bound (the
§2a-enforced quantity), plus n behind every prior.

PHI rule (per Gate's standard): this output is computed on demand for
Diagnostics/chat and NEVER persisted. persist.py must not receive it —
enforced by test_tracing.py.
"""

from __future__ import annotations

from app.services.router.tracing import DecisionTrace, SlotTrace

_DEPTH_LABELS = {0: "tight", 1: "tight-moderate", 2: "moderate", 3: "broad-moderate", 4: "broad"}


def _narrate_slot(st: SlotTrace, adjusted_bar: float, mode: str, level_pct: str) -> str:
    lines = []
    lines.append(
        f'Slot "{st.slot_id}" (priority {st.priority}, role={st.slot_semantics}'
        f"{'' if st.required else ', OPTIONAL'}): depth_bucket={st.depth_bucket} "
        f"({_DEPTH_LABELS.get(st.depth_bucket, '?')}; pool_size={st.pool_size}, "
        f"top_score={st.top_score_percentile}"
        + (f", distinct_topk={st.distinct_content_topk}"
           if st.distinct_content_topk is not None else "")
        + ")."
    )
    verb = "Chose" if mode == "optimizer" else "Tried"
    for step in st.steps:
        if step.action == "added":
            lines.append(
                f"  {verb} '{step.strategy_id}' — prior[{step.prior_source}]: "
                f"recall_lift={step.recall_lift:.3f} (n={step.n}, lb{level_pct}={step.lb_lift:.3f}), "
                f"latency_p50={step.latency_p50_ms}ms, "
                f"accuracy={step.accuracy_estimate:.3f}, cost={step.cost:g}."
            )
            lines.append(
                f"    Cumulative mean: {step.arithmetic()} | "
                f"LB: {step.lb_before:.4f}→{step.lb_after:.4f}. "
                f"Chain latency: {step.latency_before_ms}→{step.latency_after_ms}ms."
            )
        else:
            prior_bit = (
                f" (prior recall_lift={step.recall_lift:.3f}, latency={step.latency_p50_ms}ms)"
                if step.prior_source else ""
            )
            lines.append(
                f"  Skipped '{step.strategy_id}' — {step.skip_reason}{prior_bit}."
            )
    chain = ",".join(st.final_chain) if st.final_chain else "(empty)"
    verdict = st.status
    if st.status == "OPTIONAL":
        verdict = "OPTIONAL (supplementary slot — reported, not gated)"
    terminal_bit = ""
    if st.terminal_action == "clarify_low_confidence":
        terminal_bit = (" Terminal leg: clarify_low_confidence (Filler q — "
                        "confidence too low to answer outright).")
    elif st.terminal_action == "fast_exit_no_viable":
        terminal_bit = (" Terminal leg: fast_exit_no_viable (Filler e — "
                        "no viable strategy for this slot).")
    payload_bit = (
        f" worst-case payload {st.payload_tokens_worst_case} tokens;"
        if st.payload_tokens_worst_case is not None else ""
    )
    lines.append(
        f"  RESULT: sequence=[{chain}], mean={st.final_confidence:.4f}, "
        f"lb{level_pct}={st.final_lb:.4f} vs adjusted bar {adjusted_bar:.4f} → "
        f"{verdict}; worst-case latency {st.final_latency_ms}ms;{payload_bit} "
        f"stopped: {st.stop_reason}."
        f"{terminal_bit}"
    )
    return "\n".join(lines)


def narrate(trace: DecisionTrace) -> str:
    """Render the full decision trace as prose. Never persist this output."""
    lines = []
    role = f", {trace.role.upper()} plan" if trace.role else ""
    level_pct = f"{trace.confidence_level:.0%}".rstrip("%") if trace.confidence_level else "95"
    lines.append(
        f"ROUTER [{trace.mode.upper()}{role}] — {trace.mode_reason}"
    )
    lines.append(
        f"Priors: {trace.priors_version} ({trace.priors_source}). "
        f"Caller mode: {trace.caller_mode} → tolerance ±{trace.tolerance_pct:.0%}. "
        f"Confidence bar {trace.confidence_bar:.2f} → adjusted {trace.adjusted_confidence_bar:.4f}, "
        f"enforced on the {trace.confidence_level:.0%} Wilson lower bound PER SLOT. "
        f"Budget {trace.speed_budget_ms}ms → per-slot allowance {trace.latency_allowance_ms:.0f}ms."
    )
    for st in trace.slots:
        lines.append("")
        lines.append(_narrate_slot(st, trace.adjusted_confidence_bar, trace.mode, level_pct))
    lines.append("")
    if trace.outcome and trace.outcome not in ("forced",):
        verdict = trace.outcome.upper()
        if trace.infeasibility_reason:
            verdict += f" — {trace.infeasibility_reason}"
    else:
        verdict = "FORCED BYPASS" if trace.outcome == "forced" else (
            "FEASIBLE" if trace.feasible else f"INFEASIBLE — {trace.infeasibility_reason}")
    lines.append(
        f"OUTCOME: {verdict}. Aggregate mean {trace.aggregate_confidence:.4f} "
        f"(telemetry only — {trace.aggregate_arithmetic or 'per-slot LB gate in force'})."
    )
    if trace.helpers:
        lines.append(
            "HELPERS (recall loop fell short — user-pain aids, no priors): "
            + ", ".join(trace.helpers) + "."
        )
    return "\n".join(lines)
