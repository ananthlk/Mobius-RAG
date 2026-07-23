"""Friendly narrative generation for GateResult — the user-facing "thinking"
translation of the gate's structural decision.

Deterministic, template-based, NO LLM call — same constraint as the rest of
shape. Design: state the J/D/P path plainly (what was actually matched, per
axis) rather than trying to synthesize a single "best" topic phrase — a
prior version tried picking one representative phrase and got it wrong twice
(picked "claims" over "timely filing," then "sunshine health" over "timely
filing" — a domain-vs-jurisdiction salience problem that's genuinely hard to
get right by guessing). Just showing the path is more transparent and
sidesteps the guessing problem entirely.

Placement (per UX review, 2026-07-22):
  - narrate()      -> wire into the 12-field contract's `thinking_trace` slot.
                      Default-visible, chat-bubble register.
  - narrate_full()  -> Diagnostics-only (`shape_gate.reasoning_trace`), behind
                      an expandable "show reasoning" affordance — never
                      default-visible. This is debugging-adjacent prose, not
                      chat-bubble prose, per UX's read of the product's voice.

PHI-safety, STRENGTHENED per TECH's 2026-07-22 review (supersedes the earlier
"redact before persisting" framing UX proposed): `narrate_full()` echoes
`result.query` verbatim in its first line ("You asked: ..."). Per this
fleet's standing "PHI default OFF, fail closed" policy, a rule that depends
on every future caller remembering to redact before persisting is not
fail-closed — it fails open the moment someone forgets. Instead:
**`narrate_full()` output must NEVER be written to `rag_query_traces` or any
other persisted storage.** Compute it on-demand only, for the live
Diagnostics view, and discard it. `narrate()` has no such exposure (states
codes/counts only, never the raw query string) and is fine to persist
normally (e.g. into `thinking_trace`).
"""

from __future__ import annotations

from .contracts import Contour, GateResult


def _leaf_label(code: str) -> str:
    """``d:claims.timely_filing`` -> ``timely filing``; ``j:payor.sunshine_health`` -> ``sunshine health``."""
    stripped = code.split(":", 1)[1] if ":" in code else code
    leaf = stripped.split(".")[-1]
    return leaf.replace("_", " ")


def _axis_phrase(codes: list[str]) -> str | None:
    if not codes:
        return None
    labels = sorted({_leaf_label(c) for c in codes if _leaf_label(c) != "general"} or {_leaf_label(codes[0])})
    return ", ".join(labels)


# Structured, trackable prediction — the gate's forecast for what the REST of
# the pipeline (pool/fillers/observe/synthesis) will do with this query. This
# is a contract, not decoration: (label, text). `label` is meant to be logged
# alongside the eventual actual outcome (converged/diverged/empty/escalated)
# so the prediction can be checked, not just stated — "gate predicted X, did
# the pipeline actually deliver X?" is a real quality signal over time, the
# same instinct as Eval's contour-distribution-stability tracking.
_EXPECTATION: dict[tuple[Contour, str | None], tuple[str, str]] = {
    (Contour.EXACT, None): (
        "likely_answer",
        "I expect to have a solid answer for you.",
    ),
    (Contour.VICINITY, None): (
        "partial_answer",
        "I likely won't nail this precisely, but I'll do my best with what's related "
        "and flag anything I'm unsure about.",
    ),
    (Contour.UNDERSPECIFIED, "explore_siblings"): (
        "likely_answer_after_explore",
        "I expect to work this out myself once I've explored the likely angles — "
        "I'll come back to you only if it's still genuinely unclear.",
    ),
    (Contour.UNDERSPECIFIED, "missing_domain"): (
        "may_need_clarification",
        "I may need a bit more from you before I can answer with confidence.",
    ),
    (Contour.UNDERSPECIFIED, "missing_jurisdiction"): (
        "may_need_clarification",
        "I may need to know who this applies to before I can answer with confidence.",
    ),
    (Contour.CORPUS_GAP, None): (
        "no_answer_expected",
        "I don't expect we have this covered — I'll tell you plainly rather than guess.",
    ),
    (Contour.OUT_OF_SCOPE, None): (
        "cannot_help",
        "I can't help with this one — it's outside what I cover here.",
    ),
    (Contour.UNCLEAR, None): (
        "cannot_proceed",
        "I can't proceed until I understand the question better.",
    ),
}


def expectation(result: GateResult) -> tuple[str, str]:
    """(label, text) — the gate's forward prediction. `label` is the field to
    log and later compare against the pipeline's actual outcome; `text` is
    what a user reads."""
    key = (result.contour, result.underspecified_kind)
    if key in _EXPECTATION:
        return _EXPECTATION[key]
    return _EXPECTATION[(result.contour, None)]


def narrate(result: GateResult) -> str:
    """Render GateResult as the J/D/P path found, the resulting posture, and
    a forward expectation the rest of the pipeline will prove or disprove."""
    path = _found_path(result)
    _, expect_text = expectation(result)

    if result.contour == Contour.EXACT:
        return f"{path}Checked, and I have exact material for this — {result.probe.intersection_docs} document(s) cover it directly. {expect_text}"
    if result.contour == Contour.VICINITY:
        return f"{path}Checked, and I have material in this area ({result.probe.union_docs} documents), but nothing covers this exact combination — I'll need to piece it together from related content. {expect_text}"
    if result.contour == Contour.UNDERSPECIFIED:
        if result.underspecified_kind == "explore_siblings":
            n = len(result.fanout_codes)
            return f"{path}Checked — this domain has {n} more specific facets and nothing narrowed which one you mean, so I'll explore the likely ones myself rather than guess or ask right away. {expect_text}"
        return f"{path}Checked, but {_missing_axis_label(result)} — too broad to answer confidently without it. {expect_text}"
    if result.contour == Contour.CORPUS_GAP:
        return f"{path}Checked, and I know what to look for, but I don't have any documents covering it — a real gap, not something I can answer from what's available. {expect_text}"
    if result.contour == Contour.OUT_OF_SCOPE:
        return f"I understood your question, but it doesn't match anything in scope here. {expect_text}"
    return f"I wasn't able to make sense of that question — could you rephrase it? {expect_text}"


def _missing_axis_label(r: GateResult) -> str:
    if r.underspecified_kind == "missing_jurisdiction":
        return "I don't know who this applies to (which payer or state)"
    return "I don't know the specific topic within this area"


# Closing line: what happens next, in collaborative/first-person terms — never
# the internal archetype name (EXACT/UNDERSPECIFIED/...). That name is for
# Diagnostics; the user reads what WE are going to do about it, together.
def _closing_line(result: GateResult) -> str:
    if result.contour == Contour.EXACT:
        return "So I'll go ahead and answer this directly."
    if result.contour == Contour.VICINITY:
        return "So I'll pull together an answer from the related material I found."
    if result.contour == Contour.UNDERSPECIFIED:
        if result.underspecified_kind == "explore_siblings":
            return "So I'll explore the likely angles myself and see what applies — happy to collaborate if you want to point me at a specific one."
        return "So I may need to check back with you — let's narrow this down together."
    if result.contour == Contour.CORPUS_GAP:
        return "So I'll let you know we don't have this covered, rather than guess."
    if result.contour == Contour.OUT_OF_SCOPE:
        return "So I'll let you know this isn't something covered here."
    return "So I'll ask you to rephrase that for me."


def narrate_full(result: GateResult) -> str:
    """The full step-by-step reasoning trace — every stage of the gate's
    thinking, ending in the resolved archetype (contour). This is the
    "show your work" version; `narrate()` above is the short user-facing
    summary. Same underlying data, different depth.
    """
    steps: list[str] = []

    steps.append(f'You asked: "{result.query}"')

    if result.kinds_matched == 0:
        steps.append(
            "I checked this against our domain/jurisdiction/process lexicon and matched "
            "nothing at all on any axis."
        )
        if result.contour == Contour.UNCLEAR:
            steps.append(
                "The question itself is too short or doesn't contain enough real words for "
                "me to parse — so I can't even tell what topic it's pointing at."
            )
        else:
            steps.append(
                "But the question itself is well-formed and clearly understandable — it's "
                "just not about anything in this corpus's domain."
            )
        return _finish(steps, result)

    d = _axis_phrase(result.d_codes)
    j = _axis_phrase(result.j_codes)
    p = _axis_phrase(result.p_codes)
    found_bits = [f"domain: {d}" if d else "domain: (none)",
                  f"jurisdiction: {j}" if j else "jurisdiction: (none)",
                  f"process: {p}" if p else "process: (none)"]
    steps.append("I expanded this through our lexicon and matched — " + "; ".join(found_bits) + ".")

    steps.append(
        f"I then checked what we actually have on file: {result.probe.union_docs} documents carry at "
        f"least one of these tags, and {result.probe.intersection_docs} documents carry "
        f"the full combination together."
    )

    if result.probe.union_docs == 0:
        steps.append(
            "Zero documents carry any of the matched tags at all — that's not a missing "
            "piece of information, it's a real gap in what we've ingested."
        )
        return _finish(steps, result)

    missing_required = [k for k in ("d", "j") if k in result.missing_kinds]

    if not missing_required:
        if result.underspecified_kind == "explore_siblings":
            n = len(result.fanout_codes)
            steps.append(
                f"Domain and jurisdiction both matched, but the domain match landed only on "
                f"the broad umbrella category — the lexicon actually has {n} more specific "
                f"facets under that same topic that didn't fire, and nothing in the question "
                f"(no explicit process wording, no 'how do I...' phrasing) pointed at which "
                f"one you meant."
            )
            steps.append(
                "Since I know the topic and jurisdiction are right, and I have a known, "
                "bounded list of the specific facets to check, this isn't a dead end — I can "
                "go explore the likely candidates myself instead of asking you to clarify."
            )
        elif result.probe.intersection_docs > 0:
            steps.append(
                "Domain and jurisdiction both matched a specific enough combination, and "
                "documents genuinely cover that exact combination — nothing further needed."
            )
        else:
            steps.append(
                "Domain and jurisdiction both matched, but no single document covers this "
                "exact combination — the topic and the documents about it exist separately, "
                "just not together in one place."
            )
    else:
        missing_label = "the jurisdiction (which payer/state)" if "j" in missing_required else "the domain (what specific topic)"
        anchor = result.probe.intersection_docs if result.kinds_matched > 1 else result.probe.union_docs
        if anchor and anchor <= 25:  # _BROAD_MIN_DOCS, kept in sync manually
            steps.append(
                f"{missing_label.capitalize()} didn't match, but what I do have already "
                f"narrows things down to only {anchor} documents — small enough that the "
                f"corpus itself acts as the missing specifier."
            )
        else:
            steps.append(
                f"{missing_label.capitalize()} didn't match, and what's left is still too "
                f"broad ({result.probe.union_docs} documents) to guess confidently without it."
            )

    return _finish(steps, result)


def _finish(steps: list[str], result: GateResult) -> str:
    _, expect_text = expectation(result)
    steps.append(_closing_line(result))
    steps.append(expect_text)
    return "\n".join(steps)


def _found_path(r: GateResult) -> str:
    """"I found you are asking about D..., J..., and P..." — plain, transparent, no guessing."""
    d = _axis_phrase(r.d_codes)
    j = _axis_phrase(r.j_codes)
    p = _axis_phrase(r.p_codes)

    parts = []
    if d:
        parts.append(f"about **{d}**")
    if j:
        parts.append(f"for **{j}**")
    if p:
        parts.append(f"regarding **{p}**")

    if not parts:
        return ""
    return "I found you are asking " + ", ".join(parts) + ". "
