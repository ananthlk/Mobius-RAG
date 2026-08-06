"""Friendly narrative generation for ReformatResult — the user-facing
"thinking" translation of Reformat's decision. Mirrors shape/narrate.py's
(Gate's) split exactly, in its own file (Gate's narrate.py is a separate,
already-committed, TECH-signed-off module for GateResult — do not merge or
overwrite it).

DRAFT v1 — built so UX has real output to react to (2026-07-23), not a
from-scratch UX ask. Same split as Gate's narrate.py:
  - narrate()      -> chat-bubble register, default-visible, PHI-safe (never
                      persists an echo of the raw query).
  - narrate_full()  -> Diagnostics-only step-by-step trace. Echoes the raw
                      query in its first line — per Gate's PHI rule (TECH,
                      2026-07-22), NEVER persist this output, compute
                      on-demand only.

Known gap, flagged for UX rather than hidden: FanoutTheme.theme_label is
currently a raw lexicon spec description (e.g. "Policies and status related
to newly enrolled individuals who are non-participants") — fine for
Diagnostics, too clunky for chat-bubble prose as-is. `_soften_theme_label()`
below does a light mechanical cleanup (lowercase, trim trailing punctuation,
cut at first comma/semicolon); it is NOT a polish pass. Real fix is UX's
call — either a proper leaf-code humanization step (mirroring gate/
narrate.py's `_leaf_label`) or accepting the mechanical cleanup as good
enough. Don't treat the current output as "real good" without UX signing
off on it as such.

REVISED 2026-07-23 per Ananth's direct feedback: v1's narrate() was too
thin ("This looks like a precise, well-formed question...") — didn't state
what was actually found the way gate/narrate.py's narrate() does ("I found
you are asking about X, for Y. Checked, and I have exact material..."), and
FAN_OUT folded the angles into one dense run-on sentence instead of
presenting them clearly. `narrate()`/`narrate_full()` now take the source
GateResult as well as the ReformatResult, so they can state the found J/D/P
path (same pattern as Gate's own `_found_path`, reimplemented locally here
rather than importing Gate's private helpers — keeps this module
self-contained). Still a draft for UX to react to, not their final word.
"""

from __future__ import annotations

from .contracts import FanoutTheme, GateResult, ReformatPosture, ReformatResult


def _leaf_label(code: str) -> str:
    """``d:claims.timely_filing`` -> ``timely filing``. Mirrors gate/narrate.py's
    helper of the same name (reimplemented, not imported, to keep this
    module independent of Gate's private API)."""
    stripped = code.split(":", 1)[1] if ":" in code else code
    leaf = stripped.split(".")[-1]
    return leaf.replace("_", " ")


def _axis_phrase(codes: list[str]) -> str | None:
    if not codes:
        return None
    labels = sorted({_leaf_label(c) for c in codes if _leaf_label(c) != "general"} or {_leaf_label(codes[0])})
    return ", ".join(labels)


def _found_path(gate: GateResult) -> str:
    """"about X, for Y" — same shape as Gate's own found-path statement, so
    Reformat's narration reads as a continuation of Gate's, not a
    disconnected restatement."""
    d = _axis_phrase(gate.d_codes)
    j = _axis_phrase(gate.j_codes)
    parts = []
    if d:
        parts.append(f"about **{d}**")
    if j:
        parts.append(f"for **{j}**")
    return ", ".join(parts)


# UX sign-off 2026-07-23: mechanical cleanup (Option A) locked in, not an
# LLM pass — "reuse what you already generated ... don't LLM them; they're
# good." Boilerplate lead-ins strip to match UX's own worked examples
# exactly (e.g. "Policies and criteria related to gross income for
# eligibility" -> "gross income for eligibility").
_BOILERPLATE_PREFIXES = (
    "policies and services related to ",
    "policies and criteria related to ",
    "policies and status related to ",
    "policies and processes related to ",
    "policies related to ",
)


def _soften_theme_label(label: str) -> str:
    """Mechanical cleanup, UX-approved 2026-07-23 — not an LLM pass."""
    text = label.strip().rstrip(".")
    for cut in (",", ";"):
        if cut in text:
            text = text.split(cut, 1)[0]
    if text:
        text = text[0].lower() + text[1:]
    for prefix in _BOILERPLATE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text or "another angle"


def _compose(found: str | None, rest: str) -> str:
    """Glue the found-path clause onto a lowercase-starting continuation, or
    stand the continuation alone (capitalized) when there's no found path to
    state — e.g. missing_domain, where D matched nothing at all. Fixes a
    real bug caught live 2026-07-23: the naive version produced "I see I
    want to make sure I get this right..." (double subject, broken
    grammar) whenever `found` was empty."""
    if found:
        return f"I see you're asking {found} — {rest}"
    return rest[0].upper() + rest[1:] if rest else rest


def narrate(gate: GateResult, result: ReformatResult) -> str:
    """One-paragraph, collaborative, chat-bubble register — states what was
    found (mirroring Gate's own pattern) before what's being done about it.
    Never leaks the internal ReformatPosture enum name."""
    found = _found_path(gate)

    if result.posture == ReformatPosture.PRECISE:
        return _compose(found, "this can be answered directly, so I'm searching for it now.")

    if result.posture == ReformatPosture.FAN_OUT:
        # UX sign-off 2026-07-23: tightened to Gate's pattern (state the
        # angles found, then the action) — not a bulleted list. Matches
        # UX's own worked example almost verbatim.
        named = [t for t in result.fanout_themes if not t.is_catchall]
        labels = [_soften_theme_label(t.theme_label) for t in named]
        if not labels:
            angles = "a few different angles"
        elif len(labels) == 1:
            angles = labels[0]
        else:
            angles = ", ".join(labels[:-1]) + f", and {labels[-1]}"
        return _compose(
            found,
            f"I found several angles on this: {angles}. I'll explore each to find the best fit for you.",
        )

    if result.posture == ReformatPosture.CLARIFY:
        first_q = result.clarify_questions[0] if result.clarify_questions else "Could you clarify what you mean?"
        return _compose(found, f"I want to make sure I get this right — {first_q}")

    if result.posture == ReformatPosture.RELY_ON_EXTERNAL:
        if result.external_reason == "vicinity":
            return _compose(
                found,
                "I have related material, but nothing that covers this exact combination, "
                "so I'll look a bit further to piece together a complete answer.",
            )
        return _compose(
            found,
            "I don't have anything on file for this specific one, so I'll look beyond our "
            "documents to make sure I can still help.",
        )

    if result.posture == ReformatPosture.DECLINE:
        return "This doesn't look like something in scope for what I can help with here."

    # CLARIFY_REPHRASE — tentative, see schematic spec §6.
    return "I wasn't able to make sense of that — could you rephrase it?"


def narrate_full(gate: GateResult, result: ReformatResult, *, redact: bool = False) -> str:
    """Step-by-step trace, Diagnostics-only. NEVER persist — see module
    docstring's PHI note.

    `redact` (2026-08-06, same ruling/pattern as shape/narrate.py's
    narrate_full -- see that function's docstring for the full
    rationale): this module has exactly two raw-query-echo spots
    ("You asked" + the PRECISE-posture "Search query" line, which falls
    back to the raw query when there's no rewrite) -- both gated below.
    Default (redact=False) is unchanged, still the live-only contract.
    """
    steps: list[str] = [] if redact else [f'You asked: "{result.query}"']
    found = _found_path(gate)
    if found:
        steps.append(f"Gate found {found}.")

    if result.posture == ReformatPosture.PRECISE:
        steps.append("The gate found this precise enough to search for directly, without any rewriting.")
        if not redact:
            steps.append(f"Search query: \"{result.rewritten_queries[0] if result.rewritten_queries else result.query}\"")

    elif result.posture == ReformatPosture.FAN_OUT:
        n_candidates_note = "a large set of" if len(result.fanout_themes) else "several"
        steps.append(
            f"The gate found this maps to a broad topic with {n_candidates_note} more specific "
            f"facets, and nothing narrowed which one you meant."
        )
        for t in result.fanout_themes:
            if t.is_catchall:
                steps.append(
                    "One additional angle isn't drawn from our documents at all — it's an open "
                    "check for anything the other angles might have missed."
                )
            else:
                steps.append(
                    f"Angle: \"{t.theme_label}\" — {len(t.member_codes)} related facet(s), "
                    f"{t.prevalence_docs} document(s), relevance score {t.score:.2f}."
                )
        steps.append(f"Reformat took {result.reformat_ms}ms to cluster and rank these.")

    elif result.posture == ReformatPosture.CLARIFY:
        steps.append("The gate found this topic too broad to explore on its own, with nothing to enumerate.")
        for q in result.clarify_questions:
            steps.append(f"Candidate clarifying question: \"{q}\"")

    elif result.posture == ReformatPosture.RELY_ON_EXTERNAL:
        steps.append(f"Reason: {result.external_reason}.")
        steps.append("Deferring to the router's external strategies rather than forcing an internal answer.")

    elif result.posture == ReformatPosture.DECLINE:
        steps.append(f"Reason: {result.decline_reason}.")
        steps.append("No fallback attempted — this is a hard boundary, not a coverage gap.")

    else:
        steps.append("The gate couldn't parse this into anything actionable.")

    steps.append(f"({result.reason})")
    return "\n".join(steps)
