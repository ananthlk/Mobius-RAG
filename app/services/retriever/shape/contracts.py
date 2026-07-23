"""Dataclass contracts for the shape module (Step 1 of the answer engine).

GateResult is the output of gate.run_gate() — the grounded intent
classification that drives everything downstream. Contours are decided
from J/P/D tag completeness plus a cheap corpus probe (union/intersection
document counts over document_tags), never from vibes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Contour(str, Enum):
    """Response posture decided at the gate."""

    EXACT = "exact"                    # complete slots + docs cover the combination
    VICINITY = "vicinity"              # right area, no doc covers the full combo
    UNDERSPECIFIED = "underspecified"  # missing slot(s) + corpus too broad to guess
    CORPUS_GAP = "corpus_gap"          # tags matched but no documents carry them
    OUT_OF_SCOPE = "out_of_scope"      # well-formed question, zero tags — not our domain
    UNCLEAR = "unclear"                # malformed / can't parse at all


@dataclass
class CorpusProbe:
    """Document-level coverage counts for the matched tag codes.

    All counts come from one SELECT over document_tags (doc-grain,
    ~thousands of rows) — never the 1.94M-row chunk index. This is the
    "range of potential corpus" the answer will be built from.
    """

    d_docs: int = 0            # docs matching ANY matched d-tag
    j_docs: int = 0
    p_docs: int = 0
    union_docs: int = 0        # docs matching ANY matched tag (widest pool)
    intersection_docs: int = 0 # docs matching one tag of EVERY matched kind (tightest pool)
    probe_ms: int = 0


@dataclass
class GateResult:
    """Everything the gate learned about one query."""

    query: str = ""
    normalized: str = ""

    # Lexicon expansion, split by kind (codes carry no kind prefix).
    d_codes: list[str] = field(default_factory=list)
    j_codes: list[str] = field(default_factory=list)
    p_codes: list[str] = field(default_factory=list)
    expansion_phrases: list[str] = field(default_factory=list)

    probe: CorpusProbe = field(default_factory=CorpusProbe)

    # Structural signal, independent of lexicon phrase matching: does the
    # query ask "how do I / how to / what's the process for ..." (an action
    # request) vs a bare fact lookup? Cheap regex, not an LLM call. Used to
    # disambiguate general-only D matches (e.g. "eligibility") without
    # needing every synonym ("check", "confirm", "validate", ...) enumerated
    # as a lexicon p-tag alias.
    process_intent: bool = False

    contour: Contour = Contour.UNCLEAR
    missing_kinds: list[str] = field(default_factory=list)  # e.g. ["j"] → the fan-out axis
    reason: str = ""            # human-readable one-liner for the trace UI
    gate_ms: int = 0

    # Only meaningful when contour == UNDERSPECIFIED. Distinguishes strategies
    # downstream must treat differently:
    #   "explore_siblings" — D matched only a general/umbrella bucket, but the
    #      lexicon has a KNOWN, ENUMERABLE set of specific siblings (fanout_codes).
    #      We know the corpus has an answer, just not which facet. Reformat can
    #      proactively fan out across fanout_codes and explore before ever
    #      asking the user — this is capability, not a user error.
    #   "missing_domain" / "missing_jurisdiction" — D or J matched NOTHING.
    #      There is no sibling set to enumerate; fan-out has nothing to fan out
    #      over. Different downstream handling (relax/escalate/lexicon-gap
    #      flag), not explore.
    underspecified_kind: str | None = None
    fanout_codes: list[str] = field(default_factory=list)  # enumerable siblings, if any

    @property
    def kinds_matched(self) -> int:
        return sum(1 for c in (self.d_codes, self.j_codes, self.p_codes) if c)
