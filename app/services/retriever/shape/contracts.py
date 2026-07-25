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

    # Lexicon expansion, split by kind. Codes DO carry the kind prefix
    # (e.g. "d:eligibility.general", "j:payor.sunshine_health") — this
    # docstring previously (wrongly) claimed no prefix; verified 2026-07-23
    # against real output, the UX-signed-off emit spec (gate-emit-schema-spec.md),
    # narrate.py's _leaf_label() (which strips it), and 33+ existing tests, all
    # of which already assume/require the prefix. Downstream consumers should
    # split on ":" if they need the bare code (see gate.py's own _strip_kind()
    # for the pattern, used internally when querying document_tags, which
    # stores keys WITHOUT the prefix).
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


class ReformatPosture(str, Enum):
    """What Reformat decided to do with a GateResult (Step 1b).

    DRAFT — design locked per docs/rag-agents/shape-reformat-schematic-spec.md
    §2, pending cross-agent sign-off (same process as Contour/GateResult).
    """

    PRECISE = "precise"                    # EXACT contour, pass-through
    FAN_OUT = "fan_out"                    # explore_siblings, bounded ranked fan-out
    CLARIFY = "clarify"                    # missing_domain/missing_jurisdiction, suggested questions
    RELY_ON_EXTERNAL = "rely_on_external"  # VICINITY or CORPUS_GAP, defer to Router c/d
    DECLINE = "decline"                    # OUT_OF_SCOPE, no fallback
    CLARIFY_REPHRASE = "clarify_rephrase"  # UNCLEAR passthrough — tentative, not yet confirmed by Ananth


@dataclass
class FanoutCandidate:
    """One scored sibling code considered for a FAN_OUT posture.

    DEPRECATED as the basis for rewritten_queries — superseded 2026-07-23 by
    FanoutTheme (Ananth: raw per-code fan-out, up to 80 codes for a domain like
    eligibility, is too wide; cluster into themes first, cap at 3-4 angles).
    Kept as the pre-clustering scored unit — FanoutTheme.member_codes references
    these — not emitted as rewritten queries directly anymore.

    score = w_p * prevalence_norm + w_l * lexicon_proximity (weights Eval-tuned,
    proposed defaults w_p=0.4/w_l=0.6 per Eval's queries_reformat_postures.yaml §6).
    """

    code: str
    prevalence_docs: int = 0
    prevalence_norm: float = 0.0
    lexicon_proximity: float = 0.0
    score: float = 0.0


# Hard ceiling on question angles surfaced per FAN_OUT — Ananth, 2026-07-23:
# "we cannot have 80 .. no more than 3-4 question angles at any point."
MAX_FANOUT_THEMES = 4


@dataclass
class FanoutTheme:
    """One thematic cluster of fanout_codes — the actual FAN_OUT unit.

    Adopted 2026-07-23 (Ananth's correction): raw fanout_codes (up to ~80 for
    "eligibility", ~631 for "health_care_services") are collapsed into a small
    number of THEMES via vector clustering (embed each sibling's lexicon
    phrases, group by similarity), not fanned out code-by-code. One rewritten
    query per theme, capped at MAX_FANOUT_THEMES — "3-4 question angles", not
    3-4 arbitrary codes. Clustering mechanism (k-means vs. similarity-threshold
    agglomeration, which embedding) is a build-time decision, not fixed here.
    """

    theme_label: str                                  # human-readable, e.g. "income eligibility"
    member_codes: list[str] = field(default_factory=list)   # fanout_codes folded into this theme
    prevalence_docs: int = 0                          # doc count across the theme (sum or representative)
    lexicon_proximity: float = 0.0                    # theme-level match to query tokens
    score: float = 0.0                                # same hybrid intuition, computed per theme now
    is_catchall: bool = False                         # Ananth 2026-07-23: the residual "what else"
    # angle — deliberately NOT corpus/lexicon-derived (member_codes stays empty), routed
    # preferentially to Router strategy c/d (LLM synthesis + Vertex-grounded search) rather
    # than internal Pool search, since its entire purpose is covering what our corpus/lexicon
    # has never seen (e.g. "eligibility for Medicare" when only Medicaid is corpus-tagged).


@dataclass
class ReformatResult:
    """Everything Reformat decided about one GateResult (Step 1b output).

    DRAFT — contract per docs/rag-agents/shape-reformat-schematic-spec.md §4,
    pending cross-agent sign-off. Do not wire into Structure/Pool until signed.
    """

    query: str = ""
    posture: ReformatPosture = ReformatPosture.CLARIFY_REPHRASE

    rewritten_queries: list[str] = field(default_factory=list)   # what Pool actually runs
    fanout_themes: list[FanoutTheme] = field(default_factory=list)  # selected themes, FAN_OUT only, len <= MAX_FANOUT_THEMES

    clarify_questions: list[str] = field(default_factory=list)   # CLARIFY posture only
    external_reason: str | None = None                            # RELY_ON_EXTERNAL posture only
    decline_reason: str | None = None                             # DECLINE posture only

    reason: str = ""            # human-readable trace string, same pattern as GateResult.reason
    reformat_ms: int = 0
    # Per-segment timing, added 2026-07-23 per DB's condition for TECH sign-off
    # (module-gates.md "timing" cross-cut requirement — every DB call/LLM call
    # timed, not just a single total). Keys populated depend on posture: FAN_OUT
    # gets lexicon_fetch_ms/embed_ms/prevalence_ms/clustering_ms; other postures
    # get whatever DB calls they actually made (e.g. clarify_lookup_ms). Empty
    # dict for postures with zero DB/compute cost (DECLINE, CLARIFY_REPHRASE).
    segment_ms: dict = field(default_factory=dict)


@dataclass
class ResourcePosture:
    """How much retrieval effort one query gets — Structure's real new
    output (Step 1c). NOT an answer-format hint — that's legacy
    `answer_shape` on the original request (essay/structured/binary/any),
    Synthesis/Chat's concern, never touched or re-emitted by Structure.

    v1: resolved via a lookup table keyed by (ReformatPosture, caller_mode),
    not a weighted formula — Eval sign-off 2026-07-23, same reasoning
    Router's own linear formula followed (calibration data before trusting
    weights; Structure has none yet). Full record + cross-agent sign-off:
    docs/rag-agents/shape-structure-schematic-spec.md.
    """

    breadth: int            # target chunk count / pool depth
    confidence_bar: float   # convergence threshold, same 0-1 scale as
                             # corpus_search_router.py's existing accuracy_need
    speed_budget: str       # PRIMARY effort constraint (Ananth correction,
                             # 2026-07-23 — see schematic spec addendum §10).
                             # Reused AS-IS from corpus_search_router.py's
                             # existing Literal["real_time","interactive",
                             # "background","none"] — deliberately not a new
                             # ms figure; none exists anywhere in this
                             # codebase to anchor one to (verified live).
    max_attempts: int       # NOT the planned attempt count — Structure has no
                             # visibility into per-strategy latency, so it
                             # cannot know how many tries fit in speed_budget's
                             # time window. That derivation is Router's job:
                             # attempts_planned = time_budget // per_strategy_ms,
                             # then min()'d against this field. This is a soft
                             # SAFETY CEILING only — "never exceed N regardless
                             # of how fast each attempt is" (cost/pathological-
                             # retry protection), subordinate to speed_budget.
                             # Corrected 2026-07-23 after a real seam bug: Router
                             # expected a per-slot attempt count that Structure
                             # was never positioned to know. Posture-only for
                             # v1 — caller_mode-sensitivity is blocked pending
                             # DB's fix for the 3-way caller_mode vocabulary bug.

    # PER-SLOT ceiling on evidence volume a Filler may hand back (Ananth via
    # Retriever, 2026-07-23) — same family as confidence_bar/speed_budget:
    # Structure hands down a target, Fillers enforce it with their own
    # selection logic (e.g. Filler d ranks chunks by its existing BM25
    # scores and keeps the best within budget, not a blind truncate). Real
    # correctness issue, not just cost: strategy d can extract whole-
    # document-sized text with no aggregate cap today (only a PER-PASSAGE
    # cap exists — corpus_search_strategy_d.py's _MAX_PASSAGE_CHARS=2000 ×
    # up to _MAX_FETCH=5 passages, unbounded in total) — a filler could
    # "cover" a query by volume, not relevance. Grounded in a real
    # measurement, not guessed: a full d-slot at capacity=5 measured
    # ~2,021 tokens (Filler d, live). Tentative for auth_agent —
    # accuracy_need=1.00 there tracks PRECISION not volume (answer_shape=
    # binary), open question, not resolved here. Defaulted (unlike the
    # other fields) so pre-existing callers built before this field existed
    # (Pool's/Slots' test fixtures) aren't broken — 10**9 == "unbounded",
    # matching the exact pre-field status quo (no cap existed at all) and
    # mirroring Router's own "none" speed_budget sentinel
    # (_SPEED_BUDGET_MS["none"]=10**9). run_structure() always overrides
    # this with a real per-caller_mode value; the default is a compat
    # shim, not a real answer.
    token_budget: int = 10**9

    # Caller-DECLARED citability requirement (Ananth via Router, 2026-07-23)
    # — NOT computed by Structure, purely threaded through, same "consume
    # don't assemble" pattern already applied to scope/auth and caller_mode.
    # Router already built + tested its consuming side against this exact
    # contract (allocation.py's AUTHORITY_ANY="any" / AUTHORITY_CITABLE_
    # REQUIRED="citable_required", strategy_authority_eligible()): "any"
    # (default, fail-open — zero behavior change until a caller declares)
    # lets non-citable strategies (web search/d) compete normally;
    # "citable_required" makes them ineligible for REQUIRED (evidence-
    # bearing) slots only — optional/external_context slots keep them,
    # since web context is still useful even when not citable to a payor.
    # Whether callers can actually distinguish appeal-vs-casual context to
    # set this meaningfully is still being worked out with Chat — Structure
    # just needs to carry the value faithfully, not resolve that question.
    authority_requirement: str = "any"


@dataclass
class StructureResult:
    """Everything Structure decided about one ReformatResult (Step 1c
    output) — the actual Shape→Pool contract.

    Cross-agent sign-off (UX/Chat/Eval/DB/TECH) complete 2026-07-23 per
    docs/rag-agents/shape-structure-schematic-spec.md. Scope/auth context is
    deliberately NOT a field here — resolved to be orchestrator-owned, not
    Shape-internal (same spec, §1).

    2026-07-23: fanout_themes added for Shape:Slots (Step 1d) — FAN_OUT
    postures only, carries theme scores for accurate slot prioritization.
    """

    query: str = ""
    rewritten_queries: list[str] = field(default_factory=list)   # passthrough from Reformat
    posture: ReformatPosture = ReformatPosture.CLARIFY_REPHRASE  # carried forward from Reformat
    fanout_themes: list[FanoutTheme] = field(default_factory=list)  # FAN_OUT only, len <= MAX_FANOUT_THEMES; for Slots
    resource_posture: ResourcePosture = field(
        default_factory=lambda: ResourcePosture(breadth=0, confidence_bar=0.0, max_attempts=0, speed_budget="none")
    )
    reason: str = ""
    structure_ms: int = 0
