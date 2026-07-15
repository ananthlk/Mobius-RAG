"""Reusable LLM fact-checker / grounding critic.

Scores a QA turn on **honesty-weighted grounding**, given the question, the
retrieved CHUNKS, and the system's ANSWER:

  * **+ credit** for each expected fact the answer states AND that is grounded
    in the chunks (graded 0.0 / 0.5 / 1.0 — partial credit; perfect coverage of
    every qualifier is neither expected nor usually possible).
  * **− penalty** for each incorrect / forbidden fact the answer asserts, or any
    claim it makes that the chunks do NOT support (hallucination).
  * **full credit** when the answer HONESTLY ABSTAINS — declines / says the
    information isn't in the sources — instead of fabricating. In healthcare an
    honest "I can't find this" is a correct outcome, not a miss.

This isolates what we care about — did the system behave honestly given what was
retrievable — from raw answer prose. It intentionally does NOT penalize a strategy
for facts that simply aren't in the corpus (unretrievable); it penalizes making
things up.

Reusable as:
  * **eval calibration** — score each forced strategy's (chunks, answer) per query.
  * **runtime critic** — same call on a live turn to gate/flag ungrounded answers.

LLM stage: ``rag_fact_check`` (registered in RAG_STAGES + chat allowlist).
Eval harness uses the same stage; OBSERVE block in corpus_search_agent calls it
at query time. FACT_CHECKER_VERSION is owned by the EVAL agent — bump it there
on any rubric change so all callers' rows stay comparable.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.services import llm_manager_client

logger = logging.getLogger(__name__)

_FACT_CHECK_STAGE = "rag_fact_check"
FACT_CHECKER_VERSION = "fact_check_v1.2026-07-15"  # EVAL-owned; bump via EVAL agent only
_HALLUCINATION_PENALTY = 1.0  # each hallucinated/forbidden claim cancels one grounded fact


@dataclass
class FactVerdict:
    fact: str
    support: float                  # graded 0.0 (absent) / 0.5 (partial) / 1.0 (fully grounded+stated)
    grounded: bool = False          # stated in the answer AND backed by a chunk
    contradicted: bool = False      # a passage asserts something CONFLICTING (retrieval error)
    passage: int | None = None
    evidence: str | None = None

    @property
    def supported(self) -> bool:
        return self.support >= 0.5


@dataclass
class FactCheckResult:
    verdicts: list[FactVerdict] = field(default_factory=list)
    hallucinated_claims: list[str] = field(default_factory=list)  # answer claims not backed by chunks
    forbidden_present: list[str] = field(default_factory=list)     # forbidden facts asserted
    honest_abstain: bool = False    # answer declined / cited lack of info, asserted nothing unsupported
    score: float = 0.0              # honesty-weighted, in [0,1]
    coverage: float = 0.0           # mean graded support of must_facts (retrieval-only view)
    model: str = ""
    elapsed_ms: int = 0
    reasoning: str = ""

    @property
    def n_supported(self) -> int:
        return sum(1 for v in self.verdicts if v.supported)

    def ledger(self) -> list[dict[str, Any]]:
        """Compact per-claim ledger — EVAL-owned schema, versioned under
        FACT_CHECKER_VERSION. Consumers (OBSERVE row, answer card) persist this
        verbatim. status ∈ validated|unvalidated|contradicted; chunk_id is the
        1-based passage index (resolve against the query's retrieved chunks at
        display time); NO verbatim quote persisted (kept out of the hot row —
        UX pulls the quote from the chunk via chunk_id)."""
        out: list[dict[str, Any]] = []
        for v in self.verdicts:
            status = ("contradicted" if v.contradicted
                      else "validated" if v.support >= 0.5
                      else "unvalidated")
            out.append({"fact": v.fact[:160], "status": status,
                        "chunk_id": v.passage, "support": v.support})
        return out


_SYSTEM = (
    "You are a strict grounding critic for a healthcare-policy assistant. You are "
    "given a QUESTION, the RETRIEVED PASSAGES the system had, the system's ANSWER, "
    "and a list of expected FACTS. Judge ONLY against the passages — never your own "
    "outside knowledge.\n\n"
    "For EACH expected fact, grade how well the ANSWER states it AND whether the "
    "PASSAGES back it:\n"
    "  support = 1.0  answer states the fact and a passage clearly backs it\n"
    "  support = 0.5  answer states the core but misses a qualifier, or a passage "
    "only partially backs it\n"
    "  support = 0.0  the answer does not state it (or no passage backs it)\n"
    "A fact counts as ``grounded`` only when the ANSWER states it AND a PASSAGE "
    "backs it. If the answer states a fact that NO passage supports, that is a "
    "HALLUCINATION, not credit.\n\n"
    "Separately determine:\n"
    "  - hallucinated_claims: claims the ANSWER makes that the passages do NOT "
    "support (including expected facts stated without passage backing).\n"
    "  - forbidden_present: any FORBIDDEN fact the answer asserts.\n"
    "  - honest_abstain: true when the answer explicitly declines or says the "
    "information is not in the provided sources AND asserts no unsupported facts. "
    "An honest abstention is GOOD, not a failure.\n\n"
    "Output ONLY strict JSON, no markdown:\n"
    "{\n"
    '  "results": [{"fact": "<verbatim>", "support": 0.0|0.5|1.0, "grounded": true|false, '
    '"passage": <int|null>, "evidence": "<verbatim quote|null>"}],\n'
    '  "hallucinated_claims": ["<unsupported claim>", ...],\n'
    '  "forbidden_present": ["<forbidden fact asserted>", ...],\n'
    '  "honest_abstain": true|false,\n'
    '  "reasoning": "<one terse sentence>"\n'
    "}\n"
)


_SYSTEM_CHUNK = (
    "You are a retrieval validator for a healthcare-policy corpus. You are given a "
    "QUESTION, numbered RETRIEVED PASSAGES, and a list of FACTS. For EACH fact "
    "decide whether it is present / supported IN THE PASSAGES — judge only the "
    "passage text, never your own outside knowledge. This measures RETRIEVAL: is "
    "the fact in the chunks we pulled, regardless of any answer.\n"
    "For EACH fact decide TWO things:\n"
    "  support — is it present in the passages? (partial credit; perfect coverage "
    "of every qualifier is not expected):\n"
    "    1.0 the passages clearly contain the fact (synonyms/paraphrase fine)\n"
    "    0.5 the passages contain the CORE but miss a qualifier / state it partially\n"
    "    0.0 the passages do not contain the fact at all\n"
    "  contradicted — does ANY passage assert something that CONFLICTS with the "
    "fact (a different number, policy, entity, or answer)? true|false. This flags "
    "a passage that would MISLEAD a reader — a retrieval ERROR — which is distinct "
    "from the fact simply being ABSENT.\n"
    "Cite the passage number and a short verbatim quote for any fact scored > 0 or "
    "marked contradicted.\n\n"
    "Output ONLY strict JSON, no markdown:\n"
    "{\n"
    '  "results": [{"fact": "<verbatim>", "support": 0.0|0.5|1.0, '
    '"contradicted": true|false, "passage": <int|null>, "evidence": "<verbatim quote|null>"}],\n'
    '  "reasoning": "<one terse sentence>"\n'
    "}\n"
)


_SYSTEM_GROUNDING = (
    "You are a strict faithfulness critic for a healthcare-policy assistant. You are "
    "given a QUESTION, the numbered RETRIEVED PASSAGES the system had, and the system's "
    "ANSWER. There is NO answer key — judge the ANSWER only against the PASSAGES, never "
    "your own outside knowledge. Your job: decide whether the answer is GROUNDED.\n\n"
    "Enumerate EACH distinct factual claim the ANSWER asserts (a specific number, date, "
    "code, policy, entity, timeframe, or requirement). For EACH claim decide:\n"
    "  grounded — does a PASSAGE clearly back it? true|false. Paraphrase / synonyms are "
    "fine.\n"
    "  contradicted — does ANY passage assert something that CONFLICTS with the claim (a "
    "different number, policy, entity, date)? true|false. A contradicted claim is the "
    "WORST case — the answer would MISLEAD a reader (e.g. a wrong billing date).\n"
    "  support = 1.0 fully backed · 0.5 core backed but a qualifier is off · 0.0 not "
    "backed.\n"
    "A claim that is neither grounded nor contradicted is a HALLUCINATION — asserted with "
    "no passage support.\n\n"
    "If the ANSWER makes no factual assertions — it declines, or says the information is "
    "not in the provided sources — set honest_abstain=true and return an EMPTY results "
    "list. An honest abstention is GOOD, not a failure. Do NOT invent claims the answer "
    "did not make, and judge only what the answer actually asserts.\n\n"
    "Output ONLY strict JSON, no markdown:\n"
    "{\n"
    '  "results": [{"claim": "<verbatim claim>", "grounded": true|false, '
    '"contradicted": true|false, "support": 0.0|0.5|1.0, "passage": <int|null>, '
    '"evidence": "<verbatim quote|null>"}],\n'
    '  "honest_abstain": true|false,\n'
    '  "reasoning": "<one terse sentence>"\n'
    "}\n"
)


def _build_grounding_prompt(query, answer, chunks) -> str:
    parts = [f"QUESTION:\n{query}", "", "RETRIEVED PASSAGES:"]
    if not chunks:
        parts.append("  (no passages retrieved)")
    for i, c in enumerate(chunks, start=1):
        text = (c.get("text") or "")[:700]
        doc = c.get("document_name") or c.get("document_display_name") or "?"
        parts.append(f"  [{i}] {doc} p.{c.get('page_number')}: {text}")
    parts += ["", "SYSTEM ANSWER (enumerate and check EACH claim it asserts):",
              (answer or "(empty / no answer)")[:1800]]
    return "\n".join(parts)


def _build_chunk_prompt(query, must_facts, chunks) -> str:
    parts = [f"QUESTION:\n{query}", "", "RETRIEVED PASSAGES:"]
    if not chunks:
        parts.append("  (no passages retrieved)")
    for i, c in enumerate(chunks, start=1):
        text = (c.get("text") or "")[:700]
        doc = c.get("document_name") or c.get("document_display_name") or "?"
        parts.append(f"  [{i}] {doc} p.{c.get('page_number')}: {text}")
    parts += ["", "FACTS TO LOCATE IN THE PASSAGES:"]
    for f in must_facts:
        parts.append(f"  - {f}")
    return "\n".join(parts)


def _build_prompt(query, answer, must_facts, forbidden_facts, chunks) -> str:
    parts = [f"QUESTION:\n{query}", "", "RETRIEVED PASSAGES:"]
    if not chunks:
        parts.append("  (no passages retrieved)")
    for i, c in enumerate(chunks, start=1):
        text = (c.get("text") or "")[:700]
        doc = c.get("document_name") or c.get("document_display_name") or "?"
        parts.append(f"  [{i}] {doc} p.{c.get('page_number')}: {text}")
    parts += ["", "SYSTEM ANSWER:", (answer or "(empty / no answer)")[:1800]]
    parts += ["", "EXPECTED FACTS:"]
    for f in must_facts:
        parts.append(f"  - {f}")
    if forbidden_facts:
        parts += ["", "FORBIDDEN FACTS (penalize if asserted):"]
        for f in forbidden_facts:
            parts.append(f"  - {f}")
    return "\n".join(parts)


def _parse(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return {}
    return {}


async def check_facts(
    *,
    query: str,
    must_facts: Sequence[str],
    chunks: Sequence[dict[str, Any]],
    answer: str | None = None,
    forbidden_facts: Sequence[str] | None = None,
    correlation_id: str | None = None,
    max_tokens: int = 4096,
) -> FactCheckResult:
    """Honesty-weighted grounding score for a (query, chunks, answer) turn.

    ``score`` in [0,1]:  +graded support for grounded facts, −penalty per
    hallucinated/forbidden claim, and **1.0 for a clean honest abstention**.
    ``coverage`` is the retrieval-only view (mean graded support), kept for
    comparison.
    """
    must_facts = [f for f in (must_facts or []) if str(f).strip()]
    forbidden_facts = [f for f in (forbidden_facts or []) if str(f).strip()]
    n_must = max(1, len(must_facts))
    has_answer = bool(answer and str(answer).strip())
    if not must_facts and not has_answer:
        return FactCheckResult(reasoning="nothing to check (no must_facts, no answer)")

    # THREE grading modes:
    #  - grounding_only: no gold must_facts but an answer IS present → reference-free
    #    faithfulness critic. Enumerate the ANSWER's own claims and check each vs the
    #    chunks (grounded / contradicted / hallucinated / honest-abstain). This is the
    #    PROD / runtime-critic path — the compliance surface with no answer key, and the
    #    real-world synthesis grade (chat has no gold facts at inference time).
    #  - chunk_only: gold must_facts, no answer → pure retrieval coverage (calibration);
    #    does not penalize a strategy for a synthesizer that failed to use a chunk.
    #  - full: gold must_facts + answer → coverage + grounding (offline synth grade).
    grounding_only = not must_facts and has_answer
    chunk_only = bool(must_facts) and not has_answer

    if grounding_only:
        system, user = _SYSTEM_GROUNDING, _build_grounding_prompt(query, answer, chunks)
    elif chunk_only:
        system, user = _SYSTEM_CHUNK, _build_chunk_prompt(query, must_facts, chunks)
    else:
        system, user = _SYSTEM, _build_prompt(query, answer, must_facts, forbidden_facts, chunks)

    t0 = time.monotonic()
    try:
        raw, meta = await llm_manager_client.generate(
            system=system, user=user,
            stage=_FACT_CHECK_STAGE, max_tokens=max_tokens, correlation_id=correlation_id,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.monotonic() - t0) * 1000)
        logger.warning("fact_checker LLM call failed: %s", exc)
        return FactCheckResult(model="error", elapsed_ms=elapsed, reasoning=f"fact_check_error: {exc}")

    elapsed = int((time.monotonic() - t0) * 1000)
    model = (meta or {}).get("model") or (meta or {}).get("provider") or "unknown"
    parsed = _parse(raw)

    if grounding_only:
        # Answer-driven: each result is a claim the answer asserted, judged vs chunks.
        verdicts = []
        for r in (parsed.get("results") or []):
            if not isinstance(r, dict):
                continue
            claim = str(r.get("claim") or r.get("fact") or "").strip()
            if not claim:
                continue
            grounded = bool(r.get("grounded"))
            try:
                sup = float(r.get("support", 1.0 if grounded else 0.0) or 0.0)
            except (TypeError, ValueError):
                sup = 1.0 if grounded else 0.0
            sup = max(0.0, min(1.0, sup))
            if not grounded:
                sup = 0.0  # ungrounded claim earns no credit (it's a hallucination)
            passage = r.get("passage")
            try:
                passage = int(passage) if passage is not None else None
            except (TypeError, ValueError):
                passage = None
            ev = r.get("evidence")
            verdicts.append(FactVerdict(fact=claim, support=sup, grounded=grounded,
                                        contradicted=bool(r.get("contradicted")),
                                        passage=passage, evidence=str(ev)[:200] if ev else None))
        honest_abstain = bool(parsed.get("honest_abstain"))
        n_claims = max(1, len(verdicts))
        support_sum = sum(v.support for v in verdicts)
        contradicted = [v.fact for v in verdicts if v.contradicted]
        hallucinated = [v.fact for v in verdicts if not v.grounded and not v.contradicted]
        if honest_abstain and not verdicts:
            score = 1.0  # honest abstention = full credit (asserted nothing unsupported)
        else:
            # contradicted = confident-wrong → an EXTRA penalty beyond the 0 support it
            # already earns; a mere unbacked claim is only penalized by dragging the mean.
            penalty = _HALLUCINATION_PENALTY * len(contradicted)
            score = max(0.0, min(1.0, (support_sum - penalty) / n_claims))
        return FactCheckResult(
            verdicts=verdicts, hallucinated_claims=hallucinated,
            honest_abstain=honest_abstain, score=round(score, 3),
            coverage=round(support_sum / n_claims, 3), model=f"factcheck/{model}",
            elapsed_ms=elapsed, reasoning=str(parsed.get("reasoning") or "")[:300],
        )

    by_text = {str(r.get("fact", "")).strip().lower(): r
               for r in (parsed.get("results") or []) if isinstance(r, dict)}
    verdicts: list[FactVerdict] = []
    for i, f in enumerate(must_facts):
        r = by_text.get(f.strip().lower())
        if r is None:
            results = parsed.get("results") or []
            r = results[i] if i < len(results) and isinstance(results[i], dict) else {}
        try:
            sup = float(r.get("support", 0.0) or 0.0)
        except (TypeError, ValueError):
            sup = 0.0
        sup = max(0.0, min(1.0, sup))
        # In chunk-only mode support IS presence (no "grounded" concept).
        grounded = True if chunk_only else bool(r.get("grounded"))
        if not grounded:
            sup = 0.0  # ungrounded "support" is not credit (grounding mode only)
        passage = r.get("passage")
        try:
            passage = int(passage) if passage is not None else None
        except (TypeError, ValueError):
            passage = None
        ev = r.get("evidence")
        verdicts.append(FactVerdict(fact=f, support=sup, grounded=grounded,
                                    contradicted=bool(r.get("contradicted")),
                                    passage=passage, evidence=str(ev)[:200] if ev else None))

    support_sum = sum(v.support for v in verdicts)
    if chunk_only:
        hallucinated, forbidden_present, honest_abstain = [], [], False
        score = round(support_sum / n_must, 3)  # pure retrieval coverage (facts present in chunks)
    else:
        hallucinated = [str(x) for x in (parsed.get("hallucinated_claims") or []) if x]
        forbidden_present = [str(x) for x in (parsed.get("forbidden_present") or []) if x]
        honest_abstain = bool(parsed.get("honest_abstain"))
        penalty = _HALLUCINATION_PENALTY * (len(hallucinated) + len(forbidden_present))
        if honest_abstain and support_sum == 0 and not hallucinated and not forbidden_present:
            score = 1.0  # full credit for honest abstention
        else:
            score = max(0.0, min(1.0, (support_sum - penalty) / n_must))

    return FactCheckResult(
        verdicts=verdicts, hallucinated_claims=hallucinated, forbidden_present=forbidden_present,
        honest_abstain=honest_abstain, score=round(score, 3),
        coverage=round(support_sum / n_must, 3), model=f"factcheck/{model}",
        elapsed_ms=elapsed, reasoning=str(parsed.get("reasoning") or "")[:300],
    )
