"""LLM judge for eval harness.

Two scoring modes — picked by what the bank entry carries:

1. **Rubric mode** (preferred, used when ``expected.golden_answer`` set):
   We hand the LLM the question, the canonical answer, and three lists
   of facts (must / bonus / forbidden). The LLM returns per-fact hit /
   miss decisions; we compute the score deterministically from those
   booleans. The verdict band is fixed:
       correct  if normalized >= 0.85
       partial  if 0.5 <= normalized < 0.85
       wrong    if normalized < 0.5

   This is the honest path — it removes "vibes-based" judgment and lets
   the same response be scored consistently across runs.

2. **Keyword mode** (legacy fallback, used when the entry only has
   ``answer_keywords``): the LLM picks one of the four verdicts based
   on a freeform comparison. Kept for the original 22-question bank
   that doesn't have golden answers yet.

Both modes go through the shared LLM Manager (Thompson-bandit-routed),
so cross-model variance is preserved.

Output of ``adjudicate``: ``(verdict, score, reasoning, model_used, elapsed_ms)``.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.services import llm_manager_client


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode 1: Rubric scoring (golden answer + must/bonus/forbidden facts)
# ---------------------------------------------------------------------------

_RUBRIC_SYSTEM = (
    "You are a strict evaluation judge for a healthcare-policy retrieval "
    "system. The user asked a question. We will give you a CANONICAL "
    "ANSWER plus three lists of facts (must / bonus / forbidden). Your "
    "job is to decide, for each fact, whether the system's response "
    "contains that fact. Answer only with a strict JSON object — no "
    "markdown.\n\n"
    "OUTPUT FORMAT:\n"
    "{\n"
    '  "must_hits":      ["<fact_text>", ...]   // must-facts the response stated\n'
    '  "must_misses":    ["<fact_text>", ...]   // must-facts the response failed to state\n'
    '  "bonus_hits":     ["<fact_text>", ...]   // bonus-facts present\n'
    '  "forbidden_hits": ["<fact_text>", ...]   // forbidden facts present (HALLUCINATIONS)\n'
    '  "honest_abstain": true | false            // see rules below\n'
    '  "reasoning":      "<one sentence summary>"\n'
    "}\n\n"
    "Rules:\n"
    "- Lift the original fact text into the output lists VERBATIM. Do "
    "  not paraphrase.\n"
    "- A fact is a hit if the response asserts it directly OR clearly "
    "  paraphrases it. Synonyms are fine.\n"
    "- A forbidden_fact is a hit if the response asserts that wrong "
    "  claim, even partially.\n"
    "- A must-fact is a miss if the response does not address it at all "
    "  or contradicts it.\n"
    "- If the response is empty / fail-fast, all must-facts are misses "
    "  and there should be no hits.\n"
    "- ``honest_abstain`` = true when the response EXPLICITLY says it "
    "  cannot answer from the provided sources (e.g. \"the passages do "
    "  not contain this information\", \"no information was found\", "
    "  \"I cannot determine from these chunks\") AND has zero "
    "  forbidden_hits. This is qualitatively better than a confidently "
    "  wrong answer — the system declined to hallucinate. Set false "
    "  otherwise (including when the response asserts wrong facts OR "
    "  is simply incomplete without explicitly disclaiming).\n"
    "- Be terse. The reasoning is one sentence."
)


def _build_rubric_prompt(
    query: str,
    expected: dict[str, Any],
    response: dict[str, Any],
) -> str:
    # Show up to 20 chunks (was 5). With neighborhood expansion the
    # answer often lives in a NEIGHBOR of the top seed, not in the seed
    # itself — capping at 5 hid those from the judge entirely. Keep
    # per-chunk preview at 500 chars to bound prompt size; mark
    # neighbors so the judge knows they're supporting context.
    chunks = response.get("chunks") or []
    chunk_lines: list[str] = []
    for i, c in enumerate(chunks[:20], start=1):
        text = (c.get("text") or "")[:500]
        doc = c.get("document_name", "?")
        page = c.get("page_number")
        tag = " (neighbor)" if c.get("is_neighbor") else ""
        chunk_lines.append(f"  [{i}]{tag} {doc} p.{page}: {text}")
    chunks_block = "\n".join(chunk_lines) if chunk_lines else "  (no chunks)"

    llm_answer = response.get("llm_answer") or ""
    fail_fast = response.get("fail_fast")

    parts = [
        f"QUESTION:\n{query}",
        "",
        "CANONICAL ANSWER:",
        (expected.get("golden_answer") or "(none provided)").rstrip(),
        "",
        "FACTS TO CHECK:",
    ]
    for fact in (expected.get("must_facts") or []):
        parts.append(f"  must:      {fact}")
    for fact in (expected.get("bonus_facts") or []):
        parts.append(f"  bonus:     {fact}")
    for fact in (expected.get("forbidden_facts") or []):
        parts.append(f"  forbidden: {fact}")

    parts += [
        "",
        "SYSTEM RESPONSE:",
        f"  strategy_used: {response.get('strategy_used')}",
        f"  confidence: {response.get('confidence')}",
    ]
    if fail_fast:
        parts.append(f"  fail_fast: {fail_fast.get('reason')}")
    if llm_answer:
        parts.append(f"  llm_answer: {llm_answer[:1500]}")
    parts += ["  chunks:", chunks_block]

    return "\n".join(parts)


def _parse_rubric_output(raw: str) -> dict[str, Any]:
    """Parse rubric JSON from the LLM. Returns dict with the four lists."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    def _strlist(v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(x) for x in v if x]

    return {
        "must_hits":      _strlist(parsed.get("must_hits")),
        "must_misses":    _strlist(parsed.get("must_misses")),
        "bonus_hits":     _strlist(parsed.get("bonus_hits")),
        "forbidden_hits": _strlist(parsed.get("forbidden_hits")),
        "honest_abstain": bool(parsed.get("honest_abstain")),
        "reasoning":      str(parsed.get("reasoning") or "").strip()[:500],
    }


def _score_rubric(
    expected: dict[str, Any],
    rubric: dict[str, Any],
) -> tuple[str, float, str]:
    """Compute (verdict, score, reasoning) from per-fact decisions.

    Scoring formula (matches the comment in queries_cmhc.yaml):
        raw = sum(must_hits) - sum(must_misses)
            + 0.5 * sum(bonus_hits)
            - 2.0 * sum(forbidden_hits)
        normalized = max(0, raw / max(1, len(must_facts)))
    Verdict bands:
        correct  >= 0.85
        partial  0.5 .. 0.85
        wrong    < 0.5
    """
    must_facts = expected.get("must_facts") or []
    n_must = max(1, len(must_facts))

    n_must_hit = len(rubric.get("must_hits") or [])
    n_must_miss = len(rubric.get("must_misses") or [])
    n_bonus_hit = len(rubric.get("bonus_hits") or [])
    n_forbidden_hit = len(rubric.get("forbidden_hits") or [])

    raw = (
        n_must_hit
        - n_must_miss
        + 0.5 * n_bonus_hit
        - 2.0 * n_forbidden_hit
    )
    normalized = max(0.0, raw / n_must)
    # Cap at 1.0 — bonus facts and very-good responses can push raw
    # above n_must, but the verdict bands top out at 1.0.
    normalized = min(1.0, normalized)

    # Honest-abstain promotion. If the LLM judge flagged the response
    # as an explicit "I cannot answer from these passages" with zero
    # hallucinations, it gets the ``honest_abstain`` band — between
    # ``partial`` and ``wrong`` in quality. A system that knows what it
    # doesn't know is materially better than one that confidently
    # makes things up.
    #
    # Promotion only applies when:
    #   * judge flagged honest_abstain
    #   * no forbidden_hits (still hallucination-free)
    #   * the rubric score itself wouldn't already be in the partial /
    #     correct bands (don't downgrade a good answer that happened
    #     to also include a disclaimer)
    #
    # Score normalised to max(rubric_score, 0.30) so partial-fact
    # honest abstains still get their fact credit, but the floor
    # ensures the abstain at minimum scores above the wrong band.
    honest_abstain = bool(rubric.get("honest_abstain"))
    if honest_abstain and n_forbidden_hit == 0 and normalized < 0.5:
        verdict = "honest_abstain"
        normalized = max(normalized, 0.30)
    elif normalized >= 0.85:
        verdict = "correct"
    elif normalized >= 0.5:
        verdict = "partial"
    else:
        verdict = "wrong"

    bits = []
    if n_must_hit:     bits.append(f"+{n_must_hit} must")
    if n_must_miss:    bits.append(f"-{n_must_miss} miss")
    if n_bonus_hit:    bits.append(f"+{n_bonus_hit} bonus")
    if n_forbidden_hit: bits.append(f"-{n_forbidden_hit} forbidden")
    reasoning_summary = (
        (rubric.get("reasoning") or "").strip()
        + (f"  [{', '.join(bits)} → score={normalized:.2f}]" if bits else "")
    )
    return verdict, normalized, reasoning_summary[:500]


# ---------------------------------------------------------------------------
# Mode 2: Keyword scoring (legacy, for the original 22-query bank)
# ---------------------------------------------------------------------------

_KEYWORD_SYSTEM = (
    "You are a strict evaluation judge for a healthcare-policy retrieval "
    "system. The user asked a question; the system returned an answer "
    "or chunks. Your job is to decide whether the system's response "
    "answers the question correctly, given the expected ground truth.\n\n"
    "OUTPUT FORMAT — strict JSON, no markdown:\n"
    "{\n"
    '  "verdict": "correct" | "partial" | "wrong" | "unable_to_verify",\n'
    '  "score": <float 0..1>,\n'
    '  "reasoning": "<one sentence why>"\n'
    "}\n\n"
    "Rules for verdict:\n"
    "- 'correct': response addresses the question, key facts match the\n"
    "  expected answer (synonyms / paraphrasing OK), no contradictions.\n"
    "- 'partial': response is on-topic and partially right but misses a\n"
    "  key fact OR includes a wrong fact alongside right ones.\n"
    "- 'wrong': response contradicts expected, or hallucinates, or\n"
    "  answers a different question.\n"
    "- 'unable_to_verify': response is empty / fail-fast / refused, AND\n"
    "  the expected outcome is not also a fail-fast.\n\n"
    "Score is your confidence: 0.0 (definitely wrong) to 1.0 (definitely "
    "correct). Use 0.5 for genuinely ambiguous cases."
)


def _build_keyword_prompt(
    query: str,
    expected: dict[str, Any],
    response: dict[str, Any],
) -> str:
    chunks = response.get("chunks") or []
    chunk_lines = []
    for i, c in enumerate(chunks[:5], start=1):
        text = (c.get("text") or "")[:500]
        doc = c.get("document_name", "?")
        page = c.get("page_number")
        chunk_lines.append(f"  [{i}] {doc} p.{page}: {text}")
    chunks_block = "\n".join(chunk_lines) if chunk_lines else "  (no chunks)"

    llm_answer = response.get("llm_answer") or ""
    routing = response.get("routing") or {}
    fail_fast = response.get("fail_fast")

    parts = [
        f"QUESTION:\n{query}",
        "",
        "EXPECTED:",
        f"  strategy: {expected.get('strategy')}",
    ]
    if expected.get("answer_keywords"):
        parts.append(f"  answer_keywords (any of these is good): {expected.get('answer_keywords')}")
    if expected.get("must_cite_doc"):
        parts.append(f"  must_cite_doc (any of these is good): {expected.get('must_cite_doc')}")
    if expected.get("must_cite_url_contains"):
        parts.append(f"  must_cite_url_contains: {expected.get('must_cite_url_contains')}")
    if expected.get("fail_fast_reason"):
        parts.append(f"  fail_fast_reason: {expected.get('fail_fast_reason')}")
    if expected.get("notes"):
        parts.append(f"  notes: {expected.get('notes')}")

    parts += [
        "",
        "SYSTEM RESPONSE:",
        f"  strategy_used: {response.get('strategy_used')} (router_chose: {routing.get('strategy')})",
        f"  confidence: {response.get('confidence')}",
    ]
    if fail_fast:
        parts.append(f"  fail_fast: {fail_fast.get('reason')}")
    if llm_answer:
        parts.append(f"  llm_answer: {llm_answer[:1000]}")
    parts += ["  chunks:", chunks_block]

    return "\n".join(parts)


def _parse_keyword_output(raw: str) -> tuple[str, float, str]:
    """Parse the keyword judge's verdict JSON. Best-effort fallback."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}
    verdict = (parsed.get("verdict") or "").lower().strip()
    if verdict not in ("correct", "partial", "wrong", "unable_to_verify"):
        verdict = "unable_to_verify"
    score = parsed.get("score")
    try:
        score = float(score) if score is not None else 0.5
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))
    reasoning = (parsed.get("reasoning") or "").strip()[:500]
    return verdict, score, reasoning


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

async def adjudicate(
    query: str,
    expected: dict[str, Any],
    response: dict[str, Any],
    *,
    correlation_id: str | None = None,
) -> tuple[str, float, str, str, int]:
    """Score one (query, expected, response) triple. Picks rubric vs
    keyword mode based on whether ``expected.golden_answer`` is present.

    Returns ``(verdict, score, reasoning, model_used, elapsed_ms)``.
    """
    use_rubric = bool(expected.get("golden_answer"))
    system = _RUBRIC_SYSTEM if use_rubric else _KEYWORD_SYSTEM
    user = (
        _build_rubric_prompt(query, expected, response)
        if use_rubric
        else _build_keyword_prompt(query, expected, response)
    )
    t0 = time.monotonic()
    try:
        raw, llm_meta = await llm_manager_client.generate(
            system=system,
            user=user,
            stage="rag_eval_adjudicate",
            # gemini-2.5-flash is a reasoning model — it burns ~600-800
            # output tokens on internal "thinking" before emitting the
            # visible answer. With max=1024 the rubric JSON gets cut off
            # mid-array (finish_reason=MAX_TOKENS) and the parser falls
            # back to all-empty lists, scoring every response 0.00 with
            # an empty reasoning. 4096 leaves room for the thinking
            # budget plus a complete JSON envelope (~1000 chars). The
            # keyword-mode legacy path uses smaller prompts so 1024 is
            # still fine there.
            max_tokens=4096 if use_rubric else 1024,
            correlation_id=correlation_id,
        )
        elapsed = int((time.monotonic() - t0) * 1000)
        model = (llm_meta or {}).get("model") or (llm_meta or {}).get("provider") or "unknown"

        if use_rubric:
            rubric = _parse_rubric_output(raw)
            verdict, score, reasoning = _score_rubric(expected, rubric)
            # Tag the model name so downstream can see we used rubric.
            return verdict, score, reasoning, f"rubric/{model}", elapsed

        verdict, score, reasoning = _parse_keyword_output(raw)
        return verdict, score, reasoning, model, elapsed
    except Exception as exc:
        elapsed = int((time.monotonic() - t0) * 1000)
        logger.warning("judge LLM call failed: %s", exc)
        return "unable_to_verify", 0.5, f"judge_error: {exc}", "error", elapsed
