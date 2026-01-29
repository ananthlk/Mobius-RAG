"""Critique agent service for reviewing extraction quality."""
import json
import logging
from typing import Dict, List, Any
from app.services.llm_provider import get_llm_provider
from app.services.utils import parse_json_response

logger = logging.getLogger(__name__)

CATEGORY_KEYS = ("who_eligible", "how_verified", "conflict_resolution", "when_applies", "limitations")


def normalize_critique_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure pass, score, category_assessment, feedback, issues, confidence."""
    if "pass" not in raw:
        raw["pass"] = False
    if "feedback" not in raw:
        raw["feedback"] = None
    if "issues" not in raw:
        raw["issues"] = []
    if "confidence" not in raw:
        raw["confidence"] = 0.5
    if "score" not in raw or raw["score"] is None:
        raw["score"] = 0.5
    else:
        try:
            raw["score"] = float(raw["score"])
        except (TypeError, ValueError):
            raw["score"] = 0.5
    raw["score"] = max(0.0, min(1.0, raw["score"]))
    cat = raw.get("category_assessment")
    if not isinstance(cat, dict):
        cat = {}
    out = {}
    for key in CATEGORY_KEYS:
        v = cat.get(key)
        if isinstance(v, dict) and isinstance(v.get("score"), (int, float)):
            s = max(0.0, min(1.0, float(v["score"])))
            out[key] = {"score": s, "note": v.get("note") if isinstance(v.get("note"), str) else None}
    raw["category_assessment"] = out
    return raw


CRITIQUE_PROMPT = """You are a QA agent reviewing fact extraction about **provider operations** in Medicaid/managed care, focusing on:
- **Submitting claims** (requirements, formats, deadlines, authorization)
- **Being a compliant provider** (credentialing, claim requirements, dispute processes)
- **Communicating with members** (marketing, eligibility verification, coordination of benefits)
- **Prior authorization** requirements
- **Claim disputes** (underpayment, overpayment, denials)

We care about facts pertinent to providers submitting claims and working with members. Administrative boilerplate (definitions, contact info, table of contents) that is not pertinent to claims/members need not be extracted as facts.

Review this extraction:

Paragraph: {paragraph_text}

Extracted Summary: {summary}

Extracted Facts:
{facts_list}

**Assess by category:** For each category that the paragraph or extraction touches, give a **score** (0.0–1.0) and optional **note**. Omit categories not relevant.
- **who_eligible:** Are member qualifying criteria extracted correctly? (Only if relevant to provider operations)
- **how_verified:** Are verification methods for member eligibility or claim requirements complete and accurate?
- **conflict_resolution:** Are disputes about claims, eligibility, or benefits covered?
- **when_applies:** Are effective dates, retroactive rules, deadlines, etc. correct?
- **limitations:** Are restrictions on claims, authorizations, or provider operations captured?

**Overall score (0.0–1.0):** Single score for extraction quality. Be generous; reserve low scores for clear failures.
**Table of contents / boilerplate:** If no real content pertinent to claims/members, score **0.8+** when extraction is appropriately minimal. Do **not** demand more facts.

**When to PASS:** overall score >= 0.6. **When to FAIL:** overall score < 0.6 or clear fixable problems.

Return JSON:
{{
  "pass": true/false,
  "score": 0.0-1.0,
  "category_assessment": {{
    "who_eligible": {{ "score": 0.0-1.0, "note": "optional brief note or null" }},
    "how_verified": {{ "score": 0.0-1.0, "note": "optional or null" }},
    "conflict_resolution": {{ "score": 0.0-1.0, "note": "optional or null" }},
    "when_applies": {{ "score": 0.0-1.0, "note": "optional or null" }},
    "limitations": {{ "score": 0.0-1.0, "note": "optional or null" }}
  }},
  "feedback": "Detailed feedback or null if passed",
  "issues": [
    {{
      "type": "missing_fact|hallucination|incorrect_answer|wrong_verification",
      "description": "Description of the issue",
      "suggestion": "How to fix it"
    }}
  ],
  "confidence": 0.0-1.0
}}

Include only categories that apply; others can be omitted. Return only valid JSON, no markdown formatting. No preamble or explanation."""


async def critique_extraction(paragraph_text: str, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Critique an extraction result.
    
    Args:
        paragraph_text: Original paragraph text
        extraction_result: Result from extract_facts()
    
    Returns:
        Dict with 'pass', 'feedback', 'issues', 'confidence'
    """
    llm = get_llm_provider()
    
    # Format facts for prompt
    facts_list = []
    for fact in extraction_result.get("facts", []):
        fact_str = f"- {fact.get('fact_text', '')}"
        if fact.get('is_pertinent_to_claims_or_members') is not None:
            fact_str += f"\n  Pertinent to Claims/Members: {fact.get('is_pertinent_to_claims_or_members')}"
        if fact.get('who_eligible'):
            fact_str += f"\n  WHO: {fact.get('who_eligible')}"
        if fact.get('how_verified'):
            fact_str += f"\n  HOW: {fact.get('how_verified')}"
        if fact.get('conflict_resolution'):
            fact_str += f"\n  WHAT: {fact.get('conflict_resolution')}"
        if fact.get('when_applies'):
            fact_str += f"\n  WHEN: {fact.get('when_applies')}"
        if fact.get('limitations'):
            fact_str += f"\n  LIMITATIONS: {fact.get('limitations')}"
        # Include category scores if present
        if fact.get('category_scores'):
            relevant_cats = [(k, v) for k, v in fact.get('category_scores', {}).items() 
                           if isinstance(v, dict) and (v.get('score', 0) or 0) > 0]
            if relevant_cats:
                fact_str += "\n  Category Scores:"
                for cat, data in relevant_cats[:3]:  # Show top 3
                    score = data.get('score', 0)
                    direction = data.get('direction')
                    dir_label = {1.0: 'Encourages', 0.5: 'Neutral', 0.0: 'Restricts'}.get(direction, 'N/A')
                    fact_str += f"\n    - {cat}: {score:.2f} ({dir_label})"
        facts_list.append(fact_str)
    
    prompt = CRITIQUE_PROMPT.format(
        paragraph_text=paragraph_text,
        summary=extraction_result.get("summary", ""),
        facts_list="\n".join(facts_list) if facts_list else "No facts extracted"
    )
    
    try:
        response = await llm.generate(prompt)
        result = parse_json_response(response)
        return normalize_critique_result(result)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse critique JSON: {e}")
        return {
            "pass": False,
            "score": 0.0,
            "category_assessment": {},
            "feedback": f"Failed to parse critique response: {str(e)}",
            "issues": [{"type": "parse_error", "description": str(e)}],
            "confidence": 0.0
        }
    except Exception as e:
        logger.error(f"Critique failed: {e}", exc_info=True)
        return {
            "pass": False,
            "score": 0.0,
            "category_assessment": {},
            "feedback": f"Critique error: {str(e)}",
            "issues": [{"type": "error", "description": str(e)}],
            "confidence": 0.0
        }


async def stream_critique(paragraph_text: str, extraction_result: Dict[str, Any]):
    """
    Stream critique response.
    
    Yields:
        Chunks of text as they arrive
    
    Returns:
        Final parsed result after stream completes (handled by caller)
    """
    llm = get_llm_provider()
    
    # Format facts for prompt
    facts_list = []
    for fact in extraction_result.get("facts", []):
        fact_str = f"- {fact.get('fact_text', '')}"
        if fact.get('is_pertinent_to_claims_or_members') is not None:
            fact_str += f"\n  Pertinent to Claims/Members: {fact.get('is_pertinent_to_claims_or_members')}"
        if fact.get('who_eligible'):
            fact_str += f"\n  WHO: {fact.get('who_eligible')}"
        if fact.get('how_verified'):
            fact_str += f"\n  HOW: {fact.get('how_verified')}"
        if fact.get('conflict_resolution'):
            fact_str += f"\n  WHAT: {fact.get('conflict_resolution')}"
        if fact.get('when_applies'):
            fact_str += f"\n  WHEN: {fact.get('when_applies')}"
        if fact.get('limitations'):
            fact_str += f"\n  LIMITATIONS: {fact.get('limitations')}"
        # Include category scores if present
        if fact.get('category_scores'):
            relevant_cats = [(k, v) for k, v in fact.get('category_scores', {}).items() 
                           if isinstance(v, dict) and (v.get('score', 0) or 0) > 0]
            if relevant_cats:
                fact_str += "\n  Category Scores:"
                for cat, data in relevant_cats[:3]:  # Show top 3
                    score = data.get('score', 0)
                    direction = data.get('direction')
                    dir_label = {1.0: 'Encourages', 0.5: 'Neutral', 0.0: 'Restricts'}.get(direction, 'N/A')
                    fact_str += f"\n    - {cat}: {score:.2f} ({dir_label})"
        facts_list.append(fact_str)
    
    prompt = CRITIQUE_PROMPT.format(
        paragraph_text=paragraph_text,
        summary=extraction_result.get("summary", ""),
        facts_list="\n".join(facts_list) if facts_list else "No facts extracted"
    )
    
    # Stream response
    try:
        async for chunk in llm.stream_generate(prompt):
            yield chunk
    except Exception as e:
        logger.error(f"Error in stream_critique: {e}", exc_info=True)
        # Yield error message as chunk so caller can handle it
        yield f'\n{{"error": "Stream failed: {str(e)}"}}'
