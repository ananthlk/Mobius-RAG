"""Extraction service for extracting facts from paragraphs."""
import json
import logging
from typing import Dict, List, Any, Optional
from app.services.llm_provider import get_llm_provider
from app.services.utils import parse_json_response

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """You are extracting facts from a Medicaid/managed-care provider manual. Extract ALL relevant facts, not just eligibility-related ones.

**Important:** Do NOT extract section headers, table-of-contents entries, or standalone titles as facts. Each "fact_text" must be a complete sentence or substantive statement (e.g. "Claims must be submitted within 90 days"), not a heading like "Benefits and covered services". If the paragraph is only a header or title with no body, return an empty "facts" array and a brief "summary" only.

For each fact, provide:
1. The standard categories (who_eligible, how_verified, conflict_resolution, when_applies, limitations)
2. Category relevance scores (0.0 to 1.0) and directions for each category below

**Category Relevance Scores (0.0-1.0) and Directions:**
For each category, provide:
- A **score** from 0.0 (not relevant) to 1.0 (highly relevant)
- A **direction** (numeric): 1.0 = encourages (promotes/allows), 0.5 = neutral (just informs/notifies), 0.0 = restricts (limits/prohibits), or null (if not applicable)

Categories:
1. **contacting_marketing_members**: Related to contacting or marketing to members
2. **member_eligibility_molina**: About eligibility of members to Molina policy
3. **benefit_access_limitations**: About access or limitations of benefits to members
4. **prior_authorization_required**: Services that require prior authorization
5. **claims_authorization_submissions**: Requirements for claims, authorization, or other submissions
6. **compliant_claim_requirements**: Requirements to be a compliant claim (e.g., timely filing, format requirements)
7. **claim_disputes**: Claim disputes including underpayment, overpayment, denied claims
8. **credentialing**: Provider credentialing requirements
9. **claim_submission_important**: Other important things needed to submit claims
10. **coordination_of_benefits**: Coordination of benefits information
11. **other_important**: Other important information not covered above

**is_pertinent_to_claims_or_members:** true if this fact is pertinent to submitting claims or working with members. False for provider-only administrative content, definitions, contact info, table of contents, or other content not directly relevant to claims submission or member interactions.

**Examples:** 
- "Claims must be submitted within 90 days" → true (pertinent to submitting claims)
- "Members must be notified of benefit changes" → true (pertinent to working with members)
- "Provider must complete credentialing application" → false (provider administrative task)
- "Contact information for provider services" → false (not pertinent to claims/members)

**Source span (required for each fact):** Identify the exact span of text in the **provided paragraph** that this fact was derived from. Provide:
- **source_start**: 0-based character index of the first character of that span in the paragraph text.
- **source_end**: 0-based character index of the character *after* the last character (exclusive). So paragraph_text.slice(source_start, source_end) is the exact span. If the fact summarizes multiple disjoint phrases, use the most representative single span.

Extract:
1. A brief summary of this paragraph
2. All discrete facts stated
3. For each fact, provide the standard categories AND all category relevance scores with directions AND source_start/source_end

Return JSON:
{{
  "summary": "Brief summary of paragraph",
  "facts": [
    {{
      "fact_text": "Exact statement of the fact",
      "source_start": 0,
      "source_end": 42,
      "who_eligible": "Answer or null",
      "how_verified": "Answer or null",
      "conflict_resolution": "Answer or null",
      "when_applies": "Answer or null",
      "limitations": "Answer or null",
      "is_pertinent_to_claims_or_members": true/false,
      "fact_type": "who_eligible|verification_method|conflict|effective_date|limitation|other",
      "is_verified": true/false,
      "category_scores": {{
        "contacting_marketing_members": {{"score": 0.0, "direction": null}},
        "member_eligibility_molina": {{"score": 0.0, "direction": null}},
        "benefit_access_limitations": {{"score": 0.0, "direction": null}},
        "prior_authorization_required": {{"score": 0.0, "direction": null}},
        "claims_authorization_submissions": {{"score": 0.0, "direction": null}},
        "compliant_claim_requirements": {{"score": 0.0, "direction": null}},
        "claim_disputes": {{"score": 0.0, "direction": null}},
        "credentialing": {{"score": 0.0, "direction": null}},
        "claim_submission_important": {{"score": 0.0, "direction": null}},
        "coordination_of_benefits": {{"score": 0.0, "direction": null}},
        "other_important": {{"score": 0.0, "direction": null}}
      }}
    }}
  ]
}}

The following excerpt is from a provider manual. Section context (if any) is in markdown; extract facts from the paragraph body only.

Excerpt:
{paragraph_block}

Return only valid JSON, no markdown formatting. Do not include any text before or after the JSON. No preamble or explanation."""


RETRY_EXTRACTION_PROMPT = """Previous extraction had issues. Please re-extract with this feedback:

{critique_feedback}

Issues identified:
{issues}

Use the **same extraction format** as before, including:
- is_pertinent_to_claims_or_members (true if pertinent to submitting claims or working with members, false otherwise)
- All category relevance scores (0.0-1.0) and directions (1.0 = encourages, 0.5 = neutral, 0.0 = restricts, or null) for: contacting_marketing_members, member_eligibility_molina, benefit_access_limitations, prior_authorization_required, claims_authorization_submissions, compliant_claim_requirements, claim_disputes, credentialing, claim_submission_important, coordination_of_benefits, other_important.
- source_start and source_end (0-based character indices in the paragraph text for the span this fact was derived from; source_end is exclusive).

The following excerpt is from a provider manual. Section context (if any) is in markdown.

Excerpt:
{paragraph_block}

Return only valid JSON, no markdown formatting. Do not include any text before or after the JSON. No preamble or explanation."""


def _paragraph_block(paragraph_text: str, section_path: str | None = None) -> str:
    """Format paragraph for extraction: markdown section + body so structure is preserved."""
    if section_path and section_path.strip():
        return f"## {section_path.strip()}\n\n{paragraph_text.strip()}"
    return paragraph_text.strip()


async def extract_facts(
    paragraph_text: str,
    critique_feedback: str = None,
    issues: List[str] = None,
    section_path: str | None = None,
    extraction_prompt_body: Optional[str] = None,
    retry_extraction_prompt_body: Optional[str] = None,
    llm=None,
) -> Dict[str, Any]:
    """
    Extract facts from a paragraph using LLM.

    Args:
        paragraph_text: The paragraph text to analyze
        critique_feedback: Optional feedback from critique agent for retry
        issues: Optional list of issues identified
        section_path: Optional section path for context
        extraction_prompt_body: Optional prompt template (overrides in-code default)
        retry_extraction_prompt_body: Optional retry prompt template
        llm: Optional LLM provider instance (uses get_llm_provider() if None)

    Returns:
        Dict with 'summary' and 'facts' keys
    """
    if llm is None:
        llm = get_llm_provider()

    block = _paragraph_block(paragraph_text, section_path)
    if critique_feedback:
        template = retry_extraction_prompt_body if retry_extraction_prompt_body else RETRY_EXTRACTION_PROMPT
        prompt = template.format(
            paragraph_block=block,
            critique_feedback=critique_feedback,
            issues="\n".join(f"- {issue}" for issue in (issues or []))
        )
    else:
        template = extraction_prompt_body if extraction_prompt_body else EXTRACTION_PROMPT
        prompt = template.format(paragraph_block=block)
    try:
        response = await llm.generate(prompt)
        result = parse_json_response(response)
        if "summary" not in result:
            result["summary"] = ""
        if "facts" not in result:
            result["facts"] = []
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extraction JSON: {e}")
        return {
            "summary": "",
            "facts": [],
            "error": f"Failed to parse JSON: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return {
            "summary": "",
            "facts": [],
            "error": str(e)
        }


async def stream_extract_facts(
    paragraph_text: str,
    critique_feedback: str = None,
    issues: List[str] = None,
    section_path: str | None = None,
    extraction_prompt_body: Optional[str] = None,
    retry_extraction_prompt_body: Optional[str] = None,
    llm=None,
):
    """
    Stream extraction and collect full response.

    Yields:
        Chunks of text as they arrive

    Optional: extraction_prompt_body, retry_extraction_prompt_body, llm (provider instance).
    """
    if llm is None:
        llm = get_llm_provider()
    block = _paragraph_block(paragraph_text, section_path)
    if critique_feedback:
        template = retry_extraction_prompt_body if retry_extraction_prompt_body else RETRY_EXTRACTION_PROMPT
        prompt = template.format(
            paragraph_block=block,
            critique_feedback=critique_feedback,
            issues="\n".join(f"- {issue}" for issue in (issues or []))
        )
    else:
        template = extraction_prompt_body if extraction_prompt_body else EXTRACTION_PROMPT
        prompt = template.format(paragraph_block=block)
    
    # Stream response and collect
    try:
        async for chunk in llm.stream_generate(prompt):
            yield chunk
    except Exception as e:
        logger.error(f"Error in stream_extract_facts: {e}", exc_info=True)
        # Yield error message as chunk so caller can handle it
        yield f'\n{{"error": "Stream failed: {str(e)}"}}'
