"""Filename-based metadata classifier for Google Drive imports.

Infers payer, authority_level, state, program from Drive filenames.
Uses regex patterns first; falls back to a cheap LLM call for ambiguous names.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.metadata_canonical import (
    canonical_authority_level,
    canonical_payer,
    canonical_program,
    canonical_state,
)

logger = logging.getLogger(__name__)


# ── Payer tokens in filenames ─────────────────────────────────────────────────

_PAYER_TOKENS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bABHFL\b",                    re.I), "Aetna"),
    (re.compile(r"\bAetna\b",                    re.I), "Aetna"),
    (re.compile(r"\bSunshine\b",                 re.I), "Sunshine Health"),
    (re.compile(r"\bSimply[\s_-]?Healthcare\b",  re.I), "Simply Healthcare"),
    (re.compile(r"\bSimply\b",                   re.I), "Simply Healthcare"),
    (re.compile(r"\bMolina\b",                   re.I), "Molina Healthcare"),
    (re.compile(r"\bWellCare\b",                 re.I), "WellCare"),
    (re.compile(r"\bHumana\b",                   re.I), "Humana"),
    (re.compile(r"\bUnited[\s_-]?Health",        re.I), "United Healthcare"),
    (re.compile(r"\bUHC\b",                      re.I), "United Healthcare"),
    (re.compile(r"\bAHCA\b",                     re.I), "AHCA"),
    (re.compile(r"\bCMS\b",                      re.I), "CMS"),
]

# ── Authority level from filename tokens ──────────────────────────────────────

# Maps to canonical authority_level values. Order matters: more specific first.
_AUTHORITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Binding manuals → contract_source_of_truth (1.0)
    (re.compile(r"provider[\s_-]?manual",        re.I), "contract_source_of_truth"),
    (re.compile(r"billing[\s_-]?manual",         re.I), "contract_source_of_truth"),
    (re.compile(r"billing[\s_-]?guide",          re.I), "contract_source_of_truth"),
    (re.compile(r"member[\s_-]?handbook",        re.I), "contract_source_of_truth"),
    (re.compile(r"member[\s_-]?manual",          re.I), "contract_source_of_truth"),
    (re.compile(r"\bformulary\b",                re.I), "contract_source_of_truth"),
    (re.compile(r"\bUM[\s_-]?polic",             re.I), "contract_source_of_truth"),
    (re.compile(r"utilization[\s_-]?management", re.I), "contract_source_of_truth"),
    (re.compile(r"auth[\s_-]?polic",             re.I), "contract_source_of_truth"),
    (re.compile(r"authorization[\s_-]?polic",    re.I), "contract_source_of_truth"),
    (re.compile(r"clinical[\s_-]?polic",         re.I), "contract_source_of_truth"),
    (re.compile(r"coverage[\s_-]?polic",         re.I), "contract_source_of_truth"),
    (re.compile(r"model[\s_-]?contract",         re.I), "contract_source_of_truth"),
    (re.compile(r"LTC[\s_-]?manual",             re.I), "contract_source_of_truth"),

    # Forms / PA / templates → operational_suggested (0.65)
    (re.compile(r"\b[Pp][Aa][\s_-]?[Ff]orm",    re.I), "operational_suggested"),
    (re.compile(r"prior[\s_-]?auth",             re.I), "operational_suggested"),
    (re.compile(r"referral[\s_-]?form",          re.I), "operational_suggested"),
    (re.compile(r"appeal[\s_-]?form",            re.I), "operational_suggested"),
    (re.compile(r"grievance[\s_-]?form",         re.I), "operational_suggested"),
    (re.compile(r"\btemplate\b",                 re.I), "operational_suggested"),
    (re.compile(r"\bworkflow\b",                 re.I), "operational_suggested"),
    (re.compile(r"\btoolkit\b",                  re.I), "operational_suggested"),
    (re.compile(r"\bguide(?!line)\b",            re.I), "operational_suggested"),
    (re.compile(r"\bSBIRT\b",                    re.I), "operational_suggested"),

    # Fee schedules → payer_policy (0.50)
    (re.compile(r"fee[\s_-]?schedule",           re.I), "payer_policy"),
    (re.compile(r"rate[\s_-]?schedule",          re.I), "payer_policy"),
    (re.compile(r"reimbursement",                re.I), "payer_policy"),

    # Policies (generic) → payer_policy
    (re.compile(r"\bpolic(?:y|ies)\b",           re.I), "payer_policy"),
    (re.compile(r"\bcriteria\b",                 re.I), "payer_policy"),
]

# ── State from filename ───────────────────────────────────────────────────────

_STATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bFL\b"),   "FL"),
    (re.compile(r"\bFlorida\b", re.I), "FL"),
    (re.compile(r"\bTX\b"),   "TX"),
    (re.compile(r"\bCA\b"),   "CA"),
    (re.compile(r"\bNY\b"),   "NY"),
    (re.compile(r"\bGA\b"),   "GA"),
]

# ── Program from filename ─────────────────────────────────────────────────────

_PROGRAM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bMedicaid\b",         re.I), "Medicaid"),
    (re.compile(r"\bMMA\b"),              None),  # resolved to Medicaid below
    (re.compile(r"\bSMMC\b"),             None),
    (re.compile(r"\bLTC\b"),             "Medicaid"),
    (re.compile(r"\bMedicare\b",         re.I), "Medicare"),
    (re.compile(r"\bCommercial\b",       re.I), "Commercial"),
    (re.compile(r"\bMarketplace\b",      re.I), "Marketplace"),
    (re.compile(r"\bD-?SNP\b",           re.I), "D-SNP"),
    (re.compile(r"\bDual\b",             re.I), "Dual"),
]


def classify_filename(
    filename: str,
    context_payer: str | None = None,
    context_state: str | None = None,
    context_program: str | None = None,
) -> dict[str, str | None]:
    """Infer document metadata from a filename.

    Context params (from folder-level metadata) act as defaults when the
    filename doesn't contain enough signal.

    Returns dict with keys: payer, authority_level, state, program, confidence.
    confidence is 'high' (regex matched) or 'low' (default/context only).
    """
    stem = re.sub(r"\.(pdf|docx?|xlsx?|pptx?)$", "", filename, flags=re.I)
    # Normalize separators for easier matching
    readable = stem.replace("_", " ").replace("-", " ")

    # Payer
    payer: str | None = context_payer
    for pat, canonical in _PAYER_TOKENS:
        if pat.search(readable):
            payer = canonical
            break

    # Authority level
    authority_level: str | None = None
    for pat, level in _AUTHORITY_PATTERNS:
        if pat.search(readable):
            authority_level = level
            break

    # State
    state: str | None = context_state
    for pat, st in _STATE_PATTERNS:
        if pat.search(readable):
            state = st
            break

    # Program
    program: str | None = context_program
    for pat, prog in _PROGRAM_PATTERNS:
        if pat.search(readable):
            program = prog or "Medicaid"
            break

    confidence = "high" if authority_level else "low"
    if authority_level is None:
        # Default for unrecognized filenames: payer_policy if payer known, else unknown
        authority_level = "payer_policy" if payer else None

    return {
        "payer":           canonical_payer(payer),
        "authority_level": canonical_authority_level(authority_level),
        "state":           canonical_state(state),
        "program":         canonical_program(program),
        "confidence":      confidence,
    }


async def classify_filename_llm(
    filename: str,
    context_payer: str | None = None,
    context_state: str | None = None,
) -> dict[str, str | None]:
    """LLM fallback for filenames the regex can't classify with confidence.

    Only called when confidence='low'. Returns same shape as classify_filename().
    """
    try:
        from app.services.llm_provider import get_llm_client

        client = get_llm_client()
        prompt = (
            f"Classify this healthcare document filename into metadata fields.\n"
            f"Filename: {filename}\n"
            f"Context payer: {context_payer or 'unknown'}\n"
            f"Context state: {context_state or 'unknown'}\n\n"
            "Return JSON with exactly these keys:\n"
            '{"payer": "<canonical name or null>", '
            '"authority_level": "<one of: contract_source_of_truth|payer_website|operational_suggested|payer_policy|fyi_not_citable|null>", '
            '"state": "<2-letter USPS code or null>", '
            '"program": "<Medicaid|Medicare|Commercial|Marketplace|D-SNP|Dual|null>", '
            '"confidence": "<high|low>"}\n\n'
            "authority_level rules:\n"
            "- contract_source_of_truth: provider manual, member handbook, billing manual, UM policy, auth policy, formulary\n"
            "- operational_suggested: PA forms, referral forms, templates, toolkits, workflows, guides\n"
            "- payer_policy: fee schedules, reimbursement policies, clinical criteria\n"
            "- fyi_not_citable: reports, newsletters, announcements\n"
            "- payer_website: docs sourced from payor's website\n"
            "Respond with JSON only, no explanation."
        )
        resp = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        data = json.loads(raw)
        return {
            "payer":           canonical_payer(data.get("payer")),
            "authority_level": canonical_authority_level(data.get("authority_level")),
            "state":           canonical_state(data.get("state")),
            "program":         canonical_program(data.get("program")),
            "confidence":      data.get("confidence", "low"),
        }
    except Exception as e:
        logger.warning("LLM filename classifier failed for %s: %s", filename, e)
        return classify_filename(filename, context_payer, context_state)


async def classify_files(
    files: list[dict[str, Any]],
    context_payer: str | None = None,
    context_state: str | None = None,
    context_program: str | None = None,
    use_llm_fallback: bool = True,
) -> list[dict[str, Any]]:
    """Classify a list of Drive file dicts in-place.

    Each file dict gets a 'classification' key added with the inferred metadata.
    Files with low confidence are sent through the LLM fallback if enabled.
    """
    results = []
    for f in files:
        name = f.get("name", "")
        cls = classify_filename(name, context_payer, context_state, context_program)

        if cls["confidence"] == "low" and use_llm_fallback:
            cls = await classify_filename_llm(name, context_payer, context_state)

        results.append({**f, "classification": cls})
    return results
