"""Drive document classifier — delegates to the Payor Platform registry service.

Primary path: POST /api/registry/classify on the payor service (deterministic rules).
Fallback: regex patterns when the payor service is unavailable or returns confidence=none.

After ingest, call link_doc_to_registry() to register the document_id in the
payor registry so corpus_present/coverage updates automatically.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from app.services.metadata_canonical import (
    canonical_authority_level,
    canonical_payer,
    canonical_program,
    canonical_state,
)

logger = logging.getLogger(__name__)

PAYOR_SERVICE_URL = "https://mobius-payor-ortabkknqa-uc.a.run.app"

# ── asset_type → authority_level mapping ─────────────────────────────────────

_ASSET_AUTHORITY: dict[str, str] = {
    "state_contract":   "contract_source_of_truth",
    "provider_manual":  "contract_source_of_truth",
    "member_handbook":  "contract_source_of_truth",
    "billing_manual":   "contract_source_of_truth",
    "um_policies":      "contract_source_of_truth",
    "medical_policies": "contract_source_of_truth",
    "formulary":        "contract_source_of_truth",
    "fee_schedule":     "payer_policy",
    "provider_directory": "payer_policy",
    "quick_reference":  "operational_suggested",
    "useful_forms":     "operational_suggested",
    "newsletter":       "fyi_not_citable",
}


async def classify_via_registry(
    filename: str,
    text_snippet: str | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Call payor registry /classify and return enriched classification dict.

    Returns:
      asset_type, sub_type, confidence, authority_level, registry_raw
    """
    payload: dict[str, Any] = {"filename": filename}
    if text_snippet:
        payload["text"] = text_snippet[:2000]   # cap to keep request small
    if source_url:
        payload["source_url"] = source_url

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{PAYOR_SERVICE_URL}/api/registry/classify",
                json=payload,
            )
        if resp.status_code == 200:
            data = resp.json()
            asset_type = data.get("asset_type")
            authority_level = _ASSET_AUTHORITY.get(asset_type or "", "payer_policy")
            return {
                "asset_type":      asset_type,
                "sub_type":        data.get("sub_type"),
                "confidence":      data.get("confidence", "low"),
                "authority_level": canonical_authority_level(authority_level),
                "registry_raw":    data,
            }
        else:
            logger.warning("Payor classify returned %s for %s", resp.status_code, filename)
    except Exception as e:
        logger.warning("Payor classify failed for %s: %s", filename, e)

    # Fallback to local regex
    return _classify_regex_fallback(filename)


async def link_doc_to_registry(
    payor: str,
    document_id: str,
    filename: str,
    asset_type: str,
    sub_type: str | None = None,
    authority_level: str | None = None,
    source_url: str | None = None,
) -> bool:
    """Register a successfully imported document in the payor registry.

    Returns True if the link-doc call succeeded.
    """
    # Canonicalize payor to registry slug form (lowercase, underscore)
    payor_slug = re.sub(r"[^a-z0-9]+", "_", (payor or "").lower()).strip("_")
    if not payor_slug:
        logger.warning("link_doc_to_registry: empty payor slug, skipping")
        return False

    payload: dict[str, Any] = {
        "document_id":     document_id,
        "asset_type":      asset_type,
        "filename":        filename,
    }
    if sub_type:
        payload["sub_type"] = sub_type
    if authority_level:
        payload["authority_level"] = authority_level
    if source_url:
        payload["source_url"] = source_url

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{PAYOR_SERVICE_URL}/api/registry/payors/{payor_slug}/link-doc",
                json=payload,
            )
        if resp.status_code in (200, 201):
            logger.info("link-doc OK: %s → %s (%s)", document_id, payor_slug, asset_type)
            return True
        else:
            logger.warning(
                "link-doc failed %s for %s/%s: %s",
                resp.status_code, payor_slug, document_id, resp.text[:200]
            )
    except Exception as e:
        logger.warning("link-doc error for %s/%s: %s", payor_slug, document_id, e)
    return False


# ── Local regex fallback ──────────────────────────────────────────────────────

_PAYER_TOKENS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bABHFL\b",                    re.I), "Aetna"),
    (re.compile(r"\bAetna\b",                    re.I), "Aetna"),
    (re.compile(r"\bSunshine\b",                 re.I), "Sunshine Health"),
    (re.compile(r"\bSimply[\s_-]?Healthcare\b",  re.I), "Simply Healthcare"),
    (re.compile(r"\bMolina\b",                   re.I), "Molina Healthcare"),
    (re.compile(r"\bWellCare\b",                 re.I), "WellCare"),
    (re.compile(r"\bHumana\b",                   re.I), "Humana"),
    (re.compile(r"\bUnited[\s_-]?Health",        re.I), "United Healthcare"),
    (re.compile(r"\bUHC\b",                      re.I), "United Healthcare"),
    (re.compile(r"\bAHCA\b",                     re.I), "AHCA"),
]

_AUTHORITY_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # (pattern, asset_type, authority_level)
    (re.compile(r"provider[\s_-]?manual",        re.I), "provider_manual",  "contract_source_of_truth"),
    (re.compile(r"billing[\s_-]?manual",         re.I), "billing_manual",   "contract_source_of_truth"),
    (re.compile(r"member[\s_-]?handbook",        re.I), "member_handbook",  "contract_source_of_truth"),
    (re.compile(r"\bformulary\b",                re.I), "formulary",        "contract_source_of_truth"),
    (re.compile(r"\bUM[\s_-]?polic",             re.I), "um_policies",      "contract_source_of_truth"),
    (re.compile(r"utilization[\s_-]?management", re.I), "um_policies",      "contract_source_of_truth"),
    (re.compile(r"auth[\s_-]?polic",             re.I), "um_policies",      "contract_source_of_truth"),
    (re.compile(r"[Pp][Aa][\s_-]?[Ff]orm",      re.I), "useful_forms",     "operational_suggested"),
    (re.compile(r"prior[\s_-]?auth.*form",       re.I), "useful_forms",     "operational_suggested"),
    (re.compile(r"appeal[\s_-]?form",            re.I), "useful_forms",     "operational_suggested"),
    (re.compile(r"\btemplate\b",                 re.I), "useful_forms",     "operational_suggested"),
    (re.compile(r"\btoolkit\b",                  re.I), "quick_reference",  "operational_suggested"),
    (re.compile(r"fee[\s_-]?schedule",           re.I), "fee_schedule",     "payer_policy"),
    (re.compile(r"\bpolic(?:y|ies)\b",           re.I), "um_policies",      "payer_policy"),
]

_STATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bFL\b"),              "FL"),
    (re.compile(r"\bFlorida\b", re.I),  "FL"),
]

_PROGRAM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bMedicaid\b", re.I), "Medicaid"),
    (re.compile(r"\bMMA\b"),            "Medicaid"),
    (re.compile(r"\bLTC\b"),            "Medicaid"),
    (re.compile(r"\bMedicare\b", re.I), "Medicare"),
    (re.compile(r"\bCommercial\b", re.I), "Commercial"),
]


def _classify_regex_fallback(
    filename: str,
    context_payer: str | None = None,
    context_state: str | None = None,
    context_program: str | None = None,
) -> dict[str, Any]:
    readable = re.sub(r"\.(pdf|docx?|xlsx?)$", "", filename, flags=re.I).replace("_", " ").replace("-", " ")

    payer: str | None = context_payer
    for pat, canonical in _PAYER_TOKENS:
        if pat.search(readable):
            payer = canonical
            break

    asset_type: str | None = None
    authority_level: str | None = None
    for pat, at, al in _AUTHORITY_PATTERNS:
        if pat.search(readable):
            asset_type, authority_level = at, al
            break

    state: str | None = context_state
    for pat, st in _STATE_PATTERNS:
        if pat.search(readable):
            state = st
            break

    program: str | None = context_program
    for pat, prog in _PROGRAM_PATTERNS:
        if pat.search(readable):
            program = prog
            break

    confidence = "high" if asset_type else "low"
    return {
        "asset_type":      asset_type or "useful_forms",
        "sub_type":        None,
        "confidence":      confidence,
        "authority_level": canonical_authority_level(authority_level or "payer_policy"),
        "payer":           canonical_payer(payer),
        "state":           canonical_state(state),
        "program":         canonical_program(program),
        "registry_raw":    None,
    }


async def classify_files(
    files: list[dict[str, Any]],
    context_payer: str | None = None,
    context_state: str | None = None,
    context_program: str | None = None,
    use_llm_fallback: bool = True,
) -> list[dict[str, Any]]:
    """Classify a list of Drive file dicts.

    Primary: payor registry /classify (deterministic rules).
    Fallback: local regex when registry unavailable or confidence=none.
    Adds 'classification' key to each file dict.
    """
    results = []
    for f in files:
        name = f.get("name", "")
        cls = await classify_via_registry(name)

        if cls.get("confidence") == "none":
            cls = _classify_regex_fallback(name, context_payer, context_state, context_program)

        # Merge context defaults for payer/state/program (registry doesn't infer these)
        cls.setdefault("payer", canonical_payer(context_payer))
        cls.setdefault("state", canonical_state(context_state))
        cls.setdefault("program", canonical_program(context_program))
        if not cls.get("payer"):
            cls["payer"] = canonical_payer(context_payer)
        if not cls.get("state"):
            cls["state"] = canonical_state(context_state)
        if not cls.get("program"):
            cls["program"] = canonical_program(context_program)

        results.append({**f, "classification": cls})
    return results
