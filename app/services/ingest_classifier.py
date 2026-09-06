"""Payor Platform ingestion classification gate.

Calls mobius-payor /api/registry/ingest/classify before a document is chunked.
The caller gates on ``may_index`` — ``True`` means chunk+embed as normal,
``False`` means store and wait for human review.

Guarantees from the endpoint (per contract v1):
  * Always returns HTTP 200 — never raises for an unclassifiable document.
  * Unknown document_id / unknown payor → 200 + decision=hold.
  * Unreachable → caller must treat as hold and carry on (see _FALLBACK).
"""

import logging
import httpx

logger = logging.getLogger(__name__)

_CLASSIFY_URL = (
    "https://mobius-payor-ortabkknqa-uc.a.run.app/api/registry/ingest/classify"
)
_TIMEOUT = 10.0

_FALLBACK: dict = {
    "decision": "hold",
    "may_index": False,
    "needs_human": True,
    "why": "payor-platform classifier unreachable — held for human review",
    "contract_version": "fallback",
    "attributed_to": "payor_platform",
}


async def classify_for_ingest(
    *,
    document_id: str | None = None,
    payor: str | None = None,
    filename: str | None = None,
    source_url: str | None = None,
    source_page_url: str | None = None,
    text_sample: str | None = None,
    caller: str = "mobius-rag:unknown",
) -> dict:
    """Call the Payor Platform classification endpoint and return the result.

    Pass ``document_id`` when the row already exists (post-commit).
    Pass ``payor``/``filename``/``source_url`` for pre-ingest classification.

    ``source_page_url`` is the page that LINKED the document, and it is sent
    ALONGSIDE document_id rather than instead of it (Fact Store A-55). The
    classifier derives product_line from it: Sunshine serves every PDF from
    /content/dam/centene/..., which carries no product path, so only the
    linking page can decide the line. Fact Store owns that mapping — we send
    the input and persist the verdict; we do not compute it.

    Never raises — returns _FALLBACK (hold) on any network failure.
    """
    payload: dict = {"caller": caller}
    # Sent regardless of which identification path is used below: document_id
    # identifies the row, source_page_url is evidence about it. The original
    # source_url/document_id branch dropped every non-id field, which is how
    # source_url silently stopped reaching the classifier for months.
    if source_page_url:
        payload["source_page_url"] = source_page_url
    if document_id:
        payload["document_id"] = str(document_id)
    else:
        for key, val in (
            ("payor", payor),
            ("filename", filename),
            ("source_url", source_url),
        ):
            if val:
                payload[key] = val
        if text_sample:
            payload["text_sample"] = text_sample[:500]

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(_CLASSIFY_URL, json=payload)
            resp.raise_for_status()
            result = resp.json()
            result["attributed_to"] = "payor_platform"
            return result
    except Exception as exc:
        logger.warning(
            "[ingest_classifier] %s unreachable (%s) — applying hold fallback",
            caller,
            exc,
        )
        return dict(_FALLBACK)
