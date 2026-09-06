"""Ingest stages that run OUTSIDE the API request path.

2026-08-21. `/documents/import-from-gcs` used to do the whole ingest inline:
create the row, download the blob, extract every page, persist tables, call
the classifier, then queue chunking — all before returning 200.

That is why the AHCA push would not drain. `mobius-rag` is pinned
`min=max=1` as a correctness constraint (in-process eval/nightly job state
lives on one instance), so those 20 concurrency slots are the entire
service. Extraction held a slot for the length of a PDF — measured p50
16.6s, p90 38.6s, max 381s — and Cloud Run 429'd everything that arrived
while the slots were parked. Measured over 15 minutes: 309 push requests,
295 of them 429, 6 admitted. 0.4 documents/minute against ~6,000 waiting.

The stages live here so both the API and the chunking worker can call them,
and so the worker can run them off the request path. Import now does row +
job + return; the worker does extract → classify → chunk when it claims the
job and finds no pages.

Nothing here is new logic. It is the same code, moved, so a slot is not held
open for six minutes doing work no caller is waiting on.
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.models import ChunkingEvent, Document, DocumentPage
from app.services.ingest_classifier import classify_for_ingest
from app.services.utils import sanitize_text_for_db

logger = logging.getLogger(__name__)


async def persist_classification(db, doc, clf: dict) -> None:
    """Merge Payor Platform classification into source_metadata, emit an event.

    Moved verbatim from app.main._persist_classification so the worker can
    call it without importing the FastAPI app. main.py re-exports it under
    the old private name; all existing call sites are unchanged.
    """
    existing = doc.source_metadata or {}
    doc.source_metadata = {
        **existing,
        "payor_classification": {
            "decision": clf.get("decision"),
            "may_index": clf.get("may_index"),
            "needs_human": clf.get("needs_human"),
            "why": clf.get("why"),
            "review_url": clf.get("review_url"),
            "contract_version": clf.get("contract_version"),
            "attributed_to": "payor_platform",
            "stages": clf.get("stages"),
        },
    }
    # PRODUCT LINE — PERSISTED VERBATIM, NEVER COMPUTED (Fact Store A-55).
    #
    # Fact Store owns and versions the path->line mapping; it lives in their
    # classifier and comes back in the verdict. We are the scribe. Writing our
    # own value here would give one field two authors, which is the defect
    # A-23 named and the reason my own derive_product_line was deleted.
    #
    # ABSENT MEANS NOT-PATH-SCOPED, AND ABSENT MUST STAY ABSENT. When no path
    # decides, the verdict carries no product_line key and we write nothing.
    # Defaulting to "all_products" here would make silence and a decision look
    # identical afterwards — exactly what mark_product_variant's 409 exists to
    # prevent.
    _pl = clf.get("product_line")
    if _pl:
        doc.source_metadata = {**doc.source_metadata, "product_line": _pl}

    flag_modified(doc, "source_metadata")

    # MAKE THE HOLD VISIBLE.
    #
    # A classifier hold stops the pipeline before chunking, so the document
    # ends up status=completed, unchunked, unpublished — and, until this was
    # added, carrying no failure reason at all. It looked healthy on every
    # dashboard and was invisible to retrieval.
    #
    #   classifier_unavailable — the service was unreachable and the FALLBACK
    #     fired. Not a decision about the document; a decision about our
    #     ability to ask. Retryable.
    #   classifier_held — the classifier answered and said no. A real policy
    #     verdict that a person owns; retrying changes nothing.
    if not clf.get("may_index", True):
        if str(clf.get("contract_version") or "") == "fallback":
            doc.ingest_failure_reason = "classifier_unavailable"
            doc.ingest_error_message = (clf.get("why") or "classifier unreachable")[:1000]
        else:
            doc.ingest_failure_reason = "classifier_held"
            doc.ingest_error_message = (clf.get("why") or "held by classifier")[:1000]
        doc.ingest_last_attempt_at = datetime.utcnow()
        doc.ingest_attempts = (doc.ingest_attempts or 0) + 1
    elif doc.ingest_failure_reason in ("classifier_unavailable", "classifier_held"):
        # It cleared — a later run got an answer. Do not leave the old hold behind.
        doc.ingest_failure_reason = None
        doc.ingest_error_message = None

    db.add(ChunkingEvent(
        document_id=doc.id,
        event_type="payor_classification",
        event_data={
            "decision": clf.get("decision"),
            "may_index": clf.get("may_index"),
            "needs_human": clf.get("needs_human"),
            "why": clf.get("why"),
            "review_url": clf.get("review_url"),
            "attributed_to": "payor_platform",
        },
    ))
    await db.commit()


async def extract_and_persist_pages(db, document: Document) -> int:
    """Download the document's blob, extract pages, persist pages + tables.

    Sets ``document.status`` to ``extracting`` → ``completed``/``failed`` the
    same way the inline import did. Returns the number of pages written (0 on
    failure). Never raises: an extraction failure marks the document failed
    and is reported by the caller, exactly as before.
    """
    gcs_path = document.file_path or ""
    if not gcs_path.startswith("gs://"):
        document.status = "failed"
        document.ingest_failure_reason = "no_gcs_path"
        document.ingest_error_message = f"file_path is not a GCS URI: {gcs_path!r}"
        await db.commit()
        return 0

    from app.services.extract_text import extract_text_from_gcs
    from app.services.page_to_markdown import raw_page_to_markdown

    written = 0
    try:
        document.status = "extracting"
        await db.commit()

        pages = await extract_text_from_gcs(gcs_path)
        for page_data in pages:
            raw_text = sanitize_text_for_db(page_data.get("text") or "") or ""
            md = raw_page_to_markdown(raw_text) if raw_text else None
            db.add(DocumentPage(
                document_id=document.id,
                page_number=page_data["page_number"],
                text=raw_text,
                text_markdown=sanitize_text_for_db(md),
                extraction_status=page_data.get("extraction_status", "failed"),
                extraction_error=page_data.get("extraction_error"),
                text_length=page_data.get("text_length", 0),
            ))
            written += 1
        document.status = "completed"
        await db.commit()

        # Tables AFTER the page commit — document_tables carries a composite FK
        # to (document_id, page_number), so the pages must already exist. Never
        # raises: a table that cannot be stored must not cost the document its
        # ingest.
        if any("tables" in p for p in (pages or [])):
            from app.services.table_persist import persist_document_tables
            _t = await persist_document_tables(db, str(document.id), pages)
            await db.commit()
            if _t["failed"]:
                logger.warning(
                    "document_tables: %s table(s) lost for %s — breadcrumbs in "
                    "page text have no row behind them", _t["failed"], document.id)
    except Exception as exc:
        document.status = "failed"
        document.ingest_failure_reason = document.ingest_failure_reason or "extraction_failed"
        document.ingest_error_message = str(exc)[:1000]
        await db.commit()
        logger.warning("[extract] failed for %s (%s): %s", document.id, gcs_path, exc)
        return 0

    return written


async def classify_and_gate(db, document: Document, *, caller: str) -> bool:
    """Classify the document and persist the verdict. Returns ``may_index``.

    False means the caller must stop before chunking — the document is held
    and ``ingest_failure_reason`` states why.
    """
    meta = document.source_metadata or {}
    clf = await classify_for_ingest(
        document_id=str(document.id),
        source_url=meta.get("source_url"),
        source_page_url=meta.get("source_page_url"),
        caller=caller,
    )
    await persist_classification(db, document, clf)
    return bool(clf.get("may_index", True))


async def has_pages(db, document_id) -> bool:
    r = await db.execute(
        select(DocumentPage.id).where(DocumentPage.document_id == document_id).limit(1)
    )
    return r.scalar_one_or_none() is not None


# ── Product-line attribution ─────────────────────────────────────────────
#
# derive_product_line() lived here from 2026-09-06 and was DELETED the same
# day. Fact Store took the mapping into their classifier (A-55) and now
# returns product_line in the ingest verdict; persist_classification writes it
# verbatim.
#
# Deliberately not kept "just in case". A second copy of a mapping someone
# else versions is the two-authors defect wearing a different hat: it would
# drift silently and there would be no way to tell which value a document
# carried. One author, their pen, our scribe.
