"""Filler c -- LLM Retrieval strategy (Step 3c of the answer engine).

Live external call: asks an LLM for an answer + citations, then locates
and validates each citation against our corpus (title/URL/quote lookup
chains). NOT a PoolResult consumer like Fillers a/b -- same documented
exception c/d/s already have in the parent Fillers spec.

Ported (not imported) from `corpus_search_strategy_c.py`'s
`_ask_llm`/`_locate_by_title`/`_locate_by_url`/`_locate_by_quote`/
`_retrieve_in_doc_by_query`/`_retrieve_at_section_page` per fleet directive
2026-07-23: reference material to re-implement and verify fresh, not a
live dependency on the legacy file.

v1 scope, explicit: the location chain covers title -> URL -> quote
(steps 1-3 of the legacy chain). Google-external validation (step 4,
`_locate_by_google` in the legacy file) is deliberately NOT ported for
v1 -- it pulls in strategy (d)'s web-fetch infra and roughly doubles
worst-case latency for citations already in the minority (no
title/URL/quote hit). Citations that would have gone external instead
classify as `doc_not_found` in v1, same outcome as the legacy code's own
"external search failed" path. Flagged as a known v1 gap, not silently
dropped -- fast-follow once Filler d's fetch infra exists as its own
ported module (avoids re-depending on strategy_d.py directly).

See docs/rag-agents/filler-c-llm-retrieval-kickoff.md for the full design
history (Chat/Eval/DB sign-off on the reshaping below) and
docs/rag-agents/fillers-schematic-spec.md for the parent contract.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import llm_manager_client
from app.services.corpus_search import CorpusSearchRequest, corpus_search
from app.services.retriever.pool.contracts import PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult
from app.services.retriever.fillers.contracts import (
    ASSIGNMENT_REASON_LLM_PARTIAL_MATCH,
    ASSIGNMENT_REASON_LLM_RETRIEVED,
    ASSIGNMENT_REASON_LLM_RETRIEVED_EXTERNAL,
    FilledChunk,
    FilledSlot,
    FilledShape,
)

logger = logging.getLogger(__name__)

_MIN_QUOTE_CHARS = 12
_MIN_BODY_CHARS = 200

_TITLE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "and", "or", "in", "on", "to",
    "with", "from", "by", "at", "is", "are", "this", "that",
})

# Same cross-payer safeguard as legacy strategy_c -- verified needed
# (caught real cross-payer false-matches in the original code's history).
_PAYER_TOKENS = frozenset({
    "aetna", "humana", "cigna", "sunshine", "wellcare", "molina",
    "unitedhealthcare", "uhc", "centene", "anthem", "amerigroup",
    "blue", "bcbs", "kaiser", "magellan", "optum",
})

_SECTION_NUMBER_RE = re.compile(r"^\s*(\d+\.?)+\s*")


# ---------------------------------------------------------------------------
# Data classes (ported from corpus_search_strategy_c.py's shape, this
# module's own copies -- not imported)
# ---------------------------------------------------------------------------


@dataclass
class CitationCandidate:
    """What the LLM emitted, before any validation."""

    document_title: str | None = None
    page: int | None = None
    section: str | None = None
    url: str | None = None
    quote: str | None = None


@dataclass
class ValidatedCitation:
    """A citation after sitemap lookup + content check."""

    candidate: CitationCandidate
    status: str  # retrieved | retrieved_external | doc_found_section_missing
                 # | doc_in_sitemap_not_ingested | doc_robots_blocked | doc_not_found
    document_id: str | None = None
    document_display_name: str | None = None
    document_filename: str | None = None
    matched_chunk_text: str | None = None
    matched_page: int | None = None
    discovered_source_url: str | None = None
    last_fetch_status: int | None = None
    locate_method: str = ""
    notes: str = ""
    # Tri-state, per Eval's ruling 2026-07-23 (synthesis-module-spec.md §9.1):
    # True=LLM gave a quote and it was found in the served text; False=LLM
    # gave a quote but it was NOT found anywhere we looked (today's
    # llm_partial_match downgrade); None=LLM gave no quote at all, so
    # nothing was checked -- collapsing this into status="retrieved" alone
    # destroyed the distinction before FilledChunk ever saw it.
    quote_verified: bool | None = None


@dataclass
class _LocateResult:
    document_id: str | None = None
    display_name: str | None = None
    filename: str | None = None
    discovered_source_url: str | None = None
    last_fetch_status: int | None = None
    sitemap_kind: str = "none"  # doc_ingested | sitemap_robots_blocked | sitemap_needs_scrape | none
    locate_method: str = ""


# ---------------------------------------------------------------------------
# Step 1: LLM generation -- same stage as legacy so bandit history/roster
# carries over untouched (Eval-confirmed 2026-07-23: call as-is, no
# rag-side bandit).
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise policy assistant for FL Medicaid behavioral health. "
    "Answer the user's question briefly (3 sentences max). For every "
    "claim, cite your source.\n\n"
    "OUTPUT FORMAT — strict JSON, no markdown:\n"
    "{\n"
    '  "answer": "<your answer>",\n'
    '  "citations": [\n'
    "    {\n"
    '      "document_title": "<title or filename>",\n'
    '      "page": <integer or null>,\n'
    '      "section": "<section number/name or null>",\n'
    '      "url": "<URL or null>",\n'
    '      "quote": "<verbatim quote, 5-30 words, from the source>"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Every claim in the answer must have a citation.\n"
    "- The quote MUST be a verbatim copy from the source (we will grep "
    "for it). Do not paraphrase.\n"
    "- If you don't know, say so and emit an empty citations array. "
    "Do not invent sources."
)


def _parse_llm_json(raw: str) -> dict[str, Any]:
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
                pass
        return {"answer": "", "citations": [], "_parse_error": True, "_raw": text}


async def _ask_llm(query: str, *, correlation_id: str | None) -> tuple[str, dict[str, Any], dict]:
    """Call the LLM through the shared bandit-routed path. Returns
    (answer_text, parsed_json, llm_telemetry). `llm_telemetry["llm_meta"]`
    carries `model`/`llm_call_id` straight through from mobius-chat's
    ModelRouter -- verified in code 2026-07-23, must not be discarded by
    callers (see fill_shape_llm_retrieval's emit below)."""
    t0 = time.monotonic()
    raw, llm_tel = await llm_manager_client.generate(
        system=_SYSTEM_PROMPT,
        user=query,
        stage="rag_strategy_c_validate",
        max_tokens=2048,
        correlation_id=correlation_id,
    )
    elapsed = (time.monotonic() - t0) * 1000.0
    parsed = _parse_llm_json(raw)
    answer = (parsed.get("answer") or "").strip()
    return answer, parsed, {
        "llm_ms": int(elapsed),
        "llm_meta": llm_tel,
        "parse_error": bool(parsed.get("_parse_error")),
    }


def _coerce_citation(d: dict[str, Any]) -> CitationCandidate | None:
    if not isinstance(d, dict):
        return None
    title = (d.get("document_title") or "").strip() or None
    url = (d.get("url") or "").strip() or None
    quote = (d.get("quote") or "").strip() or None
    section = (d.get("section") or "").strip() or None
    raw_page = d.get("page")
    page = None
    if isinstance(raw_page, int):
        page = raw_page
    elif isinstance(raw_page, str) and raw_page.strip().isdigit():
        page = int(raw_page.strip())
    if not (title or url):
        return None
    return CitationCandidate(document_title=title, page=page, section=section, url=url, quote=quote)


# ---------------------------------------------------------------------------
# Step 2: Locate citation -- title -> URL -> quote (Google-external
# deliberately not ported for v1, see module docstring).
# ---------------------------------------------------------------------------


def _tokenize_title(title: str) -> list[str]:
    if not title:
        return []
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", title)]
    return [t for t in tokens if len(t) >= 3 and t not in _TITLE_STOPWORDS]


def _overlap_coefficient(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


async def _locate_by_title(db: AsyncSession, title: str) -> _LocateResult | None:
    tokens = _tokenize_title(title)
    if not tokens:
        return None

    where_clauses = []
    params: dict[str, str] = {}
    for i, tok in enumerate(tokens):
        p = f"t{i}"
        where_clauses.append(f"LOWER(COALESCE(display_name, '') || ' ' || filename) LIKE :{p}")
        params[p] = f"%{tok}%"
    sql = (
        "SELECT id::text AS id, display_name, filename FROM documents "
        "WHERE expires_at IS NULL AND " + " AND ".join(where_clauses) + " "
        "ORDER BY LENGTH(COALESCE(display_name, filename)) ASC LIMIT 5"
    )
    rows = (await db.execute(sql_text(sql), params)).mappings().all()

    if not rows:
        any_clauses = [
            f"LOWER(COALESCE(display_name, '') || ' ' || filename) LIKE :{p}" for p in params
        ]
        sql_any = (
            "SELECT id::text AS id, display_name, filename FROM documents "
            "WHERE expires_at IS NULL AND (" + " OR ".join(any_clauses) + ") LIMIT 50"
        )
        rows = (await db.execute(sql_text(sql_any), params)).mappings().all()

    if not rows:
        return None

    title_set = set(tokens)
    best: tuple[float, dict, set[str]] | None = None
    for r in rows:
        name = (r["display_name"] or r["filename"] or "").lower()
        name_tokens = {
            t for t in re.findall(r"[a-z0-9]+", name)
            if len(t) >= 3 and t not in _TITLE_STOPWORDS
        }
        score = _overlap_coefficient(title_set, name_tokens)
        if best is None or score > best[0]:
            best = (score, dict(r), name_tokens)

    if best and best[0] >= 0.65 and len(title_set & best[2]) >= 2:
        title_payers = title_set & _PAYER_TOKENS
        match_payers = best[2] & _PAYER_TOKENS
        if title_payers and match_payers and not (title_payers & match_payers):
            logger.info(
                "filler_c._locate_by_title: rejecting cross-payer false match "
                "title_payers=%s match_payers=%s", title_payers, match_payers,
            )
            return None
        method = "title_strict" if best[0] >= 0.90 else "title_relaxed"
        return _LocateResult(
            document_id=best[1]["id"], display_name=best[1]["display_name"],
            filename=best[1]["filename"], sitemap_kind="doc_ingested",
            locate_method=f"{method}(overlap={best[0]:.2f})",
        )
    return None


_ROBOTS_BLOCKED_STATUSES = {401, 403, 451}


async def _locate_by_url(db: AsyncSession, url: str) -> _LocateResult | None:
    rows = await db.execute(
        sql_text(
            "SELECT d.id::text AS id, d.display_name, d.filename FROM documents d "
            "WHERE d.source_metadata ->> 'url' = :u LIMIT 1"
        ),
        {"u": url},
    )
    row = rows.mappings().first()
    if row:
        return _LocateResult(
            document_id=row["id"], display_name=row["display_name"], filename=row["filename"],
            discovered_source_url=url, sitemap_kind="doc_ingested", locate_method="url_exact_doc",
        )

    rows = await db.execute(
        sql_text(
            "SELECT id::text AS id, url, ingested_doc_id::text AS ingested_doc_id, "
            "last_fetch_status FROM discovered_sources WHERE url = :u LIMIT 1"
        ),
        {"u": url},
    )
    row = rows.mappings().first()
    if row:
        ingest_id = row["ingested_doc_id"]
        if ingest_id:
            doc = (await db.execute(
                sql_text("SELECT display_name, filename FROM documents WHERE id::text = :i LIMIT 1"),
                {"i": ingest_id},
            )).mappings().first()
            return _LocateResult(
                document_id=ingest_id, display_name=doc["display_name"] if doc else None,
                filename=doc["filename"] if doc else None, discovered_source_url=row["url"],
                last_fetch_status=row["last_fetch_status"], sitemap_kind="doc_ingested",
                locate_method="url_exact_sitemap",
            )
        status = row["last_fetch_status"]
        kind = "sitemap_robots_blocked" if status in _ROBOTS_BLOCKED_STATUSES else "sitemap_needs_scrape"
        return _LocateResult(
            discovered_source_url=row["url"], last_fetch_status=status,
            sitemap_kind=kind, locate_method="url_exact_sitemap",
        )
    return None


def _quote_present(chunk_text: str | None, quote: str | None) -> bool:
    """Normalized substring check: does `quote` actually appear in `chunk_text`?
    Same probe logic as the location chain's own quote-verification (first
    60-80 chars, normalized whitespace) -- reused here to verify a chunk
    fetched by a DIFFERENT signal (raw user query / section name) actually
    backs the LLM's specific claim, not just that it's "a" chunk from the
    right document. See _run_llm_retrieval for why this matters: a real
    2026-07-23 live-call finding showed by_user_query BM25 within a large
    multi-topic document can return an unrelated but non-empty top-1 match
    (e.g. an EDI/clearinghouse passage) ahead of the section the LLM
    actually cited and quoted correctly (timely filing) -- undetected
    without this check, since "chunk_text is non-empty" was the only prior
    gate.
    """
    if not chunk_text or not quote:
        return False
    body_norm = " ".join(chunk_text.lower().split())
    quote_norm = " ".join(quote.lower().split())
    probe = quote_norm if len(quote_norm) <= 80 else quote_norm[:60]
    return bool(probe) and probe in body_norm


async def _locate_by_quote(db: AsyncSession, quote: str) -> _LocateResult | None:
    if not quote or len(quote) < _MIN_QUOTE_CHARS:
        return None
    rows = await db.execute(
        sql_text(
            "SELECT rpe.document_id::text AS id, "
            "       rpe.text AS chunk_text, "
            "       rpe.document_display_name AS display_name, "
            "       rpe.document_filename AS filename "
            "FROM rag_published_embeddings rpe "
            "WHERE rpe.search_vec @@ phraseto_tsquery('english', :q) LIMIT 5"
        ),
        {"q": quote},
    )
    for row in rows.mappings():
        if _quote_present(row.get("chunk_text"), quote):
            return _LocateResult(
                document_id=row["id"], display_name=row["display_name"], filename=row["filename"],
                sitemap_kind="doc_ingested", locate_method="quote_phrase_verified",
            )
    return None


async def _locate_citation(db: AsyncSession, cand: CitationCandidate) -> _LocateResult:
    """title -> URL -> quote. No Google-external step in v1 (see module docstring)."""
    if cand.document_title:
        r = await _locate_by_title(db, cand.document_title)
        if r is not None:
            return r
    if cand.url:
        r = await _locate_by_url(db, cand.url)
        if r is not None:
            return r
    if cand.quote:
        r = await _locate_by_quote(db, cand.quote)
        if r is not None:
            return r
    return _LocateResult(sitemap_kind="none")


# ---------------------------------------------------------------------------
# Step 3: fetch our own chunk at the located doc (LLM is a retrieval hint,
# not the fact-check target -- we serve our chunk, not the LLM's words).
# ---------------------------------------------------------------------------


def _section_topic(section: str | None) -> str:
    if not section:
        return ""
    return _SECTION_NUMBER_RE.sub("", section).strip()


async def _retrieve_in_doc_by_query(
    db: AsyncSession, document_id: str, user_query: str, section: str | None,
) -> tuple[str | None, int | None, str]:
    sec_topic = _section_topic(section)
    candidates: list[tuple[str, str]] = []
    if user_query:
        candidates.append((user_query, "by_user_query"))
    if sec_topic and len(sec_topic) >= 4:
        candidates.append((sec_topic, "by_section_topic"))

    for q, method in candidates:
        sub_req = CorpusSearchRequest(
            query=q, k=1, mode="precision", tag_mode="none",
            include_document_ids=[document_id], min_similarity=None,
        )
        sub_resp = await corpus_search(db, sub_req, caller=f"filler_c:retrieve_in_doc:{method}")
        if sub_resp.chunks:
            c = sub_resp.chunks[0]
            return (c.text or "")[:600], c.page_number, method
    return None, None, ""


async def _retrieve_at_section_page(
    db: AsyncSession, document_id: str, page: int | None, section: str | None, quote: str | None,
) -> tuple[str | None, int | None, str]:
    if section:
        sec_pat = f"%{section.lower()}%"
        rows = await db.execute(
            sql_text(
                "SELECT page_number, text FROM rag_published_embeddings "
                "WHERE document_id::text = :d AND LOWER(section_path) LIKE :sec "
                "AND LENGTH(text) >= :minc ORDER BY page_number, paragraph_index LIMIT 1"
            ),
            {"d": document_id, "sec": sec_pat, "minc": _MIN_BODY_CHARS},
        )
        row = rows.mappings().first()
        if row:
            return (row["text"] or "")[:600], int(row["page_number"]), "by_section"

    if quote and len(quote) >= _MIN_QUOTE_CHARS:
        rows = await db.execute(
            sql_text(
                "SELECT page_number, text FROM rag_published_embeddings "
                "WHERE document_id::text = :d AND search_vec @@ plainto_tsquery('english', :q) "
                "AND LENGTH(text) >= :minc ORDER BY paragraph_index LIMIT 1"
            ),
            {"d": document_id, "q": quote, "minc": _MIN_BODY_CHARS},
        )
        row = rows.mappings().first()
        if row:
            return (row["text"] or "")[:600], int(row["page_number"]), "by_quote_tokens"

    if page is not None:
        rows = await db.execute(
            sql_text(
                "SELECT page_number, text FROM rag_published_embeddings "
                "WHERE document_id::text = :d AND page_number = :p "
                "AND LENGTH(text) >= :minc ORDER BY paragraph_index LIMIT 1"
            ),
            {"d": document_id, "p": page, "minc": _MIN_BODY_CHARS},
        )
        row = rows.mappings().first()
        if row:
            return (row["text"] or "")[:600], int(row["page_number"]), "by_page"

    rows = await db.execute(
        sql_text(
            "SELECT page_number, text FROM rag_published_embeddings "
            "WHERE document_id::text = :d AND page_number > 1 "
            "AND LENGTH(text) >= :minc ORDER BY page_number, paragraph_index LIMIT 1"
        ),
        {"d": document_id, "minc": _MIN_BODY_CHARS},
    )
    row = rows.mappings().first()
    if row:
        return (row["text"] or "")[:600], int(row["page_number"]), "doc_first_chunk"

    return None, None, ""


# ---------------------------------------------------------------------------
# Main citation pipeline -- ask the LLM, locate + fetch each citation.
# ---------------------------------------------------------------------------


async def _run_llm_retrieval(
    db: AsyncSession, raw_query: str, *, agent_id: str, correlation_id: str | None = None,
) -> tuple[str, list[ValidatedCitation], dict]:
    """Returns (llm_answer, validated_citations, telemetry)."""
    t_start = time.monotonic()

    answer, parsed, llm_telemetry = await _ask_llm(raw_query, correlation_id=correlation_id)
    raw_citations = parsed.get("citations") or []
    candidates: list[CitationCandidate] = []
    for raw_cite in raw_citations:
        c = _coerce_citation(raw_cite)
        if c is not None:
            candidates.append(c)

    logger.info(
        "[%s] [trace:filler_c:llm] answer_len=%d n_citations_raw=%d n_citations_valid=%d "
        "parse_error=%s elapsed=%dms",
        agent_id, len(answer), len(raw_citations), len(candidates),
        llm_telemetry["parse_error"], llm_telemetry["llm_ms"],
    )

    t_validate = time.monotonic()
    validated: list[ValidatedCitation] = []
    for cand in candidates:
        loc = await _locate_citation(db, cand)

        if loc.sitemap_kind == "doc_ingested" and loc.document_id:
            chunk_text, matched_page, retrieval_method = await _retrieve_in_doc_by_query(
                db, loc.document_id, raw_query, cand.section,
            )
            # Verify the chunk we're about to serve actually backs the LLM's
            # specific claim, not just that it's SOME chunk from the right
            # document. Real bug found via a live call 2026-07-23: BM25-by-
            # raw-user-query within a large multi-topic document returned a
            # non-empty but topically wrong top-1 match (an EDI/clearinghouse
            # passage) ahead of the "10.1.1 Timely Filing" section the LLM
            # correctly cited and quoted -- "chunk_text is non-empty" alone
            # is not enough to accept "by_user_query" as ground truth.
            quote_verified = _quote_present(chunk_text, cand.quote)
            if not chunk_text or (cand.quote and not quote_verified):
                alt_text, alt_page, alt_method = await _retrieve_at_section_page(
                    db, loc.document_id, cand.page, cand.section, cand.quote,
                )
                alt_verified = _quote_present(alt_text, cand.quote)
                # Prefer the alternate result only if it's actually verified,
                # or if the first attempt returned nothing at all -- don't
                # discard an unverified-but-present chunk for an equally
                # unverified alternate.
                if alt_verified or not chunk_text:
                    chunk_text, matched_page, retrieval_method = alt_text, alt_page, alt_method
                    quote_verified = alt_verified

            if chunk_text:
                if not cand.quote or quote_verified:
                    # Either the LLM gave no quote to check against (nothing
                    # to verify, same as before this fix), or we confirmed
                    # the returned text actually contains it.
                    if retrieval_method in ("by_section_topic", "by_user_query", "by_section", "by_page"):
                        status, notes = "retrieved", f"retrieved by {retrieval_method}"
                    elif retrieval_method == "by_quote_tokens":
                        status, notes = "retrieved", "retrieved via quote-token match (LLM page may be off)"
                    else:
                        status = "doc_found_section_missing"
                        notes = "doc located but cited page/section not in our copy; returning first chunk"
                else:
                    # Got a non-empty chunk from the right document, but the
                    # LLM's quote isn't actually in it -- downgrade rather
                    # than silently claim "retrieved" for unverified content.
                    status = "doc_found_section_missing"
                    notes = (
                        f"retrieved by {retrieval_method} but LLM's cited quote was not found "
                        "in the returned text -- unverified, not treated as confirmed"
                    )
                validated.append(ValidatedCitation(
                    candidate=cand, status=status, document_id=loc.document_id,
                    document_display_name=loc.display_name, document_filename=loc.filename,
                    matched_chunk_text=chunk_text, matched_page=matched_page,
                    discovered_source_url=loc.discovered_source_url,
                    locate_method=loc.locate_method, notes=notes,
                    # Tri-state: None if the LLM gave no quote to check at all
                    # (quote_verified local is always False in that case, per
                    # _quote_present's own guard -- collapse that back to the
                    # correct "nothing was checked" state here rather than
                    # letting it read as "checked and failed").
                    quote_verified=(quote_verified if cand.quote else None),
                ))
            else:
                validated.append(ValidatedCitation(
                    candidate=cand, status="doc_found_section_missing",
                    document_id=loc.document_id, document_display_name=loc.display_name,
                    document_filename=loc.filename, locate_method=loc.locate_method,
                    notes="doc located but no chunks indexed",
                ))
        elif loc.sitemap_kind == "sitemap_robots_blocked":
            validated.append(ValidatedCitation(
                candidate=cand, status="doc_robots_blocked",
                discovered_source_url=loc.discovered_source_url,
                last_fetch_status=loc.last_fetch_status, locate_method=loc.locate_method,
                notes="sitemap entry exists but robots-blocked; LLM citation passed through unverified",
            ))
        elif loc.sitemap_kind == "sitemap_needs_scrape":
            validated.append(ValidatedCitation(
                candidate=cand, status="doc_in_sitemap_not_ingested",
                discovered_source_url=loc.discovered_source_url,
                last_fetch_status=loc.last_fetch_status, locate_method=loc.locate_method,
                notes="sitemap hit, not ingested -- strategy (d) on-demand fetch",
            ))
        else:
            # No Google-external step in v1 -- see module docstring.
            validated.append(ValidatedCitation(
                candidate=cand, status="doc_not_found",
                notes="no match in corpus or sitemap (external web validation not in v1) "
                      "-- LLM citation passed through unverified",
            ))

    validate_ms = (time.monotonic() - t_validate) * 1000.0
    outcome_counts = {
        s: sum(1 for v in validated if v.status == s)
        for s in (
            "retrieved", "retrieved_external", "doc_found_section_missing",
            "doc_in_sitemap_not_ingested", "doc_robots_blocked", "doc_not_found",
        )
    }
    logger.info(
        "[%s] [trace:filler_c:validate] n=%d outcomes=%s elapsed=%dms",
        agent_id, len(validated), outcome_counts, int(validate_ms),
    )

    telemetry = {
        "llm_ms": llm_telemetry["llm_ms"],
        "validate_ms": int(validate_ms),
        "total_ms": int((time.monotonic() - t_start) * 1000),
        "model_used": (llm_telemetry.get("llm_meta") or {}).get("model"),
        "llm_call_id": (llm_telemetry.get("llm_meta") or {}).get("llm_call_id"),
        "parse_error": llm_telemetry["parse_error"],
        "outcome_counts": outcome_counts,
    }
    return answer, validated, telemetry


# ---------------------------------------------------------------------------
# Reshaping: ValidatedCitation -> FilledChunk (resolved design, see
# docs/rag-agents/filler-c-llm-retrieval-kickoff.md)
# ---------------------------------------------------------------------------

# status -> (source_type, original_score, assignment_reason). Only statuses
# with usable matched_chunk_text produce a FilledChunk at all -- the other
# three (doc_in_sitemap_not_ingested/doc_robots_blocked/doc_not_found) never
# reach this table because they have no matched_chunk_text.
_CHUNK_SHAPE_BY_STATUS = {
    "retrieved": ("llm_hinted_retrieval", 1.0, ASSIGNMENT_REASON_LLM_RETRIEVED),
    "retrieved_external": ("external_validated", 0.9, ASSIGNMENT_REASON_LLM_RETRIEVED_EXTERNAL),
    "doc_found_section_missing": ("llm_hinted_retrieval", 0.5, ASSIGNMENT_REASON_LLM_PARTIAL_MATCH),
}


def _chunk_from_citation(v: ValidatedCitation) -> FilledChunk | None:
    if not v.matched_chunk_text or v.status not in _CHUNK_SHAPE_BY_STATUS:
        return None
    source_type, original_score, assignment_reason = _CHUNK_SHAPE_BY_STATUS[v.status]

    # contracts.py now states the convention explicitly (DB landed this
    # 2026-07-23, caught stale by Retriever's independent re-verification --
    # this file previously used ""/omitted-url from before that landed):
    # internal (document_id populated, url None) vs external (url populated,
    # document_id None) are mutually exclusive, not both-when-available.
    is_external = v.document_id is None

    chunk_id = (
        f"ext-{hash(v.discovered_source_url or v.matched_chunk_text) & 0xffffffff:x}"
        if is_external else
        f"llm-{v.document_id}-{v.matched_page}-{hash(v.matched_chunk_text) & 0xffffffff:x}"
    )

    return FilledChunk(
        chunk_id=chunk_id,
        document_id=None if is_external else v.document_id,
        text=v.matched_chunk_text,
        url=v.discovered_source_url if is_external else None,
        document_status=None,  # not looked up in v1 -- open question with DB, unresolved as of this writing
        content_sha=None,      # open question with DB, unresolved as of this writing
        source_type=source_type,
        page_number=v.matched_page,
        tags={"locate_method": v.locate_method, "notes": v.notes},
        is_neighbor=False,
        original_score=original_score,
        assignment_reason=assignment_reason,
        # Was silently dropped here -- v already carries the correct tri-state
        # (computed in _run_llm_retrieval), but this constructor never passed
        # it through, so FilledChunk.quote_verified stayed at its None default
        # for every citation regardless of actual verification. Caught by
        # Eval tracing the full producer->consumer chain 2026-07-23 --
        # Synthesis had already started consuming this field, so the gap was
        # live-flipping every filler-c citation to verified=False, including
        # genuinely quote-confirmed ones.
        quote_verified=v.quote_verified,
    )


# ---------------------------------------------------------------------------
# Filler entry point -- same async live-call signature pattern as filler_s
# (pool_result accepted for signature consistency, unused: Filler c doesn't
# rank Pool candidates, it generates its own).
# ---------------------------------------------------------------------------


async def fill_shape_llm_retrieval(
    pool_result: PoolResult | None,
    shape_result: AnswerShapeResult,
    raw_query: str,
    *,
    db: AsyncSession,
    agent_id: str = "filler_c",
    correlation_id: str | None = None,
    tag_matches: list[str] | None = None,
) -> FilledShape:
    """Ask the LLM for facts+citations per slot, validate against the
    corpus, reshape into FilledChunk. `pool_result`/`tag_matches` accepted
    for signature consistency with a/b/s; unused in v1 (same "unused v1"
    convention as those fillers' own unused params).

    One real LLM call per slot, using that slot's own `rewritten_query`
    when set (FAN_OUT themes), falling back to `raw_query` otherwise --
    asking the top-level query for a thematic sub-slot would defeat the
    point of per-theme filling.
    """
    filled_slots: list[FilledSlot] = []
    total_assigned = 0
    per_slot_emit: list[dict] = []

    for slot in shape_result.slots:
        filled_slot = FilledSlot(
            slot_id=slot.slot_id, slot_semantics=slot.slot_semantics,
            capacity=slot.capacity, required=slot.required,
        )

        query_for_slot = slot.rewritten_query or raw_query
        slot_telemetry: dict = {}
        try:
            llm_answer, citations, slot_telemetry = await _run_llm_retrieval(
                db, query_for_slot, agent_id=agent_id, correlation_id=correlation_id,
            )
            chunks = [c for c in (_chunk_from_citation(v) for v in citations) if c is not None]
            filled_slot.chunks = chunks[: slot.capacity]
        except Exception as exc:
            logger.warning(
                "[filler_c] slot=%s query_len=%d error=%r", slot.slot_id, len(query_for_slot), exc,
            )
            llm_answer = ""

        filled_slot.occupancy = len(filled_slot.chunks)
        filled_slot.under_filled = filled_slot.occupancy < slot.capacity
        filled_slot.over_filled = False
        filled_slots.append(filled_slot)
        total_assigned += filled_slot.occupancy

        per_slot_emit.append({
            "slot_id": slot.slot_id,
            "llm_answer": llm_answer,  # diagnostic-only -- Chat-verified 2026-07-23, never consumed downstream
            **slot_telemetry,
        })

    emit = {
        "fillers_decision": "llm_retrieval",
        "slots_filled": len([s for s in filled_slots if s.occupancy > 0]),
        "empty_slots": len([s for s in filled_slots if s.occupancy == 0]),
        "under_filled": len([s for s in filled_slots if s.under_filled]),
        "total_chunks_assigned": total_assigned,
        "per_slot_details": per_slot_emit,
    }

    return FilledShape(
        slots=filled_slots,
        total_chunks_assigned=total_assigned,
        filling_strategy="llm_retrieval",
        emit=emit,
    )
