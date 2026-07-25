"""Pure union/dedup logic across Pool's strategies (Step 2, pool-schematic-spec.md S3.5).

No DB, no I/O -- unit-testable in isolation. This is Pool's actual novel
contribution (per target-structure-spec.md S1): legacy strategies never
unioned, so this dedup step never existed until now.

Two-tier dedup, same pattern _expand_with_neighbors() already applies to
siblings (corpus_search.py:3079): dedup by chunk id first, then by content
(content_sha, falling back to a normalized body-text prefix when
content_sha is missing) so multiple ingests of the same document/paragraph
don't show up as distinct candidates.
"""

from __future__ import annotations

from app.services.retriever.pool.contracts import PoolCandidate

_CONTENT_PREFIX_LEN = 200


def _content_key(c: PoolCandidate) -> str:
    sha = (c.content_sha or "").strip()
    if sha:
        return f"sha:{sha}"
    body = " ".join((c.text or "").lower().split())[:_CONTENT_PREFIX_LEN]
    return f"body:{body}" if body else ""


def dedup_candidates(candidates: list[PoolCandidate]) -> list[PoolCandidate]:
    """Union-then-dedup: first candidate wins on both id and content collision.

    Preserves input order, so callers that want one strategy preferred on a
    tie should order their concatenation accordingly (e.g. tag_select
    before vector before inherited -- the caller's choice, not this
    function's).
    """
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    out: list[PoolCandidate] = []
    for c in candidates:
        if c.chunk_id in seen_ids:
            continue
        key = _content_key(c)
        if key and key in seen_content:
            continue
        seen_ids.add(c.chunk_id)
        if key:
            seen_content.add(key)
        out.append(c)
    return out
