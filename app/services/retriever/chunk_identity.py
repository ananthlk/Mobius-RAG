"""Shared chunk-identity key logic (Step 5, extracted 2026-07-24 to break a
circular import between synthesis.py and fusion.py -- both need to answer
"is this the same chunk" and must agree on the answer, but synthesis.py
now calls INTO fusion.py's rrf_fuse()/mmr_select() as part of the blend
rewrite, so the identity helper can no longer live inside either of them
without one importing the other's business logic backward.
"""

from __future__ import annotations

from app.services.retriever.fillers.contracts import FilledChunk

_CONTENT_PREFIX_LEN = 200


def content_keys(c: FilledChunk) -> list[str]:
    """All content-identity keys for a chunk, checked INDEPENDENTLY, not
    prioritized (Eval's correctness finding, 2026-07-24, citing this
    fleet's own established result: content_sha is salted per-document in
    this schema, not a pure content hash -- Router's diversity-signal work
    found 5 byte-identical-text chunks with 5 distinct content_sha values,
    and Filler b's tracker documented the same for 15 rows). That means
    content_sha under-fires as a duplicate signal (false negatives) when
    used as the sole/primary key -- checking it only when body-text is
    absent (the original pool/dedup.py-mirrored approach) let a same-fact
    chunk duplicated across two different documents (identical text,
    different per-document salt) survive dedup entirely. Body-text is the
    reliable identity signal here and is always checked; content_sha stays
    a fast-path co-key that can ALSO catch a match, but a chunk is a
    duplicate if EITHER key was already seen -- never gated behind the
    other's absence.
    """
    keys = []
    body = " ".join((c.text or "").lower().split())[:_CONTENT_PREFIX_LEN]
    if body:
        keys.append(f"body:{body}")
    sha = (c.content_sha or "").strip()
    if sha:
        keys.append(f"sha:{sha}")
    return keys
