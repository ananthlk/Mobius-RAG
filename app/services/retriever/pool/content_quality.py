"""Content-quality filters -- Pool (Step 2), applied before candidates ever
reach Filler A's ranking.

TOC / index leader-dot filter (2026-07-29, Ananth's catch): a table-of-
contents or index entry like "Chapter 1: Welcome to Sunshine Health
.............................. 9" can score artificially HIGH on multiple
Filler A signals despite carrying zero substantive content:
  - length_sig: leader dots pad raw character count into the "good length"
    bucket even though there's no real information.
  - meta_boost_sig: a required phrase (e.g. the payer name) can occupy a
    large fraction of a short chunk's total words, so density-normalized
    scoring rewards it MORE than a longer, genuinely-correct chunk where
    the same phrase match is diluted by surrounding substantive content.
Confirmed live on cmhc010: exactly this chunk ranked #1 by composite
score, edging out the actual answer (ranked #4, cut by budget) by a
margin of 0.0024.

Corpus-verified scope (2026-07-29): 3,135 chunks corpus-wide contain 4+
consecutive dots; 1,208 of those are short (<600 chars) -- but a real
example caught live (cmhc010's #1-ranked chunk) is a FULL multi-chapter
table of contents at 6,302 characters, not short at all -- an absolute
length cap misses exactly the case it was built for. Fixed by keying off
DENSITY/FREQUENCY of the leader-dot pattern instead of raw chunk length:
a real TOC page has many separate dotted lines (one per entry) regardless
of the chunk's total character count; real prose essentially never
contains even one run of 4+ literal consecutive dots (ellipses are
conventionally 3), let alone several.
"""

from __future__ import annotations

import re

from app.services.retriever.pool.contracts import PoolCandidate

# 4+ consecutive dots -- deliberately above the 3-dot ellipsis convention
# to avoid false-positiving on legitimate "..." usage in real prose.
_LEADER_DOTS_RE = re.compile(r"\.{4,}")

# >=3 separate leader-dot runs: unambiguous multi-entry TOC/index, any length.
_MIN_LEADER_DOT_LINES = 3
# For a single-entry (or short) TOC-style chunk, dot characters make up a
# large fraction of the total text -- catches "Chapter 1: ... .......... 9"
# on its own, where there's only one dot run but it's most of the chunk.
_DOT_DENSITY_THRESHOLD = 0.15


def is_toc_noise(text: str) -> bool:
    """True if this chunk is very likely pure table-of-contents / index
    filler (leader-dot formatting), not substantive content."""
    if not text:
        return False
    matches = _LEADER_DOTS_RE.findall(text)
    if not matches:
        return False
    if len(matches) >= _MIN_LEADER_DOT_LINES:
        return True
    dot_chars = sum(len(m) for m in matches)
    return (dot_chars / len(text)) > _DOT_DENSITY_THRESHOLD


def filter_toc_noise(candidates: list[PoolCandidate]) -> list[PoolCandidate]:
    """Drop TOC/index-noise candidates before they ever reach Filler A's
    ranking -- exclusion, not down-weighting, since this content has zero
    informational value regardless of how any signal might score it."""
    return [c for c in candidates if not is_toc_noise(c.text)]
