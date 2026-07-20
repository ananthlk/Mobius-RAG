"""Fire-and-forget stage-progress emits for the RAG pipeline.

Transport: POST {CHAT_INTERNAL_URL}/internal/progress/{cid}
Body:      {"event": <str>, "seq": <int>, **fields}

No-op when cid is falsy (eval path has no chat cid).
Swallows ALL exceptions — must never raise into the query path.
fields must contain ONLY counts/enums — never query text or chunk text (PHI §4).
"""

import json
import logging
import os
import threading
import urllib.request

logger = logging.getLogger(__name__)

# Per-cid monotonic sequence counter (fire-and-forget; reset when cid expires).
_seq_lock = threading.Lock()
_seq: dict[str, int] = {}


def emit_progress(cid: str | None, event: str, **fields: object) -> None:
    """Emit a pipeline stage event to chat's progress channel.

    No-op when cid is None/empty (eval path, unit tests).
    Never raises — all exceptions are swallowed silently.
    """
    if not cid:
        return
    chat_url = (os.environ.get("CHAT_INTERNAL_URL") or "").rstrip("/")
    if not chat_url:
        return
    try:
        with _seq_lock:
            _seq[cid] = _seq.get(cid, 0) + 1
            seq = _seq[cid]
        body = json.dumps({"event": event, "seq": seq, **fields}).encode()
        req = urllib.request.Request(
            f"{chat_url}/internal/progress/{cid}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
        logger.debug("emit_progress cid=%s event=%s seq=%d", cid[:8], event, seq)
    except Exception:
        pass
