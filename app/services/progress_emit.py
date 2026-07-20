"""Fire-and-forget stage-progress emits for the RAG pipeline.

Transport: POST {CHAT_INTERNAL_URL}/internal/progress/{cid}
Body:      {"event": <str>, "seq": <int>, **fields}

No-op when cid is falsy (eval path has no chat cid).
Swallows ALL exceptions — must never raise into the query path.
fields must contain ONLY counts/enums — never query text or chunk text (PHI §4).

Delivery: a single per-process asyncio.Queue + sequential consumer task.
emit_progress does queue.put_nowait() (instant, never blocks the loop);
the consumer awaits each item and POSTs one at a time in enqueue order.
This gives both non-blocking caller AND ordered delivery (no race on DB insert).
Safe at max_instances=1 (in-process queue is the authoritative ordering source).
"""

import asyncio
import json
import logging
import os
import threading
import urllib.request

logger = logging.getLogger(__name__)

_seq_lock = threading.Lock()
_seq: dict[str, int] = {}

_queue: asyncio.Queue | None = None
_consumer_task: asyncio.Task | None = None
_queue_lock = threading.Lock()


def _urlopen_post(url: str, body: bytes) -> None:
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=2)


async def _consumer() -> None:
    """Sequential consumer — posts emits one at a time in enqueue order."""
    global _queue
    while True:
        item = await _queue.get()
        url, body, log_hint = item
        try:
            await asyncio.to_thread(_urlopen_post, url, body)
            logger.debug("emit_progress delivered %s", log_hint)
        except Exception:
            pass
        _queue.task_done()


def _ensure_queue_and_consumer(loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
    global _queue, _consumer_task
    with _queue_lock:
        if _queue is None:
            _queue = asyncio.Queue()
        if _consumer_task is None or _consumer_task.done():
            _consumer_task = loop.create_task(_consumer())
    return _queue


def emit_progress(cid: str | None, event: str, **fields: object) -> None:
    """Enqueue a pipeline stage event for ordered delivery to chat's progress channel.

    No-op when cid is None/empty (eval path, unit tests).
    Never raises — all exceptions are swallowed silently.
    put_nowait returns immediately; the sequential consumer task posts in order.
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
        url = f"{chat_url}/internal/progress/{cid}"
        log_hint = f"cid={cid[:8]} event={event} seq={seq}"
        logger.debug("emit_progress enqueued %s", log_hint)
        try:
            loop = asyncio.get_running_loop()
            queue = _ensure_queue_and_consumer(loop)
            queue.put_nowait((url, body, log_hint))
        except RuntimeError:
            pass  # No running loop (sync test context) — skip.
    except Exception:
        pass
