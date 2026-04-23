"""Structured JSON logging for mobius-rag.

Writes one JSON object per log record to **stderr** (never stdout, so
stdout-buffered child processes like asyncpg don't delay log emission
— observed 28-minute log gaps during dev-smoke 2026-04-23).

Each record includes:
    ts          ISO-8601 UTC timestamp
    level       INFO | WARNING | ERROR | ...
    logger      module name (e.g. app.worker.main)
    service     "mobius-rag-api" | "mobius-rag-chunker" | "mobius-rag-embedder"
    message     the formatted log message
    job_id      (when present, via LogRecord extra)
    document_id (when present)
    worker_id   (when present)
    exc_info    traceback string when exception was logged

Cloud Run / Google Cloud Logging auto-parses JSON on stderr and
indexes every top-level key as a label, so `job_id:abc123` becomes a
first-class filter in the log viewer — no more grep-by-eye.

Design choices:

1. **Opt-in for prod, pretty for dev.** ENV=dev keeps the human
   format (timestamp + level + message), hosted (prod/staging)
   switches to JSON. Dev devs don't need JSON; ops does.
2. **Flush after every record.** `StreamHandler` with
   ``force=True`` and ``flush=True`` on the underlying stream
   (line-buffered on a terminal, but in Cloud Run stderr is NOT
   line-buffered — we explicitly flush).
3. **Bounded message size.** Messages >10K chars are truncated with
   a ``...(truncated N chars)`` marker so a stringified traceback
   can't blow up the log ingest budget.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


_MAX_MSG_LEN = 10_000


class _JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        msg = record.getMessage()
        if len(msg) > _MAX_MSG_LEN:
            msg = msg[:_MAX_MSG_LEN] + f"...(truncated {len(msg) - _MAX_MSG_LEN} chars)"

        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "message": msg,
        }
        # Attach known structured fields from LogRecord.extra
        for key in ("job_id", "document_id", "worker_id", "stage", "paragraph_id"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = str(val)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit.

    Cloud Run captures stderr but doesn't guarantee line buffering;
    an unflushed handler can hide a log line for minutes until the
    kernel page fills. Flushing on every record trades a bit of
    throughput for operator sanity during incidents.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def configure_logging(service: str, *, level: int | None = None) -> None:
    """Install structured logging for this process.

    Call once at module-top of each entrypoint (``app.main``,
    ``app.worker.main``, ``app.embedding_worker``, and the HTTP
    wrappers ``worker_server_*.py``). Idempotent: repeated calls
    replace the handler list.
    """
    env = os.getenv("ENV", "dev").strip().lower()
    hosted = env in ("prod", "staging")
    lvl = level if level is not None else (logging.INFO if hosted else logging.INFO)

    root = logging.getLogger()
    # Clear any handlers a prior basicConfig() installed so we don't
    # double-log.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = _FlushingStreamHandler(stream=sys.stderr)
    if hosted:
        handler.setFormatter(_JsonFormatter(service))
    else:
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s - [{service}] - %(levelname)s - %(name)s - %(message)s"
            )
        )
    root.addHandler(handler)
    root.setLevel(lvl)

    # Silence a few chatty libs at INFO so real worker events stand
    # out in the feed. Explicit so nothing surprising happens.
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
