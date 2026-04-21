"""Graceful shutdown primitive for rag workers.

Shared ``asyncio.Event`` that every worker loop polls between
iterations. SIGTERM handler sets it; in-flight jobs finish naturally
under their own deadlines, then the loop exits cleanly.

Why this matters (pre-2026-04-21):
    Cloud Run sends SIGTERM + gives the container 10 seconds to drain
    before SIGKILL. The rag chunking/embedding workers had no signal
    handler — SIGKILL landed mid-extraction, leaving the claimed job
    row locked by a dead session and the document stuck in
    ``chunking_status=processing`` forever. Operators had to POST to
    ``/admin/cleanup-stale-jobs`` to recover. Not a real ops story.

Contract:
    * ``install_handlers()`` is idempotent — safe to call from each
      entry point (``worker_loop``, ``embedding_worker.main``, and
      the HTTP-wrapper ``@app.on_event("shutdown")`` hooks).
    * ``is_shutting_down()`` is cheap; call on every loop iteration.
    * After the loop breaks, the DB session context-manager's
      rollback releases the FOR UPDATE SKIP LOCKED row. A fresh
      worker (or the stale-recovery sweep) picks up the job on
      restart — no permanent stuck-in-processing state.

What this doesn't do (yet):
    * Cancel in-flight LLM calls. A 40s Vertex extraction will run
      to completion on SIGTERM, exceeding the 10s drain window.
      Cloud Run then SIGKILLs and DB rollback recovers the job.
      Acceptable because stale-recovery already catches this pattern.
      If we need finer-grained mid-LLM cancellation, wrap extraction
      in ``asyncio.wait_for(..., timeout=drain_seconds)`` gated on
      ``is_shutting_down()``.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import threading

logger = logging.getLogger(__name__)

# Module-level event, set exactly once on first SIGTERM/SIGINT. Using
# threading.Event (not asyncio.Event) so workers that run in daemon
# threads (worker_server_*.py) can also poll it synchronously.
_shutdown = threading.Event()
_handlers_installed = False
_lock = threading.Lock()


def install_handlers(worker_name: str = "worker") -> None:
    """Install SIGTERM + SIGINT handlers on this process.

    Idempotent. Safe to call multiple times (e.g. from both
    ``worker_loop`` and FastAPI startup hooks).
    """
    global _handlers_installed
    with _lock:
        if _handlers_installed:
            return
        _handlers_installed = True

    def _handler(signum, frame):  # noqa: ARG001
        sig = signal.Signals(signum).name
        if _shutdown.is_set():
            logger.warning(
                "[%s] second %s received — process will exit when "
                "current loop iteration returns", worker_name, sig,
            )
            return
        logger.info(
            "[%s] received %s — draining. Current job (if any) will "
            "finish; no new jobs will be claimed.",
            worker_name, sig,
        )
        _shutdown.set()

    # ``signal.signal`` only works in the main thread. Workers started
    # by FastAPI lifespan hooks run the loop in a daemon thread; the
    # FastAPI process itself catches SIGTERM on the main thread and
    # we bridge via the shared event. Workers launched directly (via
    # ``python -m app.worker``) ARE on the main thread and get the
    # direct handler.
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError) as e:
            # ValueError: signal only works in main thread
            # OSError: platform doesn't support this signal
            logger.debug(
                "[%s] could not install %s handler (%s) — "
                "shutdown must be triggered via request_shutdown()",
                worker_name, sig.name, e,
            )


def request_shutdown(source: str = "external") -> None:
    """Explicitly signal shutdown. Used by FastAPI wrapper's
    ``@app.on_event('shutdown')`` hook which runs on the main thread
    while the worker loop runs on a daemon thread — signal.signal
    can't bridge that automatically.
    """
    if not _shutdown.is_set():
        logger.info("shutdown requested (source=%s)", source)
    _shutdown.set()


def is_shutting_down() -> bool:
    """Cheap check for the loop's ``while`` guard."""
    return _shutdown.is_set()


async def sleep_or_shutdown(seconds: float) -> bool:
    """Sleep up to ``seconds``, returning early if shutdown fires.

    Returns True if shutdown was requested during the sleep.
    Use in place of ``await asyncio.sleep(n)`` inside worker loops so
    a SIGTERM during an idle poll doesn't have to wait out the full
    poll interval.
    """
    # Poll at a reasonable granularity (~100ms) so SIGTERM-to-exit
    # latency stays under half a second. We keep it here instead of
    # asyncio.wait_for on a future because threading.Event integrates
    # cleanly across daemon threads; asyncio.Event would need per-loop
    # setup in the signal handler.
    step = 0.1
    remaining = seconds
    while remaining > 0 and not _shutdown.is_set():
        await asyncio.sleep(min(step, remaining))
        remaining -= step
    return _shutdown.is_set()
