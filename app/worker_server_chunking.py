"""HTTP server for Cloud Run: runs the RAG chunking worker in a
supervised background thread and serves ``/health`` so Cloud Run
keeps the instance alive.

Use: ``uvicorn app.worker_server_chunking:app --host 0.0.0.0 --port 8080``

Supervisor design (2026-04-24)
------------------------------
Previously this wrapper spawned a single daemon thread that called
``app.worker.main()``; if that function ever returned (clean exit) or
raised an uncaught exception, the thread died and nothing polled the
queue — while ``/health`` kept returning 200 because FastAPI was
still alive. Cloud Run happily kept routing traffic and charging for
an instance that did nothing. Observed twice in a single afternoon
on mobius-os-dev; each time required manual bouncing.

Two fixes layered together:

1. **Supervisor restart loop.** ``_worker_loop_supervisor`` wraps
   ``main()`` in a ``while not shutting_down: ...`` that restarts
   the worker on any exit (clean or exception) with exponential
   backoff capped at 60s. A true shutdown (SIGTERM/SIGINT from Cloud
   Run) is distinguished via ``is_shutting_down()`` and breaks the
   loop.

2. **Liveness signal via ``/health``.** The supervisor publishes a
   ``_worker_last_tick`` timestamp every iteration. ``/health``
   fails (503) when the tick is older than ``WORKER_LIVENESS_STALE_S``
   (default 120s), which causes Cloud Run's startup/liveness probe
   to mark the instance unhealthy and spin up a replacement. This
   is the real "dead worker" detector; the supervisor alone handles
   the common case, and the /health fail handles pathological cases
   (thread completely gone, event-loop deadlock).
"""
import logging
import os
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.logging_setup import configure_logging
configure_logging("mobius-rag-chunker")
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobius RAG Chunking Worker", version="0.1.0")

# Supervisor state.
_worker_thread: threading.Thread | None = None
_worker_last_tick: float = 0.0  # monotonic timestamp of last supervisor heartbeat
_worker_restart_count: int = 0

# Liveness threshold — if the supervisor hasn't ticked in this many
# seconds, ``/health`` reports unhealthy. Default 120s covers the
# longest single-call window we expect (a DB statement_timeout is
# 10 min but individual commits should fire every few seconds as
# the chunker processes paragraphs).
_LIVENESS_STALE_S = int(os.getenv("WORKER_LIVENESS_STALE_S", "120"))


def _worker_loop_supervisor() -> None:
    """Restart ``app.worker.main()`` on any exit until shutdown.

    Called exactly once, in its own thread, at FastAPI startup. Runs
    an inner event loop: each iteration calls ``main()``, which
    normally blocks forever polling the job queue. If it returns
    (clean exit) or raises, the supervisor logs and retries with a
    backoff. Shutdown is signalled by Cloud Run SIGTERM → FastAPI
    ``@app.on_event("shutdown")`` → ``request_shutdown()`` — the
    supervisor loop exits cleanly when ``is_shutting_down()``
    returns True.
    """
    global _worker_last_tick, _worker_restart_count

    from app.worker.shutdown import is_shutting_down

    backoff = 1.0  # seconds
    while not is_shutting_down():
        _worker_last_tick = time.monotonic()
        try:
            from app.worker import main as worker_main
            logger.info(
                "[supervisor] starting worker.main() (restart_count=%d)",
                _worker_restart_count,
            )
            worker_main()  # normally blocks forever
            # If we get here, main() returned cleanly — unusual but
            # not fatal. Treat it like any other exit and retry.
            logger.warning(
                "[supervisor] worker.main() returned without error — "
                "should have blocked forever. Restarting after %.1fs.",
                backoff,
            )
        except SystemExit as se:
            logger.warning(
                "[supervisor] worker.main() raised SystemExit(%s) — "
                "restarting after %.1fs.", se.code, backoff,
            )
        except Exception as exc:
            logger.exception(
                "[supervisor] worker.main() crashed: %s — restarting after %.1fs.",
                exc, backoff,
            )

        if is_shutting_down():
            break
        _worker_restart_count += 1
        # Tick before sleep so /health doesn't flap during a brief crash.
        _worker_last_tick = time.monotonic()
        time.sleep(backoff)
        backoff = min(backoff * 2, 60.0)

    logger.info(
        "[supervisor] shutdown signalled, exiting (total restarts=%d)",
        _worker_restart_count,
    )


@app.on_event("startup")
def start_worker():
    global _worker_thread, _worker_last_tick
    _worker_last_tick = time.monotonic()
    _worker_thread = threading.Thread(
        target=_worker_loop_supervisor,
        name="rag-chunker-supervisor",
        daemon=True,
    )
    _worker_thread.start()
    logger.info("RAG chunking worker supervisor started in background thread")


@app.on_event("shutdown")
def stop_worker():
    """Cloud Run SIGTERM drain hook.

    Signal handlers installed by the worker's own thread don't fire
    on process SIGTERM (signal.signal is main-thread only), so
    FastAPI's shutdown lifecycle — which runs on the main thread —
    bridges via ``request_shutdown()``. The supervisor's
    ``while not is_shutting_down()`` guard breaks on its next
    iteration and the worker's own loop breaks within ~100ms of the
    same signal. Any in-flight chunking job that can't finish in
    Cloud Run's 10s drain window gets SIGKILLed; the DB row's
    FOR UPDATE lock releases on session rollback, and
    heartbeat-aware stale-recovery picks it up on the next worker
    start.
    """
    try:
        from app.worker.shutdown import request_shutdown
        request_shutdown(source="fastapi-shutdown")
        logger.info("Shutdown requested; waiting up to 10s for worker to drain")
        if _worker_thread:
            _worker_thread.join(timeout=10)
            if _worker_thread.is_alive():
                logger.warning("Worker did not drain within 10s — Cloud Run will SIGKILL")
    except Exception as e:
        logger.warning("Shutdown hook failed (non-fatal): %s", e)


@app.get("/health")
def health():
    """Liveness probe.

    Returns 503 if the supervisor thread is no longer alive — that's
    the real "chunking worker is completely dead" signal. The
    supervisor restarts ``main()`` internally on any inner-loop
    exit, so if the supervisor itself is alive, jobs will get
    processed eventually. If the supervisor died (its own uncaught
    exception in the outer loop, OS-level thread kill, etc.), 503
    gives Cloud Run a liveness signal to replace the instance.

    We deliberately do NOT use a time-since-last-tick check: normal
    operation has ``worker.main()`` blocking inside the chunking
    event loop for hours at a time; a tick-based check would flap
    during long jobs. Thread-alive is a weaker but more stable signal.
    """
    alive = bool(_worker_thread and _worker_thread.is_alive())
    if not alive:
        return JSONResponse(
            {
                "status": "unhealthy",
                "reason": "supervisor thread is not alive — instance needs restart",
                "restart_count": _worker_restart_count,
            },
            status_code=503,
        )
    return {
        "status": "ok",
        "service": "mobius-rag-chunking-worker",
        "restart_count": _worker_restart_count,
    }
