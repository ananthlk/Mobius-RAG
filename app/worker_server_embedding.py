"""HTTP server for Cloud Run: runs the RAG embedding worker in a
supervised background thread and serves ``/health`` so Cloud Run
keeps the instance alive.

Use: ``uvicorn app.worker_server_embedding:app --host 0.0.0.0 --port 8080``

See ``app/worker_server_chunking.py`` for the supervisor design
rationale — this is the same pattern applied to the embedding
worker.
"""
import logging
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.logging_setup import configure_logging
configure_logging("mobius-rag-embedder")
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobius RAG Embedding Worker", version="0.1.0")

_worker_thread: threading.Thread | None = None
_worker_restart_count: int = 0


def _worker_loop_supervisor() -> None:
    """Restart ``app.embedding_worker.main()`` on any exit until shutdown."""
    global _worker_restart_count

    from app.worker.shutdown import is_shutting_down
    import time

    backoff = 1.0
    while not is_shutting_down():
        try:
            from app.embedding_worker import main as embed_main
            logger.info(
                "[supervisor] starting embedding_worker.main() (restart_count=%d)",
                _worker_restart_count,
            )
            embed_main()
            logger.warning(
                "[supervisor] embedding_worker.main() returned — restarting after %.1fs.",
                backoff,
            )
        except SystemExit as se:
            logger.warning(
                "[supervisor] embedding_worker.main() raised SystemExit(%s) — "
                "restarting after %.1fs.", se.code, backoff,
            )
        except Exception as exc:
            logger.exception(
                "[supervisor] embedding_worker.main() crashed: %s — restarting after %.1fs.",
                exc, backoff,
            )

        if is_shutting_down():
            break
        _worker_restart_count += 1
        time.sleep(backoff)
        backoff = min(backoff * 2, 60.0)

    logger.info(
        "[supervisor] shutdown signalled, exiting (total restarts=%d)",
        _worker_restart_count,
    )


@app.on_event("startup")
def start_worker():
    global _worker_thread
    _worker_thread = threading.Thread(
        target=_worker_loop_supervisor,
        name="rag-embedder-supervisor",
        daemon=True,
    )
    _worker_thread.start()
    logger.info("RAG embedding worker supervisor started in background thread")


@app.on_event("shutdown")
def stop_worker():
    """Cloud Run SIGTERM drain hook. See ``worker_server_chunking.py``
    for the full rationale; same pattern applies here.
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
    """503 when supervisor thread is dead. See chunker /health for rationale."""
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
        "service": "mobius-rag-embedding-worker",
        "restart_count": _worker_restart_count,
    }
