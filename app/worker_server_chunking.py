"""
Minimal HTTP server for Cloud Run: runs the RAG chunking worker in a background thread
and serves /health so Cloud Run keeps the instance alive.
Use: uvicorn app.worker_server_chunking:app --host 0.0.0.0 --port 8080
"""
import logging
import threading

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [CHUNK-WORKER] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobius RAG Chunking Worker", version="0.1.0")

_worker_thread: threading.Thread | None = None


@app.on_event("startup")
def start_worker():
    global _worker_thread

    def _run():
        try:
            from app.worker import main
            main()
        except Exception as e:
            logger.exception("Chunking worker loop exited: %s", e)

    _worker_thread = threading.Thread(target=_run, daemon=True)
    _worker_thread.start()
    logger.info("RAG chunking worker started in background thread")


@app.on_event("shutdown")
def stop_worker():
    """Cloud Run SIGTERM drain hook.

    Signal handlers installed by the worker's own thread don't fire
    on process SIGTERM (signal.signal is main-thread only), so
    FastAPI's shutdown lifecycle — which runs on the main thread —
    bridges via request_shutdown(). The worker loop's
    ``while not is_shutting_down():`` guard breaks on its next
    iteration, and any in-flight chunking job finishes naturally
    within Cloud Run's 10s drain window (DB row FOR UPDATE lock
    releases on session rollback if SIGKILL lands mid-job, and
    stale-recovery picks up the job on the next worker start).
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
    return {"status": "ok", "service": "mobius-rag-chunking-worker"}
