"""
Chunking worker configuration.

Single source of truth for worker-level defaults and tunables.
Job-level fields (threshold, critique_enabled, max_retries, etc.) can override
these defaults per run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable worker configuration loaded once at startup."""

    # --- Polling / sleep ---
    poll_interval_seconds: float = 2.0
    error_sleep_seconds: float = 5.0

    # --- Job defaults (overridden by ChunkingJob fields when present) ---
    default_threshold: float = 0.6
    default_critique_enabled: bool = True
    default_max_retries: int = 2
    default_extraction_enabled: bool = True

    # --- Path B caps (used in extract_candidates_for_document) ---
    path_b_cap_ngrams: int = 500
    path_b_cap_abbrevs: int = 200
    path_b_min_occurrences: int = 1

    # --- Production knobs ---
    fail_job_on_first_paragraph_error: bool = False
    job_timeout_seconds: float | None = None  # None = no timeout
    llm_call_timeout_seconds: float | None = None
    heartbeat_interval_seconds: float = 30.0

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - [WORKER] - %(levelname)s - %(message)s"


def load_worker_config() -> WorkerConfig:
    """Build WorkerConfig from environment variables (with defaults)."""
    def _float(key: str, default: float) -> float:
        raw = os.getenv(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _int(key: str, default: int) -> int:
        raw = os.getenv(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def _bool(key: str, default: bool) -> bool:
        raw = os.getenv(key)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes")

    def _opt_float(key: str) -> float | None:
        raw = os.getenv(key)
        if not raw:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    return WorkerConfig(
        poll_interval_seconds=_float("WORKER_POLL_INTERVAL", 2.0),
        error_sleep_seconds=_float("WORKER_ERROR_SLEEP", 5.0),
        default_threshold=_float("CRITIQUE_RETRY_THRESHOLD", 0.6),
        default_critique_enabled=_bool("WORKER_CRITIQUE_ENABLED", True),
        default_max_retries=_int("WORKER_MAX_RETRIES", 2),
        default_extraction_enabled=_bool("WORKER_EXTRACTION_ENABLED", True),
        path_b_cap_ngrams=_int("WORKER_PATH_B_CAP_NGRAMS", 500),
        path_b_cap_abbrevs=_int("WORKER_PATH_B_CAP_ABBREVS", 200),
        path_b_min_occurrences=_int("WORKER_PATH_B_MIN_OCCURRENCES", 1),
        fail_job_on_first_paragraph_error=_bool("WORKER_FAIL_ON_FIRST_ERROR", False),
        job_timeout_seconds=_opt_float("WORKER_JOB_TIMEOUT"),
        llm_call_timeout_seconds=_opt_float("WORKER_LLM_CALL_TIMEOUT"),
        heartbeat_interval_seconds=_float("WORKER_HEARTBEAT_INTERVAL", 30.0),
        log_level=os.getenv("WORKER_LOG_LEVEL", "INFO"),
        log_format=os.getenv(
            "WORKER_LOG_FORMAT",
            "%(asctime)s - [WORKER] - %(levelname)s - %(message)s",
        ),
    )
