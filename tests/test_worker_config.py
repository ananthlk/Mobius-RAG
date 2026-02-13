"""Unit tests for app.worker.config."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.worker.config import WorkerConfig, load_worker_config


# --- WorkerConfig defaults ---

def test_worker_config_defaults():
    cfg = WorkerConfig()
    assert cfg.poll_interval_seconds == 2.0
    assert cfg.error_sleep_seconds == 5.0
    assert cfg.default_threshold == 0.6
    assert cfg.default_critique_enabled is True
    assert cfg.default_max_retries == 2
    assert cfg.default_extraction_enabled is True
    assert cfg.path_b_cap_ngrams == 500
    assert cfg.path_b_cap_abbrevs == 200
    assert cfg.path_b_min_occurrences == 1
    assert cfg.fail_job_on_first_paragraph_error is False
    assert cfg.job_timeout_seconds is None
    assert cfg.llm_call_timeout_seconds is None
    assert cfg.heartbeat_interval_seconds == 30.0
    assert cfg.log_level == "INFO"


def test_worker_config_is_frozen():
    cfg = WorkerConfig()
    with pytest.raises(AttributeError):
        cfg.poll_interval_seconds = 99.0  # type: ignore[misc]


# --- load_worker_config with no env vars ---

def test_load_worker_config_defaults():
    """With no env vars set, load_worker_config returns the same as WorkerConfig()."""
    # Clear any env vars that might be set
    keys = [
        "WORKER_POLL_INTERVAL", "WORKER_ERROR_SLEEP", "CRITIQUE_RETRY_THRESHOLD",
        "WORKER_CRITIQUE_ENABLED", "WORKER_MAX_RETRIES", "WORKER_EXTRACTION_ENABLED",
        "WORKER_PATH_B_CAP_NGRAMS", "WORKER_PATH_B_CAP_ABBREVS", "WORKER_PATH_B_MIN_OCCURRENCES",
        "WORKER_FAIL_ON_FIRST_ERROR", "WORKER_JOB_TIMEOUT", "WORKER_LLM_CALL_TIMEOUT",
        "WORKER_HEARTBEAT_INTERVAL", "WORKER_LOG_LEVEL", "WORKER_LOG_FORMAT",
    ]
    env = {k: v for k, v in os.environ.items() if k not in keys}
    with patch.dict(os.environ, env, clear=True):
        cfg = load_worker_config()
    assert cfg.default_threshold == 0.6
    assert cfg.default_critique_enabled is True
    assert cfg.job_timeout_seconds is None


# --- load_worker_config with overrides ---

def test_load_worker_config_float_override():
    with patch.dict(os.environ, {"CRITIQUE_RETRY_THRESHOLD": "0.8", "WORKER_POLL_INTERVAL": "5.5"}, clear=False):
        cfg = load_worker_config()
    assert cfg.default_threshold == 0.8
    assert cfg.poll_interval_seconds == 5.5


def test_load_worker_config_int_override():
    with patch.dict(os.environ, {"WORKER_MAX_RETRIES": "5", "WORKER_PATH_B_CAP_NGRAMS": "1000"}, clear=False):
        cfg = load_worker_config()
    assert cfg.default_max_retries == 5
    assert cfg.path_b_cap_ngrams == 1000


def test_load_worker_config_bool_override():
    with patch.dict(os.environ, {"WORKER_CRITIQUE_ENABLED": "false", "WORKER_FAIL_ON_FIRST_ERROR": "true"}, clear=False):
        cfg = load_worker_config()
    assert cfg.default_critique_enabled is False
    assert cfg.fail_job_on_first_paragraph_error is True


def test_load_worker_config_bool_truthy_values():
    for truthy in ("1", "true", "True", "TRUE", "yes", "YES"):
        with patch.dict(os.environ, {"WORKER_FAIL_ON_FIRST_ERROR": truthy}, clear=False):
            cfg = load_worker_config()
        assert cfg.fail_job_on_first_paragraph_error is True, f"Expected True for {truthy!r}"


def test_load_worker_config_bool_falsy_values():
    for falsy in ("0", "false", "no", "random"):
        with patch.dict(os.environ, {"WORKER_FAIL_ON_FIRST_ERROR": falsy}, clear=False):
            cfg = load_worker_config()
        assert cfg.fail_job_on_first_paragraph_error is False, f"Expected False for {falsy!r}"


def test_load_worker_config_opt_float_set():
    with patch.dict(os.environ, {"WORKER_JOB_TIMEOUT": "300.0"}, clear=False):
        cfg = load_worker_config()
    assert cfg.job_timeout_seconds == 300.0


def test_load_worker_config_opt_float_empty():
    with patch.dict(os.environ, {"WORKER_JOB_TIMEOUT": ""}, clear=False):
        cfg = load_worker_config()
    assert cfg.job_timeout_seconds is None


# --- Invalid env values fall back to defaults ---

def test_load_worker_config_invalid_float_falls_back():
    with patch.dict(os.environ, {"CRITIQUE_RETRY_THRESHOLD": "not_a_number"}, clear=False):
        cfg = load_worker_config()
    assert cfg.default_threshold == 0.6


def test_load_worker_config_invalid_int_falls_back():
    with patch.dict(os.environ, {"WORKER_MAX_RETRIES": "abc"}, clear=False):
        cfg = load_worker_config()
    assert cfg.default_max_retries == 2


def test_load_worker_config_log_level_override():
    with patch.dict(os.environ, {"WORKER_LOG_LEVEL": "DEBUG"}, clear=False):
        cfg = load_worker_config()
    assert cfg.log_level == "DEBUG"
