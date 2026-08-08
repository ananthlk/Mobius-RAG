"""Test for _retriever_hard_timeout_s (app/main.py, 2026-08-08).

Real bug found live: the /api/retriever/answer hard wall-clock timeout was
flat across every caller_mode (45s), but chat.thinking's own per-slot
latency_allowance_ms is 16000ms and a FAN_OUT query (multiple slots, each
independently budgeted, plus Router's own multi-turn loop) has a
legitimate, by-design upper bound that can exceed 45s for that mode.
Confirmed via Cloud Run logs: 17 hard timeouts in 6h, still occurring on
genuinely organic traffic. Fix: caller_mode-aware ceiling, mirroring
allocation.py's own _BACKGROUND_MODES split.
"""
from app.main import _retriever_hard_timeout_s, _RETRIEVER_HARD_TIMEOUT_S, _RETRIEVER_HARD_TIMEOUT_BACKGROUND_S


class TestRetrieverHardTimeoutS:
    def test_chat_thinking_gets_the_background_ceiling(self):
        assert _retriever_hard_timeout_s("chat.thinking") == _RETRIEVER_HARD_TIMEOUT_BACKGROUND_S
        assert _RETRIEVER_HARD_TIMEOUT_BACKGROUND_S > _RETRIEVER_HARD_TIMEOUT_S

    def test_batch_and_background_also_get_the_background_ceiling(self):
        assert _retriever_hard_timeout_s("batch") == _RETRIEVER_HARD_TIMEOUT_BACKGROUND_S
        assert _retriever_hard_timeout_s("background") == _RETRIEVER_HARD_TIMEOUT_BACKGROUND_S

    def test_real_time_modes_keep_the_original_flat_ceiling(self):
        assert _retriever_hard_timeout_s("chat.default") == _RETRIEVER_HARD_TIMEOUT_S
        assert _retriever_hard_timeout_s("chat.copilot") == _RETRIEVER_HARD_TIMEOUT_S

    def test_none_or_unknown_caller_mode_fails_closed_to_the_tight_ceiling(self):
        """Unrecognized/unset caller_mode must NOT accidentally get the
        generous background ceiling -- fail closed to the tight one,
        same fail-closed convention as call_number/authority_requirement
        elsewhere in this pipeline."""
        assert _retriever_hard_timeout_s(None) == _RETRIEVER_HARD_TIMEOUT_S
        assert _retriever_hard_timeout_s("some_future_mode") == _RETRIEVER_HARD_TIMEOUT_S
