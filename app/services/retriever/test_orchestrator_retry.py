"""Unit tests for run_retriever_partial_with_retry: whole-loop retry on a
TECHNICAL failure only (2026-07-24, Ananth's "ask once, try our best to
get first-pass resolution" principle). Mocks run_retriever_partial itself
(the function being wrapped) -- this isolates the retry CONTROL FLOW from
the real pipeline's internals, which is what this wrapper actually owns.
"""

from unittest.mock import patch

import pytest

from app.services.retriever.orchestrator import (
    RetrieverPartialResult,
    run_retriever_partial_with_retry,
)


@pytest.mark.asyncio
async def test_retries_once_on_technical_failure_and_succeeds():
    """First attempt raises (simulating a dropped DB connection or similar
    infra failure); the retry succeeds. Caller gets a real result, not an
    exception -- exactly "ask once, we try our best" for a technical
    failure, not a low-confidence answer."""
    call_count = {"n": 0}

    async def flaky(db, query, caller_mode=None, attempt=0, retry_of_decision_id=None, token_budget_for_retrieval=None, forced_strategy=None):
        call_count["n"] += 1
        if attempt == 0:
            raise ConnectionError("connection was closed in the middle of operation")
        return RetrieverPartialResult(query=query)

    with patch("app.services.retriever.orchestrator.run_retriever_partial", side_effect=flaky):
        result = await run_retriever_partial_with_retry(db=None, query="q", max_retries=1)

    assert call_count["n"] == 2
    assert result.query == "q"


@pytest.mark.asyncio
async def test_gives_up_and_reraises_after_max_retries_exhausted():
    """Both attempts fail -- the caller must see the real exception, not a
    silently-swallowed empty result. "We try our best" has a real limit,
    not infinite retries."""
    async def always_fails(db, query, caller_mode=None, attempt=0, retry_of_decision_id=None, token_budget_for_retrieval=None, forced_strategy=None):
        raise ConnectionError("still broken")

    with patch("app.services.retriever.orchestrator.run_retriever_partial", side_effect=always_fails):
        with pytest.raises(ConnectionError, match="still broken"):
            await run_retriever_partial_with_retry(db=None, query="q", max_retries=1)


@pytest.mark.asyncio
async def test_succeeds_first_try_without_ever_retrying():
    """The common case: no failure at all -- exactly one call, no retry
    logic even triggered."""
    call_count = {"n": 0}

    async def clean(db, query, caller_mode=None, attempt=0, retry_of_decision_id=None, token_budget_for_retrieval=None, forced_strategy=None):
        call_count["n"] += 1
        return RetrieverPartialResult(query=query)

    with patch("app.services.retriever.orchestrator.run_retriever_partial", side_effect=clean):
        result = await run_retriever_partial_with_retry(db=None, query="q", max_retries=1)

    assert call_count["n"] == 1
    assert result.query == "q"


@pytest.mark.asyncio
async def test_each_attempt_passed_the_correct_attempt_number():
    """Real requirement (Eval calibration dedup): each retry must be
    threaded a distinct attempt number so Router's persisted decision row
    is distinguishable, not a silent duplicate query observation."""
    seen_attempts = []

    async def record_attempt(db, query, caller_mode=None, attempt=0, retry_of_decision_id=None, token_budget_for_retrieval=None, forced_strategy=None):
        seen_attempts.append(attempt)
        if attempt == 0:
            raise RuntimeError("boom")
        return RetrieverPartialResult(query=query)

    with patch("app.services.retriever.orchestrator.run_retriever_partial", side_effect=record_attempt):
        await run_retriever_partial_with_retry(db=None, query="q", max_retries=1)

    assert seen_attempts == [0, 1]


@pytest.mark.asyncio
async def test_retry_of_decision_id_threaded_when_attempt0_persisted_before_dying():
    """Router's ask (2026-07-24): exact retry pairing. If attempt-0's
    exception carries `retriever_partial_decision_id` (Router already
    persisted a decision row before the pipeline died later, e.g. during
    Fillers), the next attempt must receive it as `retry_of_decision_id`
    so Router can stamp `retry_of_decision` into feature_vector JSONB for
    exact pairing instead of fuzzy matching."""
    seen_retry_of_decision_ids = []

    async def dies_after_persisting(db, query, caller_mode=None, attempt=0, retry_of_decision_id=None, token_budget_for_retrieval=None, forced_strategy=None):
        seen_retry_of_decision_ids.append(retry_of_decision_id)
        if attempt == 0:
            exc = RuntimeError("fillers blew up after router persisted")
            exc.retriever_partial_decision_id = "real-decision-id-abc123"
            raise exc
        return RetrieverPartialResult(query=query)

    with patch("app.services.retriever.orchestrator.run_retriever_partial", side_effect=dies_after_persisting):
        await run_retriever_partial_with_retry(db=None, query="q", max_retries=1)

    assert seen_retry_of_decision_ids == [None, "real-decision-id-abc123"]


@pytest.mark.asyncio
async def test_retry_of_decision_id_none_when_failure_precedes_router():
    """Complementary case: if the failure happens before Router ever ran
    (e.g. during Gate/Pool), there's no decision_id to pair against --
    retry_of_decision_id must stay None, not a stale/wrong value."""
    seen_retry_of_decision_ids = []

    async def dies_before_router(db, query, caller_mode=None, attempt=0, retry_of_decision_id=None, token_budget_for_retrieval=None, forced_strategy=None):
        seen_retry_of_decision_ids.append(retry_of_decision_id)
        if attempt == 0:
            raise ConnectionError("gate's db call dropped")  # no decision_id attribute set
        return RetrieverPartialResult(query=query)

    with patch("app.services.retriever.orchestrator.run_retriever_partial", side_effect=dies_before_router):
        await run_retriever_partial_with_retry(db=None, query="q", max_retries=1)

    assert seen_retry_of_decision_ids == [None, None]
