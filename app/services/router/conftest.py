"""Router test configuration — frozen-priors fixture for machinery tests.

The live eval/priors_bootstrap.yaml is an EVAL-OWNED, MUTABLE artifact: the
whole point of the file-swappable priors design is that Eval rewrites it
(first real fold: 2026-07-23 Beta update from the forced-filler bank). Tests
that verify Router MACHINERY (allocation order, chain LB arithmetic, solver
choice, trace replayability) pin their hand-derived expectations to a frozen
snapshot — testdata/priors_frozen_seed.yaml — so live-data folds never break
math tests.

Tests that deliberately change-detect the LIVE file (test_priors_loading's
file-is-primary-source checks) opt out with @pytest.mark.live_priors; they
are the one place that must be re-derived when Eval folds new data.
"""

from pathlib import Path

import pytest

FROZEN_PRIORS_PATH = Path(__file__).parent / "testdata" / "priors_frozen_seed.yaml"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_priors: test reads the LIVE eval priors file (deliberate "
        "change-detector — re-derive expectations when Eval folds new data)",
    )


@pytest.fixture(autouse=True)
def _frozen_priors(request, monkeypatch):
    if request.node.get_closest_marker("live_priors"):
        yield
        return
    monkeypatch.setenv("ROUTER_PRIORS_PATH", str(FROZEN_PRIORS_PATH))
    yield
