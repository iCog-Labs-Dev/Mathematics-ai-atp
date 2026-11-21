"""Opt-in real Lean checks are required, not skipped, in the designated run."""

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-pantograph-live", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "pantograph_live: actual custom Lean executable integration"
    )


@pytest.fixture
def live_env(request):
    """Explicit live paths; a required live run fails instead of skipping."""
    from pantograph_async import PantographEnv

    source = os.environ.get("PANTOGRAPH_TEST_SOURCE_ROOT")
    repl = os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail(
                "Required live run needs PANTOGRAPH_TEST_SOURCE_ROOT and PANTOGRAPH_TEST_REPL."
            )
        pytest.skip("Explicit live paths not configured.")
    return PantographEnv(
        source_root=Path(source), pantograph_repl=Path(repl), imports=("Mathlib",)
    )
