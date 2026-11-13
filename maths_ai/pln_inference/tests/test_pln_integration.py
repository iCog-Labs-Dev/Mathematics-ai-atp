"""Real-runtime PeTTa integration tests.

The unit tests in ``test_pln_model.py`` use stub ``petta`` executables with
fabricated output — they can pass CI while the real startup probe fails.
These tests run the **actual** ``petta`` executable against the vendored
PeTTaChainer modules, so they only run where a real runtime is configured
(inside the Docker image, or on a host with petta + PeTTaChainer installed):

    docker compose run --rm api \
        python -m pytest maths_ai/pln_inference/tests/test_pln_integration.py -v

They are skipped automatically everywhere else.
"""

from __future__ import annotations

import os
import shutil

import pytest

from maths_ai.data_models.proof_components import STV
from maths_ai.pln_inference.model import PLNInference


def _real_runtime_configured() -> bool:
    if not (os.getenv("PETTA_BIN") or shutil.which("petta")):
        return False
    chainer_dir = os.getenv("PETTA_CHAINER_DIR", "/opt/pettachainer/pettachainer/metta")
    return os.path.isdir(chainer_dir)


pytestmark = pytest.mark.skipif(
    not _real_runtime_configured(),
    reason=(
        "real PeTTa runtime not configured "
        "(requires PETTA_BIN/PETTA_CHAINER_DIR; run inside the Docker image)"
    ),
)


@pytest.fixture(scope="module")
def pln() -> PLNInference:
    instance = PLNInference()
    instance.initialize()  # the startup probe itself is under test
    return instance


class TestRealRuntime:
    def test_startup_probe_succeeds(self, pln: PLNInference):
        """The readiness probe (query P given hypothesis P) passes on a
        healthy install."""
        assert pln.is_available is True
        assert pln.initialization_error is None

    def test_known_fact_query_is_real_non_fallback(self, pln: PLNInference):
        """P given hypothesis P must yield a real, non-fallback STV."""
        result = pln.evaluate("P", ["P"])
        assert result.is_fallback is False, (
            f"expected a real STV, got status={result.status} "
            f"stv={result.stv} raw_output:\n{result.raw_output[-2000:]}"
        )
        assert result.status == "ok"
        assert result.stv.score > 0.0

    def test_unknown_formula_is_zero_evidence_fallback(self, pln: PLNInference):
        """P without evidence: a successful run with no query STV is the
        normal no-proof outcome, not an error."""
        result = pln.evaluate("P")
        assert result.is_fallback is True
        assert result.status == "no_stv_found", result.raw_output[-2000:]
        assert result.stv == STV(strength=0.0, confidence=0.0)

    def test_false_is_zero_evidence_fallback(self, pln: PLNInference):
        result = pln.evaluate("False")
        assert result.is_fallback is True
        assert result.status == "no_stv_found", result.raw_output[-2000:]
        assert result.stv == STV(strength=0.0, confidence=0.0)

    def test_contradiction_is_zero_evidence_fallback(self, pln: PLNInference):
        result = pln.evaluate("P ∧ ¬ P")
        assert result.is_fallback is True
        assert result.stv == STV(strength=0.0, confidence=0.0)

    def test_inconsistent_arithmetic_is_zero_evidence_fallback(self, pln: PLNInference):
        result = pln.evaluate("0 = 1")
        assert result.is_fallback is True
        assert result.stv == STV(strength=0.0, confidence=0.0)

    def test_repeated_requests_are_deterministic(self, pln: PLNInference):
        first = pln.evaluate("P", ["P"])
        second = pln.evaluate("P", ["P"])
        assert first.stv == second.stv
        assert first.status == second.status
        assert first.is_fallback == second.is_fallback
