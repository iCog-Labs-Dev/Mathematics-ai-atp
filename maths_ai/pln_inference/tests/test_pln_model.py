"""Tests for :mod:`maths_ai.pln_inference.model` (PeTTaChainer wrapper).

Uses stub ``petta`` executables so no real PeTTa install is needed. The stub
output mimics the *ordering* of real PeTTa runs: imported-module/axiom
compilation and ``compileadd`` hypothesis facts are printed first (these may
mention arbitrary formulas — including the queried one — and contain words
like "error"), then the query is echoed/compiled to a Prolog ``findall``
goal, and only the output after that marker is the query's own result. The
formula-scoped, marker-scoped extraction must respect that ordering.

These stubs cannot prove output-format fidelity — that is the job of the
real-runtime integration tests in ``test_pln_integration.py``, which run
inside the Docker image against the actual ``petta`` executable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maths_ai.data_models.proof_components import STV
from maths_ai.pln_inference.model import PLNInference


# --- real-run output building blocks (in emission order) ---------------------

# Compiled axiom records: PeTTaChainer prints an STV for every compiled
# axiom. These may mention the queried formula and are printed BEFORE the
# query — they are KB state, never query results.
AXIOM_RECORDS = (
    "(: (kb MAIN Nil) axiom (Implication (Premises ($phi)) "
    "(Conclusions ($phi))) (STV 1.0 1.0))\n"
    "(: (kb MAIN Nil) axiom-p (P) (STV 1.0 1.0))\n"
)

# compileadd of a local hypothesis: also pre-query, also carries an STV.
HYPOTHESIS_RECORDS = "(: (kb MAIN Nil) local-hyp-0 (P) (STV 1.0 1.0))\n"

# The query marker: the interpreter echoes the `!(query ...)` runnable and
# prints its compiled Prolog `findall(A, query(...))` goal.
QUERY_MARKER = (
    "!(query 10 kb (: $prf (P) $tv))\n"
    ":- findall(A, query(10, kb, [:, _, ['P'], _], A), _).\n"
)

# Query output after the marker: the actual result records.
QUERY_RESULT_P = "(: (kb MAIN Nil) query-proof (P) (STV 0.35 0.8))\n"

# Imported source / compiler chatter: contains error-ish words but is NOT a
# runtime failure. Real logs of successful runs contain such lines.
PRE_QUERY_CHATTER = (
    "warning: unknown procedure in imported module (ignored)\n"
    "ERROR: user:some-imported-module library/3: stale directive\n"
)


def successful_run(result_record: str = QUERY_RESULT_P) -> str:
    """A run where the chainer loaded and the query produced ``result_record``."""
    return (
        PRE_QUERY_CHATTER
        + AXIOM_RECORDS
        + HYPOTHESIS_RECORDS
        + QUERY_MARKER
        + result_record
    )


# A run where the chainer import failed: the query remains an unevaluated
# echo — no query marker and no result.
CHAINER_NOT_LOADED_RUN = (
    "(import! &self petta_chainer)\n"
    "(compileadd kb (: local-hyp-0 (P) (STV 1.0 1.0)))\n"
    "(query 10 kb (: $prf (P) $tv))\n"
)

CHAINER_IMPORT_CRASH_RUN = (
    "ERROR: user:main library/3: Unknown procedure: library_path/1\n"
)


def make_petta_stub(tmp_path: Path, output: str, returncode: int = 0) -> Path:
    """Create an executable that pretends to be the ``petta`` interpreter."""
    stub = tmp_path / "bin" / "petta"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        f"#!/bin/sh\nprintf '%s' '{output}'\nexit {returncode}\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def make_pln(
    tmp_path: Path,
    axioms_file: Path,
    chainer_dir: Path,
    petta_bin: Path | None = None,
    **overrides,
) -> PLNInference:
    kwargs = dict(
        petta_bin=str(petta_bin) if petta_bin else None,
        axioms_path=axioms_file,
        chainer_dir=chainer_dir,
        workspace_dir=tmp_path / "petta_workspace",
    )
    kwargs.update(overrides)
    return PLNInference(**kwargs)


@pytest.fixture
def axioms_file(tmp_path: Path) -> Path:
    path = tmp_path / "metamath_axioms.metta"
    path.write_text(";; stub axioms\n", encoding="utf-8")
    return path


@pytest.fixture
def chainer_dir(tmp_path: Path) -> Path:
    """A stand-in for the vendored PeTTaChainer metta directory."""
    chainer = tmp_path / "pettachainer" / "metta"
    chainer.mkdir(parents=True)
    (chainer / "petta_chainer.metta").write_text("!(import! &self compile)\n", encoding="utf-8")
    (chainer / "compile.metta").write_text(";; compile stub\n", encoding="utf-8")
    return chainer


class TestWorkspace:
    def test_chainer_modules_linked_into_workspace(
        self, tmp_path, axioms_file, chainer_dir
    ):
        workspace = tmp_path / "petta_workspace"

        pln = make_pln(
            tmp_path, axioms_file, chainer_dir, petta_bin=(tmp_path / "absent")
        )
        assert workspace.is_dir()
        for name in ("petta_chainer.metta", "compile.metta"):
            assert (workspace / name).is_symlink(), f"workspace missing link to {name}"
        assert pln._workspace_entries  # links were registered

    def test_workspace_reuse_without_recreated_entries(self, tmp_path, axioms_file, chainer_dir):
        """A fresh evaluator over an existing workspace must still register the links.

        Regression: the first process creates the symlinks; if a second process
        found them 'already present' without registering anything, its query
        files would be written outside the workspace and the chainer import
        would silently fail.
        """
        make_pln(tmp_path, axioms_file, chainer_dir, petta_bin=(tmp_path / "absent"))
        workspace = tmp_path / "petta_workspace"
        before = sorted(p.name for p in workspace.iterdir())

        second = make_pln(tmp_path, axioms_file, chainer_dir, petta_bin=(tmp_path / "absent"))
        assert second._workspace_entries, "workspace links were not re-registered"
        assert sorted(p.name for p in workspace.iterdir()) == before
        assert (workspace / "petta_chainer.metta").is_symlink()

    def test_missing_chainer_dir_is_tolerated(self, tmp_path, axioms_file):
        pln = make_pln(
            tmp_path, axioms_file, tmp_path / "does-not-exist",
            petta_bin=make_petta_stub(tmp_path, successful_run()),
        )
        assert pln._workspace_entries == {}
        # petta still runs (evaluate succeeds) — probe coverage is separate.
        assert pln.evaluate("P").is_fallback is False


class TestEvaluate:
    def test_successful_run_is_real_result(self, tmp_path, axioms_file, chainer_dir):
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, successful_run()),
        )
        result = pln.evaluate("P")
        assert result.is_fallback is False
        assert result.status == "ok"
        # The post-marker query result, not the perfect (1.0, 1.0) KB records.
        assert result.stv == STV(strength=0.35, confidence=0.8)

    def test_query_given_hypothesis_is_real_result(
        self, tmp_path, axioms_file, chainer_dir
    ):
        """P given hypothesis P is the canonical real, non-fallback query."""
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, successful_run()),
        )
        result = pln.evaluate("P", ["P"])
        assert result.is_fallback is False
        assert result.status == "ok"
        assert result.stv.score > 0.0

    def test_identical_requests_are_consistent(self, tmp_path, axioms_file, chainer_dir):
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, successful_run()),
        )
        first, second = pln.evaluate("P"), pln.evaluate("P")
        assert (first.status, first.stv == second.stv) == ("ok", True)

    def test_unrelated_axiom_stv_is_not_reused(self, tmp_path, axioms_file, chainer_dir):
        """A pre-query KB fact for the same formula must not score the query.

        Both the compiled axiom `axiom-p` and the asserted hypothesis carry
        (STV 1.0 1.0) for P, yet the query result is weaker — the result must
        come from post-marker output only.
        """
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, successful_run()),
        )
        result = pln.evaluate("P")
        assert result.stv == STV(strength=0.35, confidence=0.8)
        assert result.stv != STV(strength=1.0, confidence=1.0)

    def test_hypothesis_stv_not_reused_without_query_result(
        self, tmp_path, axioms_file, chainer_dir
    ):
        """P without evidence: pre-query P facts exist but the query itself
        finds nothing -> zero-evidence fallback, never the KB fact's STV."""
        no_result_run = PRE_QUERY_CHATTER + AXIOM_RECORDS + HYPOTHESIS_RECORDS + QUERY_MARKER
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, no_result_run),
        )
        result = pln.evaluate("P", ["P"])
        assert result.is_fallback is True
        assert result.status == "no_stv_found"
        assert result.stv == STV(strength=0.0, confidence=0.0)

    def test_false_is_zero_evidence_fallback(self, tmp_path, axioms_file, chainer_dir):
        no_result_run = PRE_QUERY_CHATTER + AXIOM_RECORDS + QUERY_MARKER
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, no_result_run),
        )
        result = pln.evaluate("False")
        assert result.is_fallback is True
        assert result.status == "no_stv_found"
        assert result.stv == STV(strength=0.0, confidence=0.0)

    def test_pre_query_error_chatter_is_not_an_error(
        self, tmp_path, axioms_file, chainer_dir
    ):
        """Imported-source/compiler output may contain ERROR lines; a run
        that exits 0 with no query result is a normal no_stv_found."""
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, PRE_QUERY_CHATTER + QUERY_MARKER),
        )
        result = pln.evaluate("P")
        assert result.status == "no_stv_found"
        assert result.is_fallback is True

    def test_query_runtime_error_is_flagged(self, tmp_path, axioms_file, chainer_dir):
        """An ERROR line in the query's own output classifies as log_error."""
        error_in_query = PRE_QUERY_CHATTER + QUERY_MARKER + "ERROR: findall/3: evaluation failed\n"
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, error_in_query),
        )
        result = pln.evaluate("P")
        assert result.status == "log_error_no_stv"
        assert result.is_fallback is True


class TestFallbackScores:
    def test_random_fallback_disabled_by_default(self, tmp_path, axioms_file, chainer_dir):
        """No petta => a deterministic zero-evidence placeholder, not noise."""
        pln = make_pln(tmp_path, axioms_file, chainer_dir, petta_bin=None)
        first, second = pln.evaluate("P"), pln.evaluate("P")

        assert first.is_fallback is True
        assert first.status == "petta_unavailable"
        assert first.stv == STV(strength=0.0, confidence=0.0)
        assert first.stv == second.stv

    def test_missing_binary_falls_back(self, tmp_path, axioms_file):
        pln = make_pln(
            tmp_path, axioms_file, tmp_path / "chainer", petta_bin=None
        )
        result = pln.evaluate("P")
        assert result.status == "petta_unavailable"
        assert result.is_fallback is True

    def test_random_fallback_sampled_when_enabled(self, tmp_path, axioms_file, chainer_dir):
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=None,
            fallback_enabled=True,
            random_seed=1,
        )
        result = pln.evaluate("P")
        assert result.is_fallback is True
        assert result.stv.confidence == 1.0
        assert 0.0 <= result.stv.strength <= 1.0

    def test_failed_process_is_transparent(self, tmp_path, axioms_file, chainer_dir):
        """A non-zero petta exit is flagged as fallback, never presented as evidence."""
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, successful_run(), returncode=2),
        )
        result = pln.evaluate("P")
        assert result.is_fallback is True
        assert result.stv.score == 0.0
        assert pln.is_available is False


class TestInitialize:
    def test_happy_path(self, tmp_path, axioms_file, chainer_dir):
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, successful_run()),
        )
        pln.initialize()
        assert pln.is_available is True
        assert pln.initialization_error is None

    def test_missing_binary(self, tmp_path, axioms_file, chainer_dir):
        pln = make_pln(tmp_path, axioms_file, chainer_dir, petta_bin=None)
        with pytest.raises(RuntimeError, match="not found"):
            pln.initialize()
        assert pln.is_available is False

    def test_silent_chainer_import_failure_detected(
        self, tmp_path, axioms_file, chainer_dir
    ):
        """petta exits 0 while the chainer import failed silently -> probe fails."""
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, CHAINER_NOT_LOADED_RUN),
        )
        with pytest.raises(RuntimeError, match="PeTTaChainer did not load"):
            pln.initialize()
        assert pln.is_available is False

    def test_interpreter_crash_detected(self, tmp_path, axioms_file, chainer_dir):
        """petta crashing (e.g. import errors under newer PeTTa) fails the probe."""
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, CHAINER_IMPORT_CRASH_RUN, returncode=2),
        )
        with pytest.raises(RuntimeError, match="startup probe failed"):
            pln.initialize()
        assert pln.is_available is False

    def test_no_stv_detected(self, tmp_path, axioms_file, chainer_dir):
        """Query ran (marker present) but produced no STV -> probe fails."""
        empty_run = PRE_QUERY_CHATTER + AXIOM_RECORDS + HYPOTHESIS_RECORDS + QUERY_MARKER
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, empty_run),
        )
        with pytest.raises(RuntimeError, match="no STV in PeTTa output"):
            pln.initialize()
        assert pln.is_available is False


class TestEnvironmentOverrides:
    def test_env_bin_chainer_and_workspace(
        self, tmp_path, axioms_file, monkeypatch
    ):
        petta_bin = make_petta_stub(tmp_path, successful_run())
        chainer = tmp_path / "custom_chainer"
        chainer.mkdir()
        (chainer / "petta_chainer.metta").write_text("", encoding="utf-8")
        monkeypatch.setenv("PETTA_BIN", str(petta_bin))
        monkeypatch.setenv("PETTA_CHAINER_DIR", str(chainer))
        monkeypatch.setenv("PETTA_WORKSPACE", str(tmp_path / "custom_workspace"))

        pln = PLNInference(axioms_path=axioms_file)
        assert pln.petta_bin == str(petta_bin)
        assert pln.chainer_dir == chainer
        assert pln.workspace_dir == tmp_path / "custom_workspace"
        assert (tmp_path / "custom_workspace" / "petta_chainer.metta").is_symlink()
        assert pln.evaluate("P").is_fallback is False
    def test_explicit_args_win_over_env(
        self, tmp_path, axioms_file, monkeypatch
    ):
        monkeypatch.setenv("PETTA_CHAINER_DIR", str(tmp_path / "env_chainer"))
        pln = make_pln(
            tmp_path, axioms_file, tmp_path / "arg_chainer", petta_bin=None
        )
        assert pln.chainer_dir == Path(tmp_path / "arg_chainer")
