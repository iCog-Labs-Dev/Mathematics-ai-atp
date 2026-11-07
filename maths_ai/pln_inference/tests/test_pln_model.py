"""Tests for :mod:`maths_ai.pln_inference.model` (PeTTaChainer wrapper).

Uses stub ``petta`` executables so no real PeTTa install is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maths_ai.data_models.proof_components import STV
from maths_ai.pln_inference.model import PLNInference


# Representative petta output: axioms compiled, the query compiled to a prolog
# findall goal (proof of chainer engagement), real STVs in the log.
SUCCESSFUL_RUN = (
    "(rules (((: (kb $0 $1) $2 (P) $3) (CPU MpFormula ((STV 1.0 1.0) $3) $4)) |"
    "- (: (kb $0 $1) ($2 rule) (Q) $4)))\n"
    "(: (kb MAIN Nil) probe-hyp (P) (STV 1.0 1.0))\n"
    ":- findall(A, query(10, kb, [:, _, ['P'], _], A), _).\n"
)

# A run where the chainer import failed: the query remains an unevaluated
# echo — no `findall(A, query(...))` prolog goal appears.
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
            petta_bin=make_petta_stub(tmp_path, SUCCESSFUL_RUN),
        )
        assert pln._workspace_entries == {}
        # petta still runs (evaluate succeeds) — probe coverage is separate.
        assert pln.evaluate("P").is_fallback is False


class TestEvaluate:
    def test_successful_run_is_real_result(self, tmp_path, axioms_file, chainer_dir):
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, SUCCESSFUL_RUN),
        )
        result = pln.evaluate("P")
        assert result.is_fallback is False
        assert result.status == "ok"
        assert result.stv == STV(strength=1.0, confidence=1.0)

    def test_identical_requests_are_consistent(self, tmp_path, axioms_file, chainer_dir):
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, SUCCESSFUL_RUN),
        )
        first, second = pln.evaluate("P"), pln.evaluate("P")
        assert (first.status, first.stv == second.stv) == ("ok", True)


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
            petta_bin=make_petta_stub(tmp_path, SUCCESSFUL_RUN, returncode=2),
        )
        result = pln.evaluate("P")
        assert result.is_fallback is True
        assert result.stv.score == 0.0
        assert pln.is_available is False


class TestInitialize:
    def test_happy_path(self, tmp_path, axioms_file, chainer_dir):
        pln = make_pln(
            tmp_path, axioms_file, chainer_dir,
            petta_bin=make_petta_stub(tmp_path, SUCCESSFUL_RUN),
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
        empty_run = ":- findall(A, query(10, kb, [:, _, ['P'], _], A), _).\n"
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
        petta_bin = make_petta_stub(tmp_path, SUCCESSFUL_RUN)
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
