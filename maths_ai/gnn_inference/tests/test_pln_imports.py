"""Import configuration, live/batch parity, and native diagnostic evidence."""

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from maths_ai.pln_inference.dependencies import (
    AXIOMS_MODULE,
    CHAINER_MODULE,
    PLNConfigurationError,
    PLNDependencies,
    module_import,
)
from maths_ai.pln_inference.diagnose import diagnose, proof_contains, query_answers
from maths_ai.pln_inference.model import PLNInference


@pytest.fixture
def installation(tmp_path):
    root = tmp_path / "chainer"
    for relative in (CHAINER_MODULE, AXIOMS_MODULE):
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("; test module\n")
    binary = tmp_path / "petta"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return root, binary


def test_fork_defaults_and_extensionless_imports(installation, monkeypatch):
    root, _ = installation
    monkeypatch.setenv("PETTACHAINER_ROOT", str(root))
    resolved = PLNDependencies.resolve()
    assert resolved.axioms_file == root / AXIOMS_MODULE
    assert resolved.imports() == (
        f"!(import! &self {(root / CHAINER_MODULE).with_suffix('')})\n\n"
        f"!(import! &self {(root / AXIOMS_MODULE).with_suffix('')})"
    )
    assert "git-import!" not in resolved.imports()
    monkeypatch.setenv("PETTACHAINER_ROOT", "/does/not/exist")
    assert PLNDependencies.resolve(root).chainer_root == root


def test_missing_dependencies_and_unsupported_paths(
    installation, monkeypatch, tmp_path
):
    root, _ = installation
    monkeypatch.delenv("PETTACHAINER_ROOT", raising=False)
    with pytest.raises(PLNConfigurationError, match="PETTACHAINER_ROOT"):
        PLNDependencies.resolve()
    (root / AXIOMS_MODULE).unlink()
    with pytest.raises(PLNConfigurationError, match="metamath_axioms"):
        PLNDependencies.resolve(root)
    unsupported = tmp_path / "space in module.metta"
    unsupported.write_text("")
    with pytest.raises(PLNConfigurationError, match="module paths"):
        module_import(unsupported)


def test_explicit_axioms_override(installation, tmp_path):
    root, _ = installation
    override = tmp_path / "rules.metta"
    override.write_text("; alternate rules\n")
    assert PLNDependencies.resolve(root, override).axioms_file == override


def test_live_source_uses_shared_imports(installation):
    root, binary = installation
    model = PLNInference(chainer_root=root, petta_bin=str(binary))
    source = model._render_query_source("Q", ["P", "P → Q"])
    assert source.startswith(PLNDependencies.resolve(root).imports())
    assert "!(query 10 kb (: $prf (Q) $tv))" in source
    with patch("maths_ai.pln_inference.model.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(
            [], 2, "partial output", "Lean import failed"
        )
        result = model.evaluate("Q", ["P"])
    assert result.status == "engine_error" and result.is_fallback
    assert "code 2" in result.raw_output and "Lean import failed" in result.raw_output
    generated = Path(run.call_args.args[0][1])
    assert not generated.exists()


def test_timeout_preserves_diagnostics(installation):
    root, binary = installation
    model = PLNInference(chainer_root=root, petta_bin=str(binary))
    with patch(
        "maths_ai.pln_inference.model.subprocess.run",
        side_effect=subprocess.TimeoutExpired(
            "petta", 1, output=b"partial stdout", stderr=b"partial stderr"
        ),
    ):
        result = model.evaluate("P")
    assert result.status == "timeout"
    assert (
        "partial stdout" in result.raw_output and "partial stderr" in result.raw_output
    )


def test_configuration_errors_are_not_random_scores(installation, monkeypatch):
    root, _ = installation
    monkeypatch.delenv("PETTACHAINER_ROOT", raising=False)
    with pytest.raises(PLNConfigurationError):
        PLNInference().evaluate("P")
    with pytest.raises(PLNConfigurationError, match="executable"):
        PLNInference(chainer_root=root, petta_bin="/missing/petta").evaluate("P")


@pytest.mark.parametrize("complete", [True, False])
def test_both_batch_paths_use_shared_imports(installation, tmp_path, complete):
    from maths_ai.pln_inference.metta.translator.translator_modules import cli

    root, _ = installation
    input_file = tmp_path / "input.json"
    input_file.write_text(
        json.dumps([{"name": "example", "goal": "True", "tactics": []}])
    )

    class Environment:
        def verify(self):
            pass

    class Client:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_):
            pass

    with (
        patch.object(
            cli, "create_model_sexpr_client", AsyncMock(return_value=Client())
        ),
        patch.object(
            cli,
            "proof_state_after_tactics",
            AsyncMock(return_value=[] if complete else [object()]),
        ),
        patch.object(
            cli,
            "essentialize_subgoal",
            return_value=("!(query 10 kb (: $prf (P) $tv))", "", [], "(P)"),
        ),
    ):
        asyncio.run(
            cli.run_demo(
                str(input_file),
                str(tmp_path / "output"),
                env=Environment(),
                chainer_root=root,
            )
        )
    files = list((tmp_path / "output").glob("*.metta"))
    assert len(files) == 1
    assert files[0].read_text().startswith(PLNDependencies.resolve(root).imports())
    assert "git-import!" not in files[0].read_text()


def test_diagnostic_rejects_echoed_facts_wrong_targets_and_invalid_scores():
    output = "\n".join(
        [
            "(: (kb MAIN Nil) fact (P) (STV 1 1))",
            "(rules (: rule (P) (STV 1 1)))",
            "(: wrong (Q) (STV 1 1))",
            "(: invalid (P) (STV nan 1))",
            "(: invalid (P) (STV 2 1))",
            "(: (rule-proof imported fact) (P) (STV 1 0.9))",
        ]
    )
    answers = query_answers(output, "(P)")
    assert len(answers) == 1
    assert answers[0]["stv"] == [1, 0.9]
    assert proof_contains(answers[0]["proof"], "imported")
    assert not proof_contains(answers[0]["proof"], "import")


def test_diagnostic_failure_records_and_no_overwrite(installation, tmp_path):
    root, binary = installation
    evidence = tmp_path / "evidence"
    # The executable intentionally returns no answers. Positive controls must fail.
    summary = diagnose(evidence, chainer_root=root, petta_bin=str(binary))
    assert not summary["passed"]
    assert len(summary["cases"]) == 6
    assert json.loads((evidence / "summary.json").read_text()) == summary
    assert len(list(evidence.glob("*.stderr.log"))) == 6
    with pytest.raises(FileExistsError):
        diagnose(evidence, chainer_root=root, petta_bin=str(binary))
