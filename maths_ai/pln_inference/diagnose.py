"""Native import verification: python -m maths_ai.pln_inference.diagnose."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from maths_ai.lean_replay.sexp import parse_sexp_string

from .dependencies import PLNDependencies, module_import, resolve_petta


def query_answers(output: str, target: str) -> list[dict]:
    """Read returned statements for this target, excluding compilation output."""
    expected = parse_sexp_string(target)
    answers = []
    for line in output.splitlines():
        try:
            statement = parse_sexp_string(line.strip())
        except ValueError:
            continue
        if not isinstance(statement, list) or len(statement) != 4:
            continue
        if statement[0] != ":" or statement[2] != expected:
            continue
        truth = statement[3]
        if not isinstance(truth, list) or len(truth) != 3 or truth[0] != "STV":
            continue
        try:
            values = [float(item) for item in truth[1:]]
        except (ValueError, TypeError):
            continue
        if all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            answers.append({"proof": statement[1], "stv": values})
    return answers


def proof_contains(proof: object, symbol: str) -> bool:
    if isinstance(proof, list):
        return any(proof_contains(item, symbol) for item in proof)
    return proof == symbol


def _decode(value: str | bytes | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")


def _revision(root: Path | None) -> str | None:
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def diagnose(
    output: Path,
    *,
    chainer_root: Path | None = None,
    axioms_path: Path | None = None,
    petta_bin: str | None = None,
    petta_source_root: Path | None = None,
    timeout: float = 60,
) -> dict:
    dependencies = PLNDependencies.resolve(chainer_root, axioms_path)
    executable = resolve_petta(petta_bin)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    output = output.expanduser().resolve()
    # The control axiom is itself imported by absolute path.
    if any(char.isspace() for char in str(output)):
        raise ValueError("Diagnostic output directory must not contain whitespace")
    output.mkdir(parents=True, exist_ok=False)
    control = output / "control_axiom.metta"
    control.write_text(
        "!(compileadd kb (: diagnostic-import-rule "
        "(Implication (Premises (DiagnosticPremise)) "
        "(Conclusions (DiagnosticConclusion))) (STV 1.0 1.0)))\n",
        encoding="utf-8",
    )
    core = dependencies.imports(include_axioms=False)
    premise = "!(compileadd kb (: diagnostic-fact (DiagnosticPremise) (STV 1.0 1.0)))"
    identity = "(Implication (Premises (DiagnosticP)) (Conclusions (DiagnosticP)))"
    # Both full-collection and no-collection runs may prove identity. The
    # distinguishing evidence is the imported proof identifiers, not provability.
    cases = [
        (
            "core_implication",
            core,
            "(DiagnosticConclusion)",
            premise + "\n" + control.read_text(),
            "diagnostic-import-rule",
            True,
        ),
        (
            "imported_control",
            core + "\n" + module_import(control),
            "(DiagnosticConclusion)",
            premise,
            "diagnostic-import-rule",
            True,
        ),
        (
            "without_control",
            core,
            "(DiagnosticConclusion)",
            premise,
            "diagnostic-import-rule",
            False,
        ),
        ("fork_axioms", dependencies.imports(), identity, "", "pm2.18", True),
        ("without_fork_axioms", core, identity, "", "pm2.18", False),
        (
            "conjunction_observation",
            dependencies.imports(),
            "(∧ (DiagnosticP) (DiagnosticQ))",
            "!(compileadd kb (: diagnostic-p (DiagnosticP) (STV 1.0 1.0)))\n"
            "!(compileadd kb (: diagnostic-q (DiagnosticQ) (STV 1.0 1.0)))",
            None,
            None,
        ),
    ]
    summary = {
        "chainer_revision": _revision(dependencies.chainer_root),
        "petta_revision": _revision(petta_source_root),
        "chainer": _identity(dependencies.chainer_file),
        "axioms": _identity(dependencies.axioms_file),
        "executable": _identity(Path(executable)),
        "control_axiom": _identity(control),
        "timeout_s": timeout,
        "cases": [],
    }
    for name, imports, target, facts, symbol, required in cases:
        source = output / f"{name}.metta"
        source.write_text(
            f"{imports}\n{facts}\n!(query 10 kb (: $prf {target} $tv))\n",
            encoding="utf-8",
        )
        command = [executable, str(source)]
        returncode, error = None, None
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout
            )
            stdout, stderr, returncode = (
                completed.stdout,
                completed.stderr,
                completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, error = _decode(exc.stdout), _decode(exc.stderr), "timeout"
        except OSError as exc:
            stdout, stderr, error = "", str(exc), "launch-error"
        (output / f"{name}.stdout.log").write_text(stdout, encoding="utf-8")
        (output / f"{name}.stderr.log").write_text(stderr, encoding="utf-8")
        answers = query_answers(stdout, target)
        found = (
            any(proof_contains(answer["proof"], symbol) for answer in answers)
            if symbol
            else None
        )
        passed = (
            returncode == 0
            and error is None
            and (required is None or found == required)
        )
        if name == "without_control":
            passed = passed and not answers
        if name == "fork_axioms":
            passed = passed and any(
                proof_contains(answer["proof"], "pm2.18i") for answer in answers
            )
        summary["cases"].append(
            {
                "name": name,
                "command": command,
                "returncode": returncode,
                "error": error,
                "answers": answers,
                "expected_rule": symbol,
                "expected_rule_present": required,
                "passed": passed,
                "observation_only": required is None,
            }
        )
        summary["passed"] = all(case["passed"] for case in summary["cases"])
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chainer-root", type=Path)
    parser.add_argument("--axioms-path", type=Path)
    parser.add_argument("--petta-bin")
    parser.add_argument("--petta-source-root", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = diagnose(**vars(args))
    for case in summary["cases"]:
        label = (
            "OBSERVATION"
            if case["observation_only"] and case["passed"]
            else ("PASS" if case["passed"] else "FAIL")
        )
        print(f"{case['name']}: {label}; answers={len(case['answers'])}")
    print(f"Evidence: {args.output.resolve() / 'summary.json'}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
