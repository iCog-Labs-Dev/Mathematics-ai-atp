"""Target-bound rule-proof validation for PeTTaChainer answer statements."""

from __future__ import annotations
import math
from maths_ai.lean_replay.sexp import parse_sexp_string


def has_rule_proof(proof) -> bool:
    if not isinstance(proof, list):
        return False
    return (len(proof) >= 2 and proof[0] == "rule-proof") or any(
        has_rule_proof(v) for v in proof
    )


def rule_answers(output: str, target: str) -> list[tuple[float, float]]:
    expected = parse_sexp_string(target)
    answers = []
    # Supported PeTTa prints returned statements on individual lines. Ignore
    # diagnostic lines; never borrow a proof marker or score from another line.
    for line in output.splitlines():
        if len(line) > 1_000_000:
            continue
        try:
            statement = parse_sexp_string(line.strip())
        except (ValueError, RecursionError):
            continue
        if (
            not isinstance(statement, list)
            or len(statement) != 4
            or statement[0] != ":"
            or statement[2] != expected
        ):
            continue
        try:
            proof_ok = has_rule_proof(statement[1])
        except RecursionError:
            continue
        if not proof_ok:
            continue
        stv = statement[3]
        if not isinstance(stv, list) or len(stv) != 3 or stv[0] != "STV":
            continue
        try:
            pair = (float(str(stv[1])), float(str(stv[2])))
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(v) and 0 <= v <= 1 for v in pair):
            answers.append(pair)
    return answers
