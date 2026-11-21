from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .dataset import DatasetRow
from .graph import DAGBuilder, proof_state_to_dag
from .labels import label_example
from .state import ProofState, parse_state


PREPARATION_PHASES = ("parse_state", "proof_state_to_dag", "label_example")


@dataclass(frozen=True)
class PreparedExample:
    row: DatasetRow
    parsed_state: ProofState
    dag: DAGBuilder
    tactic_name: str


class PreparationPhaseError(Exception):
    def __init__(self, *, phase: str, cause: Exception):
        self.phase = phase
        self.cause = cause
        super().__init__(str(cause))


class SExprCache:
    """Disk cache for validated source-invocation S-expression records."""

    SCHEMA_VERSION = 4
    EXTRACTOR_VERSION = "source-invocation-v4"

    @staticmethod
    def row_state_sha256(row: DatasetRow) -> str:
        return hashlib.sha256(row.state.encode("utf-8")).hexdigest()

    @staticmethod
    def row_tactic_sha256(row: DatasetRow) -> str:
        return hashlib.sha256(row.tactic.encode("utf-8")).hexdigest()

    @staticmethod
    def row_target_state_sha256(row: DatasetRow) -> str:
        return hashlib.sha256(row.target_state.encode("utf-8")).hexdigest()

    def __init__(
        self,
        output_root: Path,
        project_path: str,
        enabled: bool = True,
    ):
        self.output_root = Path(output_root)
        self.project_path = project_path
        self.enabled = enabled

    def _sexpr_dir(self, split: str) -> Path:
        d = self.output_root / split / "sexpr"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load(self, split: str, row_index: int) -> Optional[dict]:
        sexpr_file = self._sexpr_dir(split) / f"{row_index:09d}.json"
        if sexpr_file.exists():
            try:
                with open(sexpr_file, encoding="utf-8") as f:
                    payload = json.load(f)
                if payload.get("schema_version") != self.SCHEMA_VERSION:
                    return None
                return payload
            except Exception:
                pass
        return None

    def save(self, split: str, row_index: int, data: dict) -> None:
        sexpr_file = self._sexpr_dir(split) / f"{row_index:09d}.json"
        temporary = sexpr_file.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sexpr_file)

    def load_for_row(
        self,
        row: DatasetRow,
        *,
        step_index: int | None = None,
        extractor_version: str | None = None,
    ) -> Optional[dict]:
        payload = self.load(row.split, row.row_index)
        if payload is None:
            return None
        expected = {
            "dataset": row.dataset_name,
            "split": row.split,
            "row_index": row.row_index,
            "theorem": row.theorem,
            "state_sha256": self.row_state_sha256(row),
            "tactic_sha256": self.row_tactic_sha256(row),
            "target_state_sha256": self.row_target_state_sha256(row),
            "repo_commit": row.repo_commit,
            "file_path": row.file_path,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return None
        if step_index is not None and payload.get("step_index") != step_index:
            return None
        if (
            extractor_version is not None
            and payload.get("extractor_version") != extractor_version
        ):
            return None
        return payload


class SExprUnavailableError(RuntimeError):
    """Raised when strict S-expression mode has no validated cache record."""


def _fix_pp_for_goal_start(pp: str) -> str:
    """Preprocess a pretty-printed Lean 4 type so goal_start can parse it.

    Fixes:
    - Universe variables: ``Type u_1`` / ``Type u₁`` → ``Type``
    - Sort expressions: ``Sort (+ u_1 1)`` → ``Type``
    - Grouped binders: ``{a b : T}`` → ``{a : T} {b : T}``
    - Universe-polymorphic instances: ``Category.{v_1, u_1} C`` → ``Category C``
    """
    s = pp.strip()

    # Subscript unicode digits: u₁ → u1
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    s = s.translate(subscript_map)

    # Type u_1, Type u1, Sort u_1 etc → Type
    s = re.sub(r"(?:Type|Sort)\s+u_?\d+", "Type", s)
    s = re.sub(r"Sort\s*\(\+\s*u_?\d+\s+\d+\)", "Type", s)

    # Universe-polymorphic instances: Category.{v_1, u_1} C → Category C
    s = re.sub(r"\.\{[^}]*\}", "", s)

    def _split_grouped(m: re.Match) -> str:
        content = m.group(1).strip()
        close = m.group(0)[0]
        other_close = "}" if close == "{" else ")"
        if " : " in content:
            names_part, type_part = content.split(" : ", 1)
        elif ":" in content:
            names_part, type_part = content.split(":", 1)
        else:
            return m.group(0)
        names = names_part.split()
        if len(names) <= 1:
            return m.group(0)
        return " ".join(
            f"{close}{n} : {type_part.strip()}{other_close}" for n in names
        )

    s = re.sub(r"\{([^{}]+)\}", _split_grouped, s)
    s = re.sub(r"\(([^()]+)\)", _split_grouped, s)
    return s


def prepare_example(
    row: DatasetRow,
    *,
    sexpr_cache: Optional[SExprCache] = None,
    sexpr_data: Optional[dict] = None,
    use_sexpr: bool = False,
) -> PreparedExample:
    """Prepare a single example with phase-specific error reporting.

    If ``sexpr_data`` is provided, uses it for S-expression DAG construction.
    Otherwise falls back to the text parser.
    """
    try:
        parsed_state = parse_state(row.state)
    except Exception as exc:
        raise PreparationPhaseError(phase="parse_state", cause=exc) from exc

    goal_sexp = None
    hyp_sexps = None

    if use_sexpr and sexpr_data is not None:
        goal_sexp = sexpr_data.get("goal_sexp")
        hyp_sexps = [
            (item["name"], item["sexp"]) for item in sexpr_data.get("hyp_sexps", [])
        ]
    elif use_sexpr and sexpr_cache is not None and sexpr_cache.enabled:
        cached = sexpr_cache.load_for_row(row)
        if cached is not None:
            goal_sexp = cached.get("goal_sexp")
            hyp_sexps = [
                (item["name"], item["sexp"]) for item in cached.get("hyp_sexps", [])
            ]

    if use_sexpr and (goal_sexp is None or hyp_sexps is None):
        raise SExprUnavailableError(
            f"No validated S-expression cache record for "
            f"{row.split} row {row.row_index} ({row.theorem})."
        )

    try:
        if goal_sexp is not None or hyp_sexps is not None:
            dag = proof_state_to_dag(
                parsed_state,
                goal_sexp=goal_sexp,
                hyp_sexps=hyp_sexps,
            )
        else:
            dag = proof_state_to_dag(parsed_state)
    except Exception as exc:
        raise PreparationPhaseError(phase="proof_state_to_dag", cause=exc) from exc

    try:
        label_info = label_example(row.tactic)
    except Exception as exc:
        raise PreparationPhaseError(phase="label_example", cause=exc) from exc
    tactic_name = str(label_info["tactic_name"])

    return PreparedExample(
        row=row,
        parsed_state=parsed_state,
        dag=dag,
        tactic_name=tactic_name,
    )
