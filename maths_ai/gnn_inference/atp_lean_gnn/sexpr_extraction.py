from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .dataset import DATASET_NAME, DatasetRow, canonicalize_split_name, iter_dataset_rows
from .preparation import (
    ActionTraceCache,
    ModelSExprCache,
    SExprCache,
    SourceSyntaxTraceCache,
)
from .pilot_sampling import selected_extraction_row_indices
from .reporting import console_print
from .state import parse_state


EXTRACTION_VERSION = SExprCache.EXTRACTOR_VERSION
DATASET_MATHLIB_COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
# First fork commit that emitted normalized model S-expressions and elaborated
# action traces.  Recorded for provenance of caches produced before the compact
# tactic-syntax trace existed; those records stay valid.
LEGACY_MODEL_PANTOGRAPH_COMMIT = "e6a8d53165a987d59c5780d2dd287d8ed4c95147"
# Current pin.  It adds the compact annotated tactic-syntax trace and keeps
# every earlier field, so raw and normalized extraction are unaffected.
PANTOGRAPH_COMMIT = "73781c2d58456e4bf369dadd4a501e1b78a0b177"
MODEL_PANTOGRAPH_COMMIT = PANTOGRAPH_COMMIT


@dataclass(frozen=True)
class SExprExtractionConfig:
    prepared_root: Path
    source_root: Path
    pantograph_repl: Path
    dataset_name: str = DATASET_NAME
    splits: tuple[str, ...] = ("train", "val", "test")
    sample_per_split: int | None = None
    resume: bool = True
    expected_commit: str = DATASET_MATHLIB_COMMIT
    expected_pantograph_commit: str = PANTOGRAPH_COMMIT
    server_startup_timeout: int = 120
    file_timeout: int = 600
    buffer_limit: int = 256 * 1024 * 1024
    workers: int = 1
    recycle_worker_files: int = 10
    verify_source_commit: bool = True
    model_sexprs: bool = False
    action_traces: bool = False
    source_syntax_traces: bool = False
    theorem: str | None = None
    selection_manifest: Path | None = None


class SourceExtractionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        phase: str,
        theorem: str = "",
        row_index: int | None = None,
    ) -> None:
        self.phase = phase
        self.theorem = theorem
        self.row_index = row_index
        super().__init__(message)


class PantographInvocationClient:
    """Small client for the Lean-4.10-compatible Pantograph JSON REPL.

    We intentionally do not import the Python Pantograph package: the project's
    regular environment may use another Lean/Pantograph version.  ``lake env``
    selects the exact toolchain recorded by the dataset checkout.
    """

    def __init__(
        self,
        *,
        source_root: Path,
        pantograph_repl: Path,
        startup_timeout: int = 120,
        file_timeout: int = 600,
        buffer_limit: int = 256 * 1024 * 1024,
        capture_model_sexprs: bool = False,
        imports: tuple[str, ...] = ("Init",),
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.pantograph_repl = Path(pantograph_repl).resolve()
        self.startup_timeout = startup_timeout
        self.file_timeout = file_timeout
        self.buffer_limit = buffer_limit
        self.capture_model_sexprs = capture_model_sexprs
        self.imports = imports
        self.proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self.stderr_tail: list[str] = []

    async def start(self) -> "PantographInvocationClient":
        if self.proc is not None:
            return self
        self.proc = await asyncio.create_subprocess_exec(
            "lake",
            "env",
            "stdbuf",
            "-oL",
            str(self.pantograph_repl),
            *self.imports,
            cwd=self.source_root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self.buffer_limit,
        )
        assert self.proc.stdout is not None
        try:
            ready = await asyncio.wait_for(
                self.proc.stdout.readline(), timeout=self.startup_timeout
            )
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError("Pantograph did not become ready in time.") from exc
        if ready.decode("utf-8", errors="replace").strip() != "ready.":
            await self.close()
            raise RuntimeError(f"Pantograph emitted an invalid ready signal: {ready!r}")
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        options = {"printExprAST": True}
        if self.capture_model_sexprs:
            options["printExprModelAST"] = True
        await self.call("options.set", options, timeout=self.startup_timeout)
        return self

    async def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        while line := await self.proc.stderr.readline():
            self.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())
            del self.stderr_tail[:-30]

    async def call(
        self, command: str, payload: dict[str, object], *, timeout: int
    ) -> dict[str, object]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Pantograph client is not running.")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.proc.stdin.write(f"{command} {encoded}\n".encode("utf-8"))
        await self.proc.stdin.drain()
        try:
            raw = await asyncio.wait_for(self.proc.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError(f"Pantograph command '{command}' timed out.") from exc
        if not raw:
            detail = "\n".join(self.stderr_tail[-10:])
            await self.close()
            raise RuntimeError(f"Pantograph exited during '{command}'. {detail}")
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            await self.close()
            raise RuntimeError(f"Pantograph returned invalid JSON for '{command}'.") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Pantograph returned a non-object for '{command}'.")
        if "error" in result:
            detail = json.dumps(result, ensure_ascii=False, sort_keys=True)
            raise RuntimeError(f"Pantograph '{command}' failed: {detail}")
        return result

    async def process_file(self, file_path: str) -> list[dict[str, object]]:
        if self.proc is None:
            await self.start()
        absolute = (self.source_root / file_path).resolve()
        try:
            absolute.relative_to(self.source_root)
        except ValueError as exc:
            raise RuntimeError(f"Dataset file escapes the source checkout: {file_path}") from exc
        descriptor, output_name = tempfile.mkstemp(
            prefix="maths_ai_invocations_", suffix=".json"
        )
        os.close(descriptor)
        output_path = Path(output_name)
        try:
            await self.call(
                "frontend.process",
                {
                    "fileName": str(absolute),
                    "invocations": True,
                    "sorrys": False,
                    "outputFile": str(output_path),
                },
                timeout=self.file_timeout,
            )
            try:
                with output_path.open(encoding="utf-8") as handle:
                    result = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "Pantograph did not write a valid frontend trace."
                ) from exc
            if not isinstance(result, dict):
                raise RuntimeError("Pantograph frontend trace is not an object.")
            units = result.get("units")
            if not isinstance(units, list):
                raise RuntimeError("Pantograph frontend trace has no compilation units.")
            return units
        finally:
            output_path.unlink(missing_ok=True)

    async def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
            try:
                await proc.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
        if proc is not None and proc.returncode is None:
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=2)
            except asyncio.TimeoutError:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=5)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    await proc.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._stderr_task = None


def _normalized_text(text: str) -> str:
    normalized = "\n".join(
        line.rstrip()
        for line in text.replace("\r\n", "\n").strip().splitlines()
        if line.strip()
    )
    if normalized.lower() in {"no goals", "no goals to be solved"}:
        return ""
    return normalized


def _strip_lean_comments(text: str) -> str:
    """Remove Lean comments without treating comment markers in strings as syntax."""
    output: list[str] = []
    index = 0
    block_depth = 0
    in_string = False
    escaped = False
    while index < len(text):
        if block_depth:
            if text.startswith("/-", index):
                block_depth += 1
                index += 2
            elif text.startswith("-/", index):
                block_depth -= 1
                index += 2
            else:
                if text[index] == "\n":
                    output.append("\n")
                index += 1
            continue

        character = text[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
        elif text.startswith("--", index):
            newline = text.find("\n", index + 2)
            if newline < 0:
                break
            output.append("\n")
            index = newline + 1
        elif text.startswith("/-", index):
            block_depth = 1
            index += 2
        else:
            output.append(character)
            if character == '"':
                in_string = True
            index += 1
    return "".join(output)


def _undecorated_tactic(text: str) -> str:
    # LeanDojo adds HTML-like links around referenced declarations, and the
    # link text may be fully qualified even when Lean's tactic pretty-printer
    # chooses the short name. These decorations are not part of Lean syntax.
    text = _strip_lean_comments(text)
    text = re.sub(r"</?a(?:\s[^>]*)?>", "", text)
    return re.sub(
        r"(?<![\w'])(?:[\w'][\w'!?]*\.)+([\w'][\w'!?]*)",
        r"\1",
        text,
    )


def _normalized_tactic(text: str) -> str:
    text = _undecorated_tactic(text)
    return " ".join(text.split())


def _branch_opener_tactic(text: str) -> str | None:
    """Return the source line before an attached branch body, if present.

    Lean's info tree can associate a tactic such as ``by_cases h`` with all
    following ``·`` branches. LeanDojo records the opener as its own action.
    The opener's input goal is still authentic, but its info-tree after-state
    is the state after the attached branches rather than immediately after the
    opener.
    """
    lines = _undecorated_tactic(text).strip().splitlines()
    if len(lines) < 2:
        return None
    for index, line in enumerate(lines[1:], 1):
        stripped = line.lstrip()
        if stripped.startswith("·") or re.match(r"(?:case|next)\b", stripped):
            head = " ".join(part.strip() for part in lines[:index] if part.strip())
            return " ".join(head.split()) or None
    return None


def _expanded_hypothesis_names(state: str) -> list[str]:
    names: list[str] = []
    for hypothesis in parse_state(state).hypotheses:
        names.extend(hypothesis.name.split() or [hypothesis.name])
    return names


def _group_rows_by_file(rows: Iterable[DatasetRow]) -> dict[str, dict[str, list[DatasetRow]]]:
    grouped: dict[str, dict[str, list[DatasetRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row.file_path:
            raise SourceExtractionError(
                "Dataset row has no file_path metadata.",
                phase="source_metadata",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        if not row.repo_commit:
            raise SourceExtractionError(
                "Dataset row has no commit metadata.",
                phase="source_metadata",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        grouped[row.file_path][row.theorem].append(row)
    return {
        file_path: {
            theorem: sorted(theorem_rows, key=lambda item: item.row_index)
            for theorem, theorem_rows in theorem_groups.items()
        }
        for file_path, theorem_groups in grouped.items()
    }


def _unit_source(source_bytes: bytes, unit: dict[str, object]) -> str:
    boundary = unit.get("boundary")
    if not isinstance(boundary, list) or len(boundary) != 2:
        return ""
    try:
        return source_bytes[int(boundary[0]) : int(boundary[1])].decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""


def _unit_declares_theorem(source: str, theorem: str) -> bool:
    if not theorem:
        return False
    short_name = theorem.rsplit(".", 1)[-1].strip("«»")
    declaration = re.compile(
        rf"\b(?:theorem|lemma)\s+(?:[A-Za-z0-9_'.]+\.)?«?{re.escape(short_name)}»?(?=\s|[:(])"
    )
    return declaration.search(source) is not None


def _candidate_invocations(
    *, source_bytes: bytes, units: list[dict[str, object]], theorem: str
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    ordinal = 0
    for unit_index, unit in enumerate(units):
        source = _unit_source(source_bytes, unit)
        invocations = unit.get("invocations") or []
        if not isinstance(invocations, list):
            continue
        is_theorem_unit = _unit_declares_theorem(source, theorem)
        for invocation_index, invocation in enumerate(invocations):
            if is_theorem_unit and isinstance(invocation, dict):
                candidates.append(
                    {
                        "ordinal": ordinal,
                        "unit_index": unit_index,
                        "invocation_index": invocation_index,
                        "unit_boundary": unit.get("boundary"),
                        "invocation": invocation,
                    }
                )
            ordinal += 1
    return candidates


def _invocation_match_kind(
    row: DatasetRow, candidate: dict[str, object]
) -> str | None:
    invocation = candidate["invocation"]
    assert isinstance(invocation, dict)
    if _normalized_text(str(invocation.get("goalBefore", ""))) != _normalized_text(row.state):
        return None

    invocation_tactic = str(invocation.get("tactic", ""))
    row_tactic = _normalized_tactic(row.tactic)
    target_matches = _normalized_text(
        str(invocation.get("goalAfter", ""))
    ) == _normalized_text(row.target_state)
    if _normalized_tactic(invocation_tactic) == row_tactic and target_matches:
        return "exact"
    if _branch_opener_tactic(invocation_tactic) == row_tactic:
        return "branch_opener"
    return None


def _capture_signature(candidate: dict[str, object]) -> str:
    """Identify candidates that provide exactly the same model input."""
    invocation = candidate["invocation"]
    assert isinstance(invocation, dict)
    payload = {
        "capture_error": invocation.get("captureError"),
        "goal_before": _normalized_text(str(invocation.get("goalBefore", ""))),
        "goals_before": invocation.get("goalsBefore"),
        "terms": invocation.get("terms"),
        "syntax_args": invocation.get("syntaxArgs"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _align_row(
    row: DatasetRow, candidates: list[dict[str, object]]
) -> dict[str, object]:
    """Return the one invocation that reproduces this row, or raise."""
    matches = [
        (item, kind)
        for item in candidates
        if (kind := _invocation_match_kind(row, item)) is not None
    ]
    if not matches:
        raise SourceExtractionError(
            "No original tactic invocation matches state + tactic + target_state.",
            phase="invocation_alignment",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    exact = [(item, kind) for item, kind in matches if kind == "exact"]
    preferred = exact or matches
    usable = [
        (item, kind)
        for item, kind in preferred
        if not (
            isinstance(item["invocation"], dict)
            and item["invocation"].get("captureError")
        )
    ]
    preferred = usable or preferred

    signatures = {_capture_signature(item) for item, _kind in preferred}
    if len(signatures) > 1:
        raise SourceExtractionError(
            "More than one invocation with a different serialized input goal "
            "matches this dataset row; refusing a guessed cache.",
            phase="ambiguous_invocation",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    selected, match_kind = min(preferred, key=lambda pair: int(pair[0]["ordinal"]))
    selected = dict(selected)
    selected["alignment_kind"] = match_kind
    return selected


def _align_rows(
    rows: list[DatasetRow], candidates: list[dict[str, object]]
) -> list[dict[str, object] | SourceExtractionError]:
    """Align every row independently, returning the error for those that fail.

    A row that matches no invocation, or matches two that disagree, says nothing
    about its neighbours, so the failure is returned instead of raised and the
    caller keeps the rows that did align.
    """
    aligned: list[dict[str, object] | SourceExtractionError] = []
    for row in rows:
        try:
            aligned.append(_align_row(row, candidates))
        except SourceExtractionError as error:
            aligned.append(error)
    return aligned


def _make_record(row: DatasetRow, candidate: dict[str, object]) -> dict[str, object]:
    invocation = candidate["invocation"]
    assert isinstance(invocation, dict)
    capture_error = invocation.get("captureError")
    if capture_error:
        raise SourceExtractionError(
            f"Pantograph could not serialize this tactic's proof state: {capture_error}",
            phase="sexpr_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    goals = invocation.get("goalsBefore")
    if not isinstance(goals, list) or len(goals) != 1:
        raise SourceExtractionError(
            f"Invocation has {len(goals) if isinstance(goals, list) else 'invalid'} active goals; expected one.",
            phase="goal_cardinality",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    goal = goals[0]
    if not isinstance(goal, dict):
        raise SourceExtractionError("Serialized goal is invalid.", phase="sexpr_capture", theorem=row.theorem)
    target = goal.get("target")
    variables = goal.get("vars")
    if not isinstance(target, dict) or not isinstance(target.get("sexp"), str):
        raise SourceExtractionError("Goal S-expression is missing.", phase="sexpr_capture", theorem=row.theorem)
    if not isinstance(variables, list):
        raise SourceExtractionError("Goal variables are missing.", phase="sexpr_capture", theorem=row.theorem)

    hyp_sexps: list[dict[str, str]] = []
    actual_names: list[str] = []
    for variable in variables:
        if not isinstance(variable, dict) or not isinstance(variable.get("type"), dict):
            raise SourceExtractionError("Variable serialization is invalid.", phase="sexpr_capture", theorem=row.theorem)
        sexp = variable["type"].get("sexp")
        if not isinstance(sexp, str):
            raise SourceExtractionError("Variable type S-expression is missing.", phase="sexpr_capture", theorem=row.theorem)
        name = str(variable.get("userName", "_"))
        actual_names.append(name)
        hyp_sexps.append({"name": name, "sexp": sexp})

    expected_names = _expanded_hypothesis_names(row.state)
    return {
        "schema_version": SExprCache.SCHEMA_VERSION,
        "extractor_version": EXTRACTION_VERSION,
        "dataset": row.dataset_name,
        "split": row.split,
        "row_index": row.row_index,
        "theorem": row.theorem,
        "repo_url": row.repo_url,
        "repo_commit": row.repo_commit,
        "pantograph_commit": PANTOGRAPH_COMMIT,
        "file_path": row.file_path,
        "unit_index": candidate["unit_index"],
        "unit_boundary": candidate["unit_boundary"],
        "invocation_index": candidate["invocation_index"],
        "alignment_kind": candidate.get("alignment_kind", "exact"),
        "target_state_matches_invocation": _normalized_text(
            str(invocation.get("goalAfter", ""))
        )
        == _normalized_text(row.target_state),
        "state_sha256": SExprCache.row_state_sha256(row),
        "tactic_sha256": SExprCache.row_tactic_sha256(row),
        "target_state_sha256": SExprCache.row_target_state_sha256(row),
        "tactic": row.tactic,
        "pending_goal_count": len(goals),
        "hypothesis_count": len(hyp_sexps),
        "hypothesis_names_match": actual_names == expected_names,
        "goal_sexp": target["sexp"],
        "hyp_sexps": hyp_sexps,
        "text_state": str(invocation.get("goalBefore", "")),
        "text_target_state": str(invocation.get("goalAfter", "")),
    }


def _make_model_record(
    row: DatasetRow,
    candidate: dict[str, object],
    raw_record: dict[str, object],
) -> dict[str, object]:
    invocation = candidate["invocation"]
    assert isinstance(invocation, dict)
    goals = invocation.get("goalsBefore")
    if not isinstance(goals, list) or len(goals) != 1:
        raise SourceExtractionError(
            "Normalized capture requires exactly one active goal.",
            phase="model_goal_cardinality",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    goal = goals[0]
    if not isinstance(goal, dict):
        raise SourceExtractionError(
            "Normalized goal is invalid.",
            phase="model_sexpr_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    target = goal.get("target")
    variables = goal.get("vars")
    if (
        not isinstance(target, dict)
        or not isinstance(target.get("modelSexp"), str)
        or target.get("modelSexpVersion") != ModelSExprCache.EXPRESSION_VERSION
    ):
        raise SourceExtractionError(
            "Goal model S-expression is missing or has an unsupported version.",
            phase="model_sexpr_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    if not isinstance(variables, list):
        raise SourceExtractionError(
            "Normalized goal variables are missing.",
            phase="model_sexpr_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        )

    hypotheses: list[dict[str, str]] = []
    for variable in variables:
        variable_type = variable.get("type") if isinstance(variable, dict) else None
        if (
            not isinstance(variable, dict)
            or not isinstance(variable_type, dict)
            or not isinstance(variable_type.get("modelSexp"), str)
            or variable_type.get("modelSexpVersion")
            != ModelSExprCache.EXPRESSION_VERSION
        ):
            raise SourceExtractionError(
                "Hypothesis model S-expression is missing or unsupported.",
                phase="model_sexpr_capture",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        context_index = variable.get("contextIndex")
        binder_role = variable.get("binderRole")
        if not isinstance(context_index, int) or not isinstance(binder_role, str):
            raise SourceExtractionError(
                "Hypothesis context index or binder role is missing.",
                phase="model_context_metadata",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        hypotheses.append(
            {
                "name": str(variable.get("userName", "_")),
                "internal_name": str(variable.get("name", "")),
                "context_index": context_index,
                "binder_role": binder_role,
                "is_instance": bool(variable.get("isInstance", False)),
                "is_let": bool(variable.get("isLet", False)),
                "sexp": variable_type["modelSexp"],
            }
        )

    return {
        "schema_version": ModelSExprCache.SCHEMA_VERSION,
        "normalization": ModelSExprCache.NORMALIZATION,
        "dataset": row.dataset_name,
        "split": row.split,
        "row_index": row.row_index,
        "theorem": row.theorem,
        "repo_commit": row.repo_commit,
        "file_path": row.file_path,
        "pantograph_commit": MODEL_PANTOGRAPH_COMMIT,
        "raw_record_sha256": ModelSExprCache.raw_record_sha256(raw_record),
        "goal_sexp": target["modelSexp"],
        "hyp_sexps": hypotheses,
    }


def _make_action_trace_record(
    row: DatasetRow,
    candidate: dict[str, object],
    raw_record: dict[str, object],
) -> dict[str, object]:
    invocation = candidate["invocation"]
    assert isinstance(invocation, dict)
    return {
        "schema_version": ActionTraceCache.SCHEMA_VERSION,
        "extractor_version": ActionTraceCache.EXTRACTOR_VERSION,
        "dataset": row.dataset_name,
        "split": row.split,
        "row_index": row.row_index,
        "theorem": row.theorem,
        "repo_commit": row.repo_commit,
        "file_path": row.file_path,
        "pantograph_commit": PANTOGRAPH_COMMIT,
        "raw_record_sha256": ActionTraceCache.raw_record_sha256(raw_record),
        "state_sha256": SExprCache.row_state_sha256(row),
        "tactic_sha256": SExprCache.row_tactic_sha256(row),
        "target_state_sha256": SExprCache.row_target_state_sha256(row),
        "tactic": row.tactic,
        "terms": _normalized_action_terms(row, invocation, include_action_sexp=True),
        "syntax_args": _normalized_action_syntax_arguments(row, invocation),
        "local_context": _normalized_action_local_context(row, invocation),
    }


def _make_source_syntax_trace_record(
    row: DatasetRow,
    candidate: dict[str, object],
    raw_record: dict[str, object],
) -> dict[str, object]:
    """Build a version-3 sidecar around the compact annotated tactic syntax.

    The elaborated term trees stay in the version-2 sidecars: repeating them here
    would restore exactly the kernel-term blowup this schema exists to avoid, so
    only their source ranges are kept as alignment diagnostics.
    """
    invocation = candidate["invocation"]
    assert isinstance(invocation, dict)
    encoded = invocation.get("sourceSyntax")
    if not isinstance(encoded, str) or not encoded:
        raise SourceExtractionError(
            "Pantograph invocation has no compact tactic-syntax trace.",
            phase="source_syntax_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    try:
        source_syntax = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise SourceExtractionError(
            f"Compact tactic-syntax trace is not valid JSON: {exc}",
            phase="source_syntax_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        ) from exc
    if not isinstance(source_syntax, dict) or not isinstance(
        source_syntax.get("tag"), str
    ):
        raise SourceExtractionError(
            "Compact tactic-syntax trace has no tagged root node.",
            phase="source_syntax_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        )

    return {
        "schema_version": SourceSyntaxTraceCache.SCHEMA_VERSION,
        "extractor_version": SourceSyntaxTraceCache.EXTRACTOR_VERSION,
        "dataset": row.dataset_name,
        "split": row.split,
        "row_index": row.row_index,
        "theorem": row.theorem,
        "repo_commit": row.repo_commit,
        "file_path": row.file_path,
        "pantograph_commit": PANTOGRAPH_COMMIT,
        "raw_record_sha256": SourceSyntaxTraceCache.raw_record_sha256(raw_record),
        "state_sha256": SExprCache.row_state_sha256(row),
        "tactic_sha256": SExprCache.row_tactic_sha256(row),
        "target_state_sha256": SExprCache.row_target_state_sha256(row),
        "tactic": row.tactic,
        "source_syntax": source_syntax,
        "syntax_args": _normalized_action_syntax_arguments(row, invocation),
        "term_ranges": _normalized_action_terms(
            row, invocation, include_action_sexp=False
        ),
        "local_context": _normalized_action_local_context(row, invocation),
    }


def _normalized_action_terms(
    row: DatasetRow,
    invocation: dict[str, object],
    *,
    include_action_sexp: bool,
) -> list[dict[str, object]]:
    terms = invocation.get("terms")
    if not isinstance(terms, list):
        raise SourceExtractionError(
            "Pantograph invocation has no structured tactic-term trace.",
            phase="action_trace_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    normalized_terms: list[dict[str, object]] = []
    for term in terms:
        if not isinstance(term, dict) or not isinstance(term.get("actionSexp"), str):
            raise SourceExtractionError(
                "Pantograph returned an invalid structured tactic term.",
                phase="action_trace_capture",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        start = term.get("sourceStart")
        end = term.get("sourceEnd")
        if not isinstance(start, int) or not isinstance(end, int):
            raise SourceExtractionError(
                "Structured tactic term has no byte range.",
                phase="action_trace_capture",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        normalized: dict[str, object] = {
            "source": str(term.get("source", "")),
            "syntax_kind": str(term.get("syntaxKind", "")),
            "source_start": start,
            "source_end": end,
        }
        if include_action_sexp:
            normalized["action_sexp"] = term["actionSexp"]
        normalized_terms.append(normalized)
    return normalized_terms


def _normalized_action_syntax_arguments(
    row: DatasetRow, invocation: dict[str, object]
) -> list[dict[str, object]]:
    syntax_args = invocation.get("syntaxArgs")
    if not isinstance(syntax_args, list):
        raise SourceExtractionError(
            "Pantograph invocation has no structured syntax-argument trace.",
            phase="action_trace_capture",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    normalized_syntax_args: list[dict[str, object]] = []
    for argument in syntax_args:
        if not isinstance(argument, dict) or not isinstance(argument.get("role"), str):
            raise SourceExtractionError(
                "Pantograph returned an invalid structured syntax argument.",
                phase="action_trace_capture",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        start = argument.get("sourceStart")
        end = argument.get("sourceEnd")
        if not isinstance(start, int) or not isinstance(end, int):
            raise SourceExtractionError(
                "Structured syntax argument has no byte range.",
                phase="action_trace_capture",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        normalized_syntax_args.append(
            {
                "role": argument["role"],
                "source": str(argument.get("source", "")),
                "syntax_kind": str(argument.get("syntaxKind", "")),
                "source_start": start,
                "source_end": end,
            }
        )
    return normalized_syntax_args


def _normalized_action_local_context(
    row: DatasetRow, invocation: dict[str, object]
) -> list[dict[str, object]]:
    goals = invocation.get("goalsBefore")
    if not isinstance(goals, list) or len(goals) != 1 or not isinstance(goals[0], dict):
        raise SourceExtractionError(
            "Action tracing requires exactly one serialized input goal.",
            phase="action_goal_cardinality",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    variables = goals[0].get("vars")
    if not isinstance(variables, list):
        raise SourceExtractionError(
            "Action trace input goal has no local context.",
            phase="action_context_metadata",
            theorem=row.theorem,
            row_index=row.row_index,
        )
    local_context: list[dict[str, object]] = []
    for variable in variables:
        if not isinstance(variable, dict) or not isinstance(
            variable.get("contextIndex"), int
        ):
            raise SourceExtractionError(
                "Action trace local variable has no stable context index.",
                phase="action_context_metadata",
                theorem=row.theorem,
                row_index=row.row_index,
            )
        local_context.append(
            {
                "context_index": variable["contextIndex"],
                "user_name": str(variable.get("userName", "_")),
                "internal_name": str(variable.get("name", "")),
                "binder_role": str(variable.get("binderRole", "")),
                "is_instance": bool(variable.get("isInstance", False)),
                "is_let": bool(variable.get("isLet", False)),
            }
        )
    return local_context


def _row_is_cached(cache: SExprCache, row: DatasetRow) -> bool:
    record = cache.load_for_row(row, extractor_version=EXTRACTION_VERSION)
    return (
        record is not None
        and isinstance(record.get("goal_sexp"), str)
        and isinstance(record.get("hyp_sexps"), list)
    )


def _row_is_model_cached(
    raw_cache: SExprCache, model_cache: ModelSExprCache, row: DatasetRow
) -> bool:
    raw_record = raw_cache.load_for_row(row, extractor_version=EXTRACTION_VERSION)
    if raw_record is None:
        return False
    record = model_cache.load_for_raw_record(row.split, row.row_index, raw_record)
    return (
        record is not None
        and isinstance(record.get("goal_sexp"), str)
        and isinstance(record.get("hyp_sexps"), list)
    )


def _row_is_action_cached(
    raw_cache: SExprCache, action_cache: ActionTraceCache, row: DatasetRow
) -> bool:
    raw_record = raw_cache.load_for_row(row, extractor_version=EXTRACTION_VERSION)
    if raw_record is None:
        return False
    record = action_cache.load_for_raw_record(row.split, row.row_index, raw_record)
    if record is None:
        return False
    if any(
        not isinstance(record.get(field), list)
        for field in action_cache.REQUIRED_LIST_FIELDS
    ):
        return False
    if isinstance(action_cache, SourceSyntaxTraceCache):
        return isinstance(record.get("source_syntax"), dict)
    return True


def _row_targets_cached(
    raw_cache: SExprCache,
    row: DatasetRow,
    *,
    model_cache: ModelSExprCache | None,
    action_cache: ActionTraceCache | None,
) -> bool:
    """True when every cache this run writes already holds a valid record."""
    if not _row_is_cached(raw_cache, row):
        return False
    if model_cache is not None and not _row_is_model_cached(
        raw_cache, model_cache, row
    ):
        return False
    if action_cache is not None and not _row_is_action_cached(
        raw_cache, action_cache, row
    ):
        return False
    return True


def _build_records(
    row: DatasetRow,
    candidate: dict[str, object],
    *,
    raw_cache: SExprCache,
    model_cache: ModelSExprCache | None,
    action_cache: ActionTraceCache | None,
    resume: bool,
) -> list[tuple[object, dict[str, object]]]:
    """Build every enabled record for one row out of one elaboration.

    Pantograph returns all the payloads on every invocation -- the raw
    S-expression, the normalized model one, and the compact annotated syntax --
    so requesting several generations costs one file compile rather than one
    each.

    The raw record is the digest anchor the sidecars are bound to.  It is built
    here when absent and reused when already cached, so a validated raw record
    survives untouched; ``resume=False`` rebuilds it along with everything else,
    which is what re-extraction means.  A sidecar of a generation this run did
    not enable then fails its digest check and is regenerated on its own next
    run, rather than being read as though it still described the raw record.
    """
    pending: list[tuple[object, dict[str, object]]] = []
    raw_record = (
        raw_cache.load_for_row(row, extractor_version=EXTRACTION_VERSION)
        if resume
        else None
    )
    if raw_record is None:
        raw_record = _make_record(row, candidate)
        pending.append((raw_cache, raw_record))
    if model_cache is not None and (
        not resume
        or model_cache.load_for_raw_record(row.split, row.row_index, raw_record) is None
    ):
        pending.append((model_cache, _make_model_record(row, candidate, raw_record)))
    if action_cache is not None and (
        not resume
        or action_cache.load_for_raw_record(row.split, row.row_index, raw_record) is None
    ):
        make_action_record = (
            _make_source_syntax_trace_record
            if isinstance(action_cache, SourceSyntaxTraceCache)
            else _make_action_trace_record
        )
        pending.append((action_cache, make_action_record(row, candidate, raw_record)))
    return pending


def _failure_record(exc: Exception, file_path: str, theorem: str, rows: list[DatasetRow]) -> dict[str, object]:
    return {
        "file_path": file_path,
        "theorem": theorem,
        "split": rows[0].split if rows else None,
        "row_indices": [row.row_index for row in rows],
        "failed_row_count": len(rows),
        "phase": getattr(exc, "phase", "unexpected_error"),
        "row_index": getattr(exc, "row_index", None),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _report_roots(
    prepared_root: Path,
    *,
    model_cache: ModelSExprCache | None,
    action_cache: ActionTraceCache | None,
) -> list[Path]:
    """One reports directory per enabled generation.

    Each consumer reads the manifest from its own generation's root -- the
    structured target audit opens ``{root}/manifests/{split}.json`` -- so a run
    that writes several generations writes the same counts into each of them.
    That is accurate rather than duplicated bookkeeping: a row either produced
    every output the run enabled or it failed.
    """
    roots = [Path(prepared_root) / SExprCache.REPORT_ROOT]
    if model_cache is not None:
        roots.append(Path(prepared_root) / model_cache.REPORT_ROOT)
    if action_cache is not None:
        roots.append(Path(prepared_root) / action_cache.REPORT_ROOT)
    return roots


def _sidecar_versions(
    *,
    model_cache: ModelSExprCache | None,
    action_cache: ActionTraceCache | None,
) -> tuple[int, str]:
    if action_cache is not None:
        return action_cache.SCHEMA_VERSION, action_cache.EXTRACTOR_VERSION
    if model_cache is not None:
        return ModelSExprCache.SCHEMA_VERSION, ModelSExprCache.NORMALIZATION
    return SExprCache.SCHEMA_VERSION, EXTRACTION_VERSION


def _target_names(
    *,
    model_cache: ModelSExprCache | None,
    action_cache: ActionTraceCache | None,
) -> list[str]:
    """Name every cache this run writes, for the manifest."""
    names = ["raw"]
    if model_cache is not None:
        names.append("model_sexpr_v2")
    if action_cache is not None:
        names.append(action_cache.SIDECAR_DIR)
    return names


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# S-expression Extraction Summary",
        "",
        f"- extractor: `{summary['extractor_version']}`",
        f"- dataset: `{summary['dataset']}`",
        f"- source commit: `{summary['source_commit']}`",
        f"- attempted rows: `{summary['attempted_rows']}`",
        f"- covered rows: `{summary['covered_rows']}`",
        f"- failed rows: `{summary['failed_rows']}`",
        f"- coverage: `{float(summary['coverage']):.4%}`",
        "",
        "| Split | Files | Theorems | Cached rows | Extracted rows | Failed rows | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    manifests = summary["manifests"]
    for split in summary["splits"]:
        manifest = manifests[split]
        lines.append(
            f"| {split} | {manifest['attempted_files']} | {manifest['attempted_theorems']} | "
            f"{manifest['cached_rows']} | {manifest['extracted_rows']} | "
            f"{manifest['failed_rows']} | {float(manifest['coverage']):.4%} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _extract_file_with_client(
    client,
    *,
    file_path: str,
    theorem_groups: dict[str, list[DatasetRow]],
    cache: SExprCache,
    source_root: Path,
    expected_commit: str,
    resume: bool,
    model_cache: ModelSExprCache | None = None,
    action_cache: ActionTraceCache | None = None,
) -> dict[str, object]:
    def row_is_cached(row: DatasetRow) -> bool:
        return _row_targets_cached(
            cache, row, model_cache=model_cache, action_cache=action_cache
        )

    file_rows = [
        row for theorem_rows in theorem_groups.values() for row in theorem_rows
    ]
    bad_commits = sorted(
        {row.repo_commit for row in file_rows if row.repo_commit != expected_commit}
    )
    if bad_commits:
        exc = SourceExtractionError(
            f"Dataset row commit does not match extractor checkout: {bad_commits}",
            phase="commit_mismatch",
        )
        units = None
        compiled = False
    else:
        try:
            units = await client.process_file(file_path)
            source_bytes = (Path(source_root) / file_path).read_bytes()
            compiled = True
        except Exception as caught:
            exc = SourceExtractionError(str(caught), phase="file_compile")
            units = None
            compiled = False

    extracted_rows = failed_rows = 0
    failures: list[dict[str, object]] = []
    failure_phases: Counter[str] = Counter()
    for theorem, theorem_rows in theorem_groups.items():
        uncached_rows = [
            row
            for row in theorem_rows
            if not (resume and row_is_cached(row))
        ]
        if not uncached_rows:
            continue
        try:
            if units is None:
                raise exc
            candidates = _candidate_invocations(
                source_bytes=source_bytes, units=units, theorem=theorem
            )
            if not candidates:
                raise SourceExtractionError(
                    "Could not identify the theorem's compilation unit.",
                    phase="theorem_identity",
                    theorem=theorem,
                )
        except Exception as caught:
            # Nothing was compiled, or the theorem has no compilation unit at
            # all, so every one of its rows is genuinely lost.  These are the
            # only failures that still cost a whole group.
            failure = _failure_record(caught, file_path, theorem, uncached_rows)
            failures.append(failure)
            failed_rows += len(uncached_rows)
            failure_phases[str(failure["phase"])] += len(uncached_rows)
            continue

        aligned = _align_rows(uncached_rows, candidates)
        for row, candidate in zip(uncached_rows, aligned):
            try:
                if isinstance(candidate, SourceExtractionError):
                    raise candidate
                # Build every enabled record before saving any of them, so a
                # capture error cannot leave one row half written.  The theorem
                # is deliberately no longer the unit: one unalignable row used
                # to void all of its neighbours, which is how 20658 otherwise
                # extractable rows were lost.  Whole-theorem selection is still
                # available to callers that need it -- build_sexpr_pilot's
                # --require-cached-train skips a partly cached theorem, exactly
                # as it already skipped a theorem holding a failed row.
                records = _build_records(
                    row,
                    candidate,
                    raw_cache=cache,
                    model_cache=model_cache,
                    action_cache=action_cache,
                    resume=resume,
                )
                for target_cache, record in records:
                    target_cache.save(row.split, row.row_index, record)
            except Exception as caught:
                failure = _failure_record(caught, file_path, theorem, [row])
                failures.append(failure)
                failed_rows += 1
                failure_phases[str(failure["phase"])] += 1
            else:
                extracted_rows += 1

    return {
        "compiled": compiled,
        "extracted_rows": extracted_rows,
        "failed_rows": failed_rows,
        "failures": failures,
        "failure_phases": failure_phases,
    }


async def extract_split_with_client(
    client,
    *,
    rows: list[DatasetRow],
    cache: SExprCache,
    prepared_root: Path,
    source_root: Path,
    split: str,
    expected_commit: str = DATASET_MATHLIB_COMMIT,
    resume: bool = True,
    recycle_worker_files: int = 0,
    model_cache: ModelSExprCache | None = None,
    action_cache: ActionTraceCache | None = None,
) -> dict[str, object]:
    clients = list(client) if isinstance(client, (list, tuple)) else [client]
    if not clients:
        raise ValueError("At least one Pantograph client is required.")
    report_roots = _report_roots(
        prepared_root, model_cache=model_cache, action_cache=action_cache
    )
    failure_paths = [root / "failures" / f"{split}.jsonl" for root in report_roots]
    for path in failure_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    try:
        file_groups = _group_rows_by_file(rows)
    except Exception as exc:
        file_groups = {}
        failure = _failure_record(exc, "", getattr(exc, "theorem", ""), rows)
        for path in failure_paths:
            path.write_text(
                json.dumps(failure, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        failure_phases = Counter({str(failure["phase"]): len(rows)})
        cached_rows = extracted_rows = compiled_files = 0
        failed_rows = len(rows)
    else:
        def row_is_cached(row: DatasetRow) -> bool:
            return _row_targets_cached(
                cache, row, model_cache=model_cache, action_cache=action_cache
            )

        cached_rows = sum(1 for row in rows if resume and row_is_cached(row))
        extracted_rows = failed_rows = compiled_files = 0
        failure_phases: Counter[str] = Counter()
        console_print(f"  [{split}] source extraction: {len(file_groups)} files, {len(rows)} rows")

        pending_jobs = []
        for file_index, (file_path, theorem_groups) in enumerate(file_groups.items(), 1):
            file_rows = [row for theorem_rows in theorem_groups.values() for row in theorem_rows]
            uncached_file_rows = [
                row for row in file_rows if not (resume and row_is_cached(row))
            ]
            if not uncached_file_rows:
                continue
            pending_jobs.append((file_index, file_path, theorem_groups))

        console_print(
            f"  [{split}] workers={len(clients)} | pending files={len(pending_jobs)}"
        )
        job_queue: asyncio.Queue[tuple[int, str, dict[str, list[DatasetRow]]]] = (
            asyncio.Queue()
        )
        result_queue: asyncio.Queue[tuple[int, str, dict[str, object]]] = (
            asyncio.Queue()
        )
        for job in pending_jobs:
            job_queue.put_nowait(job)

        async def worker(worker_client) -> None:
            processed_files = 0
            while True:
                try:
                    file_index, file_path, theorem_groups = job_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    try:
                        result = await _extract_file_with_client(
                            worker_client,
                            file_path=file_path,
                            theorem_groups=theorem_groups,
                            cache=cache,
                            source_root=source_root,
                            expected_commit=expected_commit,
                            resume=resume,
                            model_cache=model_cache,
                            action_cache=action_cache,
                        )
                    except Exception as caught:
                        file_rows = [
                            row
                            for theorem_rows in theorem_groups.values()
                            for row in theorem_rows
                            if not (resume and row_is_cached(row))
                        ]
                        failure = _failure_record(
                            caught, file_path, "", file_rows
                        )
                        result = {
                            "compiled": False,
                            "extracted_rows": 0,
                            "failed_rows": len(file_rows),
                            "failures": [failure],
                            "failure_phases": Counter(
                                {str(failure["phase"]): len(file_rows)}
                            ),
                        }
                    await result_queue.put((file_index, file_path, result))
                finally:
                    job_queue.task_done()
                    processed_files += 1
                    if (
                        recycle_worker_files > 0
                        and processed_files % recycle_worker_files == 0
                    ):
                        # Lean frontend environments retain substantial memory
                        # across files. Recycling bounds each REPL's lifetime;
                        # process_file() starts it again for the next queued job.
                        await worker_client.close()

        worker_tasks = [
            asyncio.create_task(worker(worker_client)) for worker_client in clients
        ]
        try:
            for completed in range(1, len(pending_jobs) + 1):
                _file_index, _file_path, result = await result_queue.get()
                compiled_files += int(bool(result["compiled"]))
                extracted_rows += int(result["extracted_rows"])
                failed_rows += int(result["failed_rows"])
                failure_phases.update(result["failure_phases"])
                for failure in result["failures"]:
                    encoded = json.dumps(failure, ensure_ascii=False, sort_keys=True)
                    for path in failure_paths:
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write(encoded + "\n")
                    console_print(
                        f"  [{split}] failed {failure['theorem']}: "
                        f"{failure['phase']}: {failure['error']}"
                    )

                if completed == 1 or completed % 25 == 0 or completed == len(pending_jobs):
                    console_print(
                        f"  [{split}] completed {completed}/{len(pending_jobs)} pending files "
                        f"| cached={cached_rows} extracted={extracted_rows} "
                        f"failed={failed_rows}"
                    )
        finally:
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

    attempted_rows = len(rows)
    covered_rows = cached_rows + extracted_rows
    schema_version, extractor_version = _sidecar_versions(
        model_cache=model_cache, action_cache=action_cache
    )
    manifest: dict[str, object] = {
        "schema_version": schema_version,
        "extractor_version": extractor_version,
        "enabled_targets": _target_names(
            model_cache=model_cache, action_cache=action_cache
        ),
        "dataset": rows[0].dataset_name if rows else None,
        "source_commit": expected_commit,
        "split": split,
        "workers": len(clients),
        "recycle_worker_files": recycle_worker_files,
        "attempted_files": len(file_groups),
        "compiled_files": compiled_files,
        "attempted_theorems": sum(len(groups) for groups in file_groups.values()),
        "attempted_rows": attempted_rows,
        "cached_rows": cached_rows,
        "extracted_rows": extracted_rows,
        "failed_rows": failed_rows,
        "covered_rows": covered_rows,
        "coverage": covered_rows / attempted_rows if attempted_rows else 0.0,
        "failure_phases": dict(sorted(failure_phases.items())),
        "failure_log": str(failure_paths[0]),
        "failure_logs": [str(path) for path in failure_paths],
    }
    for root, path in zip(report_roots, failure_paths):
        # Each generation's manifest points at the failure log in its own tree,
        # so a consumer reading one root never has to look inside another.
        _write_json(
            root / "manifests" / f"{split}.json",
            {**manifest, "failure_log": str(path)},
        )
    return manifest


def _git_head(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


async def extract_sexpressions(
    config: SExprExtractionConfig, *, client_factory=None
) -> dict[str, object]:
    if config.action_traces and config.source_syntax_traces:
        raise ValueError(
            "action_traces and source_syntax_traces both write an action-trace "
            "sidecar and cannot be combined; run them in separate passes."
        )
    if config.selection_manifest is not None and config.sample_per_split is not None:
        raise ValueError(
            "--selection-manifest and --max-items cannot be combined; "
            "the manifest already defines the exact rows."
        )
    if config.workers < 1:
        raise ValueError("workers must be at least 1.")
    if config.recycle_worker_files < 0:
        raise ValueError("recycle_worker_files cannot be negative.")
    source_root = Path(config.source_root).resolve()
    repl = Path(config.pantograph_repl).resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"Mathlib source checkout does not exist: {source_root}")
    if client_factory is None and not repl.is_file():
        raise RuntimeError(f"Patched Pantograph REPL does not exist: {repl}")
    if config.verify_source_commit:
        actual_commit = _git_head(source_root)
        if actual_commit != config.expected_commit:
            raise RuntimeError(
                f"Mathlib checkout is {actual_commit}, but dataset requires {config.expected_commit}."
            )
    if client_factory is None:
        try:
            pantograph_root = repl.parents[3]
            actual_pantograph_commit = _git_head(pantograph_root)
        except (IndexError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Could not identify the Pantograph checkout containing {repl}."
            ) from exc
        if actual_pantograph_commit != config.expected_pantograph_commit:
            raise RuntimeError(
                "Pantograph checkout is "
                f"{actual_pantograph_commit}, but extraction requires "
                f"{config.expected_pantograph_commit}."
            )

    clients = []
    for _worker_index in range(config.workers):
        if client_factory is None:
            client = PantographInvocationClient(
                source_root=source_root,
                pantograph_repl=repl,
                startup_timeout=config.server_startup_timeout,
                file_timeout=config.file_timeout,
                buffer_limit=config.buffer_limit,
                capture_model_sexprs=(
                    config.model_sexprs
                    or config.action_traces
                    or config.source_syntax_traces
                ),
            )
        else:
            client = client_factory(config)
            if asyncio.iscoroutine(client):
                client = await client
        clients.append(client)
    cache = SExprCache(config.prepared_root, str(source_root), enabled=True)
    model_cache = (
        ModelSExprCache(config.prepared_root, enabled=True)
        if config.model_sexprs
        else None
    )
    action_cache: ActionTraceCache | None = None
    if config.action_traces:
        action_cache = ActionTraceCache(config.prepared_root, enabled=True)
    elif config.source_syntax_traces:
        action_cache = SourceSyntaxTraceCache(config.prepared_root, enabled=True)
    manifests: dict[str, dict[str, object]] = {}
    started = False
    try:
        for raw_split in config.splits:
            split = canonicalize_split_name(raw_split)
            rows = list(
                iter_dataset_rows(
                    dataset_name=config.dataset_name,
                    split=split,
                    sample_limit=config.sample_per_split,
                )
            )
            if config.selection_manifest is not None:
                selected = selected_extraction_row_indices(
                    config.selection_manifest,
                    split,
                    dataset_name=config.dataset_name,
                )
                rows = [row for row in rows if row.row_index in selected]
                missing = selected - {row.row_index for row in rows}
                if missing:
                    raise RuntimeError(
                        f"Selection manifest contains {len(missing)} row indices "
                        f"not present in split '{split}'."
                    )
            if config.theorem is not None:
                rows = [row for row in rows if row.theorem == config.theorem]
                if not rows:
                    raise RuntimeError(
                        f"Theorem '{config.theorem}' was not found in split '{split}'."
                    )
            if not started:
                await asyncio.gather(*(client.start() for client in clients))
                started = True
            manifests[split] = await extract_split_with_client(
                clients,
                rows=rows,
                cache=cache,
                prepared_root=config.prepared_root,
                source_root=source_root,
                split=split,
                expected_commit=config.expected_commit,
                resume=config.resume,
                recycle_worker_files=config.recycle_worker_files,
                model_cache=model_cache,
                action_cache=action_cache,
            )
    finally:
        if started:
            await asyncio.gather(
                *(client.close() for client in clients), return_exceptions=True
            )

    attempted_rows = sum(int(item["attempted_rows"]) for item in manifests.values())
    covered_rows = sum(int(item["covered_rows"]) for item in manifests.values())
    schema_version, extractor_version = _sidecar_versions(
        model_cache=model_cache, action_cache=action_cache
    )
    summary: dict[str, object] = {
        "schema_version": schema_version,
        "extractor_version": extractor_version,
        "enabled_targets": _target_names(
            model_cache=model_cache, action_cache=action_cache
        ),
        "dataset": config.dataset_name,
        "source_root": str(source_root),
        "source_commit": config.expected_commit,
        "pantograph_commit": config.expected_pantograph_commit,
        "pantograph_repl": str(repl),
        "workers": config.workers,
        "recycle_worker_files": config.recycle_worker_files,
        "prepared_root": str(config.prepared_root),
        "splits": list(manifests),
        "manifests": manifests,
        "attempted_rows": attempted_rows,
        "covered_rows": covered_rows,
        "failed_rows": attempted_rows - covered_rows,
        "coverage": covered_rows / attempted_rows if attempted_rows else 0.0,
        "selection_manifest": (
            str(config.selection_manifest)
            if config.selection_manifest is not None
            else None
        ),
    }
    for report_root in _report_roots(
        config.prepared_root, model_cache=model_cache, action_cache=action_cache
    ):
        _write_json(report_root / "summary.json", summary)
        _write_summary_markdown(report_root / "summary.md", summary)
    return summary
