"""Dataset loading and streaming for the generated Mathlib proof-state corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generator, Iterable

from maths_ai.lean_replay import LeanGoalSeed, seed_from_records
from maths_ai.lean_replay.columns import check_raw_schema, dataset_locals, decode_json_list

# The Hub now redirects this name to iCogLabs-Dev/Mathlib-Normalized-Sexpr.
DATASET_NAME = "jajostrains/Mathlib-Normalized-Sexpr"
# Pinned dataset commit; rows are only comparable across runs at one revision.
DATASET_REVISION = "f44841f98193c09b20c6092198a22e98ff2cd7a8"
CANONICAL_SPLITS = ("train", "val", "test")

_SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
}

_DATASET_SPLIT_NAMES = {
    "train": "train",
    "val": "val",
    "test": "test",
}


@dataclass(frozen=True)
class DatasetRow:
    state: str
    theorem: str
    tactic: str
    split: str
    row_index: int
    dataset_name: str = DATASET_NAME
    target_state: str = ""
    repo_url: str = ""
    repo_commit: str = ""
    file_path: str = ""
    model_goal_sexp: str = ""
    model_hyp_sexps: tuple[dict[str, object], ...] = ()
    dataset_revision: str | None = None
    # Raw execution columns, kept exactly as stored and decoded only by
    # ``structured_seed``, which rejects malformed values instead of coercing them.
    raw_goal_sexp: object = None
    raw_hyp_sexps: object = None
    local_context: object = None
    raw_schema_version: object = None
    pending_goal_count: object = None
    # The dataset's own `row_index` column; `row_index` above is the position
    # in the split stream, and the two differ once the producer skipped rows.
    source_row_index: object = None

    def source_identity(self) -> dict[str, object]:
        return {
            "dataset": self.dataset_name,
            "revision": self.dataset_revision,
            "split": self.split,
            "row_index": self.row_index,
            "source_row_index": self.source_row_index,
            "theorem": self.theorem,
            "file_path": self.file_path,
            "commit": self.repo_commit,
        }

    def structured_seed(self) -> LeanGoalSeed:
        """The row's focused goal from its raw expressions and local context.

        Raises ``ReplayInputError`` for a missing, malformed or unsupported
        record; a structured row never falls back to its printed state.
        """
        check_raw_schema(self.raw_schema_version)
        locals_ = dataset_locals(
            decode_json_list(self.raw_hyp_sexps, "raw_hyp_sexps"),
            decode_json_list(self.local_context, "local_context"),
            list(self.model_hyp_sexps) or None,
        )
        return seed_from_records(
            self.raw_goal_sexp,
            locals_,
            source=self.source_identity(),
            model_target=self.model_goal_sexp or None,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "source": "dataset",
            "dataset": self.dataset_name,
            "split": self.split,
            "row_index": self.row_index,
            "theorem": self.theorem,
            "tactic": self.tactic,
            "target_state": self.target_state,
            "url": self.repo_url,
            "commit": self.repo_commit,
            "file_path": self.file_path,
            "has_model_sexprs": bool(self.model_goal_sexp and self.model_hyp_sexps),
        }


def _json_object_list(sample: dict[str, object], field: str) -> tuple[dict[str, object], ...]:
    """Decode a normalized S-expression column stored as a JSON string."""
    value = sample.get(field)
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Dataset field '{field}' is not valid JSON.") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Dataset field '{field}' must be a JSON list of objects.")
    return tuple(value)


def _dataset_row_from_sample(
    sample: dict[str, object],
    *,
    split: str,
    row_index: int,
    dataset_name: str,
    dataset_revision: str | None = None,
) -> DatasetRow:
    """Normalize the generated dataset schema into the RL row contract."""
    # The generated dataset calls these fields ``text_state`` and ``theorem``;
    # the aliases retain compatibility with the original LeanDojo row schema.
    state = sample.get("text_state", sample.get("state"))
    if not isinstance(state, str) or not state.strip():
        raise ValueError(
            "Dataset row is missing a non-empty 'text_state' field; "
            f"available fields: {sorted(sample)}"
        )
    target_state = sample.get("text_target_state", sample.get("target_state", ""))
    return DatasetRow(
        state=state,
        theorem=str(sample.get("theorem", sample.get("full_name", ""))),
        tactic=str(sample.get("tactic", "")),
        target_state=str(target_state),
        repo_url=str(sample.get("repo_url", sample.get("url", ""))),
        repo_commit=str(sample.get("repo_commit", sample.get("commit", ""))),
        file_path=str(sample.get("file_path", "")),
        split=split,
        row_index=row_index,
        dataset_name=dataset_name,
        model_goal_sexp=str(sample.get("model_goal_sexp", "")),
        model_hyp_sexps=_json_object_list(sample, "model_hyp_sexps"),
        dataset_revision=dataset_revision,
        raw_goal_sexp=sample.get("raw_goal_sexp"),
        raw_hyp_sexps=sample.get("raw_hyp_sexps"),
        local_context=sample.get("local_context"),
        raw_schema_version=sample.get("raw_schema_version"),
        pending_goal_count=sample.get("pending_goal_count"),
        source_row_index=sample.get("row_index"),
    )


def canonicalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    try:
        return _SPLIT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported split '{split}'. Use one of: train, val, validation, test."
        ) from exc


def dataset_split_name(split: str) -> str:
    canonical = canonicalize_split_name(split)
    return _DATASET_SPLIT_NAMES[canonical]


def _load_hf_split(
    split: str, *, dataset_name: str = DATASET_NAME, revision: str | None = None
):
    """Return a HuggingFace streaming dataset (lazy import)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Run: pip install datasets"
        ) from exc
    return load_dataset(
        dataset_name, split=dataset_split_name(split), streaming=True, revision=revision
    )


def get_dataset_stream(dataset_name: str = DATASET_NAME, *, split: str = "train") -> Iterable[dict[str, object]]:
    return _load_hf_split(split, dataset_name=dataset_name)


def load_dataset_row(
    row_index: int,
    *,
    split: str = "train",
    dataset_name: str = DATASET_NAME,
) -> DatasetRow:
    """Load a single row by index (for the interactive CLI)."""
    canonical_split = canonicalize_split_name(split)
    ds = _load_hf_split(canonical_split, dataset_name=dataset_name)
    for index, sample in enumerate(ds):
        if index == row_index:
            return _dataset_row_from_sample(
                sample,
                split=canonical_split,
                row_index=index,
                dataset_name=dataset_name,
            )
    raise IndexError(f"Row {row_index} not found in split '{canonical_split}'.")


def stream_split(
    split: str = "train",
    *,
    limit: int | None = None,
    dataset_name: str = DATASET_NAME,
    revision: str | None = None,
) -> Generator[DatasetRow, None, None]:
    """
    Yield ``DatasetRow`` objects for every example in *split*.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, ``"validation"``, or ``"test"``.
    limit : int or None
        If set, stop after this many rows (useful for dry runs).
    dataset_name : str
        Override the default HuggingFace dataset identifier.
    revision : str or None
        Dataset commit to read; ``None`` reads the current head.
    """
    canonical_split = canonicalize_split_name(split)
    ds = _load_hf_split(canonical_split, dataset_name=dataset_name, revision=revision)
    for index, sample in enumerate(ds):
        if limit is not None and index >= limit:
            return
        yield _dataset_row_from_sample(
            sample,
            split=canonical_split,
            row_index=index,
            dataset_name=dataset_name,
            dataset_revision=revision,
        )


def iter_dataset_rows(
    *,
    dataset_name: str = DATASET_NAME,
    split: str = "train",
    sample_limit: int | None = None,
    revision: str | None = None,
):
    yield from stream_split(
        split=split, limit=sample_limit, dataset_name=dataset_name, revision=revision
    )
