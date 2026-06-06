"""
Dataset loading and streaming for the LeanDojo benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable


DATASET_NAME = "cat-searcher/leandojo-benchmark-4-random"
CANONICAL_SPLITS = ("train", "val", "test")

_SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
}

_DATASET_SPLIT_NAMES = {
    "train": "train",
    "val": "validation",
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

    def metadata(self) -> dict[str, object]:
        return {
            "source": "dataset",
            "dataset": self.dataset_name,
            "split": self.split,
            "row_index": self.row_index,
            "theorem": self.theorem,
            "tactic": self.tactic,
        }


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


def _load_hf_split(split: str, *, dataset_name: str = DATASET_NAME):
    """Return a HuggingFace streaming dataset (lazy import)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Run: pip install datasets"
        ) from exc
    return load_dataset(dataset_name, split=dataset_split_name(split), streaming=True)


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
            return DatasetRow(
                state=sample["state"],
                theorem=sample.get("full_name", ""),
                tactic=sample.get("tactic", ""),
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
    """
    canonical_split = canonicalize_split_name(split)
    ds = _load_hf_split(canonical_split, dataset_name=dataset_name)
    for index, sample in enumerate(ds):
        if limit is not None and index >= limit:
            return
        yield DatasetRow(
            state=sample["state"],
            theorem=sample.get("full_name", ""),
            tactic=sample.get("tactic", ""),
            split=canonical_split,
            row_index=index,
            dataset_name=dataset_name,
        )


def iter_dataset_rows(
    *,
    dataset_name: str = DATASET_NAME,
    split: str = "train",
    sample_limit: int | None = None,
):
    yield from stream_split(split=split, limit=sample_limit, dataset_name=dataset_name)
