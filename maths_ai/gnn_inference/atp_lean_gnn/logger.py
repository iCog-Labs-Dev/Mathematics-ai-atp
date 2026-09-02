from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence


def _append_jsonl(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return path


def _append_csv(path: Path, fieldnames: Sequence[str], payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: payload.get(key, "") for key in fieldnames})
    return path


class TrainingLogger:
    """Log epoch metrics for training curves to JSONL and CSV."""

    def __init__(self, output_dir: Path, prefix: str = "learning_curve") -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / f"{prefix}.jsonl"
        self.csv_path = self.output_dir / f"{prefix}.csv"
        self._fieldnames: list[str] | None = None

    def log_epoch(self, epoch: int, metrics: dict[str, float | int]) -> None:
        record = {"epoch": epoch, **metrics}
        if self._fieldnames is None:
            self._fieldnames = ["epoch", *sorted(k for k in record.keys() if k != "epoch")]
        _append_jsonl(self.jsonl_path, record)
        _append_csv(self.csv_path, self._fieldnames, record)