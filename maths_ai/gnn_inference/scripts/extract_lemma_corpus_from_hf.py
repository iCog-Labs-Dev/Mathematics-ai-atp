"""Extract a lemma corpus from the LeanDojo benchmark on HuggingFace.

This script streams the LeanDojo benchmark dataset, collects unique theorem
names and their proof-state goals, and writes a JSONL corpus compatible with
``atp_lean_gnn.lemma_corpus.LemmaRecord``.

Usage::

    python scripts/extract_lemma_corpus_from_hf.py \
        --output-dir artifacts/lemmas/v1/corpus \
        --sample-limit 500 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from atp_lean_gnn.dataset import DATASET_NAME, stream_split
from atp_lean_gnn.lemma_corpus import LemmaRecord, write_lemma_corpus
from atp_lean_gnn.state import parse_state


DEFAULT_OUTPUT_DIR = Path("artifacts") / "lemmas" / "v1" / "corpus"
SAMPLE_SIZE = 50


@dataclass(frozen=True)
class ExtractionResult:
    total_rows_scanned: int
    unique_theorems: int
    parse_failures: int
    sample_count: int


def _extract_goal_from_state(state_str: str) -> str | None:
    """Parse a proof state string and return the goal expression.

    Returns ``None`` if the state cannot be parsed.
    """
    try:
        parsed = parse_state(state_str)
        return parsed.goal if parsed.goal else None
    except Exception:
        return None


def _infer_namespace(full_name: str) -> str:
    """Infer namespace from a dotted fully-qualified name."""
    if "." in full_name:
        return full_name.rsplit(".", 1)[0]
    return ""


def extract_corpus(
    *,
    output_dir: Path,
    dataset_name: str = DATASET_NAME,
    split: str = "train",
    sample_limit: int | None = None,
    force: bool = False,
) -> ExtractionResult:
    """Stream the dataset and write a deduplicated lemma corpus.

    For each unique ``full_name`` (theorem), we take the goal expression from
    its first occurrence as the lemma *statement*.  This gives us a reasonable
    approximation of each Mathlib theorem's type signature.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "lemmas.jsonl"
    manifest_path = output_dir / "manifest.json"
    sample_path = output_dir / "lemmas_sample.jsonl"
    failures_path = output_dir / "failures.jsonl"

    if corpus_path.exists() and not force:
        raise FileExistsError(
            f"Output '{corpus_path}' already exists. Use --force to overwrite."
        )

    seen_names: dict[str, str] = {}  # full_name → goal statement
    parse_failures: list[dict[str, object]] = []
    total_rows = 0

    print(f"  Streaming {dataset_name} split='{split}'...")

    for row in stream_split(split, limit=sample_limit, dataset_name=dataset_name):
        total_rows += 1
        full_name = row.theorem.strip()

        if not full_name:
            parse_failures.append(
                {"row_index": row.row_index, "reason": "empty_full_name"}
            )
            continue

        # Only take the first occurrence of each theorem
        if full_name in seen_names:
            continue

        goal = _extract_goal_from_state(row.state)
        if goal is None:
            parse_failures.append(
                {
                    "row_index": row.row_index,
                    "full_name": full_name,
                    "reason": "state_parse_failure",
                }
            )
            continue

        seen_names[full_name] = goal

        if total_rows % 10000 == 0:
            print(
                f"    scanned {total_rows} rows, "
                f"{len(seen_names)} unique theorems so far..."
            )

    # Build LemmaRecord list
    records: list[LemmaRecord] = []
    for lemma_id, (name, statement) in enumerate(seen_names.items()):
        records.append(
            LemmaRecord(
                lemma_id=lemma_id,
                name=name,
                statement=statement,
                namespace=_infer_namespace(name),
                module="",
            )
        )

    # Write full corpus
    write_lemma_corpus(corpus_path, records)

    # Write sample (first N records)
    sample_records = records[:SAMPLE_SIZE]
    write_lemma_corpus(sample_path, sample_records)

    # Write failures
    if parse_failures:
        with failures_path.open("w", encoding="utf-8") as handle:
            for failure in parse_failures:
                handle.write(
                    json.dumps(failure, ensure_ascii=False, sort_keys=True)
                )
                handle.write("\n")

    # Write manifest
    manifest = {
        "source_dataset": dataset_name,
        "source_split": split,
        "total_rows_scanned": total_rows,
        "unique_theorems": len(records),
        "parse_failures": len(parse_failures),
        "sample_limit": sample_limit,
        "sample_count": len(sample_records),
        "corpus_path": str(corpus_path),
        "sample_path": str(sample_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    result = ExtractionResult(
        total_rows_scanned=total_rows,
        unique_theorems=len(records),
        parse_failures=len(parse_failures),
        sample_count=len(sample_records),
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a lemma corpus from the LeanDojo HuggingFace dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for corpus artifacts",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DATASET_NAME,
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to extract from (default: train)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on total rows streamed from the dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing corpus artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = extract_corpus(
            output_dir=Path(args.output_dir),
            dataset_name=args.dataset_name,
            split=args.split,
            sample_limit=args.sample_limit,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(
        f"\nExtracted lemma corpus:\n"
        f"  rows scanned   = {result.total_rows_scanned}\n"
        f"  unique theorems = {result.unique_theorems}\n"
        f"  parse failures  = {result.parse_failures}\n"
        f"  sample entries  = {result.sample_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
