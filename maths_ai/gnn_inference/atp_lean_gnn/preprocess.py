from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from .cache import (
    SplitReport,
    append_failure_record,
    build_failure_record,
    build_json_payload,
    build_summary,
    prepare_output_root,
    write_json_artifact,
    write_manifest,
    write_pyg_artifact,
    write_summary_json,
    write_summary_markdown,
    write_vocab,
)
from .dataset import DATASET_NAME, DatasetRow, canonicalize_split_name, iter_dataset_rows
from .preparation import (
    prepare_example,
    SExprCache,
)
from .labels import build_tactic_vocab, encode_tactic_name
from .lemma_corpus import load_lemma_name_index
from .pyg import build_vocab_from_labels, dag_to_pyg
from .reporting import console_print


DEFAULT_OUTPUT_ROOT = Path("artifacts") / "prepared" / "v1"


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str = "cat-searcher/leandojo-benchmark-4-random"
    splits: tuple[str, ...] = ("train", "val", "test")
    output_root: Path = Path("artifacts") / "prepared" / "v1"
    sample_per_split: int | None = None
    lemma_corpus_path: Path | None = None
    force: bool = False
    use_sexpr: bool = False
    sexpr_cache_root: Path | None = None
    project_path: str = "maths_ai/lean_mathlib"


def _normalize_splits(raw_splits: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(raw_splits, str):
        candidates = [part.strip() for part in raw_splits.split(",")]
    else:
        candidates = [part.strip() for part in raw_splits]

    splits: list[str] = []
    for split in candidates:
        if not split:
            continue
        canonical_split = canonicalize_split_name(split)
        if canonical_split not in splits:
            splits.append(canonical_split)

    if not splits:
        raise ValueError("At least one split must be provided.")
    if "train" not in splits:
        raise ValueError("The requested splits must include 'train' so train-only vocabularies can be built.")
    return ["train", *[split for split in splits if split != "train"]]


def _load_rows(
    dataset_name: str,
    split: str,
    sample_limit: int | None = None,
) -> list[DatasetRow]:
    """Load all rows for a split into memory, grouped by order."""
    return list(iter_dataset_rows(
        dataset_name=dataset_name,
        split=split,
        sample_limit=sample_limit,
    ))


def _build_sexpr_map(
    rows: list[DatasetRow],
    project_path: str,
    use_sexpr: bool,
    sexpr_cache: Optional[SExprCache] = None,
    split_label: str = "",
) -> dict[int, dict]:
    """Load validated Phase 2 records without silently generating or falling back."""
    if not use_sexpr:
        return {}

    sexpr_map: dict[int, dict] = {}

    if sexpr_cache is None:
        raise ValueError("S-expression mode requires a Phase 2 cache root.")
    theorem_rows: dict[str, list[DatasetRow]] = {}
    for row in rows:
        theorem_rows.setdefault(row.theorem, []).append(row)
    for grouped_rows in theorem_rows.values():
        for step_index, row in enumerate(
            sorted(grouped_rows, key=lambda item: item.row_index)
        ):
            cached = sexpr_cache.load_for_row(
                row,
                step_index=step_index,
                extractor_version=SExprCache.EXTRACTOR_VERSION,
            )
            if cached is not None:
                sexpr_map[row.row_index] = cached

    console_print(
        f"    Validated S-expression cache: {len(sexpr_map)}/{len(rows)} rows"
    )

    return sexpr_map


def scan_train_split(
    *,
    dataset_name: str,
    sample_per_split: int | None,
    output_root: Path,
    sexpr_cache: Optional[SExprCache],
    project_path: str = "maths_ai/lean_mathlib",
    use_sexpr: bool = False,
) -> tuple[dict[str, int], dict[str, int], SplitReport]:
    node_labels: set[str] = set()
    tactic_names: list[str] = []
    report = SplitReport(split="train")

    rows = _load_rows(dataset_name, "train", sample_per_split)
    sexpr_map = _build_sexpr_map(rows, project_path, use_sexpr, sexpr_cache, "train")

    for row in rows:
        try:
            sd = sexpr_map.get(row.row_index)
            example = prepare_example(
                row,
                sexpr_cache=sexpr_cache,
                sexpr_data=sd,
                use_sexpr=use_sexpr,
            )
        except Exception as exc:
            failure_record = build_failure_record(row, exc)
            report.record_failure(
                category=str(failure_record["failure_category"]),
                phase=str(failure_record["phase"]),
            )
            continue

        report.record_success(dag=example.dag, tactic_name=example.tactic_name)
        node_labels.update(node.label for node in example.dag.nodes)
        tactic_names.append(example.tactic_name)

    if report.success_count == 0:
        raise RuntimeError("The train split produced zero successful examples while building vocabularies.")

    node_vocab = build_vocab_from_labels(node_labels)
    tactic_vocab = build_tactic_vocab(tactic_names)
    return node_vocab, tactic_vocab, report


def _resolve_arg_node_indices(dag, arg_tokens: list[str]) -> list[int]:
    """Best-effort: match each argument token to a DAG node index by label.

    Returns a list of node indices (one per argument token), using ``-1``
    when no matching node is found in the graph.
    """
    # Build label → node_id map (first match wins)
    label_to_id: dict[str, int] = {}
    for node in dag.nodes:
        if node.label not in label_to_id:
            label_to_id[node.label] = node.id

    return [label_to_id.get(token, -1) for token in arg_tokens]


def _resolve_arg_lemma_ids(
    arg_tokens: list[str],
    lemma_name_index: dict[str, int] | None,
) -> list[int]:
    if not arg_tokens:
        return []
    if lemma_name_index is None:
        return [-1 for _ in arg_tokens]
    return [lemma_name_index.get(token, -1) for token in arg_tokens]


def process_split(
    *,
    dataset_name: str,
    split: str,
    sample_per_split: int | None,
    output_root: Path,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    lemma_name_index: dict[str, int] | None,
    sexpr_cache: Optional[SExprCache],
    project_path: str = "maths_ai/lean_mathlib",
    use_sexpr: bool = False,
) -> tuple[SplitReport, dict[str, object]]:
    import torch

    from .labels import parse_tactic_arguments
    from .pyg import build_premise_mask

    report = SplitReport(split=split)

    rows = _load_rows(dataset_name, split, sample_per_split)
    sexpr_map = _build_sexpr_map(rows, project_path, use_sexpr, sexpr_cache, split)

    for row in rows:
        try:
            sd = sexpr_map.get(row.row_index)
            example = prepare_example(
                row,
                sexpr_cache=sexpr_cache,
                sexpr_data=sd,
                use_sexpr=use_sexpr,
            )
        except Exception as exc:
            failure_record = build_failure_record(row, exc)
            append_failure_record(output_root, split=split, record=failure_record)
            report.record_failure(
                category=str(failure_record["failure_category"]),
                phase=str(failure_record["phase"]),
            )
            continue

        json_payload = build_json_payload(
            row,
            parsed_state=example.parsed_state,
            dag=example.dag,
            tactic_name=example.tactic_name,
        )
        write_json_artifact(
            output_root,
            split=split,
            row_index=row.row_index,
            payload=json_payload,
        )

        dag = example.dag
        tactic_name = example.tactic_name
        data = dag_to_pyg(dag, node_vocab)
        data.y = torch.tensor(
            [encode_tactic_name(tactic_name, tactic_vocab)],
            dtype=torch.long,
        )
        data.split = split
        data.row_index = row.row_index
        data.dataset_name = row.dataset_name
        data.theorem = row.theorem
        data.tactic_raw = row.tactic
        data.tactic_name = tactic_name

        # Materialize the readout node once. Older prepared datasets omit this
        # field and remain supported by PreparedGraphDataset's fallback.
        state_node_ids = [node.id for node in dag.nodes if node.label == "State"]
        source_node_ids = {source for source, _ in dag.edges}
        root_state_ids = [node_id for node_id in state_node_ids if node_id not in source_node_ids]
        if len(root_state_ids) != 1:
            raise ValueError(
                f"Expected exactly one root State node, found {len(root_state_ids)}."
            )
        data.state_node_index = torch.tensor(root_state_ids, dtype=torch.long)

        # --- Argument-selection ground truth (additive) ---------------
        premise_mask = build_premise_mask(dag)
        data.premise_mask = torch.tensor(premise_mask, dtype=torch.bool)

        _, arg_tokens = parse_tactic_arguments(row.tactic)
        arg_indices = _resolve_arg_node_indices(dag, arg_tokens)
        arg_lemma_ids = _resolve_arg_lemma_ids(arg_tokens, lemma_name_index)
        for idx, node_id in enumerate(arg_indices):
            if node_id >= 0 and idx < len(arg_lemma_ids):
                arg_lemma_ids[idx] = -1
        data.arg_node_indices = torch.tensor(arg_indices, dtype=torch.long) if arg_indices else torch.tensor([], dtype=torch.long)
        data.arg_lemma_ids = torch.tensor(arg_lemma_ids, dtype=torch.long) if arg_lemma_ids else torch.tensor([], dtype=torch.long)
        data.arg_count = len(arg_indices)
        # --------------------------------------------------------------

        write_pyg_artifact(
            output_root,
            split=split,
            row_index=row.row_index,
            data=data,
        )

        report.record_success(dag=dag, tactic_name=tactic_name)

    if report.success_count == 0:
        raise RuntimeError(f"Split '{split}' produced zero successful examples.")

    manifest = report.to_manifest(
        dataset_name=dataset_name,
        output_root=output_root,
        vocab_source="train",
        sample_limit=sample_per_split,
    )
    write_manifest(output_root, split=split, manifest=manifest)
    return report, manifest


def run_preprocessing(config: PreprocessConfig) -> dict[str, object]:
    output_root = Path(config.output_root)
    if output_root.exists() and not config.force:
        raise FileExistsError(
            f"Output root '{output_root}' already exists. Re-run with --force to overwrite it."
        )

    sexpr_cache = None
    if config.use_sexpr:
        if config.sexpr_cache_root is None:
            raise ValueError(
                "S-expression preprocessing requires --sexpr-cache-root from "
                "the completed theorem-replay extraction stage."
            )
        if config.sexpr_cache_root.resolve() == output_root.resolve():
            raise ValueError(
                "--sexpr-cache-root must differ from --output-root because "
                "preprocessing replaces the output directory."
            )
        console_print("  S-expression mode: consuming validated theorem-replay cache")
        sexpr_cache = SExprCache(
            config.sexpr_cache_root,
            config.project_path,
            enabled=True,
        )

    console_print(
        f"\n  Scanning train split from {config.dataset_name} to build train-only vocabularies..."
    )
    node_vocab, tactic_vocab, train_scan = scan_train_split(
        dataset_name=config.dataset_name,
        sample_per_split=config.sample_per_split,
        output_root=config.output_root,
        sexpr_cache=sexpr_cache,
        project_path=config.project_path,
        use_sexpr=config.use_sexpr,
    )
    console_print(
        f"  Train scan complete: attempted={train_scan.attempted_count}, "
        f"success={train_scan.success_count}, failure={train_scan.failure_count}"
    )

    lemma_name_index = None
    if config.lemma_corpus_path is not None:
        lemma_name_index = load_lemma_name_index(config.lemma_corpus_path)

    prepare_output_root(output_root, splits=list(config.splits), force=config.force)
    write_vocab(output_root, name="node_vocab.json", vocab=node_vocab)
    write_vocab(output_root, name="tactic_vocab.json", vocab=tactic_vocab)

    split_reports: dict[str, SplitReport] = {}
    manifests: dict[str, dict[str, object]] = {}
    for split in config.splits:
        console_print(f"\n  Processing split '{split}'...")
        report, manifest = process_split(
            dataset_name=config.dataset_name,
            split=split,
            sample_per_split=config.sample_per_split,
            output_root=output_root,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            lemma_name_index=lemma_name_index,
            sexpr_cache=sexpr_cache,
            project_path=config.project_path,
            use_sexpr=config.use_sexpr,
        )
        split_reports[split] = report
        manifests[split] = manifest
        console_print(
            f"  Finished '{split}': attempted={report.attempted_count}, "
            f"success={report.success_count}, failure={report.failure_count}"
        )

    summary = build_summary(
        dataset_name=config.dataset_name,
        output_root=output_root,
        splits=list(config.splits),
        manifests=manifests,
        split_reports=split_reports,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
    )
    summary_json_path = write_summary_json(output_root, summary)
    summary_md_path = write_summary_markdown(output_root, summary)

    console_print(f"\n  Wrote node vocab     : {output_root / 'vocab' / 'node_vocab.json'}")
    console_print(f"  Wrote tactic vocab   : {output_root / 'vocab' / 'tactic_vocab.json'}")
    console_print(f"  Wrote JSON summary   : {summary_json_path}")
    console_print(f"  Wrote Markdown summary: {summary_md_path}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare cached graph artifacts from LeanDojo proof states")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME, help="Dataset name to stream from Hugging Face")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits to preprocess (must include train)")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output directory for prepared artifacts")
    parser.add_argument("--sample-per-split", type=int, default=None, help="Optional limit of examples to process per split")
    parser.add_argument("--lemma-corpus", type=str, default=None, help="Optional lemma corpus JSONL for library premise labels")
    parser.add_argument("--force", action="store_true", help="Overwrite the output root if it already exists")
    parser.add_argument("--use-sexpr", action="store_true", default=False, help="Use S-expressions from Pantograph (default: False, text parser only for training)")
    parser.add_argument("--no-sexpr", action="store_false", dest="use_sexpr", help="Disable S-expressions, use text parser only")
    parser.add_argument("--project-path", type=str, default="maths_ai/lean_mathlib", help="Path to Lean project for Pantograph")
    parser.add_argument("--sexpr-cache-root", type=str, default=None, help="Validated cache root produced by generate_sexprs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = PreprocessConfig(
            dataset_name=args.dataset_name,
            splits=tuple(_normalize_splits(args.splits)),
            output_root=Path(args.output_root),
            sample_per_split=args.sample_per_split,
            lemma_corpus_path=None if args.lemma_corpus is None else Path(args.lemma_corpus),
            force=args.force,
            use_sexpr=args.use_sexpr,
            sexpr_cache_root=None if args.sexpr_cache_root is None else Path(args.sexpr_cache_root),
            project_path=args.project_path,
        )
        run_preprocessing(config)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
