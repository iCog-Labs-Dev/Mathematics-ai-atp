from __future__ import annotations

import argparse
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .cache import (
    SplitReport,
    append_failure_record,
    build_failure_record,
    prepare_audit_output_root,
    write_manifest,
    write_parser_audit_json,
    write_parser_audit_markdown,
)
from .dataset import DATASET_NAME, canonicalize_split_name, iter_dataset_rows
from .preparation import prepare_example
from .reporting import console_print


DEFAULT_AUDIT_OUTPUT_ROOT = Path("artifacts") / "audits" / "parser" / "v1"


@dataclass(frozen=True)
class ParserAuditConfig:
    dataset_name: str = DATASET_NAME
    splits: tuple[str, ...] = ("train", "val", "test")
    output_root: Path = DEFAULT_AUDIT_OUTPUT_ROOT
    sample_per_split: int | None = None
    max_examples_per_category: int = 5
    force: bool = False


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
    return splits


def _counter_summary(counter: Counter[str], *, limit: int = 10) -> list[dict[str, object]]:
    return [{"name": name, "count": count} for (name, count) in counter.most_common(limit)]


def _merge_representative_examples(
    split_reports: dict[str, SplitReport],
    *,
    max_examples_per_category: int,
    top_categories: list[str],
) -> dict[str, list[dict[str, object]]]:
    representative_examples: dict[str, list[dict[str, object]]] = {}
    for category in top_categories:
        examples: list[dict[str, object]] = []
        for split in split_reports:
            for example in split_reports[split].representative_failures.get(category, []):
                if len(examples) >= max_examples_per_category:
                    break
                examples.append(example)
            if len(examples) >= max_examples_per_category:
                break
        representative_examples[category] = examples
    return representative_examples


def _safe_mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _safe_median(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _build_parser_audit_summary(
    *,
    config: ParserAuditConfig,
    manifests: dict[str, dict[str, object]],
    split_reports: dict[str, SplitReport],
) -> dict[str, object]:
    overall_attempted = sum(report.attempted_count for report in split_reports.values())
    overall_success = sum(report.success_count for report in split_reports.values())
    overall_failure = sum(report.failure_count for report in split_reports.values())

    overall_failure_categories: Counter[str] = Counter()
    overall_failure_phases: Counter[str] = Counter()
    all_node_counts: list[int] = []
    all_edge_counts: list[int] = []
    all_reused_counts: list[int] = []

    for report in split_reports.values():
        overall_failure_categories.update(report.failure_categories)
        overall_failure_phases.update(report.failure_phases)
        all_node_counts.extend(report.node_counts)
        all_edge_counts.extend(report.edge_counts)
        all_reused_counts.extend(report.reused_node_counts)

    top_failure_categories = _counter_summary(overall_failure_categories)
    top_failure_phases = _counter_summary(overall_failure_phases)
    recommended_follow_up_categories = _counter_summary(overall_failure_categories, limit=5)
    representative_examples = _merge_representative_examples(
        split_reports,
        max_examples_per_category=config.max_examples_per_category,
        top_categories=[item["name"] for item in recommended_follow_up_categories],
    )

    return {
        "dataset": config.dataset_name,
        "output_root": str(Path(config.output_root)),
        "splits": list(config.splits),
        "sample_per_split": config.sample_per_split,
        "max_examples_per_category": config.max_examples_per_category,
        "overall": {
            "attempted_count": overall_attempted,
            "success_count": overall_success,
            "failure_count": overall_failure,
            "parser_success_rate": 0.0 if overall_attempted == 0 else overall_success / overall_attempted,
        },
        "overall_graph_stats": {
            "node_count": {
                "mean": _safe_mean(all_node_counts),
                "median": _safe_median(all_node_counts),
            },
            "edge_count": {
                "mean": _safe_mean(all_edge_counts),
                "median": _safe_median(all_edge_counts),
            },
            "reused_node_count": {
                "mean": _safe_mean(all_reused_counts),
                "median": _safe_median(all_reused_counts),
            },
        },
        "splits_summary": manifests,
        "top_failure_categories": top_failure_categories,
        "top_failure_phases": top_failure_phases,
        "representative_examples": representative_examples,
        "recommended_follow_up_categories": recommended_follow_up_categories,
    }


def _render_parser_audit_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Parser Coverage Audit",
        "",
        f"- dataset: `{summary['dataset']}`",
        f"- output root: `{summary['output_root']}`",
        f"- processed splits: `{', '.join(summary['splits'])}`",
        f"- sample per split: `{summary['sample_per_split']}`",
        f"- max examples per category: `{summary['max_examples_per_category']}`",
        f"- attempted examples: `{summary['overall']['attempted_count']}`",
        f"- successful examples: `{summary['overall']['success_count']}`",
        f"- failed examples: `{summary['overall']['failure_count']}`",
        f"- parser success rate: `{summary['overall']['parser_success_rate']:.3f}`",
        "",
        "## Split Metrics",
        "",
        "| Split | Attempted | Success | Failure | Success Rate | Mean Nodes | Median Nodes | Mean Edges | Median Edges |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for split in summary["splits"]:
        split_summary = summary["splits_summary"][split]
        lines.append(
            "| "
            f"{split} | "
            f"{split_summary['attempted_count']} | "
            f"{split_summary['success_count']} | "
            f"{split_summary['failure_count']} | "
            f"{split_summary['parser_success_rate']:.3f} | "
            f"{split_summary['graph_stats']['node_count']['mean']:.2f} | "
            f"{split_summary['graph_stats']['node_count']['median']:.2f} | "
            f"{split_summary['graph_stats']['edge_count']['mean']:.2f} | "
            f"{split_summary['graph_stats']['edge_count']['median']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Top Failure Categories",
            "",
            "| Category | Count |",
            "| --- | ---: |",
        ]
    )
    if summary["top_failure_categories"]:
        for item in summary["top_failure_categories"]:
            lines.append(f"| `{item['name']}` | {item['count']} |")
    else:
        lines.append("| `none` | 0 |")

    lines.extend(
        [
            "",
            "## Top Failure Phases",
            "",
            "| Phase | Count |",
            "| --- | ---: |",
        ]
    )
    if summary["top_failure_phases"]:
        for item in summary["top_failure_phases"]:
            lines.append(f"| `{item['name']}` | {item['count']} |")
    else:
        lines.append("| `none` | 0 |")

    lines.extend(["", "## Recommended Follow-Up Categories", ""])
    if summary["recommended_follow_up_categories"]:
        for item in summary["recommended_follow_up_categories"]:
            lines.append(f"- `{item['name']}`: {item['count']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Representative Examples", ""])
    if summary["representative_examples"]:
        for category, examples in summary["representative_examples"].items():
            lines.append(f"### `{category}`")
            if not examples:
                lines.append("")
                lines.append("- no saved examples")
                lines.append("")
                continue
            lines.append("")
            for example in examples:
                lines.append(
                    "- "
                    f"split=`{example['split']}` "
                    f"row=`{example['row_index']}` "
                    f"theorem=`{example['theorem'] or '<unknown>'}` "
                    f"error=`{example['error_message']}`"
                )
                state_preview = str(example.get("state_preview", "")).strip()
                if state_preview:
                    lines.append("")
                    lines.append("```text")
                    lines.append(state_preview)
                    lines.append("```")
            lines.append("")
    else:
        lines.append("- no representative failures recorded")

    return "\n".join(lines).rstrip() + "\n"


def run_parser_audit(config: ParserAuditConfig) -> dict[str, object]:
    output_root = prepare_audit_output_root(
        config.output_root,
        splits=list(config.splits),
        force=config.force,
    )

    split_reports: dict[str, SplitReport] = {}
    manifests: dict[str, dict[str, object]] = {}

    for split in config.splits:
        console_print(f"\n  Auditing split '{split}'...")
        report = SplitReport(split=split)

        for row in iter_dataset_rows(
            dataset_name=config.dataset_name,
            split=split,
            sample_limit=config.sample_per_split,
        ):
            try:
                example = prepare_example(row)
            except Exception as exc:
                failure_record = build_failure_record(row, exc)
                append_failure_record(output_root, split=split, record=failure_record)
                report.record_failure(
                    category=str(failure_record["failure_category"]),
                    phase=str(failure_record["phase"]),
                    example=failure_record,
                    max_examples_per_category=config.max_examples_per_category,
                )
                continue

            report.record_success(dag=example.dag, tactic_name=example.tactic_name)

        manifest = report.to_audit_manifest(
            dataset_name=config.dataset_name,
            output_root=output_root,
            sample_limit=config.sample_per_split,
        )
        write_manifest(output_root, split=split, manifest=manifest)
        split_reports[split] = report
        manifests[split] = manifest
        console_print(
            f"  Finished '{split}': attempted={report.attempted_count}, "
            f"success={report.success_count}, failure={report.failure_count}"
        )

    summary = _build_parser_audit_summary(
        config=config,
        manifests=manifests,
        split_reports=split_reports,
    )
    summary_json_path = write_parser_audit_json(output_root, summary)
    summary_md_path = write_parser_audit_markdown(
        output_root,
        _render_parser_audit_markdown(summary),
    )

    console_print(f"\n  Wrote parser audit JSON    : {summary_json_path}")
    console_print(f"  Wrote parser audit Markdown: {summary_md_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit parser coverage on LeanDojo proof states")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME, help="Dataset name to stream from Hugging Face")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits to audit")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_AUDIT_OUTPUT_ROOT), help="Output directory for parser audit reports")
    parser.add_argument("--sample-per-split", type=int, default=None, help="Optional limit of examples to process per split")
    parser.add_argument("--max-examples-per-category", type=int, default=5, help="Maximum representative failure examples to keep per category")
    parser.add_argument("--force", action="store_true", help="Overwrite the output root if it already exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = ParserAuditConfig(
            dataset_name=args.dataset_name,
            splits=tuple(_normalize_splits(args.splits)),
            output_root=Path(args.output_root),
            sample_per_split=args.sample_per_split,
            max_examples_per_category=args.max_examples_per_category,
            force=args.force,
        )
        run_parser_audit(config)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0
