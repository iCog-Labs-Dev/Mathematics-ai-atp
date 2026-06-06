from __future__ import annotations

import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .dataset import canonicalize_split_name
from .labels import UNKNOWN_TACTIC
from .reporting import console_print
from .training import (
    _load_checkpoint,
    _use_cuda_amp,
    build_baseline_model,
    load_baseline_config,
    load_prepared_metadata,
    resolve_device,
    PreparedGraphDataset,
    REQUIRED_DATA_FIELDS,
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {index: token for token, index in vocab.items()}


def _normalize_batch_strings(value, batch_size: int) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value) for _ in range(batch_size)]


def _normalize_batch_ints(value, batch_size: int) -> list[int]:
    if torch.is_tensor(value):
        flattened = value.view(-1).tolist()
        return [int(item) for item in flattened]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value) for _ in range(batch_size)]


def load_run_summary(run_dir: str | Path) -> dict[str, object]:
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run directory '{run_dir}' is missing 'summary.json'.")
    return _read_json(summary_path)


def load_metrics_history(run_dir: str | Path) -> list[dict[str, object]]:
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run directory '{run_dir}' is missing 'metrics.jsonl'.")
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _build_analysis_loader(run_dir: Path, split: str):
    config = load_baseline_config(run_dir / "config.json")
    metadata = load_prepared_metadata(config.prepared_root)
    dataset = PreparedGraphDataset(
        metadata,
        split=split,
        edge_mode=config.edge_mode,
        required_fields=REQUIRED_DATA_FIELDS,
    )

    # Keep analysis loaders single-process on Windows. These reports are
    # throughput-insensitive, and this avoids multiprocessing permission issues
    # in sandboxed or desktop Python environments.
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return config, metadata, loader


def _build_per_tactic_summary(
    records: list[dict[str, object]],
    *,
    min_support: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if bool(record["is_unknown_target"]):
            continue
        grouped[str(record["true_tactic"])].append(record)

    summaries: list[dict[str, object]] = []
    for tactic_name, tactic_records in grouped.items():
        support = len(tactic_records)
        top1_correct = sum(1 for record in tactic_records if bool(record["correct_top1"]))
        top5_correct = sum(1 for record in tactic_records if bool(record["correct_top5"]))
        summaries.append(
            {
                "tactic_name": tactic_name,
                "support": support,
                "top1_accuracy": top1_correct / support,
                "top5_accuracy": top5_correct / support,
            }
        )

    summaries.sort(key=lambda item: (-int(item["support"]), str(item["tactic_name"])))
    hardest = [
        item
        for item in summaries
        if int(item["support"]) >= min_support
    ]
    hardest.sort(key=lambda item: (float(item["top1_accuracy"]), -int(item["support"]), str(item["tactic_name"])))
    return summaries, hardest[:10]


def _build_confusion_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    confusions: Counter[tuple[str, str]] = Counter()
    for record in records:
        if bool(record["is_unknown_target"]) or bool(record["correct_top1"]):
            continue
        confusions[(str(record["true_tactic"]), str(record["predicted_top1"]))] += 1
    return [
        {"true_tactic": true_tactic, "predicted_tactic": predicted_tactic, "count": count}
        for (true_tactic, predicted_tactic), count in confusions.most_common(15)
    ]


def _build_error_samples(records: list[dict[str, object]]) -> list[dict[str, object]]:
    errors = [
        record
        for record in records
        if not bool(record["is_unknown_target"]) and not bool(record["correct_top1"])
    ]
    errors.sort(key=lambda item: (-float(item["predicted_top1_confidence"]), str(item["true_tactic"])))
    return errors[:25]


def _render_analysis_markdown(analysis: dict[str, object]) -> str:
    overall = analysis["overall"]
    lines = [
        f"# Run Analysis ({analysis['split']})",
        "",
        f"- run dir: `{analysis['run_dir']}`",
        f"- checkpoint: `{analysis['checkpoint']}`",
        f"- epoch: `{analysis['epoch']}`",
        f"- top-1: `{overall['top1_accuracy']:.4f}`",
        f"- top-5: `{overall['top5_accuracy']:.4f}`",
        f"- loss: `{overall['loss']:.4f}`",
        f"- known labels: `{overall['known_label_count']}`",
        f"- unknown labels excluded: `{overall['unknown_label_excluded_count']}`",
        "",
        "## Hardest Tactics",
        "",
        "| Tactic | Support | Top-1 | Top-5 |",
        "| --- | ---: | ---: | ---: |",
    ]

    hardest_tactics = list(analysis["hardest_tactics"])
    if hardest_tactics:
        for item in hardest_tactics:
            lines.append(
                f"| {item['tactic_name']} | {item['support']} | "
                f"{item['top1_accuracy']:.4f} | {item['top5_accuracy']:.4f} |"
            )
    else:
        lines.append("| none | 0 | 0.0000 | 0.0000 |")

    lines.extend(["", "## Common Confusions", ""])
    confusions = list(analysis["common_confusions"])
    if confusions:
        for item in confusions:
            lines.append(
                f"- `{item['true_tactic']}` -> `{item['predicted_tactic']}`: {item['count']}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Sample Errors", ""])
    errors = list(analysis["sample_errors"])
    if errors:
        for record in errors[:10]:
            lines.append(
                f"- `{record['true_tactic']}` predicted as `{record['predicted_top1']}` "
                f"(confidence={record['predicted_top1_confidence']:.4f}) "
                f"at row `{record['row_index']}` in `{record['theorem']}`"
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def analyze_saved_run(
    run_dir: str | Path,
    *,
    split: str,
    top_k: int = 5,
    min_support: int = 20,
) -> dict[str, object]:
    run_directory = Path(run_dir)
    if not run_directory.exists():
        raise FileNotFoundError(f"Run directory '{run_directory}' does not exist.")

    canonical_split = canonicalize_split_name(split)
    if canonical_split not in {"val", "test"}:
        raise ValueError("Analysis split must be either 'val' or 'test'.")
    if top_k < 1:
        raise ValueError("Analysis parameter 'top_k' must be positive.")
    if min_support < 1:
        raise ValueError("Analysis parameter 'min_support' must be positive.")

    config, metadata, loader = _build_analysis_loader(run_directory, canonical_split)
    device = resolve_device(config.device)
    model = build_baseline_model(metadata, config).to(device)
    checkpoint_path = run_directory / "best.pt"
    checkpoint = _load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    use_amp = _use_cuda_amp(device, config)
    id_to_tactic = _invert_vocab(metadata.tactic_vocab)
    records: list[dict[str, object]] = []
    loss_sum = 0.0
    known_label_count = 0

    console_print(f"  Analyzing {canonical_split} split ({len(loader)} batches)...")
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch.y.numel())
            row_indices = _normalize_batch_ints(batch.row_index, batch_size)
            theorems = _normalize_batch_strings(batch.theorem, batch_size)
            true_tactic_names = _normalize_batch_strings(batch.tactic_name, batch_size)

            batch = batch.to(device, non_blocking=(device.type == "cuda" and config.training.pin_memory))
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch)
                probabilities = logits.softmax(dim=1)
                known_mask = batch.y.view(-1) != metadata.unknown_tactic_id
                if bool(known_mask.any()):
                    known_logits = logits[known_mask]
                    known_targets = batch.y.view(-1)[known_mask]
                    batch_loss = torch.nn.functional.cross_entropy(known_logits, known_targets)
                    loss_sum += float(batch_loss.item()) * int(known_targets.numel())
                    known_label_count += int(known_targets.numel())

            targets = batch.y.view(-1).detach().cpu()
            top_k_size = min(top_k, logits.size(1))
            topk = logits.topk(top_k_size, dim=1)
            topk_ids = topk.indices.detach().cpu()
            top1_confidences = probabilities.gather(1, topk.indices[:, :1]).squeeze(1).detach().cpu()

            for index in range(batch_size):
                true_id = int(targets[index].item())
                predicted_ids = [int(item) for item in topk_ids[index].tolist()]
                predicted_tactics = [
                    id_to_tactic.get(predicted_id, UNKNOWN_TACTIC)
                    for predicted_id in predicted_ids
                ]
                is_unknown_target = true_id == metadata.unknown_tactic_id
                record = {
                    "row_index": row_indices[index],
                    "theorem": theorems[index],
                    "true_tactic": true_tactic_names[index],
                    "true_tactic_id": true_id,
                    "predicted_top1": predicted_tactics[0],
                    "predicted_top1_id": predicted_ids[0],
                    "predicted_top1_confidence": float(top1_confidences[index].item()),
                    "predicted_topk": predicted_tactics,
                    "predicted_topk_ids": predicted_ids,
                    "is_unknown_target": is_unknown_target,
                    "correct_top1": (not is_unknown_target) and (predicted_ids[0] == true_id),
                    "correct_top5": (not is_unknown_target) and (true_id in predicted_ids),
                }
                records.append(record)

    known_records = [record for record in records if not bool(record["is_unknown_target"])]
    known_count = len(known_records)
    top1_correct = sum(1 for record in known_records if bool(record["correct_top1"]))
    top5_correct = sum(1 for record in known_records if bool(record["correct_top5"]))
    per_tactic_summary, hardest_tactics = _build_per_tactic_summary(records, min_support=min_support)
    common_confusions = _build_confusion_summary(records)
    sample_errors = _build_error_samples(records)

    analysis = {
        "run_dir": str(run_directory),
        "split": canonical_split,
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "overall": {
            "top1_accuracy": top1_correct / known_count if known_count else 0.0,
            "top5_accuracy": top5_correct / known_count if known_count else 0.0,
            "loss": loss_sum / known_label_count if known_label_count else 0.0,
            "known_label_count": known_count,
            "unknown_label_excluded_count": len(records) - known_count,
            "evaluated_count": len(records),
        },
        "curve_context": load_metrics_history(run_directory),
        "per_tactic_summary": per_tactic_summary,
        "hardest_tactics": hardest_tactics,
        "common_confusions": common_confusions,
        "sample_errors": sample_errors,
    }

    predictions_path = _write_jsonl(run_directory / f"predictions_{canonical_split}.jsonl", records)
    analysis_json_path = _write_json(run_directory / f"analysis_{canonical_split}.json", analysis)
    analysis_markdown_path = _write_text(
        run_directory / f"analysis_{canonical_split}.md",
        _render_analysis_markdown(analysis),
    )
    analysis["artifacts"] = {
        "predictions_jsonl": str(predictions_path),
        "analysis_json": str(analysis_json_path),
        "analysis_markdown": str(analysis_markdown_path),
    }
    _write_json(run_directory / f"analysis_{canonical_split}.json", analysis)
    return analysis


def compare_saved_runs(run_dirs: list[str | Path]) -> dict[str, object]:
    if not run_dirs:
        raise ValueError("At least one run directory must be provided for comparison.")

    runs: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_directory = Path(run_dir)
        summary = load_run_summary(run_directory)
        config = load_baseline_config(run_directory / "config.json")
        runs.append(
            {
                "run_dir": str(run_directory),
                "run_name": run_directory.name,
                "best_epoch": int(summary["best_epoch"]),
                "val_top1": float(summary["best_validation"]["top1_accuracy"]),
                "val_top5": float(summary["best_validation"]["top5_accuracy"]),
                "test_top1": float(summary["test_evaluation"]["top1_accuracy"]),
                "test_top5": float(summary["test_evaluation"]["top5_accuracy"]),
                "top1_gap": float(summary["best_validation"]["top1_accuracy"]) - float(summary["test_evaluation"]["top1_accuracy"]),
                "edge_mode": config.edge_mode,
                "hidden_dim": config.model.hidden_dim,
                "num_layers": config.model.num_layers,
                "use_node_type": config.use_node_type,
                "amp_enabled": bool(summary.get("amp_enabled", False)),
            }
        )

    runs.sort(key=lambda item: (-float(item["test_top1"]), -float(item["val_top1"]), str(item["run_name"])))
    return {"runs": runs}


def render_run_comparison_markdown(comparison: dict[str, object]) -> str:
    lines = [
        "# Run Comparison",
        "",
        "| Run | Best Epoch | Val Top-1 | Val Top-5 | Test Top-1 | Test Top-5 | Top-1 Gap | Edge Mode | Hidden | Layers | Node Type | AMP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for item in comparison["runs"]:
        lines.append(
            f"| {item['run_name']} | {item['best_epoch']} | "
            f"{item['val_top1']:.4f} | {item['val_top5']:.4f} | "
            f"{item['test_top1']:.4f} | {item['test_top5']:.4f} | "
            f"{item['top1_gap']:.4f} | {item['edge_mode']} | "
            f"{item['hidden_dim']} | {item['num_layers']} | "
            f"{item['use_node_type']} | {item['amp_enabled']} |"
        )
    return "\n".join(lines) + "\n"


def build_analyze_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a trained baseline run in detail")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to a completed run directory")
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Which split to analyze",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many predicted tactics to retain per example")
    parser.add_argument(
        "--min-support",
        type=int,
        default=20,
        help="Minimum support before a tactic is considered in the hardest-tactic table",
    )
    return parser


def build_compare_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare finished baseline runs")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the markdown comparison table",
    )
    return parser


def analyze_main(argv: list[str] | None = None) -> int:
    parser = build_analyze_arg_parser()
    args = parser.parse_args(argv)

    try:
        splits = ["val", "test"] if args.split == "both" else [args.split]
        for split in splits:
            analysis = analyze_saved_run(
                args.run_dir,
                split=split,
                top_k=args.top_k,
                min_support=args.min_support,
            )
            console_print(f"  Wrote analysis summary   : {analysis['artifacts']['analysis_json']}")
            console_print(f"  Wrote analysis markdown  : {analysis['artifacts']['analysis_markdown']}")
            console_print(f"  Wrote prediction records : {analysis['artifacts']['predictions_jsonl']}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0


def compare_main(argv: list[str] | None = None) -> int:
    parser = build_compare_arg_parser()
    args = parser.parse_args(argv)

    try:
        comparison = compare_saved_runs(args.run_dirs)
        markdown = render_run_comparison_markdown(comparison)
        if args.output:
            output_path = _write_text(Path(args.output), markdown)
            console_print(f"  Wrote comparison table   : {output_path}")
        else:
            console_print(markdown)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0
