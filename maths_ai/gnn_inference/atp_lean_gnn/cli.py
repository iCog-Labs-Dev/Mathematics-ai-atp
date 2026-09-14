from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import DATASET_NAME, load_dataset_row
from .graph import proof_state_to_dag, write_dag_json
from .pyg import build_vocab, dag_to_pyg
from .reporting import console_print, format_dag_summary
from .state import parse_state
from .visualize import visualize_dag


DEMO_STATE = (
    "n  : \u2115\n"
    "m  : \u2115\n"
    "hn : Even n\n"
    "hm : Even m\n"
    "\u22a2  Even (n + m)"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lean proof state -> DAG toolkit")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--demo", action="store_true", help="Use the built-in Even(n+m) example")
    source_group.add_argument("--state-file", type=str, help="Load a proof state from a local text file")
    source_group.add_argument("--state-text", type=str, help="Use a proof state string directly")
    source_group.add_argument("--row", type=int, default=None, help="Dataset row to load")

    parser.add_argument("--split", type=str, default="train", help="Dataset split when using --row (default: train)")
    parser.add_argument("--no-viz", action="store_true", help="Skip opening a browser")
    parser.add_argument("--out", type=str, default=None, help="Save HTML to this path")
    parser.add_argument("--json-out", type=str, default=None, help="Save the graph as JSON")
    parser.add_argument("--pyg-summary", action="store_true", help="Print PyG tensor shapes for the current graph")
    parser.add_argument("--bidirectional", action="store_true", help="Add reverse edges when building the PyG summary")
    return parser


def _resolve_source(args: argparse.Namespace) -> tuple[str, str, str, dict[str, object]]:
    if args.demo:
        console_print("\n  Using built-in demo: Even(n+m)")
        return (
            DEMO_STATE,
            "even_add (demo)",
            "(demo)",
            {"source": "demo", "theorem": "even_add (demo)", "tactic": "(demo)"},
        )

    if args.state_file:
        path = Path(args.state_file)
        state_text = path.read_text(encoding="utf-8")
        console_print(f"\n  Loading proof state from {path}...")
        return (
            state_text,
            path.stem,
            "",
            {"source": "file", "path": str(path)},
        )

    if args.state_text:
        console_print("\n  Using proof state provided on the command line...")
        return (
            args.state_text,
            "ad-hoc proof state",
            "",
            {"source": "cli"},
        )

    row_index = 0 if args.row is None else args.row
    console_print(f"\n  Loading row {row_index} from {DATASET_NAME} ({args.split})...")
    row = load_dataset_row(row_index, split=args.split)
    return row.state, row.theorem, row.tactic, row.metadata()


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        state_text, theorem, tactic, metadata = _resolve_source(args)
    except (RuntimeError, IndexError, FileNotFoundError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    parsed_state = parse_state(state_text)
    dag = proof_state_to_dag(parsed_state)

    console_print(format_dag_summary(dag, parsed_state, theorem=theorem, tactic=tactic))
    console_print()

    if args.json_out:
        json_metadata = dict(metadata)
        json_metadata["proof_state"] = parsed_state.as_dict()
        json_path = write_dag_json(dag, args.json_out, metadata=json_metadata)
        console_print(f"  Saved graph JSON: {json_path}")

    if args.pyg_summary:
        vocab = build_vocab([dag])
        data = dag_to_pyg(dag, vocab, add_reverse_edges=args.bidirectional)
        console_print(
            "  PyG summary: "
            f"x={tuple(data.x.shape)}, "
            f"node_type={tuple(data.node_type.shape)}, "
            f"edge_index={tuple(data.edge_index.shape)}, "
            f"vocab_size={len(vocab)}"
        )

    if args.out or not args.no_viz:
        html_path = visualize_dag(
            dag,
            title=f"DAG - {theorem}" if theorem else "Lean Proof State DAG",
            theorem=theorem,
            tactic=tactic,
            open_browser=not args.no_viz,
            output_path=args.out,
        )
        action = "Saved HTML" if args.no_viz else "Opened HTML"
        console_print(f"  {action}: {html_path}")
    elif args.no_viz:
        console_print("  (--no-viz flag set, skipping browser)")

    return 0
