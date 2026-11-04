"""Compare the graph the model gets for each replayed dataset row with its prepared graph.

For every row admission accepts, the replayed goal (with its source model
identity) is turned into a graph exactly as RL does, and compared node for node
with the graph built from the row's stored model columns, as the prepared data
holds it. Pass --prepared-root to also count labels outside the prepared node
vocabulary. Writes a new JSON report; an existing one is never overwritten.
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from pantograph_async import PantographEnv

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DATASET_NAME, DATASET_REVISION, iter_dataset_rows
from maths_ai.gnn_inference.atp_lean_gnn.graph import (
    dag_fingerprint,
    model_goal_to_dag,
    proof_state_to_dag,
)
from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_goal
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client,
    materialize_seed,
)
from maths_ai.lean_replay import ReplayInputError, render_replay
from pantograph_async import PantographProtocolError, PantographResourceError, PantographTransportError


def graph_view(dag) -> tuple:
    return (
        dag_fingerprint(dag),
        [node.as_dict() for node in dag.nodes],
        [list(edge) for edge in dag.edges],
        dag.expression_root_id,
        dag.state_root_id,
    )


async def run(args) -> dict:
    env = PantographEnv(
        source_root=args.source_root,
        pantograph_repl=args.pantograph_repl,
        imports=("Mathlib",),
        timeout=args.request_timeout,
        server_start_timeout_s=600,
    )
    vocab = None
    if args.prepared_root is not None:
        from maths_ai.gnn_inference.atp_lean_gnn.training import load_prepared_metadata

        vocab = load_prepared_metadata(args.prepared_root, splits=()).node_vocab
    counts = Counter()
    unknown = Counter()
    differences = []
    client = await create_model_sexpr_client(env)
    try:
        for row in iter_dataset_rows(
            dataset_name=args.dataset_name,
            split="train",
            sample_limit=args.max_scan_rows,
            revision=args.revision,
        ):
            try:
                seed = row.structured_seed()
                if len(render_replay(seed).expression) > args.max_replay_chars:
                    continue
                async with client.state_scope() as scope:
                    state, goal, _ = await materialize_seed(scope, seed)
                    unmapped = execution_goal_to_goal(state.goals[0])
            except ReplayInputError:
                continue
            except PantographResourceError:
                counts["resource_limited"] += 1
                if not client.is_usable:
                    await client.aclose()
                    client = await create_model_sexpr_client(env)
                continue
            except (PantographTransportError, PantographProtocolError):
                if client.is_usable:
                    raise
                await client.aclose()
                client = await create_model_sexpr_client(env)
                continue
            counts["compared"] += 1
            prepared = graph_view(
                proof_state_to_dag(
                    row.state, goal_sexp=row.model_goal_sexp, hyp_sexps=list(row.model_hyp_sexps)
                )
            )
            replayed = model_goal_to_dag(goal)
            if graph_view(replayed) == prepared:
                counts["identical"] += 1
            else:
                counts["different"] += 1
                differences.append({"row_index": row.row_index, "theorem": row.theorem})
            if graph_view(model_goal_to_dag(unmapped)) == prepared:
                counts["identical_without_mapping"] += 1
            if vocab is not None:
                missing = [node.label for node in replayed.nodes if node.label not in vocab]
                unknown["rows_with_unknown_labels"] += bool(missing)
                unknown["unknown_labels"] += len(missing)
    finally:
        await client.aclose()
    return {
        "dataset": args.dataset_name,
        "revision": args.revision,
        "rows": f"train[0:{args.max_scan_rows}]",
        **counts,
        **({"prepared_root": str(args.prepared_root), **unknown} if vocab is not None else {}),
        "different_rows": differences[:50],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pantograph-repl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--max-scan-rows", type=int, default=500)
    parser.add_argument("--max-replay-chars", type=int, default=50000)
    parser.add_argument("--request-timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error(f"Report already exists: {args.report}")
    report = asyncio.run(run(args))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "different_rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
