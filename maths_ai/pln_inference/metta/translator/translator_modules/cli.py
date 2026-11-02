"""
CLI argument parsing and main orchestration.

This is the outermost layer of the translator. It:
  - Parses command-line arguments
  - Loads input JSON
  - Drives the generation workflow (run_demo)
  - Dispatches ranking mode

Can be run directly: python -m translator_modules.cli --input tests.json --output ./generated_metta
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from pantograph_async import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from pantograph_async import (
    PantographGoalRejected,
    PantographTacticRejected,
)
import json
import os
import shutil
from typing import Any

from maths_ai.pln_inference.dependencies import PLNDependencies

from .extractor import essentialize_subgoal, proof_state_after_tactics
from .parser import to_plain
from .runner import (
    parse_and_rank_logs,
    print_ranked_results,
    safe_name,
    DynamicThompsonSampler,
    write_runner_script,
)


def load_input_json(input_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    if isinstance(input_data, dict):
        if "goal" in input_data:
            return [input_data]
        raise ValueError("JSON format not recognized. Missing 'goal' key.")

    if isinstance(input_data, list):
        for i, item in enumerate(input_data):
            if not isinstance(item, dict):
                raise ValueError(f"Input item {i} is not a JSON object.")
        return input_data

    raise ValueError("Input JSON must be either an object or a list of objects.")


async def run_demo(
    input_path: str,
    output_dir: str,
    *,
    depth: int = 10,
    runner_name: str = "run_all_generated.sh",
    stop_on_error: bool = False,
    env: PantographEnv,
    normalize_variables: bool = False,
    chainer_root: Path | None = None,
    axioms_path: str | None = None,
) -> dict[str, Any]:
    env.verify()
    imports = PLNDependencies.resolve(chainer_root, axioms_path).imports()
    input_data = load_input_json(input_path)
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(
            f"Refusing to overwrite nonempty translator output: {output_dir}"
        )
    async with await create_model_sexpr_client(env) as server:
        os.makedirs(output_dir, exist_ok=True)
        generated_items: list[dict[str, Any]] = []
        test_summaries: list[dict[str, Any]] = []

        for idx, test_case in enumerate(input_data):
            test_name = test_case.get("name", f"test_{idx + 1}")
            goal = test_case.get("goal")
            tactics = test_case.get("tactics", [])

            if not goal:
                print(f"⚠️ Warning: Skipping '{test_name}' because no 'goal' was found.")
                test_summaries.append(
                    {
                        "test_name": test_name,
                        "status": "skipped_no_goal",
                    }
                )
                continue

            if not isinstance(tactics, list):
                raise ValueError(
                    f"Tactics for '{test_name}' must be a list of tactic strings."
                )

            print(f"\n{'=' * 80}")
            print(f"🚀 Running Test: {test_name}")
            print(f"Loaded Goal: {goal}")
            print(f"Loaded Tactics: {tactics}")

            try:
                goals = await proof_state_after_tactics(
                    server,
                    goal,
                    tactics,
                )

            except (PantographGoalRejected, PantographTacticRejected) as e:
                print(f"❌ Error processing '{test_name}' in Pantograph: {e}")
                test_summaries.append(
                    {
                        "test_name": test_name,
                        "goal": goal,
                        "tactics": tactics,
                        "status": "pantograph_error",
                        "error": str(e),
                    }
                )
                continue

            safe_test_name = safe_name(test_name)

            if not goals:
                complete_file_path = os.path.join(
                    output_dir, f"{safe_test_name}_complete.metta"
                )
                with open(complete_file_path, "w", encoding="utf-8") as f:
                    f.write(
                        "\n\n".join(
                            [
                                imports,
                                "; No subgoals remain. Proof complete!",
                            ]
                        )
                    )

                print("✅ No subgoals remain. Proof complete.")
                test_summaries.append(
                    {
                        "test_name": test_name,
                        "goal": goal,
                        "tactics": tactics,
                        "status": "proof_complete",
                        "proof_complete_file": os.path.abspath(complete_file_path),
                    }
                )
                continue

            test_summary = {
                "test_name": test_name,
                "goal": goal,
                "tactics": tactics,
                "status": "generated",
                "num_subgoals": len(goals),
                "subgoals": [],
            }

            for i, g in enumerate(goals):
                print(f"\nGoal {i} Processed:")

                script, cleanup, hyp_records, query_formula = essentialize_subgoal(
                    g,
                    mode="dynamic",
                    depth=depth,
                    normalize_variables=normalize_variables,
                )

                print(script)

                metta_file_name = f"{safe_test_name}_goal_{i}.metta"
                metta_file_path = os.path.abspath(
                    os.path.join(output_dir, metta_file_name)
                )
                log_file_path = os.path.abspath(
                    os.path.join(output_dir, f"{safe_test_name}_goal_{i}.log")
                )

                subgoal_script = [
                    imports,
                    f"; === Test: {test_name} ===",
                    f"; === Goal index: {i} ===",
                    f"; === Target formula: {query_formula} ===",
                    "",
                    script,
                ]

                if cleanup:
                    subgoal_script.append(f"; --- Cleanup ---\n{cleanup}")

                with open(metta_file_path, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(subgoal_script))

                item = {
                    "test_name": test_name,
                    "goal_index": i,
                    "metta_path": metta_file_path,
                    "log_path": log_file_path,
                    "target_formula": query_formula,
                    "hypotheses": hyp_records,
                }

                generated_items.append(item)
                test_summary["subgoals"].append(item)

                print(f"✅ Wrote subgoal file: {metta_file_path}")
                print(f"📝 Expected log path: {log_file_path}")

            test_summaries.append(test_summary)

        runner_path = None
        if generated_items:
            runner_path = write_runner_script(
                generated_items,
                output_dir,
                runner_name=runner_name,
                stop_on_error=stop_on_error,
            )
            print("\n" + "=" * 80)
            print(f"✅ Wrote runner script to: {runner_path}")
            print(f"Run it with: bash {runner_path}")

        manifest = {
            "input_path": os.path.abspath(input_path),
            "output_dir": os.path.abspath(output_dir),
            "runner_path": os.path.abspath(runner_path) if runner_path else None,
            "depth": depth,
            "normalize_variables": normalize_variables,
            "generated_items": generated_items,
            "tests": test_summaries,
        }

        manifest_path = os.path.join(output_dir, "generated_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"✅ Wrote manifest to: {manifest_path}")

        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Lean goals to one PeTTaChainer .metta file per subgoal, "
            "generate a runner script, and optionally rank logs."
        )
    )

    mode_group = parser.add_mutually_exclusive_group(required=False)

    mode_group.add_argument(
        "--rank-manifest",
        type=str,
        default=None,
        help="Parse and rank logs using an existing generated_manifest.json.",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Path to input JSON containing goal/tactics objects.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory for generated .metta, .log, manifest, and runner files.",
    )

    parser.add_argument(
        "--axioms-path",
        type=str,
        default=None,
        help="Optional axiom file override; defaults to the configured fork collection.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="PeTTaChainer query depth used in generated .metta files.",
    )

    parser.add_argument(
        "--runner-name",
        type=str,
        default="run_all_generated.sh",
        help="Name of the generated runner shell script.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Make the runner stop if any generated .metta file fails.",
    )

    parser.add_argument(
        "--lean-import",
        action="append",
        default=None,
        help="Lean module import for Pantograph. Can be passed multiple times. Defaults to Init.",
    )

    parser.add_argument(
        "--ranking-output",
        type=str,
        default=None,
        help="Output JSON path for ranking results when using --rank-manifest.",
    )

    parser.add_argument(
        "--disable-random-fallback",
        action="store_true",
        help=(
            "Disable random fallback scoring when no STVs are found in any "
            "available subgoal log.  Only relevant when --fallback-strategy=random."
        ),
    )

    parser.add_argument(
        "--fallback-strategy",
        choices=["random", "thompson"],
        default="random",
        help=(
            "Fallback strategy when no subgoal log contains an STV.  "
            "'random' (default): sample from a configured distribution.  "
            "'thompson': per-subgoal Thompson sampling with Beta(alpha,beta) posteriors."
        ),
    )

    parser.add_argument(
        "--thompson-state-input",
        type=str,
        default=None,
        help="Path to a JSON file with previous Thompson-sampler state (subgoal_key -> {alpha, beta}).",
    )

    parser.add_argument(
        "--thompson-state-output",
        type=str,
        default=None,
        help="Save updated Thompson-sampler state to this path (default: embedded in ranking output).",
    )

    parser.add_argument(
        "--fallback-distribution",
        choices=["uniform", "beta"],
        default="uniform",
        help="Random fallback distribution used when no subgoal log contains an STV.",
    )

    parser.add_argument(
        "--fallback-low",
        type=float,
        default=0.0,
        help="Lower bound for random fallback scores.",
    )

    parser.add_argument(
        "--fallback-high",
        type=float,
        default=1.0,
        help="Upper bound for random fallback scores.",
    )

    parser.add_argument(
        "--fallback-alpha",
        type=float,
        default=2.0,
        help="Alpha parameter for beta fallback distribution.",
    )

    parser.add_argument(
        "--fallback-beta",
        type=float,
        default=2.0,
        help="Beta parameter for beta fallback distribution.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible fallback scores.",
    )

    parser.add_argument(
        "--normalize-variables",
        action="store_true",
        help=(
            "Map Lean variables/propositions to v0,v1,... as before. "
            "By default this adjusted version preserves concrete Lean names."
        ),
    )

    parser.add_argument(
        "--chainer-root",
        type=Path,
        help="Local PeTTaChainer checkout; defaults to PETTACHAINER_ROOT.",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--pantograph-repl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rank_manifest:
        thompson_sampler: ThompsonSampler | None = None
        if args.fallback_strategy == "thompson" and args.thompson_state_input:
            thompson_sampler = ThompsonSampler.load_from(args.thompson_state_input)

        ranked = parse_and_rank_logs(
            args.rank_manifest,
            ranking_output=args.ranking_output,
            random_fallback=not args.disable_random_fallback,
            fallback_strategy=args.fallback_strategy,
            fallback_distribution=args.fallback_distribution,
            fallback_low=args.fallback_low,
            fallback_high=args.fallback_high,
            fallback_alpha=args.fallback_alpha,
            fallback_beta=args.fallback_beta,
            random_seed=args.random_seed,
            thompson_sampler=thompson_sampler,
            thompson_state_output=args.thompson_state_output,
        )
        print_ranked_results(ranked)
    else:
        if not args.input:
            raise SystemExit(
                "Error: --input is required unless --rank-manifest is used."
            )
        if not args.output:
            raise SystemExit(
                "Error: --output is required unless --rank-manifest is used."
            )

        if args.source_root is None or args.pantograph_repl is None:
            raise SystemExit(
                "--source-root and --pantograph-repl are required for generation."
            )
        env = PantographEnv(
            source_root=args.source_root,
            pantograph_repl=args.pantograph_repl,
            imports=tuple(args.lean_import or ["Mathlib"]),
        )
        asyncio.run(
            run_demo(
                args.input,
                args.output,
                depth=args.depth,
                runner_name=args.runner_name,
                stop_on_error=args.stop_on_error,
                env=env,
                normalize_variables=args.normalize_variables,
                chainer_root=args.chainer_root,
                axioms_path=args.axioms_path,
            )
        )


if __name__ == "__main__":
    main()
