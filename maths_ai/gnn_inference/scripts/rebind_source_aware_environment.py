"""Verify a REPL-only update and preserve the existing actor-critic weights."""

import argparse
import asyncio
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.deployment_migration import rebind_environment
from maths_ai.gnn_inference.atp_lean_gnn.bundle_publication import RetainedBundleError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing source-aware RL JSON configuration.")
    parser.add_argument("--repl", type=Path, required=True, help="New compiled REPL executable.")
    parser.add_argument("--old-repl", type=Path, help="Required for inventories without component evidence.")
    parser.add_argument("--output", type=Path, required=True, help="New persistent bundle symlink; must not exist.")
    args = parser.parse_args()
    try:
        output = asyncio.run(rebind_environment(
            args.config, repl=args.repl, old_repl=args.old_repl, output=args.output,
        ))
    except (RetainedBundleError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Verified fresh-run configuration: {output}")
    print("All actor and critic tensors were preserved. Start a new run without --resume.")
    print(f"Keep both the publication symlink and its backing directory: {output.parent.resolve()}")


if __name__ == "__main__":
    main()
