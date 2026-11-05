"""Export allowed declarations from the actual deployed Lean environment."""

import argparse
import json
from pathlib import Path
from maths_ai.gnn_inference.atp_lean_gnn.declaration_inventory import generate_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "repl", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--imports", nargs="+", default=["Init", "Mathlib"])
    parser.add_argument(
        "--exclude-file",
        type=Path,
        required=True,
        help="JSON list of benchmark-disallowed declarations.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inventory = generate_inventory(
        args.source_root, args.repl, args.imports,
        json.loads(args.exclude_file.read_text()),
    )
    args.output.write_text(
        json.dumps(
            inventory,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
