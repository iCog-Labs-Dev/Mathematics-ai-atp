"""Export allowed declarations from the actual deployed Lean environment."""

import argparse
import json
from pathlib import Path
import subprocess
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import (
    environment_fingerprint,
)


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
    helper = (
        Path(__file__).resolve().parents[1] / "lean_extractor/export_declarations.lean"
    )
    result = subprocess.run(
        ["lake", "env", "lean", "--run", str(helper), *args.imports],
        cwd=args.source_root,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    declarations = json.loads(result.stdout)
    excluded = json.loads(args.exclude_file.read_text())
    if not isinstance(excluded, list) or not all(isinstance(x, str) for x in excluded):
        raise ValueError("Exclusions must be an explicit JSON list.")
    args.output.write_text(
        json.dumps(
            dict(
                version=1,
                environment_identity=environment_fingerprint(
                    args.source_root, args.repl, args.imports
                ),
                declarations=sorted(set(declarations)),
                excluded=sorted(set(excluded)),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
