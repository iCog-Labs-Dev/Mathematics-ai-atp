"""Export the accepted pre-template joint model as a clean-root offline release."""

import argparse
import json
import subprocess
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import export_joint_source_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("joint-checkpoint", "local-checkpoint", "text-checkpoint", "text-index", "prepared-root",
                 "candidate-dir", "corpus-path", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--license-file", type=Path,
                        default=Path(__file__).resolve().parents[1] / "release_assets/APACHE-2.0.txt")
    parser.add_argument("--license-id", default="apache-2.0", help="Model-artifact license; source code keeps its own license")
    parser.add_argument("--tokenizer-source", help="Local cached encoder/tokenizer name or directory; no downloads")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                   capture_output=True, text=True).stdout.strip()
    manifest = export_joint_source_bundle(**vars(args), source_commit=source_commit,
        progress=lambda message: print(message, flush=True))
    print(json.dumps({"bundle": str(args.output_dir.resolve()), "format": manifest["format"],
                      "files": len(manifest["files"]), "load_back_verified": True}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
