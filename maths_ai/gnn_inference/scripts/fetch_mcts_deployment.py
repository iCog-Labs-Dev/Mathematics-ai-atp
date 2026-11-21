"""Download, verify and publish a persistent MCTS serving bundle."""

import argparse
import asyncio
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.hub_deployment import DEFAULT_REPO, fetch_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", required=True, help="Full commit from the upload receipt is recommended; branch names resolve once.")
    parser.add_argument("--version", default="v2.0.1")
    parser.add_argument("--manifest-sha256", help="Trusted manifest digest from the upload receipt.")
    parser.add_argument("--output", type=Path, required=True, help="New persistent output; published as a relative link to a retained sibling directory.")
    parser.add_argument("--resume", type=Path, help="Original backing directory printed after an interrupted download.")
    parser.add_argument("--max-bytes", type=int, default=100 * 1024**3, help="Maximum total declared serving bytes (default 100 GiB).")
    args = parser.parse_args(argv)
    if args.max_bytes < 1:
        parser.error("--max-bytes must be positive")
    output = asyncio.run(fetch_release(
        args.output, repo=args.repo, revision=args.revision, version=args.version,
        expected_manifest=args.manifest_sha256, max_bytes=args.max_bytes, resume=args.resume,
    ))
    print("Verified deployment:", output)
    print("Artifact mount:", output / "artifacts")
    print("Configuration mount:", output / "config")
    print("Keep the hidden sibling backing directory; the output is a relative symlink.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
