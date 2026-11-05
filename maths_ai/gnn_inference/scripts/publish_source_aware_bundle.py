"""Revalidate and publish a retained verified deployment without recopying files."""

import argparse
import asyncio
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.bundle_publication import RetainedBundleError
from maths_ai.gnn_inference.atp_lean_gnn.deployment_migration import publish_retained_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backing", type=Path, required=True, help="Retained sibling .bundle directory.")
    parser.add_argument("--output", type=Path, required=True, help="Original intended publication path.")
    args = parser.parse_args()
    try:
        result = asyncio.run(publish_retained_bundle(args.backing, output=args.output))
    except (RetainedBundleError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Verified deployment published: {result}")
    print(f"Keep its backing directory: {result.parent.resolve()}")


if __name__ == "__main__":
    main()
