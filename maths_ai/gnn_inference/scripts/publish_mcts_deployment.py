"""Plan or upload a complete MCTS serving release without copying its index."""

import argparse
import asyncio
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.hub_deployment import (
    DEFAULT_REPO, prepare_release, upload_release,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed timestamped RL run with config.json.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Explicit best.pt or last.pt from the completed run.")
    parser.add_argument("--receipt-dir", type=Path, required=True, help="Persistent local plan/receipt directory; reuse only to retry the exact release.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--version", default="v2.0.1", help="New branch name, in vMAJOR.MINOR.PATCH format.")
    parser.add_argument("--base-revision", default="main")
    parser.add_argument("--device", default="cpu", help="Serving device in generated configuration: cpu or cuda:N.")
    parser.add_argument("--upload", action="store_true", help="Create the release branch and upload. Without this flag only validates/plans locally.")
    args = parser.parse_args(argv)
    manifest, sources = asyncio.run(prepare_release(
        args.run_dir, args.checkpoint, args.receipt_dir, version=args.version, device=args.device,
    ))
    print(f"Destination: {args.repo}, new branch {args.version}")
    for name, record in sorted(manifest["files"].items()):
        print(f"{record['size']:>14,d}  {name}")
    print(f"Total serving bytes: {sum(item['size'] for item in manifest['files'].values()):,}")
    print("Additional checkpoint fields retained:", manifest["bindings"]["extra_checkpoint_fields"])
    print("Lean runtime compatibility will also be checked at API startup.")
    if not args.upload:
        print(f"Local plan only. Review {args.receipt_dir}; rerun with --upload to publish.")
        return 0
    result = upload_release(manifest, sources, args.receipt_dir, repo=args.repo, base_revision=args.base_revision)
    print("Published:", result["url"])
    print("Pinned commit:", result["commit"])
    print("Manifest SHA-256:", result["manifest_sha256"])
    print("Remote verification:", result["remote_verification"])
    print("Receipt:", args.receipt_dir / "upload-receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
