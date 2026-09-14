#!/usr/bin/env python3
"""Pack many small prepared PyG files into sequential cache shards."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from maths_ai.gnn_inference.atp_lean_gnn.dataset import CANONICAL_SPLITS
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    PreparedGraphDataset,
    load_prepared_metadata,
)


def pack_prepared_dataset(
    prepared_root: str | Path,
    *,
    edge_mode: str = "bidirectional",
    chunk_size: int = 1024,
    io_threads: int = 8,
    force: bool = False,
) -> Path:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if io_threads < 0:
        raise ValueError("io_threads cannot be negative.")
    if edge_mode not in {"forward", "bidirectional"}:
        raise ValueError("edge_mode must be 'forward' or 'bidirectional'.")

    metadata = load_prepared_metadata(prepared_root)
    output_root = metadata.root / "packed" / edge_mode
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not force:
        return manifest_path

    building_root = output_root.with_name(f"{output_root.name}.building")
    if building_root.exists():
        shutil.rmtree(building_root)
    building_root.mkdir(parents=True)

    started = time.perf_counter()
    split_payloads: dict[str, dict[str, object]] = {}
    try:
        for split in CANONICAL_SPLITS:
            dataset = PreparedGraphDataset(
                metadata,
                split=split,
                edge_mode=edge_mode,
                io_threads=io_threads,
                cache_in_memory=False,
            )
            split_root = building_root / split
            split_root.mkdir(parents=True)
            chunk_names: list[str] = []

            for chunk_index, start in enumerate(range(0, len(dataset), chunk_size)):
                end = min(start + chunk_size, len(dataset))
                examples = dataset.__getitems__(list(range(start, end)))
                chunk_name = f"chunk_{chunk_index:06d}.pt"
                torch.save(examples, split_root / chunk_name)
                chunk_names.append(chunk_name)
                if chunk_index == 0 or (chunk_index + 1) % 25 == 0 or end == len(dataset):
                    elapsed = time.perf_counter() - started
                    print(
                        f"{split}: packed {end}/{len(dataset)} examples "
                        f"into {chunk_index + 1} chunks ({elapsed:.1f}s)",
                        flush=True,
                    )

            split_payloads[split] = {
                "count": len(dataset),
                "chunks": chunk_names,
            }

        manifest = {
            "version": 1,
            "edge_mode": edge_mode,
            "chunk_size": chunk_size,
            "splits": split_payloads,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (building_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if output_root.exists():
            if not force:
                raise FileExistsError(f"Packed cache already exists at '{output_root}'.")
            shutil.rmtree(output_root)
        building_root.rename(output_root)
    except Exception:
        shutil.rmtree(building_root, ignore_errors=True)
        raise

    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--edge-mode", choices=("forward", "bidirectional"), default="bidirectional")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--io-threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = pack_prepared_dataset(
        args.prepared_root,
        edge_mode=args.edge_mode,
        chunk_size=args.chunk_size,
        io_threads=args.io_threads,
        force=args.force,
    )
    print(f"Packed cache manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
