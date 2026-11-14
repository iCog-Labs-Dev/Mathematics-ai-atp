#!/usr/bin/env python3
"""
Mathematics AI ATP - Asset Fetcher

Fetches the pinned v2 model package from HuggingFace Hub. The package includes
the policy, tokenizer, retriever, index, and bound lemma corpus.

Responsibilities:
1. Load the YAML config pointed to by MATHS_AI_MODEL_CONFIG
2. Call huggingface_hub.snapshot_download() for the complete clean-root package
3. Skip re-download if .snapshot_revision marker file matches pinned revision (idempotent)
4. Exit non-zero with clear message if repo/revision doesn't resolve
5. No auth required for public repos (HF_TOKEN env var wired in for future private repos)

Usage:
    python scripts/fetch_assets.py

Environment variables:
    MATHS_AI_MODEL_CONFIG: Path to model config YAML (default: maths_ai/config/models/pointer_gnn.yaml)
    MATHS_AI_CORPUS_CONFIG: Optional legacy corpus config
    MATHS_AI_DATA_ROOT: Root directory for downloaded assets (default: current directory)
    HF_TOKEN: Optional HuggingFace token for private repos
"""

import os
import shutil
import sys
from pathlib import Path
import yaml
import subprocess

# Try to import huggingface_hub, provide helpful error if not installed
try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as e:
    print(f"ERROR: Required package 'huggingface_hub' not installed: {e}", file=sys.stderr)
    print("Install it with: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


def load_config(config_path: Path) -> dict:
    """Load and validate a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    
    # Validate required fields
    required_fields = ['name', 'source', 'repo_type', 'repo_id', 'revision', 'local_subdir']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config: {config_path}")
    
    return config


def get_marker_file(local_dir: Path) -> Path:
    """Get the path to the snapshot revision marker file."""
    return local_dir / ".snapshot_revision"


def check_revision_matches(local_dir: Path, expected_revision: str) -> bool:
    """Check if the existing snapshot matches the expected revision."""
    marker_file = get_marker_file(local_dir)
    if not marker_file.exists():
        return False
    
    try:
        with open(marker_file, 'r') as f:
            actual_revision = f.read().strip()
        return actual_revision == expected_revision
    except (IOError, OSError):
        return False


def write_revision_marker(local_dir: Path, revision: str):
    """Write the revision marker file after successful download."""
    marker_file = get_marker_file(local_dir)
    with open(marker_file, 'w') as f:
        f.write(revision)


def validate_reused_bundle(local_dir: Path, expected_revision: str) -> str | None:
    """Validate a reused clean-root bundle before trusting it.

    A matching ``.snapshot_revision`` marker alone must not cause a stale or
    truncated v2 volume to be accepted (the marker is written before anything
    can verify the payload). Returns ``None`` when the strict manifest/hash
    validation passes; otherwise a human-readable failure. The strict loader
    is stdlib-only, so this check is safe inside the fetch environment. If the
    workspace is not importable the volume is treated as invalid (a redundant
    re-download is cheap; silently serving a stale bundle is not).
    """
    try:
        from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle
    except Exception as exc:  # pragma: no cover - import-environment dependent
        return f"bundle validator unavailable ({exc}); rejecting reused volume"
    try:
        load_joint_source_bundle(local_dir, expected_revision=expected_revision)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def fetch_asset(config: dict, data_root: Path, hf_token: str | None = None) -> Path:
    """
    Fetch a single asset (model or dataset) from HuggingFace Hub.
    
    Args:
        config: Dictionary with asset configuration
        data_root: Root directory for downloads
        hf_token: Optional HuggingFace token
        
    Returns:
        Path to the downloaded asset directory
        
    Raises:
        SystemExit: If download fails
    """
    name = config['name']
    repo_id = config['repo_id']
    revision = config['revision']
    repo_type = config['repo_type']
    local_subdir = config['local_subdir']
    subfolder = config.get('subfolder')
    
    local_dir = data_root / local_subdir
    
    print(f"Fetching {repo_type} '{name}' from {repo_id}@{revision}...")
    print(f"  Target directory: {local_dir}")
    
    # Check if already downloaded with correct revision. For clean-root model
    # bundles the marker alone is not sufficient: the payload must also pass
    # the strict completeness validation before the download is skipped.
    if local_dir.exists() and check_revision_matches(local_dir, revision):
        if (local_dir / "bundle.json").is_file():
            validation_error = validate_reused_bundle(local_dir, revision)
            if validation_error is None:
                print(f"  Already downloaded with matching revision, skipping.")
                return local_dir
            print("  Reused model snapshot failed completeness validation; replacing it.")
            print(f"    ({validation_error})")
        else:
            print(f"  Already downloaded with matching revision, skipping.")
            return local_dir

    if local_dir.exists():
        print("  Existing snapshot is stale or unmarked; replacing it.")
        shutil.rmtree(local_dir)
    
    # Create parent directory if needed
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # A clean-root release must not be narrowed to the legacy subfolder.
    allow_patterns = None
    if subfolder:
        allow_patterns = [f"{subfolder}/**", "vocab/**"]

    try:
        print(f"  Downloading from HuggingFace Hub...")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            local_dir=str(local_dir),
            token=hf_token,
            allow_patterns=allow_patterns,
            ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.bin"],
        )
    except HfHubHTTPError as e:
        print(f"ERROR: Failed to resolve {repo_type} '{repo_id}' at revision '{revision}': {e}", file=sys.stderr)
        print(f"  This may mean the repo/revision doesn't exist or is private.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to download {repo_type} '{repo_id}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Write revision marker
    write_revision_marker(local_dir, revision)
    print(f"  Download complete. Revision marker written to {get_marker_file(local_dir)}")
    
    return local_dir


def get_config_path(env_var: str, default_path: Path | None = None) -> Path:
    """Get config path from environment variable or use default."""
    env_path = os.getenv(env_var)
    if env_path:
        return Path(env_path)
    if default_path:
        return default_path
    raise ValueError(f"Environment variable {env_var} not set and no default provided")


def main():
    # Get configuration paths
    scripts_dir = Path(__file__).resolve().parent
    workspace_root = scripts_dir.parent
    
    default_model_config = workspace_root / "maths_ai" / "config" / "models" / "pointer_gnn.yaml"
    default_corpus_config = workspace_root / "maths_ai" / "config" / "corpus" / "lemma_corpus_v1.yaml"
    
    model_config_path = get_config_path("MATHS_AI_MODEL_CONFIG", default_model_config)
    corpus_config_path = get_config_path("MATHS_AI_CORPUS_CONFIG", default_corpus_config)
    
    # Get data root
    data_root_str = os.getenv("MATHS_AI_DATA_ROOT", str(workspace_root))
    data_root = Path(data_root_str).resolve()
    
    # Get HF token (optional, for private repos)
    hf_token = os.getenv("HF_TOKEN") or None
    
    print("=" * 60)
    print("Mathematics AI ATP - Asset Fetcher")
    print("=" * 60)
    print(f"Model config: {model_config_path}")
    print(f"Corpus config: {corpus_config_path}")
    print(f"Data root: {data_root}")
    print()
    
    # Load and fetch model config
    try:
        model_config = load_config(model_config_path)
        print(f"[MODEL] {model_config['name']}")
        fetch_asset(model_config, data_root, hf_token)
    except Exception as e:
        print(f"ERROR: Failed to process model config: {e}", file=sys.stderr)
        sys.exit(1)
    
    print()
    
    # v2 packages the bound corpus and retriever in the model snapshot. Keep
    # the separate corpus fetch available only for explicit legacy deployments.
    if os.getenv("MATHS_AI_CORPUS_CONFIG"):
        try:
            corpus_config = load_config(corpus_config_path)
            print(f"[CORPUS] {corpus_config['name']}")
            fetch_asset(corpus_config, data_root, hf_token)
        except Exception as e:
            print(f"ERROR: Failed to process corpus config: {e}", file=sys.stderr)
            sys.exit(1)
    
    print()
    print("All assets fetched successfully!")


if __name__ == "__main__":
    main()
