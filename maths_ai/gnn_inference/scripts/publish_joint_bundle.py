"""Replace the model repo's current file tree in one commit, then tag that revision."""

import argparse
import json
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle


def publish_joint_bundle(bundle_dir, *, repo_id, release_tag, previous_tag=None, dry_run=False, api=None):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    if not release_tag or release_tag == previous_tag:
        raise ValueError("new and previous release tags must be distinct")
    root = Path(bundle_dir).resolve()
    system = load_joint_source_bundle(root)
    files = set(system.manifest["files"]) | {"bundle.json"}
    api = api or HfApi()
    previous_sha = api.model_info(repo_id, revision="main").sha
    tags = {tag.name: tag.target_commit for tag in api.list_repo_refs(repo_id, repo_type="model").tags}
    if release_tag in tags:
        raise ValueError("release tag already exists; published tags are immutable")
    if previous_tag and previous_tag in tags and tags[previous_tag] != previous_sha:
        raise ValueError("previous tag exists at another revision; choose the intended previous-release tag or omit it")
    remote_files = set(api.list_repo_files(repo_id, revision=previous_sha, repo_type="model"))
    deletions = sorted(remote_files - files - {".gitattributes"})
    result = {"repo_id": repo_id, "parent_commit": previous_sha, "release_tag": release_tag,
              "previous_tag": previous_tag, "add_or_replace": sorted(files), "delete_from_new_revision": deletions,
              "historical_revisions_retained": True, "dry_run": dry_run}
    if dry_run:
        return result
    if previous_tag and previous_tag not in tags:
        api.create_tag(repo_id, tag=previous_tag, revision=previous_sha, repo_type="model")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name)) for name in sorted(files)]
    operations.extend(CommitOperationDelete(path_in_repo=name) for name in deletions)
    commit = api.create_commit(repo_id, repo_type="model", revision="main", parent_commit=previous_sha,
        commit_message=f"Publish {release_tag}: complete joint-source model and matched retrieval", operations=operations)
    # Check the committed tree before attaching the release name. Keep Hub's
    # .gitattributes tracking file; old paths survive in the previous commit.
    published = set(api.list_repo_files(repo_id, revision=commit.oid, repo_type="model")) - {".gitattributes"}
    if published != files:
        raise RuntimeError(f"unexpected published tree at {commit.oid}; release tag was not created")
    api.create_tag(repo_id, tag=release_tag, revision=commit.oid, repo_type="model")
    result.update(commit=commit.oid, commit_url=commit.commit_url,
                  release_url=f"https://huggingface.co/{repo_id}/tree/{release_tag}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="iCogLabs-Dev/Mathlib-Sexpr-GNN")
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--previous-tag")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = publish_joint_bundle(**vars(args))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
