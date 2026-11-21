"""Real Git regressions for exact on-disk provenance and read-only status."""

import hashlib
import os
import subprocess
from unittest.mock import patch

import pytest

from maths_ai.core.git_worktree import (
    StatusEntry,
    dirty_provenance,
    parse_status,
    read_status,
)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    (tmp_path / "tracked").write_bytes(b"base")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path


@pytest.mark.parametrize(
    "name",
    [
        "normal",
        " leading ",
        "trailing ",
        '"quoted"',
        "a -> b",
        "tab\tfile",
        "new\nline",
        "café",
        os.fsdecode(b"invalid-\xff"),
    ],
)
def test_exact_filenames_and_disk_hashes(repository, name):
    path = repository / name
    path.write_bytes(b"staged")
    git(repository, "add", "--", name)
    git(repository, "commit", "-qm", "file")
    path.write_bytes(b"executed")
    for quote in ("true", "false"):
        git(repository, "config", "core.quotePath", quote)
        git(repository, "config", "status.relativePaths", "true")
        entries = read_status(repository)
        assert entries == (StatusEntry(" ", "M", name),)
        hashes, metadata = dirty_provenance(repository, entries)
        assert hashes == {name: hashlib.sha256(b"executed").hexdigest()}
        assert metadata[name]["kind"] == "file"


def test_staged_and_mixed_hash_disk_not_index(repository):
    path = repository / "tracked"
    path.write_bytes(b"staged")
    git(repository, "add", ".")
    assert read_status(repository)[0].index_status == "M"
    path.write_bytes(b"executed")
    entries = read_status(repository)
    assert entries[0].worktree_status == "M"
    assert (
        dirty_provenance(repository, entries)[0]["tracked"]
        == hashlib.sha256(b"executed").hexdigest()
    )


def test_rename_source_and_destination(repository):
    git(repository, "mv", "tracked", 'destination -> "new"')
    entries = read_status(repository)
    assert entries == (StatusEntry("R", " ", 'destination -> "new"', "tracked"),)
    assert (
        dirty_provenance(repository, entries)[1][entries[0].path]["original_path"]
        == "tracked"
    )


def test_copy_record(repository):
    (repository / "copy").write_bytes(b"base")
    (repository / "tracked").write_bytes(b"changed")
    git(repository, "add", ".")
    git(repository, "config", "status.renames", "copies")
    git(repository, "config", "diff.renames", "copies")
    entries = read_status(repository)
    assert StatusEntry("C", " ", "copy", "tracked") in entries


def test_deletion_untracked_and_symlink(repository, tmp_path):
    (repository / "tracked").unlink()
    (repository / "untracked").write_bytes(b"new")
    (repository / "link").symlink_to("/outside/checkout")
    hashes, metadata = dirty_provenance(repository, read_status(repository))
    assert hashes["tracked"] is None
    assert metadata["tracked"]["kind"] == "deleted"
    assert hashes["link"] == hashlib.sha256(b"/outside/checkout").hexdigest()
    assert metadata["link"]["kind"] == "symlink"
    assert metadata["untracked"]["kind"] == "file"


def test_unexpected_disappearance_and_unreadable_file_fail(repository):
    (repository / "tracked").write_bytes(b"changed")
    entries = read_status(repository)
    with patch(
        "maths_ai.core.git_worktree.os.open", side_effect=PermissionError("denied")
    ):
        with pytest.raises(RuntimeError, match="Incomplete provenance"):
            dirty_provenance(repository, entries)
    (repository / "tracked").unlink()
    with pytest.raises(RuntimeError, match="Incomplete provenance"):
        dirty_provenance(repository, entries)


@pytest.mark.parametrize(
    "data",
    [
        b" M file",
        b" M \0",
        b"XY file\0",
        b"R  new\0",
        b"R  new\0\0",
        b"?? ../outside\0",
        b" M f\0junk\0",
    ],
)
def test_malformed_status_fails(data):
    with pytest.raises(ValueError):
        parse_status(data)


def test_clean(repository):
    assert read_status(repository) == ()


def test_dirty_submodule_overrides_user_ignore(repository, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.com")
    (source / "file").write_text("base")
    git(source, "add", ".")
    git(source, "commit", "-qm", "base")
    git(
        repository,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(source),
        "sub",
    )
    git(repository, "commit", "-qam", "submodule")
    git(repository, "config", "submodule.sub.ignore", "all")
    (repository / "sub/file").write_text("changed")
    entries = read_status(repository)
    hashes, metadata = dirty_provenance(repository, entries)
    assert hashes["sub"] is None
    assert metadata["sub"]["kind"] == "submodule"
    assert metadata["sub"]["dirty"] is True


def test_diagnostic_target_identity_uses_shared_provenance(repository, monkeypatch):
    from maths_ai.hybrid_reasoner import mcts_diagnostics

    name = "tracked"
    (repository / name).write_bytes(b"executed")
    monkeypatch.setattr(
        mcts_diagnostics,
        "__file__",
        str(repository / "maths_ai/hybrid_reasoner/mcts_diagnostics.py"),
    )
    monkeypatch.setattr(mcts_diagnostics, "SEARCH_SOURCES", (name,))
    monkeypatch.setattr(
        mcts_diagnostics, "_installed_client", lambda: {"commit": "test"}
    )
    report = mcts_diagnostics._target_identity()
    assert report["worktree_clean"] is False
    assert report["dirty_sha256"] == {name: hashlib.sha256(b"executed").hexdigest()}
    assert report["dirty_entries"][name]["kind"] == "file"
    git(repository, "checkout", "--", name)
    report = mcts_diagnostics._target_identity()
    assert report["worktree_clean"] is True
    assert report["dirty_sha256"] == {}
    assert report["dirty_entries"] == {}


def test_uninitialized_submodule_is_not_the_parent_repository(repository):
    revision = git(repository, "rev-parse", "HEAD").decode().strip()
    git(repository, "update-index", "--add", "--cacheinfo", f"160000,{revision},sub")
    (repository / "sub").mkdir()
    _, metadata = dirty_provenance(repository, read_status(repository))
    assert metadata["sub"]["kind"] == "submodule"
    assert metadata["sub"]["initialized"] is False
    assert metadata["sub"]["revision"] is None
    assert metadata["sub"]["index_revision"] == revision


def test_staged_deletion_recreated_on_disk(repository):
    git(repository, "rm", "tracked")
    (repository / "tracked").write_bytes(b"recreated")
    hashes, metadata = dirty_provenance(repository, read_status(repository))
    assert hashes["tracked"] == hashlib.sha256(b"recreated").hexdigest()
    statuses = [metadata["tracked"], *metadata["tracked"]["additional_status"]]
    assert {
        (entry["index_status"], entry["worktree_status"]) for entry in statuses
    } == {("D", " "), ("?", "?")}
