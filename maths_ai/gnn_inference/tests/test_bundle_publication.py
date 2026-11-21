"""Filesystem publication owns its backing data without overwriting other output."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import errno
import os
from pathlib import Path
from threading import Barrier

import pytest

from maths_ai.gnn_inference.atp_lean_gnn import bundle_publication as publication


def owner(tmp_path):
    result = publication.BundlePublication.create(publication.preflight_output(tmp_path / "deployment"))
    (result.backing / "verified").write_text("bundle contents")
    return result


def test_relative_publication_is_complete_and_cleanup_keeps_its_backing(tmp_path):
    bundle = owner(tmp_path)
    bundle.validated = True
    bundle.publish()
    bundle.cleanup()
    assert os.readlink(bundle.output) == bundle.backing.name
    assert (bundle.output / "verified").read_text() == "bundle contents"
    assert bundle.output.resolve() == bundle.backing


@pytest.mark.parametrize("kind", ["file", "empty-directory", "directory", "symlink", "dangling-symlink"])
def test_existing_destination_is_never_replaced(tmp_path, kind):
    bundle = owner(tmp_path)
    output = bundle.output
    if kind == "file":
        output.write_text("preserve")
    elif kind in {"directory", "empty-directory"}:
        output.mkdir()
        if kind == "directory":
            (output / "preserve").write_text("preserve")
    else:
        target = tmp_path / "other"
        if kind == "symlink":
            target.mkdir()
        output.symlink_to(target)
    original = output.lstat()
    bundle.validated = True
    with pytest.raises(publication.RetainedBundleError) as failure:
        bundle.publish()
    assert isinstance(failure.value.__cause__, FileExistsError)
    assert output.lstat().st_ino == original.st_ino
    bundle.cleanup()
    assert bundle.backing.exists()


def test_concurrent_publishers_have_one_winner_and_preserve_both_bundles(tmp_path):
    one = owner(tmp_path)
    two = publication.BundlePublication.create(one.output)
    (two.backing / "verified").write_text("second bundle")
    one.validated = two.validated = True
    barrier = Barrier(2)

    def publish(bundle):
        barrier.wait()
        try:
            bundle.publish()
            return "published"
        except publication.RetainedBundleError:
            return "refused"
        finally:
            bundle.cleanup()

    with ThreadPoolExecutor(max_workers=2) as threads:
        assert sorted(threads.map(publish, [one, two])) == ["published", "refused"]
    assert one.output.resolve() in {one.backing, two.backing}
    assert one.backing.exists() and two.backing.exists()


def test_error_reported_after_link_creation_is_reconciled(tmp_path, monkeypatch):
    bundle = owner(tmp_path)
    bundle.validated = True
    symlink = os.symlink

    def reported_error(*args, **kwargs):
        symlink(*args, **kwargs)
        raise OSError(errno.EIO, "server reported failure after committing")

    monkeypatch.setattr(os, "symlink", reported_error)
    bundle.publish()
    bundle.cleanup()
    assert bundle.output.resolve() == bundle.backing and bundle.backing.exists()


def test_cancellation_after_link_creation_keeps_reachable_data(tmp_path, monkeypatch):
    bundle = owner(tmp_path)
    bundle.validated = True
    symlink = os.symlink

    def cancelled(*args, **kwargs):
        symlink(*args, **kwargs)
        raise asyncio.CancelledError()

    monkeypatch.setattr(os, "symlink", cancelled)
    with pytest.raises(asyncio.CancelledError):
        try:
            bundle.publish()
        finally:
            bundle.cleanup()
    assert bundle.backing.exists() and bundle.output.resolve() == bundle.backing


def test_cleanup_only_removes_owned_incomplete_data(tmp_path):
    incomplete = owner(tmp_path)
    incomplete.cleanup()
    assert not incomplete.backing.exists()
    uncertain = owner(tmp_path)
    moved = tmp_path / "moved"
    uncertain.backing.rename(moved)
    uncertain.backing.mkdir()
    (uncertain.backing / "foreign").write_text("preserve")
    uncertain.cleanup()
    assert (uncertain.backing / "foreign").read_text() == "preserve"
    assert (moved / "verified").exists()


def test_unsupported_symlink_preflight_is_actionable_and_cleans_probe(tmp_path, monkeypatch):
    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "symlinks unsupported")

    monkeypatch.setattr(os, "symlink", unsupported)
    with pytest.raises(OSError, match="publication preflight"):
        publication.preflight_output(tmp_path / "deployment")
    assert not list(tmp_path.iterdir())


def test_idempotency_requires_the_exact_backing_identity(tmp_path):
    bundle = owner(tmp_path)
    bundle.validated = True
    bundle.publish()
    bundle.publish()
    with pytest.raises(ValueError, match="owned sibling"):
        publication.publish_bundle(bundle.backing, bundle.output, {"device": -1, "inode": -1})
