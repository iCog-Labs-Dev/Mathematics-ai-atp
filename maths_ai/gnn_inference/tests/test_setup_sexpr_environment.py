from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pantograph_async import PantographConfigurationError, compatibility

from maths_ai.gnn_inference.scripts import setup_sexpr_environment as setup
from maths_ai.gnn_inference.scripts.setup_sexpr_environment import (
    LEAN_TOOLCHAIN,
    _build_manifest,
    _ensure_toolchain,
    _prepare_companion,
    _require_stdbuf,
)


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class SetupSExprEnvironmentTests(unittest.TestCase):
    def test_stdbuf_dependency_resolves_from_runtime_path(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "gnu-stdbuf"
            binary.touch()
            link = Path(directory) / "stdbuf"
            link.symlink_to(binary)
            for discovered in (binary, link):
                with self.subTest(discovered=discovered), patch(
                    "maths_ai.gnn_inference.scripts.setup_sexpr_environment.shutil.which",
                    return_value=str(discovered),
                ) as which:
                    self.assertEqual(_require_stdbuf(), str(binary.resolve()))
                    which.assert_called_once_with("stdbuf")

    @patch(
        "maths_ai.gnn_inference.scripts.setup_sexpr_environment.shutil.which",
        return_value=None,
    )
    def test_missing_stdbuf_is_a_named_setup_error(self, _which):
        with self.assertRaisesRegex(RuntimeError, "Install coreutils"):
            _require_stdbuf()

    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._run")
    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._output")
    def test_already_installed_toolchain_is_not_reinstalled(self, output, run):
        output.return_value = f"{LEAN_TOOLCHAIN}\nleanprover/lean4:v4.29.1 (default)"
        _ensure_toolchain(LEAN_TOOLCHAIN)
        output.assert_called_once_with("elan", "toolchain", "list")
        run.assert_not_called()

    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._run")
    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._output")
    def test_missing_toolchain_is_installed(self, output, run):
        output.return_value = "leanprover/lean4:v4.29.1"
        _ensure_toolchain(LEAN_TOOLCHAIN)
        run.assert_called_once_with("elan", "toolchain", "install", LEAN_TOOLCHAIN)


class PinsTests(unittest.TestCase):
    def test_pins_come_from_the_installed_client(self):
        base = compatibility.companion_base()
        self.assertEqual(setup.PANTOGRAPH_COMMIT, base["base_commit"])
        self.assertEqual(setup.PANTOGRAPH_URL, base["source_url"])
        self.assertEqual(setup.MATHLIB_COMMIT, base["mathlib_expected_pin"])
        self.assertEqual(LEAN_TOOLCHAIN, base["toolchain"])


class CompanionCheckoutTests(unittest.TestCase):
    """Real git checkouts with a small stand-in for the companion patch."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        upstream = self.root / "upstream"
        upstream.mkdir()
        _git("init", "-q", cwd=upstream)
        for key, value in (("user.name", "test"), ("user.email", "t@example.com")):
            _git("config", key, value, cwd=upstream)
        (upstream / "lean-toolchain").write_text("leanprover/lean4:v4.10.0-rc1\n")
        (upstream / "Repl.lean").write_text("def main := 0\n")
        _git("add", ".", cwd=upstream)
        _git("commit", "-q", "-m", "base", cwd=upstream)
        self.commit = _git("rev-parse", "HEAD", cwd=upstream)
        (upstream / "Repl.lean").write_text("def main := 1\n")
        (upstream / "Validation.lean").write_text("-- companion\n")
        _git("add", "-N", "Validation.lean", cwd=upstream)
        patch_bytes = _git("diff", "--binary", cwd=upstream).encode() + b"\n"
        digests = {
            name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
            for name in ("Repl.lean", "Validation.lean")
        }
        _git("checkout", "-q", "--", "Repl.lean", cwd=upstream)
        _git("rm", "-q", "--cached", "Validation.lean", cwd=upstream)
        (upstream / "Validation.lean").unlink()

        def verify(destination):
            for name, digest in digests.items():
                path = Path(destination) / name
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise PantographConfigurationError(f"differ: {name}")

        fake = SimpleNamespace(
            companion_patch=lambda: patch_bytes,
            companion_files=lambda: frozenset(digests),
            verify_companion_sources=verify,
        )
        self.destination = self.root / "Pantograph"
        for target, value in (
            ("compatibility", fake),
            ("PANTOGRAPH_URL", str(upstream)),
            ("PANTOGRAPH_COMMIT", self.commit),
        ):
            patcher = patch.object(setup, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare_companion(self):
        _prepare_companion(
            setup._inspect(
                setup.PANTOGRAPH_URL, self.commit, self.destination, companion=True
            )
        )

    def test_fresh_checkout_applies_and_verifies_the_patch(self):
        self.prepare_companion()
        self.assertEqual(_git("rev-parse", "HEAD", cwd=self.destination), self.commit)
        self.assertEqual(
            (self.destination / "Repl.lean").read_text(), "def main := 1\n"
        )
        self.assertTrue((self.destination / "Validation.lean").is_file())

    def test_rerun_reuses_an_exact_application_without_touching_files(self):
        self.prepare_companion()
        before = (self.destination / "Repl.lean").stat().st_mtime_ns
        self.prepare_companion()
        self.assertEqual((self.destination / "Repl.lean").stat().st_mtime_ns, before)

    def test_user_changes_are_refused_and_preserved(self):
        self.prepare_companion()
        (self.destination / "Notes.lean").write_text("-- mine\n")
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            self.prepare_companion()
        self.assertEqual((self.destination / "Notes.lean").read_text(), "-- mine\n")
        self.assertEqual(
            (self.destination / "Repl.lean").read_text(), "def main := 1\n"
        )

    def test_build_manifest_records_source_and_binary_identities(self):
        self.prepare_companion()
        repl = self.destination / ".lake" / "build" / "bin" / "repl"
        repl.parent.mkdir(parents=True)
        repl.write_bytes(b"built")
        manifest = _build_manifest(self.destination, self.destination, repl)
        self.assertEqual(manifest["pantograph_base_commit"], self.commit)
        self.assertEqual(manifest["mathlib_commit"], self.commit)
        self.assertEqual(manifest["toolchain"], LEAN_TOOLCHAIN)
        self.assertEqual(manifest["repl_sha256"], hashlib.sha256(b"built").hexdigest())
        self.assertEqual(manifest["patch_sha256"], setup.COMPANION["patch_sha256"])

    def test_edited_companion_file_is_refused_and_preserved(self):
        self.prepare_companion()
        (self.destination / "Repl.lean").write_text("def main := 2\n")
        with self.assertRaises(PantographConfigurationError):
            self.prepare_companion()
        self.assertEqual(
            (self.destination / "Repl.lean").read_text(), "def main := 2\n"
        )


class CheckoutRefusalTests(CompanionCheckoutTests):
    """Refusals preserve repository bytes and never run mutation commands."""

    def snapshot(self):
        return {
            str(path.relative_to(self.destination)): path.read_bytes()
            for path in self.destination.rglob("*")
            if path.is_file()
        }

    def assert_refusal_preserves(self, operation, pattern="Refusing|refusing"):
        before = self.snapshot()
        with patch.object(setup, "_run") as run:
            with self.assertRaisesRegex(RuntimeError, pattern):
                operation()
            run.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def clone_clean(self):
        setup._checkout(
            setup._inspect(setup.PANTOGRAPH_URL, self.commit, self.destination)
        )

    def checkout(self):
        setup._checkout(
            setup._inspect(setup.PANTOGRAPH_URL, self.commit, self.destination)
        )

    def test_dirty_matching_origin(self):
        self.clone_clean()
        (self.destination / "Repl.lean").write_text("-- user edit")
        self.assert_refusal_preserves(self.checkout)

    def test_mismatched_origin_dirty_and_clean(self):
        self.clone_clean()
        _git("remote", "set-url", "origin", "wrong-origin", cwd=self.destination)
        self.assert_refusal_preserves(self.checkout)
        (self.destination / "Repl.lean").write_text("-- user edit")
        self.assert_refusal_preserves(self.checkout)

    def test_missing_origin(self):
        self.clone_clean()
        _git("remote", "remove", "origin", cwd=self.destination)
        self.assert_refusal_preserves(self.checkout)

    def test_staged_deletion_even_with_empty_worktree(self):
        self.clone_clean()
        _git("rm", "-q", "Repl.lean", "lean-toolchain", cwd=self.destination)
        self.assert_refusal_preserves(self.checkout)

    def test_companion_reuse_requires_matching_origin(self):
        self.prepare_companion()
        _git("remote", "set-url", "origin", "wrong-origin", cwd=self.destination)
        self.assert_refusal_preserves(lambda: self.prepare_companion())

    def test_staged_companion_rejected(self):
        self.prepare_companion()
        _git("add", ".", cwd=self.destination)
        self.assert_refusal_preserves(lambda: self.prepare_companion())

    def test_archive_and_nested_root_rejected(self):
        self.destination.mkdir()
        (self.destination / "user-data").write_text("preserve")
        self.assert_refusal_preserves(self.checkout)
        self.clone_into_parent_and_check_nested()

    def clone_into_parent_and_check_nested(self):
        _git("init", "-q", cwd=self.destination)
        nested = self.destination / "nested"
        nested.mkdir()
        self.assert_refusal_preserves(
            lambda: setup._checkout(
                setup._inspect(setup.PANTOGRAPH_URL, self.commit, nested)
            ),
            "repository root",
        )

    def test_symlink_destination_rejected(self):
        self.clone_clean()
        link = self.root / "link"
        link.symlink_to(self.destination, target_is_directory=True)
        self.assert_refusal_preserves(
            lambda: setup._checkout(
                setup._inspect(setup.PANTOGRAPH_URL, self.commit, link)
            ),
            "symlink",
        )

    def test_snapshot_change_stops_materialization(self):
        self.clone_clean()
        inspection = setup._inspect(setup.PANTOGRAPH_URL, self.commit, self.destination)
        _git("config", "user.name", "concurrent-editor", cwd=self.destination)
        self.assert_refusal_preserves(
            lambda: setup._checkout(inspection), "changed after inspection"
        )

    def test_both_destinations_inspected_before_toolchain_or_checkout(self):
        output = self.root / "output"
        output.mkdir()
        pantograph = output / "Pantograph"
        setup._checkout(setup._inspect(setup.PANTOGRAPH_URL, self.commit, pantograph))
        (pantograph / "Notes").write_text("mine")
        with (
            patch.object(setup, "_ensure_toolchain") as toolchain,
            patch.object(setup, "_run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                setup.main(["--output-root", str(output)])
            toolchain.assert_not_called()
            run.assert_not_called()
        self.assertFalse((output / "mathlib4").exists())

    def test_clean_checkout_can_be_reused(self):
        self.clone_clean()
        self.checkout()
        self.assertEqual(_git("rev-parse", "HEAD", cwd=self.destination), self.commit)
        self.assertEqual(setup.read_status(self.destination), ())

    def test_linked_worktree_with_git_file_is_supported(self):
        self.clone_clean()
        linked = self.root / "linked"
        _git(
            "worktree",
            "add",
            "--detach",
            str(linked),
            self.commit,
            cwd=self.destination,
        )
        self.assertTrue((linked / ".git").is_file())
        inspection = setup._inspect(setup.PANTOGRAPH_URL, self.commit, linked)
        setup._checkout(inspection)
        self.assertEqual(_git("rev-parse", "HEAD", cwd=linked), self.commit)

    def test_dirty_submodule_cannot_be_hidden_by_ignore_config(self):
        self.clone_clean()
        _git(
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            setup.PANTOGRAPH_URL,
            "sub",
            cwd=self.destination,
        )
        _git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qam",
            "submodule",
            cwd=self.destination,
        )
        _git("config", "submodule.sub.ignore", "all", cwd=self.destination)
        (self.destination / "sub/Repl.lean").write_text("-- changed")
        self.assert_refusal_preserves(self.checkout)

    def test_companion_rename_is_refused(self):
        self.prepare_companion()
        _git("mv", "Repl.lean", "Renamed.lean", cwd=self.destination)
        self.assert_refusal_preserves(self.prepare_companion)

    def test_companion_wrong_base_is_refused(self):
        self.prepare_companion()
        _git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "different base",
            cwd=self.destination,
        )
        self.assert_refusal_preserves(self.prepare_companion)

    def test_index_flag_change_is_detected_after_inspection(self):
        self.clone_clean()
        inspection = setup._inspect(setup.PANTOGRAPH_URL, self.commit, self.destination)
        _git("update-index", "--assume-unchanged", "Repl.lean", cwd=self.destination)
        self.assert_refusal_preserves(
            lambda: setup._checkout(inspection), "changed after inspection"
        )

    def test_merge_conflict_is_refused(self):
        self.clone_clean()
        _git("config", "user.name", "Test", cwd=self.destination)
        _git("config", "user.email", "test@example.com", cwd=self.destination)
        _git("checkout", "-qb", "other", cwd=self.destination)
        (self.destination / "Repl.lean").write_text("other branch")
        _git("commit", "-qam", "other", cwd=self.destination)
        _git("checkout", "--detach", self.commit, cwd=self.destination)
        (self.destination / "Repl.lean").write_text("current branch")
        _git("commit", "-qam", "current", cwd=self.destination)
        result = subprocess.run(
            ["git", "merge", "other"], cwd=self.destination, capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assert_refusal_preserves(self.checkout)

    def test_symbolic_head_change_at_same_commit_is_detected(self):
        self.clone_clean()
        inspection = setup._inspect(setup.PANTOGRAPH_URL, self.commit, self.destination)
        branch_ref = _git(
            "for-each-ref", "--format=%(refname)", "refs/heads/", cwd=self.destination
        ).splitlines()[0]
        _git("symbolic-ref", "HEAD", branch_ref, cwd=self.destination)
        self.assertEqual(
            _git("rev-parse", "HEAD", cwd=self.destination), inspection.head
        )
        self.assert_refusal_preserves(
            lambda: setup._checkout(inspection), "changed after inspection"
        )


if __name__ == "__main__":
    unittest.main()
