"""Verification of the Lean environment value object (no Lean, no subprocess).

``PantographEnv.verify`` is the whole point of these tests: it is the guard that
turns a misconfigured run into a one-second named error instead of a multi-minute
model load followed by ``Unknown identifier ℕ`` on every goal. Every check it
makes is a filesystem read, so all of it is testable without a REPL.
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv


def _lake_project(root: Path, toolchain: str = "leanprover/lean4:v4.29.1") -> Path:
    """Build the minimum tree ``verify`` accepts: a lakefile and a toolchain."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "lakefile.lean").write_text("import Lake\n")
    (root / "lean-toolchain").write_text(toolchain + "\n")
    return root


def _repl(root: Path, toolchain: str = "leanprover/lean4:v4.29.1") -> Path:
    """Lay out a REPL the way a Pantograph checkout does.

    ``_repl_toolchain_path`` reads ``repl.parents[3]/lean-toolchain``, which for
    ``<root>/.lake/build/bin/repl`` is ``<root>/lean-toolchain``.
    """
    binary = root / ".lake" / "build" / "bin" / "repl"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    (root / "lean-toolchain").write_text(toolchain + "\n")
    return binary


class DefaultsTests(unittest.TestCase):
    def test_bare_env_verifies_and_imports_init(self):
        env = PantographEnv()
        env.verify()  # no paths to check, no toolchains to compare
        self.assertEqual(env.imports, ("Init",))
        self.assertIsNone(env.source_root)
        self.assertIsNone(env.pantograph_repl)

    def test_describe_names_both_seams(self):
        text = PantographEnv().describe()
        self.assertIn("core Lean only", text)
        self.assertIn("bundled", text)
        self.assertIn("Init", text)


class SourceRootTests(unittest.TestCase):
    def test_missing_directory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = PantographEnv(source_root=Path(tmp) / "absent")
            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                env.verify()

    def test_directory_without_lakefile_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "lakefile"):
                PantographEnv(source_root=Path(tmp)).verify()

    def test_lakefile_toml_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lakefile.toml").write_text("name = \"proj\"\n")
            (root / "lean-toolchain").write_text("leanprover/lean4:v4.29.1\n")
            PantographEnv(source_root=root, pantograph_repl=None).verify()

    def test_missing_toolchain_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            (root / "lakefile.lean").write_text("import Lake\n")
            with self.assertRaisesRegex(RuntimeError, "lean-toolchain"):
                PantographEnv(source_root=root).verify()


class ReplTests(unittest.TestCase):
    def test_missing_binary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = PantographEnv(pantograph_repl=Path(tmp) / "repl")
            with self.assertRaisesRegex(RuntimeError, "not a file"):
                env.verify()

    def test_non_executable_binary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "repl"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o644)
            if os.access(binary, os.X_OK):  # running as root ignores the mode bits
                self.skipTest("process can execute mode-0644 files")
            with self.assertRaisesRegex(RuntimeError, "not executable"):
                PantographEnv(pantograph_repl=binary).verify()


class ToolchainAgreementTests(unittest.TestCase):
    """A REPL cannot read .olean files stamped by a different Lean version."""

    def test_matching_toolchains_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = _lake_project(tmp / "mathlib", "leanprover/lean4:v4.29.1")
            binary = _repl(tmp / "pantograph", "leanprover/lean4:v4.29.1")
            PantographEnv(source_root=source, pantograph_repl=binary).verify()

    def test_mismatched_toolchains_rejected_naming_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = _lake_project(tmp / "mathlib", "leanprover/lean4:v4.10.0-rc1")
            binary = _repl(tmp / "pantograph", "leanprover/lean4:v4.29.1")
            env = PantographEnv(source_root=source, pantograph_repl=binary)
            with self.assertRaises(RuntimeError) as caught:
                env.verify()
            message = str(caught.exception)
            self.assertIn("v4.10.0-rc1", message)
            self.assertIn("v4.29.1", message)
            self.assertIn("--source-root", message)

    def test_blank_toolchain_skips_comparison(self):
        """One side declaring nothing is not evidence of disagreement."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = _lake_project(tmp / "mathlib", "leanprover/lean4:v4.29.1")
            binary = _repl(tmp / "pantograph", "")
            PantographEnv(source_root=source, pantograph_repl=binary).verify()

    def test_no_source_root_skips_comparison(self):
        """With no compiled artifacts to read, the REPL's version cannot conflict."""
        with tempfile.TemporaryDirectory() as tmp:
            binary = _repl(Path(tmp) / "pantograph", "leanprover/lean4:v4.10.0-rc1")
            PantographEnv(pantograph_repl=binary).verify()


class ReplToolchainPathTests(unittest.TestCase):
    def test_checkout_layout_resolves_to_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pantograph"
            binary = _repl(root)
            env = PantographEnv(pantograph_repl=binary)
            self.assertEqual(env._repl_toolchain_path(), root / "lean-toolchain")

    def test_shallow_path_falls_back_to_sibling(self):
        """A REPL not in a .lake/build/bin tree has no project root to walk to."""
        env = PantographEnv(pantograph_repl=Path("/repl"))
        self.assertEqual(env._repl_toolchain_path(), Path("/lean-toolchain"))


if __name__ == "__main__":
    unittest.main()
