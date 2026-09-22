"""Verification of the Lean environment value object (no Lean, no subprocess).

``PantographEnv.verify`` is the whole point of these tests: it is the guard that
turns a misconfigured run into a one-second named error instead of a multi-minute
model load followed by ``Unknown identifier ℕ`` on every goal. Every check it
makes is a filesystem read, so all of it is testable without a REPL.
"""

from __future__ import annotations

import asyncio
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

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

    def test_custom_repl_requires_stdbuf_during_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = _repl(Path(tmp) / "pantograph")
            env = PantographEnv(pantograph_repl=binary)
            with patch(
                "maths_ai.hybrid_reasoner.pantograph_env.shutil.which",
                return_value=None,
            ):
                with self.assertRaisesRegex(RuntimeError, "GNU stdbuf") as caught:
                    env.verify()
            self.assertIn(str(binary), str(caught.exception))


class ToolchainAgreementTests(unittest.TestCase):
    """A REPL cannot read .olean files stamped by a different Lean version."""

    @patch(
        "maths_ai.hybrid_reasoner.pantograph_env.shutil.which",
        return_value="/usr/bin/stdbuf",
    )
    def test_matching_toolchains_accepted(self, _which):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = _lake_project(tmp / "mathlib", "leanprover/lean4:v4.29.1")
            binary = _repl(tmp / "pantograph", "leanprover/lean4:v4.29.1")
            PantographEnv(source_root=source, pantograph_repl=binary).verify()

    @patch(
        "maths_ai.hybrid_reasoner.pantograph_env.shutil.which",
        return_value="/usr/bin/stdbuf",
    )
    def test_mismatched_toolchains_rejected_naming_both(self, _which):
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

    @patch(
        "maths_ai.hybrid_reasoner.pantograph_env.shutil.which",
        return_value="/usr/bin/stdbuf",
    )
    def test_blank_toolchain_skips_comparison(self, _which):
        """One side declaring nothing is not evidence of disagreement."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = _lake_project(tmp / "mathlib", "leanprover/lean4:v4.29.1")
            binary = _repl(tmp / "pantograph", "")
            PantographEnv(source_root=source, pantograph_repl=binary).verify()

    @patch(
        "maths_ai.hybrid_reasoner.pantograph_env.shutil.which",
        return_value="/usr/bin/stdbuf",
    )
    def test_no_source_root_skips_comparison(self, _which):
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


class _DeferredServer:
    def __init__(
        self,
        *,
        proc_path: str = "/bundled/pantograph-repl",
        args: list[str] | None = None,
        lean_path: bytes | None = b"/resolved/lean/path",
        startup_error: BaseException | None = None,
    ) -> None:
        self.proc_path = proc_path
        self.args = list(args or ["Init"])
        self.lean_path = lean_path
        self.startup_error = startup_error
        self.restart_calls = 0
        self.closed = False

    async def restart_async(self) -> None:
        self.restart_calls += 1
        if self.startup_error is not None:
            raise self.startup_error

    def _close(self) -> None:
        self.closed = True


class ServerConstructionTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_repl_uses_line_buffered_command_and_preserves_server_args(self):
        server = _DeferredServer(args=["Init", "Mathlib", "--pp.universes"])
        create = AsyncMock(return_value=server)
        env = PantographEnv(
            source_root=Path("/external/mathlib4"),
            pantograph_repl=Path("/external/Pantograph/.lake/build/bin/repl"),
            imports=("Init", "Mathlib"),
            options={"printExprAST": True},
            timeout=37,
        )
        with (
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.Server.create",
                new=create,
            ),
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.shutil.which",
                return_value="/usr/bin/stdbuf",
            ),
        ):
            returned = await env.create_server()

        self.assertIs(returned, server)
        self.assertEqual(server.proc_path, "/usr/bin/stdbuf")
        self.assertEqual(
            server.args,
            [
                "-oL",
                "/external/Pantograph/.lake/build/bin/repl",
                "Init",
                "Mathlib",
                "--pp.universes",
            ],
        )
        self.assertEqual(server.restart_calls, 1)
        create.assert_awaited_once_with(
            imports=["Init", "Mathlib"],
            project_path="/external/mathlib4",
            options={"printExprAST": True},
            timeout=37,
            start=False,
        )

        command_after_start = list(server.args)
        await server.restart_async()
        self.assertEqual(server.args, command_after_start)

    async def test_bundled_repl_remains_direct_and_does_not_resolve_stdbuf(self):
        server = _DeferredServer(args=["Init"], lean_path=None)
        create = AsyncMock(return_value=server)
        with (
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.Server.create",
                new=create,
            ),
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.shutil.which"
            ) as which,
        ):
            returned = await PantographEnv().create_server()

        self.assertIs(returned, server)
        self.assertEqual(server.proc_path, "/bundled/pantograph-repl")
        self.assertEqual(server.args, ["Init"])
        self.assertEqual(server.restart_calls, 1)
        which.assert_not_called()

    async def test_create_server_rejects_missing_stdbuf_without_starting(self):
        server = _DeferredServer()
        create = AsyncMock(return_value=server)
        env = PantographEnv(pantograph_repl=Path("/external/repl"))
        with (
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.Server.create",
                new=create,
            ),
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.shutil.which",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "GNU stdbuf"):
                await env.create_server()

        create.assert_not_awaited()
        self.assertEqual(server.restart_calls, 0)

    async def test_startup_failure_closes_partial_server_and_names_command(self):
        cause = RuntimeError("ready timeout")
        server = _DeferredServer(args=["Init", "Mathlib"], startup_error=cause)
        create = AsyncMock(return_value=server)
        env = PantographEnv(
            source_root=Path("/external/mathlib4"),
            pantograph_repl=Path("/external/repl"),
            imports=("Init", "Mathlib"),
        )
        with (
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.Server.create",
                new=create,
            ),
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.shutil.which",
                return_value="/usr/bin/stdbuf",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Pantograph startup failed") as caught:
                await env.create_server()

        self.assertTrue(server.closed)
        self.assertIs(caught.exception.__cause__, cause)
        message = str(caught.exception)
        self.assertIn("/usr/bin/stdbuf -oL /external/repl Init Mathlib", message)
        self.assertIn("source_root=/external/mathlib4", message)

    async def test_startup_cancellation_closes_partial_server_without_wrapping(self):
        server = _DeferredServer(startup_error=asyncio.CancelledError())
        create = AsyncMock(return_value=server)
        env = PantographEnv(pantograph_repl=Path("/external/repl"))
        with (
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.Server.create",
                new=create,
            ),
            patch(
                "maths_ai.hybrid_reasoner.pantograph_env.shutil.which",
                return_value="/usr/bin/stdbuf",
            ),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await env.create_server()

        self.assertTrue(server.closed)

    def test_describe_preserves_custom_repl_and_names_launcher(self):
        text = PantographEnv(pantograph_repl=Path("/external/repl")).describe()
        self.assertIn("repl=/external/repl", text)
        self.assertIn("launcher=stdbuf -oL", text)


if __name__ == "__main__":
    unittest.main()
