"""Which Lean environment a Pantograph server runs in.

The RL search, the interactive reasoner, and the smoke script all need to answer
one question before they can start a server: which compiled Lean artifacts should
the REPL be able to see, and which REPL binary should run. ``PantographEnv`` is
that answer as a single value, so the initial server and every post-crash restart
provably share one definition instead of two dicts that must be kept in sync.

Two seams inside PyPantograph do the work:

* ``project_path`` — ``Server.create`` runs ``lake env printenv LEAN_PATH`` with
  that directory as the working directory and injects the result as the REPL
  subprocess's ``LEAN_PATH``. Without it the REPL sees only the core ``Init``
  library compiled into the binary, so Mathlib notation (``ℕ``, ``⌊…⌋₊``) is
  neither declared nor lexable.
* ``Server.proc_path`` — the binary ``restart_async`` execs, initialised in
  ``Server.__init__`` to the REPL bundled with PyPantograph. There is no
  ``proc_path`` keyword on ``create``, so a caller-supplied REPL has to be
  assigned on the instance between construction and the spawn. That is why this
  is a value object with a ``create_server`` method rather than a dict of
  ``Server.create`` keyword arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pantograph.server import Server
from pantograph.utils import _get_proc_cwd

_LAKEFILES = ("lakefile.lean", "lakefile.toml")


def _read_toolchain(path: Path) -> str:
    """Return the stripped contents of a ``lean-toolchain`` file, or ``""``."""
    try:
        return path.read_text().strip()
    except OSError:
        return ""


@dataclass(frozen=True)
class PantographEnv:
    """The Lean environment one Pantograph server runs in.

    ``PantographEnv()`` with every field defaulted is behaviourally identical to
    a bare ``Server.create()``: the bundled REPL, ``imports=["Init"]``, and no
    ``project_path``. Callers that do not set the flags are unaffected.

    Args:
        source_root: Lake project root whose compiled ``.olean`` artifacts the
            REPL should see. ``None`` leaves the REPL on core Lean only.
        pantograph_repl: REPL binary to exec instead of the bundled one. Its
            toolchain must match ``source_root``'s.
        imports: Modules the server imports at startup.
        options: Pantograph options forwarded to ``Server.create``.
        timeout: Per-request timeout in seconds, forwarded to ``Server.create``.
    """

    source_root: Path | None = None
    pantograph_repl: Path | None = None
    imports: tuple[str, ...] = ("Init",)
    options: Mapping[str, Any] = field(default_factory=dict)
    timeout: int = 120

    # ------------------------------------------------------------------
    # Verification — cheap, synchronous, runs no Lean
    # ------------------------------------------------------------------

    def verify(self) -> None:
        """Check paths and toolchain agreement, raising ``RuntimeError`` on any
        mismatch.

        Runs before a checkpoint is loaded so a misconfigured run fails in under
        a second rather than after several minutes of model setup. Deliberately
        does not pin git commits: the extractor pins them because its dataset
        stores S-expressions that must match the source byte-for-byte, whereas
        the RL loop only needs tactics to elaborate.
        """
        if self.source_root is not None:
            root = Path(self.source_root)
            if not root.is_dir():
                raise RuntimeError(f"source_root is not a directory: {root}")
            if not any((root / name).is_file() for name in _LAKEFILES):
                raise RuntimeError(
                    f"source_root has no {' or '.join(_LAKEFILES)}: {root}. "
                    f"It must be a Lake project root."
                )
            if not (root / "lean-toolchain").is_file():
                raise RuntimeError(f"source_root has no lean-toolchain file: {root}")

        if self.pantograph_repl is not None:
            repl = Path(self.pantograph_repl)
            if not repl.is_file():
                raise RuntimeError(f"pantograph_repl is not a file: {repl}")
            if not os.access(repl, os.X_OK):
                raise RuntimeError(f"pantograph_repl is not executable: {repl}")

        self._verify_toolchains()

    def _verify_toolchains(self) -> None:
        """Compare ``source_root``'s toolchain against the REPL's.

        A REPL built by one Lean version cannot read ``.olean`` files compiled by
        another: the compiler stamps its version into every artifact. Mixing them
        surfaces several frames inside PyPantograph as ``KeyError('fragment')``
        during goal parsing, so catching it here converts an opaque mid-search
        failure into a named startup error.
        """
        if self.source_root is None:
            return

        source_toolchain = _read_toolchain(Path(self.source_root) / "lean-toolchain")
        repl_toolchain_path = self._repl_toolchain_path()
        repl_toolchain = _read_toolchain(repl_toolchain_path)

        if not source_toolchain or not repl_toolchain:
            # One side does not declare a toolchain; there is nothing to compare
            # and refusing to start would reject working setups.
            return

        if source_toolchain != repl_toolchain:
            raise RuntimeError(
                f"Lean toolchain mismatch. "
                f"{Path(self.source_root) / 'lean-toolchain'} declares "
                f"'{source_toolchain}', but the REPL at {repl_toolchain_path} was built "
                f"with '{repl_toolchain}'. Compiled .olean artifacts are not portable "
                f"across Lean versions; point --source-root at a project built with "
                f"'{repl_toolchain}', or pass a --pantograph-repl built with "
                f"'{source_toolchain}'."
            )

    def _repl_toolchain_path(self) -> Path:
        """Locate the ``lean-toolchain`` describing the REPL binary.

        A REPL built from a Pantograph checkout sits at
        ``<pantograph_root>/.lake/build/bin/repl``, so its project root is three
        parents up. The bundled REPL ships its toolchain file beside it.
        """
        if self.pantograph_repl is not None:
            repl = Path(self.pantograph_repl)
            if len(repl.parents) >= 4:
                return repl.parents[3] / "lean-toolchain"
            return repl.parent / "lean-toolchain"
        return Path(_get_proc_cwd()) / "lean-toolchain"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    async def create_server(self) -> Server:
        """Start a Pantograph server in this environment.

        ``start=False`` defers the subprocess spawn so ``proc_path`` can be
        overridden first. ``Server.create`` resolves ``lean_path`` before it
        consults ``start``, so the deferred spawn still receives a fully resolved
        ``LEAN_PATH``.
        """
        options = {"printExprModelAST": True, **dict(self.options)}
        server = await Server.create(
            imports=list(self.imports),
            project_path=str(self.source_root) if self.source_root else None,
            options=options,
            timeout=self.timeout,
            start=False,
        )
        if self.pantograph_repl is not None:
            server.proc_path = str(self.pantograph_repl)

        # `get_lean_path_async` runs `lake env printenv LEAN_PATH` through
        # `utils.check_output`, which returns None on a non-zero exit rather than
        # raising. An unbuilt project therefore yields lean_path=None silently and
        # the REPL starts with no Mathlib on its search path, which surfaces much
        # later as `Unknown identifier ℕ` on every goal.
        if self.source_root is not None and not server.lean_path:
            raise RuntimeError(
                f"`lake env printenv LEAN_PATH` produced nothing in {self.source_root}. "
                f"Run `cd {self.source_root} && lake exe cache get && lake build` first."
            )

        await server.restart_async()
        return server

    def describe(self) -> str:
        """One-line summary for run logs."""
        root = self.source_root or "<core Lean only>"
        repl = self.pantograph_repl or "<bundled>"
        return f"source_root={root} repl={repl} imports={list(self.imports)}"
