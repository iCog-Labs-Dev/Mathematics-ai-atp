"""One validated PLN specification and process owner for all live workers."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .compiler import TRANSLATION_VERSION
from .dependencies import PLNConfigurationError, PLNDependencies, resolve_petta
from .model import PLNInference


@dataclass(frozen=True)
class PLNRuntimeSpec:
    petta_bin: str
    chainer_root: str
    petta_source_root: str
    swipl_bin: str = "swipl"
    depth: int = 10
    timeout_s: float = 30.0
    max_concurrency: int = 2
    max_pending: int = 64
    output_limit: int = 1048576
    temp_dir: str | None = None

    def __post_init__(self):
        for name in ("depth", "max_concurrency", "max_pending", "output_limit"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise PLNConfigurationError(f"{name} must be a positive integer")
        if (
            isinstance(self.timeout_s, bool)
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise PLNConfigurationError("PLN timeout must be finite and positive")

    @classmethod
    def from_environment(cls):
        names = ("PETTA_BIN", "PETTACHAINER_ROOT", "PETTA_SOURCE_ROOT")
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise PLNConfigurationError(f"Configure live PLN before startup: {missing}")
        return cls(
            *(os.environ[name] for name in names),
            swipl_bin=os.environ.get("SWIPL_BIN", "swipl"),
        )

    def scorer(self):
        dependencies = PLNDependencies.resolve(self.chainer_root)
        root = Path(self.petta_source_root).resolve()
        if not (root / "run.sh").is_file():
            raise PLNConfigurationError(f"PeTTa source missing: {root}")
        return PLNInference(
            petta_bin=resolve_petta(self.petta_bin),
            chainer_root=dependencies.chainer_root,
            depth=self.depth,
            timeout=self.timeout_s,
            max_concurrency=self.max_concurrency,
            pending_limit=self.max_pending,
            output_limit=self.output_limit,
            temp_dir=self.temp_dir,
        )

    async def open(self):
        scorer = self.scorer()
        identity = await asyncio.to_thread(self._identity)
        # The shared compiler and answer validator must accept a real derivation.
        result = await asyncio.to_thread(
            scorer.evaluate_text,
            "Q",
            ("hP : P", "hPQ : P → Q"),
            propositions={"P": "P", "Q": "Q"},
        )
        if result.status != "ok" or result.is_fallback or result.stv is None:
            await scorer.aclose()
            raise PLNConfigurationError(
                f"PLN rule-proof readiness failed: {result.status}: {result.raw_output[-1000:]}"
            )
        return scorer, identity

    def _identity(self):
        executable = shutil.which(self.swipl_bin)
        if not executable:
            raise PLNConfigurationError("Configured SWI-Prolog is unavailable")
        # Require the launcher and preflight to resolve the same SWI executable.
        if Path(executable).resolve() != Path(shutil.which("swipl") or "").resolve():
            raise PLNConfigurationError(
                "swipl on PATH differs from the configured PLN executable"
            )
        command = [
            executable,
            "-q",
            "-g",
            "use_module(library(janus)),py_call(json:loads('[]'),[]),py_call(sys:version_info:major,A),"
            "py_call(sys:version_info:minor,B),format('~d.~d',[A,B]),halt",
        ]
        result = subprocess.run(
            command, check=True, text=True, capture_output=True, timeout=15
        )
        expected = f"{sys.version_info.major}.{sys.version_info.minor}"
        if result.stdout.strip() != expected:
            raise PLNConfigurationError(
                f"Janus embeds {result.stdout!r}, expected Python {expected}"
            )
        version = subprocess.run(
            [executable, "--version"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        ).stdout.strip()
        native = (
            subprocess.run(
                [
                    executable,
                    "-q",
                    "-g",
                    "current_prolog_flag(executable,E),current_prolog_flag(home,H),format('~w~n~w~n',[E,H]),halt",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()
        )
        native_files = [
            Path(native[0]),
            *Path(native[1]).glob("lib/*/janus.so"),
            *Path(native[1]).glob("lib/*/libswipl.so*"),
        ]
        identity = {
            "settings": asdict(self),
            "translation": TRANSLATION_VERSION,
            "swipl": version,
            "python": expected,
            "native_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in native_files
                if path.is_file()
            },
        }
        for name, root in (
            ("petta", self.petta_source_root),
            ("chainer", self.chainer_root),
        ):
            root = Path(root).resolve()
            revision = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={root}",
                    "-C",
                    str(root),
                    "rev-parse",
                    "HEAD",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            # Bind executable source contents, including local modifications.
            digest = hashlib.sha256()
            for path in sorted(root.rglob("*")):
                if any(
                    part in (".git", "__pycache__", "repos", "build", "target")
                    for part in path.relative_to(root).parts
                ):
                    continue
                if path.is_file():
                    digest.update(str(path.relative_to(root)).encode())
                    digest.update(hashlib.sha256(path.read_bytes()).digest())
            identity[name] = {"commit": revision, "source_sha256": digest.hexdigest()}
        return identity
