"""Validated local dependencies shared by live and batch PLN execution."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

CHAINER_MODULE = Path("pettachainer/metta/petta_chainer.metta")
AXIOMS_MODULE = Path("pettachainer/metta/metamath/axioms/metamath_axioms.metta")
TESTED_CHAINER_COMMIT = "b1678e97c4498b376b91e8cce215f5ab0398ddef"
TESTED_PETTA_COMMIT = "e1490899cefc67c128d5311ff4861f9997674957"


class PLNConfigurationError(ValueError):
    """The requested PLN installation cannot produce executable imports."""


def module_import(file: Path) -> str:
    """Validate a module file and emit the extensionless PeTTa module path."""
    path = file.expanduser().resolve()
    if path.suffix != ".metta" or not path.is_file() or not os.access(path, os.R_OK):
        raise PLNConfigurationError(f"Expected a readable .metta file: {path}")
    # Bare module paths are the syntax verified against the supported PeTTa.
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path)):
        raise PLNConfigurationError(
            f"PeTTa module paths must contain only ASCII letters, digits, /, _, -, or .: {path}"
        )
    return f"!(import! &self {path.with_suffix('')})"


@dataclass(frozen=True)
class PLNDependencies:
    chainer_root: Path
    chainer_file: Path
    axioms_file: Path

    @classmethod
    def resolve(
        cls,
        chainer_root: Path | str | None = None,
        axioms_path: Path | str | None = None,
    ) -> PLNDependencies:
        configured = (
            chainer_root
            if chainer_root is not None
            else os.environ.get("PETTACHAINER_ROOT")
        )
        if not configured:
            raise PLNConfigurationError(
                "Set PETTACHAINER_ROOT to a local checkout of "
                "https://github.com/123nol/petta_chainer or pass chainer_root."
            )
        root = Path(configured).expanduser().resolve()
        chainer = root / CHAINER_MODULE
        axioms = (
            Path(axioms_path).expanduser().resolve()
            if axioms_path is not None
            else root / AXIOMS_MODULE
        )
        module_import(chainer)
        module_import(axioms)
        return cls(root, chainer, axioms)

    def imports(self, *, include_axioms: bool = True) -> str:
        lines = [module_import(self.chainer_file)]
        if include_axioms:
            lines.append(module_import(self.axioms_file))
        return "\n\n".join(lines)


def resolve_petta(petta_bin: str | None = None) -> str:
    configured = (
        petta_bin if petta_bin is not None else os.environ.get("PETTA_BIN", "petta")
    )
    executable = shutil.which(os.path.expanduser(configured))
    if executable is None:
        raise PLNConfigurationError(
            f"PeTTa executable not found or not executable: {configured!r}. "
            "Install PeTTa and set PETTA_BIN to its launcher."
        )
    return str(Path(executable).resolve())
