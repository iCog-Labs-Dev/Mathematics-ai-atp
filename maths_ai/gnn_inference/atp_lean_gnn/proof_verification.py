"""Independent compilation of a fixed target header and reconstructed GNN path."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time


ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
_NAME = r"(?:«[^»\r\n]+»|[^\W\d][\w'′]*)(?:\.(?:«[^»\r\n]+»|[^\W\d][\w'′]*))*"


def checked_header(header: str) -> tuple[str, str]:
    """Normalize a trusted example header; never accept model-generated headers."""
    header = header.strip()
    if re.match(r"example\b", header):
        header = re.sub(r"^example\b", "theorem GnnSearchChecked", header, count=1)
    match = re.match(rf"(?:theorem|lemma)\s+({_NAME})(?=\s|[:({{\[])", header)
    if not match:
        raise ValueError("a named theorem/lemma or example header is required")
    return header, match.group(1)


def proof_source(header: str, steps: list[str], *, imports=("Init", "Mathlib")) -> tuple[str, str]:
    header, name = checked_header(header)
    if not steps or any(not isinstance(step, str) or not step.strip() for step in steps):
        raise ValueError("nonempty accepted tactic path required")
    # Native stepping focuses the first goal and then restores siblings. Each
    # source block must have exactly that behavior during independent replay.
    body = "\n".join("  focus\n" + "\n".join("    " + line for line in step.splitlines()) for step in steps)
    prelude = "\n".join(f"import {module}" for module in dict.fromkeys(("Lean", *imports)))
    return f"{prelude}\n\n{header} := by\n{body}\n\n#print axioms {name}\n", name


class LeanProofVerifier:
    """A fresh compiler process checks all goals and the theorem's actual axioms."""

    def __init__(self, *, source_root: Path, imports=("Init", "Mathlib")):
        self.source_root = Path(source_root).resolve()
        self.imports = tuple(imports)

    async def verify(self, header: str, steps: list[str], *, timeout: float) -> dict:
        return await asyncio.to_thread(self._verify, header, steps, timeout=timeout)

    def _verify(self, header: str, steps: list[str], *, timeout: float) -> dict:
        source, name = proof_source(header, steps, imports=self.imports)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="gnn_proof_check_") as directory:
            path = Path(directory) / "Candidate.lean"
            path.write_text(source, encoding="utf-8")
            process = subprocess.Popen(["lake", "env", "lean", str(path)], cwd=self.source_root,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                return {"status": "verification_timeout", "source": source,
                        "seconds": time.monotonic() - started}
        output = stdout + stderr
        audit = re.findall(r"^'" + re.escape(name) + r"' depends on axioms:\s*\[([^\]]*)\]", output, re.MULTILINE)
        empty = re.findall(r"^'" + re.escape(name) + r"' does not depend on any axioms$", output, re.MULTILINE)
        axioms = [value.strip() for value in audit[0].split(",") if value.strip()] if len(audit) == 1 else []
        passed = process.returncode == 0 and len(audit) + len(empty) == 1 and set(axioms) <= ALLOWED_AXIOMS
        return {"status": "passed" if passed else "lean_rejected", "exit_code": process.returncode,
                "axioms": axioms, "forbidden_axioms": sorted(set(axioms) - ALLOWED_AXIOMS),
                "seconds": time.monotonic() - started, "source": source, "diagnostics": output}
