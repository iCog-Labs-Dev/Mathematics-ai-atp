"""Independent original-target compilation and axiom audit from GNN 1315dca."""

import asyncio
import os
from pathlib import Path
import re
import signal
import tempfile
import time

from .proof_tasks import checked_header, split_prefix

ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})


def proof_source(header, steps, *, imports=("Init", "Mathlib"), prefix="", heartbeats=1000000, rec_depth=2048):
    prefix_imports, prefix = split_prefix(prefix)
    header, name = checked_header(header, prefix=prefix)
    if not steps or any(not isinstance(step, str) or not step.strip() for step in steps):
        raise ValueError("nonempty accepted tactic path required")
    body = "\n".join("  focus\n" + "\n".join("    " + line for line in step.splitlines()) for step in steps)
    prelude = "\n".join(f"import {module}" for module in dict.fromkeys(("Lean", *imports, *prefix_imports)))
    guards = (f"\nset_option maxHeartbeats {heartbeats}\nset_option maxRecDepth {rec_depth}\n"
        f"run_elab do\n  if (← _root_.Lean.MonadEnv.getEnv).contains `{name} then\n"
        '    throwError "GNN_TARGET_ALREADY_AVAILABLE"\n')
    return f"{prelude}\n{prefix}{guards}{header} := by\n{body}\n\n#print axioms _root_.{name}\n", name


class LeanProofVerifier:
    def __init__(self, *, source_root, imports=("Init", "Mathlib"), prefix="", heartbeats=1000000, rec_depth=2048):
        self.source_root, self.imports = Path(source_root).resolve(), tuple(imports)
        self.prefix, self.heartbeats, self.rec_depth = prefix, heartbeats, rec_depth

    async def verify(self, header, steps, *, timeout):
        source, name = proof_source(header, steps, imports=self.imports, prefix=self.prefix,
                                   heartbeats=self.heartbeats, rec_depth=self.rec_depth)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="gnn_proof_check_") as directory:
            path = Path(directory) / "Candidate.lean"
            path.write_text(source, encoding="utf-8")
            process = await asyncio.create_subprocess_exec("lake", "env", "lean", str(path), cwd=self.source_root,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError) as error:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.communicate()
                if isinstance(error, asyncio.CancelledError):
                    raise
                return {"status": "verification_timeout", "source": source, "seconds": time.monotonic() - started}
        output = (stdout + stderr).decode("utf-8", errors="replace")
        audit = re.findall(r"^'" + re.escape(name) + r"' depends on axioms:\s*\[([^\]]*)\]", output, re.MULTILINE)
        empty = re.findall(r"^'" + re.escape(name) + r"' does not depend on any axioms$", output, re.MULTILINE)
        axioms = [value.strip() for value in audit[0].split(",") if value.strip()] if len(audit) == 1 else []
        passed = process.returncode == 0 and len(audit) + len(empty) == 1 and set(axioms) <= ALLOWED_AXIOMS
        return {"status": "passed" if passed else "lean_rejected", "exit_code": process.returncode,
            "axioms": axioms, "forbidden_axioms": sorted(set(axioms) - ALLOWED_AXIOMS),
            "seconds": time.monotonic() - started, "source": source, "diagnostics": output}
