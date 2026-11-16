"""Lean parser-only source boundaries; caller proofs are never elaborated."""

import asyncio
import json
import os
from pathlib import Path
import tempfile

from .admission import AdmissionRejected


async def source_units(body, *, imports, source_root, timeout_s, lean=None, env=None):
    script = "\n".join(f"import {name}" for name in dict.fromkeys(("Lean", *imports))) + "\n"
    script += "def formalSource : String := " + json.dumps(body, ensure_ascii=False) + "\n"
    script += '''
open Lean in
run_elab do
  let ictx := Parser.mkInputContext formalSource "<formal-source>"
  let pctx : Parser.ParserModuleContext := { env := (← getEnv), options := {} }
  let mut ps : Parser.ModuleParserState := {}
  let mut messages : MessageLog := {}
  let mut units : Array Json := #[]
  repeat
    let before := ps.pos.byteIdx
    let (cmd, next, msgs) := Parser.parseCommand ictx pctx ps messages
    if msgs.hasErrors then
      throwError "Source cannot be parsed under the deployment syntax"
    ps := next
    messages := msgs
    units := units.push (toJson #[before, ps.pos.byteIdx])
    if Parser.isTerminalCommand cmd then break
    if ps.pos.byteIdx ≤ before then throwError "Source parser made no progress"
  IO.println ("FORMAL_UNITS " ++ (toJson units).compress)
'''
    with tempfile.TemporaryDirectory(prefix="formal_source_units_") as directory:
        path = Path(directory) / "Source.lean"
        path.write_text(script, encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            *([lean] if lean is not None else ["lake", "env", "lean"]), str(path), cwd=source_root, env=env, start_new_session=True,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(), timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:
                    pass
                await process.communicate()
            raise
    rows = [line[len("FORMAL_UNITS "):] for line in out.decode().splitlines() if line.startswith("FORMAL_UNITS ")]
    if process.returncode != 0 or len(rows) != 1:
        raise AdmissionRejected("input:unsupported-source-syntax", (out + err).decode(errors="replace")[-2000:])
    return [tuple(unit) for unit in json.loads(rows[0])]
