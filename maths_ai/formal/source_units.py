"""Lean parser-only source boundaries; caller proofs are never elaborated."""

import asyncio
import json
import os
from pathlib import Path
import tempfile

from .admission import AdmissionRejected


async def source_units(body, *, declaration, imports, source_root, timeout_s, lean=None, env=None):
    script = "\n".join(f"import {name}" for name in dict.fromkeys(("Lean", *imports))) + "\n"
    script += "def formalSource : String := " + json.dumps(body, ensure_ascii=False) + "\n"
    script += "def formalTargetName : String := " + json.dumps(declaration, ensure_ascii=False) + "\n"
    script += '''
open Lean in
partial def formalDeclId (stx : Syntax) : Option Name :=
  if stx.getKind == `Lean.Parser.Command.declId then some stx[0].getId
  else stx.getArgs.findSome? formalDeclId

open Lean in
run_elab do
  let ictx := Parser.mkInputContext formalSource "<formal-source>"
  let initial : Elab.Frontend.State := {
    commandState := Elab.Command.mkState (← getEnv) {} {}
    parserState := {}
    cmdPos := 0
  }
  let work : Elab.Frontend.FrontendM (Array Json) := do
    let mut units : Array Json := #[]
    repeat
      Elab.Frontend.updateCmdPos
      let cs ← Elab.Frontend.getCommandState
      let ps ← Elab.Frontend.getParserState
      let scope := cs.scopes.head!
      let pctx : Parser.ParserModuleContext := {
        env := cs.env, options := scope.opts
        currNamespace := scope.currNamespace, openDecls := scope.openDecls
      }
      let before := ps.pos.byteIdx
      let (cmd, next, msgs) := Parser.parseCommand ictx pctx ps cs.messages
      if msgs.hasErrors then
        throw (IO.userError "Source cannot be parsed under the deployment syntax")
      Elab.Frontend.setParserState next
      Elab.Frontend.setMessages msgs
      units := units.push (toJson #[before, next.pos.byteIdx])
      if let some name := formalDeclId cmd then
        let full := if name.toString.startsWith "_root_." then name.toString.drop 7
          else if scope.currNamespace.isAnonymous then name.toString
          else s!"{scope.currNamespace}.{name}"
        if full == formalTargetName then break
      if Parser.isTerminalCommand cmd then break
      if next.pos.byteIdx ≤ before then throw (IO.userError "Source parser made no progress")
      -- Activate parsing context, never elaborate a declaration/proof or an
      -- executable command. This preserves namespaces and local/scoped notation.
      if #[`Lean.Parser.Command.namespace, `Lean.Parser.Command.section,
           `Lean.Parser.Command.end, `Lean.Parser.Command.open,
           `Lean.Parser.Command.universe, `Lean.Parser.Command.variable,
           `Lean.Parser.Command.include, `Lean.Parser.Command.omit,
           `Lean.Parser.Command.notation, `Lean.Parser.Command.infix,
           `Lean.Parser.Command.prefix, `Lean.Parser.Command.postfix].contains cmd.getKind then
        Elab.Frontend.elabCommandAtFrontend cmd
        if (← Elab.Frontend.getCommandState).messages.hasErrors then
          throw (IO.userError "Unsupported source parsing context")
    return units
  let (units, _) ← (work.run {inputCtx := ictx}).run initial
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
