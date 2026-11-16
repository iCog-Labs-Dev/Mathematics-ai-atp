import Lean
open Lean Elab

private def runCommand (input : Parser.InputContext) (state : Command.State)
    (action : Command.CommandElabM Unit) : IO Command.State := do
  let ctx : Command.Context := {
    fileName := input.fileName, fileMap := input.fileMap, cmdPos := 0,
    tacticCache? := none, snap? := none, cancelTk? := none }
  match ← EIO.toIO' ((action ctx).run state) with
  | .ok (_, state) => return state
  | .error e => throw <| IO.userError (← e.toMessageData.toString)

-- Parse the target without elaborating its value. Prefix commands establish
-- namespaces and syntax extensions. No command after the target is processed.
def main (args : List String) : IO Unit := do
  let [path] := args | throw <| IO.userError "expected request JSON"
  initSearchPath (← findSysroot)
  let req ← IO.ofExcept <| Json.parse (← IO.FS.readFile path)
  let body ← IO.ofExcept <| (req.getObjVal? "body").bind Json.getStr?
  let target ← IO.ofExcept <| (req.getObjVal? "declaration").bind Json.getStr?
  let imports ← IO.ofExcept <| (req.getObjVal? "imports").bind Json.getArr?
  let modules ← imports.mapM fun j => do
    let name ← IO.ofExcept j.getStr?
    pure ({ module := name.toName } : Import)
  let env ← importModules modules {}
  let input := Parser.mkInputContext body "formal-source"
  let (_, initial, messages) ← Parser.parseHeader input
  if messages.hasErrors then throw <| IO.userError "invalid header"
  let mut ps := initial
  let mut state := Command.mkState env {} {}
  repeat
    let scope := state.scopes.head!
    let (cmd, next, msgs) := Parser.parseCommand input
      { env := state.env, options := scope.opts, currNamespace := scope.currNamespace,
        openDecls := scope.openDecls } ps {}
    if msgs.hasErrors then throw <| IO.userError "source parse failed"
    ps := next
    if cmd.isOfKind ``Parser.Command.eoi then
      throw <| IO.userError "declaration not found"
    if cmd.isOfKind ``Parser.Command.declaration && cmd[1].isOfKind ``Parser.Command.theorem then
      let localName := cmd[1][1][0].getId
      let full := if (`_root_).isPrefixOf localName then localName.replacePrefix `_root_ Name.anonymous
                  else scope.currNamespace ++ localName
      if full.toString == target then
        let start := cmd.getPos?.getD 0
        let proof ← match cmd[1][3].getPos? with
          | some p => pure p
          | none => throw <| IO.userError "unsupported declaration body"
        unless cmd[1][3].isOfKind ``Parser.Command.declValSimple do
          throw <| IO.userError "unsupported declaration body"
        let rewritten := body.extract 0 proof ++ ":= by sorry\n"
        (← IO.getStdout).putStrLn <| (Json.mkObj [
          ("rewritten", toJson rewritten), ("start", toJson start.byteIdx)]).compress
        return
    state ← runCommand input state (Command.elabCommand cmd)
    if state.messages.hasErrors then throw <| IO.userError "required prefix does not elaborate"
