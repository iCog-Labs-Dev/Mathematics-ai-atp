import Lean

/- Header-only extraction. No benchmark declaration or proof is elaborated. -/
open Lean Elab Meta

def field (j : Json) (key : String) : IO Json := IO.ofExcept (j.getObjVal? key)
def stringField (j : Json) (key : String) : IO String :=
  IO.ofExcept ((j.getObjVal? key).bind Json.getStr?)
def emit (j : Json) : IO Unit := do (← IO.getStdout).putStrLn j.compress

private def role (info : BinderInfo) : String :=
  match info with
  | .default => ":explicit"
  | .implicit => ":implicit"
  | .strictImplicit => ":strict-implicit"
  | .instImplicit => ":instance-implicit"

private def closedType (sig : Syntax) (levels : List Name) : Command.CommandElabM (Expr × Array Json) :=
  Command.runTermElabM fun sectionVars => do
    unless sectionVars.isEmpty do throwError "section variables are unsupported"
    -- An elaboration error must reject the statement, never become a `sorry`.
    Term.withoutErrToSorry <| Term.withLevelNames levels <| Term.elabBinders sig[0].getArgs fun xs => do
      let target ← Term.elabType sig[1][1]
      Term.synthesizeSyntheticMVarsNoPostponing
      unless ← isProp target do throwError "theorem target is not a proposition"
      let mut locals := #[]
      for x in xs do
        let d ← x.fvarId!.getDecl
        locals := locals.push <| Json.mkObj [("name", toJson d.userName.eraseMacroScopes.toString),
          ("binder_role", toJson (role d.binderInfo))]
      let e ← mkForallFVars xs (← instantiateMVars target)
      let e ← Term.levelMVarToParam e
      let e ← instantiateMVars e
      if e.hasMVar || e.hasFVar then throwError "unresolved theorem parameters"
      if e.hasSorry then throwError "theorem statement contains sorry"
      return (e, locals)

private def checkType (original : Expr) (text : String) : Command.CommandElabM Unit :=
  Command.runTermElabM fun _ => do
    let levels := (collectLevelParams {} original).params.toList
    Term.withoutErrToSorry <| Term.withLevelNames levels do
      let env ← getEnv
      let stx ← ofExcept <| Parser.runParserCategory env `term text
      let e ← Term.elabType stx
      Term.synthesizeSyntheticMVarsNoPostponing
      let e ← instantiateMVars e
      if e.hasMVar || e.hasFVar then throwError "round-trip has unresolved variables"
      if e.hasSorry then throwError "round-trip contains sorry"
      unless ← isDefEq original e do throwError "round-trip changes the theorem type"

private def extract (cmd : Syntax) (check : Option String) : Command.CommandElabM Json := do
  let thm := cmd[1]
  let name := thm[1][0].getId
  if (← getEnv).contains name then throwError "benchmark declaration already exists in reference imports"
  unless cmd[0].getArgs.all (fun s => s.isNone || s.getArgs.isEmpty) do
    throwError "declaration modifiers are unsupported"
  let levels := if thm[1][1].isNone then [] else
    thm[1][1][1].getSepArgs.toList.map Syntax.getId
  let (original, locals) ← closedType thm[2] levels
  let text ← Command.runTermElabM fun _ => do
    withOptions (fun o => o.setBool `pp.all true |>.setBool `pp.universes true
      |>.setBool `pp.fullNames true |>.setBool `pp.notation false
      -- Raw literals print as `nat_lit`, so the text re-elaborates to the same term
      -- and equivalence never has to unfold large numerals.
      |>.setBool `pp.natLit true) do
      return (← ppExpr original).pretty
  let saved ← Command.getScope
  Command.modifyScope fun s => { s with openDecls := [], currNamespace := .anonymous }
  try
    checkType original text
    if let some text := check then checkType original text
  finally
    Command.modifyScope fun _ => saved
  let start := cmd.getPos?.getD 0
  let stop := cmd.getTailPos?.getD start
  let proof := thm[3].getPos?.getD stop
  return Json.mkObj [("name", toJson name.toString), ("expression", toJson text),
    ("levels", toJson ((collectLevelParams {} original).params.map Name.toString)),
    ("locals", Json.arr locals), ("start", toJson start.byteIdx),
    ("end", toJson stop.byteIdx), ("proof_start", toJson proof.byteIdx),
    ("checked", toJson check.isSome)]

private def runCommand (input : Parser.InputContext) (state : Command.State)
    (action : Command.CommandElabM α) : IO (α × Command.State) := do
  let ctx : Command.Context := {
    fileName := input.fileName
    fileMap := input.fileMap
    cmdPos := 0
    tacticCache? := none
    snap? := none
    cancelTk? := none }
  match ← EIO.toIO' ((action ctx).run state) with
  | .ok result => return result
  | .error e => throw <| IO.userError (← e.toMessageData.toString)

def main (args : List String) : IO Unit := do
  let [requestPath] := args | throw <| IO.userError "expected request JSON path"
  initSearchPath (← findSysroot)
  let req ← IO.ofExcept <| Json.parse (← IO.FS.readFile requestPath)
  let source ← stringField req "source"
  let wrapper ← stringField req "wrapper"
  let wrapperInput := Parser.mkInputContext wrapper "miniF2F-import-wrapper"
  let (wrapperHeader, wrapperParser, wrapperMessages) ← Parser.parseHeader wrapperInput
  if wrapperMessages.hasErrors then throw <| IO.userError "invalid import wrapper"
  let imports := headerToImports wrapperHeader
  for imp in imports do
    unless imp.module == `Init || (`Mathlib).isPrefixOf imp.module do
      throw <| IO.userError "wrapper imports must be Mathlib modules"
  let env ← importModules imports {}
  let (wrapperEnd, _, wrapperErrors) := Parser.parseCommand wrapperInput { env := env, options := {} } wrapperParser {}
  unless !wrapperErrors.hasErrors && wrapperEnd.isOfKind ``Parser.Command.eoi do
    throw <| IO.userError "wrapper must contain only imports"
  let input := Parser.mkInputContext source "miniF2F-source"
  let (header, initialParser, messages) ← Parser.parseHeader input
  if messages.hasErrors then throw <| IO.userError "invalid module header"
  unless (headerToImports header).all (fun i => i.module == `Init || i.module == `MiniF2F.Minif2fImport) do
    throw <| IO.userError "unexpected source import"
  emit <| Json.mkObj [("status", toJson "header"),
    ("imports", toJson (imports.map fun i => i.module.toString))]
  let mut ps := initialParser
  let mut state := Command.mkState env {} (({} : Options).setBool `autoImplicit false)
  let checks := (req.getObjVal? "checks").toOption.getD (Json.mkObj [])
  repeat
    let scope := state.scopes.head!
    let (cmd, next, msgs) := Parser.parseCommand input
      { env := state.env, options := scope.opts, currNamespace := scope.currNamespace,
        openDecls := scope.openDecls } ps {}
    if msgs.hasErrors then
      for msg in msgs.toList do
        if msg.severity == .error then (← IO.getStderr).putStrLn (← msg.data.toString)
      throw <| IO.userError s!"source parse failed at byte {ps.pos.byteIdx}"
    ps := next
    if cmd.isOfKind ``Parser.Command.eoi then break
    if cmd.isOfKind ``Parser.Command.open then
      let (_, newState) ← runCommand input state (Command.elabCommand cmd)
      if newState.messages.hasErrors then throw <| IO.userError "unsupported open context"
      state := newState
      let start := cmd.getPos?.getD 0
      let stop := cmd.getTailPos?.getD start
      emit <| Json.mkObj [("status", toJson "context"),
        ("source", toJson (source.extract start stop))]
    else if cmd.isOfKind ``Parser.Command.declaration && cmd[1].isOfKind ``Parser.Command.theorem then
      let name := cmd[1][1][0].getId.toString
      try
        let (row, after) ← runCommand input state (extract cmd ((checks.getObjVal? name).bind Json.getStr?).toOption)
        for msg in after.messages.toList do
          if msg.severity == .error then throw <| IO.userError (← msg.data.toString)
        emit <| Json.mkObj [("status", toJson "ok"), ("problem", row)]
      catch e =>
        emit <| Json.mkObj [("status", toJson "incompatible"), ("name", toJson name),
          ("start", toJson (cmd.getPos?.getD 0).byteIdx),
          ("end", toJson (cmd.getTailPos?.getD 0).byteIdx),
          ("proof_start", toJson (cmd[1][3].getPos?.getD 0).byteIdx), ("error", toJson e.toString)]
    else
      throw <| IO.userError s!"unsupported context command: {cmd.getKind}"
