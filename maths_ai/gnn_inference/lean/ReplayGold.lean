import Pantograph
import Pantograph.Frontend

/- Replay individual source tactics from native pre-invocation snapshots.
   This is a step-level check, not a whole-theorem prover. -/
open Lean Pantograph

structure ReplayRequest where
  rowIndex : Nat
  sourceStart : Nat
  sourceEnd : Nat
  unitIndex : Option Nat := none
  invocationIndex : Option Nat := none
  renderedTactic : Option String := none
  deriving FromJson

def replayOptions : Protocol.Options := {
  printExprPretty := true
  printExprModelAST := true
  automaticMode := true
}

def replaySyntaxBranch (snapshot : GoalState) (stx : Syntax)
    (parentDecl : Option Name := none) : MetaM Json := do
  try
    let action : Elab.Tactic.TacticM GoalState := do
      snapshot.savedState.restore
      -- Restore all active goals, never focus/drop sibling obligations.
      Elab.Tactic.setGoals snapshot.goals
      Elab.Tactic.evalTactic stx
      let savedState ← MonadBacktrack.saveState
      pure { snapshot with savedState }
    let next ← (action { elaborator := .anonymous }).run' { goals := snapshot.goals }
      |>.run' (ctx := { Condensed.elabContext with declName? := parentDecl })
    let goals ← next.serializeGoals (options := replayOptions)
    return Json.mkObj [("status", toJson "executed"), ("goals", toJson goals)]
  catch exception =>
    return Json.mkObj [("status", toJson "lean_error"),
      ("message", toJson (← exception.toMessageData.toString))]

def replayBranch (snapshot : GoalState) (text : String)
    (parentDecl : Option Name := none) (category : Name := `tactic) : MetaM Json := do
  try
    snapshot.restoreMetaM
    match Parser.runParserCategory snapshot.env category text with
    | .ok stx => replaySyntaxBranch snapshot stx parentDecl
    | .error message => pure <| Json.mkObj [
        ("status", toJson "parse_error"), ("message", toJson message)]
  catch exception =>
    return Json.mkObj [("status", toJson "lean_error"),
      ("message", toJson (← exception.toMessageData.toString))]

def replayInvocation (invocation : Frontend.TacticInvocation)
    (request : ReplayRequest) : IO Json := do
  invocation.ctx.runMetaM {} do
    Meta.withMCtx invocation.info.mctxBefore do
      let root := invocation.info.goalsBefore.headD { name := .anonymous }
      let snapshot ← GoalState.createFromMVars invocation.info.goalsBefore root
      let before ← snapshot.serializeGoals (options := replayOptions)
      let category : Name := if (before[0]?).any (·.isConversion) then `conv else `tactic
      let after ← Meta.withMCtx invocation.info.mctxAfter do
        invocation.info.goalsAfter.toArray.mapM fun goal => do
          serializeGoal replayOptions goal (← goal.getDecl)
      let source := invocation.info.stx.reprint.getD (toString invocation.info.stx) |>.trim
      -- Gold control keeps original layout/hygiene in the captured syntax.
      -- Text reparsing is a separate diagnostic, never substituted for control.
      let original ← replaySyntaxBranch snapshot invocation.info.stx invocation.ctx.parentDecl?
      let originalText ← replayBranch snapshot source invocation.ctx.parentDecl? category
      let rendered ← match request.renderedTactic with
        | some text => replayBranch snapshot text invocation.ctx.parentDecl? category
        | none => pure <| Json.mkObj [("status", toJson "not_rendered")]
      return Json.mkObj [
        ("row_index", toJson request.rowIndex),
        ("original_control", toJson "captured_syntax"),
        ("source_tactic", toJson source),
        ("source_before", toJson before), ("source_after", toJson after),
        ("original", original), ("original_text", originalText), ("rendered", rendered)]

unsafe def main (args : List String) : IO UInt32 := do
  let [sourcePath, requestsPath, outputPath] := args
    | throw <| IO.userError "usage: ReplayGold.lean SOURCE REQUESTS_JSON OUTPUT_JSON"
  let requests : Array ReplayRequest ← match Json.parse (← IO.FS.readFile requestsPath)
      >>= (fromJson? : Json → Except String (Array ReplayRequest)) with
    | .ok requests => pure requests
    | .error message => throw <| IO.userError message
  Pantograph.initSearch ""
  let source ← IO.FS.readFile sourcePath
  let (context, state) ← Frontend.createContextStateFromFile source sourcePath
  let unitCounter ← IO.mkRef 0
  let captured ← IO.mkRef (#[] : Array (Nat × Json))
  let messages ← IO.mkRef (#[] : Array String)
  let errors ← IO.mkRef (#[] : Array String)
  let process := Frontend.mapCompilationSteps fun step => do
    let unitIndex ← unitCounter.get
    unitCounter.modify (· + 1)
    let stepMessages ← step.messageStrings
    messages.modify (· ++ stepMessages)
    let stepErrors ← step.msgs.filter (·.severity == .error) |>.toArray.mapM (·.toString)
    errors.modify (· ++ stepErrors)
    let invocations := step.trees.bind Frontend.collectTactics |>.filter fun invocation =>
      (Elab.Info.ofTacticInfo invocation.info).isOriginal
    for (invocationIndex, invocation) in invocations.enum do
      let some start := invocation.info.stx.getPos? | continue
      let some stop := invocation.info.stx.getTailPos? | continue
      for request in requests do
        if request.sourceStart != start.byteIdx || request.sourceEnd != stop.byteIdx then
          continue
        if request.unitIndex.isSome && request.unitIndex != some unitIndex then continue
        if request.invocationIndex.isSome && request.invocationIndex != some invocationIndex then continue
        -- Range-only requests are useful for smoke tests. Exclude wrappers.
        if request.invocationIndex.isNone && !invocation.info.isSubstantive then continue
        let result ← try
          replayInvocation invocation request
        catch exception =>
          pure <| Json.mkObj [("row_index", toJson request.rowIndex),
            ("status", toJson "capture_error"), ("message", toJson exception.toString)]
        captured.modify (·.push (request.rowIndex, result))
  discard <| process.run context |>.run' state
  let results ← requests.mapM fun request => do
    let found := (← captured.get).filter (·.1 == request.rowIndex)
    if found.size == 1 then pure found[0]!.2
    else pure <| Json.mkObj [("row_index", toJson request.rowIndex),
      ("status", toJson (if found.isEmpty then "invocation_missing" else "invocation_ambiguous"))]
  let result := Json.mkObj [("rows", toJson results), ("frontend_messages", toJson (← messages.get)),
    ("frontend_errors", toJson (← errors.get))]
  IO.FS.writeFile outputPath result.compress
  return (0 : UInt32)
