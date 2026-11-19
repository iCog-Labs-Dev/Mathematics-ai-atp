import Pantograph

/- Native theorem-context snapshots; frozen GNN implementation at 1315dca. -/
namespace GnnNativeRuntime
open Lean Elab Tactic Meta

def options : Pantograph.Protocol.Options := {
  printExprPretty := true
  printExprAST := true
  printExprModelAST := true
  noRepeat := false
  automaticMode := false
}

def send (response : Json) : TacticM Unit := do
  let out ← IO.getStdout
  out.putStrLn ("GNN_STATE " ++ response.compress)
  out.flush

def describe (state : Nat) : TacticM Json := do
  let goals ← (← getUnsolvedGoals).toArray.mapM fun goal => do
    Pantograph.serializeGoal options goal (← goal.getDecl)
  return Json.mkObj [("status", toJson "ok"), ("state", toJson state), ("goals", toJson goals)]

def parseBlock (text : String) : TacticM Syntax := do
  let wrapped := "focus\n" ++ String.intercalate "\n" (text.splitOn "\n" |>.map ("  " ++ ·))
  match Parser.runParserCategory (← getEnv) `tactic wrapped with
  | .ok stx => pure stx
  | .error error => throwError error

partial def serve (states : Array Tactic.SavedState) : TacticM Unit := do
  let line ← (← IO.getStdin).getLine
  if line.isEmpty then throwError "GNN client disconnected"
  let before ← Tactic.saveState
  let (next, finished, response) ← MonadRecDepth.withRecDepth 0 <| Core.withCurrHeartbeats do
    tryCatchRuntimeEx (do
      let request ← match Json.parse line with
        | .ok value => pure value
        | .error error => throwError error
      let state ← match request.getObjValAs? Nat "state" with
        | .ok value => pure value
        | .error error => throwError error
      let some snapshot := states[state]? | throwError "Unknown native snapshot"
      snapshot.restore (restoreInfo := true)
      let command ← match request.getObjValAs? String "command" with
        | .ok value => pure value
        | .error error => throwError error
      if command == "finish" then
        unless (← getUnsolvedGoals).isEmpty do throwError "Unsolved sibling goals remain"
        return (states, true, ← describe state)
      if command != "step" then throwError "Unknown GNN native command"
      let text ← match request.getObjValAs? String "tactic" with
        | .ok value => pure value
        | .error error => throwError error
      Term.withoutErrToSorry <| withoutRecover <| evalTactic (← parseBlock text)
      Term.synthesizeSyntheticMVarsNoPostponing
      let errors ← (← Core.getMessageLog).toList.filter (·.severity == .error) |>.mapM (·.toString)
      unless errors.isEmpty do throwError (String.intercalate "\n" errors)
      discard <| getUnsolvedGoals
      let snapshot ← Tactic.saveState
      return (states.push snapshot, false, ← describe states.size)
    ) (fun error => do
      let message ← MonadRecDepth.withRecDepth 0 do
        if error.isMaxRecDepth then return "maximum recursion depth reached"
        tryCatchRuntimeEx error.toMessageData.toString (fun _ => pure "native resource limit")
      before.restore (restoreInfo := true)
      return (states, false, Json.mkObj [("status", toJson "lean_error"), ("error", toJson message)]))
  send response
  unless finished do serve next

elab "gnn_native_search" : tactic => do
  if (← Core.getMessageLog).hasErrors then
    throwError "GNN_INPUT_ELABORATION_FAILED"
  withMainContext do
    if (← getMainTarget).hasSorry then
      throwError "GNN_INPUT_CONTAINS_SORRY"
    for localDecl in (← getLCtx) do
      if localDecl.type.hasSorry then
        throwError "GNN_LOCAL_TYPE_CONTAINS_SORRY"
      if localDecl.value?.any (·.hasSorry) then
        throwError "GNN_LOCAL_VALUE_CONTAINS_SORRY"
  let snapshot ← Tactic.saveState
  send (← describe 0)
  serve #[snapshot]

end GnnNativeRuntime
