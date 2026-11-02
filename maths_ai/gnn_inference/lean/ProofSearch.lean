import Lean

/- Native snapshots for generated proofs. No printed-goal reconstruction. -/
namespace AtpNativeSearch
open Lean Elab Tactic Meta

def send (response : Json) : TacticM Unit := do
  let out ← IO.getStdout
  out.putStrLn ("ATP_SEARCH " ++ response.compress)
  out.flush

def describe (stateId : Nat) : TacticM Json := do
  let goals ← (← getUnsolvedGoals).toArray.mapM fun goal => do
    return (← Meta.ppGoal goal).pretty
  return Json.mkObj [("status", toJson "ok"), ("state", toJson stateId),
    ("goals", toJson goals)]

def lookup (query : String) : TacticM Json := do
  let terms := query.toLower.splitOn " " |>.filter (!·.isEmpty)
  if terms.isEmpty || query.length > 128 || terms.length > 8 then
    throwError "Lookup needs one to eight name fragments (maximum 128 characters)"
  let entries := (← getEnv).constants.toList.filter fun (name, _) =>
    let text := name.toString
    -- Avoid line-leading !: open Nat enables Mathlib's postfix factorial.
    Bool.not (text.startsWith "_") && Bool.not (text.startsWith "AtpNativeSearch") &&
    Bool.not (text.startsWith "Expected_") && Bool.not (text.startsWith "Spec_") &&
    Bool.not (text.startsWith "SearchRoot_") &&
    terms.all (fun term => (text.toLower.splitOn term).length > 1)
  let sorted := entries.toArray.qsort fun a b =>
    let x := a.1.toString
    let y := b.1.toString
    x.length < y.length || (x.length == y.length && x < y)
  let hits ← (sorted.extract 0 (min 8 sorted.size)).mapM fun (name, info) => do
    return Json.mkObj [("name", toJson name.toString),
      ("type", toJson (← Meta.ppExpr info.type).pretty)]
  return Json.mkObj [("status", toJson "ok"), ("query", toJson query), ("lemmas", toJson hits)]

def firstTactic (seq : Syntax) : Option Syntax := do
  let body := if seq.getKind == ``Parser.Tactic.tacticSeq then seq[0] else seq
  let elements ← if body.getKind == ``Parser.Tactic.tacticSeqBracketed then some body[1]
    else if body.getKind == ``Parser.Tactic.tacticSeq1Indented then some body[0] else none
  elements.getArgs.find? (·.getKind != nullKind)

partial def firstNestedTactic (stx : Syntax) : Option Syntax := do
  if stx.getKind == ``Parser.Term.byTactic then firstTactic stx[1]
  else stx.getArgs.findSome? firstNestedTactic

partial def serve (states : Array Tactic.SavedState) : TacticM Unit := do
  let input ← IO.getStdin
  let line ← input.getLine
  if line.isEmpty then throwError "Native proof-search client disconnected"
  let (nextStates, finished, response) ← Core.withCurrHeartbeats do
    try
      let request ← match Json.parse line with
        | .ok value => pure value
        | .error error => throwError error
      let stateId ← match request.getObjValAs? Nat "state" with
        | .ok value => pure value
        | .error error => throwError error
      let some snapshot := states[stateId]? | throwError "Unknown native snapshot"
      snapshot.restore (restoreInfo := true)
      let command ← match request.getObjValAs? String "command" with
        | .ok value => pure value
        | .error error => throwError error
      if command == "lookup" then
        let query ← match request.getObjValAs? String "query" with
          | .ok value => pure value
          | .error error => throwError error
        return (states, false, ← lookup query)
      else if command == "finish" then
        unless (← getUnsolvedGoals).isEmpty do throwError "Unsolved sibling goals remain"
        return (states, true, ← describe stateId)
      else if command == "step" || command == "split" then
        let text ← match request.getObjValAs? String "tactic" with
          | .ok value => pure value
          | .error error => throwError error
        -- Each block focuses the first goal; Lean's focus restores its siblings.
        let wrapped := "focus\n" ++ String.intercalate "\n" (text.splitOn "\n" |>.map ("  " ++ ·))
        let stx ← match Parser.runParserCategory (← getEnv) `tactic wrapped with
          | .ok stx => pure stx
          | .error error => throwError error
        if command == "split" then
          let some first := firstTactic stx[1] | throwError "No top-level tactic in proposal"
          let some text := first.reprint | throwError "Cannot recover original tactic text"
          let inner := (firstNestedTactic first).bind Syntax.reprint |>.map String.trim
          return (states, false, Json.mkObj [("status", toJson "ok"), ("first_tactic", toJson text.trim),
            ("inner_tactic", toJson inner)])
        Term.withoutErrToSorry <| withoutRecover <| evalTactic stx
        Term.synthesizeSyntheticMVarsNoPostponing
        let errors ← (← Core.getMessageLog).toList.filter (·.severity == .error) |>.mapM (·.toString)
        unless errors.isEmpty do throwError (String.intercalate "\n" errors)
        discard <| getUnsolvedGoals
        let next ← Tactic.saveState
        return (states.push next, false, ← describe states.size)
      else
        throwError "Unknown native proof-search command"
    catch exception =>
      return (states, false, Json.mkObj [("status", toJson "lean_error"),
        ("error", toJson (← exception.toMessageData.toString))])
  send response
  unless finished do serve nextStates

elab "atp_native_search" : tactic => do
  let snapshot ← Tactic.saveState
  send (← describe 0)
  serve #[snapshot]

end AtpNativeSearch
