/-
Copyright (c) 2020 Anne Baanen. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Anne Baanen, Mario Carneiro, Alex J. Best

Functional definitions copied from Mathlib/Tactic/SimpRw.lean at
29dcec074de168ac2bf835a77ef68bbe069194c5. Test-only: exercise the pinned
Mathlib tactic without downloading the complete Mathlib build locally.
-/
import Lean

namespace Mathlib.Tactic
open Lean Parser.Tactic Elab.Tactic

def withSimpRWRulesSeq (token : Syntax) (rwRulesSeqStx : Syntax)
    (x : (symm : Bool) → (term : Syntax) → TacticM Unit) : TacticM Unit := do
  let lbrak := rwRulesSeqStx[0]
  let rules := rwRulesSeqStx[1].getArgs
  withTacticInfoContext (mkNullNode #[token, lbrak]) (pure ())
  let numRules := (rules.size + 1) / 2
  for i in [:numRules] do
    let rule := rules[i * 2]!
    let sep := rules.getD (i * 2 + 1) Syntax.missing
    withTacticInfoContext (mkNullNode #[rule, sep]) do
      withRef rule do
        let symm := !rule[0].isNone
        let term := rule[1]
        x symm term

elab s:"simp_rw " cfg:(config)? rws:rwRuleSeq g:(location)? : tactic => focus do
  let cfg' : TSyntax `Lean.Parser.Tactic.config ← do
    match cfg with
    | Option.none =>
      `(config| (config := ({ failIfUnchanged := false } : Lean.Meta.Simp.Config)))
    | Option.some c => match c with
      | `(config| (config := $cfg)) =>
        `(config| (config := ({ ($cfg : Lean.Meta.Simp.Config) with failIfUnchanged := false })))
      | _ => throwError "malformed cfg"
  evalTactic (← `(tactic| simp%$s $cfg' only $g ?))
  withSimpRWRulesSeq s rws fun symm term => do
    evalTactic (← match term with
    | `(term| $e:term) =>
      if symm then
        `(tactic| simp%$e $[$cfg]? only [← $e:term] $g ?)
      else
        `(tactic| simp%$e $[$cfg]? only [$e:term] $g ?))

end Mathlib.Tactic
