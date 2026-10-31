import Init

set_option linter.unusedVariables false

namespace AppliedRuleSmoke

def witnessProof : True := True.intro

theorem qualifiedLookup : True := by
  exact witnessProof

theorem eqLift (a b : Nat) (h : a = b) : a = b := h
theorem eqFour (a b : Nat) (h : a = b) (q : True) : a = b := h

theorem rwApplied (a : Nat) : a + 0 = a := by
  rw [Nat.add_zero a]

theorem rwConditional (P : Prop) [Decidable P] (h : P) (a b : Nat) :
    (if P then a else b) = a := by
  rw [if_pos h]

theorem simpApplied (P : Prop) [Decidable P] (h : P) (a b : Nat) :
    (if P then a else b) = a := by
  simp only [if_pos h]

theorem localApplied (a : Nat) (f : ∀ x : Nat, x + 0 = x) : a + 0 = a := by
  rw [f a]

theorem threeArguments (a b : Nat) (h : a = b) : a = b := by
  simp only [eqLift a b h]

theorem globalArgument : Nat.zero + 0 = Nat.zero := by
  rw [Nat.add_zero Nat.zero]

theorem parentheses (a : Nat) : a + 0 = a := by
  rw [(Nat.add_zero (a))]

theorem maxReferences (a b : Nat) (h : a = b) (hx : a = a) : True := by
  rw [eqLift a b h, ← eqLift a b h, eqLift a b h] at hx
  trivial

theorem nestedArgument (a : Nat) : Nat.succ a + 0 = Nat.succ a := by
  rw [Nat.add_zero (Nat.succ a)]

theorem namedArgument (a : Nat) : a + 0 = a := by
  rw [Nat.add_zero (n := a)]

theorem literalArgument : 2 + 0 = 2 := by
  rw [Nat.add_zero 2]

theorem holeArgument (a : Nat) : a + 0 = a := by
  rw [Nat.add_zero _]

theorem argumentBudget (a b : Nat) (h : a = b) : a = b := by
  simp only [eqFour a b h True.intro]

end AppliedRuleSmoke
