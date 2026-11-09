import Lean

set_option linter.unusedVariables false
macro "scope " body:tacticSeq : tactic => `(tactic| ($body))
theorem True.finish (receiver h : True) : True := h

namespace ProgramSmoke
private theorem secret {P : Prop} (h : P) : P := h
theorem known : True := True.intro
class Mark where
  value : Nat

theorem nested (P Q : Prop) (f : P → Q) : P → Q := by
  have helper : P → Q := by
    intro hp
    exact f hp
  exact helper

theorem parameterized (P Q : Prop) (f : P → Q) (hp : P) : Q := by
  have helper (p : P) : Q := f p
  exact helper hp

theorem instanceProof (P : Prop) (hp : P) : P := by
  haveI : Decidable P := .isTrue hp
  exact hp

theorem instanceLet (n : Nat) : True := by
  letI : Mark := ⟨n⟩
  trivial

theorem replaced (P : Prop) (hp : P) : P := by
  replace hp : P := hp
  exact hp

theorem enough (P : Prop) (hp : P) : P := by
  suffices h : P by
    exact h
  exact hp

theorem enoughFrom (P : Prop) (hp : P) : P := by
  suffices P from this
  exact hp

theorem letPattern (P Q : Prop) (h : P ∧ Q) : P := by
  let ⟨hp, hq⟩ := h
  exact hp

theorem patternLambda (P Q : Prop) : P ∧ Q → P := by
  exact fun ⟨hp, hq⟩ => hp

theorem recordValue (n : Nat) : Nonempty Mark := by
  exact ⟨{ value := n }⟩

theorem showProof (P : Prop) (hp : P) : P := by
  exact show P by
    exact hp

theorem calculated (a b c : Nat) (h1 : a = b) (h2 : b = c) : a = c := by
  calc
    a = b := h1
    _ = c := h2

theorem allGoals (P : Prop) (hp : P) : P ∧ P := by
  constructor <;> exact hp

theorem twoGoals (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q := by
  constructor
  · exact hp
  · exact hq

theorem scopedProof (P : Prop) (hp : P) : P := by
  scope
    have h : P := by
      exact hp
    exact h

theorem branching (P : Prop) (hp : P) : P := by
  first | exact missingName | exact hp

theorem attempted (P : Prop) (hp : P) : P := by
  try exact missingName
  exact hp

theorem privateUse (P : Prop) (hp : P) : P := by
  exact secret hp

theorem privateBare (hp : True) : True := by
  exact secret hp

theorem globalReceiver (hp : True) : True := by
  exact known.finish hp

theorem inductionProof (n : Nat) : n = n := by
  induction n with
  | zero => rfl
  | succ n ih => rfl

theorem casesProof (b : Bool) : b = b := by
  cases b <;> rfl

theorem configured (a : Nat) (h : a = 0) : a + 0 = 0 := by
  simp (config := { contextual := true }) only [h, Nat.zero_add]

theorem localTerm (n : Nat) : n = n := by
  exact (let x := n; Eq.refl x)

theorem typedTuple (a b : Nat) : (a, b) = (a, b) := by
  exact Eq.refl (a, b)

theorem converted (a b : Nat) (h : a = b) : a + 0 = b := by
  conv =>
    lhs
    rw [h, Nat.add_zero]

theorem matched (b : Bool) : b = b := by
  exact match b with
  | false => by rfl
  | true => by rfl

theorem multilineLiteral : "a
b" = "a
b" := by
  exact Eq.refl "a
b"

theorem doExpression (n : Nat) : (Id.run do
    let x := n
    pure x) = n := by
  exact (show (Id.run do let x := n; pure x) = n from Eq.refl n)
end ProgramSmoke
