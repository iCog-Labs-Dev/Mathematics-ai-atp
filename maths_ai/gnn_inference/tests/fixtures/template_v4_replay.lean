import Init

set_option linter.unusedVariables false

namespace RecursiveSmoke

theorem nested (P : Prop) (f g : P → P) (h : P) : P := by
  exact f (g h)

theorem appliedHead (P Q R : Prop) (f : P → Q → R) (hp : P) (hq : Q) : R := by
  exact (f hp) hq

theorem pair (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q := by
  exact ⟨hp, hq⟩

theorem emptyConstructor : True := by
  exact ⟨⟩

theorem twoHoles (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q := by
  refine ⟨?_, ?_⟩
  · exact hp
  · exact hq

theorem triple (P Q R : Prop) (hp : P) (hq : Q) (hr : R) : P ∧ Q ∧ R := by
  refine ⟨hp, ?_, hr⟩
  exact hq

theorem implicitHole (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q := by
  refine' And.intro hp _
  exact hq

theorem explicitHole (P : Prop) (h : P) : P := by
  refine id ?_
  exact h

theorem lowApplication (P : Prop) (h : P) : P := by
  exact id <| id h

def numeral : Nat := by
  exact (2 : Nat)

def negative : Int := by
  exact (-2 : Int)

theorem nestedRule (a : Nat) : Nat.succ a + 0 = Nat.succ a := by
  rw [Nat.add_zero (Nat.succ a)]

theorem namedNestedRule (a : Nat) : Nat.succ a + 0 = Nat.succ a := by
  rw [Nat.add_zero (n := Nat.succ a)]

theorem numericRule : 2 + 0 = 2 := by
  rw [Nat.add_zero 2]

theorem inferredRule (a : Nat) : a + 0 = a := by
  rw [Nat.add_zero _]

theorem typedRule (a : Nat) : a + 0 = a := by
  rw [Nat.add_zero (a : Nat)]

theorem projected (P Q : Prop) (h : P ∧ Q) : P := by
  exact (id h).left

inductive Related : Nat → Nat → Prop where
  | refl (a : Nat) : Related a a
  | lift (k : Nat) {a b : Nat} (h : Related a b) : Related (k + a) (k + b)

theorem relationLift (a b k : Nat) (ih : Related a b) : Related (k + a) (k + b) := by
  exact (id ih).lift k

theorem witness : ∃ n : Nat, n = n := by
  refine ⟨2, ?_⟩
  rfl

theorem dottedIdentifier (P Q : Prop) (h : P ∧ Q) : P := by
  exact h.left

theorem lambdaScope (P : Prop) : P → P := by
  exact fun hp => hp

theorem nestedProof (P : Prop) (h : P) : P := by
  exact (by exact h)

end RecursiveSmoke
