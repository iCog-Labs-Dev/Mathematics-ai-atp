import Init

set_option linter.unusedVariables false

namespace BinderSmoke

theorem introTwo (P Q : Prop) : P → Q → P := by
  intro hp hq
  exact hp

theorem introWildcard (P Q : Prop) : Q → P → P := by
  intro _ hp
  exact hp

theorem introOne (P : Prop) : P → P := by
  intro
  assumption

theorem introsAll (P Q : Prop) : P → Q → P := by
  intros
  assumption

theorem introsNames (P Q : Prop) : P → Q → P := by
  intros hp hq
  exact hp

theorem escaped (P : Prop) : P → P := by
  intro «a.b»
  exact «a.b»

theorem shadowing (P : Prop) (h : P) : P → P := by
  intro h
  exact h

theorem repeatName (P Q : Prop) : P → Q → Q := by
  intro h h
  exact h

theorem globalCollision (P : Prop) : P → P := by
  intro Nat
  exact Nat

theorem unicode (P : Prop) : P → P := by
  intro h₁
  exact h₁

theorem namedRewrite (a : Nat) : a + 0 = a := by
  rw [Nat.add_zero (n := a)]

theorem eqLift (a b : Nat) (h : a = b) : a = b := h

theorem namedSimp (a b : Nat) (h : a = b) : a = b := by
  simp only [eqLift (b := b) a h]

theorem namedProof (a b : Nat) (h : a = b) : a = b := by
  rw [eqLift (a := a) (b := b) (h := h)]

theorem namedNested (a : Nat) : Nat.succ a + 0 = Nat.succ a := by
  rw [Nat.add_zero (n := Nat.succ a)]

theorem exactApplied (P Q : Prop) (f : P → Q) (h : P) : Q := by
  exact (f h)

theorem applyApplied (P Q : Prop) (f : P → Q) (h : P) : Q := by
  apply (f h)

theorem refineApplied (P Q : Prop) (f : P → Q) (h : P) : Q := by
  refine (f h)

theorem applyPartial (P Q R : Prop) (fpartial : P → Q → R) (hp : P) (hq : Q) : R := by
  apply (fpartial hp)
  exact hq

theorem exactNamed (a b : Nat) (h : a = b) : a = b := by
  exact eqLift (a := a) (b := b) (h := h)

theorem nestedTerm (P : Prop) (f g : P → P) (h : P) : P := by
  exact f (g h)

theorem refineHole (P : Prop) (h : P) : P := by
  refine id ?_
  exact h

theorem fourRules (a b c d e : Nat) (h1 : a = b) (h2 : b = c) (h3 : c = d) (h4 : d = e) : a = e := by
  rw [h1, h2, h3, h4]

theorem sixteenRules (a b : Nat) (h : a = b) (hx : a = a) : True := by
  rw [h, ← h, h, ← h, h, ← h, h, ← h, h, ← h, h, ← h, h, ← h, h, ← h] at hx
  trivial

end BinderSmoke
