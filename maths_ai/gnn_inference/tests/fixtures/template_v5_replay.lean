/- Native test prepends the exact pinned simp_rw implementation. -/
set_option linter.unusedVariables false

namespace WrapperSmoke

attribute [symm] Eq.symm

def keep (n : Nat) := n

theorem simpaUsing (a b : Nat) (h : a + 0 = b) : a = b := by
  simpa only [Nat.add_zero] using h

theorem simpaPlain (P : Prop) (h : P) : P := by
  simpa using (id h)

theorem allOnly (P : Prop) (h : P) : P := by
  simp_all only [h]

theorem definitionAt (a : Nat) (ha : keep a = a) : True := by
  dsimp only [keep] at ha
  trivial

theorem rewriteAt (a b : Nat) (h : a = b) (ha : a = a) (hb : a = a) : True := by
  rewrite [h] at ha hb
  trivial

theorem rwaGoal (P Q : Prop) (h : P = Q) (hp : P) : Q := by
  rwa [← h]

theorem bothLocation (a b : Nat) (h : a = b) (ha : a = a) : a = b := by
  simp only [h] at ha ⊢

theorem wildcard (a b : Nat) (h : a = b) : a = b := by
  simp only [h] at *

theorem erased (a : Nat) : a + 0 = a := by
  simp only [Nat.add_zero, -Nat.add_zero, Nat.add_zero]

theorem reverseSimp (a b : Nat) (h : a = b) : b = a := by
  simp only [← h]

theorem postSimp (a : Nat) : a + 0 = a := by
  simp only [↑Nat.add_zero]

theorem allHyp (P : Prop) (h : P) : P := by
  simp only [*]

theorem unfoldAt (a : Nat) (h : keep a = a) : True := by
  unfold keep at h
  trivial

theorem deltaAt (a : Nat) (h : keep a = a) : True := by
  delta keep at h
  trivial

theorem changeAt (a : Nat) (h : keep a = a) : True := by
  change a = a at h
  trivial

theorem specializeArg (P : Prop) (f : P → P) (hp : P) : P := by
  specialize f hp
  exact f

theorem simpleHave (P : Prop) (hp : P) : P := by
  have hx := id hp
  exact hx

theorem anonymousHave (P : Prop) (hp : P) : P := by
  have := id hp
  exact this

theorem typedHave (P Q : Prop) (hp : P) (hq : Q) : P := by
  have hx : P ∧ Q := ⟨hp, hq⟩
  exact hx.1

theorem proofHave (a : Nat) : a = a := by
  have hx : a + 0 = a := by rfl
  exact hx

theorem assignedLet (a : Nat) : a = a := by
  let b := id a
  rfl

theorem typedLet (a : Nat) : a = a := by
  let b : Nat := id a
  rfl

theorem projectIndex (P Q : Prop) (h : P ∧ Q) : P := by
  exact (id h).1

theorem projectBare (P Q : Prop) (h : P ∧ Q) : P := by
  exact h.1

theorem rewriteUnderBinder (P Q : Prop) (h : P = Q) : (P → P) = (Q → Q) := by
  simp_rw [h]

theorem showType (a : Nat) : a = a := by
  show a = a
  rfl

theorem symmAt (a b : Nat) (h : a = b) : b = a := by
  symm at h
  exact h

end WrapperSmoke
