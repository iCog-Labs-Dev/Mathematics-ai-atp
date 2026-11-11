import Lean

set_option linter.unusedVariables false

namespace CodecSmoke

theorem introduction (P : Prop) : P → P := by
  intro hα
  exact hα

theorem anonymousIntroduction (P : Prop) (hp : P) : P → P := by
  intro _
  exact hp

theorem pair (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q := by
  constructor
  · exact hp
  · exact hq

theorem casesPair (P Q : Prop) (h : P ∧ Q) : P := by
  rcases eqn : h with ⟨hp, hq⟩
  exact hp

theorem nested (P Q : Prop) : P → Q → P := by
  exact fun hp => fun hq => hp

theorem shadowed (P : Prop) (hp : P) : P → P := by
  exact fun hp => hp

def dependent : ∀ {α : Type} (x : α), x = x := by
  exact fun {α : Type} (x : α) => Eq.refl x

theorem quantify (h : ∀ (α : Type) (x : α), x = x) : True := by
  have hx : ∀ (α : Type) (x : α), x = x := h
  trivial

theorem bounded (n : Nat) (h : ∀ m ≤ n, m = m) : True := by
  have hx : ∀ m ≤ n, m = m := h
  trivial

theorem existential (n : Nat) : True := by
  have hx : ∃ m : Nat, m = n := ⟨n, by rfl⟩
  trivial

theorem hole (P : Prop) (hp : P) : P ∧ P := by
  refine ⟨hp, ?_⟩
  exact hp

theorem selector (a b : Nat) : a = b → b = a := by
  exact fun h => h.symm

theorem typedRewrite (a b : Nat) (h : a = b) : a = b := by
  rw [(h : a = b)]

theorem definition (n : Nat) : n = n := by
  let copy : Nat := n
  rfl

end CodecSmoke
