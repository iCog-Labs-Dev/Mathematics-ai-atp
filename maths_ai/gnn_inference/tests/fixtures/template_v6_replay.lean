import Lean

set_option linter.unusedVariables false
theorem True.named {m : Nat} (h : True) (_ : m + 1 ≤ 1) : True := h
theorem True.typed {α : Type} (h : True) (_ : α) : True := h
namespace ScopeSmoke

macro "reverseProof " h:term : term => `(Eq.symm $h)

theorem simple (P : Prop) : P → P := by
  exact fun hp => hp
theorem typed (P : Prop) : P → P := by
  exact fun (hp : P) => hp
theorem typedBare (P : Prop) : P → P := by
  exact fun hp : P => hp
theorem dependent {α : Type} : ∀ x : α, x = x := by
  exact fun x => Eq.refl x
def implicitIdentity : ∀ {α : Type}, α → α := by
  exact fun {α} (x : α) => x
def strictIdentity : ∀ ⦃α : Type⦄, α → α := by
  exact fun ⦃α⦄ (x : α) => x
def implicitTyped : ∀ {α : Type}, α → α := by
  exact fun {α : Type} (x : α) => x
theorem multiBinders (P : Prop) : P → P → P := by
  exact fun (hp hq : P) => hp
theorem shadow (P : Prop) (hp : P) : P → P := by
  exact fun hp => hp
theorem nested (P Q : Prop) : P → Q → P := by
  exact fun hp => fun hq => hp
theorem anonymous (P Q : Prop) (h : Q) : P → Q := by
  exact fun _ => h
theorem holes (P : Prop) : P → P := by
  refine fun hp => ?_
  exact hp
theorem quantifier (P : Nat → Prop) (h : ∀ n, P n) : True := by
  have hx : ∀ n : Nat, P n := h
  trivial
theorem quantifierGrouped (h : ∀ (α : Type) (x : α), x = x) : True := by
  have hx : ∀ (α : Type) (x : α), x = x := h
  trivial
theorem quantifierImplicit (h : ∀ {P : Prop}, P → P) : True := by
  have hx : ∀ {P : Prop}, P → P := h
  trivial
theorem arrow (P : Prop) (h : P) : True := by
  have hx : P → P := id
  trivial
theorem explicitHead {α : Type} (x : α) : x = x := by
  exact @Eq.refl α x
theorem casePair (P Q : Prop) (h : P ∧ Q) : P := by
  rcases h with ⟨hp, hq⟩
  exact hp
theorem caseOr (P Q : Prop) (h : P ∨ Q) : Q ∨ P := by
  rcases h with hp | hq
  · exact Or.inr hp
  · exact Or.inl hq
theorem caseIgnore (P Q : Prop) (h : P ∧ Q) : Q := by
  rcases h with ⟨_, hq⟩
  exact hq
theorem caseClear (P Q : Prop) (h : P ∧ Q) : Q := by
  rcases h with ⟨-, hq⟩
  exact hq
theorem caseRfl (a b : Nat) (h : a = b) : b = a := by
  rcases h with rfl
  rfl
theorem caseTyped (P Q : Prop) (h : P ∧ Q) : P := by
  rcases h with (⟨hp, hq⟩ : P ∧ Q)
  exact hp
theorem caseEquation (P Q : Prop) (h : P ∧ Q) : P := by
  rcases eqn : h with ⟨hp, hq⟩
  exact hp
theorem caseTwo (P Q : Prop) (hp : P) (hq : Q) : P := by
  rcases hp, hq with ⟨hp, hq⟩
  exact hp
theorem introPattern (P Q : Prop) : P ∧ Q → P := by
  rintro ⟨hp, hq⟩
  exact hp
theorem introGrouped (P : Prop) : P → P → P := by
  rintro (hp hq : P)
  exact hp
theorem introOr (P Q : Prop) : P ∨ Q → Q ∨ P := by
  rintro (hp | hq)
  · exact Or.inr hp
  · exact Or.inl hq
theorem obtainPair (P Q : Prop) (h : P ∧ Q) : P := by
  obtain ⟨hp, hq⟩ := h
  exact hp
theorem obtainTyped (P Q : Prop) (h : P ∧ Q) : P := by
  obtain ⟨hp, hq⟩ : P ∧ Q := h
  exact hp
theorem obtainGoal (P Q : Prop) (h : P ∧ Q) : P := by
  obtain ⟨hp, hq⟩ : P ∧ Q
  · exact h
  exact hp
theorem byCase (P : Prop) [Decidable P] : P ∨ ¬P := by
  by_cases hp : P
  · exact Or.inl hp
  · exact Or.inr hp
theorem byCaseDefault (P : Prop) [Decidable P] : True := by
  by_cases P
  · trivial
  · trivial
theorem extending {α : Type} (f : α → α) : f = f := by
  ext x
  rfl
theorem receiverSymmetry (a b : Nat) (h : a = b) : b = a := by
  exact h.symm
theorem boundReceiver (a b : Nat) : a = b → b = a := by
  exact fun h => h.symm
theorem receiverTransitivity (a b c : Nat) (h : a = b) (h2 : b = c) : a = c := by
  exact h.trans h2
theorem receiverRefine (a b : Nat) (h : a = b) : b = a := by
  refine h.symm
theorem receiverRewrite (a b : Nat) (h : a = b) : b = a := by
  rw [h.symm]
theorem semanticMacro (a b : Nat) (h : a = b) : b = a := by
  exact reverseProof h
class ShiftLaw where
  shift : Nat → Nat
  shift_zero : shift 0 = 0
  shift_one : shift 1 = 1
theorem twoInstances (inst₁ inst₂ : ShiftLaw) : inst₁.shift 0 + inst₂.shift 0 = 0 := by
  rewrite [inst₁.shift_zero, inst₂.shift_zero]
  rfl
theorem twoInstancesOne (inst₁ inst₂ : ShiftLaw) : inst₁.shift 1 + inst₂.shift 1 = 2 := by
  rewrite [inst₁.shift_one, inst₂.shift_one]
  rfl
theorem namedReceiver (h : True) : True := by
  have hx := h.named (m := 0) (Nat.le_refl 1)
  exact hx
theorem typedReceiver (h : True) : True := by
  have hx := h.typed (1 : Nat)
  exact hx
theorem specializeReceiver (n : Nat) (hn : ∀ m, n ≤ m → m = m) : True := by
  specialize hn (n + 1) (n.le_add_right 1)
  trivial
end ScopeSmoke
