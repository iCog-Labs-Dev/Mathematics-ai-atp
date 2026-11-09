import Lean

set_option linter.unusedVariables false

-- Test-only carriers. The norm notation declarations match Mathlib at
-- 29dcec074de168ac2bf835a77ef68bbe069194c5, Analysis/Normed/Group/Basic.lean.
def norm (n : Nat) : Nat := n
def nnnorm (n : Nat) : Nat := n
notation "‖" e "‖" => norm e
notation "‖" e "‖₊" => nnnorm e
def inv (n : Nat) : Nat := n
postfix:max "⁻¹" => inv
def smul (a b : Nat) : Nat := a * b
infixr:73 " • " => smul
def abs (n : Nat) : Nat := n
notation "|" e "|" => abs e
abbrev Bag := Nat → Prop
instance : Membership Nat Bag := ⟨fun n p => p n⟩
instance : HasSubset Bag := ⟨fun p q => ∀ n, p n → q n⟩
instance : HasSSubset Bag := ⟨fun p q => (∀ n, p n → q n) ∧ ¬ (∀ n, q n → p n)⟩
instance : Union Bag := ⟨fun p q n => p n ∨ q n⟩
instance : Inter Bag := ⟨fun p q n => p n ∧ q n⟩
instance : SDiff Bag := ⟨fun p q n => p n ∧ ¬ q n⟩

namespace MathSmoke
theorem bounded (n : Nat) (h : ∀ x ≤ n, x ≤ n) : True := by
  have hx : ∀ x ≤ n, x ≤ n := h
  trivial
theorem existsTyped (h : ∃ x : Nat, x = x) : True := by
  have hx : ∃ x : Nat, x = x := h
  trivial
theorem existsBounded (n : Nat) (h : ∃ x ≤ n, x ≤ n) : True := by
  have hx : ∃ x ≤ n, x ≤ n := h
  trivial
theorem normed (a : Nat) (h : ‖a‖ = a) : True := by
  have hx : ‖a‖ = a := h
  trivial
theorem nonnegativeNorm (a : Nat) (h : ‖a‖₊ = a) : True := by
  have hx : ‖a‖₊ = a := h
  trivial
theorem inverse (a : Nat) (h : a⁻¹ = a) : True := by
  have hx : a⁻¹ = a := h
  trivial
theorem absolute (a : Nat) (h : |a| = a) : True := by
  have hx : |a| = a := h
  trivial
theorem negative (P : Prop) (h : ¬ P) : True := by
  have hx : ¬ P := h
  trivial
theorem scalar (a b : Nat) (h : a • b = a * b) : True := by
  have hx : a • b = a * b := h
  trivial
theorem subset (s t : Bag) (h : s ⊆ t) : True := by
  have hx : s ⊆ t := h
  trivial
theorem boundedMember (s : Bag) (h : ∀ x ∈ s, x ∈ s) : True := by
  have hx : ∀ x ∈ s, x ∈ s := h
  trivial
theorem existsUntyped (P : Nat → Prop) (h : ∃ x, P x) : True := by
  have hx : ∃ x, P x := h
  trivial
theorem existsGrouped (h : ∃ (x : Nat) (y : Nat), x = y) : True := by
  have hx : ∃ (x : Nat) (y : Nat), x = y := h
  trivial
theorem related (s t : Bag) (a b : Nat) (f g : Nat → Nat) (p : Nat × Nat)
    (hgt : a > b) (hge : a ≥ b) (hdiv : a ∣ b) (hnm : a ∉ s)
    (hproper : s ⊂ t) (hsuper : s ⊇ t) (hstrictsuper : s ⊃ t) : True := by
  have hx_gt : a > b := hgt
  have hx_ge : a ≥ b := hge
  have hx_div : a ∣ b := hdiv
  have hx_nmem : a ∉ s := hnm
  have hx_proper : s ⊂ t := hproper
  have hx_super : s ⊇ t := hsuper
  have hx_strictsuper : s ⊃ t := hstrictsuper
  have hx_union : s ∪ t = s ∪ t := by rfl
  have hx_inter : s ∩ t = s ∩ t := by rfl
  have hx_diff : s \ t = s \ t := by rfl
  have hx_comp : (f ∘ g) a = f (g a) := by rfl
  have hx_prod : Nat × Nat := p
  trivial
theorem boundedPredicates (n : Nat) (s : Bag)
    (hlt : ∀ x < n, True) (hle : ∀ x ≤ n, True) (hgt : ∀ x > n, True)
    (hge : ∀ x ≥ n, True) (hne : ∀ x ≠ n, True) (hmem : ∀ x ∈ s, True)
    (hnmem : ∀ x ∉ s, True) (hsub : ∀ x ⊆ s, True) (hssub : ∀ x ⊂ s, True)
    (hsup : ∀ x ⊇ s, True) (hssup : ∀ x ⊃ s, True) (hanon : ∀ _ ≤ n, True) : True := by
  have hx_lt : ∀ x < n, True := hlt
  have hx_le : ∀ x ≤ n, True := hle
  have hx_gt : ∀ x > n, True := hgt
  have hx_ge : ∀ x ≥ n, True := hge
  have hx_ne : ∀ x ≠ n, True := hne
  have hx_mem : ∀ x ∈ s, True := hmem
  have hx_nmem : ∀ x ∉ s, True := hnmem
  have hx_sub : ∀ x ⊆ s, True := hsub
  have hx_ssub : ∀ x ⊂ s, True := hssub
  have hx_sup : ∀ x ⊇ s, True := hsup
  have hx_ssup : ∀ x ⊃ s, True := hssup
  have hx_anon : ∀ _ ≤ n, True := hanon
  trivial
theorem shadowBound (n : Nat) (h : ∀ n : Nat, n ≤ n + 1 → True) : True := by
  have hx_shadow : ∀ n ≤ n + 1, True := h
  trivial
theorem existsEnglish (P : Nat → Prop) (h : ∃ x, P x) : True := by
  have hx_english : exists x, P x := h
  trivial
theorem existsSeveral (h : ∃ x y : Nat, x = y) : True := by
  have hx_several : ∃ x y : Nat, x = y := h
  trivial
theorem existsDependent (h : ∃ (x : Nat) (y : Fin (x + 1)), y = y) : True := by
  have hx_dependent : ∃ (x : Nat) (y : Fin (x + 1)), y = y := h
  trivial
theorem composedMath (n : Nat) (h : ∀ x ≤ n, ¬ (‖x⁻¹‖ < n • 1)) : True := by
  have hx_composed : ∀ x ≤ n, ¬ (‖x⁻¹‖ < n • 1) := h
  trivial
theorem exactNegative (P : Prop) (hp : ¬ P) : ¬ P := by
  exact (hp : ¬ P)
theorem rewriteNorm (a b : Nat) (h : ‖a‖ = b) : ‖a‖ = b := by
  rw [(h : ‖a‖ = b)]
theorem refineExists (n : Nat) (h : n = n) : ∃ x : Nat, x = n := by
  refine (⟨n, h⟩ : ∃ x : Nat, x = n)
theorem patternNot (P : Prop) : ¬ P → ¬ P := by
  rintro (hp : ¬ P)
  exact hp
theorem casesNot (P : Prop) [Decidable P] : True := by
  by_cases hp : ¬ P
  · trivial
  · trivial
theorem obtainExists (h : ∃ x : Nat, x = x) : True := by
  obtain ⟨x, hx⟩ : ∃ x : Nat, x = x := h
  trivial
theorem scopedNorm : ∀ n : Nat, ‖n‖ = ‖n‖ := by
  exact fun n => Eq.refl ‖n‖
theorem usingNot (P : Prop) (hp : ¬ P) : ¬ P := by
  simpa using (hp : ¬ P)
end MathSmoke
