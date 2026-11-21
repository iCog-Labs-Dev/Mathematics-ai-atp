import Mathlib

/-!
# Formal protocol v2: reference proofs

These declarations are independently checkable mathematical examples. The live
model demo submits only their statements and contexts, never these proof bodies.

Both trained backends independently discovered `ring` for `lagrange_identity`.
The Cauchy–Schwarz and Chinese remainder proofs below are reference proofs; their
presence does not assert that either model discovered them within its budget.
-/

set_option autoImplicit false

namespace FormalV2Examples

/-- Lagrange's identity: the three-dimensional Gram determinant is a sum of squares. -/
theorem lagrange_identity (a b c x y z : ℝ) :
    (a ^ 2 + b ^ 2 + c ^ 2) * (x ^ 2 + y ^ 2 + z ^ 2) -
        (a * x + b * y + c * z) ^ 2 =
      (a * y - b * x) ^ 2 + (a * z - c * x) ^ 2 + (b * z - c * y) ^ 2 := by
  ring

/-- Cauchy–Schwarz for three real coordinates, derived from Lagrange's identity. -/
theorem cauchy_schwarz_three (a b c x y z : ℝ) :
    (a * x + b * y + c * z) ^ 2 ≤
      (a ^ 2 + b ^ 2 + c ^ 2) * (x ^ 2 + y ^ 2 + z ^ 2) := by
  have nonnegative : 0 ≤
      (a ^ 2 + b ^ 2 + c ^ 2) * (x ^ 2 + y ^ 2 + z ^ 2) -
        (a * x + b * y + c * z) ^ 2 := by
    rw [lagrange_identity]
    positivity
  exact sub_nonneg.mp nonnegative

set_option maxHeartbeats 2000000 in
/-- A bounded Chinese remainder problem; the unique solution below 105 is 23. -/
theorem chinese_remainder_bounded (n : ℕ)
    (h3 : n % 3 = 2) (h5 : n % 5 = 3) (h7 : n % 7 = 2)
    (hbound : n < 105) : n = 23 := by
  omega

end FormalV2Examples

#print axioms FormalV2Examples.lagrange_identity
#print axioms FormalV2Examples.cauchy_schwarz_three
#print axioms FormalV2Examples.chinese_remainder_bounded
