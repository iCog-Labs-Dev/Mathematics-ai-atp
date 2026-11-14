/-
Mathematics AI ATP - Lean Project

This is a minimal Lean project that includes Mathlib as a dependency.
It exists to provide a stable Lean environment for Pantograph to resolve
real proof goals.

The project is built with:
  - lake update
  - lake exe cache get  (downloads precompiled .olean files)
  - lake build

This avoids the ~1-2 hour build time of compiling Mathlib from source.
-/
import Mathlib
