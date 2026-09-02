import Lake
open Lake DSL

package mathlib_atp where
  leanOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib MathlibAtp where

-- Pin to Mathlib commit compatible with Lean v4.29.1 (matches Pantograph)
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "5e932f97dd25535344f80f9dd8da3aab83df0fe6"
