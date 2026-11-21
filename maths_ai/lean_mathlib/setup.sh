#!/usr/bin/env bash
# Setup Lean project with Mathlib for Pantograph
set -e

PROJECT_DIR="maths_ai/lean_mathlib"

echo "=== Setting up Lean Mathlib project ==="

# Check if lake is available
if ! command -v lake &> /dev/null; then
    echo "ERROR: lake not found. Install Lean 4 first."
    echo "  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh"
    exit 1
fi

cd "$PROJECT_DIR"

# Initialize project if not already done
if [ ! -f "lakefile.lean" ]; then
    echo "Creating lakefile.lean..."
    cat > lakefile.lean << 'EOF'
import Lake
open Lake DSL

package mathlib_atp where
  leanOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib MathlibAtp where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"
EOF
fi

# Create lean-toolchain if not exists
if [ ! -f "lean-toolchain" ]; then
    echo "Creating lean-toolchain..."
    echo "leanprover/lean4:v4.31.0" > lean-toolchain
fi

# Create Main.lean if not exists
if [ ! -f "Main.lean" ]; then
    echo "Creating Main.lean..."
    cat > Main.lean << 'EOF'
-- This file exists to make the project buildable
-- The actual imports are handled by Pantograph at runtime
import Init
EOF
fi

echo "Project files created. Now run:"
echo "  cd $PROJECT_DIR && lake update && lake build"
echo ""
echo "This will download Mathlib (~2GB) and build it (~10-30 min first time)."
