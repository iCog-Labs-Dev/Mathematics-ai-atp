#! /bin/bash

# Run the proover on the given goal, using the given model.
# Usage: ./run_proover.sh 

# Example: ./run_proover.sh "theorem example : 1 = 1

#import the joint_inference.py script and run the proover on the given goal, using the given model.

# Get the directory where this script resides

# Add the current directory to PYTHONPATH so the maths_ai module is found
export PYTHONPATH="$(pwd)"

# Source elan if available, to provide Lean 4 (lake)
if [ -f "$HOME/.elan/env" ]; then
    source "$HOME/.elan/env"
fi

# Use Python from the uv virtual environment
PYTHON=".venv/bin/python"
SCRIPT="maths_ai/hybrid_reasoner/joint_inference.py"

# Check that the virtual environment exists
if [ ! -x "$PYTHON" ]; then
    echo "Error: uv virtual environment not found at .venv"
    echo "Create it with:"
    echo "  uv venv"
    exit 1
fi

# Run the script with all provided arguments
"$PYTHON" "$SCRIPT" "$@"