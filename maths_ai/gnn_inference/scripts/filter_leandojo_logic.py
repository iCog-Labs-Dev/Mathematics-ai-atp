"""
Filter the LeanDojo dataset (tasksource/leandojo on Hugging Face) down to
Propositional Logic and First-Order Logic (FOL) theorems.

WHAT THIS DOES
--------------
1. Downloads the dataset from Hugging Face (train/validation/test splits).
2. Keeps only theorems whose Mathlib file path lives under known logic
   folders (Step A), OR whose statement is built only from logic
   connectives with no heavier machinery (Step B, optional/stricter).
3. Saves the filtered result to disk as JSON, one file per split.

HOW TO RUN
----------
pip install datasets
python filter_leandojo_logic.py

Adjust LOGIC_PATH_PREFIXES and the symbol filter below to tune precision
vs. recall as you inspect results.
"""

import json
import re
from datasets import load_dataset

# ---------------------------------------------------------------------------
# STEP A: File-path allowlist.
# These are the Mathlib4 folders that correspond to propositional logic and
# first-order logic content. Adjust/add paths as you discover more via
# manual inspection (Step 3 of the plan: sanity-check the result).
# ---------------------------------------------------------------------------
LOGIC_PATH_PREFIXES = [
    "Mathlib/Logic/Basic.lean",     # core propositional logic (And, Or, Not, Iff, etc.)
    "Mathlib/Logic/Basic/",
    "Mathlib/ModelTheory/",         # first-order logic / model theory
    "Mathlib/Order/Heyting/",       # Heyting algebras -> intuitionistic propositional logic
    "Mathlib/Order/BooleanAlgebra", # Boolean algebra -> classical propositional logic
    "Mathlib/Tactic/Tauto",         # propositional tautology tactic support
]

# Subfolders/files under the above that are NOT core logic and should be
# excluded even though their parent folder matched above (Mathlib's
# "Logic/" folder is a broad catch-all for foundational utilities, not
# just propositional connectives).
LOGIC_PATH_EXCLUDE = [
    "Mathlib/Logic/Equiv/",     # type equivalences/permutations, not logic connectives
    "Mathlib/Logic/Encodable/", # encoding of data structures
    "Mathlib/Logic/Function/",  # general function utilities (curry, comp, etc.)
    "Mathlib/Logic/Denumerable",
    "Mathlib/Logic/Embedding/",
]

# ---------------------------------------------------------------------------
# STEP B (optional, stricter): symbol-based filter on the theorem statement.
# A statement "looks propositional/FOL" if it only uses logic connectives
# and quantifiers, and does NOT reference heavier machinery keywords.
# This is a heuristic, not a proof -- always spot-check the output.
# ---------------------------------------------------------------------------
LOGIC_SYMBOLS = re.compile(r"[∧∨¬→↔∀∃]")
DISALLOWED_KEYWORDS = re.compile(
    r"\b(Ring|Group|Module|Topology|Continuous|Measure|Metric|Category|"
    r"Filter|Cardinal|Deriv|Integral|Matrix|Polynomial|Ideal)\b"
)


def path_is_logic(file_path: str) -> bool:
    if any(file_path.startswith(p) for p in LOGIC_PATH_EXCLUDE):
        return False
    return any(file_path.startswith(p) for p in LOGIC_PATH_PREFIXES)


def statement_is_logic(theorem_row) -> bool:
    """Looks at the first traced tactic's state (if any) as a proxy for the
    statement content. Falls back to True (don't exclude) if no tactics."""
    tactics = theorem_row.get("traced_tactics") or []
    if not tactics:
        return True  # no tactic text to check; rely on path filter alone
    text = tactics[0].get("state_before", "")
    if not text:
        return True
    has_logic_symbol = bool(LOGIC_SYMBOLS.search(text))
    has_disallowed = bool(DISALLOWED_KEYWORDS.search(text))
    return has_logic_symbol and not has_disallowed


def filter_split(dataset_split, use_symbol_filter: bool = False):
    kept = []
    for row in dataset_split:
        if not path_is_logic(row["file_path"]):
            continue
        if use_symbol_filter and not statement_is_logic(row):
            continue
        kept.append(row)
    return kept


def main():
    print("Downloading tasksource/leandojo from Hugging Face...")
    ds = load_dataset("tasksource/leandojo")

    for split_name in ds.keys():
        print(f"\nFiltering split: {split_name} ({len(ds[split_name])} rows)")
        filtered = filter_split(ds[split_name], use_symbol_filter=False)
        print(f"  Kept {len(filtered)} / {len(ds[split_name])} theorems "
              f"({100 * len(filtered) / len(ds[split_name]):.2f}%)")

        out_path = f"leandojo_logic_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(filtered, f, indent=2)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()