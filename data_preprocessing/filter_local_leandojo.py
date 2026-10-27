"""
LeanDojo Local-Only Theorem Preprocessing Script

Filters theorems from LeanDojo (Lean 4 / Mathlib4) to extract proofs that:
1. Rely ONLY on local hypotheses in context (`state_before`).
2. Use zero external lemmas from the global Mathlib environment.
3. Exclude automated solvers (like `simp`, `linarith`, `aesop`) to prevent implicit premise retrieval.
"""

import os
import re
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset


def extract_local_hypotheses(state_before: str) -> set:
    if "⊢" not in state_before:
        return set()
    context = state_before.split("⊢")[0]
    names = set()
    for line in context.strip().split("\n"):
        if ":" in line:
            var_part = line.split(":")[0].strip()
            for name in var_part.split():
                clean_name = re.sub(r'[^A-Za-z0-9_]', '', name)
                if clean_name:
                    names.add(clean_name)
    return names


SAFE_STANDALONE = {
    "rfl", "refl", "trivial", "assumption", "constructor", 
    "left", "right", "split", "intros", "ext", "done"
}

LEAN_KEYWORDS = {
    "with", "at", "only", "using", "case", "skip", "h", "rfl", "refl"
}


def is_single_tactic_local(tactic_str: str, local_vars: set) -> bool:
    t = tactic_str.strip()
    if not t:
        return True
        
    if t in SAFE_STANDALONE:
        return True
        
    if t.startswith("intro ") or t.startswith("intros "):
        return True

    # Reject automated solvers that secretly query global library lemmas
    if any(t.startswith(solver) for solver in ["simp", "linarith", "ring", "aesop", "nlinarith", "norm_num", "omega"]):
        return False

    tokens = re.findall(r'[A-Za-z0-9_]+', t)
    if not tokens:
        return False
        
    cmd = tokens[0]
    if cmd in {"exact", "apply", "cases", "rcases", "subst", "have", "refine"}:
        args = tokens[1:]
        for arg in args:
            if arg not in local_vars and arg not in LEAN_KEYWORDS:
                return False
        return True

    return False


def is_local_tactic(tactic_str: str, local_vars: set) -> bool:
    sub_tactics = re.split(r'<;>|;', tactic_str)
    return all(is_single_tactic_local(sub, local_vars) for sub in sub_tactics)


def is_pure_local_theorem(sample) -> bool:
    tactics = sample.get("traced_tactics", [])
    if not tactics:
        return False
        
    if tactics[-1].get("state_after", "").strip() != "no goals":
        return False
        
    for step in tactics:
        local_vars = extract_local_hypotheses(step.get("state_before", ""))
        tactic_text = step.get("tactic", "")
        if not is_local_tactic(tactic_text, local_vars):
            return False
            
    return True


def main():
    parser = argparse.ArgumentParser(description="Filter local-only proofs from LeanDojo")
    parser.add_argument("--output_dir", type=str, default="./data_preprocessing/output", help="Directory to save output jsonl files")
    parser.add_argument("--max_samples", type=int, default=2000, help="Maximum number of theorems to filter")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/validation/test)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_theorems = os.path.join(args.output_dir, f"local_theorems_{args.split}.jsonl")
    output_pairs = os.path.join(args.output_dir, f"local_state_tactic_pairs_{args.split}.jsonl")

    print(f"Loading LeanDojo Mathlib4 ({args.split} split)...")
    dataset = load_dataset("tasksource/leandojo", split=args.split, streaming=True)

    theorems_count = 0
    pairs_count = 0

    with open(output_theorems, "w", encoding="utf-8") as f_thm, \
         open(output_pairs, "w", encoding="utf-8") as f_pairs:
        
        pbar = tqdm(total=args.max_samples, desc="Filtering Local Theorems")
        for sample in dataset:
            if is_pure_local_theorem(sample):
                f_thm.write(json.dumps(sample) + "\n")
                theorems_count += 1
                
                for step in sample["traced_tactics"]:
                    pair = {
                        "theorem_name": sample.get("full_name"),
                        "file_path": sample.get("file_path"),
                        "input": f"[GOAL]\n{step['state_before']}",
                        "output": step["tactic"]
                    }
                    f_pairs.write(json.dumps(pair) + "\n")
                    pairs_count += 1
                    
                pbar.update(1)
                if theorems_count >= args.max_samples:
                    break
        pbar.close()

    print(f"\nCompleted! Saved:")
    print(f"  - {output_theorems} ({theorems_count} theorems)")
    print(f"  - {output_pairs} ({pairs_count} pairs)")


if __name__ == "__main__":
    main()
