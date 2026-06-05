"""
translator_modules — Modularized Lean/PyPantograph → PeTTaChainer translator.

This package re-exports all public symbols from its submodules so that
``from translator_modules import X`` works for any symbol that was previously
available in the monolithic ``translator.py``.
"""

# --- Layer 0: pure data ---
from .constants import LEAN_TO_PETTA, TYPE_ATOMS, map_const

# --- Layer 1: symbol handling ---
from .normalizer import VariableNormalizer, sanitize_metta_symbol

# --- Layer 2: rendering (no internal deps) ---
from .renderer import build_query, render_formula, render_term, validate_formula_shape

# --- Layer 3: expression parsing ---
from .parser import (
    canonicalize_application,
    expr_from_field,
    flatten_app,
    literal_to_concept,
    normalize_logic,
    parse_sexp_string,
    parse_simple_pp,
    process_foralls,
    split_top_level_infix,
    to_plain,
    translate_expr_dict,
    translate_sexp_obj,
)

# --- Layer 4: extraction & subgoal processing ---
from .extractor import (
    essentialize_subgoal,
    extract_hypotheses,
    extract_target,
    is_has_type_expr,
    is_nonlogical_type_expr,
    proof_state_after_tactics,
    recurry,
)

# --- Layer 5: runner, log parsing, ranking ---
from .runner import (
    extract_stv_scores,
    parse_and_rank_logs,
    print_ranked_results,
    safe_name,
    sample_fallback_score,
    score_from_stv,
    shell_quote,
    write_runner_script,
)

# --- Layer 6: CLI & orchestration ---
from .cli import load_input_json, main, parse_args, run_demo
