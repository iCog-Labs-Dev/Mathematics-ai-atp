"""Import smoke test for the modularized translator_modules package."""

import sys
from pathlib import Path

# Ensure translator directory is on sys.path for direct imports
_TRANSLATOR_DIR = str(Path(__file__).parent)
if _TRANSLATOR_DIR not in sys.path:
    sys.path.insert(0, _TRANSLATOR_DIR)


def test_translator_imports():
    # Layer 0: constants
    from translator_modules.constants import LEAN_TO_PETTA, TYPE_ATOMS, map_const
    assert len(LEAN_TO_PETTA) > 0
    assert len(TYPE_ATOMS) > 0
    assert map_const("And") == "∧"

    # Layer 1: normalizer
    from translator_modules.normalizer import sanitize_metta_symbol, VariableNormalizer
    assert sanitize_metta_symbol("P") == "P"
    assert sanitize_metta_symbol("?m.123") == "mm.123"
    vn = VariableNormalizer(normalize=False)
    assert vn.get("Q") == "Q"
    vn2 = VariableNormalizer(normalize=True)
    assert vn2.get("P") == "v0"
    assert vn2.get("Q") == "v1"
    assert vn2.get("P") == "v0"  # should be cached

    # Layer 2: renderer
    from translator_modules.renderer import render_formula, render_term, validate_formula_shape, build_query
    assert render_formula("P") == "(P)"
    assert render_formula(["Not", "P"]) == "(Not (P))"
    assert "Implication" in render_formula(["Implication", "P", "Q"])
    assert "query" in build_query("P")

    # Layer 3: parser
    from translator_modules.parser import (
        to_plain, literal_to_concept, parse_sexp_string, split_top_level_infix,
        parse_simple_pp, flatten_app, canonicalize_application, translate_expr_dict,
        translate_sexp_obj, process_foralls, normalize_logic, expr_from_field
    )
    assert literal_to_concept(0) == "Zero"
    assert literal_to_concept(1) == "One"
    assert parse_sexp_string("(a b c)") == ["a", "b", "c"]
    norm = VariableNormalizer(normalize=False)
    assert parse_simple_pp("P", norm) == "P"

    # Layer 4: extractor
    from translator_modules.extractor import (
        is_nonlogical_type_expr, is_has_type_expr, extract_hypotheses,
        extract_target, recurry, essentialize_subgoal, proof_state_after_tactics
    )
    assert is_nonlogical_type_expr("PROP") is True
    assert is_nonlogical_type_expr("P") is False
    assert is_has_type_expr(["HasType", "x", "Nat"]) is True
    result = recurry(["P"], "Q")
    assert result == ["Implication", "P", "Q"]

    # Layer 5: runner
    from translator_modules.runner import (
        safe_name, shell_quote, extract_stv_scores, score_from_stv,
        sample_fallback_score, parse_and_rank_logs, print_ranked_results,
        write_runner_script
    )
    assert safe_name("test-1") == "test-1"
    assert safe_name("") == "test"
    assert score_from_stv(0.8, 0.6) == 0.48
    stvs = extract_stv_scores("Result: (: proof1 (Q) (STV 0.85 0.72))")
    assert len(stvs) == 1
    assert stvs[0] == (0.85, 0.72)

    # Layer 6: cli (skip Pantograph-dependent functions)
    from translator_modules.cli import load_input_json, parse_args, run_demo, main

    # Test the facade
    from translator import (
        LEAN_TO_PETTA, TYPE_ATOMS, map_const,
        sanitize_metta_symbol, VariableNormalizer,
        render_formula, build_query,
        parse_sexp_string, to_plain,
        essentialize_subgoal, recurry,
        safe_name, extract_stv_scores,
        load_input_json, parse_args,
    )


if __name__ == "__main__":
    test_translator_imports()
    print("=== ALL IMPORT TESTS PASSED! ===")
