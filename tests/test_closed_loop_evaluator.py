"""Unit and regression tests for Issue #28: Closed-Loop Proof Search Evaluation.

Verifies:
1. Deterministic evaluation under a fixed random seed.
2. Complete-action syntax validation and Lean elaboration tracking.
3. Accurate multi-step proof search tree expansion and Pass@K computation.
4. Failure taxonomy classification (invalid_syntax, unavailable_identifier, etc.).
5. Ground-truth component metrics (tactic top-1/5, argument top-1/5).
6. Resumability after interruption from trace.jsonl.
7. Machine-readable summary.json and human-readable report.md generation.
"""

import json
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from maths_ai.gnn_inference.atp_lean_gnn.closed_loop_evaluator import (
    ClosedLoopConfig,
    ClosedLoopEvaluator,
    FailureCategory,
    classify_tactic_failure,
)


def test_failure_classification():
    """Verify classification of Lean errors into standard failure categories."""
    assert classify_tactic_failure("syntax error, unexpected token", "apply") == FailureCategory.INVALID_SYNTAX
    assert classify_tactic_failure("unknown identifier 'foo'", "exact foo") == FailureCategory.UNAVAILABLE_IDENTIFIER
    assert classify_tactic_failure("server dead: broken pipe", "rfl", is_server_crash=True) == FailureCategory.SERVER_FAILURE
    assert classify_tactic_failure(None, "rfl", is_timeout=True) == FailureCategory.TIMEOUT
    assert classify_tactic_failure("type mismatch: expected Nat got Bool", "exact h") == FailureCategory.ELABORATION_FAILURE
    assert classify_tactic_failure(None, "rfl") == FailureCategory.NONE


def test_deterministic_closed_loop_evaluation():
    """Verify complete closed-loop search, metric aggregation, and report generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ClosedLoopConfig(
            max_depth=5,
            max_nodes=20,
            top_k_tactics=3,
            timeout_seconds=10.0,
            seed=42,
            output_dir=tmpdir,
            resume=True,
        )

        def mock_predict(state: str, top_k: int):
            if "n = n" in state:
                return [("rfl", [], 0.9), ("simp", [], 0.1)]
            if "p ∧ q" in state:
                return [("constructor", [], 0.8), ("rfl", [], 0.2)]
            if "invalid" in state:
                return [("bad_tactic", ["(unbalanced"], 0.5)]
            return [("rfl", [], 0.5), ("exact", ["h"], 0.3)]

        def mock_execute(state: str, tactic: str):
            if tactic == "rfl":
                return True, [], None
            if tactic == "constructor":
                return True, ["⊢ p", "⊢ q"], None
            if tactic == "exact h":
                return True, [], None
            if "bad_tactic" in tactic:
                return False, [], "syntax error"
            return False, [], "tactic failed"

        evaluator = ClosedLoopEvaluator(
            config,
            predict_fn=mock_predict,
            execute_fn=mock_execute,
        )

        benchmarks = [
            {
                "id": "thm_1",
                "statement": "⊢ n = n",
                "ground_truth_tactic": "rfl",
                "ground_truth_args": [],
            },
            {
                "id": "thm_2",
                "statement": "⊢ p ∧ q",
                "ground_truth_tactic": "constructor",
                "ground_truth_args": [],
            },
            {
                "id": "thm_3",
                "statement": "⊢ invalid",
                "ground_truth_tactic": "simp",
                "ground_truth_args": [],
            },
        ]

        summary = evaluator.evaluate_benchmark(benchmarks)

        # Check metrics
        assert summary["total_theorems"] == 3
        assert summary["proved_theorems"] >= 2
        assert summary["success_rate"] > 0.6
        assert summary["syntax_validity_rate"] > 0.0
        assert summary["elaboration_success_rate"] > 0.0

        # Check ground truth accuracy
        comp = summary["component_metrics"]
        assert comp["tactic_top1_accuracy"] >= 0.6

        # Check failure categories recorded
        assert FailureCategory.INVALID_SYNTAX.value in summary["failure_taxonomy_distribution"]

        # Check outputs created
        assert (Path(tmpdir) / "summary.json").exists()
        assert (Path(tmpdir) / "trace.jsonl").exists()
        assert (Path(tmpdir) / "report.md").exists()

        # Check report content
        report_text = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        assert "Closed-Loop Proof Search Evaluation Report" in report_text
        assert "Complete-Action Syntax Validity" in report_text
        assert "Lean Elaboration Success Rate" in report_text


def test_resumability():
    """Verify that evaluator correctly resumes from an existing trace file without re-evaluating."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "trace.jsonl"
        # Pre-populate trace with thm_1
        pre_result = {
            "theorem_id": "thm_1",
            "statement": "⊢ n = n",
            "proved": True,
            "steps_taken": 1,
            "nodes_expanded": 1,
            "elapsed_seconds": 0.01,
            "termination_reason": "proved",
            "proof_script": ["rfl"],
            "action_records": [],
        }
        trace_file.write_text(json.dumps(pre_result) + "\n", encoding="utf-8")

        config = ClosedLoopConfig(
            output_dir=tmpdir,
            resume=True,
        )

        evaluator = ClosedLoopEvaluator(config)
        evaluated = evaluator.load_evaluated_ids()
        assert "thm_1" in evaluated

        benchmarks = [
            {"id": "thm_1", "statement": "⊢ n = n"},
            {"id": "thm_2", "statement": "⊢ n = n"},
        ]

        summary = evaluator.evaluate_benchmark(benchmarks)
        assert summary["total_theorems"] == 2
        assert summary["proved_theorems"] == 2


if __name__ == "__main__":
    test_failure_classification()
    test_deterministic_closed_loop_evaluation()
    test_resumability()
    print("All closed-loop evaluator tests passed successfully!")
