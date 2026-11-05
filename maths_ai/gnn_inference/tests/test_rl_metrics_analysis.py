import json

import pytest

from maths_ai.gnn_inference.scripts import analyze_rl_run as analysis


def training(index=0, **changes):
    return {
        "round": index, "search_evidence": analysis.SEARCH_EVIDENCE_VERSION,
        "attempted": 5, "solved": 1, "seeds_inadmissible": 1,
        "searches_failed": 1, "searches_resource_limited": 1,
        "searches_hard_timed_out": 1, "actor_loss": 0.0, "critic_loss": 0.2,
        "onpolicy_optimizer_step": 1, "htps_optimizer_steps": 2,
        "parameter_updates_this_round": 3, "wall_clock_s": 2,
        **changes,
    }


def stats(solved=1):
    return {
        "attempted": 4, "solved": solved, "searched": 4, "proof_rate": solved / 4,
        "eval_subset_sha256": "same-ordered-inputs", "eval_subset_size": 4,
        "evaluation": {"metric": analysis.EVAL_METRIC_VERSION, "mode": "sample",
                       "seed": 42, "repeats": 1, "seed_derivation": "test-v1",
                       "pln_translation": "test-v1"},
        "repeats": [{"repeat": 0, "proposal_seeds_sha256": "same-draw-seeds"}],
        "pln_runtime": None,
    }


def evaluation_report(solved=1):
    report = {
        "kind": "rl-evaluation-report", "result": stats(solved), "checkpoint": "base.pt",
        "search_settings": {"selection_policy": "puct", "use_pln": False},
        "budgets": {"max_nodes": 64}, "dataset": {"split": "test"},
        "workers": 1, "admission": {"environment": {"identity": "same-env"}},
        "path": "base.json",
    }
    report["computed"] = analysis.evaluation(report["result"])
    return report


def make_run(tmp_path, rows):
    root = tmp_path / "run"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"search_evidence": analysis.SEARCH_EVIDENCE_VERSION}))
    (root / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def test_training_and_evaluation_are_separate_and_updates_are_explicit(tmp_path):
    root = make_run(tmp_path, [training(pln_ok=2), {"round": 0, "eval": stats()}, training(1)])
    output = tmp_path / "analysis"
    report = analysis.analyze(root, output, window=1)
    train = report["training"]
    assert (train["attempted"], train["searched"], train["solved"]) == (10, 4, 2)
    assert train["proof_rate"] == 0.5
    assert train["counters"]["onpolicy_optimizer_step"] == 2
    assert train["counters"]["htps_optimizer_steps"] == 4
    assert train["counters"]["parameter_updates_this_round"] == 6
    assert train["counters"]["num_transitions"] is None
    assert train["pln_counts"] == {"pln_ok": 2}
    assert train["first_window"]["means_of_reported_round_metrics"]["actor_loss"] == 0
    assert len(report["periodic_evaluations"]) == 1
    assert report["checkpoint_comparison"] is None
    assert {p.name for p in output.iterdir()} == {
        "report.md", "summary.json", "training_rounds.csv", "evaluations.csv",
    }
    assert "not an untrained baseline" in (output / "report.md").read_text()
    with pytest.raises(FileExistsError):
        analysis.analyze(root, output)


def test_outcome_does_not_double_subtract_hard_timeouts():
    assert analysis.outcome(training())["searched"] == 2
    assert analysis.outcome({"attempted": 1, "solved": 0, "searches_failed": 1})["proof_rate"] is None
    with pytest.raises(ValueError, match="Solved count"):
        analysis.outcome(training(solved=3))
    with pytest.raises(ValueError, match="Excluded attempts"):
        analysis.outcome(training(attempted=2))


def test_matching_evaluations_produce_a_descriptive_difference():
    base, trained = evaluation_report(), evaluation_report(3)
    result = analysis.comparison(base, trained)
    assert result["recorded_conditions_match"]
    assert result["proof_rate_difference_percentage_points"] == 50


@pytest.mark.parametrize("field,value", [
    ("eval_subset_sha256", "different-inputs"), ("eval_subset_size", 5),
    ("pln_runtime", {"commit": "different"}),
    ("repeats", [{"repeat": 0, "proposal_seeds_sha256": "different"}]),
    ("repeats", [{"proposal_seeds_sha256": "same-draw-seeds"}]),
])
def test_different_evaluation_inputs_suppress_comparison(field, value):
    base, trained = evaluation_report(), evaluation_report(3)
    trained["result"][field] = value
    result = analysis.comparison(base, trained)
    assert not result["recorded_conditions_match"]
    assert result["proof_rate_difference_percentage_points"] is None


@pytest.mark.parametrize("field,value", [
    ("budgets", {"max_nodes": 128}), ("workers", 2),
    ("search_settings", {"selection_policy": "legacy", "use_pln": False}),
    ("admission", None), ("dataset", {}),
])
def test_different_or_missing_runtime_provenance_suppresses_difference(field, value):
    base, trained = evaluation_report(), evaluation_report(3)
    trained[field] = value
    result = analysis.comparison(base, trained)
    assert not result["recorded_conditions_match"]
    assert result["mismatches"]


def test_zero_searches_and_missing_pln_identity_suppress_difference():
    base, trained = evaluation_report(), evaluation_report(0)
    trained["result"].update(attempted=0, searched=0)
    trained["computed"] = analysis.evaluation(trained["result"])
    assert not analysis.comparison(base, trained)["recorded_conditions_match"]
    base, trained = evaluation_report(), evaluation_report()
    base["search_settings"]["use_pln"] = trained["search_settings"]["use_pln"] = True
    assert "Missing PLN runtime identity." in analysis.comparison(base, trained)["mismatches"]


def test_invalid_evaluation_denominators_and_contracts_are_rejected():
    value = stats()
    value["searched"] = 3
    with pytest.raises(ValueError, match="denominator"):
        analysis.evaluation(value)
    value = stats()
    value["proof_rate"] = 0.9
    with pytest.raises(ValueError, match="proof rate"):
        analysis.evaluation(value)
    value = stats()
    value["evaluation"]["repeats"] = 0
    with pytest.raises(ValueError, match="Invalid evaluation"):
        analysis.evaluation(value)


def test_incomplete_tail_is_not_a_complete_bad_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(training()) + '\n{"round":')
    rows, notes = analysis.load_metrics(path)
    assert len(rows) == 1 and len(notes) == 1
    path.write_text('{"round":\n' + json.dumps(training()) + "\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        analysis.load_metrics(path)


def test_duplicate_rounds_and_wrong_evidence_are_rejected(tmp_path):
    root = make_run(tmp_path, [training(), training()])
    with pytest.raises(ValueError, match="Duplicate training"):
        analysis.analyze(root, tmp_path / "out")
    (root / "metrics.jsonl").write_text(json.dumps(training(search_evidence="old")))
    with pytest.raises(ValueError, match="different search evidence"):
        analysis.analyze(root, tmp_path / "out")


def test_nonfinite_values_are_flagged_and_serialized_as_null(tmp_path):
    root = make_run(tmp_path, [training(actor_loss=float("nan"))])
    output = tmp_path / "out"
    analysis.analyze(root, output)
    report = json.loads((output / "summary.json").read_text())
    assert report["training"]["nonfinite_fields"] == {"actor_loss": 1}
    assert report["training"]["first_window"]["means_of_reported_round_metrics"]["actor_loss"] is None
    assert "nan" not in (output / "training_rounds.csv").read_text().lower()


def test_cli_with_paired_reports(tmp_path):
    root = make_run(tmp_path, [training()])
    reports = []
    for name, solved in (("base", 1), ("trained", 3)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(evaluation_report(solved)))
        reports.append(path)
    assert analysis.main([
        "--run-dir", str(root), "--output", str(tmp_path / "out"),
        "--baseline-eval", str(reports[0]), "--trained-eval", str(reports[1]),
    ]) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["checkpoint_comparison"]["proof_rate_difference_percentage_points"] == 50
