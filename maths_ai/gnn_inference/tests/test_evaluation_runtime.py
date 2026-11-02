"""The shared evaluation summary: chunked and whole-round evaluation agree."""

from __future__ import annotations

from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import (
    EVAL_METRIC_VERSION,
    EvaluationSettings,
    chunk_stats,
    merge_stats,
    repeat_row,
    summarize_evaluation,
)
from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import TheoremItem
from maths_ai.lean_replay import seed_from_text


def _items(count: int) -> list[TheoremItem]:
    return [TheoremItem(seed=seed_from_text(goal=f"p{i}"), tactic_label="", size=i) for i in range(count)]


def _stats(attempted, solved, inadmissible=0, failed=0, extra=0.0):
    return {
        "attempted": float(attempted),
        "collected": float(attempted - inadmissible),
        "solved": float(solved),
        "searches_failed": float(failed),
        "searches_cooperative_deadline": 0.0,
        "searches_hard_timed_out": 0.0,
        "seeds_inadmissible": float(inadmissible),
        # A per-round mean: it must not be summed across chunks.
        "queue_wait_mean_s": extra,
    }


def test_chunked_repeats_summarize_like_whole_rounds():
    items = _items(4)
    settings = EvaluationSettings(mode="greedy", seed=7, repeats=2)
    whole = [
        repeat_row(r, settings.proposals(items, r), chunk_stats(_stats(4, 2, inadmissible=1)))
        for r in range(2)
    ]
    chunked = []
    for r in range(2):
        total: dict[str, float] = {}
        merge_stats(total, _stats(2, 1, inadmissible=1, extra=3.0))
        merge_stats(total, _stats(2, 1, extra=5.0))
        chunked.append(repeat_row(r, settings.proposals(items, r), total))
    assert summarize_evaluation(settings, items, chunked) == summarize_evaluation(settings, items, whole)
    assert "queue_wait_mean_s" not in chunked[0]


def test_proof_rate_counts_searched_seeds_only():
    items = _items(3)
    settings = EvaluationSettings(seed=1)
    row = repeat_row(0, settings.proposals(items, 0), chunk_stats(_stats(3, 1, inadmissible=1)))
    summary = summarize_evaluation(settings, items, [row])
    assert summary["searched"] == 2.0
    assert summary["proof_rate"] == 0.5
    assert summary["evaluation"]["metric"] == EVAL_METRIC_VERSION
