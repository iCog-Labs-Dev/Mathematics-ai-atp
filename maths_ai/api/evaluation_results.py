"""Assemble results exclusively from durable, handle-free job records."""

from __future__ import annotations


def evaluation_result(job, state, attempts, stats):
    from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import (
        EvaluationSettings,
        repeat_row_for_seeds,
        summarize_rows,
    )

    complete = state == "completed"
    progress = job["progress"]
    result = {
        "kind": "mcts-evaluation-result",
        "version": "mcts-service-result-v2",
        "job_id": job["id"],
        "state": state,
        "complete": complete,
        "partial": not complete,
        "settings": job["settings"],
        "provenance": job["provenance"],
        "admission": progress.get("admission"),
        "attempts": attempts,
        "input_records": progress.get("input_records", []),
        "proposal_records": progress.get("proposal_records", []),
    }
    if progress.get("subset_sha256") and "evaluation" in job["settings"]:
        settings = EvaluationSettings(**job["settings"]["evaluation"])
        digests = progress.get("proposal_seeds_sha256", [])
        rows = [
            repeat_row_for_seeds(r, digests[r], values)
            for r, values in sorted(stats.items())
            if r < len(digests)
        ]
        result["summary" if complete else "partial_summary"] = summarize_rows(
            settings,
            progress["subset_size"],
            progress["subset_sha256"],
            rows,
        )
        result["repeats_completed"] = progress.get("repeats_completed", 0)
    return result
