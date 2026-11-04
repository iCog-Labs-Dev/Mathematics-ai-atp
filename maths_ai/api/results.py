"""Plain-data views of search results, safe to send over IPC and store.

Only kernel-certified roots count as proofs. A READY graph candidate, a
successful exploratory tactic and an empty goal list are reported as such and
never as a proof. Traces keep the exact commands and discharged obligations
the certified replay recorded.
"""

from __future__ import annotations

import math
from typing import Any

# Outcomes a search attempt can end with (``CollectionRecord.outcome``).
SEARCH_OUTCOMES = (
    "solved",
    "unsolved",
    "cooperative_deadline",
    "inadmissible",
    "hard_timeout",
    "error",
)


def plain(value: Any, *, depth: int = 0) -> Any:
    """``value`` as JSON-safe data: no tensors, handles or non-finite floats."""
    if depth > 64:
        raise ValueError("Result nesting is too deep to serialize")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value  # enums
    if isinstance(value, dict):
        return {str(key): plain(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item, depth=depth + 1) for item in value]
    if hasattr(value, "as_dict"):
        return plain(value.as_dict(), depth=depth + 1)
    if hasattr(value, "item") and callable(value.item):  # a 0-d tensor or numpy scalar
        return plain(value.item(), depth=depth + 1)
    return str(value)


def graph_view(graph) -> dict[str, Any]:
    """The certified outcome and bounded diagnostics of one finished search."""
    summary = graph.summary()
    return {
        "certified": bool(graph.is_solved()),
        "candidate": bool(graph.has_candidate()),
        "end_reason": plain(graph.end_reason),
        "certificate": plain(summary.get("certificate")),
        "proof_trace": plain(graph.proof_trace()),
        "artifact": plain(graph.artifact) if graph.artifact is not None else None,
        "execution": plain(graph.execution),
        "search": plain(graph.search_counters()),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
    }


def record_view(record, *, proposal=None, include_trace: bool = True) -> dict[str, Any]:
    """One attempted search: its outcome, timing and, when it ran, its graph view."""
    view: dict[str, Any] = {
        "outcome": record.outcome,
        "worker_id": record.worker_id,
        "queue_wait_s": record.queue_wait_s,
        "search_s": record.search_s,
        "exception_category": record.exception_category,
    }
    if proposal is not None:
        view["proposal"] = {"mode": proposal.mode, "seed": proposal.seed, "derivation": proposal.derivation}
    result = record.result
    if result is not None and getattr(result, "graph", None) is not None:
        graph = graph_view(result.graph)
        if not include_trace and not graph["certified"]:
            graph.pop("proof_trace")
            graph.pop("artifact")
        view.update(graph)
        view["observations"] = len(result.observations)
        view["draws"] = result.draws()
    return view
