"""Bounded best-first proof search driven exclusively by GNN action probabilities.

The heap stores unexplored parent/action edges. Each accepted child is expanded
once, so the next native trial has the smallest known cumulative negative log
probability. No printed-state merging, proof LLM, value model, or tactic fallback.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import heapq
import math
import time

from .live_state import LiveStateError, TacticFailure
from .proof_verification import checked_header


@dataclass(frozen=True)
class GnnSearchBudget:
    model_calls: int = 32
    tactic_trials: int = 96
    nodes: int = 64  # Includes the root and closed/rejected-verification children.
    depth: int = 16
    top_k: int = 8
    seconds: float = 120
    verification_seconds: float = 30

    def __post_init__(self):
        for name in ("model_calls", "tactic_trials", "nodes", "depth", "top_k"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seconds", "verification_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def accepted_path(nodes: list[dict], identity: int) -> list[str]:
    path = []
    while nodes[identity]["parent"] is not None:
        path.append(nodes[identity]["action"])
        identity = nodes[identity]["parent"]
    return list(reversed(path))


async def search_gnn(policy, session, header: str, verifier, *, budget=None, checkpoint=None) -> dict:
    budget = budget or GnnSearchBudget()
    header, _ = checked_header(header)
    started = time.monotonic()
    report = {"format": "gnn_best_first_v1", "generator": "gnn_joint_source_flat_v2",
        "target_header": header, "target_sha256": hashlib.sha256(header.encode()).hexdigest(),
        "budget": asdict(budget), "priority": "cumulative_action_negative_log_probability",
        "retrieval_text_policy": "live_pretty_fields_v1", "status": "running", "proof_verified": False,
        "model_calls": 0, "tactic_trials": 0, "model_seconds": 0.0, "nodes": [], "expansions": [],
        "trials": [], "verifications": [], "accepted_path": [], "pruned_depth": 0,
        "duplicate_candidates": 0, "frontier_size": 0}
    frontier, snapshots = [], {}
    sequence = 0
    call_limited = False

    def remaining():
        return budget.seconds - (time.monotonic() - started)

    def save():
        report["seconds"] = time.monotonic() - started
        report["frontier_size"] = len(frontier)
        if checkpoint is not None:
            checkpoint(report)

    def expand(identity):
        nonlocal sequence, call_limited
        node, snapshot = report["nodes"][identity], snapshots[identity]
        if node["depth"] >= budget.depth:
            report["pruned_depth"] += 1
            return
        if report["model_calls"] >= budget.model_calls:
            call_limited = True
            return
        if remaining() <= 0:
            return
        report["model_calls"] += 1
        expansion = {"node": identity, "candidates": [], "rejected": []}
        report["expansions"].append(expansion)
        save()
        model_started = time.monotonic()
        try:
            prediction = policy.predict(snapshot.goals[0], top_k=budget.top_k)
        except LiveStateError as error:
            expansion["state_error"] = str(error)
            return
        finally:
            report["model_seconds"] += time.monotonic() - model_started
        expansion["rejected"] = prediction.rejected
        seen = set()
        for candidate in prediction.candidates:
            score = candidate.log_probability
            if not math.isfinite(score) or score > 0:
                expansion["rejected"].append({"tactic": candidate.tactic, "reason": "invalid_log_probability"})
                continue
            if candidate.code in seen:
                report["duplicate_candidates"] += 1
                continue
            seen.add(candidate.code)
            cost = node["cost"] - score
            expansion["candidates"].append({**candidate.to_dict(), "path_cost": cost})
            heapq.heappush(frontier, (cost, sequence, identity, candidate))
            sequence += 1
        save()

    try:
        root = await asyncio.wait_for(session.start_declaration(header), timeout=max(remaining(), 0.001))
        snapshots[0] = root
        report["nodes"].append({"id": 0, "parent": None, "action": None, "cost": 0.0,
            "depth": 0, "state_id": root.state_id, "goals": list(root.goals)})
        expand(0)
        while frontier:
            if remaining() <= 0:
                report["status"] = "time_budget"
                break
            if report["tactic_trials"] >= budget.tactic_trials:
                report["status"] = "trial_budget"
                break
            if len(report["nodes"]) >= budget.nodes:
                report["status"] = "node_budget"
                break
            cost, _, parent, candidate = heapq.heappop(frontier)
            report["tactic_trials"] += 1
            trial = {"parent": parent, "code": candidate.code, "path_cost": cost, "status": "pending"}
            report["trials"].append(trial)
            save()
            try:
                child = await asyncio.wait_for(session.step(snapshots[parent], candidate.code),
                    timeout=max(min(session.timeout, remaining()), 0.001))
            except TacticFailure as error:
                trial.update(status="lean_error", error=str(error))
                save()
                continue
            identity = len(report["nodes"])
            snapshots[identity] = child
            node = {"id": identity, "parent": parent, "action": candidate.code, "cost": cost,
                    "depth": report["nodes"][parent]["depth"] + 1,
                    "state_id": child.state_id, "goals": list(child.goals)}
            report["nodes"].append(node)
            trial.update(status="executed", child=identity, pending_goals=len(child.goals))
            save()
            if child.closed:
                if remaining() <= 0:
                    report["status"] = "time_budget"
                    break
                path = accepted_path(report["nodes"], identity)
                verdict = await verifier.verify(header, path,
                    timeout=min(budget.verification_seconds, remaining()))
                report["verifications"].append({"node": identity, **verdict})
                if verdict.get("status") == "passed":
                    report.update(status="solved", proof_verified=True, winning_node=identity, accepted_path=path)
                    break
                node["verification_rejected"] = True
                save()
            else:
                expand(identity)
        else:
            report["status"] = ("time_budget" if remaining() <= 0 else "model_call_budget" if call_limited
                                else "depth_budget" if report["pruned_depth"] else "frontier_exhausted")
    except asyncio.TimeoutError:
        report["status"] = "time_budget" if remaining() <= 0 else "native_timeout"
    except LiveStateError as error:
        report.update(status="input_error", error=str(error))
    except (RuntimeError, OSError) as error:
        report.update(status="runtime_error", error=str(error))
    finally:
        save()
    return report
