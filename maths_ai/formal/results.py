"""The canonical version-2 formal result of a run.

Dispositions follow the persistent-atp contract and never overstate:

* only a certified attempt is ``proved-pending-replay``; an MCTS READY or
  JOINT graph candidate is not proof evidence and stays a bounded outcome;
* exhausting a step, node, simulation, time or Lean resource bound with the
  root still open is ``budget-exhausted``; it is not a dead root;
* running out of search alternatives is ``stagnated``, not a refutation;
* input, environment and infrastructure faults keep their own dispositions.

A certified proof is exported as states and tactic edges with every AND
child of each step, under run-scoped identities. A state's exact hash binds
the run's admitted input, environment and the state's position in the proof,
so equal goal texts in different contexts are never conflated. The complete
backend search summary is stored as an artifact rather than inlined.
"""

from __future__ import annotations

from typing import Any, Optional

from .canonical import object_ref

DISPOSITIONS = {
    "certified": "proved-pending-replay",
    "step-budget": "budget-exhausted",
    "node-budget": "budget-exhausted",
    "simulation-budget": "budget-exhausted",
    "deadline": "budget-exhausted",
    "lean-resource": "budget-exhausted",
    "frontier-exhausted": "stagnated",
    "alternatives-exhausted": "stagnated",
    "invalid-input": "invalid-request",
    "environment-mismatch": "environment-error",
    "crash": "internal-error",
    "interrupted": "internal-error",
    "cancelled": "cancelled",
}


def _namespace(run_id: str) -> str:
    return run_id.split("/", 1)[0]


def _goal_text(node: dict[str, Any]) -> str:
    return str(node.get("goal", ""))


def _command(node: dict[str, Any]) -> tuple[str, list[str]]:
    if "command" in node:
        return str(node["command"]), []
    return str(node.get("tactic", "")), [str(argument) for argument in node.get("arguments", [])]


def proof_graph(
    run_id: str, trace: dict[str, Any], *, input_artifact_hash: str, environment_hash: str
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    namespace = _namespace(run_id)
    states: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def visit(node: dict[str, Any], path: tuple[int, ...]) -> str:
        state_id = f"{namespace}/fs-{len(states) + 1}"
        goal = _goal_text(node)
        state = {
            "state_id": state_id,
            "kind": "goal",
            "goal_text": goal,
            "exact_state_hash": object_ref({
                "run_id": run_id,
                "input_artifact_hash": input_artifact_hash,
                "environment_hash": environment_hash,
                "occurrence": list(path),
                "goal": goal,
            }),
            "semantic_signature": object_ref({"goal": " ".join(goal.split())}),
            "status": "formally-closed",
        }
        states.append(state)
        children = [visit(child, (*path, index)) for index, child in enumerate(node.get("subgoals", []))]
        label, arguments = _command(node)
        edge = {
            "tactic_id": f"{namespace}/ta-{len(edges) + 1}",
            "source_state_id": state_id,
            "tactic_label": label,
            "executor_result": "lean-accepted",
            "subgoal_count": len(children),
            "produced_goal_ids": children,
            "status": "closed",
        }
        if arguments:
            edge["tactic_args"] = arguments
        edges.append(edge)
        return state_id

    root = visit(trace, ())
    return root, states, edges


def open_root(run_id: str, goal: str, *, input_artifact_hash: str, environment_hash: str):
    state_id = f"{_namespace(run_id)}/fs-1"
    return state_id, [{
        "state_id": state_id,
        "kind": "goal",
        "goal_text": goal,
        "exact_state_hash": object_ref({
            "run_id": run_id, "input_artifact_hash": input_artifact_hash,
            "environment_hash": environment_hash, "occurrence": [], "goal": goal,
        }),
        "semantic_signature": object_ref({"goal": " ".join(goal.split())}),
        "status": "open",
    }]


def build_result(
    record,
    *,
    termination: str,
    error: Optional[str],
    trace: Optional[dict[str, Any]],
    certificate_hash: Optional[str],
    details_hash: Optional[str],
    consumed_steps: int,
    consumed_exact: bool,
    attempt_cost: dict[str, Any],
    display_goal: str,
) -> dict[str, Any]:
    disposition = DISPOSITIONS[termination]
    input_hash = record.input_artifact_hash or object_ref({"run_id": record.run_id, "input": None})
    if termination == "certified" and trace is not None:
        root, states, edges = proof_graph(
            record.run_id, trace, input_artifact_hash=input_hash, environment_hash=record.environment_hash
        )
    else:
        root, states = open_root(
            record.run_id, display_goal, input_artifact_hash=input_hash,
            environment_hash=record.environment_hash,
        )
        edges = []
    request = record.request
    result: dict[str, Any] = {
        "protocol_version": 2,
        "run_id": record.run_id,
        "proof_id": request["proof_id"],
        "declaration_id": request["formal_declaration_id"],
        "environment_hash": record.environment_hash,
        "disposition": disposition,
        "termination": {"reason": termination, **({"detail": error} if error else {})},
        "root_state_id": root,
        "states": states,
        "tactic_edges": edges,
        "obstructions": [],
        "artifacts": [],
        "budget_used": {
            "tactic_steps": consumed_steps,
            "tactic_steps_exact": consumed_exact,
            "attempt": attempt_cost,
        },
        "budget": {
            "admitted_max_steps": record.admitted_max_steps,
            "remaining_steps": max(0, record.admitted_max_steps - consumed_steps),
            "admitted_wall_clock_ms": record.admitted_wall_clock_ms,
        },
        "input_kind": record.input_kind,
        "input_artifact_hash": record.input_artifact_hash,
        "source_artifact_hash": record.source_artifact_hash,
        "source_origin": record.source_origin,
        "backend": {"name": record.backend, "profile": record.profile,
                    "deployment_id": record.deployment_id,
                    "profile_config_hash": record.profile_config_hash},
        "request_digest": record.request_digest,
        "coordinator_binding": {
            key: request[key]
            for key in ("claim_id", "base_revision", "lease_id", "fencing_token")
        },
        "attempts": record.attempts,
    }
    if certificate_hash is not None:
        result["certificate"] = {
            "artifact_hash": certificate_hash,
            "status": "replay-pending",
            "producer_run_id": record.run_id,
        }
        result["artifacts"].append({"sha256": certificate_hash, "role": "certificate"})
    if record.input_artifact_hash:
        result["artifacts"].append({"sha256": record.input_artifact_hash, "role": "admitted-input"})
    if record.source_artifact_hash:
        result["artifacts"].append({
            "sha256": record.source_artifact_hash,
            "role": "source" if record.source_origin == "caller-artifact" else "generated-source",
        })
    if details_hash is not None:
        result["artifacts"].append({"sha256": details_hash, "role": "search-summary"})
    return result
