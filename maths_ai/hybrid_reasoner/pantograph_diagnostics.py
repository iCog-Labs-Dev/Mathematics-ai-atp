"""CPU-only live verification of this repository's graph contract over the client.

Protocol checks run inside the standalone client. This module adds the
application side: canonical goal conversion, DAG goldens, and replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from pantograph_async import PantographClient, PantographEnv, PantographProtocolError
from pantograph_async.diagnostics import verify_live_client
from pantograph_async.probes import require

from .pantograph_adapter import execution_goal_to_goal, require_client_compatibility
from .pantograph_model_sexpr import materialize_seed


GOLDEN_PATH = (
    Path(__file__).parents[1]
    / "gnn_inference/tests/fixtures/pantograph/canonical_graph_expectations.json"
)
GRAPH_CASES = (
    ("identity_intro", "∀ (p : Prop), p → p", "intro p h"),
    ("conjunction_split", "True ∧ True", "constructor"),
    (
        "instance_intro",
        "∀ (α : Type) [inst : Inhabited α] (x : α), x = x",
        "intro α inst x",
    ),
    ("let_intro", "∀ (n : Nat), let x := n; x = x", "intro n x"),
)


# Compiles the on-demand splitter of `Nat.pow.match_1`, continues the retained
# state, and closes it. Validation version 2 ended the session at the unfold.
COMPILER_CONTROL = (
    "∀ m n : Nat, Nat.pow m n = Nat.pow m n",
    ("intro m n", "unfold Nat.pow", "split <;> rfl"),
)


async def certify_compiler_control(scope):
    """Run the compiler-producing control and return its kernel receipt."""
    expression, tactics = COMPILER_CONTROL
    state = await scope.start_goal(expression, levels=())
    for tactic in tactics:
        state = await scope.apply_tactic(state, tactic)
    require(not state.goals, "Compiler-producing control left goals open.")
    certificate = await scope.certify(state)
    require(certificate.is_certified, f"Compiler-producing control did not certify: {certificate.reason}")
    auxiliary = certificate.receipt.auxiliary
    require(
        not any("_cstage" in name for name in auxiliary),
        "A compiler-only entry reached the kernel certificate.",
    )
    return certificate.receipt


async def verify_canonical_graphs(client: PantographClient) -> list[str]:
    """Live goals must reproduce the canonical states, DAGs, and replays."""
    from maths_ai.gnn_inference.atp_lean_gnn.graph import (
        dag_fingerprint,
        model_goal_to_dag,
    )

    outcomes = []
    goldens = json.loads(GOLDEN_PATH.read_text())["goals"]
    async with client.state_scope() as scope:
        for label, expression, tactic in GRAPH_CASES:
            root = await scope.start_goal(expression, levels=())
            state = await scope.apply_tactic(root, tactic)
            require(
                len(state.goals) == len(goldens[label]),
                f"Goal count changed: {label}",
            )
            for execution_goal, expected in zip(state.goals, goldens[label]):
                goal = execution_goal_to_goal(execution_goal)
                require(
                    goal.state_fingerprint() == expected["state_fingerprint"],
                    f"Canonical model state changed: {label}",
                )
                dag = model_goal_to_dag(goal)
                require(
                    dag_fingerprint(dag) == expected["dag_fingerprint"],
                    f"DAG changed: {label}",
                )
                require(
                    [node.as_dict() for node in dag.nodes] == expected["nodes"],
                    f"Features changed: {label}",
                )
                require(
                    [list(edge) for edge in dag.edges] == expected["edges"],
                    f"Edges changed: {label}",
                )
                require(
                    dag.expression_root_id == expected["expression_root_id"]
                    and dag.state_root_id == expected["state_root_id"],
                    f"Graph roots changed: {label}",
                )
                _, replayed, _ = await materialize_seed(scope, goal.replay_seed())
                require(
                    replayed.state_fingerprint() == goal.state_fingerprint(),
                    f"Canonical replay round trip changed: {label}",
                )
            outcomes.append(f"canonical-graph-{label}")
    async with client.state_scope() as scope:
        for expression, intro, closing in (
            ("let x : Nat := 1; x = x", "intro x", "rfl"),
            (
                "∀ (n : Nat), let x := n; ∀ (h : x = n), x = n",
                "intro n x h",
                "exact h",
            ),
        ):
            start = await scope.start_goal(expression, levels=())
            state = await scope.apply_tactic(start, intro)
            goal = execution_goal_to_goal(state.goals[0])
            require(
                any(local.is_let for local in goal.locals),
                "Local-let regression lost its definition.",
            )
            replayed, replayed_goal, _ = await materialize_seed(
                scope, goal.replay_seed()
            )
            require(
                replayed_goal.state_fingerprint() == goal.state_fingerprint(),
                "Dependent local-let replay changed the model state.",
            )
            require(
                not (await scope.apply_tactic(replayed, closing)).goals,
                "Known local-let closing tactic failed after replay.",
            )
    outcomes.append("constant-and-dependent-let-replay")
    async with client.state_scope() as scope:
        await certify_compiler_control(scope)
    outcomes.append("compiler-producing-continuation-certified")
    return outcomes


async def verify_live_application(env: PantographEnv, *, proofs: int = 100) -> dict:
    """Run the client's protocol gate, then this repository's graph gate."""
    require_client_compatibility()
    report = await verify_live_client(env, proofs=proofs)
    client = await PantographClient.create(env)
    try:
        baseline = (await client.stat()).n_goals
        checks = await verify_canonical_graphs(client)
        final = (await client.stat()).n_goals
        if final != baseline:
            raise PantographProtocolError("Application graph checks leaked states.")
    finally:
        await client.aclose()
    report["checks"] = [*report["checks"], *checks]
    report["application"] = {
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parents[2],
            text=True,
            timeout=5,
        ).strip(),
        "python_sources_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "pantograph_adapter.py",
                "pantograph_model_sexpr.py",
                "pantograph_diagnostics.py",
            )
        },
        "initial_states": baseline,
        "final_states": final,
        "phase": client.phase.value,
        "subprocess_reaped": client.phase.value == "CLOSED",
    }
    return report
