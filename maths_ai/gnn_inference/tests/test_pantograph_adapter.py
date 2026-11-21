"""Canonical goal conversion over the standalone client's execution types."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import (
    MODEL_SEXPR_GRAPH_SPEC,
    GraphRepresentationSpec,
)
from maths_ai.hybrid_reasoner.pantograph_adapter import (
    execution_goal_to_goal,
    execution_goal_to_translator,
    execution_local_to_goal_local,
    execution_state_to_goals,
    require_client_compatibility,
)
from pantograph_async import (
    MODEL_SEXP_VERSION,
    PantographConfigurationError,
    StateHandle,
    TacticState,
    decode_execution_goal,
    decode_response,
)


FIXTURES = Path(__file__).parent / "fixtures" / "pantograph"
FUTURE = json.loads((FIXTURES / "future_contract.json").read_text())
UPSTREAM = json.loads((FIXTURES / "upstream_pr1" / "capture.json").read_text())
UPSTREAM_CAPTURES = {entry["label"]: entry for entry in UPSTREAM["interactions"]}


def success(name="identity_intro"):
    return deepcopy(FUTURE["successes"][name])


def test_client_import_does_not_load_application_models():
    code = """
import sys
import pantograph_async
assert not any(name.split(".")[0] == "maths_ai" for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


def test_graph_representation_matches_the_client_wire_version():
    require_client_compatibility()
    assert MODEL_SEXPR_GRAPH_SPEC.model_sexp_version == MODEL_SEXP_VERSION
    incompatible = GraphRepresentationSpec(
        representation=MODEL_SEXPR_GRAPH_SPEC.representation,
        model_sexp_version=MODEL_SEXP_VERSION + 1,
        graph_schema_version=MODEL_SEXPR_GRAPH_SPEC.graph_schema_version,
        hypothesis_schema=MODEL_SEXPR_GRAPH_SPEC.hypothesis_schema,
    )
    with pytest.raises(PantographConfigurationError):
        require_client_compatibility(incompatible)


def test_case_tags_are_not_hypotheses():
    result = decode_response("goal.tactic", success("conjunction_split"))
    goals = execution_state_to_goals(result)
    assert [goal.case_tag for goal in goals] == ["left", "right"]
    assert "left" not in goals[0].hypotheses
    payload = deepcopy(FUTURE["successes"]["conjunction_split"]["goals"][0])
    payload.update(isConversion=True, fragment="conv")
    assert execution_goal_to_goal(decode_execution_goal(payload)).case_tag == "left"


def test_canonical_copies_are_fresh_and_translator_projection_is_independent():
    let_goal = decode_response("goal.tactic", success("let_intro")).goals[0]
    projection = execution_goal_to_translator(let_goal)
    assert projection["fragment"] == let_goal.fragment == "tactic"
    assert projection["target"]["sexp"] == let_goal.target.sexp
    assert projection["target"]["sexp"] != projection["target"]["model_sexp"]
    copied = execution_goal_to_goal(let_goal)
    copied.expression = "changed"
    copied.locals[0].user_name = "changed"
    fresh = execution_goal_to_goal(let_goal)
    assert fresh.expression != "changed"
    assert fresh.locals[0].user_name == "n"
    projection["locals"][0]["type"]["pp"] = "changed"
    assert let_goal.locals[0].type.pp == "ℕ"


def test_locals_keep_let_values_and_context_gaps():
    let_goal = decode_response("goal.tactic", success("let_intro")).goals[0]
    local = execution_local_to_goal_local(let_goal.locals[1])
    assert local.is_let and local.value_model_sexp == "(:fv FV0)"
    payload = success("instance_intro")["goals"][0]
    del payload["vars"][1]
    goal = execution_goal_to_goal(decode_execution_goal(payload))
    assert [item.context_index for item in goal.locals] == [0, 2]
    assert "FV2" in goal.goal_model_sexp


def test_state_conversion_matches_per_goal_conversion():
    result = decode_response("goal.tactic", success())
    state = TacticState(
        StateHandle("session", "scope", result.next_state_id),
        result.goals,
        result.validation,
        result.messages,
        result.transition,
    )
    assert [goal.model_dump() for goal in execution_state_to_goals(state)] == [
        execution_goal_to_goal(goal).model_dump() for goal in state.goals
    ]


def test_omitted_case_name_stays_none():
    goal = decode_response("goal.tactic", success()).goals[0]
    assert execution_goal_to_goal(goal).case_tag is None


@pytest.mark.parametrize("fragment", ["tactic", "conv", "calc"])
def test_fragment_is_execution_metadata_not_a_graph_feature(fragment):
    payload = success()["goals"][0]
    expected = execution_goal_to_goal(decode_execution_goal(payload)).model_dump()
    payload.update(fragment=fragment, isConversion=(fragment == "conv"))
    goal = decode_execution_goal(payload)
    assert execution_goal_to_translator(goal)["fragment"] == fragment
    assert execution_goal_to_goal(goal).model_dump() == expected


@pytest.mark.parametrize(
    "label", ["identity_intro", "instance_intro", "let_intro", "conjunction_split"]
)
@pytest.mark.parametrize("source", ["authored", "upstream"])
def test_exact_canonical_fingerprints_and_graph_features(label, source):
    from maths_ai.gnn_inference.atp_lean_gnn.graph import (
        dag_fingerprint,
        model_goal_to_dag,
    )

    expected = json.loads((FIXTURES / "canonical_graph_expectations.json").read_text())[
        "goals"
    ][label]
    goals = (
        decode_response("goal.tactic", success(label)).goals
        if source == "authored"
        else tuple(
            decode_execution_goal(goal)
            for goal in UPSTREAM_CAPTURES[label]["response"]["goals"]
        )
    )
    assert len(goals) == len(expected)
    for execution, golden in zip(goals, expected):
        goal = execution_goal_to_goal(execution)
        # Raw expressions and live identity are execution metadata outside the
        # canonical model view; a fresh REPL goal has no separate live identity.
        live_fields = {"type_sexp", "value_sexp", "execution_name", "execution_index"}
        canonical = goal.model_dump(
            exclude={"goal_sexp": True, "locals": {"__all__": live_fields}}
        )
        assert all(
            local.execution_name is None and local.execution_index is None
            for local in goal.locals
        )
        assert canonical == golden["canonical"]
        assert goal.goal_sexp == execution.target.sexp
        assert [(local.type_sexp, local.value_sexp) for local in goal.locals] == [
            (local.type.sexp, local.value.sexp if local.value else None)
            for local in execution.locals
        ]
        assert goal.state_fingerprint() == golden["state_fingerprint"]
        dag = model_goal_to_dag(goal)
        assert dag_fingerprint(dag) == golden["dag_fingerprint"]
        assert [node.as_dict() for node in dag.nodes] == golden["nodes"]
        assert [list(edge) for edge in dag.edges] == golden["edges"]
        assert dag.expression_root_id == golden["expression_root_id"]
        assert dag.state_root_id == golden["state_root_id"]


def test_capture_utility_refuses_overwriting_evidence(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "capture_pantograph_fixtures.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-root",
            "/does/not/exist",
            "--repl-source",
            "/does/not/exist",
            "--repl",
            "/does/not/exist",
            "--verified-base",
            "/does/not/exist",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "existing evidence is never overwritten" in result.stderr
    assert list(tmp_path.iterdir()) == []
