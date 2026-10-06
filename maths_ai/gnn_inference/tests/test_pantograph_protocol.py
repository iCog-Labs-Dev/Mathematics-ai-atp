"""Pure final-contract tests; these do not launch Lean or import PyPantograph."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys

import pytest

from maths_ai.hybrid_reasoner.pantograph_protocol import (
    COMMAND_ERROR_CATEGORIES,
    PROTOCOL_DESCRIPTION,
    REQUIRED_OPTIONS,
    CommandAcknowledgement,
    CommandError,
    GoalStartResult,
    PantographError,
    PantographConfigurationError,
    PantographGoalRejected,
    PantographProtocolError,
    PantographRequestTimeout,
    PantographStateError,
    PantographStats,
    PantographTransportError,
    ProtocolDescription,
    SessionOptions,
    StartedGoal,
    StateHandle,
    TacticFailure,
    TacticParseFailure,
    TacticState,
    TacticSuccess,
    decode_execution_goal,
    decode_json_response,
    decode_messages,
    decode_response,
    encode_request,
    execution_goal_to_translator,
    execution_state_to_goals,
)


FIXTURES = Path(__file__).parent / "fixtures" / "pantograph"
BASELINE = json.loads((FIXTURES / "baseline" / "baseline.json").read_text())
CAPTURES = {entry["label"]: entry for entry in BASELINE["interactions"]}
FUTURE = json.loads((FIXTURES / "future_contract.json").read_text())


def success(name="identity_intro"):
    return deepcopy(FUTURE["successes"][name])


def test_protocol_import_is_independent_of_upstream_and_models():
    # A fresh interpreter makes this independent of pytest's import order.
    code = """
import importlib.abc
import sys
class ForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pantograph" or fullname.startswith("pantograph.") or fullname == "torch" or fullname.startswith("torch.") or fullname.endswith((".inference", ".training", ".joint_inference")):
            raise AssertionError("Forbidden dependency: " + fullname)
sys.meta_path.insert(0, ForbiddenImports())
from maths_ai.hybrid_reasoner.pantograph_protocol import decode_response
assert decode_response("goal.start", {"stateId": 0, "root": "g"}).state_id == 0
assert not any(name.startswith(("pantograph", "torch")) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        cwd=Path(__file__).resolve().parents[3],
    )


def test_lazy_exports_preserve_existing_package_api():
    from maths_ai.gnn_inference import atp_lean_gnn
    from maths_ai.gnn_inference.atp_lean_gnn.graph import GraphNode

    assert atp_lean_gnn.GraphNode is GraphNode
    assert set(atp_lean_gnn.__all__) == set(atp_lean_gnn._EXPORT_MODULES)
    with pytest.raises(AttributeError):
        getattr(atp_lean_gnn, "not_an_export")
    from maths_ai import hybrid_reasoner

    assert "HybridReasoner" in dir(hybrid_reasoner)
    with pytest.raises(AttributeError):
        getattr(hybrid_reasoner, "not_an_export")


@pytest.mark.parametrize(
    "raw",
    [
        '{"stateId":0,"stateId":1}',
        '{"x":{"a":1,"a":2}}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        "[]",
        "null",
        "ready.",
        "{",
        "{} {}",
        b"\xff",
    ],
)
def test_reject_malformed_json(raw):
    with pytest.raises(PantographProtocolError):
        decode_json_response(raw)


def test_request_framing_preserves_unicode_and_escaped_source():
    tactic = 'have h : α = α := by\n  trace "quoted"\n  rfl'
    raw = encode_request("goal.tactic", {"stateId": 0, "goalId": 2, "tactic": tactic})
    assert raw.count(b"\n") == 1 and raw.endswith(b"\n")
    assert raw.startswith(b'goal.tactic {"stateId":0,"goalId":2,')
    assert "α".encode() in raw
    assert json.loads(raw.decode().split(" ", 1)[1])["tactic"] == tactic
    assert decode_json_response('{"root":"α","stateId":0}\n')["root"] == "α"


@pytest.mark.parametrize(
    "command,payload",
    [
        ("goal.tactic", {"stateId": 0, "tactic": "skip"}),
        ("goal.tactic", {"stateId": True, "goalId": 0, "tactic": "skip"}),
        ("goal.tactic", {"stateId": 0, "goalId": -1, "tactic": "skip"}),
        ("goal.tactic", {"stateId": 0, "goalId": False, "tactic": "skip"}),
        ("goal.tactic", {"stateId": 0, "goalId": 0, "tactic": None}),
        ("goal.start", {"expr": "True", "copyFrom": "Nat.add_comm"}),
        ("goal.start", {"expr": ""}),
        ("goal.delete", {"stateIds": [False]}),
        ("stat", {"unused": 1}),
        ("options.set", {"printExprAST": 1}),
    ],
)
def test_request_shape_is_strict(command, payload):
    with pytest.raises(PantographProtocolError):
        encode_request(command, payload)


def test_request_options_and_deletions():
    for key, value in REQUIRED_OPTIONS.items():
        with pytest.raises(PantographConfigurationError):
            encode_request("options.set", {key: not value})
    with pytest.raises(PantographStateError):
        encode_request("goal.delete", {"stateIds": [0, 0]})
    with pytest.raises(PantographConfigurationError):
        encode_request("goal.print", {})
    assert (
        encode_request("goal.delete", {"stateIds": [0, 1]})
        == b'goal.delete {"stateIds":[0,1]}\n'
    )


@pytest.mark.parametrize("state_id", [0, 17])
def test_goal_start_does_not_fabricate_solved_goals(state_id):
    response = decode_response("goal.start", {"stateId": state_id, "root": "g"})
    assert response == GoalStartResult(state_id, "g")
    state = StartedGoal(
        StateHandle("session", "scope", response.state_id), response.root
    )
    assert not hasattr(state, "goals")
    with pytest.raises(FrozenInstanceError):
        state.root = "changed"


@pytest.mark.parametrize("value", [True, False, -1, 0.0, "0", None])
@pytest.mark.parametrize(
    "command,field",
    [("goal.start", "stateId"), ("goal.tactic", "nextStateId"), ("stat", "nGoals")],
)
def test_integer_fields_reject_booleans_and_invalid_ids(value, command, field):
    payload = (
        {"stateId": 0, "root": "g"}
        if command == "goal.start"
        else success() if command == "goal.tactic" else {"nGoals": 0}
    )
    payload[field] = value
    with pytest.raises(PantographProtocolError) as exc:
        decode_response(command, payload)
    assert exc.value.command == command
    assert exc.value.payload == payload
    assert exc.value.__cause__ is not None


def test_all_response_variants():
    assert isinstance(
        decode_response("protocol.describe", FUTURE["descriptor"]), ProtocolDescription
    )
    assert isinstance(decode_response("goal.tactic", success()), TacticSuccess)
    assert isinstance(
        decode_response("goal.tactic", FUTURE["parse_failure"]), TacticParseFailure
    )
    assert isinstance(
        decode_response("goal.tactic", FUTURE["tactic_failure"]), TacticFailure
    )
    assert isinstance(
        decode_response("goal.tactic", FUTURE["logged_error_failure"]), TacticFailure
    )
    assert decode_response("goal.delete", {}) == CommandAcknowledgement("goal.delete")
    assert decode_response("options.set", {}) == CommandAcknowledgement("options.set")
    assert decode_response("stat", {"nGoals": 0}) == PantographStats(0)
    options = decode_response("options.print", CAPTURES["options_print"]["response"])
    assert isinstance(options, SessionOptions)
    assert options.to_dict() == CAPTURES["options_print"]["response"]


@pytest.mark.parametrize("key", list(PROTOCOL_DESCRIPTION))
def test_descriptor_versions_and_identity_are_exact(key):
    descriptor = deepcopy(FUTURE["descriptor"])
    del descriptor[key]
    with pytest.raises(PantographProtocolError):
        decode_response("protocol.describe", descriptor)
    descriptor[key] = True if type(PROTOCOL_DESCRIPTION[key]) is int else "incompatible"
    with pytest.raises(PantographProtocolError):
        decode_response("protocol.describe", descriptor)


@pytest.mark.parametrize("command,categories", list(COMMAND_ERROR_CATEGORIES.items()))
def test_error_categories_are_known_per_command(command, categories):
    for category in categories:
        response = decode_response(command, {"error": category, "desc": "detail"})
        assert isinstance(response, CommandError)
        exc = response.as_exception()
        assert isinstance(exc, PantographError)
        assert exc.command == command and exc.payload == {
            "error": category,
            "desc": "detail",
        }
    with pytest.raises(PantographProtocolError):
        decode_response(command, {"error": "unknown", "desc": "detail"})


@pytest.mark.parametrize(
    "label",
    [
        "goal_parse_error",
        "goal_elab_error",
        "invalid_state",
        "invalid_goal",
        "missing_goal_id",
        "unsupported_descriptor",
    ],
)
def test_captured_error_mapping(label):
    capture = CAPTURES[label]
    response = decode_response(capture["command"], capture["response"])
    expected = (
        PantographGoalRejected
        if label.startswith("goal_")
        else (
            PantographStateError
            if label.startswith("invalid_")
            else PantographProtocolError
        )
    )
    assert isinstance(response.as_exception(), expected)


def test_timeout_is_distinct_from_domain_failure():
    assert issubclass(PantographRequestTimeout, PantographTransportError)
    assert not issubclass(PantographTransportError, PantographGoalRejected)


@pytest.mark.parametrize(
    "label",
    [
        entry["label"]
        for entry in BASELINE["interactions"]
        if entry["command"] == "goal.tactic" and "error" not in entry["response"]
    ],
)
def test_every_baseline_tactic_response_is_rejected(label):
    # Even valid old successes lack required evidence. Failures lack diagnostics.
    with pytest.raises(PantographProtocolError):
        decode_response("goal.tactic", CAPTURES[label]["response"])


@pytest.mark.parametrize(
    "change",
    [
        {"parseError": "failure"},
        {"tacticErrors": ["failure"]},
        {"error": "index", "desc": "bad"},
        {"goals": None},
        {"goals": {}},
        {"messages": None},
        {"validation": None},
        {"fragment": "tactic"},
        {"hasSorry": False},
    ],
)
def test_mixed_and_unknown_tactic_variants_are_rejected(change):
    payload = success()
    payload.update(change)
    with pytest.raises(PantographProtocolError):
        decode_response("goal.tactic", payload)


@pytest.mark.parametrize(
    "field", ["version", "scope", "checked", "hasSorry", "hasUnsafe"]
)
def test_validation_fields_are_required(field):
    payload = success()
    del payload["validation"][field]
    with pytest.raises(PantographProtocolError):
        decode_response("goal.tactic", payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("scope", "root-proof"),
        ("checked", False),
        ("checked", 1),
        ("hasSorry", None),
        ("hasUnsafe", 0),
    ],
)
def test_validation_evidence_is_strict(field, value):
    payload = success()
    payload["validation"][field] = value
    with pytest.raises(PantographProtocolError):
        decode_response("goal.tactic", payload)


@pytest.mark.parametrize(
    "name,flag", [("admission", "has_sorry"), ("unsafe", "has_unsafe")]
)
def test_positive_evidence_retains_allocation_for_later_release(name, flag):
    result = decode_response("goal.tactic", success(name))
    assert isinstance(result, TacticSuccess)
    assert getattr(result.validation, flag) is True
    assert result.validation.is_rejection
    assert result.next_state_id == FUTURE["successes"][name]["nextStateId"]


def test_message_positions_and_warning_preservation():
    result = decode_response("goal.tactic", FUTURE["warning_success"])
    message = result.messages[0]
    assert (message.pos.line, message.pos.column) == (2, 0)
    assert (message.end_pos.line, message.end_pos.column) == (2, 4)
    assert message.file_name == "<Pantograph>" and message.kind == "example"
    assert message.severity == "warning"
    payload = success()
    payload["messages"] = [{"severity": "error", "data": "logged error"}]
    with pytest.raises(PantographProtocolError):
        decode_response("goal.tactic", payload)
    assert decode_messages([{"severity": "information", "data": "info"}])[0].pos is None


@pytest.mark.parametrize(
    "message",
    [
        {"severity": "info", "data": "bad"},
        {"severity": "warning"},
        {"severity": "error", "data": 0},
        {"severity": "warning", "data": "bad", "pos": None},
        {"severity": "warning", "data": "bad", "pos": {"line": 0, "column": 0}},
        {"severity": "warning", "data": "bad", "pos": {"line": True, "column": 0}},
        {"severity": "warning", "data": "bad", "endPos": {"line": 1, "column": 0}},
        {
            "severity": "warning",
            "data": "bad",
            "pos": {"line": 2, "column": 0},
            "endPos": {"line": 1, "column": 0},
        },
    ],
)
def test_message_schema_is_strict(message):
    with pytest.raises(PantographProtocolError):
        decode_messages([message])


@pytest.mark.parametrize("variant", ["parseError", "tacticErrors"])
@pytest.mark.parametrize("bad", [None, False, 0, {}, [], ""])
def test_failure_payloads_are_not_empty_or_wrong_type(variant, bad):
    payload = {variant: bad, "messages": []}
    # Nonempty strings only for parseError; nonempty string arrays for tacticErrors.
    with pytest.raises(PantographProtocolError):
        decode_response("goal.tactic", payload)


def test_goals_keep_order_case_tags_and_conversion_metadata():
    result = decode_response("goal.tactic", success("conjunction_split"))
    wire = FUTURE["successes"]["conjunction_split"]["goals"]
    assert [goal.goal_name for goal in result.goals] == [goal["name"] for goal in wire]
    assert [goal.case_tag for goal in result.goals] == ["left", "right"]
    goal_payload = deepcopy(wire[0])
    goal_payload["isConversion"] = True
    decoded = decode_execution_goal(goal_payload)
    assert decoded.is_conversion is True
    assert decoded.goal.case_tag == "left"
    assert "left" not in decoded.goal.hypotheses


def test_locals_preserve_instances_lets_raw_views_and_fresh_canonical_copies():
    instance = decode_response("goal.tactic", success("instance_intro")).goals[0]
    assert [local.context_index for local in instance.locals] == [0, 1, 2]
    assert instance.locals[1].is_instance is True
    assert instance.locals[1].binder_role == ":instance-implicit"
    let_goal = decode_response("goal.tactic", success("let_intro")).goals[0]
    assert let_goal.locals[1].is_let is True
    assert let_goal.locals[1].value.pp == "n"
    assert let_goal.locals[1].value.model_sexp == "(:fv FV0)"
    projection = execution_goal_to_translator(let_goal)
    assert projection["target"]["sexp"] == let_goal.target.sexp
    assert projection["target"]["sexp"] != projection["target"]["model_sexp"]
    copied = let_goal.goal
    copied.expression = "changed"
    copied.locals[0].user_name = "changed"
    assert let_goal.goal.expression != "changed"
    assert let_goal.goal.locals[0].user_name == "n"
    projection["locals"][0]["type"]["pp"] = "changed"
    assert let_goal.locals[0].type.pp == "ℕ"


def test_context_gaps_are_not_renumbered():
    payload = success("instance_intro")["goals"][0]
    del payload["vars"][1]
    goal = decode_execution_goal(payload)
    assert [local.context_index for local in goal.locals] == [0, 2]
    assert "FV2" in goal.goal.goal_model_sexp


@pytest.mark.parametrize("role", [":explicit", ":implicit", ":strict-implicit"])
def test_supported_declaration_binder_roles(role):
    payload = success()["goals"][0]
    payload["vars"][0]["binderRole"] = role
    assert decode_execution_goal(payload).locals[0].binder_role == role


@pytest.mark.parametrize(
    "field,value",
    [
        ("contextIndex", True),
        ("contextIndex", -1),
        ("contextIndex", 0.5),
        ("binderRole", ":instImplicit"),
        ("isInstance", "false"),
        ("isLet", 0),
        ("isInaccessible", None),
        ("type", {"pp": "Prop"}),
        ("value", {"pp": "unexpected"}),
    ],
)
def test_local_schema_is_strict(field, value):
    payload = success()["goals"][0]
    payload["vars"][0][field] = value
    with pytest.raises(PantographProtocolError):
        decode_execution_goal(payload)


def test_missing_let_value_and_metadata_are_rejected():
    for field in ("value", "isLet", "contextIndex", "binderRole", "isInstance"):
        payload = success("let_intro")["goals"][0]
        del payload["vars"][1][field]
        with pytest.raises(PantographProtocolError):
            decode_execution_goal(payload)


@pytest.mark.parametrize("indices", [[0, 0], [1, 0]])
def test_duplicate_and_reordered_context_indices_are_rejected(indices):
    payload = success()["goals"][0]
    for local, index in zip(payload["vars"], indices):
        local["contextIndex"] = index
    with pytest.raises(PantographProtocolError):
        decode_execution_goal(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("modelSexpVersion", True),
        ("modelSexpVersion", 2),
        ("modelSexp", ""),
        ("sexp", None),
        ("pp", None),
    ],
)
def test_expression_views_and_versions_are_required(field, value):
    for location in ("target", "type", "value"):
        payload = success("let_intro")["goals"][0]
        expr = (
            payload["target"] if location == "target" else payload["vars"][1][location]
        )
        expr[field] = value
        with pytest.raises(PantographProtocolError):
            decode_execution_goal(payload)


def test_only_known_optional_expression_diagnostics_are_allowed():
    payload = success()["goals"][0]
    payload["target"]["dependentMVars"] = ["g2"]
    assert decode_execution_goal(payload).target.dependent_mvars == ("g2",)
    payload["target"]["inventedDiagnostic"] = []
    with pytest.raises(PantographProtocolError):
        decode_execution_goal(payload)


def test_immutable_containers_and_state_conversion():
    result = decode_response("goal.tactic", success())
    state = TacticState(
        StateHandle("session", "scope", result.next_state_id),
        result.goals,
        result.validation,
        result.messages,
    )
    with pytest.raises(FrozenInstanceError):
        result.next_state_id = 5
    with pytest.raises(FrozenInstanceError):
        result.goals[0].target.pp = "changed"
    assert (
        execution_state_to_goals(state)[0].model_dump()
        == state.goals[0].goal.model_dump()
    )
    for bad in (True, -1, "0"):
        with pytest.raises(PantographStateError):
            StateHandle("session", "scope", bad)


@pytest.mark.parametrize(
    "label", ["identity_intro", "instance_intro", "let_intro", "conjunction_split"]
)
def test_exact_canonical_fingerprints_and_graph_features(label):
    from maths_ai.gnn_inference.atp_lean_gnn.graph import (
        dag_fingerprint,
        model_goal_to_dag,
    )

    expected = json.loads((FIXTURES / "canonical_graph_expectations.json").read_text())[
        "goals"
    ][label]
    result = decode_response("goal.tactic", success(label))
    assert len(result.goals) == len(expected)
    for execution, golden in zip(result.goals, expected):
        goal = execution.goal
        assert goal.model_dump() == golden["canonical"]
        assert goal.state_fingerprint() == golden["state_fingerprint"]
        dag = model_goal_to_dag(goal)
        assert dag_fingerprint(dag) == golden["dag_fingerprint"]
        assert [node.as_dict() for node in dag.nodes] == golden["nodes"]
        assert [list(edge) for edge in dag.edges] == golden["edges"]
        assert dag.expression_root_id == golden["expression_root_id"]
        assert dag.state_root_id == golden["state_root_id"]


def test_baseline_evidence_is_not_final_contract_evidence():
    assert BASELINE["kind"] == "unmodified-custom-repl-baseline"
    assert FUTURE["kind"] == "authored-future-contract"
    assert BASELINE["readiness_line"] == "ready."
    for entry in BASELINE["interactions"]:
        assert json.loads(entry["response_line"]) == entry["response"]
    for label in ("admission_unchecked", "logged_error_unchecked"):
        assert CAPTURES[label]["response"]["goals"] == []
        assert "validation" not in CAPTURES[label]["response"]
    assert CAPTURES["focused_all_goals"]["response"]["goals"]
    assert CAPTURES["identity_after_parent_delete"]["response"]["goals"] == []
    assert CAPTURES["identity_alternative"]["response"]["goals"] == []
    assert (
        CAPTURES["initial_stat"]["response"]
        == CAPTURES["final_stat"]["response"]
        == {"nGoals": 0}
    )


def test_frozen_contract_descriptor_matches_the_codec():
    contract = json.loads(
        (FIXTURES.parents[2] / "pantograph_backport" / "contract.json").read_text()
    )
    assert contract["descriptor"] == dict(PROTOCOL_DESCRIPTION)
    assert contract["required_options"] == dict(REQUIRED_OPTIONS)
    assert {name: set(codes) for name, codes in contract["commands"].items()} == dict(
        COMMAND_ERROR_CATEGORIES
    )
    assert contract["complete_theorem_certification"] is False


def test_goal_names_and_omitted_case_names_are_not_interchangeable():
    payload = success()
    assert "userName" not in payload["goals"][0]
    goal = decode_response("goal.tactic", payload).goals[0]
    assert goal.case_tag is None and goal.goal.case_tag is None
    assert goal.goal_name != ""
    payload["goals"].append(deepcopy(payload["goals"][0]))
    with pytest.raises(PantographProtocolError):
        decode_response("goal.tactic", payload)


def test_errors_keep_independent_payload_snapshots():
    payload = success()
    payload["validation"]["checked"] = False
    with pytest.raises(PantographProtocolError) as failure:
        decode_response("goal.tactic", payload)
    payload["validation"]["checked"] = True
    assert failure.value.payload["validation"]["checked"] is False
    copied = failure.value.payload
    copied["validation"]["checked"] = True
    assert failure.value.payload["validation"]["checked"] is False


def test_capture_utility_refuses_overwriting_evidence(tmp_path):
    script = FIXTURES.parents[2] / "scripts" / "capture_pantograph_fixtures.py"
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
