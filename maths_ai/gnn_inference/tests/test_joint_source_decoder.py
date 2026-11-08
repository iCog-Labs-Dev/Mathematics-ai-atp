"""Contract tests for the source-aware joint v2 argument decoder."""

import asyncio

import pytest
import torch

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import JointSourcePolicy
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import (
    CandidateSource,
    SourceCandidates,
    decode_action,
    goal_first_text,
    goal_state_text,
)
from maths_ai.hybrid_reasoner.pantograph_adapter import (
    DEFAULT_CAPABILITY_PROBES,
    all_capabilities_verified,
    capability_findings,
    probe_live_capabilities,
)

HIDDEN = 8
TEXT_DIM = 384
NODE_LABELS = 10
TACTICS = 6


def _policy(seed: int = 0) -> JointSourcePolicy:
    torch.manual_seed(seed)
    return JointSourcePolicy(
        num_node_labels=NODE_LABELS,
        num_tactics=TACTICS,
        num_node_types=4,
        hidden_dim=HIDDEN,
        num_layers=2,
        heads=2,
        dropout=0.0,
        use_node_type=True,
        max_args=3,
    )


def _candidates(
    *,
    local_ids=(0, 1),
    lemma_ids=(7,),
    local_scale=(2.0, 0.0),
    hidden: int = HIDDEN,
    text_dim: int = TEXT_DIM,
) -> SourceCandidates:
    vectors = torch.zeros(len(local_ids), hidden)
    for position, scale in enumerate(local_scale[: len(local_ids)]):
        vectors[position, 0] = scale
    return SourceCandidates(
        local_ids=tuple(local_ids),
        local_vectors=vectors,
        lemma_ids=tuple(lemma_ids),
        lemma_vectors=torch.zeros(len(lemma_ids), text_dim),
        lemma_scores=torch.tensor([0.9] * len(lemma_ids)),
    )


def _force_count(policy: JointSourcePolicy, count: int) -> None:
    torch.nn.init.zeros_(policy.arg_count_head.weight)
    torch.nn.init.zeros_(policy.arg_count_head.bias)
    with torch.no_grad():
        policy.arg_count_head.bias[count] = 1.0


def _force_source(policy: JointSourcePolicy, source: int) -> None:
    torch.nn.init.zeros_(policy.source_head.weight)
    torch.nn.init.zeros_(policy.source_head.bias)
    with torch.no_grad():
        policy.source_head.bias[source] = 3.0


def _force_local_query(policy: JointSourcePolicy) -> None:
    """Make the pointer query a constant positive unit direction.

    With ``query = e0`` the local score of a pool row is exactly its first
    coordinate, so ordering is deterministic regardless of the GRU state.
    """
    out_proj = policy.pointer.argument_selector.out_proj
    torch.nn.init.zeros_(out_proj.weight)
    torch.nn.init.zeros_(out_proj.bias)
    with torch.no_grad():
        out_proj.bias[0] = 1.0


def test_decode_appreciates_zero_argument_actions() -> None:
    policy = _policy()
    _force_count(policy, 0)
    action = decode_action(policy, torch.randn(HIDDEN), 1, _candidates())
    assert action.arguments == ()
    assert action.complete is True


def test_decode_resolves_local_identity_and_preserves_repeats() -> None:
    policy = _policy(seed=1)
    _force_count(policy, 2)
    _force_source(policy, 0)
    _force_local_query(policy)
    candidates = _candidates(local_ids=(0, 1, 2))
    action = decode_action(policy, torch.randn(HIDDEN), 2, candidates)
    assert len(action.arguments) == 2
    assert all(argument.source is CandidateSource.LOCAL for argument in action.arguments)
    # Position 0 dominates by construction; the second decode is independent,
    # so repeats of a high-scoring candidate are legitimate, not placeholders.
    assert action.arguments[0].stable_id == 0
    assert action.arguments[0].source_position == 0
    assert action.arguments[1].stable_id == 0
    assert action.complete is True


def test_decode_falls_back_to_library_without_locals() -> None:
    policy = _policy(seed=2)
    _force_count(policy, 1)
    candidates = _candidates(local_ids=())
    action = decode_action(policy, torch.randn(HIDDEN), 3, candidates)
    assert len(action.arguments) == 1
    assert action.arguments[0].source is CandidateSource.LIBRARY
    assert action.arguments[0].stable_id == 7


def test_decode_fails_closed_on_empty_pools() -> None:
    policy = _policy(seed=3)
    _force_count(policy, 1)
    candidates = _candidates(local_ids=(), lemma_ids=())
    action = decode_action(policy, torch.randn(HIDDEN), 0, candidates)
    assert action.complete is False


def test_source_candidates_reject_misaligned_pools() -> None:
    with pytest.raises(ValueError, match="align uniquely"):
        SourceCandidates(
            local_ids=(0, 5),
            local_vectors=torch.zeros(1, HIDDEN),
            lemma_ids=(7,),
            lemma_vectors=torch.zeros(1, TEXT_DIM),
            lemma_scores=torch.tensor([0.9]),
        )


def test_goal_state_text_and_goal_first_policy() -> None:
    goal = Goal(
        expression="n = n",
        goal_model_sexp="(sort Prop)",
        locals=[
            GoalLocal(
                user_name="n",
                context_index=0,
                binder_role="default",
                type_pp="Nat",
                type_model_sexp="(sort Type)",
            )
        ],
        model_sexp_version=1,
    )
    state_text = goal_state_text(goal)
    assert state_text == "n : Nat\n⊢ n = n"
    transformed = goal_first_text(state_text)
    assert transformed == "Goal: n = n\nContext: n : Nat"
    assert goal_first_text("⊢ p") == "Goal: p"


def _capability_payload(variables: list[dict]) -> dict:
    return {
        "name": "goal",
        "target": {"pp": "p", "modelSexp": "(sort Prop)", "modelSexpVersion": 1},
        "variables": variables,
    }


def _variable(**overrides: object) -> dict:
    field = {
        "userName": "h",
        "name": "h✝",
        "contextIndex": 0,
        "binderRole": "default",
        "isInstance": False,
        "isLet": False,
        "type": {"pp": "Nat", "modelSexp": "(sort Type)", "modelSexpVersion": 1},
        "value": None,
    }
    field.update(overrides)
    return field


def test_capability_findings_cover_binders_shadowing_lets_instances() -> None:
    findings = capability_findings(
        _capability_payload(
            [
                _variable(userName="a", name="a✝¹", contextIndex=0),
                _variable(userName="a", name="a✝²", contextIndex=1),
            ]
        )
    )
    assert findings["binders"] is True
    assert findings["shadowing"] is True
    assert findings["instances"] is True
    assert findings["local_lets"] is False

    findings = capability_findings(
        _capability_payload(
            [
                _variable(
                    userName="x",
                    name="x✝",
                    isLet=True,
                    value={"pp": "1", "modelSexp": "(lit 1)", "modelSexpVersion": 1},
                )
            ]
        )
    )
    assert findings["local_lets"] is True


def test_capability_findings_fail_closed_on_missing_fields() -> None:
    findings = capability_findings(
        _capability_payload(
            [_variable(contextIndex=None, isInstance=None, binderRole=None)]
        )
    )
    assert findings["binders"] is False
    assert findings["instances"] is False


class _FakeServer:
    """Minimal async stand-in mirroring the live client's payload contract."""

    def __init__(self, payloads: dict[str, dict]) -> None:
        self._payloads = payloads

    async def goal_start_async(self, expression: str) -> dict:
        payload = self._payloads.get(expression)
        if payload is None:
            raise RuntimeError(f"server could not start: {expression!r}")
        return payload


def _live_payloads() -> dict[str, dict]:
    double_binder = _capability_payload(
        [
            _variable(userName="a", name="a✝¹", contextIndex=0),
            _variable(userName="a", name="a✝²", contextIndex=1),
        ]
    )
    let_goal = _capability_payload(
        [
            _variable(
                userName="x",
                name="x✝",
                contextIndex=0,
                isLet=True,
                value={"pp": "1", "modelSexp": "(lit 1)", "modelSexpVersion": 1},
            )
        ]
    )
    let_goal["target"] = {
        "pp": "x = x",
        "modelSexp": "(app (lit =) (lit 1))",
        "modelSexpVersion": 1,
    }
    wrapped = {
        DEFAULT_CAPABILITY_PROBES[0].expression: double_binder,
        DEFAULT_CAPABILITY_PROBES[1].expression: double_binder,
        DEFAULT_CAPABILITY_PROBES[2].expression: let_goal,
        DEFAULT_CAPABILITY_PROBES[3].expression: double_binder,
    }
    return {expression: {"goals": [goal]} for expression, goal in wrapped.items()}


def test_probe_live_capabilities_reports_verified_state() -> None:
    server = _FakeServer(_live_payloads())
    results = asyncio.run(probe_live_capabilities(server))
    assert all_capabilities_verified(results) is True
    assert set(results) == {probe.name for probe in DEFAULT_CAPABILITY_PROBES}


def test_probe_live_capabilities_fails_closed_on_misbehaving_server() -> None:
    payloads = _live_payloads()
    let_variables = payloads[DEFAULT_CAPABILITY_PROBES[2].expression]["goals"][0]["variables"]
    let_variables[0]["value"] = None

    server = _FakeServer(payloads)
    results = asyncio.run(probe_live_capabilities(server))
    assert all_capabilities_verified(results) is False
    assert results["binders"].ok is True
    assert results["local_lets"].ok is False and results["local_lets"].detail
    # A crashing server for one probe must not break the whole report.
    assert results["shadowing"].ok is True
