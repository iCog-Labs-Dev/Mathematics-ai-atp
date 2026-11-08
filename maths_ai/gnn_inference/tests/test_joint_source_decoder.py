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
    """Minimal async stand-in mirroring the live client's raw payload contract.

    Payloads are keyed by probe name; ``goal.start`` hands back a fresh
    stateId and ``goal.tactic`` replays return the recorded raw payload.
    """

    def __init__(self, payloads: dict[str, dict], *, expose_run_async: bool = True) -> None:
        self._payloads = payloads
        self._current_expression: str | None = None
        self._replay_calls = 0
        self.run_async = self._run_async if expose_run_async else None

    async def _run_async(self, cmd: str, payload: dict) -> dict:
        if cmd == "goal.start":
            self._current_expression = payload.get("expr")
            self._replay_calls = 0
            return {"stateId": 0, "root": "_fake"}
        if cmd == "goal.tactic":
            self._replay_calls += 1
            # Distinguish probes sharing an expression by their replay length;
            # the payload is delivered on the probe's final replay step.
            candidates = {
                probe.name: probe
                for probe in DEFAULT_CAPABILITY_PROBES
                if probe.expression == self._current_expression
            }
            for name, probe in candidates.items():
                if len(probe.replay) == self._replay_calls:
                    result = self._payloads.get(name)
                    if result is None:
                        raise RuntimeError(f"server could not replay: {self._current_expression!r}")
                    return result
            if any(len(p.replay) > self._replay_calls for p in candidates.values()):
                # Intermediate replay step: no goals yet, just advance state.
                return {"nextStateId": 1}
            raise RuntimeError(
                f"server could not replay step {self._replay_calls} of: {self._current_expression!r}"
            )
        raise RuntimeError(f"unexpected command: {cmd}")


def _live_payloads() -> dict[str, dict]:
    """Payloads captured live from the pinned Pantograph fork repl.

    Each is the exact raw ``goal.tactic`` wire payload after the probe's
    replay tactics with ``printExprAST``/``printExprModelAST`` enabled.
    """
    # binders/shadowing/instances-replays: two binder intros on
    # ∀ (a : Nat), ∀ (a : Nat), a = a — Pantograph renames the shadowed
    # binder to 'a✝' (inaccessible) at the second intro.
    shadow_binder_goal = {
        "name": "_uniq.20",
        "target": {
            "sexp": "((:c Eq) (:c Nat) (:fv _uniq.19) (:fv _uniq.19))",
            "pp": "a = a",
            "modelSexpVersion": 1,
            "modelSexp": "(:app (:c Eq) (:arg :implicit-type 0 (:c Nat)) (:arg :explicit 1 (:fv FV1)) (:arg :explicit 2 (:fv FV1)))",
        },
        "vars": [
            {
                "userName": "a✝",
                "name": "_uniq.11",
                "isLet": False,
                "isInstance": False,
                "isInaccessible": True,
                "contextIndex": 0,
                "binderRole": ":explicit",
                "type": {"sexp": "(:c Nat)", "pp": "Nat", "modelSexpVersion": 1, "modelSexp": "(:c Nat)"},
            },
            {
                "userName": "a",
                "name": "_uniq.19",
                "isLet": False,
                "isInstance": False,
                "isInaccessible": False,
                "contextIndex": 1,
                "binderRole": ":explicit",
                "type": {"sexp": "(:c Nat)", "pp": "Nat", "modelSexpVersion": 1, "modelSexp": "(:c Nat)"},
            },
        ],
        "isConversion": False,
        "fragment": "tactic",
    }
    let_goal = {
        "name": "_uniq.44",
        "target": shadow_binder_goal["target"],
        "vars": [
            {
                "userName": "x",
                "name": "_uniq.43",
                "isLet": True,
                "isInstance": False,
                "isInaccessible": False,
                "contextIndex": 0,
                "binderRole": ":let",
                "type": {"sexp": "(:c Nat)", "pp": "Nat", "modelSexpVersion": 1, "modelSexp": "(:c Nat)"},
                "value": {
                    "sexp": "((:c OfNat.ofNat) (:c Nat) (:lit 2) ((:c instOfNatNat) (:lit 2)))",
                    "pp": "2",
                    "modelSexpVersion": 1,
                    "modelSexp": "(:app (:c OfNat.ofNat) (:arg :implicit-type 0 (:c Nat)) (:arg :explicit 1 (:lit 2)) (:arg :instance 2 (:instance-of (:app (:c OfNat) (:arg :explicit 0 (:c Nat)) (:arg :explicit 1 (:lit 2))))))",
                },
            }
        ],
        "isConversion": False,
        "fragment": "tactic",
    }
    implicit_goal = {
        "name": "_uniq.9",
        "target": {
            "sexp": "((:c Eq) (:c Nat) (:fv _uniq.8) (:fv _uniq.8))",
            "pp": "a = a",
            "modelSexpVersion": 1,
            "modelSexp": "(:app (:c Eq) (:arg :implicit-type 0 (:c Nat)) (:arg :explicit 1 (:fv FV0)) (:arg :explicit 2 (:fv FV0)))",
        },
        "vars": [
            {
                "userName": "a",
                "name": "_uniq.8",
                "isLet": False,
                "isInstance": False,
                "isInaccessible": False,
                "contextIndex": 0,
                "binderRole": ":implicit",
                "type": {"sexp": "(:c Nat)", "pp": "Nat", "modelSexpVersion": 1, "modelSexp": "(:c Nat)"},
            }
        ],
        "isConversion": False,
        "fragment": "tactic",
    }
    return {
        probe.name: {"goals": [goal]}
        for probe, goal in zip(
            DEFAULT_CAPABILITY_PROBES,
            [shadow_binder_goal, shadow_binder_goal, let_goal, implicit_goal],
        )
    }


def test_probe_live_capabilities_reports_verified_state() -> None:
    server = _FakeServer(_live_payloads())
    results = asyncio.run(probe_live_capabilities(server))
    assert all_capabilities_verified(results) is True
    assert set(results) == {probe.name for probe in DEFAULT_CAPABILITY_PROBES}


def test_probe_live_capabilities_fails_closed_on_misbehaving_server() -> None:
    payloads = _live_payloads()
    let_variables = payloads[DEFAULT_CAPABILITY_PROBES[2].name]["goals"][0]["vars"]
    let_variables[0]["value"] = None

    server = _FakeServer(payloads)
    results = asyncio.run(probe_live_capabilities(server))
    assert all_capabilities_verified(results) is False
    assert results["binders"].ok is True
    assert results["local_lets"].ok is False and results["local_lets"].detail
    # A crashing server for one probe must not break the whole report.
    assert results["shadowing"].ok is True


def test_probe_live_capabilities_fails_closed_without_raw_payload_access() -> None:
    class _ParsedOnlyServer:
        """Stands for the pinned client: parses payloads, strips v2 fields."""

        async def goal_start_async(self, expression: str) -> dict:
            return {"stateId": 0, "goals": [], "messages": []}

    results = asyncio.run(probe_live_capabilities(_ParsedOnlyServer()))
    assert all_capabilities_verified(results) is False
    for probe in DEFAULT_CAPABILITY_PROBES:
        assert "does not expose raw payload access" in results[probe.name].detail


# Exact raw goal payload emitted by the pinned PyPantograph rev
# (3322696b72e...) with "printExprAST" and "printExprModelAST" enabled,
# captured live against the bundled pantograph-repl with core Lean v4.29.1:
#   options.set accepted both options, but the payload carries plain "sexp"
#   and the variables lack every v2-specific field.
PINNED_REV_PAYLOAD = {
    "goals": [
        {
            "fragment": "tactic",
            "name": "_uniq.7",
            "target": {"pp": "True", "sexp": "(:c True)"},
            "vars": [
                {
                    "isInaccessible": False,
                    "name": "_uniq.6",
                    "type": {"pp": "Prop", "sexp": "(:sort 0)"},
                    "userName": "p",
                }
            ],
        }
    ],
    "hasSorry": False,
    "hasUnsafe": False,
    "messages": [],
    "nextStateId": 1,
}


def test_pinned_rev_payload_fails_closed_with_actionable_gap_report() -> None:
    from maths_ai.hybrid_reasoner.pantograph_adapter import (
        pantograph_state_to_goals,
        protocol_gap_report,
    )

    # The pinned client payload fails closed at the adapter normalization
    # layer; ModelSExprProtocolError (a ValueError) covers the strict layer.
    with pytest.raises(ValueError, match="modelSexp"):
        pantograph_state_to_goals(PINNED_REV_PAYLOAD)

    report = protocol_gap_report(PINNED_REV_PAYLOAD)
    assert "plain-sexp (pinned rev)" in report["expression_style"]
    # None of the v2 model-S-expression fields exist on the pinned client.
    assert report["missing_expression_fields"] == [
        "modelSexp",
        "modelSexpVersion",
    ]
    assert set(report["missing_variable_fields"]) >= {
        "binderRole",
        "contextIndex",
        "isInstance",
        "isLet",
    }


def test_probe_live_capabilities_attaches_gap_report_for_pinned_rev() -> None:
    server = _FakeServer(
        {"binders": {"goals": [PINNED_REV_PAYLOAD["goals"][0]]}}
    )
    results = asyncio.run(probe_live_capabilities(server))
    assert results["binders"].ok is False
    assert "gap_report=" in results["binders"].detail
    assert "plain-sexp (pinned rev)" in results["binders"].detail
