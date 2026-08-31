from __future__ import annotations

import hashlib

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    ModelSExprGoal,
    ModelSExprVariable,
)

from maths_ai.gnn_inference.atp_lean_gnn.model_factory import (
    build_actor_critic_model,
    build_pointer_model,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec


def spec(*, hidden_dim: int = 16, num_layers: int = 2, dropout: float = 0.1, max_args: int = 2):
    return ModelSpec.from_dict(
        {
            "architecture": "graphsage",
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "encoder": {"num_layers": num_layers},
            "use_node_type": True,
            "max_args": max_args,
        }
    )


def pointer(num_node_labels: int, num_tactics: int, **kwargs):
    return build_pointer_model(
        model_spec=spec(**kwargs),
        num_node_labels=num_node_labels,
        num_tactics=num_tactics,
    )


def actor_critic(num_node_labels: int, num_tactics: int, **kwargs):
    return build_actor_critic_model(
        model_spec=spec(**kwargs),
        num_node_labels=num_node_labels,
        num_tactics=num_tactics,
    )


def structured_goal(
    expression: str = "p → p", hypotheses: list[str] | None = None
) -> Goal:
    hypotheses = hypotheses or []
    locals_: list[GoalLocal] = []
    for index, hypothesis in enumerate(hypotheses):
        name, separator, type_pp = hypothesis.partition(":")
        if not separator:
            raise ValueError(f"Test hypothesis must have 'name : type' form: {hypothesis!r}")
        locals_.append(
            GoalLocal(
                user_name=name.strip(),
                internal_name=f"_test.{index}",
                context_index=index,
                binder_role=":explicit",
                type_pp=type_pp.strip(),
                type_model_sexp=f"(:c Test.Type{index})",
            )
        )
    digest = hashlib.sha256(expression.encode("utf-8")).hexdigest()[:12]
    return Goal(
        expression=expression,
        goal_model_sexp=f"(:c Test.Goal{digest})",
        locals=locals_,
        model_sexp_version=1,
    )


def pantograph_goal(goal: Goal) -> ModelSExprGoal:
    goal.require_model_state()
    variables = [
        ModelSExprVariable(
            t=local.type_pp,
            v=local.value_pp,
            name=None if local.user_name == "_" else local.user_name,
            model_sexp=local.type_model_sexp,
            model_sexp_version=local.model_sexp_version,
            internal_name=local.internal_name,
            context_index=local.context_index,
            binder_role=local.binder_role,
            is_instance=local.is_instance,
            is_let=local.is_let,
            value_model_sexp=local.value_model_sexp,
        )
        for local in goal.locals or []
    ]
    return ModelSExprGoal(
        id="test-goal",
        variables=variables,
        target=goal.expression,
        name=goal.case_tag,
        model_sexp=goal.goal_model_sexp or "",
        model_sexp_version=goal.model_sexp_version or 0,
    )
