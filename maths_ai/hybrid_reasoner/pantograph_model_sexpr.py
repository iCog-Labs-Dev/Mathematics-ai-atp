from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

import pantograph.expr as expr_mod
from pantograph.server import Server

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import (
    MODEL_SEXPR_GRAPH_SPEC,
)

if TYPE_CHECKING:
    from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv


MODEL_SEXPR_SERVER_OPTIONS: dict[str, bool] = {
    "printExprAST": True,
    "printExprModelAST": True,
}


class PantographModelSExprError(RuntimeError):
    pass


class UnsupportedPyPantographError(PantographModelSExprError):
    pass


class PantographModelSExprProtocolError(PantographModelSExprError):
    pass


@dataclass(frozen=True)
class ModelSExprVariable(expr_mod.Variable):
    model_sexp: str = ""
    model_sexp_version: int = 0
    internal_name: str = ""
    context_index: int = -1
    binder_role: str = ""
    is_instance: bool = False
    is_let: bool = False
    value_model_sexp: str | None = None


@dataclass(frozen=True)
class ModelSExprGoal(expr_mod.Goal):
    model_sexp: str = ""
    model_sexp_version: int = 0


def _required_expression(payload: Any, *, owner: str) -> tuple[str, str, int]:
    if not isinstance(payload, dict):
        raise PantographModelSExprProtocolError(f"{owner} expression payload is missing.")
    pp = payload.get("pp")
    model_sexp = payload.get("modelSexp")
    version = payload.get("modelSexpVersion")
    if not isinstance(pp, str) or not isinstance(model_sexp, str):
        raise PantographModelSExprProtocolError(
            f"{owner} requires string pp and modelSexp fields."
        )
    if version != MODEL_SEXPR_GRAPH_SPEC.model_sexp_version:
        raise PantographModelSExprProtocolError(
            f"{owner} modelSexpVersion is {version!r}; expected "
            f"{MODEL_SEXPR_GRAPH_SPEC.model_sexp_version}."
        )
    return pp, model_sexp, int(version)


def install_model_sexpr_pypantograph_patch() -> None:
    """Install the strict parser extension required by model-S-expression search."""
    if getattr(expr_mod.Goal.parse, "_maths_ai_model_sexpr_patch", False):
        return
    variable_fields = {item.name for item in fields(expr_mod.Variable)}
    goal_fields = {item.name for item in fields(expr_mod.Goal)}
    if variable_fields != {"t", "v", "name"}:
        raise UnsupportedPyPantographError(
            f"Unsupported PyPantograph Variable fields: {sorted(variable_fields)}."
        )
    if not {"id", "variables", "target", "sibling_dep", "name", "mode"}.issubset(
        goal_fields
    ):
        raise UnsupportedPyPantographError(
            f"Unsupported PyPantograph Goal fields: {sorted(goal_fields)}."
        )

    original_goal_parse = expr_mod.Goal.parse

    def parse_variable(payload: dict) -> expr_mod.Variable:
        type_pp, type_model_sexp, type_version = _required_expression(
            payload.get("type"), owner="local type"
        )
        context_index = payload.get("contextIndex")
        binder_role = payload.get("binderRole")
        if not isinstance(context_index, int) or not isinstance(binder_role, str):
            raise PantographModelSExprProtocolError(
                "Every local requires integer contextIndex and string binderRole."
            )
        value_payload = payload.get("value")
        value_pp: str | None = None
        value_model_sexp: str | None = None
        if value_payload is not None:
            value_pp, value_model_sexp, value_version = _required_expression(
                value_payload, owner="local value"
            )
            if value_version != type_version:
                raise PantographModelSExprProtocolError(
                    "Local type and value use different model S-expression versions."
                )
        return ModelSExprVariable(
            t=type_pp,
            v=value_pp,
            name=payload.get("userName"),
            model_sexp=type_model_sexp,
            model_sexp_version=type_version,
            internal_name=str(payload.get("name") or ""),
            context_index=context_index,
            binder_role=binder_role,
            is_instance=bool(payload.get("isInstance", False)),
            is_let=bool(payload.get("isLet", False)),
            value_model_sexp=value_model_sexp,
        )

    def parse_goal(payload: dict, sibling_map: dict[str, int]) -> expr_mod.Goal:
        try:
            parsed = original_goal_parse(payload, sibling_map)
        except (KeyError, TypeError, ValueError) as exc:
            raise PantographModelSExprProtocolError(
                f"Pantograph goal payload is incompatible: {exc}."
            ) from exc
        _target_pp, target_model_sexp, target_version = _required_expression(
            payload.get("target"), owner="goal target"
        )
        return ModelSExprGoal(
            id=parsed.id,
            variables=parsed.variables,
            target=parsed.target,
            sibling_dep=parsed.sibling_dep,
            name=parsed.name,
            mode=parsed.mode,
            model_sexp=target_model_sexp,
            model_sexp_version=target_version,
        )

    parse_goal._maths_ai_model_sexpr_patch = True
    expr_mod.Variable.parse = staticmethod(parse_variable)
    expr_mod.Goal.parse = staticmethod(parse_goal)


def pantograph_goal_to_goal(goal: expr_mod.Goal) -> Goal:
    if not isinstance(goal, ModelSExprGoal):
        raise PantographModelSExprProtocolError(
            "Pantograph goal was not parsed by the model S-expression codec."
        )
    locals_: list[GoalLocal] = []
    for variable in goal.variables:
        if not isinstance(variable, ModelSExprVariable):
            raise PantographModelSExprProtocolError(
                "Pantograph local was not parsed by the model S-expression codec."
            )
        locals_.append(
            GoalLocal(
                user_name=variable.name or "_",
                internal_name=variable.internal_name,
                context_index=variable.context_index,
                binder_role=variable.binder_role,
                is_instance=variable.is_instance,
                is_let=variable.is_let,
                type_pp=variable.t,
                type_model_sexp=variable.model_sexp,
                value_pp=variable.v,
                value_model_sexp=variable.value_model_sexp,
                model_sexp_version=variable.model_sexp_version,
            )
        )
    return Goal(
        expression=str(goal.target),
        goal_model_sexp=goal.model_sexp,
        locals=locals_,
        case_tag=goal.name,
        model_sexp_version=goal.model_sexp_version,
    )


def pantograph_state_to_goals(state: expr_mod.GoalState) -> list[Goal]:
    return [pantograph_goal_to_goal(goal) for goal in state.goals]


async def probe_model_sexpr_capability(server: Server) -> None:
    state = await server.goal_start_async(
        "∀ (α : Type) [inst : Inhabited α] (x : α), x = x"
    )
    state = await server.goal_tactic_async(state, "intro α inst x")
    goals = pantograph_state_to_goals(state)
    if len(goals) != 1 or len(goals[0].locals or []) != 3:
        raise PantographModelSExprProtocolError(
            "Model S-expression probe returned an unexpected goal or local count."
        )
    locals_ = goals[0].locals or []
    labels = [local.context_index for local in locals_]
    if labels != [0, 1, 2]:
        raise PantographModelSExprProtocolError(
            f"Model S-expression probe returned context indices {labels}, expected [0, 1, 2]."
        )
    if not locals_[1].is_instance:
        raise PantographModelSExprProtocolError(
            "Model S-expression probe did not preserve the instance local."
        )
    for local in locals_:
        if f"FV{local.context_index}" not in local.type_model_sexp and local.context_index:
            # The first occurrence of a local need not refer to itself. Later local
            # types and the target must nevertheless expose free-variable identities.
            continue
    combined = " ".join(
        [goals[0].goal_model_sexp or "", *(local.type_model_sexp for local in locals_)]
    )
    if "FV0" not in combined or "FV2" not in combined:
        raise PantographModelSExprProtocolError(
            "Model S-expression probe did not expose expected FV context labels."
        )


async def create_model_sexpr_server(env: "PantographEnv") -> Server:
    install_model_sexpr_pypantograph_patch()
    if env.source_root is None or env.pantograph_repl is None:
        raise RuntimeError(
            "Model S-expression search requires both source_root and pantograph_repl."
        )
    if "Mathlib" not in env.imports:
        raise RuntimeError("Model S-expression search requires the Mathlib import.")
    options = dict(env.options)
    for key, required in MODEL_SEXPR_SERVER_OPTIONS.items():
        if key in options and options[key] is not required:
            raise RuntimeError(f"Pantograph option {key} conflicts with the graph contract.")
        options[key] = required
    strict_env = replace(env, options=options)
    strict_env.verify()
    server = await strict_env.create_server()
    try:
        await probe_model_sexpr_capability(server)
    except Exception:
        server._close()
        raise
    return server
