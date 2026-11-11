"""Native snapshots and strict normalized inputs for GNN-only execution.

No goal is reconstructed by elaborating its printed text. Text assembled from
Lean's pretty fields is used only as a retrieval query; graph inputs come from
the same versioned model S-expressions as offline preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Batch

from .graph import DAGBuilder, parse_sexp_string, proof_state_to_dag, root_state_node_id
from .preparation import ModelSExprCache
from .pyg import build_premise_mask, dag_to_pyg
from .sexpr_extraction import PantographInvocationClient
from .training import transform_edge_index


class LiveStateError(ValueError):
    """The runtime did not supply a complete, supported model input."""


class TacticFailure(RuntimeError):
    """A candidate failed without invalidating its parent snapshot."""


def _expression(payload):
    if (not isinstance(payload, dict)
            or type(payload.get("modelSexpVersion")) is not int
            or payload["modelSexpVersion"] != ModelSExprCache.EXPRESSION_VERSION
            or not isinstance(payload.get("modelSexp"), str)
            or not payload["modelSexp"].strip()
            or not isinstance(payload.get("pp"), str)):
        raise LiveStateError("complete version-1 model S-expression and pretty text required")
    return payload["modelSexp"]


def normalized_goal_record(goal: dict) -> dict:
    """Match offline model sidecar fields, without legacy/text fallbacks."""
    if not isinstance(goal, dict) or goal.get("isConversion") is not False:
        raise LiveStateError("ordinary tactic-mode goal required")
    target = _expression(goal.get("target"))
    variables = goal.get("vars")
    if not isinstance(variables, list):
        raise LiveStateError("complete local context required")
    hypotheses, indices = [], set()
    for variable in variables:
        if not isinstance(variable, dict):
            raise LiveStateError("invalid local declaration")
        index, role = variable.get("contextIndex"), variable.get("binderRole")
        if type(index) is not int or index < 0 or index in indices or not isinstance(role, str):
            raise LiveStateError("unique nonnegative context indices and binder roles required")
        if not isinstance(variable.get("userName"), str):
            raise LiveStateError("local display name required")
        indices.add(index)
        hypotheses.append({
            "name": variable["userName"], "internal_name": str(variable.get("name", "")),
            "context_index": index, "binder_role": role,
            "is_instance": bool(variable.get("isInstance", False)),
            "is_let": bool(variable.get("isLet", False)),
            "sexp": _expression(variable.get("type")),
        })
    for expression in [target, *(hypothesis["sexp"] for hypothesis in hypotheses)]:
        pending = [parse_sexp_string(expression)]
        while pending:
            node = pending.pop()
            if not isinstance(node, list):
                continue
            if node and node[0] == ":fv":
                if (len(node) != 2 or not isinstance(node[1], str)
                        or not node[1].startswith("FV") or not node[1][2:].isdecimal()
                        or int(node[1][2:]) not in indices):
                    raise LiveStateError("model expression references an unknown input local")
            pending.extend(node)
    return {"goal_sexp": target, "hyp_sexps": hypotheses}


def retrieval_state_text(goal: dict) -> str:
    """Readable live query, not a claim of byte parity with dataset state text."""
    lines = []
    for variable in goal["vars"]:
        line = f"{variable['userName'] or '_'} : {variable['type']['pp']}"
        if variable.get("isLet"):
            value = variable.get("value")
            if not isinstance(value, dict) or not isinstance(value.get("pp"), str):
                raise LiveStateError("pretty value required for a local let")
            line += f" := {value['pp']}"
        lines.append(line)
    return "\n".join(lines + [f"⊢ {goal['target']['pp']}"])


@dataclass
class LiveGraph:
    dag: DAGBuilder
    batch: Batch
    normalized: dict
    text: str
    local_node_context: dict[int, int]
    local_context: list[dict]


def prepare_live_goal(goal: dict, node_vocab: dict[str, int], *, edge_mode: str) -> LiveGraph:
    record = normalized_goal_record(goal)
    text = retrieval_state_text(goal)
    dag = proof_state_to_dag(text, **record)
    data = dag_to_pyg(dag, node_vocab)
    data.state_node_index = torch.tensor([root_state_node_id(dag)], dtype=torch.long)
    data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
    data.edge_index = transform_edge_index(data.edge_index, edge_mode=edge_mode)
    # Resolve using original DAG labels, not vocabulary IDs: unseen names/FV
    # labels may map to <UNK>, but their runtime identities must remain distinct.
    context_indices = {item["context_index"] for item in record["hyp_sexps"]}
    local_nodes = {}
    for node in dag.nodes:
        if node.label.startswith("FV") and node.label[2:].isdecimal():
            index = int(node.label[2:])
            if index in context_indices and node.node_type == "var":
                local_nodes[node.id] = index
    # The accepted pointer pool also includes Hyp nodes. Those are legitimate
    # whole-local references; unrelated expression/type nodes are not.
    for node in dag.nodes:
        if node.label == "Hyp" and len(node.children) == 4 and node.children[0] in local_nodes:
            local_nodes[node.id] = local_nodes[node.children[0]]
    context = [{"context_index": v["contextIndex"], "user_name": v["userName"],
                "isInaccessible": bool(v.get("isInaccessible", False))} for v in goal["vars"]]
    return LiveGraph(dag, Batch.from_data_list([data]), record, text, local_nodes, context)


@dataclass(frozen=True)
class NativeSnapshot:
    state_id: int
    goals: tuple[dict, ...]

    @property
    def closed(self) -> bool:
        return not self.goals


class LiveLeanSession:
    """Keep all pending goals while executing alternatives from saved parents.

    The pinned backend's automatic resumption duplicates goals for no-op
    tactics. Use explicit, name-deduplicated resumption instead. Lean filters
    assigned goals from that complete pending-goal list.
    """

    def __init__(self, *, source_root: Path, pantograph_repl: Path,
                 imports=("Init", "Mathlib"), timeout: int = 30):
        self.client = PantographInvocationClient(
            source_root=source_root, pantograph_repl=pantograph_repl,
            capture_model_sexprs=True, imports=tuple(imports),
        )
        self.timeout = timeout

    async def __aenter__(self):
        try:
            await self.client.start()
            await self.client.call("options.set", {
                "automaticMode": False, "noRepeat": False, "printExprPretty": True,
            }, timeout=self.timeout)
        except BaseException:
            await self.client.close()
            raise
        return self

    async def __aexit__(self, *exc):
        await self.client.close()

    async def start(self, expression: str) -> NativeSnapshot:
        response = await self.client.call("goal.start", {"expr": expression}, timeout=self.timeout)
        # goal.start returns only an ID. A no-op gets authoritative serialized
        # goals without rebuilding/re-elaborating their printed representations.
        result = await self.client.call("goal.tactic", {
            "stateId": response["stateId"], "goalId": 0, "tactic": "skip",
        }, timeout=self.timeout)
        return self._snapshot(result)

    async def start_declaration(self, header: str) -> NativeSnapshot:
        """Elaborate a trusted declaration header and reopen its single proof hole.

        Unlike quantifying/re-elaborating printed locals, the frontend retains
        declaration-context indices, including hidden local-context slots. The
        initialization stub is never a generated action or a verified proof.
        """
        result = await self.client.call("frontend.process", {
            "file": header.rstrip() + " := by\n  sorry\n",
            "invocations": False, "sorrys": True,
        }, timeout=self.timeout)
        units = result.get("units", [])
        errors = [message for unit in units for message in unit.get("messages", [])
                  if ": error:" in message or message.startswith("error:")]
        if errors:
            raise LiveStateError(f"declaration elaboration failed: {errors}")
        states = [unit for unit in units if "goalStateId" in unit]
        if len(states) != 1 or len(states[0].get("goals", [])) != 1:
            raise LiveStateError(f"declaration must produce one native proof obligation: {result}")
        unit = states[0]
        return self._snapshot({"nextStateId": unit["goalStateId"], "goals": unit["goals"]})

    @staticmethod
    def _snapshot(response) -> NativeSnapshot:
        if "parseError" in response or "tacticErrors" in response:
            raise TacticFailure(str(response.get("parseError", response.get("tacticErrors"))))
        identity = response.get("nextStateId")
        goals = response.get("goals")
        if type(identity) is not int or not isinstance(goals, list):
            raise LiveStateError("native tactic response lacks snapshot/goals")
        if any(not isinstance(goal, dict) for goal in goals):
            raise LiveStateError("invalid serialized pending goal")
        names = [goal.get("name") for goal in goals]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise LiveStateError("native pending goals must have unique identities")
        return NativeSnapshot(identity, tuple(goals))

    async def step(self, parent: NativeSnapshot, tactic: str) -> NativeSnapshot:
        if parent.closed:
            raise LiveStateError("cannot step a closed snapshot")
        result = await self.client.call("goal.tactic", {
            "stateId": parent.state_id, "goalId": 0, "tactic": tactic,
        }, timeout=self.timeout)
        child = self._snapshot(result)
        # A focused tactic response omits siblings. Restore them in the child's
        # metavariable context, where Lean can discard goals solved indirectly.
        names = list(dict.fromkeys(goal["name"] for goal in (*child.goals, *parent.goals)))
        resumed = await self.client.call("goal.continue", {
            "target": child.state_id, "goals": names,
        }, timeout=self.timeout)
        return self._snapshot(resumed)
