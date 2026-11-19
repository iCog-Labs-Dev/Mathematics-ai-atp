"""Frozen accepted proposals over the shared graph, loader and flat decoder."""

from dataclasses import dataclass, replace
import copy
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from maths_ai.gnn_inference.atp_lean_gnn.joint_loader import load_joint_neural_bundle
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import (
    CandidateSource, SourceCandidates, argument_count_logits, decode_action,
    initial_decoder_state, local_type_signals, source_logits, goal_first_text,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_graph import goal_to_model_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg, build_premise_mask
from maths_ai.gnn_inference.atp_lean_gnn.training import transform_edge_index
from maths_ai.gnn_inference.atp_lean_gnn.native_session import LiveStateError
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import file_hash
from maths_ai.lean_replay.sexp import parse_sexp_string
from .flat_actions import FlatReference, RenderError, render_flat_action
from .live_state import goal_payload_to_goal


@dataclass(frozen=True)
class LiveCandidate:
    code: str
    tactic: str
    log_probability: float
    references: tuple

    def to_dict(self):
        return {"code": self.code, "tactic": self.tactic, "log_probability": self.log_probability,
            "references": [{"source": ref.source, "identity": ref.identity} for ref in self.references]}


def prepare_live_goal(payload, vocab):
    if payload.get("isConversion") is not False or not isinstance(payload.get("vars"), list):
        raise LiveStateError("ordinary normalized goal and complete context required")
    if type(payload.get("target", {}).get("modelSexpVersion")) is not int:
        raise LiveStateError("integer expression version required")
    for variable in payload["vars"]:
        if type(variable.get("contextIndex")) is not int or type(variable.get("type", {}).get("modelSexpVersion")) is not int:
            raise LiveStateError("integer local index and expression version required")
    goal = goal_payload_to_goal({**payload, "variables": payload["vars"]})
    indices = {local.context_index for local in goal.locals}
    for text in (goal.goal_model_sexp, *(local.type_model_sexp for local in goal.locals)):
        pending = [parse_sexp_string(text)]
        while pending:
            node = pending.pop()
            if not isinstance(node, list):
                continue
            if node and node[0] == ":fv" and (len(node) != 2 or not isinstance(node[1], str)
                    or not node[1].startswith("FV") or not node[1][2:].isdecimal() or int(node[1][2:]) not in indices):
                raise LiveStateError("model expression references an unknown input local")
            pending.extend(node)
    dag = goal_to_model_dag(goal)
    data = dag_to_pyg(dag, vocab)
    data.state_node_index = torch.tensor([next(node.id for node in dag.nodes if node.label == "State")])
    data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
    data.edge_index = transform_edge_index(data.edge_index, edge_mode="bidirectional", deduplicate=False)
    local_nodes = {node.id: int(node.label[2:]) for node in dag.nodes if node.node_type == "var" and not node.is_bound
        and node.label.startswith("FV") and node.label[2:].isdecimal() and int(node.label[2:]) in indices}
    for node in dag.nodes:
        if node.label == "Hyp" and len(node.children) == 4 and node.children[0] in local_nodes:
            local_nodes[node.id] = local_nodes[node.children[0]]
    text = "\n".join([local.declaration_pp for local in goal.locals] + [f"⊢ {goal.expression}"])
    context = [{"context_index": variable["contextIndex"], "user_name": variable["userName"],
                "isInaccessible": bool(variable.get("isInaccessible", False))} for variable in payload["vars"]]
    return goal, dag, Batch.from_data_list([data]), local_nodes, text, context


class LiveJointPolicy:
    def __init__(self, bundle):
        self.system = bundle
        self.library_names = {key: value.name for key, value in bundle.lemma_corpus.items()}
        self.id_to_node = {value: key for key, value in bundle.node_vocab.items()}
        self.id_to_tactic = {value: key for key, value in bundle.tactic_vocab.items()}
        self.excluded_library_names = frozenset()

    @classmethod
    def from_bundle(cls, root, *, device="cpu"):
        if file_hash(Path(root) / "bundle.json") != "ebc7620282838e7a0d26250ddbf159ec02c859700c17217d9a43db3ef4145107":
            raise ValueError("bundle differs from pinned accepted release")
        return cls(load_joint_neural_bundle(root, device=device))

    def excluding(self, names):
        policy = copy.copy(self)
        policy.excluded_library_names = frozenset(names)
        return policy

    @torch.no_grad()
    def predict(self, payload, *, top_k=8):
        try:
            goal, dag, batch, local_nodes, text, context = prepare_live_goal(payload, self.system.node_vocab)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
            raise LiveStateError(str(error)) from error
        batch = batch.to(self.system.device)
        policy = self.system.policy
        nodes = policy.pointer.backbone.encode_nodes(batch)
        state = policy.pointer.backbone.readout(nodes, batch)[0]
        tactic_logps = F.log_softmax(policy.pointer.backbone.classifier(state), dim=-1)
        ids, vectors, scores = self.system.retrieve_library_pool(goal_first_text(text), k=200)
        keep = [index for index, key in enumerate(ids) if self.library_names[key] not in self.excluded_library_names
                and not self.library_names[key].startswith("GnnNativeRuntime.")]
        local_ids = batch.premise_mask.nonzero(as_tuple=False).view(-1)
        base_pool = SourceCandidates(tuple(local_ids.tolist()), nodes.index_select(0, local_ids), tuple(ids[i] for i in keep),
            vectors[keep].to(nodes), scores[keep].to(nodes))
        strengths = self.system.artifact.inference_config["profiles"]["accepted_local_cues"]["local_type_bonus_strengths"]
        candidates, rejected, seen = [], [], set()
        for tactic_id in tactic_logps.topk(min(top_k, len(tactic_logps))).indices.tolist():
            tactic = self.id_to_tactic[tactic_id]
            signal = local_type_signals(base_pool, batch.x, self.id_to_node, tactic_name=tactic,
                goal_sexp=goal.goal_model_sexp, hyp_sexps=goal.graph_locals())
            pool = replace(base_pool, local_type_signal=signal, local_type_bonus=float(strengths.get(tactic, 0)))
            action = decode_action(policy, state, tactic_id, pool)
            if not action.complete:
                rejected.append({"tactic": tactic, "reason": "incomplete_decode"})
                continue
            try:
                if tactic.startswith("<"):
                    raise RenderError("unknown tactic label")
                refs = []
                for arg in action.arguments:
                    if arg.source is CandidateSource.LOCAL:
                        if arg.stable_id not in local_nodes:
                            raise RenderError("selected graph node is not an input local")
                        refs.append(FlatReference("local", local_nodes[arg.stable_id]))
                    else:
                        refs.append(FlatReference("library", self.library_names[arg.stable_id]))
                code = render_flat_action(tactic, refs, context, max_args=policy.pointer.max_args)
            except RenderError as error:
                rejected.append({"tactic": tactic, "reason": str(error)})
                continue
            hidden = initial_decoder_state(policy, state, tactic_id)
            score = tactic_logps[tactic_id] + F.log_softmax(argument_count_logits(policy, hidden), dim=0)[len(action.arguments)]
            for step, arg in enumerate(action.arguments):
                source, local, library, adapted = source_logits(policy, hidden, pool, step=step)
                source_id = 0 if arg.source is CandidateSource.LOCAL else 1
                scores_ = local if source_id == 0 else library
                score = score + F.log_softmax(source, dim=0)[source_id] + F.log_softmax(scores_, dim=0)[arg.source_position]
                selected = pool.local_vectors[arg.source_position] if source_id == 0 else adapted[arg.source_position]
                hidden = policy.pointer.argument_selector.gru(selected.view(1, -1), hidden)
            score = float(score)
            if not math.isfinite(score):
                rejected.append({"tactic": tactic, "reason": "nonfinite_action_score"})
            elif code not in seen:
                candidates.append(LiveCandidate(code, tactic, score, tuple(refs)))
                seen.add(code)
        candidates.sort(key=lambda candidate: candidate.log_probability, reverse=True)
        from types import SimpleNamespace
        return SimpleNamespace(candidates=candidates, rejected=rejected)
