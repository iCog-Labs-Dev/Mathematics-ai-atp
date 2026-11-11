"""Rank executable flat actions from the accepted joint bundle on live states.

One greedy argument sequence per top-ranked tactic family. No hand-written
fallback tactics, proof LLM, or changes to the accepted candidate distribution.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .gold_replay import FlatReference, RenderError, render_flat_action
from .joint_source_decoder import candidate_groups_from_batch, local_type_signals
from .live_state import LiveGraph, prepare_live_goal
from .premise_pool import CandidateSource


@dataclass(frozen=True)
class LiveCandidate:
    code: str
    tactic: str
    log_probability: float
    references: tuple[FlatReference, ...]

    def to_dict(self):
        return {"code": self.code, "tactic": self.tactic, "log_probability": self.log_probability,
                "references": [{"source": r.source, "identity": r.identity} for r in self.references]}


@dataclass
class LivePrediction:
    graph: LiveGraph
    candidates: list[LiveCandidate]
    rejected: list[dict]


def _action_log_probability(policy, state, tactic_id, pool, arguments):
    """Count + source + identity probability, matching predicted-count decode."""
    hidden = policy.initial_state(state, tactic_id)
    score = F.log_softmax(policy.arg_count_head(hidden).view(-1), dim=0)[len(arguments)]
    adapted = policy.lemma_adapter(pool.lemma_vectors) if pool.lemma_ids and arguments else None
    for step, argument in enumerate(arguments):
        source, local, library = policy.score_step(hidden, pool, adapted=adapted, step=step)
        source_id = 0 if argument.source is CandidateSource.LOCAL else 1
        identities = pool.local_ids if source_id == 0 else pool.lemma_ids
        position = identities.index(argument.stable_id)
        scores = local if source_id == 0 else library
        score = score + F.log_softmax(source, dim=0)[source_id] + F.log_softmax(scores, dim=0)[position]
        hidden = policy.pointer.argument_selector.gru(
            policy.selected_vector(pool, argument.source, position, adapted=adapted), hidden)
    return float(score)


class LiveJointPolicy:
    def __init__(self, system, *, edge_mode: str):
        self.system = system
        self.edge_mode = edge_mode
        self.id_to_tactic = {identity: name for name, identity in system.tactic_vocab.items()}
        self.id_to_node = {identity: name for name, identity in system.node_vocab.items()}
        self.library_names = dict(zip(system.lemma_index.lemma_ids, system.lemma_index.lemma_names))

    @classmethod
    def from_bundle(cls, root, *, device="cpu", profile="accepted_local_cues"):
        from .joint_bundle import load_joint_source_bundle
        system = load_joint_source_bundle(root, device=device, profile=profile)
        config = json.loads((system.root / "policy/config.json").read_text())
        return cls(system, edge_mode=config["input_contract"]["edge_mode"])

    @torch.no_grad()
    def predict(self, goal: dict, *, top_k: int = 8) -> LivePrediction:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        graph = prepare_live_goal(goal, self.system.node_vocab, edge_mode=self.edge_mode)
        policy = self.system.policy
        batch = graph.batch.clone().to(next(policy.parameters()).device)
        batch.y = torch.zeros(1, dtype=torch.long, device=batch.x.device)
        nodes = policy.pointer.backbone.encode_nodes(batch)
        state = policy.pointer.backbone.readout(nodes, batch)[0]
        tactic_log_probs = F.log_softmax(policy.pointer.backbone.classifier(state), dim=-1)
        ids, vectors, scores = self.system.retrieve([graph.text])
        base_pool = candidate_groups_from_batch(batch, nodes, ids, vectors, scores)[0]
        candidates, rejected, seen = [], [], set()
        for tactic_id in tactic_log_probs.topk(min(top_k, tactic_log_probs.numel())).indices.tolist():
            tactic = self.id_to_tactic[tactic_id]
            pool = base_pool
            if self.system.profile == "accepted_local_cues":
                signal = local_type_signals(pool, batch.x, self.id_to_node,
                    tactic_name=tactic, **graph.normalized)
                pool = replace(pool, local_type_signal=signal, local_type_tactic=tactic)
            arguments, complete = policy.decode(state, tactic_id, pool)
            if not complete:
                rejected.append({"tactic": tactic, "reason": "incomplete_decode"})
                continue
            try:
                if tactic.startswith("<"):
                    raise RenderError("unknown tactic label")
                references = []
                for argument in arguments:
                    if argument.source is CandidateSource.LOCAL:
                        if argument.stable_id not in graph.local_node_context:
                            raise RenderError("selected graph node is not an input local")
                        references.append(FlatReference("local", graph.local_node_context[argument.stable_id]))
                    else:
                        if argument.stable_id not in self.library_names:
                            raise RenderError("selected lemma absent from bound corpus")
                        references.append(FlatReference("library", self.library_names[argument.stable_id]))
                code = render_flat_action(tactic, references, graph.local_context,
                                          max_args=policy.pointer.max_args)
            except RenderError as exc:
                rejected.append({"tactic": tactic, "reason": str(exc)})
                continue
            score = float(tactic_log_probs[tactic_id]) + _action_log_probability(
                policy, state, tactic_id, pool, arguments)
            if not math.isfinite(score):
                rejected.append({"tactic": tactic, "reason": "nonfinite_action_score"})
            elif code not in seen:
                candidates.append(LiveCandidate(code, tactic, score, tuple(references)))
                seen.add(code)
        candidates.sort(key=lambda candidate: candidate.log_probability, reverse=True)
        return LivePrediction(graph, candidates, rejected)
