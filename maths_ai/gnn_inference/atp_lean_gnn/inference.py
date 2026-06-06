"""Inference pipeline for end-to-end tactic prediction.

This module provides the ``InferencePipeline`` which integrates graph conversion,
tactic prediction, premise retrieval, and candidate scoring to produce a final
tactic string.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch

from .argument_selector import TacticWithArgsClassifier
from .graph import DAGBuilder, GraphNode, proof_state_to_dag
from .labels import get_tactic_arity
from .lemma_index import LemmaIndex
from .premise_pool import build_unified_pools
from .premise_scoring import PremiseScorer
from .pyg import build_premise_mask, dag_to_pyg
from .state import ProofState, parse_state


def _resolve_local_node_name(node: GraphNode, dag: DAGBuilder) -> str:
    """Attempt to extract a readable hypothesis or variable name from a node."""
    if node.label == "Hyp" and node.children:
        name_node = dag.nodes[node.children[0]]
        return name_node.label
    return node.label


class InferencePipeline:
    """End-to-end tactic prediction pipeline."""

    def __init__(
        self,
        model: TacticWithArgsClassifier,
        scorer: PremiseScorer,
        lemma_index: LemmaIndex,
        node_vocab: dict[str, int],
        tactic_vocab: dict[str, int],
        device: torch.device,
        k: int = 500,
        lemma_corpus: dict[int, LemmaRecord] | None = None,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.lemma_index = lemma_index
        self.node_vocab = node_vocab
        self.tactic_vocab = tactic_vocab
        self.device = device
        self.k = k
        self.lemma_corpus = lemma_corpus

        # Invert tactic vocab for decoding
        self.id_to_tactic = {idx: name for name, idx in tactic_vocab.items()}

        self.model.eval()
        self.scorer.eval()

    @torch.no_grad()
    def predict_tactic(self, state_str: str) -> str:
        """Predict a full tactic string given a Lean proof state."""
        # Parse the raw state string into a ProofState
        state = parse_state(state_str)
        
        # 1. Graph construction
        dag = proof_state_to_dag(state)
        data = dag_to_pyg(dag, self.node_vocab)
        
        # Find the root State node for graph readout
        try:
            state_idx = next(i for i, n in enumerate(dag.nodes) if n.label == "State")
        except StopIteration:
            state_idx = 0
        data.state_node_index = torch.tensor([state_idx], dtype=torch.long)
        
        # Build premise mask for local candidates
        premise_mask = build_premise_mask(dag)
        data.premise_mask = torch.tensor(premise_mask, dtype=torch.bool)
        
        # Move to device and batch
        data = data.to(self.device)
        batch = Batch.from_data_list([data])

        # 2. Encode state and predict tactic
        node_embeddings = self.model.backbone.encode_nodes(batch)
        state_emb = self.model.backbone.readout(node_embeddings, batch)
        
        tactic_logits = self.model.backbone.classifier(state_emb)
        tactic_id = tactic_logits.argmax(dim=-1).item()
        tactic_name = self.id_to_tactic.get(tactic_id, "<UNK>")
        
        arity = get_tactic_arity(tactic_name)
        if arity == 0:
            return tactic_name
            
        # 3. Retrieve and Score Premises
        # Convert integer tactic ID to a tensor for the embedding layer
        tactic_id_tensor = torch.tensor([tactic_id], dtype=torch.long, device=self.device)
        tactic_emb = self.model.tactic_embedding(tactic_id_tensor)
        
        pools = build_unified_pools(
            state_emb,
            node_embeddings,
            batch.premise_mask,
            batch.batch,
            lemma_index=self.lemma_index,
            k=self.k,
        )
        
        pool = pools[0]
        if not pool.candidate_ids:
            return tactic_name
            
        # Score candidates
        scores = self.scorer.score(state_emb.squeeze(0), tactic_emb.squeeze(0), pool.candidate_vectors)
        
        # Pick top `arity` arguments
        sorted_indices = scores.argsort(descending=True)
        top_indices = sorted_indices[:arity].tolist()
        
        arguments = []
        for idx in top_indices:
            source = pool.candidate_sources[idx]
            cid = pool.candidate_ids[idx]
            
            if source == "local":
                node = dag.nodes[cid]
                arg_str = _resolve_local_node_name(node, dag)
            else:
                if self.lemma_corpus and cid in self.lemma_corpus:
                    arg_str = self.lemma_corpus[cid].name
                else:
                    arg_str = f"<lemma_{cid}>"
                
            arguments.append(arg_str)
            
        return f"{tactic_name} {' '.join(arguments)}"
