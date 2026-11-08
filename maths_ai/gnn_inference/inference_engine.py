from pathlib import Path
from collections.abc import Mapping
from typing import List, Optional, cast

import torch

from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.data_models.proof_components import Goal
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
from maths_ai.gnn_inference.atp_lean_gnn.training import load_baseline_config, load_prepared_metadata
from maths_ai.gnn_inference.atp_lean_gnn.joint_loader import load_joint_neural_bundle
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import (
    CandidateSource,
    DecodedAction,
    SourceCandidates,
    decode_action,
    goal_first_text,
    goal_state_text,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_graph import goal_to_model_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_premise_mask, dag_to_pyg
from maths_ai.gnn_inference.atp_lean_gnn.graph import DAGBuilder
from maths_ai.gnn_inference.atp_lean_gnn.training import transform_edge_index
from torch_geometric.data import Batch

from .model import GNNPredictor


def _build_source_candidates(bundle, node_embeddings, local_positions, goal: Goal) -> SourceCandidates:
    """One argument pool: local premise-mask rows plus the library text pool."""
    lemma_ids, lemma_vectors, lemma_scores = bundle.retrieve_library_pool(
        goal_first_text(goal_state_text(goal)), k=bundle.pool_size()
    )
    return SourceCandidates(
        local_ids=tuple(int(position) for position in local_positions),
        local_vectors=node_embeddings.index_select(0, node_embeddings.new_tensor(
            local_positions, dtype=torch.long
        )) if local_positions else node_embeddings.new_zeros((0, node_embeddings.size(1))),
        lemma_ids=lemma_ids,
        lemma_vectors=lemma_vectors.to(node_embeddings.device, dtype=node_embeddings.dtype),
        lemma_scores=lemma_scores.to(node_embeddings.device, dtype=node_embeddings.dtype),
    )


class GNNModelEngine:
    @classmethod
    def from_bundle(
        cls,
        bundle_path: Path,
        *,
        index_path: Optional[Path] = None,
        corpus_path: Optional[Path] = None,
        k: int = 500,
        device: str = "cuda",
    ) -> "GNNModelEngine":
        instance = cls.__new__(cls)
        instance.device = torch.device(device if torch.cuda.is_available() else "cpu")
        loaded = load_joint_neural_bundle(bundle_path, device=instance.device)
        instance._joint_bundle = loaded
        instance._is_joint_v2 = True
        return instance

    def __init__(
        self,
        config_path: Path,
        tactic_predictor_model_path: Path,
        argument_predictor_model_path: Path,
        *,
        index_path: Optional[Path] = None,
        corpus_path: Optional[Path] = None,
        scorer_mode: str = "dot",
        k: int = 500,
        device: str = "cuda",
    ):
        """
            Args:
                config_path: path to the baseline training config.json (model architecture
                    and the prepared dataset root used to recover the node/tactic vocabs)
                tactic_predictor_model_path: checkpoint (.pt) holding "model_state_dict" for
                    the trained TacticWithArgsClassifier
                argument_predictor_model_path: checkpoint (.pt) holding "scorer_state_dict"
                    for the trained PremiseScorer
                index_path: optional path to a FAISS lemma index directory
                corpus_path: optional path to lemmas.jsonl for decoding retrieved lemma IDs
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        config = load_baseline_config(config_path)
        metadata = load_prepared_metadata(config.prepared_root)

        tactic_model = TacticWithArgsClassifier(
            num_node_labels=len(metadata.node_vocab),
            num_tactics=len(metadata.tactic_vocab),
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            use_node_type=config.use_node_type,
            max_args=getattr(config, "max_args", 3),
        )
        tactic_checkpoint = torch.load(tactic_predictor_model_path, map_location=self.device, weights_only=False)
        tactic_model.load_state_dict(tactic_checkpoint.get("model_state_dict", tactic_checkpoint))
        tactic_model = tactic_model.to(self.device)
        tactic_model.eval()

        argument_model = PremiseScorer(hidden_dim=config.model.hidden_dim, mode=scorer_mode)
        argument_checkpoint = torch.load(argument_predictor_model_path, map_location=self.device, weights_only=False)
        argument_model.load_state_dict(argument_checkpoint.get("scorer_state_dict", argument_checkpoint))
        argument_model = argument_model.to(self.device)
        argument_model.eval()

        lemma_index = self._load_lemma_index(index_path, config.model.hidden_dim)
        lemma_corpus = self._load_lemma_corpus(corpus_path)
        self._validate_lemma_artifacts(lemma_index, lemma_corpus)

        self.gnn_inference = GNNPredictor(
            tactic_model=tactic_model,
            argument_model=argument_model,
            lemma_index=lemma_index,
            node_vocab=metadata.node_vocab,
            tactic_vocab=metadata.tactic_vocab,
            device=self.device,
            k=k,
            lemma_corpus=lemma_corpus,
        )
        self._joint_bundle = None
        self._is_joint_v2 = False

    @staticmethod
    def _load_lemma_index(index_path: Optional[Path], hidden_dim: int) -> LemmaIndex:
        if index_path is not None and Path(index_path).exists():
            return LemmaIndex.load(index_path)

        import faiss
        import numpy as np

        return LemmaIndex(
            index=faiss.IndexFlatL2(hidden_dim),
            lemma_ids=[],
            lemma_vectors=np.empty((0, hidden_dim), dtype=np.float32),
        )

    @staticmethod
    def _load_lemma_corpus(corpus_path: Optional[Path]):
        if corpus_path is None or not Path(corpus_path).exists():
            return None

        records = load_lemma_corpus(corpus_path)
        return {record.lemma_id: record for record in records}

    @staticmethod
    def _validate_lemma_artifacts(
        lemma_index: LemmaIndex,
        lemma_corpus: Mapping[int, object] | None,
    ) -> None:
        if lemma_corpus is None:
            return
        missing_ids = sorted(set(lemma_index.lemma_ids) - set(lemma_corpus))
        if missing_ids:
            preview = ", ".join(str(item) for item in missing_ids[:5])
            suffix = "..." if len(missing_ids) > 5 else ""
            raise ValueError(
                "Lemma index contains IDs missing from the loaded lemma corpus: "
                f"{preview}{suffix}."
            )

    def inference(
        self,
        goal_expression: str,
        top_k: int = 3,
        *,
        hypotheses: Optional[List[str]] = None,
        structured_goal: Optional[Goal] = None,
    ) -> List[TacticCandidate]:
        """Predict ranked tactic candidates for ``goal_expression``.

        Contract (depended on by ``HybridReasoner.predict_next_tactic``):
        return up to ``top_k`` candidates, each a ``TacticCandidate``
        carrying the tactic family name, its selected argument/premise
        names, and the model's predicted probability — sorted by
        probability, descending.
        """
        if self._is_joint_v2:
            if structured_goal is None:
                raise ValueError(
                    "The v2 bundle requires a structured lean-model-sexp-v2 Goal; "
                    "printed goal text cannot be converted losslessly."
                )
            return self._joint_inference(structured_goal, top_k)

        predictions = self.gnn_inference.predict_tactics_with_arguments(
            goal_expression,
            top_k=top_k,
            hypotheses=hypotheses,
        )
        print(*predictions, "\n")
        return [
            TacticCandidate(
                tactic_name=str(prediction["tactic_name"]),
                arguments=[str(argument) for argument in cast(list, prediction["selected_arguments"])],
                probability=float(cast(float, prediction["probability"])),
            )
            for prediction in predictions
        ]

    @torch.no_grad()
    def _joint_inference(self, goal: Goal, top_k: int) -> List[TacticCandidate]:
        bundle = self._joint_bundle
        if bundle is None:
            raise RuntimeError("v2 inference bundle is not initialized.")
        dag = goal_to_model_dag(goal)
        data = dag_to_pyg(dag, bundle.node_vocab)
        state_idx = next((node.id for node in dag.nodes if node.label == "State"), 0)
        data.state_node_index = torch.tensor([state_idx], dtype=torch.long)
        data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
        data = data.to(self.device)
        data.edge_index = transform_edge_index(data.edge_index, edge_mode="bidirectional")
        batch = Batch.from_data_list([data])
        policy = bundle.policy
        tactic_logits, _, _, _, state_embeddings = policy(batch)
        probabilities = torch.softmax(tactic_logits[0], dim=-1)
        id_to_tactic = {value: key for key, value in bundle.tactic_vocab.items()}
        selected = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
        node_embeddings = policy.pointer.backbone.encode_nodes(batch)
        dag_rows = list(dag.nodes)
        local_positions = (
            data.premise_mask[0] if isinstance(data.premise_mask, list)
            else data.premise_mask
        ).nonzero(as_tuple=False).view(-1).tolist()
        pool = _build_source_candidates(
            bundle, node_embeddings[0], local_positions, goal
        )
        candidates: list[TacticCandidate] = []
        diagnostics: list[str] = []
        for tactic_id, probability in zip(selected.indices.tolist(), selected.values.tolist(), strict=False):
            tactic_name = id_to_tactic.get(int(tactic_id), "<UNK>")
            action = decode_action(
                policy, state_embeddings[0], int(tactic_id), pool
            )
            rendered, complete = self._render_action(dag, dag_rows, bundle, action)
            if not complete:
                diagnostics.append(
                    f"{tactic_name}: rendered "
                    f"{len(rendered)}/{len(action.arguments)} arguments; candidate dropped."
                )
                continue
            candidates.append(
                TacticCandidate(
                    tactic_name=tactic_name,
                    arguments=rendered,
                    probability=float(probability),
                )
            )
        for diagnostic in diagnostics:
            print(f"[joint-v2] {diagnostic}")
        return candidates

    @staticmethod
    def _render_action(
        dag: DAGBuilder,
        dag_rows: list,
        bundle,
        action: DecodedAction,
    ) -> tuple[List[str], bool]:
        """Render decoded identities into usable Lean references.

        Local identities map to per-graph node names; library identities map to
        corpus lemma names. Unresolvable identities make the whole action
        incomplete — placeholders are never stringified into arguments.
        """
        arguments: List[str] = []
        corpus = bundle.lemma_corpus or {}
        for argument in action.arguments:
            if argument.source is CandidateSource.LOCAL:
                if not 0 <= argument.stable_id < len(dag_rows):
                    return [], False
                node = dag_rows[argument.stable_id]
                if node.label == "Hyp" and node.children:
                    name_node = dag.nodes[node.children[0]]
                    if not name_node.label:
                        return [], False
                    arguments.append(name_node.label)
                else:
                    return [], False
            else:
                record = corpus.get(argument.stable_id)
                if record is None or not getattr(record, "name", ""):
                    return [], False
                arguments.append(record.name)
        return arguments, True
