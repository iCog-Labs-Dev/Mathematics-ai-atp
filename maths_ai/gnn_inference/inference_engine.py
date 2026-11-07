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
from maths_ai.gnn_inference.atp_lean_gnn.model_graph import goal_to_model_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_premise_mask, dag_to_pyg
from maths_ai.gnn_inference.atp_lean_gnn.labels import get_tactic_arity
from maths_ai.gnn_inference.atp_lean_gnn.training import transform_edge_index
from torch_geometric.data import Batch

from .model import GNNPredictor


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
        if self._joint_bundle is None:
            raise RuntimeError("v2 inference bundle is not initialized.")
        dag = goal_to_model_dag(goal)
        data = dag_to_pyg(dag, self._joint_bundle.node_vocab)
        state_idx = next((node.id for node in dag.nodes if node.label == "State"), 0)
        data.state_node_index = torch.tensor([state_idx], dtype=torch.long)
        data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
        data = data.to(self.device)
        data.edge_index = transform_edge_index(data.edge_index, edge_mode="bidirectional")
        batch = Batch.from_data_list([data])
        tactic_logits, _, _, arg_count_logits, _ = self._joint_bundle.policy(batch)
        probabilities = torch.softmax(tactic_logits[0], dim=-1)
        id_to_tactic = {value: key for key, value in self._joint_bundle.tactic_vocab.items()}
        selected = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
        candidates: list[TacticCandidate] = []
        local_nodes = {
            node.id: node
            for node in dag.nodes
            if node.label == "Hyp" and node.children
        }
        for tactic_id, probability in zip(selected.indices.tolist(), selected.values.tolist(), strict=False):
            tactic_name = id_to_tactic.get(int(tactic_id), "<UNK>")
            arity = min(get_tactic_arity(tactic_name), int(self._joint_bundle.policy.pointer.max_args))
            if arity == 0:
                candidates.append(
                    TacticCandidate(
                        tactic_name=tactic_name,
                        arguments=[],
                        probability=float(probability),
                    )
                )
                continue
            conditioned = torch.tensor([int(tactic_id)], device=self.device)
            _, arg_logits, stop_logits, count_logits, _ = self._joint_bundle.policy(
                batch, tactic_ids=conditioned
            )
            predicted_count = int(count_logits[0].argmax().item())
            count = min(arity, predicted_count, len(arg_logits))
            arguments: list[str] = []
            for step in range(count):
                position = int(arg_logits[step][0].argmax().item())
                node_ids = [
                    node.id
                    for node in dag.nodes
                    if node.id < data.num_nodes
                ]
                if position >= len(node_ids):
                    break
                node = dag.nodes[node_ids[position]]
                if node.id in local_nodes:
                    name_node = dag.nodes[node.children[0]]
                    arguments.append(name_node.label)
            if count and len(arguments) < count:
                continue
            candidates.append(
                TacticCandidate(
                    tactic_name=tactic_name,
                    arguments=arguments,
                    probability=float(probability),
                )
            )
        return candidates
