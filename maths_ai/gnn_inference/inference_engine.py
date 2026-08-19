from pathlib import Path
from typing import List, Optional, cast

import torch

from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import build_model_from_checkpoint
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import (
    PremiseScorer,
    load_scorer_checkpoint,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    load_actor_critic_config,
    load_pointer_config,
    load_prepared_metadata,
)

from .model import GNNPredictor


class GNNModelEngine:
    def __init__(
        self,
        config_path: Path,
        tactic_predictor_model_path: Path,
        argument_predictor_model_path: Path,
        *,
        model_type: str = "pointer",
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

        if model_type == "actor_critic":
            config = load_actor_critic_config(config_path)
        else:
            config = load_pointer_config(config_path)

        metadata = load_prepared_metadata(config.prepared_root)

        tactic_checkpoint = torch.load(tactic_predictor_model_path, map_location=self.device, weights_only=False)
        expected_kind = "actor_critic_with_args" if model_type == "actor_critic" else "tactic_with_args"
        tactic_model, checkpoint_manifest, checkpoint_spec = build_model_from_checkpoint(
            tactic_checkpoint,
            node_vocab=metadata.node_vocab,
            tactic_vocab=metadata.tactic_vocab,
            expected_model_kind=expected_kind,
        )

        tactic_model = tactic_model.to(self.device)
        tactic_model.eval()

        argument_model = PremiseScorer(hidden_dim=checkpoint_spec.hidden_dim, mode=scorer_mode)
        argument_checkpoint = torch.load(argument_predictor_model_path, map_location=self.device, weights_only=False)
        expected_fingerprint = str(checkpoint_manifest["encoder_fingerprint"])
        load_scorer_checkpoint(
            argument_model,
            argument_checkpoint,
            expected_encoder_fingerprint=expected_fingerprint,
        )
        argument_model = argument_model.to(self.device)
        argument_model.eval()

        lemma_index = self._load_lemma_index(
            index_path,
            checkpoint_spec.hidden_dim,
            expected_fingerprint,
        )
        lemma_corpus = self._load_lemma_corpus(corpus_path)

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

    @staticmethod
    def _load_lemma_index(
        index_path: Optional[Path],
        hidden_dim: int,
        encoder_fingerprint: str,
    ) -> LemmaIndex:
        if index_path is not None and Path(index_path).exists():
            lemma_index = LemmaIndex.load(index_path)
            lemma_index.validate_encoder_fingerprint(encoder_fingerprint)
            return lemma_index

        import faiss
        import numpy as np

        return LemmaIndex(
            index=faiss.IndexFlatL2(hidden_dim),
            lemma_ids=[],
            lemma_vectors=np.empty((0, hidden_dim), dtype=np.float32),
            encoder_fingerprint=encoder_fingerprint,
        )

    @staticmethod
    def _load_lemma_corpus(corpus_path: Optional[Path]):
        if corpus_path is None or not Path(corpus_path).exists():
            return None

        records = load_lemma_corpus(corpus_path)
        return {record.lemma_id: record for record in records}

    def inference(self, goal_expression: str, top_k: int = 3) -> List[TacticCandidate]:
        """Predict ranked tactic candidates for ``goal_expression``.

        Contract (depended on by ``HybridReasoner.predict_next_tactic``):
        return up to ``top_k`` candidates, each a ``TacticCandidate``
        carrying the tactic family name, its selected argument/premise
        names, and the model's predicted probability — sorted by
        probability, descending.
        """
        predictions = self.gnn_inference.predict_tactics_with_arguments(goal_expression, top_k=top_k)
        print(*predictions, "\n")
        result =  [
            TacticCandidate(
                tactic_name=str(prediction["tactic_name"]),
                arguments=[str(argument) for argument in cast(list, prediction["selected_arguments"])],
                probability=float(cast(float, prediction["probability"])),
            )
            for prediction in predictions
        ]

        for res in result:
            if res.tactic_name == "rw":
                res.arguments = "[" + ",".join(res.arguments) + "]"

        return result
