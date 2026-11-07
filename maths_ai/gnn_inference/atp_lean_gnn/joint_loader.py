"""Neural construction for the published Mathlib-Sexpr-GNN v2 bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .argument_selector import JointSourcePolicy
from .joint_bundle import JointBundle, load_joint_source_bundle
from .lemma_corpus import load_lemma_corpus
from .lemma_index import LemmaIndex


def mean_pool_embeddings(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Pool encoder tokens using the release's attention-mask policy."""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return torch.nn.functional.normalize(pooled, p=2, dim=-1) if normalize else pooled


@dataclass
class JointNeuralBundle:
    """Constructed v2 policy and text-retrieval components."""

    artifact: JointBundle
    policy: JointSourcePolicy
    tokenizer: Any
    text_encoder: Any
    lemma_index: LemmaIndex
    lemma_corpus: dict[int, Any]
    device: torch.device

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode goal-first state text with the published retriever policy."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(self.artifact.retriever_config["max_length"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.text_encoder(**encoded)
        return mean_pool_embeddings(
            outputs.last_hidden_state,
            encoded["attention_mask"],
            normalize=bool(self.artifact.retriever_config.get("normalize", True)),
        )


def _pointer_config(bundle: JointBundle) -> dict[str, Any]:
    config = bundle.policy_config.get("pointer_config")
    if not isinstance(config, dict):
        raise ValueError("Joint bundle policy is missing pointer_config metadata.")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("Joint bundle policy is missing model architecture metadata.")
    return config


def _load_policy(bundle: JointBundle, device: torch.device) -> JointSourcePolicy:
    config = _pointer_config(bundle)
    model_config = config["model"]
    node_vocab_path = bundle.root / "vocab" / "node_vocab.json"
    tactic_vocab_path = bundle.root / "vocab" / "tactic_vocab.json"
    import json

    node_vocab = json.loads(node_vocab_path.read_text(encoding="utf-8"))
    tactic_vocab = json.loads(tactic_vocab_path.read_text(encoding="utf-8"))
    model = JointSourcePolicy(
        num_node_labels=len(node_vocab),
        num_tactics=len(tactic_vocab),
        num_node_types=len(bundle.policy_config["input_contract"]["node_type_to_id"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        heads=int(model_config["heads"]),
        dropout=float(model_config["dropout"]),
        use_node_type=bool(config["use_node_type"]),
        max_args=int(model_config["max_args"]),
    )
    weights = load_file(str(bundle.root / "policy" / "model.safetensors"), device=str(device))
    model.load_state_dict(weights)
    return model.to(device).eval()


def _load_retriever(bundle: JointBundle, device: torch.device) -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer, BertConfig
    except ImportError as exc:
        raise RuntimeError("Transformers is required to load the v2 text retriever.") from exc

    tokenizer_root = bundle.root / "retriever" / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True)
    config = bundle.retriever_config.get("encoder_config")
    if not isinstance(config, dict):
        raise ValueError("Joint bundle retriever is missing encoder_config metadata.")
    encoder = AutoModel.from_config(BertConfig.from_dict(config))
    weights = load_file(str(bundle.root / "retriever" / "model.safetensors"), device=str(device))
    encoder.load_state_dict(weights)
    return tokenizer, encoder.to(device).eval()


def load_joint_neural_bundle(
    root: str | Path,
    *,
    expected_revision: str | None = None,
    device: str | torch.device = "cpu",
    require_live_parity: bool = False,
) -> JointNeuralBundle:
    """Validate and construct every neural component in the v2 release."""
    artifact = load_joint_source_bundle(
        root,
        expected_revision=expected_revision,
        require_live_parity=require_live_parity,
    )
    resolved_device = torch.device(device)
    policy = _load_policy(artifact, resolved_device)
    tokenizer, text_encoder = _load_retriever(artifact, resolved_device)
    index = LemmaIndex.load(
        artifact.root / "retriever" / "index",
        normalize_queries=bool(artifact.retriever_config.get("normalize", False)),
    )
    corpus = {
        record.lemma_id: record
        for record in load_lemma_corpus(artifact.root / "retriever" / "corpus" / "lemmas.jsonl")
    }
    return JointNeuralBundle(
        artifact=artifact,
        policy=policy,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        lemma_index=index,
        lemma_corpus=corpus,
        device=resolved_device,
    )
