"""Portable pre-template joint-policy release with its matched text retrieval system."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .bundle import file_sha256
from .joint_source_decoder import JointSourceDecoder, SourceCandidates, candidate_groups_from_batch, local_type_signals
from .lemma_index import LemmaIndex, state_dict_sha256
from .text_retriever import TextDualEncoder, goal_first_text, tokenize_texts
from .pyg import NODE_TYPE_TO_ID
from .training import PointerConfig, PreparedMetadata, build_pointer_model


FORMAT = "joint_source_system_v1"
GRAPH_FORMAT = "lean-model-sexp-v2"
INDEX_FILES = ("faiss.index", "lemma_vectors.npy", "lemma_ids.json", "lemma_names.json", "manifest.json")
LOCAL_CUES = {"exact": 8.0, "rw": 8.0, "simp": 4.0, "subst": 8.0}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _vocab(path):
    value = _json(path)
    if (not isinstance(value, dict) or not value or any(type(i) is not int for i in value.values())
            or sorted(value.values()) != list(range(len(value)))):
        raise ValueError(f"vocabulary must have unique contiguous IDs: {path}")
    return value


def _tensor_state(model):
    return {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}


def _tokenizer_hash(tokenizer):
    return hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _public_name(name):
    return "local_encoder" if Path(name).is_absolute() or name.startswith(".") else name


def _encoder_config(value):
    result = copy.deepcopy(value)
    result.pop("_name_or_path", None)
    result.pop("name_or_path", None)
    return result


def _build_retriever(config):
    from transformers import AutoConfig, AutoModel

    encoder = copy.deepcopy(config["encoder_config"])
    model_type = encoder.pop("model_type")
    return TextDualEncoder(AutoModel.from_config(AutoConfig.for_model(model_type, **encoder)))


def _build_policy(config, node_vocab, tactic_vocab, text_dim):
    metadata = PreparedMetadata.from_vocabs(node_vocab=node_vocab, tactic_vocab=tactic_vocab)
    pointer = build_pointer_model(metadata, PointerConfig.from_dict(config["pointer_config"]))
    return JointSourceDecoder(pointer, text_dim=text_dim)


def _validate_index(index, retriever, tokenizer, config, corpus_path):
    import faiss

    expected = {"model_type": "pretrained_text_dual_encoder_index_v1", "model_name": config["model_name"],
                "tokenizer_sha256": _tokenizer_hash(tokenizer), "state_text_policy": "goal_first_v1",
                "corpus_sha256": file_sha256(corpus_path),
                "lemma_encoder_sha256": state_dict_sha256(retriever.lemma_encoder.state_dict()),
                "max_length": config["max_length"], "normalize": True}
    if index.manifest != expected:
        raise ValueError("retriever/tokenizer/corpus/index manifest binding mismatch")
    size = len(index.lemma_ids)
    if (not size or len(set(index.lemma_ids)) != size or len(index.lemma_names) != size
            or len(set(index.lemma_names)) != size or index.index.ntotal != size
            or index.index.metric_type != faiss.METRIC_INNER_PRODUCT or index.lemma_vectors.dtype != np.float32
            or index.lemma_vectors.shape != (size, config["embedding_dim"])):
        raise ValueError("invalid FAISS/vector/ID/name population or metric")
    # Same dimensions are not enough: score and GRU feedback must use the exact
    # same rows, and the corpus ID/name mapping must identify those rows.
    for start in range(0, size, 8192):
        vectors = index.lemma_vectors[start:start + 8192]
        if (not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
                or not np.array_equal(index.index.reconstruct_n(start, len(vectors)), vectors)):
            raise ValueError("FAISS rows differ from normalized lemma_vectors.npy")
    names = dict(zip(index.lemma_ids, index.lemma_names))
    seen, corpus_names = set(), set()
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            identity, name = record["lemma_id"], record["name"]
            if (type(identity) is not int or identity in seen or name in corpus_names
                    or names.get(identity) != name or not isinstance(record.get("statement"), str)
                    or not record["statement"].strip()):
                raise ValueError("lemma corpus differs from index ID/name mapping")
            seen.add(identity)
            corpus_names.add(name)
    if seen != set(names):
        raise ValueError("lemma corpus/index population mismatch")


@dataclass
class LoadedJointSourceBundle:
    root: Path
    policy: JointSourceDecoder
    retriever: TextDualEncoder
    tokenizer: object
    lemma_index: LemmaIndex
    node_vocab: dict
    tactic_vocab: dict
    manifest: dict
    retrieval_config: dict
    inference_config: dict
    profile: str

    @torch.no_grad()
    def retrieve(self, states, *, k=None):
        k = self.inference_config["retrieval_k"] if k is None else k
        if not isinstance(states, (list, tuple)) or not states or not all(isinstance(state, str) and state.strip() for state in states):
            raise ValueError("nonempty original proof-state texts required")
        if type(k) is not int or not 1 <= k <= len(self.lemma_index.lemma_ids):
            raise ValueError("retrieval K outside corpus bounds")
        device = next(self.retriever.parameters()).device
        tokens = tokenize_texts(self.tokenizer, [goal_first_text(state) for state in states],
                                device=device, max_length=self.retrieval_config["max_length"])
        return self.lemma_index.search(self.retriever.encode_states(tokens), k=k)

    @torch.no_grad()
    def predict(self, batch, states, *, graph_format, normalized_records=None, k=None):
        """Predict from normalized prepared PyG graphs plus their original text.

        This does not reconstruct graphs from printed Lean goals or establish
        live Pantograph graph parity. Callers must supply the declared graph form.
        """
        if graph_format != GRAPH_FORMAT:
            raise ValueError("joint model requires normalized model-S-expression graphs")
        if not isinstance(states, (list, tuple)):
            raise ValueError("state texts must be an ordered list or tuple")
        count = int(batch.num_graphs)
        if len(states) != count:
            raise ValueError("one original state text required per graph")
        if self.profile == "accepted_local_cues" and (normalized_records is None or len(normalized_records) != count):
            raise ValueError("local cue profile requires one normalized goal/hypothesis record per graph")
        device = next(self.policy.parameters()).device
        batch = batch.clone().to(device)
        batch.y = torch.zeros(count, dtype=torch.long, device=device)  # Pool grouping needs count, not gold tactic labels.
        nodes = self.policy.pointer.backbone.encode_nodes(batch)
        pooled = self.policy.pointer.backbone.readout(nodes, batch)
        logits = self.policy.pointer.backbone.classifier(pooled)
        ids, vectors, scores = self.retrieve(states, k=k)
        pools = candidate_groups_from_batch(batch, nodes, ids, vectors, scores)
        id_to_tactic = {identity: name for name, identity in self.tactic_vocab.items()}
        id_to_node = {identity: name for name, identity in self.node_vocab.items()}
        output = []
        for row, pool in enumerate(pools):
            tactic_id = int(logits[row].argmax())
            tactic = id_to_tactic[tactic_id]
            if self.profile == "accepted_local_cues":
                record = normalized_records[row]
                start, end = int(batch.ptr[row]), int(batch.ptr[row + 1])
                signal = local_type_signals(pool, batch.x[start:end], id_to_node,
                    tactic_name=tactic, goal_sexp=record["goal_sexp"], hyp_sexps=record["hyp_sexps"])
                pool = replace(pool, local_type_signal=signal, local_type_tactic=tactic)
            arguments, complete = self.policy.decode(pooled[row], tactic_id, pool)
            output.append({"tactic_id": tactic_id, "tactic": tactic, "complete": complete, "arguments": [
                {"source": argument.source.value, "identity": argument.stable_id} for argument in arguments]})
        return output


def load_joint_source_bundle(root, *, device="cpu", profile="base"):
    """Load a complete clean-root release offline; no training paths or pickle."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    root = Path(root).resolve()
    manifest = _json(root / "bundle.json")
    if manifest.get("format") != FORMAT or manifest.get("model_type") != "joint_source_flat_decoder_v2":
        raise ValueError("not a supported joint-source system bundle")
    for relative, digest in manifest["files"].items():
        member = PurePosixPath(relative)
        path = root / relative
        # Hub snapshots legitimately symlink files to a sibling blob cache.
        # Validate logical member paths and bytes, not their physical target root.
        if member.is_absolute() or ".." in member.parts or file_sha256(path) != digest:
            raise ValueError(f"bundle file missing, outside root or changed: {relative}")
    required = {"policy/model.safetensors", "policy/config.json", "policy/inference.json", "vocab/node_vocab.json",
                "vocab/tactic_vocab.json", "retriever/model.safetensors", "retriever/config.json", "retriever/corpus/lemmas.jsonl",
                *("retriever/index/" + name for name in INDEX_FILES)}
    if not required.issubset(manifest["files"]) or not any(name.startswith("retriever/tokenizer/") for name in manifest["files"]):
        raise ValueError("bundle manifest omits a required component")
    tokenizer_files = {str(path.relative_to(root)) for path in (root / "retriever/tokenizer").rglob("*") if path.is_file()}
    if tokenizer_files != {name for name in manifest["files"] if name.startswith("retriever/tokenizer/")}:
        raise ValueError("unmanifested/missing tokenizer files")
    node_vocab, tactic_vocab = (_vocab(root / "vocab" / name) for name in ("node_vocab.json", "tactic_vocab.json"))
    policy_config, retrieval_config = _json(root / "policy/config.json"), _json(root / "retriever/config.json")
    if (policy_config["input_contract"]["graph_format"] != GRAPH_FORMAT
            or policy_config["input_contract"]["node_type_to_id"] != NODE_TYPE_TO_ID):
        raise ValueError("unsupported graph-input contract")
    if (retrieval_config.get("model_type") != "pretrained_text_dual_encoder"
            or retrieval_config.get("pooling") != "attention_mask_mean" or retrieval_config.get("normalize") is not True
            or type(retrieval_config.get("max_length")) is not int or retrieval_config["max_length"] < 1):
        raise ValueError("unsupported retriever encoding contract")
    retriever = _build_retriever(retrieval_config)
    retriever.load_state_dict(load_file(str(root / "retriever/model.safetensors")), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(root / "retriever/tokenizer", local_files_only=True, trust_remote_code=False)
    index = LemmaIndex.load(root / "retriever/index")
    _validate_index(index, retriever, tokenizer, retrieval_config, root / "retriever/corpus/lemmas.jsonl")
    policy = _build_policy(policy_config, node_vocab, tactic_vocab, index.lemma_vectors.shape[1])
    policy.load_state_dict(load_file(str(root / "policy/model.safetensors")), strict=True)
    inference = _json(root / "policy/inference.json")
    if profile not in inference["profiles"] or inference["argument_count_policy"] != "predicted_count":
        raise ValueError("unsupported inference profile/decoding policy")
    if (inference["max_args"] != policy.pointer.max_args or type(inference["retrieval_k"]) is not int
            or not 1 <= inference["retrieval_k"] <= len(index.lemma_ids) or inference.get("allow_repeated_arguments") is not True
            or manifest["policy_state_sha256"] != state_dict_sha256(policy.state_dict())
            or manifest["retriever_state_sha256"] != state_dict_sha256(retriever.state_dict())):
        raise ValueError("bundle architecture/runtime/tensor binding mismatch")
    policy.local_type_bonus_strengths = inference["profiles"][profile]["local_type_bonus_strengths"]
    policy.to(device).eval().requires_grad_(False)
    retriever.to(device).eval().requires_grad_(False)
    return LoadedJointSourceBundle(root, policy, retriever, tokenizer, index, node_vocab, tactic_vocab,
                                    manifest, retrieval_config, inference, profile)


def export_joint_source_bundle(*, joint_checkpoint, local_checkpoint, text_checkpoint, text_index,
                               prepared_root, candidate_dir, corpus_path, output_dir, source_commit,
                               license_file, license_id, tokenizer_source=None, progress=None):
    """Export accepted stages as one weight set, keeping original inputs read-only."""
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    def announce(message):
        if progress is not None:
            progress(message)

    announce("Validating source checkpoint lineage and prepared-data bindings")
    joint_checkpoint, local_checkpoint, text_checkpoint, text_index, prepared_root, candidate_dir, corpus_path, output = (
        Path(path).resolve() for path in (joint_checkpoint, local_checkpoint, text_checkpoint, text_index,
                                         prepared_root, candidate_dir, corpus_path, output_dir))
    for source in (prepared_root, candidate_dir, text_index, joint_checkpoint.parent, local_checkpoint.parent,
                   text_checkpoint.parent, corpus_path.parent):
        if output == source or source in output.parents:
            raise ValueError("export must be outside source artifact roots")
    if output.exists():
        raise FileExistsError(f"export output already exists: {output}")
    license_file = Path(license_file).resolve()
    if not license_file.is_file() or not license_file.read_text(encoding="utf-8").strip() or not license_id:
        raise ValueError("explicit release license text and identifier required")
    checkpoint_hashes = {"joint": file_sha256(joint_checkpoint), "local": file_sha256(local_checkpoint),
                         "retriever": file_sha256(text_checkpoint)}
    joint = torch.load(joint_checkpoint, map_location="cpu", weights_only=False)
    local = torch.load(local_checkpoint, map_location="cpu", weights_only=False)
    text = torch.load(text_checkpoint, map_location="cpu", weights_only=False)
    if (joint.get("model_type") != "joint_source_flat_decoder_v2" or joint.get("phase") != "complete"
            or local.get("model_type") != "frozen_graph_local_identity_v1" or local.get("phase") != "complete"
            or text.get("model_type") != "pretrained_text_dual_encoder" or text.get("phase") != "complete"):
        raise ValueError("completed joint/local/text checkpoints required")
    binding, local_binding = joint["binding"], local["binding"]
    candidate = _json(candidate_dir / "manifest.json")
    index_manifest = _json(text_index / "manifest.json")
    text_sha, candidate_sha = file_sha256(text_checkpoint), file_sha256(candidate_dir / "manifest.json")
    if (local_binding["joint_checkpoint_sha256"] != file_sha256(joint_checkpoint)
            or any(owner["text_checkpoint_sha256"] != text_sha or owner["text_index_manifest"] != index_manifest
                   or owner["candidate_manifest_sha256"] != candidate_sha for owner in (binding, local_binding))
            or candidate["text_checkpoint_sha256"] != text_sha or candidate["text_index_manifest"] != index_manifest
            or candidate["corpus_sha256"] != file_sha256(corpus_path) or text["corpus_sha256"] != file_sha256(corpus_path)
            or candidate["text_epoch"] != text["epoch"] or candidate["k"] < binding["k"]
            or candidate["query_policy"] != "goal_first_v1"
            or candidate["selection_policy"] != "exact_cosine_no_gold_injection"):
        raise ValueError("checkpoint lineage or matched retrieval/candidate binding mismatch")
    if (any(owner["prepared_root"] != candidate["prepared_root"] for owner in (binding, local_binding))
            or local_binding.get("trained_components") != "local_pointer_out_proj_only"):
        raise ValueError("joint/local prepared population or refinement policy differs")
    for field, relative in (("node_vocab_sha256", "vocab/node_vocab.json"), ("tactic_vocab_sha256", "vocab/tactic_vocab.json"),
                            ("argument_label_binding_sha256", "argument_labels.json"),
                            ("prepared_packed_manifest_sha256", "packed/bidirectional/manifest.json")):
        if candidate[field] != file_sha256(prepared_root / relative):
            raise ValueError(f"prepared vocabulary/label population differs: {field}")
    source_state, refined_state = joint["model_state_dict"], local["model_state_dict"]
    if (set(source_state) != set(refined_state)
            or any(not torch.equal(source_state[name], refined_state[name]) for name in source_state
                   if not name.startswith("pointer.argument_selector.out_proj."))):
        raise ValueError("local checkpoint changed components outside declared local projection")
    node_vocab, tactic_vocab = (_vocab(prepared_root / "vocab" / name) for name in ("node_vocab.json", "tactic_vocab.json"))
    pointer_config = {key: copy.deepcopy(value) for key, value in joint["config"].items()
                      if key in {"seed", "edge_mode", "use_node_type", "gnn_type", "max_args", "arg_loss_weight", "model"}}
    pointer_config.update(prepared_root=".", run_root=".", device="cpu", initialization_checkpoint=None)
    policy_config = {"model_type": "joint_source_flat_decoder_v2", "pointer_config": pointer_config,
        "input_contract": {"graph_format": GRAPH_FORMAT, "model_expression_version": 1,
                           "node_type_to_id": NODE_TYPE_TO_ID,
                           "edge_mode": pointer_config["edge_mode"], "original_state_text_required": True,
                           "live_pantograph_parity_verified": False}}
    source_name = tokenizer_source or text["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(source_name, local_files_only=True)
    encoder_config = AutoConfig.from_pretrained(source_name, local_files_only=True)
    retriever = TextDualEncoder(AutoModel.from_config(encoder_config))
    retriever.load_state_dict(text["model_state_dict"], strict=True)
    if any(value.dtype != retriever.state_dict()[name].dtype or not torch.equal(value, retriever.state_dict()[name])
           for name, value in text["model_state_dict"].items()):
        raise ValueError("retriever construction changed checkpoint tensor precision")
    retriever.eval()
    if _tokenizer_hash(tokenizer) != text["tokenizer_sha256"] or text["state_text_policy"] != "goal_first_v1":
        raise ValueError("source tokenizer/state-text policy differs")
    retrieval_config = {"model_type": "pretrained_text_dual_encoder", "model_name": text["model_name"],
        "encoder_config": _encoder_config(encoder_config.to_dict()), "max_length": text["max_length"],
        "embedding_dim": int(encoder_config.hidden_size), "pooling": "attention_mask_mean", "normalize": True}
    index = LemmaIndex.load(text_index)
    announce("Checking retriever/index/corpus compatibility and every vector row")
    _validate_index(index, retriever, tokenizer, retrieval_config, corpus_path)
    policy = _build_policy(policy_config, node_vocab, tactic_vocab, index.lemma_vectors.shape[1])
    policy.load_state_dict(refined_state, strict=True)
    if any(value.dtype != policy.state_dict()[name].dtype or not torch.equal(value, policy.state_dict()[name])
           for name, value in refined_state.items()):
        raise ValueError("policy construction changed checkpoint tensor precision")
    policy.eval()
    if not 1 <= binding["k"] <= len(index.lemma_ids):
        raise ValueError("joint retrieval budget outside corpus")
    # Stage in a new sibling, then publish the directory only after load-back.
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"unfinished export exists: {building}")
    building.mkdir(parents=True)
    announce("Writing clean-root policy, retrieval components, vocabularies and corpus")
    for name in ("policy", "vocab", "retriever/index", "retriever/corpus", "retriever/tokenizer", "evaluation", "provenance"):
        (building / name).mkdir(parents=True, exist_ok=True)
    save_file(_tensor_state(policy), str(building / "policy/model.safetensors"))
    save_file(_tensor_state(retriever), str(building / "retriever/model.safetensors"))
    _write(building / "policy/config.json", policy_config)
    _write(building / "policy/inference.json", {"argument_count_policy": "predicted_count", "max_args": policy.pointer.max_args,
        "retrieval_k": binding["k"], "allow_repeated_arguments": True, "default_profile": "base",
        "profiles": {"base": {"local_type_bonus_strengths": {}},
                     "accepted_local_cues": {"local_type_bonus_strengths": LOCAL_CUES,
                         "requires_normalized_goal_and_hypotheses": True, "scope": "first_argument_only"}}})
    retrieval_config["model_name"] = _public_name(text["model_name"])
    _write(building / "retriever/config.json", retrieval_config)
    tokenizer.save_pretrained(building / "retriever/tokenizer")
    for path in (building / "retriever/tokenizer").glob("*.json"):
        value = _json(path)
        if path.name == "tokenizer_config.json":
            for key in ("name_or_path", "_name_or_path", "tokenizer_file", "vocab_file"):
                if isinstance(value.get(key), str) and Path(value[key]).is_absolute():
                    value.pop(key)
            _write(path, value)
    for name in ("node_vocab.json", "tactic_vocab.json"):
        shutil.copyfile(prepared_root / "vocab" / name, building / "vocab" / name)
    for name in INDEX_FILES:
        shutil.copyfile(text_index / name, building / "retriever/index" / name)
    published_index = dict(index_manifest, model_name=retrieval_config["model_name"])
    _write(building / "retriever/index/manifest.json", published_index)
    shutil.copyfile(corpus_path, building / "retriever/corpus/lemmas.jsonl")
    _write(building / "evaluation/metrics.json", {"joint_training_checkpoint": joint.get("val_metrics"),
        "local_refinement_checkpoint": local.get("val_metrics"), "retriever_checkpoint": text.get("val_metrics"),
        "full_system_reevaluated": False})
    _write(building / "evaluation/protocol.json", {"metric_scope": "offline stage diagnostics; denominators differ",
        "default_profile": "base", "optional_cues": "accepted_local_cues requires normalized sidecars",
        "template_prediction_trained": False, "whole_proof_success_measured": False,
        "prepared_population": {split: entry["count"] for split, entry in _json(prepared_root / "packed/bidirectional/manifest.json")["splits"].items()}})
    _write(building / "provenance/training.json", {"joint_epoch": joint["epoch"], "local_refinement_epoch": local["epoch"],
        "retriever_epoch": text["epoch"], "local_trained_components": local_binding["trained_components"],
        "source_checkpoint_sha256": checkpoint_hashes,
        "source_candidate_manifest_sha256": candidate_sha,
        "prepared_binding_sha256": {key: candidate[key] for key in (
            "node_vocab_sha256", "tactic_vocab_sha256", "argument_label_binding_sha256", "prepared_packed_manifest_sha256")}})
    import transformers
    import safetensors
    _write(building / "provenance/versions.json", {"source_code_commit": source_commit, "torch": torch.__version__,
        "transformers": transformers.__version__, "safetensors": safetensors.__version__,
        "exporter_sha256": file_sha256(Path(__file__)),
        "dataset_name": text["dataset_name"], "corpus_sha256": text["corpus_sha256"]})
    (building / "README.md").write_text(_model_card(joint["epoch"], local["epoch"], text["epoch"], license_id), encoding="utf-8")
    shutil.copyfile(license_file, building / "LICENSE")
    (building / "NOTICE").write_text(
        "Mathlib-Sexpr-GNN model-artifact release.\n"
        "Learned model artifacts use the license in LICENSE.\n"
        "The separate Mathematics-ai-atp implementation retains its GPLv2 source-code license.\n"
        "Source: https://github.com/iCog-Labs-Dev/Mathematics-ai-atp\n"
        "Base encoder: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0).\n"
        "Declaration corpus derives from Mathlib4 (Apache-2.0) and LeanDojo benchmark provenance (MIT).\n"
        "Mathlib: https://github.com/leanprover-community/mathlib4\n"
        "LeanDojo: https://github.com/lean-dojo/LeanDojo\n", encoding="utf-8")
    manifest = {"format": FORMAT, "model_type": "joint_source_flat_decoder_v2", "weights_format": "safetensors",
        "license_id": license_id,
        "policy_state_sha256": state_dict_sha256(policy.state_dict()), "retriever_state_sha256": state_dict_sha256(retriever.state_dict()),
        "files": {str(path.relative_to(building)): file_sha256(path) for path in sorted(building.rglob("*")) if path.is_file()}}
    _write(building / "bundle.json", manifest)
    announce("Reloading exported bundle offline and comparing tensors, retrieval and decoder outputs")
    loaded = load_joint_source_bundle(building)
    for expected_model, actual in ((policy, loaded.policy), (retriever, loaded.retriever)):
        if any(not torch.equal(tensor, actual.state_dict()[name]) for name, tensor in expected_model.state_dict().items()):
            raise ValueError("exported tensors changed on round-trip")
    probe = ["⊢ True", "n : Nat\n⊢ n = n"]
    tokens = tokenize_texts(tokenizer, [goal_first_text(value) for value in probe], device=torch.device("cpu"), max_length=text["max_length"])
    with torch.no_grad():
        expected_query = retriever.encode_states(tokens)
        actual_query = loaded.retriever.encode_states(tokens)
    torch.testing.assert_close(expected_query, actual_query, rtol=0, atol=0)
    expected_ids, _, expected_scores = index.search(expected_query, k=min(binding["k"], 8))
    actual_ids, _, actual_scores = loaded.retrieve(probe, k=min(binding["k"], 8))
    if expected_ids != actual_ids or not np.allclose(expected_scores, actual_scores, rtol=0, atol=1e-6):
        raise ValueError("exported retrieval results changed on round-trip")
    local_vectors = torch.arange(2 * policy.pointer.hidden_dim, dtype=torch.float32).reshape(2, -1) / 100
    selected_ids, selected_vectors, selected_scores = index.search(expected_query[:1], k=min(binding["k"], 8))
    candidates = SourceCandidates((0, 1), local_vectors, tuple(selected_ids[0]),
                                 torch.from_numpy(selected_vectors[0]), torch.from_numpy(selected_scores[0]))
    state = torch.linspace(-1, 1, policy.pointer.hidden_dim)
    with torch.no_grad():
        for tactic in range(min(3, len(tactic_vocab))):
            torch.testing.assert_close(policy.count_logits(state, tactic), loaded.policy.count_logits(state, tactic), rtol=0, atol=0)
            if policy.decode(state, tactic, candidates) != loaded.policy.decode(state, tactic, candidates):
                raise ValueError("joint GRU/count/source/identity decoding changed on round-trip")
    if checkpoint_hashes != {"joint": file_sha256(joint_checkpoint), "local": file_sha256(local_checkpoint),
                             "retriever": file_sha256(text_checkpoint)}:
        raise ValueError("source checkpoints changed during export")
    building.rename(output)
    announce(f"Verified export ready: {output}")
    return manifest


def _model_card(joint_epoch, local_epoch, text_epoch, license_id):
    return f'''---
license: {license_id}
library_name: pytorch
tags: [lean-4, mathlib, theorem-proving, gnn, tactic-prediction, premise-retrieval]
language: [en]
---

# Mathlib joint-source GNN + GRU

Complete pre-template research system: normalized graph encoder, tactic head,
GRU argument decoder, explicit argument-count head, per-position local/library
source gate, local selector and contextual library scoring. Maximum three flat
arguments. Weights include joint stage epoch {joint_epoch}, local-only refinement
epoch {local_epoch}, and matched dual-encoder/index epoch {text_epoch}.

## Loading

```python
from huggingface_hub import snapshot_download
from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle
path = snapshot_download("iCogLabs-Dev/Mathlib-Sexpr-GNN", revision="<release-tag-or-commit>")
system = load_joint_source_bundle(path, device="cpu")
```

`system.policy`, `system.retriever`, `system.lemma_index`, and vocabularies are
available separately. `retrieve(original_state_texts)` performs live text
retrieval. `predict(batch, original_state_texts, graph_format="{GRAPH_FORMAT}")`
expects correctly normalized prepared PyG graphs and their original state text.
It does not convert arbitrary printed Lean goals into compatible graphs.
The optional `accepted_local_cues` profile requires normalized goal/hypothesis
records; base inference keeps those runtime-only cues off.

## Reproducibility and limitations

`bundle.json` binds every file. Retrieval validates tower/tokenizer/corpus hashes,
the ID/name population and equality of FAISS rows with the vector matrix.
No training-machine paths or optimizer checkpoints are needed for loading.
Source-code/dependency versions and training stages are in `provenance/`.
The implementation's separate GPLv2 code license is described in `NOTICE`.
Stage diagnostics are in `evaluation/`; their cohorts differ and they are not
a re-evaluation of this whole export. This is an offline experimental system,
not a measured whole-theorem prover or a model trained on the newer template
representations. Live normalized-state serving parity remains a separate gate.
Pin a release tag or full Hugging Face commit for reproducible downloads.
'''
