"""Full joint-system export must survive removal of every training artifact."""

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from maths_ai.gnn_inference.atp_lean_gnn import joint_bundle
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import state_dict_sha256
from maths_ai.gnn_inference.atp_lean_gnn.text_retriever import TextDualEncoder, tokenize_texts
from maths_ai.gnn_inference.scripts.publish_joint_bundle import publish_joint_bundle


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path):
    import faiss
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    torch.manual_seed(37)
    source = tmp_path / "source"
    encoder_dir, prepared, index_dir, candidate_dir = (source / name for name in ("encoder", "prepared", "index", "candidates"))
    vocab = {word: i for i, word in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "true", "nat", "goal", "context", "n", "="])}
    backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                       cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    tokenizer.save_pretrained(encoder_dir)
    encoder_config = BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=64, hidden_dropout_prob=0,
        attention_probs_dropout_prob=0)
    encoder_config.save_pretrained(encoder_dir)
    retriever = TextDualEncoder(BertModel(encoder_config)).eval()
    records = [{"lemma_id": i, "name": f"A.lemma{i}", "statement": "True" if i % 2 else "Nat", "namespace": "A", "module": "A"}
               for i in range(4)]
    corpus = source / "lemmas.jsonl"
    corpus.write_text("".join(json.dumps(record) + "\n" for record in records))
    with torch.no_grad():
        vectors = retriever.encode_lemmas(tokenize_texts(tokenizer, [f"{r['name']} : {r['statement']}" for r in records],
            device=torch.device("cpu"), max_length=32)).numpy().copy()
    index_dir.mkdir()
    index = faiss.IndexFlatIP(8)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    np.save(index_dir / "lemma_vectors.npy", vectors)
    write(index_dir / "lemma_ids.json", [record["lemma_id"] for record in records])
    write(index_dir / "lemma_names.json", [record["name"] for record in records])
    tokenizer_sha = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"model_type": "pretrained_text_dual_encoder_index_v1", "model_name": str(encoder_dir),
        "tokenizer_sha256": tokenizer_sha, "state_text_policy": "goal_first_v1", "corpus_sha256": file_sha256(corpus),
        "lemma_encoder_sha256": state_dict_sha256(retriever.lemma_encoder.state_dict()), "max_length": 32, "normalize": True}
    write(index_dir / "manifest.json", manifest)
    text_checkpoint = source / "text.pt"
    torch.save({"model_type": "pretrained_text_dual_encoder", "phase": "complete", "model_name": str(encoder_dir),
        "model_state_dict": retriever.state_dict(), "epoch": 5, "corpus_sha256": file_sha256(corpus),
        "tokenizer_sha256": tokenizer_sha, "max_length": 32, "state_text_policy": "goal_first_v1",
        "dataset_name": "fixture", "val_metrics": {"recall_at_200": 0.5}}, text_checkpoint)
    node_vocab = {"<UNK>": 0, "State": 1, "FV0": 2, "Hyp": 3, "True": 4, "Nat": 5}
    tactic_vocab = {"<UNK_TACTIC>": 0, "exact": 1, "rw": 2}
    write(prepared / "vocab/node_vocab.json", node_vocab)
    write(prepared / "vocab/tactic_vocab.json", tactic_vocab)
    write(prepared / "argument_labels.json", {"policy": "flat_reference_v2"})
    write(prepared / "packed/bidirectional/manifest.json", {"splits": {"train": {"count": 1}, "val": {"count": 1}, "test": {"count": 1}}})
    candidate = {"schema": "text_library_candidates_v1", "prepared_root": str(prepared), "k": 2, "text_epoch": 5,
        "text_checkpoint_sha256": file_sha256(text_checkpoint), "text_index_manifest": manifest,
        "corpus_sha256": file_sha256(corpus), "query_policy": "goal_first_v1",
        "selection_policy": "exact_cosine_no_gold_injection"}
    for key, path in (("node_vocab_sha256", "vocab/node_vocab.json"), ("tactic_vocab_sha256", "vocab/tactic_vocab.json"),
                      ("argument_label_binding_sha256", "argument_labels.json"),
                      ("prepared_packed_manifest_sha256", "packed/bidirectional/manifest.json")):
        candidate[key] = file_sha256(prepared / path)
    write(candidate_dir / "manifest.json", candidate)
    pointer_config = {"prepared_root": str(prepared), "run_root": str(source), "device": "cpu", "gnn_type": "sage",
        "use_node_type": True, "edge_mode": "bidirectional", "max_args": 3,
        "model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0, "max_args": 3}}
    policy = joint_bundle._build_policy({"pointer_config": pointer_config}, node_vocab, tactic_vocab, 8).eval()
    with torch.no_grad():
        policy.arg_count_head.weight.zero_()
        policy.arg_count_head.bias.fill_(-10)
        policy.arg_count_head.bias[1] = 10
        policy.pointer.backbone.classifier.weight.zero_()
        policy.pointer.backbone.classifier.bias.zero_()
        policy.pointer.backbone.classifier.bias[1] = 10
        policy.source_head.weight.zero_()
        policy.source_head.bias.copy_(torch.tensor([5.0, 0.0]))
    binding = {"prepared_root": str(prepared), "text_checkpoint_sha256": file_sha256(text_checkpoint),
        "text_index_manifest": manifest, "candidate_manifest_sha256": file_sha256(candidate_dir / "manifest.json"), "k": 2}
    joint_checkpoint, local_checkpoint = source / "joint.pt", source / "local.pt"
    torch.save({"model_type": "joint_source_flat_decoder_v2", "phase": "complete", "epoch": 3,
        "binding": binding, "config": pointer_config, "model_state_dict": policy.state_dict(), "val_metrics": {}}, joint_checkpoint)
    with torch.no_grad():
        policy.pointer.argument_selector.out_proj.weight.add_(0.1)
    torch.save({"model_type": "frozen_graph_local_identity_v1", "phase": "complete", "epoch": 8,
        "binding": {**binding, "joint_checkpoint_sha256": file_sha256(joint_checkpoint),
                    "trained_components": "local_pointer_out_proj_only"},
        "model_state_dict": policy.state_dict(), "val_metrics": {"all_local_positions": 322}}, local_checkpoint)
    license_file = source / "RELEASE_LICENSE"
    license_file.write_text("Fixture model license text.\n")
    return {"joint_checkpoint": joint_checkpoint, "local_checkpoint": local_checkpoint,
        "text_checkpoint": text_checkpoint, "text_index": index_dir, "prepared_root": prepared,
        "candidate_dir": candidate_dir, "corpus_path": corpus, "source_commit": "a" * 40,
        "license_file": license_file, "license_id": "apache-2.0"}, policy


def test_joint_release_roundtrip_is_offline_and_contains_all_heads(tmp_path, artifacts):
    args, source_policy = artifacts
    output = tmp_path / "release"
    joint_bundle.export_joint_source_bundle(**args, output_dir=output)
    source_state = copy.deepcopy(source_policy.state_dict())
    shutil.rmtree(tmp_path / "source")
    loaded = joint_bundle.load_joint_source_bundle(output)
    for name, value in source_state.items():
        assert torch.equal(value, loaded.policy.state_dict()[name])
    assert any("argument_selector.gru" in name for name in source_state)
    assert any("arg_count_head" in name for name in source_state)
    assert any("source_head" in name for name in source_state)
    assert loaded.lemma_index.lemma_names == [f"A.lemma{i}" for i in range(4)]
    batch = Batch.from_data_list([Data(x=torch.tensor([1, 2, 3]), node_type=torch.tensor([0, 0, 0]),
        edge_index=torch.tensor([[1, 2, 0, 0], [0, 0, 1, 2]]), state_node_index=torch.tensor([0]),
        premise_mask=torch.tensor([False, True, False]))])
    predicted = loaded.predict(batch, ["h : True\n⊢ True"], graph_format=joint_bundle.GRAPH_FORMAT)
    assert predicted == [{"tactic_id": 1, "tactic": "exact", "complete": True,
                          "arguments": [{"source": "local", "identity": 1}]}]
    with pytest.raises(ValueError, match="normalized"):
        loaded.predict(batch, ["⊢ True"], graph_format="text_parser")
    cues = joint_bundle.load_joint_source_bundle(output, profile="accepted_local_cues")
    with pytest.raises(ValueError, match="normalized goal"):
        cues.predict(batch, ["⊢ True"], graph_format=joint_bundle.GRAPH_FORMAT)
    records = [{"goal_sexp": "(:c True)", "hyp_sexps": [{"context_index": 0, "sexp": "(:c True)"}]}]
    assert cues.predict(batch, ["⊢ True"], graph_format=joint_bundle.GRAPH_FORMAT, normalized_records=records)[0]["complete"]
    versions = json.loads((output / "provenance/versions.json").read_text())
    assert versions["source_code_commit"] == "a" * 40
    assert not list(output.rglob("*.pt"))
    assert str(tmp_path / "source") not in (output / "policy/config.json").read_text()
    assert str(tmp_path / "source") not in (output / "retriever/config.json").read_text()
    # Hub cache snapshots use file symlinks to sibling blob storage.
    blob = tmp_path / "blob.safetensors"
    shutil.move(output / "policy/model.safetensors", blob)
    (output / "policy/model.safetensors").symlink_to(blob)
    joint_bundle.load_joint_source_bundle(output)


@pytest.mark.parametrize("change", ["wrong_retriever", "wrong_vocab", "wrong_head", "index_vectors"])
def test_release_refuses_incompatible_training_artifacts(tmp_path, artifacts, change):
    args, _ = artifacts
    if change == "wrong_retriever":
        saved = torch.load(args["text_checkpoint"], weights_only=False)
        saved["epoch"] = 7
        torch.save(saved, args["text_checkpoint"])
    elif change == "wrong_vocab":
        vocab = json.loads((args["prepared_root"] / "vocab/node_vocab.json").read_text())
        vocab["State"], vocab["FV0"] = vocab["FV0"], vocab["State"]
        write(args["prepared_root"] / "vocab/node_vocab.json", vocab)
    elif change == "wrong_head":
        saved = torch.load(args["local_checkpoint"], weights_only=False)
        saved["model_state_dict"]["arg_count_head.bias"].add_(1)
        torch.save(saved, args["local_checkpoint"])
    else:
        vectors = np.load(args["text_index"] / "lemma_vectors.npy")
        np.save(args["text_index"] / "lemma_vectors.npy", vectors[::-1].copy())
    with pytest.raises(ValueError, match="binding|vocabulary|outside declared|FAISS"):
        joint_bundle.export_joint_source_bundle(**args, output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_release_detects_tampering_and_unknown_tokenizer_files(tmp_path, artifacts):
    args, _ = artifacts
    output = tmp_path / "release"
    joint_bundle.export_joint_source_bundle(**args, output_dir=output)
    extra = output / "retriever/tokenizer/extra.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="unmanifested"):
        joint_bundle.load_joint_source_bundle(output)
    extra.unlink()
    with (output / "retriever/index/lemma_names.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="changed"):
        joint_bundle.load_joint_source_bundle(output)


def test_publication_uses_one_parent_bound_commit_and_immutable_tags(tmp_path, artifacts):
    args, _ = artifacts
    output = tmp_path / "release"
    manifest = joint_bundle.export_joint_source_bundle(**args, output_dir=output)

    class Api:
        def __init__(self):
            self.tags, self.calls, self.new_files = {}, [], None
        def model_info(self, *args, **kwargs):
            return SimpleNamespace(sha="old")
        def list_repo_refs(self, *args, **kwargs):
            return SimpleNamespace(tags=[SimpleNamespace(name=name, target_commit=sha) for name, sha in self.tags.items()])
        def list_repo_files(self, *args, revision, **kwargs):
            return [".gitattributes", "pointer-gat-gru/model.safetensors", "vocab/node_vocab.json"] if revision == "old" else self.new_files
        def create_tag(self, repo_id, *, tag, revision, **kwargs):
            self.tags[tag] = revision
            self.calls.append(("tag", tag, revision))
        def create_commit(self, repo_id, *, operations, parent_commit, **kwargs):
            from huggingface_hub import CommitOperationAdd, CommitOperationDelete
            assert parent_commit == "old"
            assert {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)} == {
                "pointer-gat-gru/model.safetensors"}
            self.new_files = [op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)] + [".gitattributes"]
            self.calls.append(("commit", parent_commit))
            return SimpleNamespace(oid="new", commit_url="https://huggingface.co/commit/new")

    api = Api()
    planned = publish_joint_bundle(output, repo_id="org/model", release_tag="v2.0.0", previous_tag="v1.0.0", dry_run=True, api=api)
    assert not api.calls and planned["historical_revisions_retained"]
    published = publish_joint_bundle(output, repo_id="org/model", release_tag="v2.0.0", previous_tag="v1.0.0", api=api)
    assert api.calls == [("tag", "v1.0.0", "old"), ("commit", "old"), ("tag", "v2.0.0", "new")]
    assert published["commit"] == "new" and set(api.new_files) - {".gitattributes"} == set(manifest["files"]) | {"bundle.json"}
    with pytest.raises(ValueError, match="already exists"):
        publish_joint_bundle(output, repo_id="org/model", release_tag="v2.0.0", api=api)
