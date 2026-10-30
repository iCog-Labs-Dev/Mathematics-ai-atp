"""Export only state-tower weights and verified precomputed library bindings."""

import argparse
import json
from pathlib import Path
import torch
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import fingerprint
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import file_hash


def export(checkpoint_path, index_root, corpus, inventory_path, output):
    from transformers import AutoTokenizer, AutoConfig

    checkpoint_path, index_root, output = (
        Path(checkpoint_path),
        Path(index_root),
        Path(output),
    )
    if output.exists():
        raise FileExistsError(output)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        saved["model_type"] != "pretrained_text_dual_encoder"
        or saved["phase"] != "complete"
        or saved["state_text_policy"] != "goal_first_v1"
    ):
        raise ValueError("A complete compatible dual encoder is required.")
    from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import state_dict_sha256

    state = {
        key.removeprefix("state_encoder."): value
        for key, value in saved["model_state_dict"].items()
        if key.startswith("state_encoder.")
    }
    lemma = {
        key.removeprefix("lemma_encoder."): value
        for key, value in saved["model_state_dict"].items()
        if key.startswith("lemma_encoder.")
    }
    source = json.loads((index_root / "manifest.json").read_text())
    expected = dict(
        model_type="pretrained_text_dual_encoder_index_v1",
        model_name=saved["model_name"],
        tokenizer_sha256=saved["tokenizer_sha256"],
        state_text_policy="goal_first_v1",
        corpus_sha256=file_hash(corpus),
        lemma_encoder_sha256=state_dict_sha256(lemma),
        max_length=saved["max_length"],
        normalize=True,
    )
    if source != expected or saved["corpus_sha256"] != file_hash(corpus):
        raise ValueError("Dual encoder, tokenizer, corpus, and index are not paired.")
    inventory = json.loads(Path(inventory_path).read_text())
    deployment = dict(
        schema="lemma_serving_v1",
        source_manifest=source,
        text_checkpoint_sha256=file_hash(checkpoint_path),
        query_encoder_sha256=state_dict_sha256(state),
        corpus_sha256=file_hash(corpus),
        environment_identity=inventory["environment_identity"],
        files={
            name: file_hash(index_root / name)
            for name in (
                "manifest.json",
                "faiss.index",
                "lemma_vectors.npy",
                "lemma_ids.json",
                "lemma_names.json",
            )
        },
    )
    deployment_path = index_root / "deployment_manifest.json"
    if (
        deployment_path.exists()
        and json.loads(deployment_path.read_text()) != deployment
    ):
        raise ValueError("An incompatible deployment manifest already exists.")
    output.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(
        saved["model_name"], local_files_only=True
    )
    # The training code uses JSON's default ASCII escaping for tokenizer identity.
    import hashlib

    actual = hashlib.sha256(
        json.dumps(
            tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if actual != saved["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from training.")
    tokenizer.save_pretrained(output)
    AutoConfig.from_pretrained(
        saved["model_name"], local_files_only=True
    ).save_pretrained(output)
    torch.save(state, output / "query.pt")
    manifest = dict(
        schema="query_only_v1",
        text_policy="goal_first_v1",
        max_length=saved["max_length"],
        paired_index_fingerprint=fingerprint(deployment),
        source_checkpoint_sha256=file_hash(checkpoint_path),
        lemma_encoder_sha256=state_dict_sha256(lemma),
        query_encoder_sha256=state_dict_sha256(state),
        files={
            str(p.relative_to(output)): file_hash(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    )
    (output / "query_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not deployment_path.exists():
        deployment_path.write_text(json.dumps(deployment, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "index-root", "corpus", "inventory", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.index_root, args.corpus, args.inventory, args.output)


if __name__ == "__main__":
    main()
