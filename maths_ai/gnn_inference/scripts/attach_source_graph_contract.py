"""Attach AC's explicit graph contract to verified pinned-source prepared data.

This operation adds metadata only. It does not rebuild graphs, relabel arguments,
renumber vocabularies, or rewrite any checkpoint-bound prepared file.
"""

import argparse
import json
from pathlib import Path
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import file_hash


def attach(root, candidate_manifest):
    root = Path(root)
    contract = root / "metadata/graph_representation.json"
    if contract.exists():
        raise FileExistsError(contract)
    manifest = json.loads(Path(candidate_manifest).read_text())
    fields = {
        "node_vocab_sha256": "vocab/node_vocab.json",
        "tactic_vocab_sha256": "vocab/tactic_vocab.json",
        "argument_label_binding_sha256": "argument_labels.json",
        "prepared_packed_manifest_sha256": "packed/bidirectional/manifest.json",
    }
    for key, relative in fields.items():
        if file_hash(root / relative) != manifest[key]:
            raise ValueError(f"Prepared source artifact differs: {relative}")
    vocab = json.loads((root / "vocab/node_vocab.json").read_text())
    if not {"State", "Hyp", "Goal", "HypRole:explicit"} <= set(vocab) or not any(
        name.startswith("FV") and name[2:].isdigit() for name in vocab
    ):
        raise ValueError(
            "Source vocabulary does not describe the pinned model-expression context schema."
        )
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(json.dumps(MODEL_SEXPR_GRAPH_SPEC.to_dict(), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    args = parser.parse_args()
    attach(args.prepared_root, args.candidate_manifest)


if __name__ == "__main__":
    main()
