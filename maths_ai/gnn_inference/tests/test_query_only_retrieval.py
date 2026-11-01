"""Offline transformer/index fixtures exercise serving without any downloads."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import (
    RetrievalConfig,
    LiveLemmaProvider,
    RetrievalError,
    file_hash,
)
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import state_dict_sha256
from maths_ai.gnn_inference.scripts.export_query_artifact import export


def artifacts(tmp):
    import faiss

    model_root = tmp / "bert"
    model_root.mkdir()
    config = BertConfig(
        vocab_size=9,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    config.save_pretrained(model_root)
    (model_root / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ntrue\ngoal\ncontext\nnat\n"
    )
    tokenizer = BertTokenizerFast(vocab_file=str(model_root / "vocab.txt"))
    tokenizer.save_pretrained(model_root)
    state = BertModel(config).eval()
    lemma = BertModel(config).eval()
    weights = {
        **{"state_encoder." + k: v for k, v in state.state_dict().items()},
        **{"lemma_encoder." + k: v for k, v in lemma.state_dict().items()},
    }
    corpus = tmp / "lemmas.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"lemma_id": i, "name": n})
            for i, n in [
                (10, "Fixture.first"),
                (11, "Fixture.second"),
                (12, "Fixture.self"),
            ]
        )
    )
    index_root = tmp / "index"
    index_root.mkdir()
    vectors = np.eye(3, 8, dtype=np.float32)
    np.save(index_root / "lemma_vectors.npy", vectors)
    (index_root / "lemma_ids.json").write_text(json.dumps([10, 11, 12]))
    (index_root / "lemma_names.json").write_text(
        json.dumps(["Fixture.first", "Fixture.second", "Fixture.self"])
    )
    index = faiss.IndexFlatIP(8)
    index.add(vectors)
    faiss.write_index(index, str(index_root / "faiss.index"))
    token_hash = hashlib.sha256(
        json.dumps(
            tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    source_manifest = {
        "model_type": "pretrained_text_dual_encoder_index_v1",
        "model_name": str(model_root),
        "state_text_policy": "goal_first_v1",
        "max_length": 32,
        "normalize": True,
        "lemma_encoder_sha256": state_dict_sha256(lemma.state_dict()),
        "corpus_sha256": file_hash(corpus),
        "tokenizer_sha256": token_hash,
    }
    (index_root / "manifest.json").write_text(json.dumps(source_manifest))
    checkpoint = tmp / "text.pt"
    torch.save(
        dict(
            model_type="pretrained_text_dual_encoder",
            corpus_sha256=file_hash(corpus),
            phase="complete",
            state_text_policy="goal_first_v1",
            model_name=str(model_root),
            max_length=32,
            tokenizer_sha256=token_hash,
            model_state_dict=weights,
        ),
        checkpoint,
    )
    inventory = tmp / "inventory.json"
    inventory.write_text(
        json.dumps(
            dict(
                version=1,
                environment_identity="fixture",
                declarations=["Fixture.first", "Fixture.second", "Fixture.self"],
                excluded=[],
            )
        )
    )
    query = tmp / "query"
    export(checkpoint, index_root, corpus, inventory, query)
    return (
        RetrievalConfig(
            query, index_root, corpus, inventory, top_k=2, microbatch_wait_ms=1
        ),
        state,
        tokenizer,
    )


def test_query_only_load_matches_source_tower_and_batches_live_states(
    tmp_path, monkeypatch
):
    cfg, state, tokenizer = artifacts(tmp_path)
    import transformers

    original = transformers.AutoModel.from_config
    calls = []

    def build(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoModel, "from_config", build)
    provider = LiveLemmaProvider(cfg)
    assert len(calls) == 1
    assert not hasattr(provider.query, "lemma_encoder")
    text = ["Goal: True"]
    tokens = tokenizer(
        text, padding=True, truncation=True, max_length=32, return_tensors="pt"
    )
    with torch.no_grad():
        hidden = state(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        expected = torch.nn.functional.normalize(
            (hidden * mask).sum(1) / mask.sum(1), dim=-1
        )
    np.testing.assert_allclose(
        provider.query.encode(text), expected.numpy(), rtol=1e-5, atol=1e-6
    )
    goal = SimpleNamespace(expression="True", locals=[])

    async def run():
        first, second = await asyncio.gather(
            provider.retrieve(goal, (0,), "graph", excluded=("Fixture.self",)),
            provider.retrieve(goal, (0,), "graph", excluded=("Fixture.self",)),
        )
        assert first.fingerprint == second.fingerprint
        assert len(first.lemma_ids) == 2 and 12 not in first.lemma_ids
        assert first.library is second.library
        changed = await provider.retrieve(
            SimpleNamespace(expression="Nat", locals=[]), (0,), "graph"
        )
        assert changed.query_fingerprint != first.query_fingerprint
        await provider.aclose()
        assert provider._task.done()

    asyncio.run(run())


def test_altered_index_files_fail_before_model_loading(tmp_path):
    cfg, _, _ = artifacts(tmp_path)
    (cfg.lemma_index / "lemma_ids.json").write_text("[11,10,12]")
    with pytest.raises(RetrievalError, match="differs"):
        LiveLemmaProvider(cfg)


def test_close_settles_active_and_queue_blocked_requests(tmp_path):
    from dataclasses import replace
    from threading import Event

    cfg, _, _ = artifacts(tmp_path)
    provider = LiveLemmaProvider(
        replace(cfg, queue_max_size=1, microbatch_max_size=1, microbatch_wait_ms=0)
    )
    started = Event()
    release = Event()

    def blocked_search(texts, excluded):
        started.set()
        assert release.wait(5)
        return [((), ()) for _ in texts]

    provider._search = blocked_search

    async def run():
        requests = [
            asyncio.create_task(
                provider.retrieve(
                    SimpleNamespace(expression=str(i), locals=[]), (0,), str(i)
                )
            )
            for i in range(5)
        ]
        assert await asyncio.to_thread(started.wait, 2)
        closing = asyncio.create_task(provider.aclose())
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.wait_for(closing, 3)
        outcomes = await asyncio.wait_for(
            asyncio.gather(*requests, return_exceptions=True), 3
        )
        assert all(isinstance(item, RetrievalError) for item in outcomes)
        assert provider._task.done()
        await provider.aclose()

    asyncio.run(run())


def test_empty_allowed_inventory_is_not_an_artifact_failure(tmp_path):
    cfg, _, _ = artifacts(tmp_path)
    inventory = json.loads(cfg.declaration_inventory.read_text())
    inventory["declarations"] = []
    cfg.declaration_inventory.write_text(json.dumps(inventory))
    provider = LiveLemmaProvider(cfg)

    async def run():
        pool = await provider.retrieve(
            SimpleNamespace(expression="True", locals=[]), (0,), "graph"
        )
        assert not pool.lemma_ids and pool.local_ids == (0,)
        await provider.aclose()

    asyncio.run(run())


def test_changed_query_binding_is_rejected_even_with_matching_index_files(tmp_path):
    cfg, _, _ = artifacts(tmp_path)
    path = cfg.query_artifact / "query_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_checkpoint_sha256"] = "different-text-checkpoint"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RetrievalError, match="deployment binding"):
        LiveLemmaProvider(cfg)
