"""Independent source admission, provenance and exact-resume contracts."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn import theorem_sources as sources
from maths_ai.gnn_inference.atp_lean_gnn import rl_training_driver as driver
from maths_ai.gnn_inference.atp_lean_gnn.dataset import (
    DATASET_NAME,
    DATASET_REVISION,
    dataset_split_name,
)
from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import (
    AdmissionReport,
    TheoremItem,
)
from maths_ai.lean_replay import seed_from_text


def dataset(split, **kwargs):
    return sources.TheoremSourceConfig(
        data_source="dataset",
        dataset_name=DATASET_NAME,
        dataset_revision=DATASET_REVISION,
        split=split,
        **kwargs,
    )


def item(goal, row):
    return TheoremItem(
        seed_from_text(goal=goal, source={"row_index": row}), "", len(goal)
    )


def config(**kwargs):
    return driver.RLTrainingConfig(
        warmstart_checkpoint=Path("/checkpoint"),
        prepared_root=Path("/prepared"),
        **kwargs,
    )


def test_split_alias_is_the_published_name():
    assert dataset_split_name("val") == dataset_split_name("validation") == "val"
    assert dataset("val").contract() == dataset("validation").contract()
    assert dataset_split_name("test") == "test"


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_revision": "main"},
        {"split": "eval"},
        {"max_pool_size": 0},
        {"max_scan_rows": True},
        {"max_replay_chars": -1},
        {"admission_time_budget_s": float("inf")},
        {"theorem_file": "/unrelated"},
    ],
)
def test_invalid_source_settings_fail(changes):
    with pytest.raises(ValueError):
        replace(dataset("train"), **changes)


def test_source_config_roundtrip_and_retired_fields(tmp_path):
    cfg = config(
        training_data=dataset("train", seed_cache_path=tmp_path / "train.json"),
        evaluation_data=dataset("val", max_pool_size=20),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert driver.RLTrainingConfig.from_json(path).to_dict() == cfg.to_dict()
    path.write_text(json.dumps({**cfg.to_dict(), "leandojo_split": "train"}))
    with pytest.raises(ValueError, match="Retired flat"):
        driver.RLTrainingConfig.from_json(path)


def test_training_and_evaluation_are_admitted_independently(tmp_path, monkeypatch):
    calls = []
    train = [item("True", 0), item("True ∧ True", 1)]
    evaluation = [item("False → False", 2)]

    async def admit(source, env, **kwargs):
        calls.append((source.split, kwargs["report_dir"]))
        rows = train if source.split == "train" else evaluation
        return sources.finalize_admission(source, rows, AdmissionReport())

    monkeypatch.setattr(driver, "admit_source", admit)
    cfg = config(
        training_data=dataset("train"), evaluation_data=dataset("val"), eval_every=1
    )
    pool = asyncio.run(driver.build_theorem_pool(cfg, None, report_dir=tmp_path))
    assert {x.seed.source_dict["row_index"] for x in pool.train_items} == {0, 1}
    assert {x.seed.source_dict["row_index"] for x in pool.eval_items} == {2}
    assert calls == [("train", tmp_path / "training"), ("val", tmp_path / "evaluation")]
    for _ in range(10):
        assert all(
            row.seed.source_dict["row_index"] != 2 for row in pool.sample_batch(2)
        )
    assert pool.admission["evaluation"]["source"]["split"] == "val"
    assert (
        pool.admission["training"]["pool_identity"]
        != pool.admission["evaluation"]["pool_identity"]
    )


def test_eval_only_never_admits_train_and_disabled_periodic_never_admits_val(
    monkeypatch,
):
    calls = []

    async def admit(source, env, **kwargs):
        calls.append(source.split)
        return sources.finalize_admission(source, [item("True", 0)], AdmissionReport())

    monkeypatch.setattr(driver, "admit_source", admit)
    cfg = config(
        training_data=dataset("train"), evaluation_data=dataset("test"), eval_every=0
    )
    pool = asyncio.run(driver.build_theorem_pool(cfg, None, evaluation_only=True))
    assert calls == ["test"] and not pool.train_items and len(pool.eval_items) == 1
    calls.clear()
    pool = asyncio.run(driver.build_theorem_pool(cfg, None))
    assert calls == ["train"] and len(pool.train_items) == 1 and not pool.eval_items


def test_missing_required_sources_and_same_source_are_rejected():
    with pytest.raises(ValueError, match="missing"):
        asyncio.run(driver.build_theorem_pool(config(), None, evaluation_only=True))
    with pytest.raises(ValueError, match="independent"):
        sources.validate_source_separation(dataset("train"), dataset("train"))
    with pytest.raises(ValueError, match="separate admission cache"):
        sources.validate_source_separation(
            dataset("train", seed_cache_path="cache.json"),
            dataset("val", seed_cache_path="cache.json"),
        )


def test_file_identity_and_pool_identity_bind_contents_and_labels(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text('{"goal":"True"}\n')
    source = sources.TheoremSourceConfig(data_source="file", theorem_file=path)
    before = source.contract()
    assert (
        replace(source, seed_cache_path=tmp_path / "other-cache").contract() == before
    )
    path.write_text('{"goal":"False"}\n')
    assert source.contract() != before
    original = item("True", 0)
    assert sources.pool_identity([original]) != sources.pool_identity(
        [replace(original, tactic_label="trivial")]
    )
    assert sources.ordered_items(
        [item("True", 2), item("True", 1)]
    ) == sources.ordered_items([item("True", 1), item("True", 2)])


def test_resume_rejects_changed_pool_and_accepts_new_warm_start(tmp_path):
    from maths_ai.gnn_inference.tests.test_rl_training_driver import (
        _write_config,
        _pool,
        _qed_factory,
    )

    cfg = _write_config(tmp_path, num_rounds=1)
    pool = _pool()
    asyncio.run(driver.run_rl_training(cfg, reasoner_factory=_qed_factory, pool=pool))
    run = next((tmp_path / "runs").iterdir())
    checkpoint = torch.load(run / "last.pt", weights_only=False)
    assert checkpoint["data_contract"]["version"] == sources.DATA_CONTRACT_VERSION
    cfg.num_rounds = 2
    pool.eval_items = [item("False", 100)]
    with pytest.raises(ValueError, match="different training/evaluation pools"):
        asyncio.run(
            driver.run_rl_training(
                cfg, resume_run_dir=run, reasoner_factory=_qed_factory, pool=pool
            )
        )
    # Existing checkpoint weights do not require an exact-resume data contract.
    cfg.warmstart_checkpoint = run / "last.pt"
    cfg.run_root = tmp_path / "fresh"
    cfg.num_rounds = 0
    asyncio.run(
        driver.run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool())
    )


@pytest.mark.pantograph_live
def test_native_independent_admission_and_certified_evaluation(live_env, tmp_path):
    from maths_ai.hybrid_reasoner.mcts_diagnostics import ScriptedSearchReasoner
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
        create_model_sexpr_client,
    )
    from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import (
        evaluate_proof_rate,
        EvaluationSettings,
    )
    from maths_ai.gnn_inference.atp_lean_gnn.evaluation_archive import EvaluationArchive

    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text(json.dumps({"goal": "True"}) + "\n")
    val.write_text(json.dumps({"goal": "True ∧ True"}) + "\n")
    cfg = config(
        training_data=sources.TheoremSourceConfig(
            data_source="file",
            theorem_file=train,
            seed_cache_path=tmp_path / "train-cache.json",
            max_pool_size=1,
        ),
        evaluation_data=sources.TheoremSourceConfig(
            data_source="file",
            theorem_file=val,
            seed_cache_path=tmp_path / "val-cache.json",
            max_pool_size=1,
        ),
        eval_every=1,
    )

    async def run():
        pool = await driver.build_theorem_pool(
            cfg, live_env, report_dir=tmp_path / "admission"
        )
        assert len(pool.train_items) == len(pool.eval_items) == 1
        assert pool.train_items[0].seed.source_dict["file"] == str(train)
        assert pool.eval_items[0].seed.source_dict["file"] == str(val)
        async with await create_model_sexpr_client(live_env) as client:
            for policy in ("legacy", "puct"):
                search = ScriptedSearchReasoner(
                    client,
                    {
                        "True ∧ True": (("constructor", 1.0),),
                        "True": (("trivial", 1.0),),
                    },
                    selection_policy=policy,
                    top_k_tactics=1,
                    max_nodes=16,
                    max_depth=4,
                    **(
                        {"num_simulations": 16, "sim_batch_size": 2}
                        if policy == "puct"
                        else {}
                    ),
                )

                class Reasoner:
                    model = torch.nn.Linear(1, 1)

                    async def prove(self, seed, **kwargs):
                        graph = await search.prove(seed)
                        return SimpleNamespace(
                            graph=graph, observations=[], draws=lambda: 0
                        )

                archive = EvaluationArchive(
                    tmp_path / policy,
                    {"evaluation_source": pool.admission["evaluation"]},
                )
                stats = await evaluate_proof_rate(
                    Reasoner(),
                    pool.eval_items,
                    settings=EvaluationSettings(),
                    archive=archive,
                )
                assert stats["solved"] == stats["exported_proofs"] == 1
                row = json.loads((archive.path / "attempts.jsonl").read_text())
                assert row["source"]["file"] == str(val)
                assert (await client.stat()).n_goals == 0

    asyncio.run(run())
    assert list((tmp_path / "admission/training").glob("admission-*.json"))
    assert list((tmp_path / "admission/evaluation").glob("admission-*.json"))


def test_inactive_evaluation_source_is_not_read(tmp_path):
    cfg = config(
        training_data=dataset("train"),
        eval_every=0,
        evaluation_data=sources.TheoremSourceConfig(
            data_source="file", theorem_file=tmp_path / "absent.jsonl"
        ),
    )
    assert sources.configured_sources(cfg)["evaluation"] is None


def test_empty_evaluation_source_does_not_use_training_items(monkeypatch):
    calls = []

    async def admit(source, env, **kwargs):
        calls.append(source.split)
        if source.split == "val":
            raise RuntimeError("No seed passed Lean validation")
        return sources.finalize_admission(source, [item("True", 0)], AdmissionReport())

    monkeypatch.setattr(driver, "admit_source", admit)
    cfg = config(
        training_data=dataset("train"), evaluation_data=dataset("val"), eval_every=1
    )
    with pytest.raises(RuntimeError, match="No seed passed"):
        asyncio.run(driver.build_theorem_pool(cfg, None))
    assert calls == ["train", "val"]


def test_api_and_cli_finalization_produce_the_same_inputs(tmp_path):
    from maths_ai.api.runtime_backend import LiveBackend, _EvaluationJob
    from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import subset_identity

    path = tmp_path / "eval.jsonl"
    path.write_text('{"goal":"True"}\n')
    source = sources.TheoremSourceConfig(data_source="file", theorem_file=path)
    rows = [item("True", 7), item("True ∧ True", 9)]
    report = AdmissionReport()

    class Session:
        done = True
        rows_seen = 2
        admitted = rows

        async def step(self, count):
            return []

        async def close(self):
            pass

    backend = LiveBackend.__new__(LiveBackend)
    backend.jobs = {"job": _EvaluationJob(Session(), report, source, 42)}
    # Admission receives proposal settings, not the expanded metric contract.
    progress = asyncio.run(
        backend.admit(
            {
                "job_id": "job",
                "rows": 2,
                "evaluation": {"mode": "sample", "seed": 42, "repeats": 1},
            }
        )
    )
    expected = sources.finalize_admission(source, rows, report)
    assert progress["eval_subset_sha256"] == subset_identity(expected.items)
    assert (
        progress["admission"]["source_contract"]["pool_identity"]
        == expected.provenance["pool_identity"]
    )
