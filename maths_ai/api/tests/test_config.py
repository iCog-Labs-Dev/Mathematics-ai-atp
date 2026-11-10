"""Server-owned configuration: profiles, overrides and limits."""

from __future__ import annotations

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.theorem_sources import TheoremSourceConfig
from maths_ai.api.config import SearchProfile, ServiceConfig, ServiceLimits
from maths_ai.api.tests.helpers import write_config

PUCT = SearchProfile("puct", 4, 8, 64, num_simulations=64, sim_batch_size=4, puct_c=1.0)
BEST_FIRST = SearchProfile("legacy", 4, 8, 64)


def test_profiles_resolve_through_the_search_validators():
    assert PUCT.search_kwargs()["num_simulations"] == 64
    assert BEST_FIRST.search_kwargs()["num_simulations"] is None
    assert PUCT.search_kwargs()["use_pln"] is False
    with pytest.raises(ValueError, match="simulation budget"):
        SearchProfile("legacy", 4, 8, 64, num_simulations=8)
    with pytest.raises(ValueError, match="widen"):
        SearchProfile("legacy", 4, 8, 64, widening_max_draws=8)
    with pytest.raises(ValueError, match="num_simulations"):
        SearchProfile("puct", 4, 8, 64)


def test_overrides_only_lower_applicable_budgets():
    assert PUCT.with_overrides({"num_simulations": 16, "max_depth": 4}).max_depth == 4
    with pytest.raises(ValueError, match="exceeds"):
        PUCT.with_overrides({"max_nodes": 65})
    with pytest.raises(ValueError, match="PUCT profile"):
        BEST_FIRST.with_overrides({"num_simulations": 4})
    with pytest.raises(ValueError, match="positive integer"):
        PUCT.with_overrides({"max_depth": True})
    with pytest.raises(ValueError, match="cannot be overridden"):
        PUCT.with_overrides({"puct_c": 2.0})


def test_dataset_profiles_must_be_pinned():
    with pytest.raises(ValueError, match="pinned"):
        TheoremSourceConfig("dataset", max_pool_size=4, dataset_name="x")
    with pytest.raises(ValueError, match="theorem_file"):
        TheoremSourceConfig("file", max_pool_size=4)


def test_config_resolves_paths_against_its_file_and_rejects_unknown_fields(tmp_path):
    config = ServiceConfig.from_json(write_config(tmp_path))
    assert config.rl_config == tmp_path / "rl.json"
    assert config.dataset_profiles["smoke"].theorem_file == tmp_path / "theorems.jsonl"
    assert set(config.search_profiles) == {"puct", "best_first"}
    with pytest.raises(ValueError, match="Unknown"):
        ServiceConfig.from_json(write_config(tmp_path, checkpoint_upload="x"))
    with pytest.raises(ValueError, match="default_search_profile"):
        ServiceConfig.from_json(write_config(tmp_path, default_search_profile="none"))
    with pytest.raises(ValueError):
        ServiceLimits(default_timeout_s=100.0, max_timeout_s=10.0)


def test_pln_profile_requires_native_configuration_and_is_not_a_request_override(tmp_path):
    profiles = {"pln": {"selection_policy": "legacy", "top_k_tactics": 4,
                        "max_depth": 8, "max_nodes": 64, "use_pln": True}}
    with pytest.raises(ValueError, match="PLN profiles"):
        ServiceConfig.from_json(write_config(tmp_path, search_profiles=profiles,
                                            default_search_profile="pln"))
    with pytest.raises(ValueError, match="boolean"):
        SearchProfile("legacy", 4, 8, 64, use_pln="true")
    with pytest.raises(ValueError, match="cannot be overridden"):
        BEST_FIRST.with_overrides({"use_pln": True})


def test_the_host_chooses_the_serving_device():
    from maths_ai.api.config import serving_device
    from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import RLTrainingConfig
    from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import RetrievalConfig

    retrieval = RetrievalConfig(query_artifact="q", lemma_index="i", corpus_path="c.jsonl",
                                declaration_inventory="d.json", device="cuda:0")
    cfg = RLTrainingConfig(warmstart_checkpoint="c.pt", prepared_root="prepared", device="cuda:0",
                           update_device="cuda:0", collection_devices=["cuda:0"],
                           retrieval=retrieval)
    cpu = serving_device(cfg, "cpu")
    assert (cpu.device, cpu.update_device, cpu.collection_devices, cpu.retrieval.device) == (
        "cpu", "cpu", ["cpu"], "cpu")
    assert serving_device(cfg, None) is cfg and serving_device(cfg, "") is cfg
    assert serving_device(cpu, "cuda:1").retrieval.device == "cuda:1"
    for bad in ("gpu", "cuda", "cuda:x", "cpu "):
        with pytest.raises(ValueError, match="MATHS_AI_DEVICE"):
            serving_device(cfg, bad)
