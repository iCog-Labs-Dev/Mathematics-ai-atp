from __future__ import annotations

import json
from pathlib import Path

FAKE_BACKEND = "maths_ai.api.tests.fake_backend:FakeBackend"


def write_config(tmp: Path, **overrides) -> Path:
    payload = {
        "rl_config": "rl.json",
        "state_dir": "state",
        "default_search_profile": "puct",
        "search_profiles": {
            "puct": {"selection_policy": "puct", "top_k_tactics": 4, "max_depth": 8,
                     "max_nodes": 64, "num_simulations": 64, "sim_batch_size": 4, "puct_c": 1.0},
            "best_first": {"selection_policy": "legacy", "top_k_tactics": 4, "max_depth": 8,
                           "max_nodes": 64},
        },
        "dataset_profiles": {
            "smoke": {"data_source": "file", "theorem_file": "theorems.jsonl",
                      "eval_pool_size": 2, "max_pool_size": 4},
        },
        "limits": {"cleanup_grace_s": 2.0, "runtime_start_timeout_s": 60.0,
                   "default_timeout_s": 5.0, "max_timeout_s": 30.0,
                   "default_queue_timeout_s": 2.0, "max_queue_timeout_s": 10.0,
                   "eval_chunk_size": 1},
    }
    for key, value in overrides.items():
        if key == "limits":
            payload["limits"].update(value)
        else:
            payload[key] = value
    path = tmp / "service.json"
    path.write_text(json.dumps(payload))
    return path
