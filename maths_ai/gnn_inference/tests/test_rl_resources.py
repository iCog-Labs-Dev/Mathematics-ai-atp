from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn import rl_resources
from maths_ai.gnn_inference.atp_lean_gnn.rl_resources import plan_resources


def _cpu_capacity(monkeypatch, *, usable: int) -> None:
    monkeypatch.setattr(
        rl_resources,
        "_cpu_capacity",
        lambda: (16, 12, usable, usable),
    )


def test_cpu_resource_plan_uses_one_server_per_worker(monkeypatch):
    _cpu_capacity(monkeypatch, usable=8)
    plan = plan_resources(
        collection_workers=3,
        collection_devices=["cpu"],
        update_device="cpu",
        cpu_reserve=1,
    )
    assert plan.worker_count == 3
    assert plan.requested_worker_count == 3
    assert plan.collection_devices == (torch.device("cpu"),)
    assert "servers=3" in plan.describe()


def test_warning_check_reduces_workers_to_cpu_capacity(monkeypatch):
    _cpu_capacity(monkeypatch, usable=2)
    plan = plan_resources(
        collection_workers=8,
        collection_devices=["cpu"],
        update_device="cpu",
        cpu_reserve=1,
        resource_check="warn",
    )
    assert plan.worker_count == 1
    assert plan.requested_worker_count == 8
    assert plan.warnings


def test_strict_check_rejects_excess_workers(monkeypatch):
    _cpu_capacity(monkeypatch, usable=2)
    with pytest.raises(RuntimeError, match="CPU slots"):
        plan_resources(
            collection_workers=3,
            collection_devices=["cpu"],
            update_device="cpu",
            cpu_reserve=1,
            resource_check="strict",
        )


def test_cgroup_v2_quota_is_converted_to_whole_worker_capacity(monkeypatch):
    def fake_read(path: Path) -> str | None:
        if path == Path("/sys/fs/cgroup/cpu.max"):
            return "250000 100000"
        return None

    monkeypatch.setattr(rl_resources, "_read_text", fake_read)
    assert rl_resources._cgroup_cpu_quota_count() == 2


def test_cgroup_unlimited_quota_returns_none(monkeypatch):
    monkeypatch.setattr(
        rl_resources,
        "_read_text",
        lambda path: "max 100000" if path == Path("/sys/fs/cgroup/cpu.max") else None,
    )
    assert rl_resources._cgroup_cpu_quota_count() is None


def test_auto_devices_drop_devices_without_workers(monkeypatch):
    _cpu_capacity(monkeypatch, usable=8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type("Properties", (), {"name": f"GPU-{device.index}"})(),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4_000, 8_000))

    plan = plan_resources(
        collection_workers=1,
        collection_devices=None,
        update_device="cuda:0",
        cpu_reserve=0,
        resource_check="warn",
    )
    assert plan.collection_devices == (torch.device("cuda:0"),)
    assert len(plan.gpu_resources) == 1
    assert plan.gpu_resources[0].free_memory_bytes == 4_000
    assert any("unused devices" in warning for warning in plan.warnings)


def test_strict_check_rejects_explicit_device_without_worker(monkeypatch):
    _cpu_capacity(monkeypatch, usable=8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="cannot use every requested"):
        plan_resources(
            collection_workers=1,
            collection_devices=["cuda:0", "cuda:1"],
            update_device="cuda:0",
            cpu_reserve=0,
            resource_check="strict",
        )


def test_repository_config_omits_removed_resource_fields():
    config_path = (
        Path(__file__).parents[1] / "configs" / "rl_actor_critic.json"
    )
    payload = json.loads(config_path.read_text())
    assert "pantograph_server_pool_size" not in payload
    assert "resource_policy" not in payload
    assert "gpu_strategy" not in payload
    assert payload["collection_workers"] == 1
    assert payload["collection_devices"] is None
