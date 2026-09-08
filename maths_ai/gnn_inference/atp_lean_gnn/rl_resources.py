"""Resource discovery and validation for live RL collection.

The planner is deliberately independent of Pantograph startup. It only decides
which CPU worker count and CUDA devices the RL driver may use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResourcePlan:
    """Effective collection topology after validating the requested resources."""

    cpu_count: int
    usable_cpu_count: int
    worker_count: int
    server_count: int
    collection_devices: tuple[torch.device, ...]
    update_device: torch.device
    gpu_names: tuple[str, ...]
    gpu_memory_bytes: tuple[int, ...]

    def describe(self) -> str:
        devices = ", ".join(str(device) for device in self.collection_devices)
        gpu_details = ", ".join(
            f"{name}:{memory // (1024 ** 3)}GiB" if memory else name
            for name, memory in zip(self.gpu_names, self.gpu_memory_bytes)
        )
        return (
            f"cpus={self.usable_cpu_count}/{self.cpu_count} workers={self.worker_count} "
            f"servers={self.server_count} collection_devices=[{devices}] "
            f"update_device={self.update_device} gpus=[{gpu_details}]"
        )


def _usable_cpu_count() -> tuple[int, int]:
    reported = max(int(os.cpu_count() or 1), 1)
    try:
        usable = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        usable = reported
    return reported, max(min(usable, reported), 1)


def _parse_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def plan_resources(
    *,
    collection_workers: int,
    server_pool_size: int,
    collection_devices: list[str] | tuple[str, ...] | None,
    update_device: str,
    resource_policy: str = "auto",
    cpu_reserve: int = 2,
    resource_check: str = "strict",
) -> ResourcePlan:
    """Compute and validate the effective worker/server/device topology.

    ``resource_policy='auto'`` uses all explicitly listed collection devices and
    otherwise selects the update device. ``resource_check='warn'`` clamps worker
    and server counts but still rejects malformed device names.
    """
    if resource_policy not in {"auto", "manual"}:
        raise ValueError("resource_policy must be 'auto' or 'manual'.")
    if resource_check not in {"strict", "warn"}:
        raise ValueError("resource_check must be 'strict' or 'warn'.")
    if collection_workers < 1 or server_pool_size < 1:
        raise ValueError("collection_workers and server_pool_size must be positive.")
    if cpu_reserve < 0:
        raise ValueError("cpu_reserve must be non-negative.")

    reported_cpus, usable_cpus = _usable_cpu_count()
    cpu_capacity = max(usable_cpus - cpu_reserve, 1)
    requested_devices = tuple(
        _parse_device(value)
        for value in (collection_devices or [update_device])
    )
    update = _parse_device(update_device)

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    invalid_collection = [
        device
        for device in requested_devices
        if device.type != "cpu"
        and (device.type != "cuda" or device.index is None or device.index >= cuda_count)
    ]
    if invalid_collection and resource_check == "warn":
        requested_devices = tuple(
            device for device in requested_devices if device not in invalid_collection
        )
        if not requested_devices:
            requested_devices = (torch.device("cuda:0"),) if cuda_count else (torch.device("cpu"),)
    for device in (*requested_devices, update):
        if device.type == "cpu":
            continue
        if device.type != "cuda" or device.index is None or device.index >= cuda_count:
            raise RuntimeError(
                f"Requested CUDA device {device} is unavailable; visible CUDA devices: {cuda_count}."
            )
    if update.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The update device requests CUDA, but CUDA is unavailable.")

    if resource_check == "strict":
        if collection_workers > cpu_capacity:
            raise RuntimeError(
                f"Requested {collection_workers} collection workers, but only {cpu_capacity} "
                f"CPU slots remain after reserving {cpu_reserve}."
            )
        if server_pool_size < collection_workers:
            raise RuntimeError(
                "server_pool_size must be at least collection_workers for full concurrency."
            )
        worker_count = collection_workers
        server_count = server_pool_size
    else:
        worker_count = min(collection_workers, cpu_capacity)
        server_count = max(min(server_pool_size, worker_count), 1)

    names: list[str] = []
    memory: list[int] = []
    for device in requested_devices:
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            names.append(str(props.name))
            memory.append(int(props.total_memory))
        else:
            names.append("cpu")
            memory.append(0)

    return ResourcePlan(
        cpu_count=reported_cpus,
        usable_cpu_count=usable_cpus,
        worker_count=worker_count,
        server_count=server_count,
        collection_devices=requested_devices,
        update_device=update,
        gpu_names=tuple(names),
        gpu_memory_bytes=tuple(memory),
    )
