"""Resource discovery and validation for live RL collection.

The planner is independent of Pantograph startup. It resolves the canonical update
device, active collection devices, and the number of one-server/one-reasoner workers
that may be started.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class GPUResource:
    """Memory and identity information for one active CUDA device."""

    device: torch.device
    name: str
    free_memory_bytes: int
    total_memory_bytes: int


@dataclass(frozen=True)
class ResourcePlan:
    """Effective live-collection topology after resource validation."""

    reported_cpu_count: int
    affinity_cpu_count: int
    quota_cpu_count: int | None
    usable_cpu_count: int
    requested_worker_count: int
    worker_count: int
    collection_devices: tuple[torch.device, ...]
    update_device: torch.device
    gpu_resources: tuple[GPUResource, ...]
    warnings: tuple[str, ...] = ()

    def describe(self) -> str:
        devices = ", ".join(str(device) for device in self.collection_devices)
        quota = "unlimited" if self.quota_cpu_count is None else str(self.quota_cpu_count)
        gpu_details = ", ".join(
            f"{item.device}={item.name}:"
            f"{item.free_memory_bytes // (1024 ** 3)}GiB-free/"
            f"{item.total_memory_bytes // (1024 ** 3)}GiB-total"
            for item in self.gpu_resources
        )
        warning_text = f" warnings=[{' | '.join(self.warnings)}]" if self.warnings else ""
        return (
            f"cpus={self.usable_cpu_count}/{self.reported_cpu_count} "
            f"affinity={self.affinity_cpu_count} quota={quota} "
            f"workers={self.worker_count}/{self.requested_worker_count} "
            f"servers={self.worker_count} collection_devices=[{devices}] "
            f"update_device={self.update_device} gpus=[{gpu_details}]"
            f"{warning_text}"
        )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, OSError):
        return None


def _cgroup_cpu_quota_count() -> int | None:
    """Return the Linux cgroup CPU quota as a whole-worker capacity."""

    cpu_max = _read_text(Path("/sys/fs/cgroup/cpu.max"))
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
            except ValueError:
                pass
            else:
                if quota > 0 and period > 0:
                    return max(quota // period, 1)

    quota_text = _read_text(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period_text = _read_text(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota_text is None or period_text is None:
        return None
    try:
        quota, period = int(quota_text), int(period_text)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(quota // period, 1)


def _cpu_capacity() -> tuple[int, int, int | None, int]:
    reported = max(int(os.cpu_count() or 1), 1)
    try:
        affinity = max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        affinity = reported
    quota = _cgroup_cpu_quota_count()
    limits = [reported, affinity]
    if quota is not None:
        limits.append(quota)
    return reported, affinity, quota, max(min(limits), 1)


def _parse_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported RL device type '{device.type}'; use CPU or CUDA.")
    return device


def _deduplicate_devices(devices: tuple[torch.device, ...]) -> tuple[torch.device, ...]:
    seen: set[str] = set()
    unique: list[torch.device] = []
    for device in devices:
        key = str(device)
        if key not in seen:
            seen.add(key)
            unique.append(device)
    return tuple(unique)


def _device_is_available(device: torch.device, cuda_count: int) -> bool:
    if device.type == "cpu":
        return True
    return device.index is not None and 0 <= device.index < cuda_count


def plan_resources(
    *,
    collection_workers: int,
    collection_devices: list[str] | tuple[str, ...] | None,
    update_device: str | torch.device,
    cpu_reserve: int = 2,
    resource_check: str = "strict",
) -> ResourcePlan:
    """Resolve one-server/one-reasoner collection resources.

    ``collection_workers`` is explicit because CPU count alone cannot predict the
    memory cost of a Lean process. ``collection_devices=None`` selects visible CUDA
    devices automatically and falls back to the update device when CUDA is absent.
    """

    if resource_check not in {"strict", "warn"}:
        raise ValueError("resource_check must be 'strict' or 'warn'.")
    if collection_workers < 1:
        raise ValueError("collection_workers must be positive.")
    if cpu_reserve < 0:
        raise ValueError("cpu_reserve must be non-negative.")

    reported_cpus, affinity_cpus, quota_cpus, usable_cpus = _cpu_capacity()
    cpu_capacity = max(usable_cpus - cpu_reserve, 1)
    warnings: list[str] = []

    requested_workers = collection_workers
    if collection_workers > cpu_capacity:
        message = (
            f"Requested {collection_workers} collection workers, but only {cpu_capacity} "
            f"CPU slots remain after reserving {cpu_reserve}."
        )
        if resource_check == "strict":
            raise RuntimeError(message)
        warnings.append(message)
        collection_workers = cpu_capacity

    update = _parse_device(update_device)
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not _device_is_available(update, cuda_count):
        raise RuntimeError(
            f"Requested update device {update} is unavailable; visible CUDA devices: "
            f"{cuda_count}."
        )

    explicit_devices = collection_devices is not None
    if collection_devices is None:
        requested_devices = (
            tuple(torch.device(f"cuda:{index}") for index in range(cuda_count))
            if cuda_count
            else (update,)
        )
    else:
        if not collection_devices:
            raise ValueError("collection_devices must be null or a non-empty list.")
        requested_devices = tuple(_parse_device(value) for value in collection_devices)
    requested_devices = _deduplicate_devices(requested_devices)

    invalid_devices = tuple(
        device for device in requested_devices if not _device_is_available(device, cuda_count)
    )
    if invalid_devices:
        message = (
            f"Unavailable collection devices: {[str(device) for device in invalid_devices]}; "
            f"visible CUDA devices: {cuda_count}."
        )
        if resource_check == "strict":
            raise RuntimeError(message)
        warnings.append(message)
        requested_devices = tuple(
            device for device in requested_devices if device not in invalid_devices
        )

    if not requested_devices:
        requested_devices = (update,)
        warnings.append(f"Falling back to update device {update} for collection.")

    if len(requested_devices) > collection_workers:
        unused = requested_devices[collection_workers:]
        message = (
            f"{collection_workers} workers cannot use every requested collection device; "
            f"unused devices: {[str(device) for device in unused]}."
        )
        if explicit_devices and resource_check == "strict":
            raise RuntimeError(message)
        warnings.append(message)
        requested_devices = requested_devices[:collection_workers]

    gpu_resources: list[GPUResource] = []
    for device in requested_devices:
        if device.type != "cuda":
            continue
        properties = torch.cuda.get_device_properties(device)
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        gpu_resources.append(
            GPUResource(
                device=device,
                name=str(properties.name),
                free_memory_bytes=int(free_memory),
                total_memory_bytes=int(total_memory),
            )
        )

    return ResourcePlan(
        reported_cpu_count=reported_cpus,
        affinity_cpu_count=affinity_cpus,
        quota_cpu_count=quota_cpus,
        usable_cpu_count=usable_cpus,
        requested_worker_count=requested_workers,
        worker_count=collection_workers,
        collection_devices=requested_devices,
        update_device=update,
        gpu_resources=tuple(gpu_resources),
        warnings=tuple(warnings),
    )
