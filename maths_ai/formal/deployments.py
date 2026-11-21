"""Provider-owned deployments: which backend, profiles, ceilings and environment.

``deployment_id`` in a request names one entry of a versioned registry file.
Each entry fixes the backend, the input kinds it supports, its named profiles
with their finite tactic-step and wall-clock ceilings, the separate
verification allowance and the model identities a request may assert. A
request selects among these; it cannot change them.

The environment hash is not configured. Each backend computes it from the
runtime it actually started (``environment_manifest``), and admission and
replay compare requests against that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .canonical import object_ref
from .contract import ACCEPTED_BACKEND, MCTS_BACKEND

REGISTRY_FORMAT = "maths-ai-formal-deployments-v1"


class DeploymentConfigError(ValueError):
    pass


def _positive_int(value: Any, where: str) -> int:
    if type(value) is not int or value < 1:
        raise DeploymentConfigError(f"{where} must be a positive integer, got {value!r}.")
    return value


@dataclass(frozen=True)
class Profile:
    name: str
    max_steps: int
    wall_clock_ms: int
    default_wall_clock_ms: int
    backend_settings: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_steps": self.max_steps,
            "wall_clock_ms": self.wall_clock_ms,
            "default_wall_clock_ms": self.default_wall_clock_ms,
            "backend_settings": dict(self.backend_settings),
        }


@dataclass(frozen=True)
class Deployment:
    deployment_id: str
    backend: str
    environment_id: str
    inputs: tuple[str, ...]
    profiles: Mapping[str, Profile]
    verification_max_steps: int
    verification_wall_clock_ms: int
    model: Mapping[str, Any] = field(default_factory=dict)

    def profile_config_hash(self, profile: Profile) -> str:
        return object_ref(
            {
                "format": REGISTRY_FORMAT,
                "deployment_id": self.deployment_id,
                "backend": self.backend,
                "environment_id": self.environment_id,
                "profile": profile.as_dict(),
                "model": dict(self.model),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "backend": self.backend,
            "environment_id": self.environment_id,
            "inputs": list(self.inputs),
            "profiles": {name: profile.as_dict() for name, profile in self.profiles.items()},
            "verification": {
                "max_steps": self.verification_max_steps,
                "wall_clock_ms": self.verification_wall_clock_ms,
            },
            "model": dict(self.model),
        }


def _profile(name: str, payload: Any, where: str) -> Profile:
    if not isinstance(payload, dict):
        raise DeploymentConfigError(f"{where} must be an object.")
    unknown = set(payload) - {"max_steps", "wall_clock_ms", "default_wall_clock_ms", "backend_settings"}
    if unknown:
        raise DeploymentConfigError(f"{where} has unknown settings {sorted(unknown)}.")
    ceiling = _positive_int(payload.get("wall_clock_ms"), f"{where}.wall_clock_ms")
    default = _positive_int(payload.get("default_wall_clock_ms", ceiling), f"{where}.default_wall_clock_ms")
    if default > ceiling:
        raise DeploymentConfigError(f"{where}.default_wall_clock_ms exceeds its ceiling.")
    settings = payload.get("backend_settings", {})
    if not isinstance(settings, dict):
        raise DeploymentConfigError(f"{where}.backend_settings must be an object.")
    return Profile(
        name,
        _positive_int(payload.get("max_steps"), f"{where}.max_steps"),
        ceiling,
        default,
        dict(settings),
    )


def _deployment(deployment_id: str, payload: Any) -> Deployment:
    where = f"deployments.{deployment_id}"
    if not isinstance(payload, dict):
        raise DeploymentConfigError(f"{where} must be an object.")
    unknown = set(payload) - {"backend", "environment_id", "inputs", "profiles", "verification", "model"}
    if unknown:
        raise DeploymentConfigError(f"{where} has unknown settings {sorted(unknown)}.")
    backend = payload.get("backend")
    if backend not in (MCTS_BACKEND, ACCEPTED_BACKEND):
        raise DeploymentConfigError(f"{where}.backend {backend!r} is not a known backend.")
    environment_id = payload.get("environment_id")
    if not isinstance(environment_id, str) or not environment_id:
        raise DeploymentConfigError(f"{where}.environment_id must be a nonempty string.")
    inputs = payload.get("inputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or any(kind not in ("goal", "source") for kind in inputs)
        or len(set(inputs)) != len(inputs)
    ):
        raise DeploymentConfigError(f"{where}.inputs must list 'goal' and/or 'source'.")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise DeploymentConfigError(f"{where}.profiles must name at least one profile.")
    verification = payload.get("verification")
    if not isinstance(verification, dict) or set(verification) != {"max_steps", "wall_clock_ms"}:
        raise DeploymentConfigError(f"{where}.verification needs max_steps and wall_clock_ms.")
    model = payload.get("model", {})
    if not isinstance(model, dict):
        raise DeploymentConfigError(f"{where}.model must be an object.")
    return Deployment(
        deployment_id,
        backend,
        environment_id,
        tuple(inputs),
        {name: _profile(name, value, f"{where}.profiles.{name}") for name, value in profiles.items()},
        _positive_int(verification["max_steps"], f"{where}.verification.max_steps"),
        _positive_int(verification["wall_clock_ms"], f"{where}.verification.wall_clock_ms"),
        dict(model),
    )


@dataclass(frozen=True)
class DeploymentRegistry:
    deployments: Mapping[str, Deployment]
    registry_hash: str
    default: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> "DeploymentRegistry":
        if not isinstance(payload, dict) or payload.get("format") != REGISTRY_FORMAT:
            raise DeploymentConfigError(f"The deployment registry must declare format {REGISTRY_FORMAT!r}.")
        unknown = set(payload) - {"format", "deployments", "default"}
        if unknown:
            raise DeploymentConfigError(f"The deployment registry has unknown keys {sorted(unknown)}.")
        entries = payload.get("deployments")
        if not isinstance(entries, dict) or not entries:
            raise DeploymentConfigError("The deployment registry needs at least one deployment.")
        deployments = {key: _deployment(key, value) for key, value in entries.items()}
        default = payload.get("default")
        if default is not None:
            if not isinstance(default, dict) or set(default) - {"deployment_id", "profile", "proposal_mode", "seed"}:
                raise DeploymentConfigError("default must contain a deployment/profile selection")
            deployment = deployments.get(default.get("deployment_id"))
            if deployment is None or default.get("profile") not in deployment.profiles:
                raise DeploymentConfigError("default selects an unknown deployment/profile")
            from .contract import BackendSelection
            BackendSelection.model_validate({"name": deployment.backend, **{key: value for key, value in default.items() if key != "deployment_id"}})
        return cls(deployments, object_ref(payload), None if default is None else dict(default))

    @classmethod
    def from_json(cls, path: Path) -> "DeploymentRegistry":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DeploymentConfigError(f"Cannot read the deployment registry {path}: {exc}") from exc
        return cls.from_dict(payload)

    def for_backend(self, backend: str) -> list[Deployment]:
        return [entry for entry in self.deployments.values() if entry.backend == backend]
