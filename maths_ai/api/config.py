"""Server-owned configuration of the MCTS proving service.

Everything a request could otherwise use to reach the host (checkpoint, REPL,
imports, prepared root, retrieval artifacts, dataset files) comes from here.
Requests name a search or dataset profile and may only lower its budgets.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from maths_ai.hybrid_reasoner.selection_policy import (
    WIDENING_FIELDS,
    resolve_search_params,
    resolve_widening,
)

# Budgets a request may lower. PUCT-only ones are refused for best-first.
REDUCIBLE_BUDGETS = ("top_k_tactics", "max_depth", "max_nodes")
PUCT_BUDGETS = ("num_simulations",)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class SearchProfile:
    """A complete, validated set of reasoner search settings.

    ``selection_policy`` is ``"puct"`` or ``"legacy"`` (best-first). The
    resolvers that validate a training config validate a profile, so a
    best-first profile with a simulation budget or widening bound is refused.
    PLN stays off in served profiles.
    """

    selection_policy: str
    top_k_tactics: int
    max_depth: int
    max_nodes: int
    num_simulations: int | None = None
    sim_batch_size: int | None = None
    puct_c: float | None = None
    widening_initial_actions: int | None = None
    widening_coefficient: float | None = None
    widening_exponent: float | None = None
    widening_max_actions: int | None = None
    widening_max_draws: int | None = None

    def __post_init__(self) -> None:
        for name in REDUCIBLE_BUDGETS:
            _positive_int(name, getattr(self, name))
        resolve_search_params(
            self.selection_policy, self.num_simulations, self.sim_batch_size, self.puct_c
        )
        resolve_widening(self.selection_policy, *(getattr(self, name) for name in WIDENING_FIELDS))

    @property
    def is_puct(self) -> bool:
        return self.selection_policy == "puct"

    def search_kwargs(self) -> dict[str, Any]:
        """The reasoner constructor's search settings, as the training config builds them."""
        return {
            "top_k_tactics": self.top_k_tactics,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "selection_policy": self.selection_policy,
            "num_simulations": self.num_simulations,
            "sim_batch_size": self.sim_batch_size,
            "puct_c": self.puct_c,
            **{name: getattr(self, name) for name in WIDENING_FIELDS},
            "use_pln": False,
        }

    def with_overrides(self, overrides: dict[str, Any]) -> "SearchProfile":
        """This profile with lowered budgets; a raise or an inapplicable field is refused."""
        changes: dict[str, Any] = {}
        for name, value in overrides.items():
            if value is None:
                continue
            if name in PUCT_BUDGETS and not self.is_puct:
                raise ValueError(f"{name} applies only to a PUCT profile")
            if name not in (*REDUCIBLE_BUDGETS, *PUCT_BUDGETS):
                raise ValueError(f"{name} cannot be overridden")
            _positive_int(name, value)
            if value > getattr(self, name):
                raise ValueError(
                    f"{name}={value} exceeds the profile's limit of {getattr(self, name)}"
                )
            changes[name] = value
        return SearchProfile(**{**asdict(self), **changes})


@dataclass(frozen=True)
class DatasetProfile:
    """An allowlisted evaluation input: a pinned dataset revision or a mounted file."""

    data_source: str
    eval_pool_size: int
    max_pool_size: int
    dataset_name: str | None = None
    dataset_revision: str | None = None
    split: str = "train"
    theorem_file: str | None = None
    max_scan_rows: int | None = None
    max_state_chars: int | None = None
    max_replay_chars: int | None = None
    admission_time_budget_s: float = 600.0

    def __post_init__(self) -> None:
        if self.data_source == "dataset":
            if not self.dataset_name or not self.dataset_revision:
                raise ValueError("A dataset profile pins dataset_name and dataset_revision")
        elif self.data_source == "file":
            if not self.theorem_file:
                raise ValueError("A file profile needs theorem_file")
        else:
            raise ValueError(f"data_source must be 'dataset' or 'file', got {self.data_source!r}")
        _positive_int("eval_pool_size", self.eval_pool_size)
        _positive_int("max_pool_size", self.max_pool_size)
        _positive_float("admission_time_budget_s", self.admission_time_budget_s)

    def rl_overrides(self) -> dict[str, Any]:
        """The training-config fields this profile sets for admission.

        Optional limits the profile leaves unset keep the training config's.
        """
        overrides = {
            "data_source": self.data_source,
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "leandojo_split": self.split,
            "theorem_file": None if self.theorem_file is None else Path(self.theorem_file),
            "max_pool_size": self.max_pool_size,
            "eval_pool_size": self.eval_pool_size,
            "admission_time_budget_s": self.admission_time_budget_s,
        }
        for name in ("max_scan_rows", "max_state_chars", "max_replay_chars"):
            if getattr(self, name) is not None:
                overrides[name] = getattr(self, name)
        return overrides


@dataclass(frozen=True)
class ServiceLimits:
    """Queue, input, deadline and message bounds."""

    max_pending_proofs: int = 8
    max_pending_jobs: int = 4
    max_goal_chars: int = 4000
    max_hypotheses: int = 64
    max_hypothesis_chars: int = 2000
    default_timeout_s: float = 60.0
    max_timeout_s: float = 600.0
    default_queue_timeout_s: float = 10.0
    max_queue_timeout_s: float = 60.0
    max_job_deadline_s: float = 86400.0
    max_repeats: int = 8
    eval_chunk_size: int = 4
    admission_step_rows: int = 16
    # Interactive requests served before one waiting evaluation step.
    interactive_per_chunk: int = 2
    runtime_start_timeout_s: float = 900.0
    cleanup_grace_s: float = 60.0
    max_message_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type == "int":
                _positive_int(f.name, value)
            else:
                _positive_float(f.name, value)
        if self.default_timeout_s > self.max_timeout_s:
            raise ValueError("default_timeout_s exceeds max_timeout_s")
        if self.default_queue_timeout_s > self.max_queue_timeout_s:
            raise ValueError("default_queue_timeout_s exceeds max_queue_timeout_s")


@dataclass(frozen=True)
class ServiceConfig:
    """The whole service configuration, loaded once at startup."""

    rl_config: Path
    state_dir: Path
    search_profiles: dict[str, SearchProfile]
    default_search_profile: str
    dataset_profiles: dict[str, DatasetProfile] = field(default_factory=dict)
    checkpoint: Path | None = None
    limits: ServiceLimits = field(default_factory=ServiceLimits)

    def __post_init__(self) -> None:
        if not self.search_profiles:
            raise ValueError("At least one search profile is required")
        if self.default_search_profile not in self.search_profiles:
            raise ValueError(
                f"default_search_profile {self.default_search_profile!r} is not a configured profile"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, base: Path | None = None) -> "ServiceConfig":
        base = base or Path.cwd()

        def path(value: str | None) -> Path | None:
            if value is None:
                return None
            candidate = Path(value)
            return candidate if candidate.is_absolute() else base / candidate

        known = {"rl_config", "state_dir", "search_profiles", "default_search_profile",
                 "dataset_profiles", "checkpoint", "limits"}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown service configuration fields: {unknown}")
        return cls(
            rl_config=path(payload["rl_config"]),
            state_dir=path(payload["state_dir"]),
            search_profiles={
                name: SearchProfile(**profile)
                for name, profile in payload["search_profiles"].items()
            },
            default_search_profile=payload["default_search_profile"],
            dataset_profiles={
                name: DatasetProfile(**profile)
                for name, profile in payload.get("dataset_profiles", {}).items()
            },
            checkpoint=path(payload.get("checkpoint")),
            limits=ServiceLimits(**payload.get("limits", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ServiceConfig":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), base=path.parent)
