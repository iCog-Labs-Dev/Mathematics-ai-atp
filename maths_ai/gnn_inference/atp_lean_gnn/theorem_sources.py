"""Independent, validated theorem sources shared by training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from maths_ai.hybrid_reasoner.proposal import input_identity
from maths_ai.lean_replay import RENDERER_VERSION, render_replay
from .admission_rows import AdmissionSource
from .lean_source_rows import elaborate_candidate, read_selection
from .dataset import canonicalize_split_name
from .goal_file_records import GOAL_FILE_DECODER_VERSION
from .seed_admission import (
    AdmissionReport,
    AdmissionSession,
    TheoremItem,
    ValidationCache,
    dataset_candidates,
    environment_identity,
    file_candidates,
    write_report,
)

DATA_CONTRACT_VERSION = 1
LEAN_SOURCE_FIELDS = ("lean_source_root", "lean_source_revision", "lean_source_manifest")
SOURCE_PATH_FIELDS = ("theorem_file", "seed_cache_path", "lean_source_root", "lean_source_manifest")
RETIRED_SOURCE_FIELDS = frozenset(
    {
        *LEAN_SOURCE_FIELDS,
        "data_source",
        "theorem_file",
        "dataset_name",
        "dataset_revision",
        "leandojo_split",
        "max_pool_size",
        "max_scan_rows",
        "max_state_chars",
        "max_replay_chars",
        "admission_time_budget_s",
        "seed_cache_path",
        "eval_pool_size",
    }
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TheoremSourceConfig:
    data_source: str
    dataset_name: str | None = None
    dataset_revision: str | None = None
    split: str | None = None
    theorem_file: Path | None = None
    max_pool_size: int = 200
    max_scan_rows: int = 20000
    max_state_chars: int | None = None
    max_replay_chars: int | None = 50000
    admission_time_budget_s: float = 1800.0
    seed_cache_path: Path | None = None
    lean_source_root: Path | None = None
    lean_source_revision: str | None = None
    lean_source_manifest: Path | None = None

    def __post_init__(self):
        for name in SOURCE_PATH_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        source_fields = [getattr(self, name) for name in LEAN_SOURCE_FIELDS]
        if self.data_source != "lean_source" and any(v is not None for v in source_fields):
            raise ValueError("Lean source fields require data_source=lean_source")
        if self.data_source == "dataset":
            if not self.dataset_name or not self.dataset_revision or not self.split:
                raise ValueError(
                    "Dataset sources require dataset_name, pinned dataset_revision, and split"
                )
            if len(self.dataset_revision) != 40 or any(
                c not in "0123456789abcdef" for c in self.dataset_revision.lower()
            ):
                raise ValueError("dataset_revision must be a full pinned commit SHA")
            if self.theorem_file is not None:
                raise ValueError("Dataset sources cannot specify theorem_file")
            object.__setattr__(self, "split", canonicalize_split_name(self.split))
        elif self.data_source == "file":
            if self.theorem_file is None or any(
                v is not None
                for v in (self.dataset_name, self.dataset_revision, self.split)
            ):
                raise ValueError(
                    "File sources require theorem_file and no dataset fields"
                )
        elif self.data_source == "lean_source":
            if any(v is None for v in source_fields) or any(
                v is not None for v in (self.dataset_name, self.dataset_revision, self.split, self.theorem_file)
            ):
                raise ValueError("Lean source evaluation requires root, revision, and manifest exclusively")
            revision = self.lean_source_revision
            if not isinstance(revision, str) or len(revision) != 40 or any(
                c not in "0123456789abcdef" for c in revision
            ):
                raise ValueError("lean_source_revision must be a full lowercase Git commit SHA")
        else:
            raise ValueError("data_source must be dataset, file, or lean_source")
        for name in (
            "max_pool_size",
            "max_scan_rows",
            "max_state_chars",
            "max_replay_chars",
        ):
            value = getattr(self, name)
            if value is None and name in ("max_state_chars", "max_replay_chars"):
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.admission_time_budget_s, bool)
            or not isinstance(self.admission_time_budget_s, (int, float))
            or not math.isfinite(self.admission_time_budget_s)
            or self.admission_time_budget_s <= 0
        ):
            raise ValueError("admission_time_budget_s must be positive and finite")

    def to_dict(self):
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(self).items()
            if k not in LEAN_SOURCE_FIELDS or v is not None
        }

    def source_identity(self):
        if self.data_source == "dataset":
            return {
                "kind": "dataset",
                "dataset": self.dataset_name,
                "revision": self.dataset_revision,
                "split": self.split,
            }
        if self.data_source == "lean_source":
            return read_selection(self.lean_source_manifest, self.lean_source_revision)[0]
        path = self.theorem_file
        return {"kind": "file", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def contract(self):
        config = self.to_dict()
        config.pop("seed_cache_path")
        config.pop("theorem_file")
        config.pop("lean_source_root", None)
        config.pop("lean_source_manifest", None)
        return {
            "config": config,
            "source": self.source_identity(),
            "renderer": RENDERER_VERSION,
            "file_decoder": GOAL_FILE_DECODER_VERSION,
            "ordering": "size-input-identity-v1",
        }


def ordered_items(items):
    return sorted(items, key=lambda item: (item.size, input_identity(item.seed)))


def pool_identity(items):
    return digest(
        [
            {"seed": asdict(item.seed), "label": item.tactic_label, "size": item.size}
            for item in items
        ]
    )


@dataclass(frozen=True)
class AdmittedSource:
    items: list[TheoremItem]
    provenance: dict


def open_admission(source: TheoremSourceConfig, env, *, client_factory=None):
    report = AdmissionReport()
    report.environment = environment_identity(env)
    resolver = None
    cache_environment = report.environment
    if source.data_source == "lean_source":
        identity, selections = read_selection(
            source.lean_source_manifest, source.lean_source_revision
        )
        candidates = AdmissionSource("lean_source", {
            "root": source.lean_source_root,
            "identity": identity,
            "selections": selections,
            "max_scan_rows": source.max_scan_rows,
        })
        report.source = {**identity, "max_scan_rows": source.max_scan_rows}
        cache_environment = {**report.environment, "lean_source": identity}
        resolver = partial(
            elaborate_candidate, env=env, max_state_chars=source.max_state_chars, report=report
        )
    elif source.data_source == "file":
        candidates = file_candidates(
            source.theorem_file,
            max_scan_rows=source.max_scan_rows,
            max_state_chars=source.max_state_chars,
            report=report,
        )
    else:
        candidates = dataset_candidates(
            dataset_name=source.dataset_name,
            split=source.split,
            revision=source.dataset_revision,
            max_scan_rows=source.max_scan_rows,
            max_state_chars=source.max_state_chars,
            report=report,
        )
    session = AdmissionSession(
        candidates,
        client_factory or (lambda: create_model_sexpr_client(env)),
        max_admitted=source.max_pool_size,
        time_budget_s=source.admission_time_budget_s,
        report=report,
        cache=ValidationCache(source.seed_cache_path, cache_environment),
        elaborate_candidate=resolver,
        max_replay_chars=source.max_replay_chars,
    )
    return session, report


def finalize_admission(source, items, report):
    ordered = ordered_items(items)
    contract = source.contract()
    identity = contract["source"]
    if source.data_source == "lean_source" and identity["selection_sha256"] != report.source["selection_sha256"]:
        raise ValueError("Lean source selection manifest changed during admission")
    return AdmittedSource(
        ordered,
        {
            "source": identity,
            "configuration": source.to_dict(),
            "contract": contract,
            "admitted": len(ordered),
            "pool_identity": pool_identity(ordered),
            "environment": report.environment,
            "report_sha256": digest(report.as_dict()),
            "summary": report.summary(),
        },
    )


async def admit_source(source, env, *, report_dir=None, client_factory=None):
    session, report = open_admission(source, env, client_factory=client_factory)
    try:
        while not session.done:
            await session.step(32)
    finally:
        await session.close()
    report_path = write_report(report, report_dir) if report_dir is not None else None
    if not session.admitted:
        raise RuntimeError(
            f"No seed passed Lean validation for {source.data_source} source {source.split or source.theorem_file or source.lean_source_manifest}"
        )
    admitted = finalize_admission(source, session.admitted, report)
    if report_path is not None:
        admitted.provenance["report_file"] = str(report_path)
    return admitted


def configured_sources(cfg):
    return {
        "training": cfg.training_data.contract() if cfg.training_data else None,
        "evaluation": (
            cfg.evaluation_data.contract()
            if cfg.eval_every > 0 and cfg.evaluation_data
            else None
        ),
    }


def data_contract(cfg, pool):
    return {
        "version": DATA_CONTRACT_VERSION,
        "sources": configured_sources(cfg),
        "training_pool": pool_identity(pool.train_items),
        "evaluation_pool": pool_identity(pool.eval_items),
    }


def validate_source_separation(training, evaluation):
    if training and training.data_source == "lean_source":
        raise ValueError("Lean source inputs are supported only in evaluation_data")
    if training and evaluation:
        if training.source_identity() == evaluation.source_identity():
            raise ValueError(
                "Training and evaluation must use independent sources, not the same split/file"
            )
        if (
            training.seed_cache_path
            and evaluation.seed_cache_path
            and training.seed_cache_path.resolve()
            == evaluation.seed_cache_path.resolve()
        ):
            raise ValueError(
                "Training and evaluation require separate admission cache files"
            )


def overlap_diagnostics(training, evaluation):
    def theorem(item):
        src = item.seed.source_dict
        return (
            (src.get("commit"), src.get("file_path"), src.get("theorem"))
            if src.get("theorem")
            else None
        )

    train_statements = {render_replay(item.seed).fingerprint for item in training}
    train_theorems = {theorem(item) for item in training} - {None}
    return {
        "matching_evaluation_statements": sum(
            render_replay(item.seed).fingerprint in train_statements
            for item in evaluation
        ),
        "matching_evaluation_theorems": sum(
            theorem(item) in train_theorems for item in evaluation
        ),
    }
