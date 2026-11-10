"""The runtime process's inference backend over the shared MCTS runtime.

One ``LiveBackend`` loads the frozen policy, retrieval provider and worker
pool once (``live_runtime.open_live_runtime``) and serves proof requests and
evaluation steps through the same ``collect_round`` the CLI uses. It never
creates an optimizer or changes weights.
"""

from __future__ import annotations

import os
import time
from typing import Any

from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import (
    EvaluationSettings,
    chunk_stats,
    proposal_seeds_sha256,
    search_counters,
    subset_identity,
)
from maths_ai.gnn_inference.atp_lean_gnn.live_runtime import LiveRuntime, open_live_runtime
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import RLTrainingConfig
from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import (
    AdmissionReport,
    AdmissionSession,
    TheoremItem,
    environment_identity,
)
from maths_ai.hybrid_reasoner.proposal import proposal_specs, input_identity
from maths_ai.lean_replay import ReplayInputError, seed_from_text

from .config import DEVICE_ENV, SearchProfile, ServiceConfig, serving_device
from maths_ai.gnn_inference.atp_lean_gnn.theorem_sources import TheoremSourceConfig, open_admission, finalize_admission
from maths_ai.hybrid_reasoner.search_results import plain, record_view


class BackendError(RuntimeError):
    """An operation failed for a typed reason the API maps to a response."""

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


def _seed(goal: str, hypotheses: list[str]):
    try:
        return seed_from_text(goal=goal, hypotheses=hypotheses, source={"origin": "api"})
    except ReplayInputError as exc:
        raise BackendError(f"input:{exc.reason.value}", str(exc)) from exc
    except ValueError as exc:
        raise BackendError("input:malformed-text", str(exc)) from exc


class _EvaluationJob:
    """One resident job: its admission session, then its fixed evaluation subset."""

    def __init__(self, session: AdmissionSession, report: AdmissionReport, profile: TheoremSourceConfig, seed: int):
        self.session = session
        self.report = report
        self.profile = profile
        self.seed = seed
        self.items: list[TheoremItem] | None = None


class LiveBackend:
    """Serves proofs and evaluation steps from one loaded runtime."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.cfg = serving_device(RLTrainingConfig.from_json(config.rl_config), os.environ.get(DEVICE_ENV))
        self.runtime: LiveRuntime | None = None
        self.jobs: dict[str, _EvaluationJob] = {}

    async def start(self) -> dict[str, Any]:
        default = self.config.search_profiles[self.config.default_search_profile]
        self.runtime = await open_live_runtime(
            self.cfg, checkpoint=self.config.checkpoint, search_kwargs=default.search_kwargs(),
            pln_spec=self.config.pln, enable_pln=any(p.use_pln for p in self.config.search_profiles.values())
        )
        return self.provenance()

    def provenance(self) -> dict[str, Any]:
        runtime = self.runtime
        policy = runtime.policy
        return plain(
            {
                "checkpoint": str(policy.checkpoint),
                "checkpoint_sha256": policy.checkpoint_sha256,
                "model": {
                    "architecture": getattr(policy.model_spec, "architecture", None),
                    "retrieval_binding": getattr(policy.model_spec, "retrieval_binding", None),
                    "parameter_bytes": policy.parameter_bytes,
                },
                "graph_representation": policy.graph_representation.to_dict(),
                "environment": environment_identity(runtime.env),
                "workers": len(runtime.pool.active_workers),
                "pln": runtime.pool.pln_identity,
                "devices": [str(device) for device in runtime.plan.collection_devices],
            }
        )

    def status(self) -> dict[str, Any]:
        runtime = self.runtime
        workers = 0 if runtime is None else len(runtime.pool.active_workers)
        admitting = sum(1 for job in self.jobs.values() if job.session.has_client)
        return {
            "workers": workers,
            "usable_workers": 0 if runtime is None else sum(
                1 for worker in runtime.pool.active_workers if getattr(worker.server, "is_usable", True)
            ),
            "admission_clients": admitting,
            "live_clients": workers + admitting,
            "configured_clients": workers + bool(self.config.dataset_profiles),
            "resident_jobs": len(self.jobs),
        }

    def _use_profile(self, profile: dict[str, Any]) -> SearchProfile:
        resolved = SearchProfile(**profile)
        self.runtime.use_search_settings(resolved.search_kwargs())
        return resolved

    async def prove(self, payload: dict[str, Any]) -> dict[str, Any]:
        seed = _seed(payload["goal"], payload["hypotheses"])
        profile = self._use_profile(payload["profile"])
        item = TheoremItem(seed=seed, tactic_label="", size=len(payload["goal"]))
        (proposal,) = proposal_specs(
            [seed], mode=payload["proposal_mode"], namespace="prove",
            base_seed=payload["seed"], pass_index=0,
        )
        pool = self.runtime.pool
        collected = await pool.collect_round(
            [item],
            timeout_s=float(payload["timeout_s"]),
            proposals=[proposal],
            model_generation=self.runtime.model_generation,
        )
        (record,) = collected.records
        view = record_view(record, proposal=proposal)
        view["search_settings"] = profile.search_kwargs()
        view["pln_runtime"] = self.runtime.pool.pln_identity
        view["input_identity"] = subset_identity([item])
        return view

    async def admit(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = payload["job_id"]
        job = self.jobs.get(job_id)
        if job is None:
            if self.jobs:
                raise BackendError("busy", "Another evaluation session is resident")
            job = self._open_admission(job_id, TheoremSourceConfig(**payload["dataset_profile"]), payload["seed"])
        newly = await job.session.step(int(payload["rows"]))
        progress = {
            "rows_seen": job.session.rows_seen,
            "scanned": job.report.scanned,
            "admitted": len(job.session.admitted),
            "newly_admitted": len(newly),
            "done": job.session.done,
            "admission": plain(job.report.as_dict()),
        }
        if job.session.done:
            await job.session.close()
            admitted = finalize_admission(job.profile, job.session.admitted, job.report)
            job.items = admitted.items
            settings = EvaluationSettings(**payload["evaluation"])
            progress.update(
                input_records=[{"identity": input_identity(item.seed), "source": plain(item.seed.source_dict)} for item in job.items],
                proposal_records=[[spec.as_dict() for spec in settings.proposals(job.items, repeat)] for repeat in range(settings.repeats)],
                eval_items=len(job.items),
                eval_subset_sha256=subset_identity(job.items) if job.items else None,
                proposal_seeds_sha256=[
                    proposal_seeds_sha256(settings.proposals(job.items, repeat))
                    for repeat in range(settings.repeats)
                ],
                admission={**plain(job.report.as_dict()), "source_contract": admitted.provenance},
            )
        return progress

    def _open_admission(self, job_id: str, profile: TheoremSourceConfig, seed: int) -> _EvaluationJob:
        if len(self.runtime.pool.active_workers) + 2 > self.runtime.plan.usable_cpu_count:
            raise BackendError("resources", "Admission needs capacity for its decoder process and Lean client")
        session, report = open_admission(profile, self.runtime.env)
        job = _EvaluationJob(session, report, profile, seed)
        self.jobs[job_id] = job
        return job

    async def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        job = self.jobs.get(payload["job_id"])
        if job is None or job.items is None:
            raise BackendError("state", "The job has no admitted evaluation subset")
        self._use_profile(payload["profile"])
        settings = EvaluationSettings(
            mode=payload["proposal_mode"], seed=payload["seed"], repeats=payload["repeats"]
        )
        start, count, repeat = payload["start"], payload["count"], payload["repeat"]
        proposals = settings.proposals(job.items, repeat)[start:start + count]
        items = job.items[start:start + count]
        started = time.perf_counter()
        collected = await self.runtime.pool.collect_round(
            items,
            timeout_s=float(payload["timeout_s"]),
            proposals=proposals,
            model_generation=self.runtime.model_generation,
        )
        attempts = []
        for index, (record, proposal) in enumerate(zip(collected.records, proposals)):
            view = record_view(record, proposal=proposal, include_trace=False)
            view["input_identity"] = input_identity(items[index].seed)
            view["source"] = plain(items[index].seed.source_dict)
            view["index"] = start + index
            view["repeat"] = repeat
            attempts.append(view)
        stats = {**collected.stats, **search_counters(collected.results)}
        return {
            "attempts": attempts,
            "stats": chunk_stats(stats),
            "chunk_s": time.perf_counter() - started,
        }

    async def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        job = self.jobs.pop(payload["job_id"], None)
        if job is not None:
            await job.session.close()
        return {"released": job is not None}

    async def close(self) -> None:
        for job_id in list(self.jobs):
            await self.release({"job_id": job_id})
        if self.runtime is not None:
            await self.runtime.close()
