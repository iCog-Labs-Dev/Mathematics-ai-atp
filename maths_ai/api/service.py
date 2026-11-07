"""The service coordinator: bounded fair scheduling, deadlines and job state.

One coordinator owns all proof and evaluation admission. It counts queued and
active work toward explicit bounds and runs one runtime operation at a time:
an interactive proof, or one bounded step of the resident evaluation job (an
admission step or one evaluation chunk). Up to ``interactive_per_chunk``
waiting proofs run before one waiting evaluation step, so neither starves
the other. A proof's clock starts at admission; its queue wait is capped
separately and its search gets the time that remains.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import time
from dataclasses import asdict
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maths_ai.gnn_inference.atp_lean_gnn.evaluation_runtime import (
    EVAL_METRIC_VERSION,
    EvaluationSettings,
    merge_stats,
    repeat_row_for_seeds,
    summarize_rows,
)
from maths_ai.lean_replay import ReplayInputError, seed_from_text

from .config import ServiceConfig
from .job_store import (
    ACTIVE_STATES,
    ADMITTING,
    CANCELLED,
    CANCELLING,
    COMPLETED,
    FAILED,
    INTERRUPTED,
    QUEUED,
    RUNNING,
    TERMINAL_STATES,
    JobStore,
    JobCapacityError,
    ResultStorageError,
)
from .runtime_supervisor import (
    OperationFailed,
    OperationInterrupted,
    OperationTimeout,
    RuntimeSupervisor,
    RuntimeUnavailable,
)

logger = logging.getLogger("maths_ai.api.service")

SERVICE_RESULT_VERSION = "mcts-service-result-v1"


class ServiceError(RuntimeError):
    """A request the service refuses, with the HTTP status it maps to."""

    status = 500
    category = "internal"

    def __init__(self, detail: str, *, category: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if category is not None:
            self.category = category


class InputRejected(ServiceError):
    status, category = 422, "input"


class QueueFull(ServiceError):
    status, category = 429, "queue-full"


class Unavailable(ServiceError):
    status, category = 503, "unavailable"


class DeadlineExceeded(ServiceError):
    status, category = 504, "deadline"


class UnknownJob(ServiceError):
    status, category = 404, "unknown-job"


class NotFinished(ServiceError):
    status, category = 409, "not-finished"


class StateOwnershipError(RuntimeError):
    """Another service process owns this state directory."""


@dataclass
class _Proof:
    payload: dict[str, Any]
    admitted_at: float
    deadline: float
    queue_deadline: float
    future: asyncio.Future
    started: bool = False
    queue_timer: Any = None


@dataclass
class _ActiveJob:
    job_id: str
    settings: dict[str, Any]
    phase: str = ADMITTING
    repeat: int = 0
    start: int = 0
    subset_size: int = 0
    subset_sha256: str | None = None
    seeds: list[str] = field(default_factory=list)
    admission: dict[str, Any] | None = None
    repeat_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    cancel_requested: bool = False
    deadline: float = 0.0
    attempts_done: int = 0
    expired: bool = False
    input_records: list = field(default_factory=list)
    proposal_records: list = field(default_factory=list)


def _map_failure(exc: OperationFailed) -> ServiceError:
    if exc.category.startswith("input:"):
        return InputRejected(exc.detail, category=exc.category)
    if exc.category in ("busy", "state"):
        return Unavailable(exc.detail, category=exc.category)
    return ServiceError(exc.detail, category=exc.category)


class ProvingService:
    def __init__(self, config: ServiceConfig, config_path: Path, *, backend: str | None = None) -> None:
        self.config = config
        self.limits = config.limits
        self.store = JobStore(config.state_dir)
        self.supervisor = RuntimeSupervisor(
            config, config_path, backend=backend, on_restart=self._runtime_restarted
        )
        self._proofs: deque[_Proof] = deque()
        self._jobs: deque[str] = deque()
        self._active: _ActiveJob | None = None
        self._active_proofs = 0
        self._served_since_chunk = 0
        self._wakeup = asyncio.Event()
        self._dispatcher: asyncio.Task | None = None
        self._step: asyncio.Task | None = None
        self._lock_file = None
        self._stopping = False
        self._submissions: set[asyncio.Task] = set()
        self._job_timers: dict[str, asyncio.TimerHandle] = {}

    # Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._own_state_directory()
        interrupted = await self.store.open()
        if interrupted:
            logger.warning("interrupted %d unfinished jobs from a previous service run", len(interrupted))
        self.supervisor.start()
        self._dispatcher = asyncio.create_task(self._dispatch(), name="service-dispatcher")

    def _own_state_directory(self) -> None:
        state = Path(self.config.state_dir)
        state.mkdir(parents=True, exist_ok=True)
        handle = open(state / "service.lock", "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise StateOwnershipError(
                f"Another service process owns {state}; run exactly one service per state directory."
            ) from exc
        handle.write(str(os.getpid()))
        handle.flush()
        self._lock_file = handle

    async def stop(self) -> None:
        self._stopping = True
        await asyncio.gather(*self._submissions, return_exceptions=True)
        for timer in self._job_timers.values():
            timer.cancel()
        active_step = self._step
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
        if active_step is not None:
            active_step.cancel()
            await asyncio.gather(active_step, return_exceptions=True)
        for proof in self._proofs:
            proof.queue_timer.cancel()
            if not proof.future.done():
                proof.future.set_exception(Unavailable("The service is shutting down"))
        await self.supervisor.stop()
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    def readiness(self) -> dict[str, Any]:
        supervisor = self.supervisor
        status = supervisor.status or {}
        ready = supervisor.ready and status.get("usable_workers", 1) > 0
        return {
            "ready": ready,
            "runtime": supervisor.state,
            "generation": supervisor.generation,
            "error": supervisor.error,
            "status": status,
            "provenance": supervisor.provenance,
            "queue": {
                "pending_proofs": len(self._proofs) + self._active_proofs,
                "max_pending_proofs": self.limits.max_pending_proofs,
                "pending_jobs": len(self._jobs) + (self._active is not None),
                "max_pending_jobs": self.limits.max_pending_jobs,
            },
        }

    # Proofs --------------------------------------------------------------

    def resolve_prove(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate a proof request into the payload the runtime runs."""
        limits = self.limits
        if len(request["goal"]) > limits.max_goal_chars:
            raise InputRejected(f"goal exceeds {limits.max_goal_chars} characters", category="input:too-large")
        hypotheses = list(request.get("hypotheses") or [])
        if len(hypotheses) > limits.max_hypotheses:
            raise InputRejected(f"more than {limits.max_hypotheses} hypotheses", category="input:too-large")
        if any(len(h) > limits.max_hypothesis_chars for h in hypotheses):
            raise InputRejected(
                f"a hypothesis exceeds {limits.max_hypothesis_chars} characters", category="input:too-large"
            )
        try:
            seed_from_text(goal=request["goal"], hypotheses=hypotheses)
        except ReplayInputError as exc:
            raise InputRejected(str(exc), category=f"input:{exc.reason.value}") from exc
        except ValueError as exc:
            raise InputRejected(str(exc), category="input:malformed-text") from exc
        profile = self._profile(request.get("search_profile"), request)
        timeout = request.get("timeout_s") or limits.default_timeout_s
        queue_timeout = request.get("queue_timeout_s") or min(limits.default_queue_timeout_s, timeout)
        if timeout > limits.max_timeout_s:
            raise InputRejected(f"timeout_s exceeds {limits.max_timeout_s}", category="input:budget")
        if queue_timeout > limits.max_queue_timeout_s or queue_timeout > timeout:
            raise InputRejected("queue_timeout_s exceeds its limit or the request timeout",
                                category="input:budget")
        return {
            "goal": request["goal"],
            "hypotheses": hypotheses,
            "search_profile": profile[0],
            "profile": profile[1],
            "proposal_mode": request.get("proposal_mode") or "sample",
            "seed": int(request.get("seed") or 0),
            "timeout_s": float(timeout),
            "queue_timeout_s": float(queue_timeout),
        }

    def _profile(self, name: str | None, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = name or self.config.default_search_profile
        base = self.config.search_profiles.get(name)
        if base is None:
            raise InputRejected(f"unknown search profile {name!r}", category="input:profile")
        overrides = {key: request.get(key) for key in ("top_k_tactics", "max_depth", "max_nodes", "num_simulations")}
        try:
            resolved = base.with_overrides(overrides)
        except ValueError as exc:
            raise InputRejected(str(exc), category="input:budget") from exc
        return name, {key: getattr(resolved, key) for key in resolved.__dataclass_fields__}

    async def prove(self, request: dict[str, Any]) -> dict[str, Any]:
        """Admit, queue and run one proof; raises a ``ServiceError`` on refusal."""
        admitted_at = time.monotonic()
        payload = self.resolve_prove(request)
        if not self.supervisor.ready:
            raise Unavailable(self.supervisor.error or f"runtime is {self.supervisor.state}")
        if len(self._proofs) + self._active_proofs >= self.limits.max_pending_proofs:
            raise QueueFull("The proof queue is full; retry later")
        proof = _Proof(
            payload=payload,
            admitted_at=admitted_at,
            deadline=admitted_at + payload["timeout_s"],
            queue_deadline=admitted_at + payload["queue_timeout_s"],
            future=asyncio.get_running_loop().create_future(),
        )
        def expire_queue():
            if not proof.started and proof in self._proofs:
                self._proofs.remove(proof)
                self._fail(proof, DeadlineExceeded("The queue deadline expired", category="queue-timeout"))
        proof.queue_timer = asyncio.get_running_loop().call_at(proof.queue_deadline, expire_queue)
        self._proofs.append(proof)
        self._wakeup.set()
        try:
            async with asyncio.timeout_at(proof.deadline):
                return await asyncio.shield(proof.future)
        except TimeoutError:
            if not proof.started and proof in self._proofs:
                self._proofs.remove(proof)
            elif self._step is not None and getattr(self._step, "proof", None) is proof:
                self._step.cancel()
            proof.future.cancel()
            raise DeadlineExceeded("The request deadline expired")
        except asyncio.CancelledError:
            # The client went away: drop a queued proof, cancel a running one.
            if not proof.started and proof in self._proofs:
                self._proofs.remove(proof)
            elif self._step is not None and getattr(self._step, "proof", None) is proof:
                self._step.cancel()
            proof.future.cancel()
            raise
        finally:
            proof.queue_timer.cancel()

    async def _run_proof(self, proof: _Proof) -> None:
        proof.started = True
        proof.queue_timer.cancel()
        now = time.monotonic()
        try:
            queue_wait = now - proof.admitted_at
            remaining = proof.deadline - now
            payload = dict(proof.payload)
            # The search stops cooperatively before the response deadline, so
            # replay, certification and the reply fit inside it.
            if remaining <= 0:
                raise DeadlineExceeded("The request deadline expired before dispatch")
            payload["timeout_s"] = remaining * 0.8
            started = time.monotonic()
            result = await self.supervisor.run("prove", payload, timeout=remaining)
            outcome = result.get("outcome")
            if outcome == "inadmissible":
                raise InputRejected(
                    "Lean could not admit the goal and its declarations",
                    category=f"input:{result.get('exception_category') or 'lean-rejected'}",
                )
            if outcome == "hard_timeout":
                raise DeadlineExceeded(
                    "The search did not stop by its deadline; its worker was replaced",
                    category="hard-timeout",
                )
            if outcome == "error":
                raise Unavailable(
                    f"The search failed in the runtime ({result.get('exception_category')})",
                    category="search-infrastructure",
                )
            result = {
                **result,
                "search_profile": payload["search_profile"],
                "timing": {
                    "queue_wait_s": queue_wait,
                    "execution_s": time.monotonic() - started,
                    "total_s": time.monotonic() - proof.admitted_at,
                },
                "runtime": {"generation": self.supervisor.generation, "provenance": self.supervisor.provenance},
            }
            if not proof.future.done():
                proof.future.set_result(result)
        except ServiceError as exc:
            self._fail(proof, exc)
        except OperationTimeout as exc:
            self._fail(proof, DeadlineExceeded(str(exc)))
        except OperationFailed as exc:
            self._fail(proof, _map_failure(exc))
        except (RuntimeUnavailable, OperationInterrupted) as exc:
            self._fail(proof, Unavailable(str(exc), category="runtime-interrupted"))
        except asyncio.CancelledError:
            self._fail(proof, DeadlineExceeded("The request was cancelled", category="cancelled"))
            raise
        except Exception as exc:
            logger.exception("proof operation failed")
            self._fail(proof, ServiceError(str(exc), category="internal"))

    @staticmethod
    def _fail(proof: _Proof, error: ServiceError) -> None:
        if not proof.future.done():
            proof.future.set_exception(error)

    # Evaluation jobs -----------------------------------------------------

    async def submit_evaluation(self, request: dict[str, Any]) -> dict[str, Any]:
        # Once admitted, a dataset job is independent of the submitting connection.
        if len(self._submissions) >= self.limits.max_pending_jobs:
            raise QueueFull("The evaluation admission queue is full")
        task = asyncio.create_task(self._submit_evaluation(request))
        self._submissions.add(task)
        task.add_done_callback(self._submissions.discard)
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(task)

    async def _submit_evaluation(self, request: dict[str, Any]) -> dict[str, Any]:
        limits = self.limits
        dataset = self.config.dataset_profiles.get(request["dataset_profile"])
        if dataset is None:
            raise InputRejected(f"unknown dataset profile {request['dataset_profile']!r}", category="input:profile")
        name, profile = self._profile(request.get("search_profile"), request)
        repeats = int(request.get("repeats") or 1)
        if repeats > limits.max_repeats:
            raise InputRejected(f"repeats exceeds {limits.max_repeats}", category="input:budget")
        timeout = request.get("timeout_s") or limits.default_timeout_s
        if timeout > limits.max_timeout_s:
            raise InputRejected(f"timeout_s exceeds {limits.max_timeout_s}", category="input:budget")
        deadline = request.get("deadline_s") or limits.max_job_deadline_s
        if deadline > limits.max_job_deadline_s:
            raise InputRejected(f"deadline_s exceeds {limits.max_job_deadline_s}", category="input:budget")
        try:
            evaluation = EvaluationSettings(
                mode=request.get("proposal_mode") or "sample",
                seed=int(request.get("seed") if request.get("seed") is not None else 42),
                repeats=repeats,
            )
        except ValueError as exc:
            raise InputRejected(str(exc), category="input:budget") from exc
        settings = {
            "search_profile": name,
            "profile": profile,
            "dataset_profile": request["dataset_profile"],
            "dataset": dataset.to_dict(),
            "evaluation": {"mode": evaluation.mode, "seed": evaluation.seed, "repeats": evaluation.repeats},
            "evaluation_contract": evaluation.contract(),
            "timeout_s": float(timeout),
            "deadline_s": float(deadline),
            "deadline_at": time.time() + float(deadline),
            "chunk_size": limits.eval_chunk_size,
            "metric": EVAL_METRIC_VERSION,
            "pln_runtime": asdict(self.config.pln) if profile["use_pln"] else None,
        }
        try:
            job_id = await self.store.create(dict(request), settings, capacity=limits.max_pending_jobs)
        except JobCapacityError as exc:
            raise QueueFull(str(exc)) from exc
        self._job_timers[job_id] = asyncio.get_running_loop().call_later(
            max(settings["deadline_at"] - time.time(), 0),
            lambda: asyncio.create_task(self._expire_job(job_id)),
        )
        self._jobs.append(job_id)
        self._wakeup.set()
        return await self.store.get(job_id)

    async def _expire_job(self, job_id):
        job = await self.store.get(job_id)
        if job is None or job["state"] in TERMINAL_STATES:
            return
        if job_id in self._jobs:
            self._jobs.remove(job_id)
            await self._finish(job_id, FAILED, None, error=("deadline", "The queued job expired"))
        elif self._active is not None and self._active.job_id == job_id:
            self._active.expired = self._active.cancel_requested = True
            await self.store.transition(job_id, CANCELLING, phase="deadline-cleanup")
            if self._step is not None and getattr(self._step, "job_id", None) == job_id:
                self._step.cancel()
        self._wakeup.set()

    async def job(self, job_id: str) -> dict[str, Any]:
        job = await self.store.get(job_id)
        if job is None:
            raise UnknownJob(f"no job {job_id}")
        return job

    async def job_result(self, job_id: str) -> dict[str, Any]:
        job = await self.job(job_id)
        if job["state"] not in TERMINAL_STATES:
            raise NotFinished(f"job {job_id} is {job['state']}")
        try:
            result = await self.store.result(job_id)
        except ResultStorageError as exc:
            raise ServiceError(str(exc), category="result-storage") from exc
        if result is None:
            raise NotFinished(f"job {job_id} has no published result", category="no-result")
        return result

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = await self.job(job_id)
        if job["state"] in TERMINAL_STATES:
            return job
        if job_id in self._jobs:
            self._jobs.remove(job_id)
            return await self._finish(job_id, CANCELLED, None, error=("cancelled", "Cancelled before it ran"))
        active = self._active
        if active is not None and active.job_id == job_id:
            active.cancel_requested = True
            job = await self.store.transition(job_id, CANCELLING, phase="cancelling")
            if self._step is not None and getattr(self._step, "job_id", None) == job_id:
                self._step.cancel()
            self._wakeup.set()
        return job

    # Dispatch ------------------------------------------------------------

    async def _dispatch(self) -> None:
        """Run one operation at a time, alternating fairly between proofs and jobs."""
        while True:
            self._expire_queued_proofs()
            job_work = self._active is not None or bool(self._jobs)
            proof_work = bool(self._proofs)
            if not job_work and not proof_work:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            await self.supervisor.settled()
            if not self.supervisor.ready:
                await asyncio.sleep(0.2)
                continue
            job_work = self._active is not None or bool(self._jobs)
            proof_work = bool(self._proofs)
            if not job_work and not proof_work:
                continue
            take_proof = proof_work and (
                not job_work or self._served_since_chunk < self.limits.interactive_per_chunk
            )
            if take_proof:
                proof = self._proofs.popleft()
                self._served_since_chunk += 1
                self._active_proofs += 1
                self._step = asyncio.create_task(self._run_proof(proof))
                self._step.proof = proof
            else:
                self._served_since_chunk = 0
                self._step = asyncio.create_task(self._job_step())
                self._step.job_id = self._active.job_id if self._active else (self._jobs[0] if self._jobs else None)
            try:
                await asyncio.shield(self._step)
            except asyncio.CancelledError:
                if not self._step.done():
                    raise  # the dispatcher itself is being cancelled
            except Exception:  # noqa: BLE001 — one failed step does not stop the service
                logger.exception("service step failed")
            finally:
                if getattr(self._step, "proof", None) is not None:
                    self._active_proofs -= 1
                self._step = None

    def _expire_queued_proofs(self) -> None:
        now = time.monotonic()
        for proof in list(self._proofs):
            if now >= proof.queue_deadline:
                self._proofs.remove(proof)
                self._fail(proof, DeadlineExceeded("The request waited longer than its queue timeout",
                                                   category="queue-timeout"))

    async def _job_step(self) -> None:
        if self._active is None:
            job_id = self._jobs.popleft()
            job = await self.store.get(job_id)
            if job is None or job["state"] != QUEUED:
                return
            self._active = _ActiveJob(
                job_id, job["settings"], deadline=time.monotonic() + (job["deadline_at"] - time.time())
            )
            if time.monotonic() >= self._active.deadline:
                await self._close_active(FAILED, ("deadline", "The queued job expired"))
                return
            await self.store.transition(job_id, ADMITTING, phase="admitting",
                                        runtime_generation=self.supervisor.generation,
                                        provenance=self.supervisor.provenance)
        active = self._active
        if active.cancel_requested:
            await self._close_active(FAILED if active.expired else CANCELLED,
                                     ("deadline" if active.expired else "cancelled", "Execution stopped by request"))
            return
        if time.monotonic() >= active.deadline:
            await self._close_active(FAILED, ("deadline", "The job exceeded its deadline"))
            return
        try:
            if active.phase == ADMITTING:
                await self._admit_step(active)
            else:
                await self._evaluate_step(active)
        except asyncio.CancelledError:
            if active.cancel_requested:
                await self._close_active(FAILED if active.expired else CANCELLED,
                                     ("deadline" if active.expired else "cancelled", "Execution stopped by request"))
                return
            raise
        except OperationTimeout as exc:
            await self._close_active(FAILED, ("timeout", str(exc)))
        except OperationFailed as exc:
            await self._close_active(FAILED, (exc.category, exc.detail))
        except (RuntimeUnavailable, OperationInterrupted) as exc:
            category = "service-shutdown" if self._stopping else "runtime-interrupted"
            await self._close_active(INTERRUPTED, (category, str(exc)))
        except Exception as exc:
            logger.exception("evaluation operation failed")
            await self._close_active(FAILED, ("internal", str(exc)))

    async def _admit_step(self, active: _ActiveJob) -> None:
        settings = active.settings
        remaining = active.deadline - time.monotonic()
        progress = await self.supervisor.run(
            "admit",
            {
                "job_id": active.job_id,
                "dataset_profile": settings["dataset"],
                "rows": self.limits.admission_step_rows,
                "seed": settings["evaluation"]["seed"],
                "evaluation": settings["evaluation"],
            },
            timeout=min(remaining, settings["dataset"]["admission_time_budget_s"]),
        )
        if not progress["done"]:
            await self.store.transition(active.job_id, ADMITTING, progress={"phase": "admitting", "admission": progress["admission"],
                                                  "rows_seen": progress["rows_seen"], "admitted": progress["admitted"]})
            return
        active.subset_size = int(progress["eval_items"])
        active.subset_sha256 = progress["eval_subset_sha256"]
        active.seeds = list(progress["proposal_seeds_sha256"])
        active.admission = progress.get("admission")
        active.input_records = progress["input_records"]
        active.proposal_records = progress["proposal_records"]
        if active.subset_size == 0:
            await self._close_active(FAILED, ("no-admitted-inputs", "No seed passed Lean admission"))
            return
        active.phase = RUNNING
        await self.store.transition(
            active.job_id, RUNNING, phase="evaluating",
            progress=self._progress(active),
        )

    async def _evaluate_step(self, active: _ActiveJob) -> None:
        settings = active.settings
        count = min(settings["chunk_size"], active.subset_size - active.start)
        remaining = active.deadline - time.monotonic()
        per_search = min(settings["timeout_s"], remaining)
        # A chunk runs its searches over the pool's workers; bound it by the job deadline.
        result = await self.supervisor.run(
            "evaluate",
            {
                "job_id": active.job_id,
                "profile": settings["profile"],
                "proposal_mode": settings["evaluation"]["mode"],
                "seed": settings["evaluation"]["seed"],
                "repeats": settings["evaluation"]["repeats"],
                "repeat": active.repeat,
                "start": active.start,
                "count": count,
                "timeout_s": per_search,
            },
            timeout=min(remaining, per_search * 1.25 * count + 60.0),
        )
        merge_stats(active.repeat_stats.setdefault(active.repeat, {}), result["stats"])
        active.attempts_done += len(result["attempts"])
        active.start += count
        repeat = active.repeat
        if active.start >= active.subset_size:
            active.repeat += 1
            active.start = 0
        await self.store.record_chunk(
            active.job_id, repeat, result["attempts"], active.repeat_stats[repeat], self._progress(active)
        )
        if active.repeat >= settings["evaluation"]["repeats"]:
            await self._close_active(COMPLETED, None)

    def _progress(self, active: _ActiveJob) -> dict[str, Any]:
        repeats = active.settings["evaluation"]["repeats"]
        return {
            "phase": "evaluating" if active.phase == RUNNING else "admitting",
            "subset_size": active.subset_size,
            "proposal_seeds_sha256": active.seeds,
            "admission": active.admission,
            "input_records": active.input_records,
            "proposal_records": active.proposal_records,
            "subset_sha256": active.subset_sha256,
            "repeats": repeats,
            "repeats_completed": min(active.repeat, repeats),
            "attempts_completed": active.attempts_done,
            "attempts_total": active.subset_size * repeats,
        }

    async def _close_active(self, state: str, error: tuple[str, str] | None) -> None:
        active, self._active = self._active, None
        if active is None:
            return
        await self.supervisor.settled()
        try:
            if self.supervisor.ready:
                await self.supervisor.run("release", {"job_id": active.job_id}, timeout=self.limits.cleanup_grace_s)
        except (OperationTimeout, OperationFailed, RuntimeUnavailable, OperationInterrupted):
            logger.warning("could not release job %s in the runtime", active.job_id)
        await self._finish(active.job_id, state, active, error=error)

    async def _finish(self, job_id: str, state: str, active: _ActiveJob | None,
                      *, error: tuple[str, str] | None) -> dict[str, Any]:
        job = await self.store.get(job_id)
        if job["state"] in TERMINAL_STATES:
            return job
        attempts, stats = await self.store.attempts(job_id)
        settings = job["settings"]
        from .evaluation_results import evaluation_result
        result = evaluation_result(job, state, attempts, stats)
        timer = self._job_timers.pop(job_id, None)
        if timer is not None:
            timer.cancel()
        fields: dict[str, Any] = {"phase": state}
        if error is not None:
            fields["error_category"], fields["error_detail"] = error
        return await self.store.finish(job_id, state, result, partial=state != COMPLETED, **fields)

    async def _runtime_restarted(self, generation: int) -> None:
        """A runtime generation ended: its active job is interrupted, queued work waits."""
        active = self._active
        if active is not None:
            self._active = None
            await self._finish(active.job_id, INTERRUPTED, active,
                               error=("runtime-restart", f"runtime generation {generation} ended"))


__all__ = [
    "ACTIVE_STATES",
    "CANCELLING",
    "DeadlineExceeded",
    "InputRejected",
    "NotFinished",
    "ProvingService",
    "QueueFull",
    "ServiceError",
    "StateOwnershipError",
    "Unavailable",
    "UnknownJob",
]
