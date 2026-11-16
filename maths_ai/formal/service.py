"""The provider's formal-run service (protocol version 2).

One registry (``FormalRunStore``) owns coordinator-submitted runs; dataset
evaluation jobs stay separate. A run is admitted once:

1. The request is validated against the deployment registry: the backend,
   profile and input kind must be offered there, the environment must be the
   one the backend actually runs (its content hash), and any model or corpus
   assertion must equal the deployment's.
2. The tactic allowance is ``min(budget.max_steps, profile ceiling)`` and the
   wall clock ``min(budget.wall_clock_ms or profile default, ceiling)``; any
   reduction is acknowledged. Both are persisted with the profile's
   configuration hash.
3. The input is admitted (``admission``) into a content-addressed manifest,
   then an attempt runs on the selected backend with the run's remaining
   allowance. Its wall clock starts at that attempt's admission and includes
   queueing.

A repeated start with the same request returns the existing run; the same run
id with a different request is a conflict. Retry starts another attempt from
the bound initial input with only the remaining allowance, under the same
profile configuration and environment; frontier resume is not offered.
Cancellation is acknowledged at once and reported terminal only after the
backend's work has stopped. After a restart, unfinished attempts are
interrupted, charged their whole reservation, and never restarted without a
retry.

Replay loads the stored certificate and admitted input by hash, re-checks
every binding, and verifies in a fresh native session under the run's
environment. The attestation reports identities derived from that evidence,
never from the request.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from .admission import AdmissionRejected, AdmittedTask, admit_goal, admit_source, manifest_task
from .artifacts import ArtifactError, ArtifactStore
from .backends import AttemptOutcome
from .canonical import object_ref
from .contract import (
    PROTOCOL_VERSION,
    Admission,
    FormalReplayRequest,
    FormalSearchRequest,
    BackendSelection,
    ReplayAttestation,
)
from .deployments import Deployment, DeploymentRegistry, Profile
from .results import DISPOSITIONS, build_result
from .store import FormalRunStore, RunRecord

logger = logging.getLogger("maths_ai.formal")

_MODEL_KEYS = ("tactic_model_artifact", "argument_model_artifact", "premise_model_artifact")
_CORPUS_KEYS = ("premise_corpus_artifact", "retrieval_policy")
QUIESCE_TIMEOUT_S = 120.0


class FormalError(Exception):
    def __init__(self, status: int, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.status = status
        self.category = category
        self.detail = detail


def environment_hash(environment_id: str, manifest: dict[str, Any]) -> str:
    """The content hash of a backend runtime, as requests and attestations carry it."""
    return object_ref({"environment_id": environment_id, "manifest": manifest})


class FormalService:
    def __init__(self, *, registry: DeploymentRegistry, state_dir: Path, backends: dict[str, Any]) -> None:
        self.registry = registry
        self.backends = dict(backends)
        state_dir = Path(state_dir)
        self.artifacts = ArtifactStore(state_dir / "artifacts")
        self.runs = FormalRunStore(state_dir / "formal_runs.sqlite3")
        self._tasks: dict[str, asyncio.Task] = {}
        self._recover()

    # Lifecycle -----------------------------------------------------------------

    def _recover(self) -> None:
        for record in self.runs.interrupted_runs():
            attempt = self.runs.open_attempt(record.run_id)
            result = self._result(record, termination="interrupted",
                                  error="The service stopped before the attempt finished.",
                                  charged=record.remaining_steps if attempt is not None else 0,
                                  exact=attempt is None, attempt_cost={"interrupted": True})
            fields = {"status": "interrupted", "disposition": DISPOSITIONS["interrupted"],
                      "termination": "interrupted", "result": result}
            if attempt is None:
                self.runs.update(record.run_id, **fields)
            else:
                self.runs.settle_attempt(record.run_id, attempt, used_steps=None,
                                         outcome="interrupted", run_fields=fields)

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.runs.close()

    # Capabilities --------------------------------------------------------------

    def _environment(self, deployment: Deployment) -> Optional[str]:
        backend = self.backends.get(deployment.backend)
        if backend is None:
            return None
        manifest = backend.environment_manifest()
        return None if manifest is None else environment_hash(deployment.environment_id, manifest)

    def capabilities(self) -> dict[str, Any]:
        deployments = {}
        for deployment in self.registry.deployments.values():
            backend = self.backends.get(deployment.backend)
            deployments[deployment.deployment_id] = {
                **deployment.as_dict(),
                "available": backend is not None and backend.ready(),
                "environment_hash": self._environment(deployment),
                "profile_config_hashes": {
                    name: deployment.profile_config_hash(profile)
                    for name, profile in deployment.profiles.items()
                },
            }
        return {
            "protocol_versions": [PROTOCOL_VERSION],
            "default": dict(self.registry.default) if self.registry.default is not None else None,
            "registry_hash": self.registry.registry_hash,
            "deployments": deployments,
            "tactic_step": "one tactic command sent to Lean, including rejected commands "
                           "and proof-combination replay",
            "budget_fields": ["budget.max_steps", "budget.wall_clock_ms"],
            "wall_clock": "per attempt, from that attempt's admission, including queueing",
            "cancellation": "asynchronous; terminal after the backend's work has stopped",
            "retry": "bound-initial-input with the remaining admitted allowance",
            "frontier_resume": False,
            "replay": "independent, fresh native session, bound to stored evidence",
            "artifact_upload": {"route": "/formal-artifacts", "max_bytes": self.artifacts.max_bytes},
        }

    # Admission -----------------------------------------------------------------

    def _ack(self, record: RunRecord) -> dict[str, Any]:
        deployment = self.registry.deployments.get(record.deployment_id)
        verification = None if deployment is None else deployment.verification_max_steps
        ack = {
            "protocol_version": 2,
            "run_id": record.run_id,
            "status": record.status,
            "request_digest": record.request_digest,
            "input_kind": record.input_kind,
            "input_artifact_hash": record.input_artifact_hash,
            "source_artifact_hash": record.source_artifact_hash,
            "source_origin": record.source_origin,
            "resolved": {
                "deployment_id": record.deployment_id,
                "backend": record.backend,
                "profile": record.profile,
                "profile_config_hash": record.profile_config_hash,
                "environment_id": record.environment_id,
                "environment_hash": record.environment_hash,
                "model": record.model,
                "proposal_mode": (record.backend_selection or record.request.get("backend") or {}).get("proposal_mode"),
                "seed": (record.backend_selection or record.request.get("backend") or {}).get("seed"),
            },
            "budget": {
                "requested_max_steps": record.requested_max_steps,
                "admitted_max_steps": record.admitted_max_steps,
                "max_steps_reduced": record.admitted_max_steps < record.requested_max_steps,
                "requested_wall_clock_ms": record.requested_wall_clock_ms,
                "admitted_wall_clock_ms": record.admitted_wall_clock_ms,
                "wall_clock_reduced": (
                    record.requested_wall_clock_ms is not None
                    and record.admitted_wall_clock_ms < record.requested_wall_clock_ms
                ),
                "consumed_steps": record.consumed_steps,
                "consumed_steps_exact": record.consumed_exact,
                "remaining_steps": record.remaining_steps,
                "verification_max_steps": verification,
            },
            "attempts": record.attempts,
            "retry_from": "bound-initial-input",
        }
        return Admission.model_validate(ack).model_dump(mode="json")

    def _selection(self, request: FormalSearchRequest):
        default = self.registry.default or {}
        deployment_id = request.deployment_id or default.get("deployment_id")
        deployment = self.registry.deployments.get(deployment_id)
        if deployment is None:
            raise FormalError(422, "unknown-deployment", f"No deployment {deployment_id!r}; configure a default or select one.")
        selection = request.backend
        if selection is None:
            if default.get("deployment_id") == deployment_id:
                selected = {key: value for key, value in default.items() if key != "deployment_id"}
            elif len(deployment.profiles) == 1:
                selected = {"profile": next(iter(deployment.profiles))}
            else:
                raise FormalError(422, "profile-required", "Select a profile for this non-default deployment.")
            selection = BackendSelection(name=deployment.backend, **selected)
        if selection.name != deployment.backend:
            raise FormalError(
                422, "backend-mismatch",
                f"Deployment {deployment.deployment_id} serves {deployment.backend}, not {selection.name}.",
            )
        if deployment.backend == "accepted-gnn":
            if selection.proposal_mode not in (None, "greedy") or selection.seed is not None:
                raise FormalError(422, "unsupported-proposal", "The accepted backend is greedy and does not use a seed.")
            selection = selection.model_copy(update={"proposal_mode": "greedy"})
        else:
            selection = selection.model_copy(update={"proposal_mode": selection.proposal_mode or "sample", "seed": selection.seed if selection.seed is not None else 0})
        return deployment, selection

    def _resolve(self, request: FormalSearchRequest):
        deployment, selection = self._selection(request)
        profile = deployment.profiles.get(selection.profile)
        if profile is None:
            raise FormalError(
                422, "unknown-profile",
                f"Deployment {deployment.deployment_id} has no profile {selection.profile!r}.",
            )
        if request.input.kind not in deployment.inputs:
            raise FormalError(
                422, "unsupported-input",
                f"Deployment {deployment.deployment_id} admits {list(deployment.inputs)} inputs, "
                f"not {request.input.kind!r}.",
            )
        backend = self.backends.get(deployment.backend)
        if backend is None or not backend.ready():
            raise FormalError(503, "backend-unavailable", f"{deployment.backend} is not available.")
        if request.environment_id != deployment.environment_id:
            raise FormalError(
                409, "environment-mismatch",
                f"Deployment {deployment.deployment_id} runs {deployment.environment_id}, "
                f"not {request.environment_id}.",
            )
        actual = self._environment(deployment)
        if actual is None:
            raise FormalError(503, "backend-unavailable", "The backend has not reported its environment.")
        if request.environment_hash != actual:
            raise FormalError(
                409, "environment-mismatch",
                f"The backend's environment hash is {actual}, not {request.environment_hash}.",
            )
        assertions = {}
        if request.model_bundle is not None:
            assertions.update(request.model_bundle.model_dump(exclude_none=True))
        if request.corpus_manifest is not None:
            assertions.update(request.corpus_manifest.model_dump(exclude_none=True))
        for key, value in assertions.items():
            if deployment.model.get(key) != value:
                raise FormalError(
                    409, "model-mismatch",
                    f"Deployment {deployment.deployment_id} does not use {key} {value}.",
                )
        return deployment, profile, backend, actual, selection

    async def start(self, request: FormalSearchRequest) -> tuple[dict[str, Any], bool]:
        payload = request.model_dump(mode="json")
        digest = object_ref(payload)
        existing = self.runs.get(request.run_id)
        if existing is not None:
            return self._reconcile(existing, digest), False
        deployment, profile, backend, env_hash, selection = self._resolve(request)
        requested_wall = request.budget.wall_clock_ms
        record = RunRecord(
            run_id=request.run_id, request_digest=digest, request=payload,
            deployment_id=deployment.deployment_id, backend=deployment.backend,
            profile=profile.name, profile_config_hash=deployment.profile_config_hash(profile),
            environment_id=deployment.environment_id, environment_hash=env_hash,
            model=dict(deployment.model), input_kind=request.input.kind,
            input_artifact_hash=None, source_artifact_hash=None, source_origin=None,
            requested_max_steps=request.budget.max_steps,
            admitted_max_steps=min(request.budget.max_steps, profile.max_steps),
            consumed_steps=0, consumed_exact=True,
            requested_wall_clock_ms=requested_wall,
            admitted_wall_clock_ms=min(requested_wall or profile.default_wall_clock_ms, profile.wall_clock_ms),
            status="admitting", disposition=None, termination=None, result=None,
            certificate_hash=None, attempts=0, cancel_requested=False,
            created_at=time.time(), updated_at=time.time(),
            backend_selection=selection.model_dump(mode="json"),
        )
        if not self.runs.insert(record):
            return self._reconcile(self.runs.get(request.run_id), digest), False
        environment = {"environment_id": deployment.environment_id, "environment_hash": env_hash}
        if request.input.kind == "goal":
            try:
                task = admit_goal(
                    request.input, store=self.artifacts, imports=backend.imports(),
                    environment=environment, supports_universes=backend.supports_universes,
                )
            except AdmissionRejected as exc:
                self._reject(request.run_id, exc)
                return self._ack(self.runs.get(request.run_id)), True
            self._bind_input(request.run_id, task)
            self._spawn(request.run_id, self._attempt(request.run_id, task))
        else:
            self._spawn(request.run_id, self._admit_source_then_attempt(request, backend, environment))
        return self._ack(self.runs.get(request.run_id)), True

    def _reconcile(self, existing: RunRecord, digest: str) -> dict[str, Any]:
        if existing.request_digest != digest:
            raise FormalError(
                409, "run-conflict",
                f"Run {existing.run_id} exists with a different request ({existing.request_digest}).",
            )
        return self._ack(existing)

    def _bind_input(self, run_id: str, task: AdmittedTask) -> None:
        self.runs.update(
            run_id, input_artifact_hash=task.input_artifact_hash,
            source_artifact_hash=task.source_artifact_hash, source_origin=task.source_origin,
            status="queued",
        )

    def _reject(self, run_id: str, exc: AdmissionRejected) -> None:
        record = self.runs.get(run_id)
        termination = "environment-mismatch" if exc.category.startswith("environment:") else "invalid-input"
        result = self._result(record, termination=termination, error=f"{exc.category}: {exc.detail}",
                              charged=0, exact=True, attempt_cost={"admission": "rejected"})
        self.runs.update(run_id, status="finished", disposition=DISPOSITIONS[termination],
                         termination=termination, result=result)

    async def _admit_source_then_attempt(self, request, backend, environment) -> None:
        run_id = request.run_id
        wall_s = self.runs.get(run_id).admitted_wall_clock_ms / 1000.0

        async def elaborate(payload):
            return await backend.elaborate_source(payload, timeout_s=wall_s)

        try:
            task = await admit_source(
                request.input, store=self.artifacts, imports=backend.imports(),
                environment=environment, elaborate=elaborate,
            )
        except AdmissionRejected as exc:
            self._reject(run_id, exc)
            return
        except asyncio.CancelledError:
            self._cancelled_before_attempt(run_id)
            raise
        if self.runs.get(run_id).cancel_requested:
            self._cancelled_before_attempt(run_id)
            return
        self._bind_input(run_id, task)
        await self._attempt(run_id, task)

    def _cancelled_before_attempt(self, run_id: str) -> None:
        record = self.runs.get(run_id)
        result = self._result(record, termination="cancelled", error=None, charged=0, exact=True,
                              attempt_cost={"admission": "cancelled"})
        self.runs.update(run_id, status="finished", disposition="cancelled",
                         termination="cancelled", result=result)

    def _spawn(self, run_id: str, coroutine) -> None:
        task = asyncio.get_running_loop().create_task(coroutine, name=f"formal-run:{run_id}")
        self._tasks[run_id] = task

        def done(finished: asyncio.Task) -> None:
            if self._tasks.get(run_id) is finished:
                del self._tasks[run_id]
            if not finished.cancelled() and finished.exception() is not None:
                logger.error("formal run %s failed", run_id, exc_info=finished.exception())

        task.add_done_callback(done)

    # Attempts ------------------------------------------------------------------

    def _result(self, record, *, termination, error, charged, exact, attempt_cost,
                trace=None, certificate_hash=None, details_hash=None) -> dict[str, Any]:
        display = ""
        if record.input_kind == "goal":
            display = record.request["input"]["goal_text"]
        else:
            display = record.request["input"]["declaration"]
        return build_result(
            record, termination=termination, error=error, trace=trace,
            certificate_hash=certificate_hash, details_hash=details_hash,
            consumed_steps=record.consumed_steps + charged,
            consumed_exact=record.consumed_exact and exact,
            attempt_cost=attempt_cost, display_goal=display,
        )

    async def _attempt(self, run_id: str, task: AdmittedTask) -> None:
        admitted_at = time.monotonic()
        attempt, reserved = self.runs.begin_attempt(run_id, time.time())
        record = self.runs.get(run_id)
        deployment = self.registry.deployments[record.deployment_id]
        profile = deployment.profiles[record.profile]
        backend = self.backends[record.backend]
        if reserved == 0:
            outcome = AttemptOutcome("step-budget", steps_used=0, error="No tactic steps remain.")
        else:
            self.runs.mark_started(run_id, attempt)
            selection = BackendSelection.model_validate(record.backend_selection or record.request["backend"])
            timeout_s = record.admitted_wall_clock_ms / 1000.0
            try:
                outcome = await backend.run(
                    task, profile=profile, selection=selection, max_steps=reserved, timeout_s=timeout_s,
                )
            except asyncio.CancelledError:
                # The cost of interrupted work is unknown: charge the reservation.
                self._settle(run_id, attempt, AttemptOutcome("cancelled", steps_used=None), admitted_at)
                raise
            except Exception as exc:  # noqa: BLE001 — typed as a crash, never a proof outcome
                logger.exception("formal attempt %s/%s failed", run_id, attempt)
                outcome = AttemptOutcome("crash", steps_used=None, error=f"{type(exc).__name__}: {exc}")
        self._settle(run_id, attempt, outcome, admitted_at)

    def _settle(self, run_id: str, attempt: int, outcome: AttemptOutcome, admitted_at: float) -> None:
        record = self.runs.get(run_id)
        reserved = next(row["reserved_steps"] for row in self.runs.attempt_rows(run_id) if row["attempt"] == attempt)
        exact = outcome.steps_used is not None
        charged = reserved if not exact else min(int(outcome.steps_used), reserved)
        certificate_hash = None
        if outcome.certified and outcome.certificate is not None:
            certificate_hash = self.artifacts.put_json({
                **outcome.certificate,
                "run_id": run_id,
                "attempt": attempt,
                "backend": record.backend,
                "input_artifact_hash": record.input_artifact_hash,
                "source_artifact_hash": record.source_artifact_hash,
                "environment_hash": record.environment_hash,
            })
        details_hash = self.artifacts.put_json(outcome.details) if outcome.details else None
        termination = outcome.termination
        cost = {
            "attempt": attempt,
            "reserved_steps": reserved,
            "tactic_steps": charged,
            "tactic_steps_exact": exact,
            "wall_clock_ms": int((time.monotonic() - admitted_at) * 1000),
        }
        result = self._result(record, termination=termination, error=outcome.error, charged=charged,
                              exact=exact, attempt_cost=cost, trace=outcome.trace,
                              certificate_hash=certificate_hash, details_hash=details_hash)
        self.runs.settle_attempt(
            run_id, attempt, used_steps=outcome.steps_used, outcome=termination,
            run_fields={
                "status": "finished",
                "disposition": DISPOSITIONS[termination],
                "termination": termination,
                "result": result,
                "certificate_hash": certificate_hash,
            },
        )

    # Status, cancel, retry -------------------------------------------------------

    def get(self, run_id: str) -> dict[str, Any]:
        record = self.runs.get(run_id)
        if record is None:
            raise FormalError(404, "run-not-found", f"No run {run_id}.")
        return {
            **self._ack(record),
            "disposition": record.disposition,
            "termination": record.termination,
            "cancel_requested": record.cancel_requested,
            "result": record.result,
            "replays": [
                {"replay": item["replay"], "status": item["status"]} for item in self.runs.replays(run_id)
            ],
        }

    async def cancel(self, run_id: str) -> dict[str, Any]:
        record = self.runs.get(run_id)
        if record is None:
            raise FormalError(404, "run-not-found", f"No run {run_id}.")
        if record.status in ("finished", "interrupted") or record.cancel_requested:
            return self.get(run_id)
        self.runs.update(run_id, cancel_requested=True, status="cancelling")
        task = self._tasks.get(run_id)
        if task is not None:
            asyncio.get_running_loop().create_task(self._finish_cancel(run_id, task))
        return self.get(run_id)

    async def _finish_cancel(self, run_id: str, task: asyncio.Task) -> None:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        record = self.runs.get(run_id)
        backend = self.backends.get(record.backend)
        quiesce = getattr(backend, "quiesce", None)
        if quiesce is not None:
            try:
                await asyncio.wait_for(quiesce(), QUIESCE_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("backend %s did not stop the cancelled run %s in time", record.backend, run_id)
        record = self.runs.get(run_id)
        if record.status != "finished":
            self._cancelled_before_attempt(run_id)
        elif record.termination != "cancelled":
            return
        self.runs.update(run_id, status="finished")

    async def retry(self, run_id: str) -> dict[str, Any]:
        record = self.runs.get(run_id)
        if record is None:
            raise FormalError(404, "run-not-found", f"No run {run_id}.")
        if record.status not in ("finished", "interrupted") or run_id in self._tasks:
            raise FormalError(409, "run-active", f"Run {run_id} is {record.status}.")
        if record.disposition == "proved-pending-replay":
            raise FormalError(409, "run-proved", f"Run {run_id} already found a proof.")
        if record.disposition in ("invalid-request", "environment-error") and record.input_artifact_hash is None:
            raise FormalError(409, "input-rejected", f"Run {run_id}'s input was not admitted.")
        if record.remaining_steps == 0:
            raise FormalError(
                409, "allowance-exhausted",
                f"Run {run_id} has used its admitted {record.admitted_max_steps} steps.",
            )
        deployment = self.registry.deployments.get(record.deployment_id)
        profile = None if deployment is None else deployment.profiles.get(record.profile)
        if profile is None or deployment.profile_config_hash(profile) != record.profile_config_hash:
            raise FormalError(
                409, "profile-changed",
                f"Run {run_id}'s profile configuration changed since admission; it is not retried under another.",
            )
        if self._environment(deployment) != record.environment_hash:
            raise FormalError(409, "environment-changed", f"The backend no longer runs {record.environment_hash}.")
        backend = self.backends.get(record.backend)
        if backend is None or not backend.ready():
            raise FormalError(503, "backend-unavailable", f"{record.backend} is not available.")
        if record.input_artifact_hash is None:
            # Admission itself was interrupted: admit the bound request again.
            request = FormalSearchRequest.model_validate(record.request)
            environment = {"environment_id": record.environment_id, "environment_hash": record.environment_hash}
            self.runs.update(run_id, status="admitting", cancel_requested=False, disposition=None,
                             termination=None, result=None)
            self._spawn(run_id, self._admit_source_then_attempt(request, backend, environment))
            return self.get(run_id)
        try:
            manifest = self.artifacts.get_json(record.input_artifact_hash)
        except ArtifactError as exc:
            raise FormalError(409, "input-evidence-unavailable", exc.detail) from exc
        task = manifest_task(manifest, record.input_artifact_hash)
        self.runs.update(run_id, cancel_requested=False)
        self._spawn(run_id, self._attempt(run_id, task))
        await asyncio.sleep(0)
        return self.get(run_id)

    # Replay ----------------------------------------------------------------------

    async def replay(self, request: FormalReplayRequest) -> dict[str, Any]:
        record = self.runs.get(request.run_id)
        if record is None:
            raise FormalError(404, "run-not-found", f"No run {request.run_id}.")
        if record.certificate_hash is None:
            raise FormalError(409, "no-certificate", f"Run {request.run_id} has no certificate.")
        for field, expected in (
            ("certificate_hash", record.certificate_hash),
            ("input_artifact_hash", record.input_artifact_hash),
            ("source_artifact_hash", record.source_artifact_hash),
            ("environment_hash", record.environment_hash),
        ):
            if getattr(request, field) != expected:
                raise FormalError(
                    409, f"{field.replace('_', '-')}-mismatch",
                    f"Run {request.run_id}'s {field} is {expected}, not {getattr(request, field)}.",
                )
        deployment = self.registry.deployments[record.deployment_id]
        backend = self.backends.get(record.backend)

        def attest(status: str, reason: Optional[str], cost: dict[str, Any], env_hash: str) -> dict[str, Any]:
            attestation = ReplayAttestation.model_validate({
                "status": status, "reason": reason, "run_id": record.run_id,
                "certificate_hash": record.certificate_hash,
                "input_artifact_hash": record.input_artifact_hash,
                "source_artifact_hash": record.source_artifact_hash,
                "source_origin": record.source_origin,
                "environment_hash": env_hash, "backend": record.backend,
                "verification_cost": cost,
            }).model_dump(mode="json")
            self.runs.add_replay(record.run_id, status, attestation)
            return attestation

        try:
            certificate = self.artifacts.get_json(record.certificate_hash)
            manifest = self.artifacts.get_json(record.input_artifact_hash)
            self.artifacts.get(record.source_artifact_hash)
        except ArtifactError as exc:
            return attest("rejected", f"stored evidence does not verify: {exc.detail}", {}, record.environment_hash)
        bindings = {
            "run_id": record.run_id,
            "input_artifact_hash": record.input_artifact_hash,
            "source_artifact_hash": record.source_artifact_hash,
            "environment_hash": record.environment_hash,
        }
        if any(certificate.get(key) != value for key, value in bindings.items()) or any(
            manifest.get(key) != value
            for key, value in (("source_artifact_hash", record.source_artifact_hash),
                               ("environment_hash", record.environment_hash))
        ):
            return attest("rejected", "the certificate or input manifest is not bound to this run", {},
                          record.environment_hash)
        if backend is None or not backend.ready():
            raise FormalError(503, "backend-unavailable", f"{record.backend} is not available.")
        current = self._environment(deployment)
        if current != record.environment_hash:
            return attest("inconclusive", "the backend no longer runs the run's environment", {}, current or "")
        max_steps = min(request.budget.max_steps, deployment.verification_max_steps)
        wall_ms = min(request.budget.wall_clock_ms or deployment.verification_wall_clock_ms,
                      deployment.verification_wall_clock_ms)
        started = time.monotonic()
        outcome = await backend.verify(
            certificate, statement=manifest["statement"], max_steps=max_steps, timeout_s=wall_ms / 1000.0,
        )
        verified_env = current
        status, reason = outcome.status, outcome.reason
        if outcome.environment is not None:
            verified_env = environment_hash(deployment.environment_id, outcome.environment)
            if verified_env != record.environment_hash and status == "verified":
                status, reason = "inconclusive", "verification ran in a different environment"
        cost = {
            "allowance_max_steps": max_steps,
            "tactic_steps": outcome.steps_used,
            "wall_clock_ms": int((time.monotonic() - started) * 1000),
            **outcome.details,
        }
        return attest(status, reason, cost, verified_env)

    # Artifacts -------------------------------------------------------------------

    def put_artifact(self, data: bytes) -> str:
        try:
            return self.artifacts.put(data)
        except ArtifactError as exc:
            raise FormalError(413 if exc.category == "artifact-too-large" else 422, exc.category, exc.detail) from exc
