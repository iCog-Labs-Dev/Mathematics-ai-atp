"""The two proving backends behind the formal service.

Each adapter runs one bounded attempt on an admitted task and reports a typed
``AttemptOutcome``; it verifies a stored certificate in a fresh native session
and reports ``verified``, ``rejected`` or ``inconclusive``. Neither falls back
to the other backend.

* ``MctsFormalBackend`` queues work on the MCTS ``ProvingService``: search in
  its runtime workers, source elaboration in a dedicated REPL process, and
  verification by artifact replay in a new Pantograph client.
* ``AcceptedFormalBackend`` runs the accepted-GNN best-first search, and
  verifies by compiling the stored statement with its proof script in a new
  Lean process under the deployment's own imports, warnings as errors.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from maths_ai.hybrid_reasoner.tactic_budget import TacticAllowance

from .admission import AdmissionRejected, AdmittedTask
from .contract import ACCEPTED_BACKEND, MCTS_BACKEND

ACCEPTED_CERTIFICATE_FORMAT = "maths-ai-accepted-certificate-v1"
MCTS_CERTIFICATE_FORMAT = "maths-ai-mcts-certificate-v1"


@dataclass
class AttemptOutcome:
    termination: str
    certified: bool = False
    steps_used: Optional[int] = None  # None: unknown, the reservation is charged
    trace: Optional[dict[str, Any]] = None
    certificate: Optional[dict[str, Any]] = None
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class VerifyOutcome:
    status: str
    reason: Optional[str]
    steps_used: Optional[int]
    environment: Optional[dict[str, Any]]
    details: dict[str, Any] = field(default_factory=dict)


_MCTS_END_REASONS = {
    "root_certified": "certified",
    "step_budget": "step-budget",
    "max_nodes": "node-budget",
    "num_simulations": "simulation-budget",
    "deadline": "deadline",
    "frontier_empty": "frontier-exhausted",
    "root_dead": "alternatives-exhausted",
    "resource_limit": "lean-resource",
    "external_abort": "crash",
}


def mcts_outcome(view: dict[str, Any]) -> AttemptOutcome:
    """Classify one MCTS search view (``record_view``) as a formal attempt outcome."""
    execution = view.get("execution") or {}
    steps = execution.get("tactic_steps_issued")
    outcome = view.get("outcome")
    details = {
        key: view.get(key)
        for key in ("outcome", "end_reason", "exception_category", "resource", "search", "timing", "nodes", "edges")
        if view.get(key) is not None
    }
    if outcome == "inadmissible":
        return AttemptOutcome("invalid-input", steps_used=steps or 0, details=details,
                              error=f"Lean could not admit the goal ({view.get('exception_category')}).")
    if outcome == "error":
        return AttemptOutcome("crash", steps_used=steps, details=details,
                              error=f"The search failed in the runtime ({view.get('exception_category')}).")
    if outcome == "hard_timeout":
        return AttemptOutcome("deadline", steps_used=None, details=details,
                              error="The search did not stop by its deadline; its worker was replaced.")
    if view.get("certified") and view.get("artifact"):
        return AttemptOutcome(
            "certified", True, steps, view.get("proof_trace"),
            {"format": MCTS_CERTIFICATE_FORMAT, "artifact": view["artifact"],
             "proof_trace": view.get("proof_trace")},
            details,
        )
    if outcome == "cooperative_deadline":
        return AttemptOutcome("deadline", steps_used=steps, details=details)
    if outcome == "resource_limit":
        return AttemptOutcome("lean-resource", steps_used=steps, details=details)
    termination = _MCTS_END_REASONS.get(view.get("end_reason") or "", "frontier-exhausted")
    if termination == "certified":  # certified without an artifact is not evidence
        termination = "crash"
    return AttemptOutcome(termination, steps_used=steps, details=details)


class MctsFormalBackend:
    name = MCTS_BACKEND
    supports_universes = True

    def __init__(self, service) -> None:
        self.service = service

    def environment_manifest(self) -> Optional[dict[str, Any]]:
        provenance = getattr(self.service.supervisor, "provenance", None) or {}
        return provenance.get("formal_environment")

    def imports(self) -> list[str]:
        manifest = self.environment_manifest() or {}
        return list(manifest.get("imports") or [])

    def ready(self) -> bool:
        return bool(self.service.readiness()["ready"])

    async def quiesce(self) -> None:
        """Wait until the runtime's current operation (a cancelled attempt) has stopped."""
        while self.service._step is not None and not self.service._step.done():
            await asyncio.sleep(0.1)

    async def _run(self, operation: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        from maths_ai.api.service import DeadlineExceeded, InputRejected, ServiceError

        try:
            return await self.service.run_formal(operation, payload, timeout_s=timeout_s)
        except InputRejected as exc:
            raise AdmissionRejected(exc.category, exc.detail) from exc
        except DeadlineExceeded as exc:
            raise TimeoutError(exc.detail) from exc
        except ServiceError as exc:
            raise BackendFailure(exc.category, exc.detail) from exc

    async def elaborate_source(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        try:
            result = await self._run("admit_source", payload, timeout_s)
        except TimeoutError as exc:
            raise AdmissionRejected("input:admission-timeout", str(exc)) from exc
        if "rejected" in result:
            raise AdmissionRejected(result["rejected"]["category"], result["rejected"]["detail"])
        return result

    async def run(
        self, task: AdmittedTask, *, profile, selection, max_steps: int, timeout_s: float
    ) -> AttemptOutcome:
        settings = dict(profile.backend_settings)
        search_profile = settings.pop("search_profile", profile.name)
        payload = self.service.resolve_formal_prove(
            seed_spec=task.seed_spec,
            display_goal=task.display_goal,
            search_profile=search_profile,
            settings=settings,
            proposal_mode=selection.proposal_mode,
            seed=selection.seed,
            max_tactic_steps=max_steps,
            timeout_s=timeout_s,
        )
        try:
            view = await self._run("prove", payload, timeout_s)
        except AdmissionRejected as exc:
            return AttemptOutcome("invalid-input", steps_used=0, error=exc.detail)
        except TimeoutError as exc:
            return AttemptOutcome("deadline", steps_used=None, error=str(exc))
        except BackendFailure as exc:
            return AttemptOutcome("crash", steps_used=None, error=f"{exc.category}: {exc.detail}")
        return mcts_outcome(view)

    async def verify(self, certificate: dict[str, Any], *, statement, max_steps: int, timeout_s: float) -> VerifyOutcome:
        if certificate.get("format") != MCTS_CERTIFICATE_FORMAT:
            return VerifyOutcome("rejected", "The certificate is not an MCTS certificate.", 0, None)
        if certificate["artifact"].get("statement", {}).get("expression") != statement["expression"]:
            return VerifyOutcome("rejected", "The certificate proves a different statement.", 0, None)
        try:
            result = await self._run(
                "verify", {"artifact": certificate["artifact"], "max_tactic_steps": max_steps}, timeout_s
            )
        except TimeoutError as exc:
            return VerifyOutcome("inconclusive", f"verification deadline: {exc}", None, None)
        except (BackendFailure, AdmissionRejected) as exc:
            return VerifyOutcome("inconclusive", f"verification could not run: {exc}", None, None)
        return VerifyOutcome(
            result["status"], result.get("reason"), None, result.get("formal_environment"),
            result.get("verification") or {},
        )


class BackendFailure(RuntimeError):
    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


# Accepted-GNN backend -------------------------------------------------------------


def proof_script(trace: dict[str, Any], *, indent: int = 2, bullet: bool = False) -> list[str]:
    arguments = " ".join(str(argument).rstrip(":") for argument in trace.get("arguments", []))
    command = " ".join([str(trace.get("tactic") or trace.get("command")), arguments]).strip()
    prefix = " " * indent + ("· " if bullet else "")
    lines = [prefix + command]
    child = indent + 2 if bullet else indent
    for subgoal in trace.get("subgoals", []):
        lines.extend(proof_script(subgoal, indent=child, bullet=True))
    return lines


def trace_steps(trace: dict[str, Any]) -> int:
    return 1 + sum(trace_steps(child) for child in trace.get("subgoals", []))


def certificate_source(statement: dict[str, Any], trace: dict[str, Any], imports: list[str]) -> str:
    """The declaration a verifier compiles: the stored statement and the proof."""
    levels = statement["universe_levels"]
    universes = f".{{{', '.join(levels)}}}" if levels else ""
    lines = [
        *(f"import {name}" for name in imports),
        "set_option warningAsError true",
        "",
        f"theorem formal_certificate{universes} :",
        f"    {statement['expression']} := by",
    ]
    if statement["local_names"]:
        lines.append(f"  intro {' '.join(statement['local_names'])}")
    lines += [*proof_script(trace), ""]
    return "\n".join(lines)


class AcceptedFormalBackend:
    name = ACCEPTED_BACKEND
    supports_universes = False

    def __init__(self, service, *, project_path: Path, imports: list[str]) -> None:
        self.service = service
        self.project_path = Path(project_path)
        self._imports = list(imports)
        self._environment: Optional[dict[str, Any]] = None

    def environment_manifest(self) -> Optional[dict[str, Any]]:
        if self._environment is None:
            from maths_ai.accepted_gnn.environment import accepted_environment

            self._environment = getattr(self.service, "formal_environment", None) or accepted_environment(
                self.project_path, self._imports
            )
        return self._environment

    def imports(self) -> list[str]:
        return list(self._imports)

    def ready(self) -> bool:
        return self.service is not None and getattr(self.service, "petta_available", True) is not False

    async def elaborate_source(self, payload, *, timeout_s):
        raise AdmissionRejected("input:unsupported-input", "The accepted-GNN backend admits goal inputs only.")

    async def run(
        self, task: AdmittedTask, *, profile, selection, max_steps: int, timeout_s: float
    ) -> AttemptOutcome:
        if task.kind != "goal":
            return AttemptOutcome("invalid-input", steps_used=0, error="accepted-gnn admits goal inputs only")
        settings = dict(profile.backend_settings)
        allowance = TacticAllowance(max_steps)
        goal = task.manifest["input"]["goal_text"]
        hypotheses = list(task.manifest["input"]["hypotheses"])
        try:
            graph = await self.service.prove(
                goal,
                hypotheses,
                max_depth=settings.get("max_depth"),
                max_nodes=settings.get("max_nodes"),
                timeout=timeout_s,
                allowance=allowance,
            )
        except asyncio.TimeoutError:
            return AttemptOutcome("deadline", steps_used=allowance.issued)
        except ValueError as exc:
            return AttemptOutcome("invalid-input", steps_used=allowance.issued, error=str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — an infrastructure fault, typed as a crash
            return AttemptOutcome("crash", steps_used=allowance.issued, error=f"{type(exc).__name__}: {exc}")
        summary = graph.summary() if hasattr(graph, "summary") else {}
        details = {"summary": {key: summary.get(key) for key in ("num_nodes", "num_edges", "termination_reason")}}
        if graph.is_solved():
            trace = graph.proof_trace()
            certificate = {
                "format": ACCEPTED_CERTIFICATE_FORMAT,
                "statement": task.statement,
                "proof_trace": trace,
            }
            # Search closed every goal; only replay certifies it.
            return AttemptOutcome("search-closed", True, allowance.issued, trace, certificate, details)
        reason = getattr(graph, "termination_reason", None)
        if reason == "step_budget":
            termination = "step-budget"
        elif reason == "budget_exhausted":
            termination = "node-budget"
        elif graph.is_exhausted():
            termination = "alternatives-exhausted"
        else:
            termination = "frontier-exhausted"
        return AttemptOutcome(termination, steps_used=allowance.issued, details=details)

    async def verify(self, certificate: dict[str, Any], *, statement, max_steps: int, timeout_s: float) -> VerifyOutcome:
        if certificate.get("format") != ACCEPTED_CERTIFICATE_FORMAT:
            return VerifyOutcome("rejected", "The certificate is not an accepted-GNN certificate.", 0, None)
        if certificate.get("statement") != statement:
            return VerifyOutcome("rejected", "The certificate proves a different statement.", 0, None)
        trace = certificate.get("proof_trace")
        if not isinstance(trace, dict):
            return VerifyOutcome("rejected", "The certificate has no proof.", 0, None)
        steps = trace_steps(trace) + (1 if statement["local_names"] else 0)
        if steps > max_steps:
            return VerifyOutcome("inconclusive", f"The proof needs {steps} steps; the allowance is {max_steps}.", 0, None)
        source = certificate_source(statement, trace, self._imports)
        started = time.monotonic()
        status, reason = await asyncio.to_thread(self._compile, source, timeout_s)
        return VerifyOutcome(
            status, reason, steps, self.environment_manifest(),
            {"session": "fresh-lean-process", "elapsed_s": time.monotonic() - started},
        )

    def _compile(self, source: str, timeout_s: float) -> tuple[str, Optional[str]]:
        path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".lean", encoding="utf-8", delete=False) as handle:
                handle.write(source)
                path = Path(handle.name)
            completed = subprocess.run(
                ["lake", "env", "lean", str(path)], cwd=self.project_path,
                capture_output=True, text=True, timeout=timeout_s, check=False,
            )
        except subprocess.TimeoutExpired:
            return "inconclusive", "Lean verification timed out"
        except OSError as exc:
            return "inconclusive", f"Lean verification is unavailable: {exc}"
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout).strip()
            if "maximum recursion depth" in error or "heartbeats" in error:
                return "inconclusive", error[-2000:]
            return "rejected", error[-2000:] or "Lean rejected the proof declaration"
        return "verified", None
