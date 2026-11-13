"""Service layer for the maths_ai HTTP API.

``InferenceService`` wraps a fully-constructed ``HybridReasoner`` (GNN engine,
PLN chainer and Pantograph executor all loaded once at startup) and adds the
concurrency machinery the HTTP surface needs:

* a single shared Pantograph ``Server`` process — proof searches are
  serialized behind a proof/runtime asyncio lock because the server's goal-state
  handles and the DTS sampler are stateful and not safe to interleave;
* separate GNN and PLN locks for component-scoped synchronization, allowing
  standalone GNN/PLN calls to proceed concurrently with proof searches;
* concurrency-safe access to the DTS state file: every persist/load goes
  through its own lock, so concurrent ``/prove`` requests can't corrupt it.

The heavy imports (torch, pantograph) live inside ``build_default_service``
so the rest of the package stays importable on machines without them (tests
inject fakes; docs generation imports the app).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from maths_ai.core.config import settings
from maths_ai.data_models.proof_components import (
    Goal,
    TacticCandidate,
    filter_executable_tactic_candidates,
)
from maths_ai.pln_inference.model import PLNInference, PLNResult

logger = logging.getLogger(__name__)
@dataclass
class InferenceService:
    """Owns the loaded inference components plus the API-level locks.
    
    Concurrency contract:
    - ``proof_lock``: held start-to-finish around each ``/prove`` call, including
      all GNN/PLN search iterations and DTS persistence. Protects the shared
      Pantograph Server and DTS sampler state.
    - ``gnn_lock``: protects GNN model inference; acquired by standalone
      ``/gnn/predict_tactic`` and by proof search when calling GNN.
    - ``pln_lock``: protects PLN inference; acquired by standalone ``/pln/evaluate``
      and by proof search when calling PLN.
    - ``dts_lock``: protects atomic DTS state file writes.
    
    This allows standalone GNN/PLN calls to proceed during proof searches
    (subject to their component locks), while proof searches cannot overlap.
    """

    reasoner: Any  # HybridReasoner; typed loosely so fakes can stand in
    dts_state_path: Path = field(default_factory=lambda: settings.dts_state_file)
    proof_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    gnn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pln_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    dts_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def petta_available(self) -> Optional[bool]:
        """Expose PeTTa startup state to the readiness endpoint."""
        petta = getattr(self.reasoner, "petta_chainer", None)
        available = getattr(petta, "is_available", None)
        return available if isinstance(available, bool) else None

    async def prove(
        self,
        goal: str,
        hypotheses: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        max_nodes: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """Run one full hybrid proof search, serially.

        Per-request depth/node-budget overrides are applied inside the lock
        and restored afterwards — safe because searches never overlap.
        
        The proof_lock is held for the entire duration, including GNN/PLN
        calls and DTS persistence, because the shared Pantograph Server and
        DTS sampler are stateful and cannot be safely interleaved.
        
        Args:
            goal: Lean goal expression to prove.
            hypotheses: Optional list of hypothesis strings.
            max_depth: Per-request override of the search branch-depth bound.
            max_nodes: Per-request override of the total hypergraph-size bound.
            timeout: Optional per-request timeout in seconds. If provided,
                the proof search will be cancelled if it exceeds this duration.
                Note: this is cooperative - in-flight thread operations will
                complete before cancellation takes effect.
        
        Returns:
            ProofHypergraph with the search results.
        
        Raises:
            asyncio.TimeoutError: If the proof search exceeds the timeout.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        
        async with self.proof_lock:
            saved = (self.reasoner.max_depth, self.reasoner.max_nodes)
            try:
                if max_depth is not None:
                    self.reasoner.max_depth = max_depth
                if max_nodes is not None:
                    self.reasoner.max_nodes = max_nodes
                
                if timeout is not None:
                    graph = await asyncio.wait_for(
                        self.reasoner.prove(goal, hypotheses=hypotheses),
                        timeout=timeout,
                    )
                else:
                    graph = await self.reasoner.prove(goal, hypotheses=hypotheses)
            finally:
                self.reasoner.max_depth, self.reasoner.max_nodes = saved
        await self.persist_dts_state()
        return graph

    async def replay(self, goal: str, proof_trace: Dict[str, Any],
                     hypotheses: Optional[List[str]] = None) -> Dict[str, Any]:
        """Replay a proof trace against the live Lean/Pantograph server."""
        reasoner = self.reasoner
        executor = getattr(reasoner, "executor", None)
        start_state = getattr(reasoner, "_start_state", None)
        if executor is None or start_state is None:
            raise RuntimeError("Lean replay is unavailable for this service")

        async with self.proof_lock:
            async def visit(node: Dict[str, Any], state: Any) -> Any:
                if not state.goals:
                    raise ValueError("proof trace contains a tactic after all goals were closed")

                pending_goals = len(state.goals) - 1
                tactic_cmd = self._render_tactic(node)
                try:
                    next_state = await executor.server.goal_tactic_async(state, 0, tactic_cmd)
                except Exception as exc:
                    raise ValueError(str(exc)) from exc

                children = list(node.get("subgoals", []))
                produced_goals = len(next_state.goals) - pending_goals
                if produced_goals != len(children):
                    raise ValueError("proof trace subgoals do not match Lean")

                for child in children:
                    next_state = await visit(child, next_state)

                if len(next_state.goals) != pending_goals:
                    raise ValueError("proof trace did not close all generated subgoals")
                return next_state

            root_state = await reasoner._start_state(
                Goal(expression=goal, hypotheses=hypotheses or [])
            )
            final_state = await visit(proof_trace, root_state)
            if final_state.goals:
                raise ValueError("proof left unresolved Lean goals")

            # The pinned Pantograph protocol does not expose hasSorry/hasUnsafe.
            # Compile the same declaration with warnings-as-errors so verification
            # fails closed rather than accepting a goal closed by `sorry`.
            await asyncio.to_thread(
                self._verify_declaration_sync,
                goal,
                proof_trace,
                hypotheses or [],
            )
        return {"verified": True, "status": "verified"}

    @staticmethod
    def _render_tactic(node: Dict[str, Any]) -> str:
        arguments = " ".join(str(arg).rstrip(":") for arg in node.get("arguments", []))
        return " ".join([str(node["tactic"]), arguments]).strip()

    @classmethod
    def _render_proof_script(
        cls,
        node: Dict[str, Any],
        *,
        indent: int = 2,
        bullet: bool = False,
    ) -> List[str]:
        prefix = " " * indent + ("· " if bullet else "")
        lines = [prefix + cls._render_tactic(node)]
        child_indent = indent + 2 if bullet else indent
        for child in node.get("subgoals", []):
            lines.extend(cls._render_proof_script(child, indent=child_indent, bullet=True))
        return lines

    @classmethod
    def _verify_declaration_sync(
        cls,
        goal: str,
        proof_trace: Dict[str, Any],
        hypotheses: List[str],
    ) -> None:
        binders = " ".join(f"({hypothesis})" for hypothesis in hypotheses)
        theorem_head = "theorem replay_candidate"
        if binders:
            theorem_head += f" {binders}"
        source = "\n".join([
            "import Init",
            "set_option warningAsError true",
            f"{theorem_head} : {goal} := by",
            *cls._render_proof_script(proof_trace),
            "",
        ])

        path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", encoding="utf-8", delete=False
            ) as handle:
                handle.write(source)
                path = Path(handle.name)
            completed = subprocess.run(
                ["lake", "env", "lean", str(path)],
                cwd=settings.lean_project_path,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Lean verification timed out") from exc
        except OSError as exc:
            raise RuntimeError(f"Lean verification is unavailable: {exc}") from exc
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout).strip()
            raise ValueError(error[-4000:] or "Lean rejected the proof declaration")

    async def predict_tactics(self, goal: str, top_k: int) -> List[TacticCandidate]:
        """Predict tactics for a goal using the GNN engine."""
        async with self.gnn_lock:
            candidates = await asyncio.to_thread(
                self.reasoner.gnn_engine.inference, goal, top_k=top_k
            )
        filtered = filter_executable_tactic_candidates(candidates)
        if len(filtered) != len(candidates):
            logger.warning(
                "Filtered %d GNN candidate(s) containing unresolved lemma placeholders.",
                len(candidates) - len(filtered),
            )
        return filtered

    async def evaluate(self, expression: str, hypotheses: Optional[List[str]] = None) -> PLNResult:
        """Evaluate an expression using PLN inference."""
        async with self.pln_lock:
            return await asyncio.to_thread(
                self.reasoner.petta_chainer.evaluate, expression, hypotheses=hypotheses
            )

    async def persist_dts_state(self) -> None:
        """Persist the DTS sampler under the state-file lock."""
        sampler = getattr(self.reasoner, "dts_sampler", None)
        if sampler is None:
            return
        async with self.dts_lock:
            try:
                await asyncio.to_thread(self._save_dts_sync)
            except Exception:
                logger.exception("failed to persist DTS state to %s", self.dts_state_path)

    def _save_dts_sync(self) -> None:
        self.dts_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.reasoner.dts_sampler.save_to(str(self.dts_state_path))

    async def shutdown(self) -> None:
        """Flush DTS state and close the Pantograph server process."""
        await self.persist_dts_state()
        server = getattr(self.reasoner, "server", None)
        close = getattr(server, "_close", None)
        if close is not None:
            await asyncio.to_thread(close)


@dataclass
class FormalRun:
    """Durable state for one persistent-atp formal search."""

    run_id: str
    envelope: Dict[str, Any]
    status: str = "queued"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    task: Optional[asyncio.Task] = field(default=None, repr=False)


class FormalSearchManager:
    """Manage asynchronous formal-search runs and their checkpoints."""

    _RUN_ID_RE = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*$"
    )
    _LEGACY_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

    def __init__(self, service: InferenceService, state_dir: Optional[Path] = None):
        self.service = service
        self.state_dir = state_dir or settings.formal_runs_dir
        self.runs: Dict[str, FormalRun] = {}
        self.lock = asyncio.Lock()
        self._recover_interrupted()

    @classmethod
    def validate_run_id(cls, run_id: str) -> str:
        if len(run_id) > 256 or not cls._RUN_ID_RE.fullmatch(run_id):
            raise ValueError(
                "run_id must be at most 256 characters and contain slash-separated "
                "segments using only letters, digits, '_', '-' or '.'"
            )
        return run_id

    def _recover_interrupted(self) -> None:
        """Mark in-flight jobs interrupted after a process restart."""
        if not self.state_dir.exists():
            return
        for path in self.state_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") in {"queued", "running", "cancelling"}:
                    data["status"] = "interrupted"
                    path.write_text(json.dumps(data), encoding="utf-8")
            except (OSError, ValueError, TypeError):
                logger.exception("failed to recover formal run state from %s", path)

    def _path(self, run_id: str) -> Path:
        self.validate_run_id(run_id)
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        return self.state_dir / f"{digest}.json"

    def _legacy_path(self, run_id: str) -> Optional[Path]:
        """Return the pre-hash path only for IDs that were already filesystem-safe."""
        if self._LEGACY_RUN_ID_RE.fullmatch(run_id):
            return self.state_dir / f"{run_id}.json"
        return None

    def _save(self, run: FormalRun) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = {"run_id": run.run_id, "envelope": run.envelope, "status": run.status,
                "result": run.result, "error": run.error}
        self._path(run.run_id).write_text(json.dumps(data), encoding="utf-8")

    def _load(self, run_id: str) -> Optional[FormalRun]:
        path = self._path(run_id)
        if not path.exists():
            legacy_path = self._legacy_path(run_id)
            if legacy_path is None or not legacy_path.exists():
                return None
            path = legacy_path
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("run_id") != run_id:
            raise ValueError("stored formal run ID does not match the requested run ID")
        run = FormalRun(run_id=run_id, envelope=data["envelope"], status=data["status"],
                        result=data.get("result"), error=data.get("error"))
        self.runs[run_id] = run
        return run

    @staticmethod
    def _request_args(envelope: Dict[str, Any]) -> Dict[str, Any]:
        problem = envelope.get("problem", envelope)
        budget = envelope.get("budget", {})
        goal = problem.get("goal") or problem.get("target")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("formal-search request requires problem.goal")
        return {
            "goal": goal,
            "hypotheses": problem.get("hypotheses", []),
            "max_depth": budget.get("max_depth", envelope.get("max_depth")),
            "max_nodes": budget.get("max_nodes", envelope.get("max_nodes")),
            "timeout": budget.get("timeout", envelope.get("timeout")),
        }

    def _snapshot(self, run: FormalRun) -> Dict[str, Any]:
        return {
            "run_id": run.run_id, "status": run.status,
            "result": run.result, "error": run.error,
            "environment": run.envelope.get("environment"),
            "budget": run.envelope.get("budget"),
            "request": run.envelope,
        }

    async def _execute(self, run: FormalRun) -> None:
        run.status = "running"
        self._save(run)
        try:
            args = self._request_args(run.envelope)
            graph = await self.service.prove(**args)
            from maths_ai.api.schemas import ProveResponse
            response = ProveResponse.from_graph(graph)
            run.result = response.model_dump(mode="json")
            if response.solved and response.proof_trace is not None:
                run.status = "found"
            elif response.exhausted:
                run.status = "exhausted"
            else:
                run.status = "unsolved"
        except asyncio.CancelledError:
            run.status = "cancelled"
            self._save(run)
            raise
        except Exception as exc:
            run.status, run.error = "failed", str(exc)
        finally:
            run.task = None
            self._save(run)

    async def start(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(envelope.get("run_id") or envelope.get("id") or uuid.uuid4())
        self.validate_run_id(run_id)
        async with self.lock:
            legacy_path = self._legacy_path(run_id)
            if (
                run_id in self.runs
                or self._path(run_id).exists()
                or (legacy_path is not None and legacy_path.exists())
            ):
                raise ValueError(f"run_id already exists: {run_id}")
            normalized_envelope = dict(envelope)
            normalized_envelope["run_id"] = run_id
            run = FormalRun(run_id=run_id, envelope=normalized_envelope)
            self.runs[run_id] = run
            self._save(run)
            run.task = asyncio.create_task(self._execute(run))
            return self._snapshot(run)

    async def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        self.validate_run_id(run_id)
        async with self.lock:
            run = self.runs.get(run_id) or self._load(run_id)
            return self._snapshot(run) if run else None

    async def retry(self, run_id: str) -> Dict[str, Any]:
        """Retry an interrupted/failed search; this is not checkpoint resume."""
        self.validate_run_id(run_id)
        async with self.lock:
            run = self.runs.get(run_id) or self._load(run_id)
            if run is None:
                raise KeyError(run_id)
            if run.task is not None and not run.task.done():
                raise RuntimeError("run is already active")
            if run.status not in {"interrupted", "failed", "cancelled"}:
                raise RuntimeError("only interrupted, failed, or cancelled runs can be retried")
            run.status, run.error, run.result = "queued", None, None
            self._save(run)
            run.task = asyncio.create_task(self._execute(run))
            return self._snapshot(run)

    async def cancel(self, run_id: str) -> Dict[str, Any]:
        self.validate_run_id(run_id)
        async with self.lock:
            run = self.runs.get(run_id) or self._load(run_id)
            if run is None:
                raise KeyError(run_id)
            if run.task is not None and not run.task.done():
                run.task.cancel()
                run.status = "cancelling"
                self._save(run)
            elif run.status in {"queued", "running"}:
                run.status = "cancelled"
                self._save(run)
            return self._snapshot(run)

    async def mark_verified(self, run_id: str) -> None:
        self.validate_run_id(run_id)
        async with self.lock:
            run = self.runs.get(run_id) or self._load(run_id)
            if run is None:
                raise KeyError(run_id)
            if run.status != "found":
                raise RuntimeError("only a found proof can be verified")
            run.status = "verified"
            self._save(run)

    async def validate_replay(
        self,
        run_id: str,
        problem: Dict[str, Any],
        proof_trace: Dict[str, Any],
    ) -> None:
        """Ensure replay input is exactly the candidate stored for this run."""
        self.validate_run_id(run_id)
        async with self.lock:
            run = self.runs.get(run_id) or self._load(run_id)
            if run is None:
                raise KeyError(run_id)
            if run.status != "found":
                raise RuntimeError("only a found proof can be replayed")
            if not run.result or not run.result.get("solved") or not run.result.get("proof_trace"):
                raise RuntimeError("run has no solved proof trace")

            stored_problem = run.envelope.get("problem", {})
            normalized_stored_problem = {
                "goal": stored_problem.get("goal"),
                "hypotheses": stored_problem.get("hypotheses", []),
            }
            normalized_problem = {
                "goal": problem.get("goal"),
                "hypotheses": problem.get("hypotheses", []),
            }
            if normalized_problem != normalized_stored_problem:
                raise ValueError("replay problem does not match the stored run")
            if proof_trace != run.result["proof_trace"]:
                raise ValueError("replay proof trace does not match the stored run")

async def build_default_service(
    *,
    config_path: Optional[Path] = None,
    tactic_model_path: Optional[Path] = None,
    argument_model_path: Optional[Path] = None,
    bundle_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    top_k_tactics: int = 3,
    top_k_subgoals: int = 3,
    max_depth: Optional[int] = None,
    max_nodes: int = 500,
    dts_c: Optional[float] = None,
    dts_random_seed: Optional[int] = None,
) -> InferenceService:
    """Construct the real service from configured model paths.

    Loads the GNN checkpoints once, starts one Pantograph/Lean server process
    once, and restores any existing DTS state — everything the batch CLI
    ``joint_inference.main`` does, minus the argparse/goal plumbing.
    Paths default to the env-configurable settings so Docker/deployments can
    redirect them without code changes.
    """
    from pantograph.server import Server

    from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, PantographExecutor
    from maths_ai.pln_inference.metta.translator.translator_modules.runner import DynamicThompsonSampler

    config_path = config_path or settings.gnn_config_path
    tactic_model_path = tactic_model_path or settings.tactic_model_path
    argument_model_path = argument_model_path or settings.argument_model_path
    index_path = index_path if index_path is not None else settings.lemma_index_path
    corpus_path = corpus_path if corpus_path is not None else settings.lemma_corpus_path
    bundle_path = bundle_path if bundle_path is not None else settings.gnn_bundle_path

    server = await Server.create(project_path=str(settings.lean_project_path))
    try:
        dts_input = settings.dts_state_file
        dts_sampler = None
        if dts_input.exists() and dts_input.stat().st_size > 0:
            try:
                dts_sampler = DynamicThompsonSampler.load_from(
                    str(dts_input), C=dts_c if dts_c is not None else settings.dts_default_c
                )
            except Exception as exc:
                logger.warning("could not load DTS state from %s (%s); starting fresh", dts_input, exc)

        reasoner = HybridReasoner(
            config_path=config_path,
            tactic_model_path=tactic_model_path,
            argument_model_path=argument_model_path,
            bundle_path=bundle_path if bundle_path.exists() else None,
            index_path=index_path,
            corpus_path=corpus_path,
            executor=PantographExecutor(server=server),
            top_k_tactics=top_k_tactics,
            top_k_subgoals=top_k_subgoals,
            max_depth=max_depth if max_depth is not None else settings.proof_depth,
            max_nodes=max_nodes,
            dts_sampler=dts_sampler,
            dts_c=dts_c,
            dts_random_seed=dts_random_seed,
        )
        reasoner.petta_chainer.initialize()
    except Exception:
        server._close()
        raise

    return InferenceService(reasoner=reasoner)
