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
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from maths_ai.core.config import settings
from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.data_models.proof_components import Goal
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
            async def visit(node: Dict[str, Any], state: Any) -> None:
                tactic = TacticCandidate(
                    tactic_name=node["tactic"],
                    arguments=list(node.get("arguments", [])),
                    probability=1.0,
                )
                outcome = await executor.apply(executor.server, state, tactic)
                if not outcome.success:
                    raise ValueError(outcome.error or "Lean rejected tactic")
                children = list(node.get("subgoals", []))
                if len(children) != len(outcome.subgoals):
                    raise ValueError("proof trace subgoals do not match Lean")
                for child, subgoal in zip(children, outcome.subgoals):
                    child_state = await reasoner._start_state(subgoal)
                    await visit(child, child_state)

            root_state = await reasoner._start_state(
                Goal(expression=goal, hypotheses=hypotheses or [])
            )
            await visit(proof_trace, root_state)
        return {"verified": True, "status": "verified"}

    async def predict_tactics(self, goal: str, top_k: int) -> List[TacticCandidate]:
        """Predict tactics for a goal using the GNN engine."""
        async with self.gnn_lock:
            return await asyncio.to_thread(self.reasoner.gnn_engine.inference, goal, top_k=top_k)

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

    _RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

    def __init__(self, service: InferenceService, state_dir: Optional[Path] = None):
        self.service = service
        self.state_dir = state_dir or settings.formal_runs_dir
        self.runs: Dict[str, FormalRun] = {}
        self.lock = asyncio.Lock()
        self._recover_interrupted()

    @classmethod
    def validate_run_id(cls, run_id: str) -> str:
        if not cls._RUN_ID_RE.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, digits, '_' or '-'")
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
        return self.state_dir / f"{run_id}.json"

    def _save(self, run: FormalRun) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = {"run_id": run.run_id, "envelope": run.envelope, "status": run.status,
                "result": run.result, "error": run.error}
        self._path(run.run_id).write_text(json.dumps(data), encoding="utf-8")

    def _load(self, run_id: str) -> Optional[FormalRun]:
        path = self._path(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
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
            run.result = ProveResponse.from_graph(graph).model_dump(mode="json")
            run.status = "found"
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
            if run_id in self.runs or self._path(run_id).exists():
                raise ValueError(f"run_id already exists: {run_id}")
            run = FormalRun(run_id=run_id, envelope=dict(envelope))
            self.runs[run_id] = run
            run.task = asyncio.create_task(self._execute(run))
            self._save(run)
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
            run.task = asyncio.create_task(self._execute(run))
            self._save(run)
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
    except Exception:
        server._close()
        raise

    return InferenceService(reasoner=reasoner)
