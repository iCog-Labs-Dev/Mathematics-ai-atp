"""FastAPI application for both proving backends.

Endpoints
---------
``GET  /health``                        process liveness; touches no model or Lean
``GET  /ready``                         200 when every configured backend can accept work, else 503
``POST /prove``                         one bounded proof search on the selected ``backend``
``POST /evaluations``                   accept an MCTS evaluation job (202)
``GET  /evaluations/{job_id}``          its persisted state and progress
``GET  /evaluations/{job_id}/result``   its complete or explicitly partial result
``POST /evaluations/{job_id}/cancel``   request cancellation (202 until cleaned up)
``GET  /formal-capabilities``           formal protocol version 2: deployments, limits, semantics
``POST /formal-artifacts``              upload a source artifact to the content-addressed store
``POST /formal-search/start``           admit a version-2 formal run on its selected backend (202)
``GET  /formal-search/{run_id}``        its state, admitted budget and result
``POST /formal-search/{run_id}/cancel`` request cancellation (terminal once the work stopped)
``POST /formal-search/{run_id}/retry``  another attempt with the remaining admitted allowance
``POST /formal-replay``                 independent replay of a run's stored certificate
``POST /gnn/predict_tactic``            top-k tactic candidates from the accepted v2 policy
``POST /pln/evaluate``                  PLN outcome for one structured goal

Two backends are served, each enabled by its own configuration:

* ``mcts-actor-critic``: the source-aware actor-critic policy under the MCTS
  runtime supervisor, configured by ``--config service.json`` (or
  ``MATHS_AI_SERVICE_CONFIG``).
* ``accepted-gnn``: the published Mathlib-Sexpr-GNN v2 policy with best-first
  search on the synchronous Pantograph runtime, enabled with
  ``MATHS_AI_ACCEPTED_GNN=1`` and configured by ``maths_ai.core.config``.

The formal routes are served when a deployment registry is configured
(``MATHS_AI_FORMAL_DEPLOYMENTS``); runs and artifacts persist under
``MATHS_AI_FORMAL_STATE_DIR``. They implement protocol version 2 only, for both
backends (``maths_ai.formal``).

``/prove`` requires an explicit ``backend`` and never falls back to the other
one; a backend that is not configured answers 503. Errors map by category:
invalid input is 422, a full queue 429, an unready backend 503 and an exceeded
deadline 504, which is not a mathematical refutation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Dict, Optional, Union

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from maths_ai.accepted_gnn.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    PredictTacticRequest,
    PredictTacticResponse,
    ProofTerminationReason,
)
from maths_ai.accepted_gnn.schemas import ProveRequest as AcceptedProveRequest
from maths_ai.accepted_gnn.schemas import ProveResponse as AcceptedProveResponse

from .config import ServiceConfig
from .schemas import (
    ACCEPTED_BACKEND,
    MCTS_BACKEND,
    ErrorBody,
    EvaluationRequest,
    Health,
    Job,
    ProveRequest,
    Readiness,
)
from .service import ProvingService, ServiceError

logger = logging.getLogger("maths_ai.api")

ServiceFactory = Callable[[], Awaitable[Any]]

CONFIG_ENV = "MATHS_AI_SERVICE_CONFIG"
ACCEPTED_ENV = "MATHS_AI_ACCEPTED_GNN"
FROZEN_CONFIG_ENV = "MATHS_AI_FROZEN_CONFIG"
_ERRORS = {
    422: {"model": ErrorBody}, 429: {"model": ErrorBody}, 503: {"model": ErrorBody},
    504: {"model": ErrorBody}, 404: {"model": ErrorBody}, 409: {"model": ErrorBody},
}

AnyProveRequest = Annotated[
    Union[ProveRequest, AcceptedProveRequest], Field(discriminator="backend")
]


def default_service_factory(config_path: Path | None = None) -> ServiceFactory:
    async def factory() -> ProvingService:
        path = Path(config_path or os.environ[CONFIG_ENV])
        service = ProvingService(ServiceConfig.from_json(path), path)
        await service.start()
        return service

    return factory


def default_accepted_factory() -> ServiceFactory:
    async def factory():
        from maths_ai.accepted_gnn.service import build_default_service

        return await build_default_service()

    return factory


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _error(exc: ServiceError) -> JSONResponse:
    headers = {"Retry-After": "5"} if exc.status in (429, 503) else None
    return JSONResponse(status_code=exc.status, content={"category": exc.category, "detail": exc.detail},
                        headers=headers)


def _unavailable(backend: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"category": "backend-unavailable", "detail": f"{backend}: {detail}"},
        headers={"Retry-After": "5"},
    )


def create_app(
    service_factory: Optional[ServiceFactory] = None,
    *,
    accepted_factory: Optional[ServiceFactory] = None,
    frozen_factory: Optional[ServiceFactory] = None,
    formal_registry: Any = None,
    formal_state_dir: Optional[Path] = None,
) -> FastAPI:
    """Build the app.

    ``service_factory`` builds the MCTS ``ProvingService`` and
    ``accepted_factory`` the accepted-GNN ``InferenceService``; tests inject
    either. Without factories, a backend is enabled by its configuration (see
    the module docstring), and at least one must be.
    """
    mcts_factory = service_factory
    accepted = accepted_factory
    frozen = frozen_factory
    if frozen is None and os.environ.get(FROZEN_CONFIG_ENV):
        async def frozen():
            path = Path(os.environ[FROZEN_CONFIG_ENV])
            service = ProvingService(ServiceConfig.from_json(path), path,
                                     backend="maths_ai.formal.frozen_runtime:FrozenRuntime")
            await service.start()
            return service
    if mcts_factory is None and accepted is None and frozen_factory is None:
        if os.environ.get(CONFIG_ENV):
            mcts_factory = default_service_factory()
        if _env_flag(ACCEPTED_ENV):
            accepted = default_accepted_factory()
        if mcts_factory is None and accepted is None and frozen is None:
            raise RuntimeError(
                f"No backend is configured: set {CONFIG_ENV} for {MCTS_BACKEND} "
                f"or {ACCEPTED_ENV}=1 for {ACCEPTED_BACKEND}."
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = await mcts_factory() if mcts_factory is not None else None
        app.state.accepted = None
        app.state.frozen = await frozen() if frozen is not None else None
        app.state.formal = None
        app.state.accepted_error = None if accepted is not None else "not configured"
        if accepted is not None:
            # Missing checkpoints or Lean keep the process alive but unready,
            # so orchestration sees the failure through /ready.
            try:
                app.state.accepted = await accepted()
            except Exception as exc:
                logger.exception("accepted-GNN backend initialization failed")
                app.state.accepted = None
                app.state.accepted_error = str(exc)
        app.state.formal = _formal_service(app, formal_registry, formal_state_dir)
        try:
            yield
        finally:
            if app.state.formal is not None:
                await app.state.formal.close()
            if app.state.accepted is not None:
                try:
                    await app.state.accepted.shutdown()
                except Exception:
                    logger.exception("accepted-GNN backend shutdown failed")
            if app.state.service is not None:
                await app.state.service.stop()
            if app.state.frozen is not None:
                await app.state.frozen.stop()

    app = FastAPI(
        title="Mathematics AI ATP",
        description=(
            "Kernel-certified proof search with an explicitly selected backend: the "
            "source-aware actor-critic policy under PUCT or best-first profiles, or the "
            "accepted Mathlib-Sexpr-GNN v2 policy with best-first hypergraph search. "
            "Persistent dataset evaluation jobs and the persistent-atp formal-search contract."
        ),
        version="1.1.0",
        lifespan=lifespan,
    )

    def accepted_service(request: Request) -> Any:
        service = request.app.state.accepted
        if service is None:
            raise HTTPException(
                status_code=503,
                detail=f"service unavailable: {ACCEPTED_BACKEND}: {request.app.state.accepted_error}",
            )
        return service

    def mcts_service(request: Request) -> ProvingService:
        service = request.app.state.service
        if service is None:
            raise HTTPException(status_code=503, detail=f"service unavailable: {MCTS_BACKEND}: not configured")
        return service

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return _error(exc)

    @app.get("/health", response_model=Health)
    async def health() -> Health:
        return Health()

    def accepted_readiness(request: Request) -> Dict[str, Any]:
        service = request.app.state.accepted
        if service is None:
            return {"ready": False, "error": request.app.state.accepted_error}
        # The accepted search ranks every subgoal with PLN, so unresolvable
        # PeTTa dependencies make it unready rather than silently unscored.
        petta_available = service.petta_available
        pln_error = service.pln_error
        return {
            "ready": petta_available is not False,
            "error": pln_error,
            "petta_available": petta_available,
            "pln_error": pln_error,
        }

    @app.get("/ready", response_model=Readiness, responses={503: {"model": Readiness}})
    async def ready(request: Request):
        backends: Dict[str, Any] = {}
        content: Dict[str, Any] = {}
        if request.app.state.service is not None:
            content = dict(request.app.state.service.readiness())
            backends[MCTS_BACKEND] = {"ready": content["ready"], "error": content["error"]}
        if accepted is not None:
            backends[ACCEPTED_BACKEND] = accepted_readiness(request)
        if request.app.state.frozen is not None:
            state = request.app.state.frozen.readiness()
            backends[ACCEPTED_BACKEND] = {"ready": state["ready"], "error": state["error"], "profile": "frozen-best-first"}
        ready_all = all(backend["ready"] for backend in backends.values())
        if not content:
            content = {"ready": ready_all, "error": backends[ACCEPTED_BACKEND]["error"]}
        content["ready"] = ready_all
        content["backends"] = backends
        return JSONResponse(status_code=200 if ready_all else 503, content=content)

    @app.post("/prove", responses=_ERRORS)
    async def prove(payload: Annotated[AnyProveRequest, Body()], request: Request):
        if isinstance(payload, AcceptedProveRequest):
            if request.app.state.accepted is None:
                return _unavailable(ACCEPTED_BACKEND, request.app.state.accepted_error)
            return await _accepted_prove(payload, request.app.state.accepted)
        if request.app.state.service is None:
            return _unavailable(MCTS_BACKEND, "not configured")
        service = request.app.state.service
        search = asyncio.create_task(service.prove(payload.model_dump(exclude={"backend"})))

        async def watch_disconnect() -> None:
            while not search.done():
                if await request.is_disconnected():
                    search.cancel()
                    return
                await asyncio.sleep(0.25)

        watcher = asyncio.create_task(watch_disconnect())
        try:
            return await search
        finally:
            watcher.cancel()

    @app.post("/evaluations", response_model=Job, status_code=202, responses=_ERRORS)
    async def submit(payload: EvaluationRequest, request: Request):
        job = await mcts_service(request).submit_evaluation(payload.model_dump())
        return Job(**job)

    @app.get("/evaluations/{job_id}", response_model=Job, responses=_ERRORS)
    async def job(job_id: str, request: Request):
        return Job(**await mcts_service(request).job(job_id))

    @app.get("/evaluations/{job_id}/result", responses=_ERRORS)
    async def job_result(job_id: str, request: Request):
        return await mcts_service(request).job_result(job_id)

    @app.post("/evaluations/{job_id}/cancel", response_model=Job, responses=_ERRORS)
    async def cancel(job_id: str, request: Request):
        job = await mcts_service(request).cancel_job(job_id)
        status = 200 if job["state"] in ("cancelled", "completed", "failed", "interrupted") else 202
        return JSONResponse(status_code=status, content=Job(**job).model_dump())

    @app.post("/gnn/predict_tactic", response_model=PredictTacticResponse, response_model_exclude_none=True)
    async def predict_tactic(payload: PredictTacticRequest, request: Request) -> PredictTacticResponse:
        service = accepted_service(request)
        try:
            candidates = await service.predict_tactics(
                payload.goal,
                payload.top_k,
                hypotheses=payload.hypotheses,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid goal syntax: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"tactic prediction failed: {exc}") from exc
        return PredictTacticResponse(goal=payload.goal, candidates=candidates)

    @app.post("/pln/evaluate", response_model=EvaluateResponse)
    async def evaluate(payload: EvaluateRequest, request: Request) -> EvaluateResponse:
        from maths_ai.pln_inference.compiler import GoalInput, LocalInput
        from maths_ai.pln_inference.dependencies import PLNConfigurationError

        service = accepted_service(request)
        goal = GoalInput(
            payload.target_sexp,
            tuple(
                LocalInput(local.internal_name, local.type_sexp, local.value_sexp, local.is_let)
                for local in payload.locals
            ),
        )
        try:
            result = await service.evaluate(goal)
        except PLNConfigurationError as exc:
            raise HTTPException(status_code=503, detail=f"PLN is unavailable: {exc}") from exc
        if result.status == "unsupported":
            raise HTTPException(status_code=422, detail={"status": "unsupported", "diagnostic": result.diagnostic})
        return EvaluateResponse(
            stv=result.stv,
            score=None if result.stv is None else result.stv.score,
            status=result.status,
            is_fallback=result.is_fallback,
            diagnostic=result.diagnostic,
        )

    def formal_service(request: Request):
        from maths_ai.formal.service import FormalError

        service = request.app.state.formal
        if service is None:
            raise FormalError(503, "formal-unavailable", "No formal deployment registry is configured.")
        return service

    from maths_ai.formal.routes import formal_router

    app.include_router(formal_router(formal_service))

    return app


FORMAL_REGISTRY_ENV = "MATHS_AI_FORMAL_DEPLOYMENTS"
FORMAL_STATE_ENV = "MATHS_AI_FORMAL_STATE_DIR"


def _formal_service(app: FastAPI, registry: Any, state_dir: Optional[Path]):
    """The formal service over whichever backends started, if a registry is configured."""
    from maths_ai.formal.deployments import DeploymentRegistry

    if registry is None and os.environ.get(FORMAL_REGISTRY_ENV):
        registry = Path(os.environ[FORMAL_REGISTRY_ENV])
    if registry is None:
        return None
    if not isinstance(registry, DeploymentRegistry):
        registry = DeploymentRegistry.from_json(Path(registry))
    from maths_ai.core.config import settings
    from maths_ai.formal.backends import AcceptedFormalBackend, MctsFormalBackend
    from maths_ai.formal.service import FormalService

    backends: dict[str, Any] = {}
    if app.state.service is not None:
        backends[MCTS_BACKEND] = MctsFormalBackend(app.state.service)
    if app.state.accepted is not None:
        backends[ACCEPTED_BACKEND] = AcceptedFormalBackend(
            app.state.accepted, project_path=settings.lean_project_path, imports=["Mathlib"],
        )
    if app.state.frozen is not None:
        from maths_ai.formal.frozen_backend import FrozenFormalBackend

        backends[ACCEPTED_BACKEND] = FrozenFormalBackend(app.state.frozen)
    directory = Path(state_dir or os.environ.get(FORMAL_STATE_ENV) or "/state/formal")
    return FormalService(registry=registry, state_dir=directory, backends=backends)


async def _accepted_prove(payload: AcceptedProveRequest, service: Any) -> Any:
    from maths_ai.core.config import settings

    timeout = payload.timeout if payload.timeout is not None else settings.api_default_prove_timeout
    try:
        graph = await service.prove(
            payload.goal,
            payload.hypotheses,
            max_depth=payload.max_depth,
            max_nodes=payload.max_nodes,
            timeout=timeout,
        )
        return AcceptedProveResponse.from_graph(graph, timeout=False)
    except asyncio.TimeoutError:
        # A timeout is an answer, not an error; the service has released the proof lock.
        return AcceptedProveResponse(
            solved=False,
            exhausted=False,
            termination_reason=ProofTerminationReason.TIMEOUT,
            proof_trace=None,
            summary={"error": "timeout"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Lean parsing error: {exc}") from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"proof search failed: {exc}") from exc
