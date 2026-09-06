"""FastAPI application exposing the hybrid reasoner over HTTP.

Endpoints
---------
``GET  /health``              liveness/readiness probe
``POST /prove``               full HybridReasoner.prove() search, returning the
                              verifiable tactic-sequence proof trace
``POST /gnn/predict_tactic``  top-k tactic candidates from GNNModelEngine
``POST /pln/evaluate``        STV score from PLNInference

Run locally with ``python -m maths_ai.api`` (or
``uvicorn maths_ai.api.app:create_app --factory``). All model/checkpoint paths
come from the env-configurable settings (see ``maths_ai.core.config``) and are
loaded exactly once at startup, not per request.

Error mapping: malformed request bodies → 422 (pydantic); unparseable
goal/hypothesis syntax surfacing as ``ValueError`` → 400; service components
not initialized → 503; unexpected inference failures → 500.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from maths_ai.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    PredictTacticRequest,
    PredictTacticResponse,
    ProveRequest,
    ProveResponse,
)
from maths_ai.core.config import settings
from maths_ai.pln_inference.model import PLNResult

logger = logging.getLogger(__name__)

ServiceFactory = Callable[[], Awaitable[Any]]


def _get_service(request: Request) -> Any:
    """Fetch the initialized service or fail with 503.

    Startup component loading can legitimately fail (missing checkpoints,
    Lean toolchain unavailable); keeping the app alive but unready lets
    orchestration observe the failure via ``/health`` instead of crash-looping.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        error = getattr(request.app.state, "init_error", None) or "service is starting"
        raise HTTPException(status_code=503, detail=f"service unavailable: {error}")
    return service


def create_app(service_factory: Optional[ServiceFactory] = None) -> FastAPI:
    """Build the API app.

    ``service_factory`` is an awaitable returning a fully-initialized
    ``InferenceService``-compatible object; tests inject fakes here so no
    Lean/torch stack is needed. Defaults to :func:`build_default_service`,
    which loads real models once at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        factory = service_factory
        if factory is None:
            from maths_ai.api.service import build_default_service

            factory = build_default_service
        try:
            app.state.service = await factory()
            app.state.init_error = None
        except Exception as exc:
            logger.exception("service initialization failed")
            app.state.service = None
            app.state.init_error = str(exc)
        yield
        service = getattr(app.state, "service", None)
        if service is not None:
            shutdown = getattr(service, "shutdown", None)
            if shutdown is not None:
                try:
                    await shutdown()
                except Exception:
                    logger.exception("service shutdown failed")

    app = FastAPI(
        title="Mathematics AI ATP",
        description="Hybrid GNN + PLN automated theorem proving over Lean/Pantograph.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ValueError)
    async def _value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        # Unparseable goal/hypothesis/expression syntax surfaces from the
        # inference layers as ValueError — that's a client-input problem.
        return JSONResponse(status_code=400, content={"detail": f"invalid input: {exc}"})

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        ready = getattr(request.app.state, "service", None) is not None
        return HealthResponse(
            status="ok" if ready else "degraded",
            ready=ready,
            error=getattr(request.app.state, "init_error", None),
        )

    @app.post("/prove", response_model=ProveResponse)
    async def prove(payload: ProveRequest, request: Request) -> ProveResponse:
        service = _get_service(request)
        
        # Use request timeout if provided, otherwise use service default
        timeout = payload.timeout if payload.timeout is not None else settings.api_default_prove_timeout
        
        try:
            graph = await service.prove(
                payload.goal,
                payload.hypotheses,
                max_depth=payload.max_depth,
                max_nodes=payload.max_nodes,
                timeout=timeout,
            )
            return ProveResponse.from_graph(graph, timeout=False)
        except asyncio.TimeoutError:
            # For timeout, we return a 200 with timeout termination reason
            # rather than a 408 error, so the client gets a proper response
            # The proof lock has already been released by the service layer
            from maths_ai.api.schemas import ProofTerminationReason
            return ProveResponse(
                solved=False,
                exhausted=False,
                termination_reason=ProofTerminationReason.TIMEOUT,
                proof_trace=None,
                summary={"error": "timeout"},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid goal or hypotheses: {exc}") from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"proof search failed: {exc}") from exc

    @app.post("/gnn/predict_tactic", response_model=PredictTacticResponse)
    async def predict_tactic(payload: PredictTacticRequest, request: Request) -> PredictTacticResponse:
        service = _get_service(request)
        try:
            candidates = await service.predict_tactics(payload.goal, payload.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid goal syntax: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"tactic prediction failed: {exc}") from exc
        return PredictTacticResponse(goal=payload.goal, candidates=candidates)

    @app.post("/pln/evaluate", response_model=EvaluateResponse)
    async def evaluate(payload: EvaluateRequest, request: Request) -> EvaluateResponse:
        service = _get_service(request)
        result: PLNResult = await service.evaluate(payload.expression, list(payload.hypotheses))
        if result.status == "render_error":
            # The symbolic renderer could not parse the expression/hypotheses
            # at all — surface it instead of dressing up a random fallback
            # score as evidence.
            raise HTTPException(status_code=400, detail="expression or hypotheses could not be parsed")
        return EvaluateResponse(
            expression=payload.expression,
            stv=result.stv,
            score=result.stv.score,
            status=result.status,
            is_fallback=result.is_fallback,
        )

    return app
