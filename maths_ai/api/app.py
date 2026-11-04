"""FastAPI application for MCTS actor-critic proving and evaluation.

Endpoints
---------
``GET  /health``                        process liveness; touches no model or Lean
``GET  /ready``                         200 when the runtime can accept work, else 503
``POST /prove``                         one bounded, certified-or-not proof search
``POST /evaluations``                   accept an evaluation job (202)
``GET  /evaluations/{job_id}``          its persisted state and progress
``GET  /evaluations/{job_id}/result``   its complete or explicitly partial result
``POST /evaluations/{job_id}/cancel``   request cancellation (202 until cleaned up)

Run with ``python -m maths_ai.api --config service.json``: one Uvicorn process,
which owns one runtime process. Errors map by category, never by exception
type alone: invalid input is 422, a full queue 429, an unready runtime 503 and
an exceeded deadline 504, which is not a mathematical refutation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import ServiceConfig
from .schemas import (
    ErrorBody,
    EvaluationRequest,
    Health,
    Job,
    ProveRequest,
    ProveResponse,
    Readiness,
)
from .service import ProvingService, ServiceError

logger = logging.getLogger("maths_ai.api")

ServiceFactory = Callable[[], Awaitable[Any]]

CONFIG_ENV = "MATHS_AI_SERVICE_CONFIG"
_ERRORS = {
    422: {"model": ErrorBody}, 429: {"model": ErrorBody}, 503: {"model": ErrorBody},
    504: {"model": ErrorBody}, 404: {"model": ErrorBody}, 409: {"model": ErrorBody},
}


def default_service_factory(config_path: Path | None = None) -> ServiceFactory:
    async def factory() -> ProvingService:
        path = Path(config_path or os.environ[CONFIG_ENV])
        service = ProvingService(ServiceConfig.from_json(path), path)
        await service.start()
        return service

    return factory


def _error(exc: ServiceError) -> JSONResponse:
    headers = {"Retry-After": "5"} if exc.status in (429, 503) else None
    return JSONResponse(status_code=exc.status, content={"category": exc.category, "detail": exc.detail},
                        headers=headers)


def create_app(service_factory: Optional[ServiceFactory] = None) -> FastAPI:
    """Build the app; tests inject a service factory, deployments use the config."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = await (service_factory or default_service_factory())()
        try:
            yield
        finally:
            await app.state.service.stop()

    app = FastAPI(
        title="Mathematics AI ATP: MCTS actor-critic proving",
        description=(
            "Kernel-certified proof search with the source-aware actor-critic policy, "
            "under PUCT or best-first profiles with sampled or greedy proposals, and "
            "persistent dataset evaluation jobs."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return _error(exc)

    @app.get("/health", response_model=Health)
    async def health() -> Health:
        return Health()

    @app.get("/ready", response_model=Readiness, responses={503: {"model": Readiness}})
    async def ready(request: Request):
        readiness = request.app.state.service.readiness()
        return JSONResponse(status_code=200 if readiness["ready"] else 503, content=readiness)

    @app.post("/prove", response_model=ProveResponse, responses=_ERRORS)
    async def prove(payload: ProveRequest, request: Request):
        service = request.app.state.service
        search = asyncio.create_task(service.prove(payload.model_dump()))

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
        job = await request.app.state.service.submit_evaluation(payload.model_dump())
        return Job(**job)

    @app.get("/evaluations/{job_id}", response_model=Job, responses=_ERRORS)
    async def job(job_id: str, request: Request):
        return Job(**await request.app.state.service.job(job_id))

    @app.get("/evaluations/{job_id}/result", responses=_ERRORS)
    async def job_result(job_id: str, request: Request):
        return await request.app.state.service.job_result(job_id)

    @app.post("/evaluations/{job_id}/cancel", response_model=Job, responses=_ERRORS)
    async def cancel(job_id: str, request: Request):
        job = await request.app.state.service.cancel_job(job_id)
        status = 200 if job["state"] in ("cancelled", "completed", "failed", "interrupted") else 202
        return JSONResponse(status_code=status, content=Job(**job).model_dump())

    return app
