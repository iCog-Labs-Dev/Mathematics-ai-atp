"""HTTP routes of formal protocol version 2.

Request bodies are validated here rather than by FastAPI so that every
rejection, including an unsupported ``protocol_version``, has the typed
``{protocol_version, category, detail}`` body. Run ids may contain ``/``.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .contract import PROTOCOL_VERSION, FormalReplayRequest, FormalSearchRequest
from .service import FormalError, FormalService


def _error(status: int, category: str, detail: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"protocol_version": PROTOCOL_VERSION, "category": category, "detail": detail},
    )


async def _body(request: Request, model):
    try:
        payload = json.loads(await request.body())
    except ValueError:
        raise FormalError(422, "invalid-request", "The body is not JSON.")
    if not isinstance(payload, dict):
        raise FormalError(422, "invalid-request", "The body must be a JSON object.")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise FormalError(
            422, "unsupported-protocol-version",
            f"This provider implements protocol_version {PROTOCOL_VERSION}; "
            f"got {payload.get('protocol_version')!r}.",
        )
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        errors = [
            {"loc": list(item["loc"]), "type": item["type"], "msg": item["msg"]}
            for item in exc.errors(include_url=False, include_context=False, include_input=False)
        ]
        raise FormalError(422, "invalid-request", errors) from None


def formal_router(get_service: Callable[[Request], FormalService]) -> APIRouter:
    router = APIRouter()

    async def guarded(request: Request, handler, status: int = 200):
        try:
            service = get_service(request)
            return JSONResponse(status_code=status, content=await handler(service))
        except FormalError as exc:
            return _error(exc.status, exc.category, exc.detail)

    @router.get("/formal-capabilities")
    async def capabilities(request: Request):
        async def handler(service):
            return service.capabilities()
        return await guarded(request, handler)

    @router.post("/formal-artifacts")
    async def upload(request: Request):
        async def handler(service):
            data = await request.body()
            return {"protocol_version": PROTOCOL_VERSION, "artifact_hash": service.put_artifact(data)}
        return await guarded(request, handler, 201)

    @router.get("/formal-artifacts/{artifact_hash}")
    async def download(artifact_hash: str, request: Request):
        from fastapi.responses import Response

        from .artifacts import ArtifactError

        try:
            data = get_service(request).artifacts.get(artifact_hash)
        except FormalError as exc:
            return _error(exc.status, exc.category, exc.detail)
        except ArtifactError as exc:
            status = 404 if exc.category == "artifact-missing" else 422
            return _error(status, exc.category, exc.detail)
        return Response(content=data, media_type="application/octet-stream",
                        headers={"X-Content-Hash": artifact_hash})

    @router.post("/formal-search/start")
    async def start(request: Request):
        try:
            service = get_service(request)
            payload = await _body(request, FormalSearchRequest)
            ack, created = await service.start(payload)
        except FormalError as exc:
            return _error(exc.status, exc.category, exc.detail)
        return JSONResponse(status_code=202 if created else 200, content=ack)

    @router.post("/formal-search/{run_id:path}/cancel")
    async def cancel(run_id: str, request: Request):
        async def handler(service):
            return await service.cancel(run_id)
        return await guarded(request, handler, 202)

    @router.post("/formal-search/{run_id:path}/retry")
    async def retry(run_id: str, request: Request):
        async def handler(service):
            return await service.retry(run_id)
        return await guarded(request, handler, 202)

    @router.get("/formal-search/{run_id:path}")
    async def status(run_id: str, request: Request):
        async def handler(service):
            return service.get(run_id)
        return await guarded(request, handler)

    @router.post("/formal-replay")
    async def replay(request: Request):
        try:
            service = get_service(request)
            payload = await _body(request, FormalReplayRequest)
            return JSONResponse(content=await service.replay(payload))
        except FormalError as exc:
            return _error(exc.status, exc.category, exc.detail)

    return router
