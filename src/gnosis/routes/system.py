"""Health, readiness, and diagnostics routes."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from gnosis.auth import Authenticator
from gnosis.backend import MemoryBackend
from gnosis.models import (
    BackendReadiness,
    DiagnosticsResponse,
    HealthResponse,
    ReadinessResponse,
)
from gnosis.settings import Settings


def register_health_route(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")


def register_readiness_routes(
    app: FastAPI,
    settings: Settings,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
    @app.get("/ready", response_model=ReadinessResponse)
    async def ready(
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReadinessResponse | JSONResponse:
        readiness = await memory.readiness()
        if _is_ready(readiness):
            return ReadinessResponse(status="ready")
        return JSONResponse(
            status_code=503,
            content=ReadinessResponse(status="unavailable").model_dump(by_alias=True),
        )

    @app.get(
        "/v1/diagnostics",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def diagnostics(
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> DiagnosticsResponse:
        authenticator.require_tenant(settings.gnosis_tenant_id)
        return memory.diagnostics(await memory.readiness())


def _is_ready(readiness: BackendReadiness) -> bool:
    return (
        readiness.graph == "ready"
        and readiness.schema_status == "ready"
        and readiness.buffer_status == "ready"
    )
