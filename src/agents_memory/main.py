from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI

from agents_memory.auth import build_authenticator
from agents_memory.backend import MemoryBackend, Neo4jAgentMemoryBackend
from agents_memory.models import (
    ContextRequest,
    ContextResponse,
    HealthResponse,
    MessageWriteRequest,
    MessageWriteResponse,
)
from agents_memory.settings import Settings, load_settings


def create_app(
    settings_factory: Callable[[], Settings] = load_settings,
    backend: MemoryBackend | None = None,
) -> FastAPI:
    settings = settings_factory()
    memory_backend = backend or Neo4jAgentMemoryBackend(settings)
    authenticator = build_authenticator(settings)

    app = FastAPI(title="agents-memory")

    def get_backend() -> MemoryBackend:
        return memory_backend

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post(
        "/v1/messages",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def write_message(
        request: MessageWriteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MessageWriteResponse:
        authenticator.require_scope(request.scope)
        return await memory.add_message(request)

    @app.post(
        "/v1/context",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def get_context(
        request: ContextRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ContextResponse:
        authenticator.require_scope(request.scope)
        return await memory.get_context(request)

    return app


app = create_app()
