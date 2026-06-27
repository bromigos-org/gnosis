from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from agents_memory.auth import Authenticator, build_authenticator
from agents_memory.backend import MemoryBackend, Neo4jAgentMemoryBackend
from agents_memory.models import (
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ContextRequest,
    ContextResponse,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    HealthResponse,
    MessageWriteRequest,
    MessageWriteResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
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

    _register_health_route(app)
    _register_message_routes(app, authenticator, get_backend)
    _register_event_routes(app, authenticator, get_backend)
    _register_context_routes(app, authenticator, get_backend)
    _register_skill_routes(app, authenticator, get_backend)

    return app


def _register_health_route(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")


def _register_message_routes(
    app: FastAPI,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
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


def _register_event_routes(
    app: FastAPI,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
    @app.post(
        "/v1/events",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def ingest_event(
        request: ClientEvent,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> EventIngestResult:
        authenticator.require_scope(request.scope)
        return await memory.ingest_event(request)

    @app.post(
        "/v1/events/batch",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def ingest_events(
        request: ClientEventBatchRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ClientEventBatchResponse:
        results: list[EventIngestResult | None] = [None] * len(request.events)
        accepted_events: list[ClientEvent] = []
        accepted_positions: list[int] = []

        for index, event in enumerate(request.events):
            try:
                authenticator.require_scope(event.scope)
            except HTTPException as error:
                results[index] = EventIngestResult(
                    event_id=event.event_id,
                    status=EventIngestStatus.REJECTED,
                    reason=str(error.detail),
                )
                continue
            accepted_events.append(event)
            accepted_positions.append(index)

        if accepted_events:
            accepted_response = await memory.ingest_events(
                ClientEventBatchRequest(events=accepted_events),
            )
            for index, result in zip(
                accepted_positions,
                accepted_response.results,
                strict=True,
            ):
                results[index] = result

        return ClientEventBatchResponse(
            results=[result for result in results if result is not None],
        )


def _register_context_routes(
    app: FastAPI,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
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

    @app.post(
        "/v1/graph/context",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def get_graph_context(
        request: GraphContextRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> GraphContextResponse:
        authenticator.require_scope(request.scope)
        return await memory.get_graph_context(request)


def _register_skill_routes(
    app: FastAPI,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
    @app.post(
        "/v1/skills",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def list_skills(
        request: SkillListRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> SkillListResponse:
        authenticator.require_tenant(request.tenant_id)
        return await memory.list_skills(request)

    @app.post(
        "/v1/skills/proposals",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def propose_skill(
        request: SkillProposal,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> SkillProposal:
        authenticator.require_tenant(request.tenant_id)
        return await memory.propose_skill(request)

    @app.post(
        "/v1/skills/usage",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def record_skill_usage(
        request: SkillUsage,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MessageWriteResponse:
        authenticator.require_tenant(request.tenant_id)
        result = await memory.record_skill_usage(request)
        return MessageWriteResponse(
            accepted=result.status is EventIngestStatus.ACCEPTED,
        )


app = create_app()
