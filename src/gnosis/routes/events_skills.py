"""Client event ingestion and skill registry routes."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from gnosis.auth import Authenticator
from gnosis.backend import MemoryBackend
from gnosis.models import (
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    EventIngestResult,
    EventIngestStatus,
    MessageWriteResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
)


def register_event_routes(
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
        _require_event_scope(authenticator, request)
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
                _require_event_scope(authenticator, event)
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


def _require_event_scope(authenticator: Authenticator, event: ClientEvent) -> None:
    authenticator.require_scope(event.scope)
    if event.tenant_id != event.scope.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="event tenant does not match scope tenant",
        )


def register_skill_routes(
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
