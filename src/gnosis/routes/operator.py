"""Operator-facing routes: stats, buffer, dedup, consolidation, graph, records.

Every response that can carry SDK-shaped payloads is passed through
``redact_secrets`` before leaving the service.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from gnosis.auth import Authenticator
from gnosis.backend import BackendRequestError, MemoryBackend
from gnosis.models import (
    BufferFlushResponse,
    ConsolidationApplyRequest,
    ConsolidationApplyResponse,
    ConsolidationDryRunRequest,
    ConsolidationDryRunResponse,
    DedupApplyRequest,
    DedupApplyResponse,
    DedupCandidateRequest,
    DedupCandidateResponse,
    DedupStatsRequest,
    DedupStatsResponse,
    EntityRecord,
    EntitySearchRequest,
    EntitySearchResponse,
    EntityWriteRequest,
    FactRecord,
    FactSearchRequest,
    FactSearchResponse,
    FactWriteRequest,
    GraphExportRequest,
    GraphExportResponse,
    JsonObject,
    MemoryScope,
    MemoryVisibility,
    PreferenceRecord,
    PreferenceSearchRequest,
    PreferenceSearchResponse,
    PreferenceWriteRequest,
    SdkStatsRequest,
    SdkStatsResponse,
)
from gnosis.redaction import redact_secrets
from gnosis.settings import Settings


def register_operator_routes(  # noqa: C901
    app: FastAPI,
    settings: Settings,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
    @app.get(
        "/v1/memory/stats",
        dependencies=[Depends(authenticator.require_read_operator)],
        response_model_exclude_none=True,
    )
    async def get_memory_stats(
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> SdkStatsResponse:
        response = await memory.get_sdk_stats(_tenant_stats_request(settings))
        return SdkStatsResponse(
            scope=response.scope,
            stats=redacted_object(response.stats),
        )

    @app.post(
        "/v1/sdk/stats",
        dependencies=[Depends(authenticator.require_read_operator)],
        include_in_schema=False,
    )
    async def get_sdk_stats(
        request: SdkStatsRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> SdkStatsResponse:
        authenticator.require_scope(request.scope)
        response = await memory.get_sdk_stats(request)
        return SdkStatsResponse(
            scope=response.scope,
            stats=redacted_object(response.stats),
        )

    @app.post(
        "/v1/memory/buffer/flush",
        dependencies=[Depends(authenticator.require_admin_operator)],
    )
    async def flush_buffer(
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> BufferFlushResponse:
        return await memory.flush_buffer()

    @app.get(
        "/v1/memory/dedup/stats",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def get_dedup_stats(
        request: DedupStatsRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> DedupStatsResponse:
        authenticator.require_scope(request.scope)
        response = await memory.get_dedup_stats(request)
        return DedupStatsResponse(
            scope=response.scope,
            stats=redacted_object(response.stats),
        )

    @app.post(
        "/v1/memory/dedup/candidates",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def find_dedup_candidates(
        request: DedupCandidateRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> DedupCandidateResponse:
        authenticator.require_scope(request.scope)
        return await memory.find_dedup_candidates(request)

    @app.post(
        "/v1/memory/dedup/apply",
        dependencies=[Depends(authenticator.require_admin_operator)],
    )
    async def apply_dedup_candidate(
        request: DedupApplyRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> DedupApplyResponse:
        authenticator.require_scope(request.scope)
        try:
            return await memory.apply_dedup_candidate(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error

    @app.post(
        "/v1/memory/consolidation/dry-run",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def dry_run_consolidation(
        request: ConsolidationDryRunRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ConsolidationDryRunResponse:
        authenticator.require_scope(request.scope)
        return await memory.dry_run_consolidation(request)

    @app.post(
        "/v1/memory/consolidation/apply",
        dependencies=[Depends(authenticator.require_admin_operator)],
    )
    async def apply_consolidation(
        request: ConsolidationApplyRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ConsolidationApplyResponse:
        authenticator.require_scope(request.scope)
        try:
            return await memory.apply_consolidation(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error

    @app.post(
        "/v1/memory/graph/export",
        dependencies=[Depends(authenticator.require_export_operator)],
    )
    @app.post(
        "/v1/graph/export",
        dependencies=[Depends(authenticator.require_export_operator)],
        include_in_schema=False,
    )
    async def export_graph(
        request: GraphExportRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> GraphExportResponse:
        authenticator.require_scope(request.scope)
        response = await memory.export_graph(request)
        return _redacted_graph_export(response)

    @app.post(
        "/v1/memory/entities/search",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def search_entities(
        request: EntitySearchRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> EntitySearchResponse:
        authenticator.require_scope(request.scope)
        return await memory.search_entities(request)

    @app.post(
        "/v1/memory/facts/search",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def search_facts(
        request: FactSearchRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> FactSearchResponse:
        authenticator.require_scope(request.scope)
        return await memory.search_facts(request)

    @app.post(
        "/v1/memory/preferences/search",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def search_preferences(
        request: PreferenceSearchRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> PreferenceSearchResponse:
        authenticator.require_scope(request.scope)
        return await memory.search_preferences(request)

    @app.post(
        "/v1/memory/entities",
        dependencies=[Depends(authenticator.require_write_operator)],
    )
    async def add_entity(
        request: EntityWriteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> EntityRecord:
        authenticator.require_scope(request.scope)
        return await memory.add_entity(request)

    @app.post(
        "/v1/memory/facts",
        dependencies=[Depends(authenticator.require_write_operator)],
    )
    async def add_fact(
        request: FactWriteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> FactRecord:
        authenticator.require_scope(request.scope)
        return await memory.add_fact(request)

    @app.post(
        "/v1/memory/preferences",
        dependencies=[Depends(authenticator.require_write_operator)],
    )
    async def add_preference(
        request: PreferenceWriteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> PreferenceRecord:
        authenticator.require_scope(request.scope)
        return await memory.add_preference(request)


def _tenant_stats_request(settings: Settings) -> SdkStatsRequest:
    tenant_id = settings.gnosis_tenant_id
    return SdkStatsRequest(
        scope=MemoryScope(
            tenant_id=tenant_id,
            space_id="tenant",
            agent_id="operator",
            session_id=f"tenant:{tenant_id}",
            user_id="operator",
            visibility=MemoryVisibility.TENANT,
        ),
    )


def _redacted_graph_export(response: GraphExportResponse) -> GraphExportResponse:
    return GraphExportResponse(
        scope=response.scope,
        nodes=[
            node.model_copy(update={"properties": redacted_object(node.properties)})
            for node in response.nodes
        ],
        relationships=[
            relationship.model_copy(
                update={"properties": redacted_object(relationship.properties)},
            )
            for relationship in response.relationships
        ],
        metadata=redacted_object(response.metadata),
    )


def redacted_object(value: JsonObject) -> JsonObject:
    redacted = redact_secrets(value)
    if isinstance(redacted, dict):
        return redacted
    return {}
