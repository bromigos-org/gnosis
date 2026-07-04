import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Final

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from gnosis.auth import Authenticator, MemoryCaller, build_authenticator
from gnosis.backend import (
    BackendCapabilityUnavailable,
    BackendRequestError,
    ExtractionPreviewBackend,
    MemoryBackend,
    MemoryNotFoundError,
    Neo4jAgentMemoryBackend,
    RecallFilteringBackend,
)
from gnosis.federation import (
    FederationGateway,
    FederationTransport,
    PeerNotAllowedError,
    PeerTokenUnavailableError,
    UnknownPeerError,
    merged_search_results,
    shareable_filters,
)
from gnosis.mcp_server import BearerTokenMiddleware, build_mcp_server
from gnosis.models import (
    BackendReadiness,
    BufferFlushResponse,
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ConsolidationApplyRequest,
    ConsolidationApplyResponse,
    ConsolidationDryRunRequest,
    ConsolidationDryRunResponse,
    ContextRequest,
    ContextResponse,
    DedupApplyRequest,
    DedupApplyResponse,
    DedupCandidateRequest,
    DedupCandidateResponse,
    DedupStatsRequest,
    DedupStatsResponse,
    DiagnosticsResponse,
    EntityRecord,
    EntitySearchRequest,
    EntitySearchResponse,
    EntityWriteRequest,
    EventIngestResult,
    EventIngestStatus,
    ExtractionPreviewRequest,
    ExtractionPreviewResponse,
    FactRecord,
    FactSearchRequest,
    FactSearchResponse,
    FactWriteRequest,
    GraphContextRequest,
    GraphContextResponse,
    GraphExportRequest,
    GraphExportResponse,
    HealthResponse,
    JsonObject,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryListRequest,
    MemoryListResponse,
    MemoryPromoteCandidate,
    MemoryPromoteRequest,
    MemoryPromoteResponse,
    MemoryScope,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
    MemoryVisibility,
    MessageWriteRequest,
    MessageWriteResponse,
    PreferenceRecord,
    PreferenceSearchRequest,
    PreferenceSearchResponse,
    PreferenceWriteRequest,
    ReadinessResponse,
    ReasoningContextRequest,
    ReasoningContextResponse,
    ReasoningSimilarTracesRequest,
    ReasoningSimilarTracesResponse,
    ReasoningStepRequest,
    ReasoningStepResponse,
    ReasoningStepSearchRequest,
    ReasoningStepSearchResponse,
    ReasoningToolCallRequest,
    ReasoningToolCallResponse,
    ReasoningToolStatsRequest,
    ReasoningToolStatsResponse,
    ReasoningTraceCompleteRequest,
    ReasoningTraceCompleteResponse,
    ReasoningTraceDetailRequest,
    ReasoningTraceDetailResponse,
    ReasoningTraceListRequest,
    ReasoningTraceListResponse,
    ReasoningTraceStartRequest,
    ReasoningTraceStartResponse,
    SdkStatsRequest,
    SdkStatsResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
)
from gnosis.redaction import redact_secrets
from gnosis.settings import Settings, load_settings

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_LEGACY_CONTEXT_ROUTE: Final = "/v1/context"
_MEMORY_CONTEXT_ROUTE: Final = "/v1/memory/context"
_LEGACY_CONTEXT_SUCCESSOR_LINK: Final = (
    f'<{_MEMORY_CONTEXT_ROUTE}>; rel="successor-version"'
)


def _legacy_context_warning() -> Callable[[], None]:
    """Build a warner that logs deprecated /v1/context usage once per process."""
    emitted = False

    def warn() -> None:
        nonlocal emitted
        if emitted:
            return
        emitted = True
        _LOGGER.warning(
            "Deprecated route %s was called; migrate to %s.",
            _LEGACY_CONTEXT_ROUTE,
            _MEMORY_CONTEXT_ROUTE,
            extra={
                "deprecated_route": _LEGACY_CONTEXT_ROUTE,
                "successor_route": _MEMORY_CONTEXT_ROUTE,
            },
        )

    return warn


def create_app(
    settings_factory: Callable[[], Settings] = load_settings,
    backend: MemoryBackend | None = None,
    federation_transport: FederationTransport | None = None,
) -> FastAPI:
    settings = settings_factory()
    memory_backend = backend or Neo4jAgentMemoryBackend(settings)
    authenticator = build_authenticator(settings)
    federation = FederationGateway(settings, transport=federation_transport)

    def get_backend() -> MemoryBackend:
        return memory_backend

    mcp_server = (
        build_mcp_server(settings, get_backend) if settings.gnosis_mcp_enabled else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        try:
            if mcp_server is None:
                yield
            else:
                async with mcp_server.session_manager.run():
                    yield
        finally:
            await memory_backend.shutdown()

    app = FastAPI(title="gnosis", lifespan=lifespan)
    _register_exception_handlers(app)

    if mcp_server is not None:
        app.mount(
            "/mcp",
            BearerTokenMiddleware(
                app=mcp_server.streamable_http_app(),
                token=settings.gnosis_token,
            ),
        )

    _register_health_route(app)
    _register_readiness_routes(app, settings, authenticator, get_backend)
    _register_message_routes(app, authenticator, get_backend)
    _register_memory_provider_routes(
        app,
        settings,
        authenticator,
        get_backend,
        federation,
    )
    _register_event_routes(app, authenticator, get_backend)
    _register_context_routes(app, authenticator, get_backend)
    _register_operator_routes(app, settings, authenticator, get_backend)
    _register_reasoning_routes(app, authenticator, get_backend)
    _register_skill_routes(app, authenticator, get_backend)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BackendCapabilityUnavailable)
    async def capability_unavailable(
        _request: object,
        error: BackendCapabilityUnavailable,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={"detail": "capability_unavailable", "message": error.detail},
        )

    @app.exception_handler(UnknownPeerError)
    async def unknown_peer(
        _request: object,
        error: UnknownPeerError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": error.detail})

    @app.exception_handler(PeerNotAllowedError)
    async def peer_not_allowed(
        _request: object,
        error: PeerNotAllowedError,
    ) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": error.detail})

    @app.exception_handler(PeerTokenUnavailableError)
    async def peer_token_unavailable(
        _request: object,
        error: PeerTokenUnavailableError,
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": error.detail})


def _register_health_route(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")


def _register_readiness_routes(
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
        try:
            return await memory.add_message(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error

    @app.post(
        "/v1/memory/extraction/preview",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def preview_extraction(
        request: ExtractionPreviewRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ExtractionPreviewResponse:
        authenticator.require_scope(request.scope)
        if not isinstance(memory, ExtractionPreviewBackend):
            raise HTTPException(
                status_code=501,
                detail="Extraction preview is unavailable.",
            )
        try:
            return await memory.preview_extraction(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error


def _register_memory_provider_routes(  # noqa: C901, PLR0915 - route grouping is intentional.
    app: FastAPI,
    settings: Settings,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
    federation: FederationGateway,
) -> None:
    @app.post("/v1/memories")
    async def add_memories(
        request: MemoryAddRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
        caller: Annotated[MemoryCaller, Depends(authenticator.resolve_memory_caller)],
    ) -> MemoryAddResponse:
        authenticator.require_scope(request.scope)
        if caller is MemoryCaller.FEDERATED:
            _require_federated_add_provenance(request)
        try:
            return await memory.add_memories(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error

    @app.post("/v1/memories/search")
    async def search_memories(
        request: MemorySearchRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
        caller: Annotated[MemoryCaller, Depends(authenticator.resolve_memory_caller)],
    ) -> MemorySearchResponse:
        authenticator.require_scope(request.scope)
        if caller is MemoryCaller.FEDERATED:
            _require_no_federated_fanout(request)
            request = request.model_copy(
                update={"filters": shareable_filters(request.filters)},
            )
        for peer_name in request.peers:
            _ = federation.require_pull_peer(peer_name)
        try:
            local = await memory.search_memories(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error
        if not request.peers:
            return local
        remote, peer_errors = await federation.search_peers(request)
        results = merged_search_results(local.results, remote, request.limit)
        if isinstance(memory, RecallFilteringBackend):
            # Federated searches run the recall filter here, once over the
            # merged result set; remote results are already shareable-only
            # and the filter can only remove or keep records.
            results = await memory.filter_recalled_memories(request.query, results)
        return MemorySearchResponse(results=results, peer_errors=peer_errors)

    @app.post("/v1/memories/list")
    async def list_memories(
        request: MemoryListRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
        caller: Annotated[MemoryCaller, Depends(authenticator.resolve_memory_caller)],
    ) -> MemoryListResponse:
        authenticator.require_scope(request.scope)
        if caller is MemoryCaller.FEDERATED:
            request = request.model_copy(
                update={"filters": shareable_filters(request.filters)},
            )
        try:
            return await memory.list_memories(request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error

    @app.post(
        "/v1/memories/promote",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def promote_memories(
        request: MemoryPromoteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MemoryPromoteResponse:
        authenticator.require_scope(request.scope)
        peer = federation.require_push_peer(request.peer)
        try:
            listing = await memory.list_memories(
                MemoryListRequest(
                    scope=request.scope,
                    filters=shareable_filters(request.filters),
                    page=1,
                    page_size=request.limit,
                ),
            )
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error
        candidates = [
            MemoryPromoteCandidate(
                memory_id=record.memory_id,
                content=record.content,
                metadata=record.metadata,
            )
            for record in listing.results
        ]
        if request.dry_run:
            return MemoryPromoteResponse(
                peer=peer.name,
                count=len(candidates),
                dry_run=True,
                candidates=candidates,
            )
        outcome = await federation.promote(peer, request.scope, candidates)
        _LOGGER.info(
            "memory promotion applied",
            extra={
                "peer": peer.name,
                "tenant_id": request.scope.tenant_id,
                "user_id": request.scope.user_id,
                "promoted": len(outcome.promoted),
                "failed": len(outcome.failed),
            },
        )
        return MemoryPromoteResponse(
            peer=peer.name,
            count=len(candidates),
            dry_run=False,
            promoted=outcome.promoted,
            failed=outcome.failed,
        )

    @app.patch(
        "/v1/memories/{memory_id}",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def update_memory(
        memory_id: str,
        request: MemoryUpdateRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MemoryUpdateResponse:
        _require_memory_edit_enabled(settings)
        authenticator.require_scope(request.scope)
        try:
            return await memory.update_memory(memory_id, request)
        except BackendRequestError as error:
            raise HTTPException(status_code=400, detail=error.detail) from error
        except MemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail=error.detail) from error

    @app.delete(
        "/v1/memories/{memory_id}",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def delete_memory(
        memory_id: str,
        request: MemoryDeleteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MemoryDeleteResponse:
        _require_memory_edit_enabled(settings)
        authenticator.require_scope(request.scope)
        try:
            return await memory.delete_memory(memory_id, request)
        except MemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail=error.detail) from error


def _require_memory_edit_enabled(settings: Settings) -> None:
    if not settings.gnosis_memory_edit_enabled:
        raise HTTPException(
            status_code=403,
            detail="Memory editing is disabled by service policy.",
        )


def _require_federated_add_provenance(request: MemoryAddRequest) -> None:
    promoted_from = request.metadata.get("promoted_from")
    if not (isinstance(promoted_from, str) and promoted_from):
        raise HTTPException(
            status_code=403,
            detail="federated writes require metadata.promoted_from",
        )


def _require_no_federated_fanout(request: MemorySearchRequest) -> None:
    if request.peers:
        raise HTTPException(
            status_code=403,
            detail="federated callers cannot request peer fan-out",
        )


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


def _register_context_routes(
    app: FastAPI,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
    warn_legacy_context_route_used = _legacy_context_warning()

    @app.post(
        _LEGACY_CONTEXT_ROUTE,
        dependencies=[Depends(authenticator.require_token)],
        deprecated=True,
    )
    async def get_context(
        request: ContextRequest,
        response: Response,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ContextResponse:
        authenticator.require_scope(request.scope)
        warn_legacy_context_route_used()
        response.headers["Deprecation"] = "true"
        response.headers["Link"] = _LEGACY_CONTEXT_SUCCESSOR_LINK
        return await memory.get_context(request)

    @app.post(
        _MEMORY_CONTEXT_ROUTE,
        dependencies=[Depends(authenticator.require_token)],
    )
    async def get_memory_context(
        request: MemoryContextRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> MemoryContextResponse:
        authenticator.require_scope(request.scope)
        return await memory.get_memory_context(request)

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


def _register_operator_routes(  # noqa: C901
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
            stats=_redacted_object(response.stats),
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
            stats=_redacted_object(response.stats),
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
            stats=_redacted_object(response.stats),
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


def _register_reasoning_routes(  # noqa: C901 - FastAPI route grouping is intentional.
    app: FastAPI,
    authenticator: Authenticator,
    get_backend: Callable[[], MemoryBackend],
) -> None:
    @app.post(
        "/v1/reasoning/traces",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def start_reasoning_trace(
        request: ReasoningTraceStartRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningTraceStartResponse:
        authenticator.require_scope(request.scope)
        return await memory.start_reasoning_trace(request)

    @app.post(
        "/v1/reasoning/traces/{trace_id}/steps",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def add_reasoning_step(
        trace_id: str,
        request: ReasoningStepRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningStepResponse:
        _require_matching_identifier(
            path_value=trace_id,
            body_value=request.trace_id,
            field_name="trace_id",
        )
        authenticator.require_scope(request.scope)
        return await memory.add_reasoning_step(request)

    @app.post(
        "/v1/reasoning/steps/{step_id}/tool-calls",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def record_reasoning_tool_call(
        step_id: str,
        request: ReasoningToolCallRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningToolCallResponse:
        _require_matching_identifier(
            path_value=step_id,
            body_value=request.step_id,
            field_name="step_id",
        )
        authenticator.require_scope(request.scope)
        return await memory.record_reasoning_tool_call(request)

    @app.post(
        "/v1/reasoning/traces/{trace_id}/complete",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def complete_reasoning_trace(
        trace_id: str,
        request: ReasoningTraceCompleteRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningTraceCompleteResponse:
        _require_matching_identifier(
            path_value=trace_id,
            body_value=request.trace_id,
            field_name="trace_id",
        )
        authenticator.require_scope(request.scope)
        return await memory.complete_reasoning_trace(request)

    @app.post(
        "/v1/reasoning/context",
        dependencies=[Depends(authenticator.require_token)],
    )
    async def get_reasoning_context(
        request: ReasoningContextRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningContextResponse:
        authenticator.require_scope(request.scope)
        response = await memory.get_reasoning_context(request)
        return ReasoningContextResponse(
            context=_redacted_context(response.context),
            traces=_redacted_traces(response.traces),
        )

    @app.post(
        "/v1/reasoning/traces/list",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def list_reasoning_traces(
        request: ReasoningTraceListRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningTraceListResponse:
        authenticator.require_scope(request.scope)
        return await memory.list_reasoning_traces(request)

    @app.post(
        "/v1/reasoning/traces/{trace_id}/detail",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def get_reasoning_trace(
        trace_id: str,
        request: ReasoningTraceDetailRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningTraceDetailResponse:
        authenticator.require_scope(request.scope)
        _require_matching_identifier(
            path_value=trace_id,
            body_value=request.trace_id,
            field_name="trace_id",
        )
        return await memory.get_reasoning_trace(request)

    @app.post(
        "/v1/reasoning/traces/similar",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def find_similar_reasoning_traces(
        request: ReasoningSimilarTracesRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningSimilarTracesResponse:
        authenticator.require_scope(request.scope)
        return await memory.find_similar_reasoning_traces(request)

    @app.post(
        "/v1/reasoning/steps/search",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def search_reasoning_steps(
        request: ReasoningStepSearchRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningStepSearchResponse:
        authenticator.require_scope(request.scope)
        return await memory.search_reasoning_steps(request)

    @app.post(
        "/v1/reasoning/tools/stats",
        dependencies=[Depends(authenticator.require_read_operator)],
    )
    async def get_reasoning_tool_stats(
        request: ReasoningToolStatsRequest,
        memory: Annotated[MemoryBackend, Depends(get_backend)],
    ) -> ReasoningToolStatsResponse:
        authenticator.require_scope(request.scope)
        return await memory.get_reasoning_tool_stats(request)


def _redacted_context(context: str) -> str:
    redacted = redact_secrets(context)
    if isinstance(redacted, str):
        return redacted
    return context


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
            node.model_copy(update={"properties": _redacted_object(node.properties)})
            for node in response.nodes
        ],
        relationships=[
            relationship.model_copy(
                update={"properties": _redacted_object(relationship.properties)},
            )
            for relationship in response.relationships
        ],
        metadata=_redacted_object(response.metadata),
    )


def _redacted_object(value: JsonObject) -> JsonObject:
    redacted = redact_secrets(value)
    if isinstance(redacted, dict):
        return redacted
    return {}


def _require_matching_identifier(
    *,
    path_value: str,
    body_value: str,
    field_name: str,
) -> None:
    if path_value != body_value:
        raise HTTPException(
            status_code=400,
            detail=f"Path {field_name} must match request body {field_name}.",
        )


def _redacted_traces(traces: list[JsonObject]) -> list[JsonObject]:
    redacted_traces: list[JsonObject] = []
    for trace in traces:
        redacted = redact_secrets(trace)
        if isinstance(redacted, dict):
            redacted_traces.append(redacted)
    return redacted_traces


app = create_app()
