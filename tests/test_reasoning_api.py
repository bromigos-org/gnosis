from dataclasses import dataclass, field
from os import environ

from fastapi.testclient import TestClient

_ = environ.setdefault("GNOSIS_TOKEN", "reasoning-access")
_ = environ.setdefault("GNOSIS_READ_OPERATOR_TOKEN", "read-operator-token")
_ = environ.setdefault("GNOSIS_EXPORT_OPERATOR_TOKEN", "export-operator-token")
_ = environ.setdefault("GNOSIS_WRITE_OPERATOR_TOKEN", "write-operator-token")
_ = environ.setdefault("GNOSIS_ADMIN_OPERATOR_TOKEN", "admin-operator-token")
_ = environ.setdefault("NEO4J_URI", "bolt://neo4j.local:7687")
_ = environ.setdefault("NEO4J_PASSWORD", "inert-password")
_ = environ.setdefault("LITELLM_BASE_URL", "http://litellm.local/v1")
_ = environ.setdefault("LITELLM_API_KEY", "inert-litellm-key")

from gnosis.main import create_app  # noqa: E402
from gnosis.models import (  # noqa: E402
    BackendReadiness,
    BufferFlushResponse,
    BufferStatus,
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
    DiagnosticsConfig,
    DiagnosticsResponse,
    EntityRecord,
    EntitySearchRequest,
    EntitySearchResponse,
    EntityWriteRequest,
    EventIngestResult,
    EventIngestStatus,
    FactRecord,
    FactSearchRequest,
    FactSearchResponse,
    FactWriteRequest,
    GraphContextRequest,
    GraphContextResponse,
    GraphExportRequest,
    GraphExportResponse,
    MemoryContextRequest,
    MemoryContextResponse,
    MessageWriteRequest,
    MessageWriteResponse,
    PreferenceRecord,
    PreferenceSearchRequest,
    PreferenceSearchResponse,
    PreferenceWriteRequest,
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
from gnosis.settings import Settings  # noqa: E402


def test_reasoning_lifecycle_when_request_is_scoped() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    start_response = client.post(
        "/v1/reasoning/traces",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "session_id": "guild:123:channel:456",
            "task": "discord_reply",
            "metadata": {"channel_id": "456"},
            "triggered_by_message_id": "message-999",
        },
    )
    step_response = client.post(
        "/v1/reasoning/traces/trace-1/steps",
        headers=_auth_header(),
        json=_step_payload(),
    )
    tool_response = client.post(
        "/v1/reasoning/steps/step-1/tool-calls",
        headers=_auth_header(),
        json=_tool_call_payload(),
    )
    complete_response = client.post(
        "/v1/reasoning/traces/trace-1/complete",
        headers=_auth_header(),
        json=_completion_payload(),
    )

    assert start_response.status_code == 200
    assert start_response.json() == {
        "trace_id": "trace-1",
        "session_id": "guild:123:channel:456",
        "task": "discord_reply",
    }
    assert step_response.status_code == 200
    assert step_response.json() == {
        "step_id": "step-1",
        "trace_id": "trace-1",
        "step_number": 1,
    }
    assert tool_response.status_code == 200
    assert tool_response.json() == {
        "tool_call_id": "tool-call-1",
        "trace_id": "trace-1",
        "step_id": "step-1",
    }
    assert complete_response.status_code == 200
    assert complete_response.json() == {
        "trace_id": "trace-1",
        "success": True,
        "outcome": "sent reply",
        "completed_at": "2026-06-28T01:02:03Z",
    }
    assert backend.trace_starts == [
        ReasoningTraceStartRequest.model_validate(
            {
                "scope": _scope_payload(),
                "session_id": "guild:123:channel:456",
                "task": "discord_reply",
                "metadata": {"channel_id": "456"},
                "triggered_by_message_id": "message-999",
            },
        ),
    ]
    assert backend.steps[0].action == "get_memory_context"
    assert backend.tool_calls[0].touched_entities[0].id == "entity-1"
    assert backend.completions[0].success is True


def test_reasoning_context_redacts_secret_like_content() -> None:
    backend = RecordingBackend(
        reasoning_context=ReasoningContextResponse(
            context="result included API_KEY=plain-value",
            traces=[{"authorization": "plain-value", "safe": "kept"}],
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/context",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "what matters?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "context": "result included API_KEY=[REDACTED]",
        "traces": [{"authorization": "[REDACTED]", "safe": "kept"}],
    }
    assert backend.reasoning_context_requests == [
        ReasoningContextRequest.model_validate(
            {"scope": _scope_payload(), "query": "what matters?"},
        ),
    ]


def test_reasoning_route_requires_auth() -> None:
    client = TestClient(
        create_app(settings_factory=_settings, backend=RecordingBackend()),
    )

    response = client.post(
        "/v1/reasoning/context",
        json={"scope": _scope_payload(), "query": "what matters?"},
    )

    assert response.status_code == 401


def test_reasoning_route_rejects_cross_tenant_scope() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/traces",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(tenant_id="other-tenant"),
            "session_id": "guild:123:channel:456",
            "task": "discord_reply",
        },
    )

    assert response.status_code == 403
    assert backend.trace_starts == []


def test_reasoning_step_rejects_cross_tenant_scope() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/traces/trace-1/steps",
        headers=_auth_header(),
        json=_step_payload(scope=_scope_payload(tenant_id="other-tenant")),
    )

    assert response.status_code == 403
    assert backend.steps == []


def test_reasoning_tool_call_rejects_cross_tenant_scope() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/steps/step-1/tool-calls",
        headers=_auth_header(),
        json=_tool_call_payload(scope=_scope_payload(tenant_id="other-tenant")),
    )

    assert response.status_code == 403
    assert backend.tool_calls == []


def test_reasoning_completion_rejects_cross_tenant_scope() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/traces/trace-1/complete",
        headers=_auth_header(),
        json=_completion_payload(scope=_scope_payload(tenant_id="other-tenant")),
    )

    assert response.status_code == 403
    assert backend.completions == []


def test_reasoning_step_rejects_path_body_trace_mismatch() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/traces/other-trace/steps",
        headers=_auth_header(),
        json=_step_payload(),
    )

    assert response.status_code == 400
    assert backend.steps == []


def test_reasoning_tool_call_rejects_path_body_step_mismatch() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/steps/other-step/tool-calls",
        headers=_auth_header(),
        json=_tool_call_payload(),
    )

    assert response.status_code == 400
    assert backend.tool_calls == []


def test_reasoning_completion_rejects_path_body_trace_mismatch() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/traces/other-trace/complete",
        headers=_auth_header(),
        json=_completion_payload(),
    )

    assert response.status_code == 400
    assert backend.completions == []


@dataclass(slots=True)
class RecordingBackend:
    reasoning_context: ReasoningContextResponse = field(
        default_factory=lambda: ReasoningContextResponse(
            context="### Similar Past Tasks\n- resolved",
        ),
    )
    trace_starts: list[ReasoningTraceStartRequest] = field(default_factory=list)
    steps: list[ReasoningStepRequest] = field(default_factory=list)
    tool_calls: list[ReasoningToolCallRequest] = field(default_factory=list)
    completions: list[ReasoningTraceCompleteRequest] = field(default_factory=list)
    reasoning_context_requests: list[ReasoningContextRequest] = field(
        default_factory=list,
    )
    reasoning_trace_list_requests: list[ReasoningTraceListRequest] = field(
        default_factory=list,
    )
    reasoning_trace_detail_requests: list[ReasoningTraceDetailRequest] = field(
        default_factory=list,
    )
    reasoning_similar_trace_requests: list[ReasoningSimilarTracesRequest] = field(
        default_factory=list,
    )
    reasoning_step_search_requests: list[ReasoningStepSearchRequest] = field(
        default_factory=list,
    )
    reasoning_tool_stats_requests: list[ReasoningToolStatsRequest] = field(
        default_factory=list,
    )

    async def start_reasoning_trace(
        self,
        request: ReasoningTraceStartRequest,
    ) -> ReasoningTraceStartResponse:
        self.trace_starts.append(request)
        return ReasoningTraceStartResponse(
            trace_id="trace-1",
            session_id=request.session_id,
            task=request.task,
        )

    async def add_reasoning_step(
        self,
        request: ReasoningStepRequest,
    ) -> ReasoningStepResponse:
        self.steps.append(request)
        return ReasoningStepResponse(
            step_id="step-1",
            trace_id=request.trace_id,
            step_number=1,
        )

    async def record_reasoning_tool_call(
        self,
        request: ReasoningToolCallRequest,
    ) -> ReasoningToolCallResponse:
        self.tool_calls.append(request)
        return ReasoningToolCallResponse(
            tool_call_id="tool-call-1",
            trace_id=request.trace_id,
            step_id=request.step_id,
        )

    async def complete_reasoning_trace(
        self,
        request: ReasoningTraceCompleteRequest,
    ) -> ReasoningTraceCompleteResponse:
        self.completions.append(request)
        return ReasoningTraceCompleteResponse(
            trace_id=request.trace_id,
            success=request.success,
            outcome=request.outcome,
            completed_at="2026-06-28T01:02:03Z",
        )

    async def get_reasoning_context(
        self,
        request: ReasoningContextRequest,
    ) -> ReasoningContextResponse:
        self.reasoning_context_requests.append(request)
        return self.reasoning_context

    async def list_reasoning_traces(
        self,
        request: ReasoningTraceListRequest,
    ) -> ReasoningTraceListResponse:
        self.reasoning_trace_list_requests.append(request)
        return ReasoningTraceListResponse(scope=request.scope)

    async def get_reasoning_trace(
        self,
        request: ReasoningTraceDetailRequest,
    ) -> ReasoningTraceDetailResponse:
        self.reasoning_trace_detail_requests.append(request)
        return ReasoningTraceDetailResponse(scope=request.scope)

    async def find_similar_reasoning_traces(
        self,
        request: ReasoningSimilarTracesRequest,
    ) -> ReasoningSimilarTracesResponse:
        self.reasoning_similar_trace_requests.append(request)
        return ReasoningSimilarTracesResponse(scope=request.scope)

    async def search_reasoning_steps(
        self,
        request: ReasoningStepSearchRequest,
    ) -> ReasoningStepSearchResponse:
        self.reasoning_step_search_requests.append(request)
        return ReasoningStepSearchResponse(scope=request.scope)

    async def get_reasoning_tool_stats(
        self,
        request: ReasoningToolStatsRequest,
    ) -> ReasoningToolStatsResponse:
        self.reasoning_tool_stats_requests.append(request)
        return ReasoningToolStatsResponse(scope=request.scope)

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def buffer_status(self) -> BufferStatus:
        return BufferStatus(
            write_mode="sync",
            max_pending=200,
            pending_writes=None,
            write_errors=0,
            status="ready",
        )

    async def flush_buffer(self) -> BufferFlushResponse:
        return BufferFlushResponse(flushed=True, status=await self.buffer_status())

    async def shutdown(self) -> None:
        return None

    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse:
        settings = Settings()
        return DiagnosticsResponse(
            tenant_id=settings.gnosis_tenant_id,
            config=DiagnosticsConfig(
                neo4j_uri=settings.neo4j_uri,
                neo4j_username=settings.neo4j_username,
                litellm_base_url=settings.litellm_base_url,
                gnosis_llm=settings.gnosis_llm,
                gnosis_embedding=settings.gnosis_embedding,
                gnosis_embedding_dimensions=settings.gnosis_embedding_dimensions,
                gnosis_audit_read=settings.gnosis_audit_read,
                gnosis_conversation_ttl_days=settings.gnosis_conversation_ttl_days,
                gnosis_write_mode=settings.gnosis_write_mode,
                gnosis_max_pending=settings.gnosis_max_pending,
                gnosis_fact_deduplication_enabled=(
                    settings.gnosis_fact_deduplication_enabled
                ),
                gnosis_trace_embedding_enabled=settings.gnosis_trace_embedding_enabled,
                gnosis_extract_entities_enabled=(
                    settings.gnosis_extract_entities_enabled
                ),
                gnosis_extract_relations_enabled=(
                    settings.gnosis_extract_relations_enabled
                ),
                gnosis_extraction_preview_enabled=(
                    settings.gnosis_extraction_preview_enabled
                ),
                gnosis_extraction_batch_size=settings.gnosis_extraction_batch_size,
                gnosis_extraction_max_concurrency=(
                    settings.gnosis_extraction_max_concurrency
                ),
                gnosis_extraction_chunk_size=settings.gnosis_extraction_chunk_size,
                gnosis_extraction_chunk_overlap=(
                    settings.gnosis_extraction_chunk_overlap
                ),
                gnosis_ocr_enabled=settings.gnosis_ocr_enabled,
                gnosis_ocr_model=settings.gnosis_ocr_model,
                gnosis_ocr_max_image_bytes=settings.gnosis_ocr_max_image_bytes,
                gnosis_rustfs_enabled=settings.gnosis_rustfs_enabled,
                gnosis_rustfs_bucket=settings.gnosis_rustfs_bucket,
                gnosis_rustfs_prefix=settings.gnosis_rustfs_prefix,
                gnosis_rustfs_endpoint=settings.gnosis_rustfs_endpoint,
                gnosis_rustfs_retention_days=settings.gnosis_rustfs_retention_days,
                gnosis_prompt_entities_enabled=settings.gnosis_prompt_entities_enabled,
                gnosis_prompt_preferences_enabled=(
                    settings.gnosis_prompt_preferences_enabled
                ),
                gnosis_prompt_reasoning_enabled=settings.gnosis_prompt_reasoning_enabled,
                gnosis_consolidation_schedule_enabled=(
                    settings.gnosis_consolidation_schedule_enabled
                ),
            ),
            backend=readiness,
        )

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        _ = request
        return MessageWriteResponse(accepted=True)

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        _ = request
        return ContextResponse(context="")

    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        _ = request
        return MemoryContextResponse()

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def ingest_events(
        self,
        request: ClientEventBatchRequest,
    ) -> ClientEventBatchResponse:
        return ClientEventBatchResponse(
            results=[await self.ingest_event(event) for event in request.events],
        )

    async def get_graph_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse:
        _ = request
        return GraphContextResponse(context="")

    async def get_sdk_stats(self, request: SdkStatsRequest) -> SdkStatsResponse:
        return SdkStatsResponse(scope=request.scope, stats={})

    async def get_dedup_stats(
        self,
        request: DedupStatsRequest,
    ) -> DedupStatsResponse:
        return DedupStatsResponse(scope=request.scope, stats={})

    async def find_dedup_candidates(
        self,
        request: DedupCandidateRequest,
    ) -> DedupCandidateResponse:
        return DedupCandidateResponse(
            scope=request.scope,
            graph_snapshot_hash="snapshot-placeholder",
            expires_at="2026-06-29T00:15:00+00:00",
        )

    async def apply_dedup_candidate(
        self,
        request: DedupApplyRequest,
    ) -> DedupApplyResponse:
        return DedupApplyResponse(
            scope=request.scope,
            operation=request.operation,
            candidate_id=request.candidate_id,
            candidate_version=request.candidate_version,
            applied=True,
            audit=request.audit,
        )

    async def dry_run_consolidation(
        self,
        request: ConsolidationDryRunRequest,
    ) -> ConsolidationDryRunResponse:
        return ConsolidationDryRunResponse(
            scope=request.scope,
            operation=request.operation,
            dry_run=True,
            graph_snapshot_hash="snapshot-placeholder",
            dry_run_token="consolidation-token",  # noqa: S106
            expires_at="2026-06-29T00:15:00+00:00",
        )

    async def apply_consolidation(
        self,
        request: ConsolidationApplyRequest,
    ) -> ConsolidationApplyResponse:
        return ConsolidationApplyResponse(
            scope=request.scope,
            operation=request.operation,
            applied=True,
            audit=request.audit,
        )

    async def export_graph(
        self,
        request: GraphExportRequest,
    ) -> GraphExportResponse:
        return GraphExportResponse(scope=request.scope)

    async def search_entities(
        self,
        request: EntitySearchRequest,
    ) -> EntitySearchResponse:
        _ = request
        return EntitySearchResponse()

    async def search_facts(self, request: FactSearchRequest) -> FactSearchResponse:
        _ = request
        return FactSearchResponse()

    async def search_preferences(
        self,
        request: PreferenceSearchRequest,
    ) -> PreferenceSearchResponse:
        _ = request
        return PreferenceSearchResponse()

    async def add_entity(self, request: EntityWriteRequest) -> EntityRecord:
        return EntityRecord(name=request.name, type=request.type)

    async def add_fact(self, request: FactWriteRequest) -> FactRecord:
        return FactRecord(
            subject=request.subject,
            predicate=request.predicate,
            object=request.object,
        )

    async def add_preference(
        self,
        request: PreferenceWriteRequest,
    ) -> PreferenceRecord:
        return PreferenceRecord(
            category=request.category,
            preference=request.preference,
        )

    async def list_skills(self, request: SkillListRequest) -> SkillListResponse:
        _ = request
        return SkillListResponse()

    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal:
        return proposal

    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult:
        return EventIngestResult(
            event_id=usage.skill_id,
            status=EventIngestStatus.ACCEPTED,
        )


def _settings() -> Settings:
    return Settings()


def _auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {environ['GNOSIS_TOKEN']}"}


def _scope_payload(*, tenant_id: str = "bromigos") -> dict[str, str]:
    return {
        "tenant_id": tenant_id,
        "space_id": "discord",
        "agent_id": "pc-principal",
        "session_id": "guild:123:channel:456",
        "user_id": "789",
        "visibility": "channel",
        "guild_id": "123",
        "channel_id": "456",
    }


def _step_payload(
    *,
    trace_id: str = "trace-1",
    scope: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "scope": scope or _scope_payload(),
        "trace_id": trace_id,
        "action": "get_memory_context",
        "observation": "combined memory returned",
    }


def _tool_call_payload(
    *,
    step_id: str = "step-1",
    scope: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "scope": scope or _scope_payload(),
        "trace_id": "trace-1",
        "step_id": step_id,
        "tool_name": "memory.get_context",
        "arguments": {"query": "what matters?"},
        "result": {"matches": 2},
        "status": "success",
        "duration_ms": 12,
        "message_id": "message-999",
        "touched_entities": [
            {"id": "entity-1", "name": "cartman", "type": "user"},
        ],
    }


def _completion_payload(
    *,
    trace_id: str = "trace-1",
    scope: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "scope": scope or _scope_payload(),
        "trace_id": trace_id,
        "outcome": "sent reply",
        "success": True,
    }
