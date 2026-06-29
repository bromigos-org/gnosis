from dataclasses import dataclass, field
from os import environ

from fastapi.testclient import TestClient

_ = environ.setdefault("AGENTS_MEMORY_TOKEN", "reasoning-access")
_ = environ.setdefault("NEO4J_URI", "bolt://neo4j.local:7687")
_ = environ.setdefault("NEO4J_PASSWORD", "inert-password")
_ = environ.setdefault("LITELLM_BASE_URL", "http://litellm.local/v1")
_ = environ.setdefault("LITELLM_API_KEY", "inert-litellm-key")

from agents_memory.main import create_app  # noqa: E402
from agents_memory.models import (  # noqa: E402
    BackendReadiness,
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ContextRequest,
    ContextResponse,
    DiagnosticsConfig,
    DiagnosticsResponse,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    MemoryContextRequest,
    MemoryContextResponse,
    MessageWriteRequest,
    MessageWriteResponse,
    ReasoningContextRequest,
    ReasoningContextResponse,
    ReasoningStepRequest,
    ReasoningStepResponse,
    ReasoningToolCallRequest,
    ReasoningToolCallResponse,
    ReasoningTraceCompleteRequest,
    ReasoningTraceCompleteResponse,
    ReasoningTraceStartRequest,
    ReasoningTraceStartResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
)
from agents_memory.settings import Settings  # noqa: E402


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

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse:
        return DiagnosticsResponse(
            tenant_id="bromigos",
            config=DiagnosticsConfig(
                neo4j_uri="bolt://neo4j.local:7687",
                neo4j_username="neo4j",
                litellm_base_url="http://litellm.local/v1",
                memory_llm="openai/gemma4",
                memory_embedding="local-qwen3-embedding-0.6b",
                memory_embedding_dimensions=1024,
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
    return {"Authorization": f"Bearer {environ['AGENTS_MEMORY_TOKEN']}"}


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


def _step_payload(*, trace_id: str = "trace-1") -> dict[str, str]:
    return {
        "trace_id": trace_id,
        "action": "get_memory_context",
        "observation": "combined memory returned",
    }


def _tool_call_payload(*, step_id: str = "step-1") -> dict[str, object]:
    return {
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


def _completion_payload(*, trace_id: str = "trace-1") -> dict[str, object]:
    return {"trace_id": trace_id, "outcome": "sent reply", "success": True}
