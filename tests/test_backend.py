from dataclasses import dataclass, field
from os import environ
from typing import TYPE_CHECKING, Self
from uuid import UUID

import pytest
from neo4j_agent_memory import MemorySettings
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolCallStatus
from neo4j_agent_memory.schema.models import EntityRef

environ["AGENTS_MEMORY_TOKEN"] = "test-token"
environ["NEO4J_URI"] = "bolt://neo4j.local:7687"
environ["NEO4J_PASSWORD"] = "test-password"
environ["LITELLM_BASE_URL"] = "http://litellm.local/v1"
environ["LITELLM_API_KEY"] = "test-litellm-key"

from agents_memory.backend import Neo4jAgentMemoryBackend, litellm_embedding_model
from agents_memory.graph_probe import DirectNeo4jProbe, GraphPersistenceUnavailableError
from agents_memory.graph_store import DirectNeo4jGraphStore, InMemoryGraphExecutor
from agents_memory.models import (
    BackendReadiness,
    ClientEvent,
    ClientEventActor,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ClientEventSubject,
    ClientEventType,
    ContextRequest,
    ContextResponse,
    DiagnosticsConfig,
    DiagnosticsResponse,
    DiscordEventContext,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    JsonObject,
    JsonValue,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSection,
    MemoryScope,
    MemoryVisibility,
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
    SourceClient,
)
from agents_memory.settings import Settings

if TYPE_CHECKING:
    from agents_memory.backend import MemoryBackend, MemoryClientContext


def test_litellm_embedding_model_when_embedding_alias_is_bare() -> None:
    # Given: homelab config uses a bare LiteLLM proxy alias for memory embeddings.
    model = "local-qwen3-embedding-0.6b"

    # When: the model is prepared for the LiteLLM SDK provider.
    sdk_model = litellm_embedding_model(model)

    # Then: only the SDK-facing model is provider-qualified.
    assert model == "local-qwen3-embedding-0.6b"
    assert sdk_model == "openai/local-qwen3-embedding-0.6b"


def test_litellm_embedding_model_when_embedding_alias_is_qualified() -> None:
    # Given: a caller already supplied a LiteLLM provider-qualified embedding model.
    model = "openai/local-qwen3-embedding-0.6b"

    # When: the model is prepared for the LiteLLM SDK provider.
    sdk_model = litellm_embedding_model(model)

    # Then: the configured provider prefix is preserved without double-prefixing.
    assert sdk_model == "openai/local-qwen3-embedding-0.6b"


@pytest.mark.anyio
async def test_backend_protocol_ingests_event_batch() -> None:
    # Given: a fake backend implementing the full persistence protocol.
    backend: MemoryBackend = RecordingBackend()
    event = _client_event()

    # When: a typed batch is ingested through the protocol seam.
    response = await backend.ingest_events(ClientEventBatchRequest(events=[event]))

    # Then: fake implementations can satisfy the seam without Neo4j.
    assert response == ClientEventBatchResponse(
        results=[
            EventIngestResult(
                event_id="discord-message-999",
                status=EventIngestStatus.ACCEPTED,
            ),
        ],
    )


@pytest.mark.anyio
async def test_direct_neo4j_probe_failure_degrades_to_clear_error() -> None:
    # Given: direct graph persistence is unavailable at startup.
    probe = DirectNeo4jProbe(driver_factory=FailingDriverFactory())

    # When / Then: the seam fails explicitly instead of silently no-oping.
    with pytest.raises(GraphPersistenceUnavailableError) as error:
        await probe.require_available()

    assert "Neo4j structured graph persistence is unavailable" in str(error.value)


@pytest.mark.anyio
async def test_backend_promotes_accepted_event_to_embedded_long_term_fact() -> None:
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=DirectNeo4jGraphStore(executor=InMemoryGraphExecutor()),
    )
    event = _client_event()

    result = await backend.ingest_event(event)

    assert result.status == EventIngestStatus.ACCEPTED
    assert fake_client.long_term.facts == [
        LongTermFactWrite(
            subject="tenant:bromigos:message:message-999",
            predicate="discord.message_created",
            obj="message message-999: remember this",
            metadata={
                "agent_id": "pc-principal",
                "channel_id": "456",
                "event_id": "discord-message-999",
                "event_type": "message_created",
                "guild_id": "123",
                "idempotency_key": "discord:message:message-999:create",
                "session_id": "guild:123:channel:456",
                "tenant_id": "bromigos",
                "user_id": "789",
                "visibility": "channel",
            },
            generate_embedding=True,
        ),
    ]


@pytest.mark.anyio
async def test_backend_repairs_duplicate_event_graph_without_promoting_fact() -> None:
    fake_client = RecordingMemoryClient()
    executor = InMemoryGraphExecutor()
    store = DirectNeo4jGraphStore(executor=executor)
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=store,
    )
    event = _client_event()
    _ = await backend.ingest_event(event)
    executor.clear_current_nodes_for_test()

    duplicate = await backend.ingest_event(event)

    assert duplicate.status == EventIngestStatus.DUPLICATE
    assert len(fake_client.long_term.facts) == 1
    assert executor.semantic_node_ids_for_test() == {
        "tenant:bromigos:agent:pc-principal",
        "tenant:bromigos:channel:456",
        "tenant:bromigos:client:discord",
        "tenant:bromigos:guild:123",
        "tenant:bromigos:message:message-999",
        "tenant:bromigos:tenant:bromigos",
        "tenant:bromigos:user:789",
    }


@pytest.mark.anyio
async def test_backend_retries_fact_promotion_after_initial_failure() -> None:
    # Given: graph persistence accepts an event before long-term fact promotion fails.
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(failed_writes_remaining=1),
    )
    executor = InMemoryGraphExecutor()
    store = DirectNeo4jGraphStore(executor=executor)
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=store,
    )
    event = _client_event()

    # When: the caller retries the same event after the promotion failure.
    with pytest.raises(PromotionFailureError):
        _ = await backend.ingest_event(event)
    retry = await backend.ingest_event(event)

    # Then: the graph stays idempotent while the missing fact is promoted once.
    assert retry.status == EventIngestStatus.DUPLICATE
    assert executor.event_count == 1
    assert len(fake_client.long_term.facts) == 1
    assert fake_client.long_term.facts[0].generate_embedding is True


@pytest.mark.anyio
async def test_memory_context_combines_labeled_sections_in_order() -> None:
    # Given: a backend with fake short-term, long-term, reasoning, and graph stores.
    fake_client = RecordingMemoryClient(
        short_term=RecordingShortTermMemory(context="recent chat"),
        long_term=RecordingLongTermMemory(context="### User Preferences\n- concise"),
        reasoning=RecordingReasoningMemory(context="### Similar Past Tasks\n- replied"),
    )
    graph_store = RecordingGraphStore(context="graph summary")
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=graph_store,
    )

    # When: combined memory context is requested with graph enabled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=_scope(),
            query="what matters?",
            max_items=4,
            graph_limit=3,
        ),
    )

    # Then: sections are labeled in the required order and scoped graph is used.
    assert response == MemoryContextResponse(
        sections=[
            MemoryContextSection(source="short_term", content="recent chat"),
            MemoryContextSection(
                source="long_term_preferences_entities",
                content="### User Preferences\n- concise",
            ),
            MemoryContextSection(
                source="reasoning",
                content="### Similar Past Tasks\n- replied",
            ),
            MemoryContextSection(
                source="graph",
                content="graph summary",
                facts=[{"kind": "graph"}],
            ),
        ],
    )
    assert fake_client.short_term.context_queries == ["what matters?"]
    assert fake_client.long_term.context_queries == ["what matters?"]
    assert fake_client.reasoning.context_queries == ["what matters?"]
    assert graph_store.context_requests == [
        GraphContextRequest(scope=_scope(), query="what matters?", limit=3),
    ]


@pytest.mark.anyio
async def test_memory_context_omits_empty_sections_and_disabled_graph() -> None:
    # Given: only long-term context has content and graph recall is disabled.
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(context="### Relevant Entities\n- Cartman"),
    )
    graph_store = RecordingGraphStore(context="graph summary")
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=graph_store,
    )

    # When: combined memory context is requested without graph context.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=_scope(),
            query="what matters?",
            include_graph=False,
        ),
    )

    # Then: empty sections are omitted and graph storage is not queried.
    assert response == MemoryContextResponse(
        sections=[
            MemoryContextSection(
                source="long_term_preferences_entities",
                content="### Relevant Entities\n- Cartman",
            ),
        ],
    )
    assert graph_store.context_requests == []


@dataclass(slots=True)
class RecordingBackend:
    events: list[ClientEvent] = field(default_factory=list)

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
        self.events.append(event)
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def ingest_events(
        self,
        request: ClientEventBatchRequest,
    ) -> ClientEventBatchResponse:
        results = [await self.ingest_event(event) for event in request.events]
        return ClientEventBatchResponse(results=results)

    async def get_graph_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse:
        _ = request
        return GraphContextResponse(context="fake graph context")

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

    async def start_reasoning_trace(
        self,
        request: ReasoningTraceStartRequest,
    ) -> ReasoningTraceStartResponse:
        _ = request
        return ReasoningTraceStartResponse(
            trace_id="trace-placeholder",
            session_id="session-placeholder",
            task="task-placeholder",
        )

    async def add_reasoning_step(
        self,
        request: ReasoningStepRequest,
    ) -> ReasoningStepResponse:
        _ = request
        return ReasoningStepResponse(
            step_id="step-placeholder",
            trace_id="trace-placeholder",
            step_number=1,
        )

    async def record_reasoning_tool_call(
        self,
        request: ReasoningToolCallRequest,
    ) -> ReasoningToolCallResponse:
        _ = request
        return ReasoningToolCallResponse(
            tool_call_id="tool-call-placeholder",
            trace_id="trace-placeholder",
            step_id="step-placeholder",
        )

    async def complete_reasoning_trace(
        self,
        request: ReasoningTraceCompleteRequest,
    ) -> ReasoningTraceCompleteResponse:
        _ = request
        return ReasoningTraceCompleteResponse(trace_id="trace-placeholder")

    async def get_reasoning_context(
        self,
        request: ReasoningContextRequest,
    ) -> ReasoningContextResponse:
        _ = request
        return ReasoningContextResponse(context="reasoning context")

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


@dataclass(frozen=True, slots=True)
class LongTermFactWrite:
    subject: str
    predicate: str
    obj: str
    metadata: dict[str, str]
    generate_embedding: bool


@dataclass(slots=True)
class RecordingLongTermMemory:
    facts: list[LongTermFactWrite] = field(default_factory=list)
    failed_writes_remaining: int = 0
    context: str = ""
    context_queries: list[str] = field(default_factory=list)

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        metadata: dict[str, str],
        generate_embedding: bool,
    ) -> None:
        if self.failed_writes_remaining > 0:
            self.failed_writes_remaining -= 1
            raise PromotionFailureError
        self.facts.append(
            LongTermFactWrite(
                subject=subject,
                predicate=predicate,
                obj=obj,
                metadata=metadata,
                generate_embedding=generate_embedding,
            ),
        )

    async def get_context(self, query: str, *, max_items: int) -> str:
        _ = max_items
        self.context_queries.append(query)
        return self.context


class PromotionFailureError(Exception):
    pass


@dataclass(slots=True)
class RecordingShortTermMemory:
    context: str = ""
    context_queries: list[str] = field(default_factory=list)

    async def add_message(  # noqa: PLR0913
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        user_identifier: str,
        metadata: dict[str, str],
        extract_entities: bool,
        extract_relations: bool,
    ) -> None:
        _ = (
            session_id,
            role,
            content,
            user_identifier,
            metadata,
            extract_entities,
            extract_relations,
        )

    async def get_context(
        self,
        query: str,
        *,
        session_id: str,
        max_messages: int,
        metadata_filters: dict[str, str],
    ) -> str:
        _ = (session_id, max_messages, metadata_filters)
        self.context_queries.append(query)
        return self.context


@dataclass(slots=True)
class RecordingReasoningMemory:
    context: str = ""
    context_queries: list[str] = field(default_factory=list)

    async def get_context(self, query: str, *, max_traces: int) -> str:
        _ = max_traces
        self.context_queries.append(query)
        return self.context

    async def start_trace(  # noqa: PLR0913
        self,
        session_id: str,
        task: str,
        *,
        generate_embedding: bool,
        metadata: JsonObject | None,
        triggered_by_message_id: str | None,
        user_identifier: str,
    ) -> SdkReasoningTrace:
        _ = (generate_embedding, metadata, triggered_by_message_id, user_identifier)
        return SdkReasoningTrace(session_id=session_id, task=task)

    async def add_step(  # noqa: PLR0913
        self,
        trace_id: UUID,
        *,
        thought: None,
        action: str | None,
        observation: str | None,
        generate_embedding: bool,
        metadata: JsonObject | None,
    ) -> SdkReasoningStep:
        _ = (thought, action, observation, generate_embedding, metadata)
        return SdkReasoningStep(trace_id=trace_id, step_number=1)

    async def record_tool_call(  # noqa: PLR0913
        self,
        step_id: UUID,
        tool_name: str,
        arguments: JsonObject,
        *,
        result: JsonValue | None,
        status: ToolCallStatus,
        duration_ms: int | None,
        error: str | None,
        message_id: str | None,
        touched_entities: list[EntityRef],
    ) -> ToolCall:
        _ = (result, message_id, touched_entities)
        return ToolCall(
            step_id=step_id,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            duration_ms=duration_ms,
            error=error,
        )

    async def complete_trace(
        self,
        trace_id: UUID,
        *,
        outcome: str | None,
        success: bool | None,
        generate_step_embeddings: bool,
    ) -> SdkReasoningTrace:
        _ = generate_step_embeddings
        return SdkReasoningTrace(
            id=trace_id,
            session_id="session-placeholder",
            task="task-placeholder",
            outcome=outcome,
            success=success,
        )


@dataclass(slots=True)
class RecordingCypherQuery:
    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        _ = (query, params)
        return []


@dataclass(slots=True)
class RecordingMemoryClient:
    short_term: RecordingShortTermMemory = field(
        default_factory=RecordingShortTermMemory,
    )
    long_term: RecordingLongTermMemory = field(default_factory=RecordingLongTermMemory)
    reasoning: RecordingReasoningMemory = field(
        default_factory=RecordingReasoningMemory,
    )
    query: RecordingCypherQuery = field(
        default_factory=RecordingCypherQuery,
    )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


@dataclass(frozen=True, slots=True)
class MemoryClientFactory:
    client: RecordingMemoryClient

    def __call__(self, settings: MemorySettings) -> "MemoryClientContext":
        _ = settings
        return self.client


@dataclass(slots=True)
class RecordingGraphStore:
    context: str = ""
    context_requests: list[GraphContextRequest] = field(default_factory=list)

    async def require_available(self) -> None:
        return None

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        self.context_requests.append(request)
        return GraphContextResponse(context=self.context, facts=[{"kind": "graph"}])


@dataclass(frozen=True, slots=True)
class FailingDriverFactory:
    def __call__(self) -> "FailingDriver":
        return FailingDriver()


@dataclass(frozen=True, slots=True)
class FailingDriver:
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)

    async def verify_connectivity(self) -> None:
        reason = "connection refused"
        raise OSError(reason)


def _scope() -> MemoryScope:
    return MemoryScope(
        tenant_id="bromigos",
        space_id="discord",
        agent_id="pc-principal",
        session_id="guild:123:channel:456",
        user_id="789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="123",
        channel_id="456",
    )


def _client_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id="discord-message-999",
        event_type=ClientEventType.MESSAGE_CREATED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="discord:message:message-999:create",
        scope=_scope(),
        actor=ClientEventActor(id="789", display_name="cartman", is_bot=False),
        subject=ClientEventSubject(id="message-999", type="message"),
        payload={"content": "remember this", "payload_version": 1},
        discord=DiscordEventContext(
            guild_id="123",
            channel_id="456",
            message_id="message-999",
        ),
    )
