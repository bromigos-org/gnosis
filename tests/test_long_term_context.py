import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self
from uuid import UUID

import pytest
from neo4j_agent_memory import MemorySettings
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolCallStatus
from neo4j_agent_memory.schema.models import EntityRef

from agents_memory.backend import Neo4jAgentMemoryBackend
from agents_memory.models import (
    BackendReadiness,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    JsonObject,
    JsonValue,
    MemoryContextRequest,
    MemoryContextSection,
    MemoryScope,
    MemoryVisibility,
)
from agents_memory.settings import Settings

if TYPE_CHECKING:
    from agents_memory.backend import MemoryClientContext


@pytest.mark.anyio
async def test_combined_context_includes_scoped_facts_preferences_entities() -> None:
    # Given: one locally stored fact plus separate upstream long-term context.
    fact = _fact_row(
        subject="tenant:bromigos:message:one",
        predicate="discord.message_created",
        object_value="message one mentions the library schedule",
        metadata=_scope_metadata(_scope()) | {"event_id": "event-one"},
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(rows=[{"f": fact}]),
        long_term=RecordingLongTermMemory(
            context=(
                "### User Preferences\n- concise updates\n\n"
                "### Relevant Entities\n- library"
            ),
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: combined memory context is requested for the matching scope.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=_scope(),
            query="what should I remember?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
            max_items=5,
        ),
    )

    # Then: facts are formatted once and preferences/entities remain separate.
    assert response.sections == [
        MemoryContextSection(
            source="long_term_facts",
            content=(
                "### Long-Term Facts\n"
                "- subject: tenant:bromigos:message:one\n"
                "  predicate: discord.message_created\n"
                "  object: message one mentions the library schedule\n"
                "  provenance: agent_id=pc-principal, channel_id=456, "
                "event_id=event-one, guild_id=123, session_id=guild:123:channel:456, "
                "tenant_id=bromigos, user_id=789, visibility=channel"
            ),
        ),
        MemoryContextSection(
            source="long_term_preferences_entities",
            content=(
                "### User Preferences\n- concise updates\n\n"
                "### Relevant Entities\n- library"
            ),
        ),
    ]
    assert response.sections[0].content.count("tenant:bromigos:message:one") == 1
    assert "tenant:bromigos:message:one" not in response.sections[1].content
    assert client.long_term.context_queries == ["what should I remember?"]
    assert client.query.cypher_calls[0].parameters == {
        "limit": 5,
        "metadata_fragments": [
            '"tenant_id": "bromigos"',
            '"agent_id": "pc-principal"',
            '"session_id": "guild:123:channel:456"',
            '"user_id": "789"',
            '"visibility": "channel"',
            '"guild_id": "123"',
            '"channel_id": "456"',
        ],
    }


@pytest.mark.anyio
async def test_fact_context_does_not_cross_tenant_or_channel_scope() -> None:
    # Given: facts for the requested channel plus facts from other scopes.
    requested_scope = _scope()
    other_tenant_scope = _scope(tenant_id="other-tenant")
    other_channel_scope = _scope(channel_id="999")
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:bromigos:message:visible",
                        predicate="discord.message_created",
                        object_value="visible channel note",
                        metadata=_scope_metadata(requested_scope),
                    ),
                },
                {
                    "f": _fact_row(
                        subject="tenant:other:message:hidden",
                        predicate="discord.message_created",
                        object_value="other tenant note",
                        metadata=_scope_metadata(other_tenant_scope),
                    ),
                },
                {
                    "f": _fact_row(
                        subject="tenant:bromigos:message:wrong-channel",
                        predicate="discord.message_created",
                        object_value="wrong channel note",
                        metadata=_scope_metadata(other_channel_scope),
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: combined memory context is requested for the original channel.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=requested_scope,
            query="channel recall",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: only the same-tenant same-channel fact appears.
    assert len(response.sections) == 1
    content = response.sections[0].content
    assert "tenant:bromigos:message:visible" in content
    assert "tenant:other:message:hidden" not in content
    assert "tenant:bromigos:message:wrong-channel" not in content
    assert "other tenant note" not in content
    assert "wrong channel note" not in content


@dataclass(frozen=True, slots=True)
class CypherCall:
    statement: str
    parameters: dict[str, JsonValue]


@dataclass(slots=True)
class RecordingQuery:
    rows: list[JsonObject] = field(default_factory=list)
    cypher_calls: list[CypherCall] = field(default_factory=list)

    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.cypher_calls.append(
            CypherCall(statement=query, parameters=params or {}),
        )
        return self.rows


@dataclass(slots=True)
class RecordingLongTermMemory:
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
        _ = (subject, predicate, obj, metadata, generate_embedding)

    async def get_context(self, query: str, *, max_items: int) -> str:
        _ = max_items
        self.context_queries.append(query)
        return self.context


@dataclass(slots=True)
class RecordingShortTermMemory:
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
        _ = (query, session_id, max_messages, metadata_filters)
        return ""


@dataclass(slots=True)
class RecordingReasoningMemory:
    async def get_context(self, query: str, *, max_traces: int) -> str:
        _ = (query, max_traces)
        return ""

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
class RecordingMemoryClient:
    query: RecordingQuery
    long_term: RecordingLongTermMemory = field(default_factory=RecordingLongTermMemory)
    short_term: RecordingShortTermMemory = field(
        default_factory=RecordingShortTermMemory,
    )
    reasoning: RecordingReasoningMemory = field(
        default_factory=RecordingReasoningMemory,
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
    async def require_available(self) -> None:
        return None

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def ingest_event(self, event: object) -> EventIngestResult:
        _ = event
        return EventIngestResult(
            event_id="event-placeholder",
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        _ = request
        return GraphContextResponse(context="")


def _settings() -> Settings:
    settings_values: JsonObject = {
        "agents_memory_token": "value",
        "agents_memory_tenant_id": "bromigos",
        "neo4j_uri": "bolt://neo4j.local:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "value",
        "litellm_base_url": "http://litellm.local/v1",
        "litellm_api_key": "value",
        "memory_llm": "openai/gemma4",
        "memory_embedding": "local-qwen3-embedding-0.6b",
        "memory_embedding_dimensions": 1024,
    }
    return Settings.model_validate(settings_values)


def _scope(
    *,
    tenant_id: str = "bromigos",
    channel_id: str = "456",
) -> MemoryScope:
    return MemoryScope(
        tenant_id=tenant_id,
        space_id="discord",
        agent_id="pc-principal",
        session_id=f"guild:123:channel:{channel_id}",
        user_id="789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="123",
        channel_id=channel_id,
    )


def _scope_metadata(scope: MemoryScope) -> dict[str, str]:
    return {
        "tenant_id": scope.tenant_id,
        "agent_id": scope.agent_id,
        "session_id": scope.session_id,
        "user_id": scope.user_id,
        "visibility": scope.visibility.value,
        "guild_id": scope.guild_id or "",
        "channel_id": scope.channel_id or "",
    }


def _fact_row(
    *,
    subject: str,
    predicate: str,
    object_value: str,
    metadata: dict[str, str],
) -> JsonObject:
    return {
        "id": subject,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "confidence": 1.0,
        "created_at": None,
        "metadata": json.dumps(metadata),
    }
