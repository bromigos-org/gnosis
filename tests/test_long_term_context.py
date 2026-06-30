import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Self, cast
from uuid import UUID

import pytest
from neo4j_agent_memory import MemorySettings
from neo4j_agent_memory.memory.long_term import EntityType
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolCallStatus, ToolStats
from neo4j_agent_memory.schema.models import EntityRef

from agents_memory.backend import Neo4jAgentMemoryBackend
from agents_memory.models import (
    BackendReadiness,
    EntityRecord,
    EventIngestResult,
    EventIngestStatus,
    FactRecord,
    GraphContextRequest,
    GraphContextResponse,
    JsonObject,
    JsonValue,
    MemoryContextRequest,
    MemoryContextSection,
    MemoryScope,
    MemoryVisibility,
    PreferenceRecord,
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
        _settings(memory_prompt_entities_enabled=True),
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

    async def search_entities(
        self,
        query: str,
        *,
        entity_types: list[EntityType | str] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[EntityRecord]:
        _ = (query, entity_types, limit, threshold)
        return []

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[FactRecord]:
        _ = (query, limit, threshold)
        return []

    async def search_preferences(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[PreferenceRecord]:
        _ = (query, category, limit, threshold)
        return []

    async def add_entity(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        name: str,
        entity_type: EntityType | str,
        *,
        subtype: str | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
        attributes: JsonObject | None = None,
        resolve: bool = True,
        generate_embedding: bool = True,
        deduplicate: bool = True,
        geocode: bool = True,
        enrich: bool = True,
        coordinates: tuple[float, float] | None = None,
        metadata: JsonObject | None = None,
    ) -> EntityRecord:
        _ = (
            name,
            entity_type,
            subtype,
            description,
            aliases,
            attributes,
            resolve,
            generate_embedding,
            deduplicate,
            geocode,
            enrich,
            coordinates,
            metadata,
        )
        return EntityRecord(name=name, type=str(entity_type))

    async def add_fact(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
    ) -> FactRecord:
        _ = (
            subject,
            predicate,
            obj,
            confidence,
            valid_from,
            valid_until,
            metadata,
            generate_embedding,
        )
        return FactRecord(subject=subject, predicate=predicate, object=obj)

    async def add_preference(  # noqa: PLR0913 - Mirrors SDK API.
        self,
        category: str,
        preference: str,
        *,
        context: str | None = None,
        confidence: float = 1.0,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
        user_identifier: str | None = None,
        applies_to: object | None = None,
    ) -> PreferenceRecord:
        _ = (
            category,
            preference,
            context,
            confidence,
            generate_embedding,
            metadata,
            user_identifier,
            applies_to,
        )
        return PreferenceRecord(category=category, preference=preference)

    async def get_preferences_for(
        self,
        user_identifier: str,
        *,
        applies_to: object | None = None,
        active_only: bool = True,
        as_of: datetime | None = None,
    ) -> list[PreferenceRecord]:
        _ = (user_identifier, applies_to, active_only, as_of)
        return []

    async def get_facts_about(
        self,
        subject: str,
        *,
        limit: int = 100,
    ) -> list[FactRecord]:
        _ = (subject, limit)
        return []

    async def link_entity_to_message(  # noqa: PLR0913 - Mirrors SDK API.
        self,
        entity: EntityRecord | UUID,
        message_id: UUID | str,
        *,
        confidence: float = 1.0,
        start_pos: int | None = None,
        end_pos: int | None = None,
        context: str | None = None,
    ) -> bool:
        _ = (entity, message_id, confidence, start_pos, end_pos, context)
        return True

    async def link_entity_to_extractor(
        self,
        entity: EntityRecord | UUID,
        extractor_name: str,
        *,
        confidence: float = 1.0,
        extraction_time_ms: float | None = None,
    ) -> bool:
        _ = (entity, extractor_name, confidence, extraction_time_ms)
        return True

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

    async def list_traces(
        self,
        *,
        session_id: str | None = None,
        limit: int = 10,
        success_only: bool | None = None,
    ) -> list[SdkReasoningTrace]:
        _ = (session_id, limit, success_only)
        return []

    async def get_trace(self, trace_id: UUID | str) -> SdkReasoningTrace | None:
        _ = trace_id
        return None

    async def get_trace_with_steps(
        self,
        trace_id: UUID | str,
    ) -> SdkReasoningTrace | None:
        _ = trace_id
        return None

    async def get_similar_traces(
        self,
        task: str,
        *,
        limit: int = 5,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[SdkReasoningTrace]:
        _ = (task, limit, success_only, threshold)
        return []

    async def search_steps(
        self,
        query: str,
        *,
        limit: int = 10,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[object]:
        _ = (query, limit, success_only, threshold)
        return []

    async def get_tool_stats(self, tool_name: str | None = None) -> list[ToolStats]:
        _ = tool_name
        return []

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
        client = cast("object", self.client)
        return cast("MemoryClientContext", client)


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


def _settings(*, memory_prompt_entities_enabled: bool = False) -> Settings:
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
        "memory_prompt_entities_enabled": memory_prompt_entities_enabled,
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
