import json
from dataclasses import dataclass, field
from typing import Final, Protocol, Self
from uuid import UUID

from neo4j_agent_memory import MemoryClient, MemoryConfig, MemorySettings, Neo4jConfig
from neo4j_agent_memory.llm.adapters.litellm import (
    LiteLLMEmbeddingProvider,
    LiteLLMProvider,
)
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolCallStatus
from neo4j_agent_memory.schema.models import EntityRef
from pydantic import SecretStr, TypeAdapter, ValidationError

from agents_memory.event_facts import EventFactPromoter
from agents_memory.graph_probe import StructuredGraphStore, direct_neo4j_driver_factory
from agents_memory.graph_store import DirectNeo4jGraphStore, Neo4jGraphExecutor
from agents_memory.models import (
    BackendReadiness,
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ContextRequest,
    ContextResponse,
    DiagnosticsConfig,
    DiagnosticsResponse,
    EventIngestResult,
    GraphContextRequest,
    GraphContextResponse,
    JsonObject,
    JsonValue,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSection,
    MemoryScope,
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
from agents_memory.redaction import redact_secrets
from agents_memory.settings import Settings
from agents_memory.skill_registry import InMemorySkillRegistry, SkillRegistry

_JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)


class MemoryBackend(Protocol):
    async def readiness(self) -> BackendReadiness: ...
    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse: ...
    async def add_message(
        self,
        request: MessageWriteRequest,
    ) -> MessageWriteResponse: ...
    async def get_context(self, request: ContextRequest) -> ContextResponse: ...
    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse: ...
    async def ingest_event(self, event: ClientEvent) -> EventIngestResult: ...
    async def ingest_events(
        self,
        request: ClientEventBatchRequest,
    ) -> ClientEventBatchResponse: ...
    async def get_graph_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse: ...
    async def list_skills(self, request: SkillListRequest) -> SkillListResponse: ...
    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal: ...
    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult: ...
    async def start_reasoning_trace(
        self,
        request: ReasoningTraceStartRequest,
    ) -> ReasoningTraceStartResponse: ...
    async def add_reasoning_step(
        self,
        request: ReasoningStepRequest,
    ) -> ReasoningStepResponse: ...
    async def record_reasoning_tool_call(
        self,
        request: ReasoningToolCallRequest,
    ) -> ReasoningToolCallResponse: ...
    async def complete_reasoning_trace(
        self,
        request: ReasoningTraceCompleteRequest,
    ) -> ReasoningTraceCompleteResponse: ...
    async def get_reasoning_context(
        self,
        request: ReasoningContextRequest,
    ) -> ReasoningContextResponse: ...


class MemoryClientFactory(Protocol):
    def __call__(self, settings: MemorySettings) -> "MemoryClientContext": ...


class ShortTermMemory(Protocol):
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
    ) -> object: ...

    async def get_context(
        self,
        query: str,
        *,
        session_id: str,
        max_messages: int,
        metadata_filters: dict[str, str],
    ) -> str: ...


class LongTermMemory(Protocol):
    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        metadata: dict[str, str],
        generate_embedding: bool,
    ) -> object: ...

    async def get_context(self, query: str, *, max_items: int) -> str: ...


class ReasoningMemory(Protocol):
    async def get_context(self, query: str, *, max_traces: int) -> str: ...
    async def start_trace(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        session_id: str,
        task: str,
        *,
        generate_embedding: bool,
        metadata: JsonObject | None,
        triggered_by_message_id: str | None,
        user_identifier: str,
    ) -> SdkReasoningTrace: ...
    async def add_step(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        trace_id: UUID,
        *,
        thought: None,
        action: str | None,
        observation: str | None,
        generate_embedding: bool,
        metadata: JsonObject | None,
    ) -> SdkReasoningStep: ...
    async def record_tool_call(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
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
    ) -> ToolCall: ...
    async def complete_trace(
        self,
        trace_id: UUID,
        *,
        outcome: str | None,
        success: bool | None,
        generate_step_embeddings: bool,
    ) -> SdkReasoningTrace: ...


class CypherQuery(Protocol):
    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]: ...


class MemoryClientContext(Protocol):
    @property
    def short_term(self) -> ShortTermMemory: ...
    @property
    def long_term(self) -> LongTermMemory: ...
    @property
    def reasoning(self) -> ReasoningMemory: ...
    @property
    def query(self) -> CypherQuery: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class LongTermFactsContext:
    context: str = ""
    markers: set[str] = field(default_factory=set)


class Neo4jAgentMemoryBackend:
    def __init__(
        self,
        settings: Settings,
        memory_client_factory: MemoryClientFactory | None = None,
        graph_store: StructuredGraphStore | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self._app_settings: Settings = settings
        self._settings: MemorySettings = _build_memory_settings(settings)
        self._memory_client_factory: MemoryClientFactory | None = memory_client_factory
        embedding_provider = LiteLLMEmbeddingProvider(
            litellm_embedding_model(settings.memory_embedding),
            dimensions=settings.memory_embedding_dimensions,
            api_base=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        self._graph_store: StructuredGraphStore = graph_store or DirectNeo4jGraphStore(
            executor=Neo4jGraphExecutor(
                driver_factory=direct_neo4j_driver_factory(settings),
                embedding_dimensions=settings.memory_embedding_dimensions,
                embedding_provider=embedding_provider,
            ),
        )
        self._skill_registry: SkillRegistry = skill_registry or InMemorySkillRegistry()
        self._event_fact_promoter: EventFactPromoter = EventFactPromoter()

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        metadata = _scope_metadata(request.scope)
        async with self._memory_client() as client:
            _ = await client.short_term.add_message(
                session_id=_session_id(request.scope),
                role=request.role.value,
                content=request.content,
                user_identifier=_user_identifier(request.scope),
                metadata=metadata,
                extract_entities=False,
                extract_relations=False,
            )
            _ = await client.long_term.add_fact(
                subject=_user_identifier(request.scope),
                predicate=f"said_{request.role.value}",
                obj=request.content,
                metadata=metadata,
                generate_embedding=True,
            )
        return MessageWriteResponse(accepted=True)

    async def readiness(self) -> BackendReadiness:
        return await self._graph_store.readiness()

    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse:
        return DiagnosticsResponse(
            tenant_id=self._app_settings.agents_memory_tenant_id,
            config=DiagnosticsConfig(
                neo4j_uri=self._app_settings.neo4j_uri,
                neo4j_username=self._app_settings.neo4j_username,
                litellm_base_url=self._app_settings.litellm_base_url,
                memory_llm=self._app_settings.memory_llm,
                memory_embedding=self._app_settings.memory_embedding,
                memory_embedding_dimensions=(
                    self._app_settings.memory_embedding_dimensions
                ),
            ),
            backend=readiness,
        )

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        async with self._memory_client() as client:
            context = await client.short_term.get_context(
                request.query,
                session_id=_session_id(request.scope),
                max_messages=request.limit,
                metadata_filters=_scope_metadata(request.scope),
            )
        return ContextResponse(context=context)

    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        sections: list[MemoryContextSection] = []
        long_term_facts = LongTermFactsContext()
        async with self._memory_client() as client:
            if request.include_short_term:
                short_term = await client.short_term.get_context(
                    request.query,
                    session_id=_session_id(request.scope),
                    max_messages=request.max_items,
                    metadata_filters=_scope_metadata(request.scope),
                )
                _append_context_section(sections, "short_term", short_term)

            if request.include_long_term:
                long_term_facts = await self._get_long_term_facts_context(
                    request,
                    client,
                )
                _append_context_section(
                    sections,
                    "long_term_facts",
                    long_term_facts.context,
                )

                long_term = await client.long_term.get_context(
                    request.query,
                    max_items=request.max_items,
                )
                _append_context_section(
                    sections,
                    "long_term_preferences_entities",
                    long_term,
                )

            if request.include_reasoning:
                reasoning = await client.reasoning.get_context(
                    request.query,
                    max_traces=request.max_items,
                )
                _append_context_section(sections, "reasoning", reasoning)

        if request.include_graph:
            graph = await self.get_graph_context(
                GraphContextRequest(
                    scope=request.scope,
                    query=request.query,
                    limit=request.graph_limit,
                ),
            )
            graph = _dedupe_graph_context(graph, long_term_facts.markers)
            if graph.context:
                sections.append(
                    MemoryContextSection(
                        source="graph",
                        content=graph.context,
                        facts=graph.facts,
                    ),
                )

        return MemoryContextResponse(sections=sections)

    async def _get_long_term_facts_context(
        self,
        request: MemoryContextRequest,
        client: MemoryClientContext,
    ) -> "LongTermFactsContext":
        metadata = _scope_metadata(request.scope)
        params: JsonObject = {
            "metadata_fragments": _metadata_fragments(metadata),
            "limit": request.max_items,
        }
        rows = await client.query.cypher(
            """
            MATCH (f:Fact)
            WHERE f.metadata IS NOT NULL
              AND all(
                fragment IN $metadata_fragments WHERE f.metadata CONTAINS fragment
              )
            RETURN f
            ORDER BY f.created_at DESC, f.subject ASC, f.predicate ASC, f.object ASC
            LIMIT $limit
            """,
            params,
        )
        facts = [
            fact
            for row in rows
            if (fact := _fact_from_row(row)) is not None
            and _fact_matches_scope(fact, metadata)
        ]
        if not facts:
            return LongTermFactsContext()
        lines = ["### Long-Term Facts"]
        for fact in facts:
            metadata_fields = _fact_metadata(fact)
            lines.extend(
                [
                    f"- subject: {fact['subject']}",
                    f"  predicate: {fact['predicate']}",
                    f"  object: {fact['object']}",
                    f"  provenance: {_format_provenance(metadata_fields)}",
                ],
            )
        return LongTermFactsContext(
            context="\n".join(lines),
            markers=_fact_markers(facts),
        )

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        result = await self._graph_store.ingest_event(event)
        await self._event_fact_promoter.promote_for_result(
            event,
            result,
            self._memory_client(),
        )
        return result

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
        return await self._graph_store.get_context(request)

    async def list_skills(self, request: SkillListRequest) -> SkillListResponse:
        await self._graph_store.require_available()
        return await self._skill_registry.list_skills(request)

    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal:
        await self._graph_store.require_available()
        return await self._skill_registry.propose_skill(proposal)

    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult:
        await self._graph_store.require_available()
        return await self._skill_registry.record_skill_usage(usage)

    async def start_reasoning_trace(
        self,
        request: ReasoningTraceStartRequest,
    ) -> ReasoningTraceStartResponse:
        async with self._memory_client() as client:
            trace = await client.reasoning.start_trace(
                request.session_id,
                _redacted_text(request.task),
                generate_embedding=True,
                metadata=_redacted_object(request.metadata),
                triggered_by_message_id=request.triggered_by_message_id,
                user_identifier=(
                    request.user_identifier or _user_identifier(request.scope)
                ),
            )
        return ReasoningTraceStartResponse(
            trace_id=str(trace.id),
            session_id=trace.session_id,
            task=trace.task,
        )

    async def add_reasoning_step(
        self,
        request: ReasoningStepRequest,
    ) -> ReasoningStepResponse:
        async with self._memory_client() as client:
            step = await client.reasoning.add_step(
                UUID(request.trace_id),
                thought=None,
                action=_redacted_optional_text(request.action),
                observation=_redacted_optional_text(request.observation),
                generate_embedding=True,
                metadata=_redacted_object(request.metadata),
            )
        return ReasoningStepResponse(
            step_id=str(step.id),
            trace_id=str(step.trace_id),
            step_number=step.step_number,
        )

    async def record_reasoning_tool_call(
        self,
        request: ReasoningToolCallRequest,
    ) -> ReasoningToolCallResponse:
        async with self._memory_client() as client:
            tool_call = await client.reasoning.record_tool_call(
                UUID(request.step_id),
                request.tool_name,
                _redacted_object(request.arguments),
                result=redact_secrets(request.result),
                status=ToolCallStatus(request.status),
                duration_ms=request.duration_ms,
                error=_redacted_optional_text(request.error),
                message_id=request.message_id,
                touched_entities=[
                    EntityRef(id=entity.id, name=entity.name, type=entity.type)
                    for entity in request.touched_entities
                ],
            )
        return ReasoningToolCallResponse(
            tool_call_id=str(tool_call.id),
            trace_id=request.trace_id,
            step_id=request.step_id,
        )

    async def complete_reasoning_trace(
        self,
        request: ReasoningTraceCompleteRequest,
    ) -> ReasoningTraceCompleteResponse:
        async with self._memory_client() as client:
            trace = await client.reasoning.complete_trace(
                UUID(request.trace_id),
                outcome=_redacted_optional_text(request.outcome),
                success=request.success,
                generate_step_embeddings=False,
            )
        completed_at = None
        if trace.completed_at is not None:
            completed_at = trace.completed_at.isoformat()
        return ReasoningTraceCompleteResponse(
            trace_id=str(trace.id),
            success=trace.success,
            outcome=trace.outcome,
            completed_at=completed_at,
        )

    async def get_reasoning_context(
        self,
        request: ReasoningContextRequest,
    ) -> ReasoningContextResponse:
        async with self._memory_client() as client:
            context = await client.reasoning.get_context(
                request.query,
                max_traces=request.max_items,
            )
        redacted_context = _redacted_text(context)
        if not redacted_context:
            redacted_context = "No similar reasoning traces found."
        return ReasoningContextResponse(context=redacted_context, traces=[])

    def _memory_client(self) -> MemoryClientContext:
        if self._memory_client_factory is not None:
            return self._memory_client_factory(self._settings)
        return MemoryClient(self._settings)


def _build_memory_settings(settings: Settings) -> MemorySettings:
    return MemorySettings(
        backend="bolt",
        neo4j=Neo4jConfig(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=SecretStr(settings.neo4j_password),
        ),
        llm=LiteLLMProvider(
            settings.memory_llm,
            api_base=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        ),
        embedding=LiteLLMEmbeddingProvider(
            litellm_embedding_model(settings.memory_embedding),
            dimensions=settings.memory_embedding_dimensions,
            api_base=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        ),
        memory=MemoryConfig(multi_tenant=True),
    )


def litellm_embedding_model(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


def _session_id(scope: MemoryScope) -> str:
    return scope.session_id


def _user_identifier(scope: MemoryScope) -> str:
    return (
        f"{scope.tenant_id}:{scope.space_id}:{scope.visibility.value}:"
        f"{scope.agent_id}:{scope.user_id}"
    )


def _scope_metadata(scope: MemoryScope) -> dict[str, str]:
    metadata = {
        "tenant_id": scope.tenant_id,
        "space_id": scope.space_id,
        "agent_id": scope.agent_id,
        "session_id": scope.session_id,
        "user_id": scope.user_id,
        "visibility": scope.visibility.value,
    }
    if scope.guild_id is not None:
        metadata["guild_id"] = scope.guild_id
    if scope.channel_id is not None:
        metadata["channel_id"] = scope.channel_id
    return metadata


def _fact_from_row(row: JsonObject) -> JsonObject | None:
    fact = row.get("f")
    if not isinstance(fact, dict):
        return None
    required_fields = ("subject", "predicate", "object")
    if all(isinstance(fact.get(field_name), str) for field_name in required_fields):
        return fact
    return None


def _fact_matches_scope(fact: JsonObject, scope_metadata: dict[str, str]) -> bool:
    metadata = _fact_metadata(fact)
    requested_scope = {
        field_name: scope_value
        for field_name, scope_value in scope_metadata.items()
        if field_name in _FACT_SCOPE_FIELDS
    }
    return all(
        metadata.get(field_name) == requested_value
        for field_name, requested_value in requested_scope.items()
    ) and all(
        requested_scope.get(field_name) == fact_value
        for field_name, fact_value in metadata.items()
        if field_name in _FACT_SCOPE_FIELDS
    )


def _fact_metadata(fact: JsonObject) -> dict[str, str]:
    metadata = fact.get("metadata")
    if isinstance(metadata, str):
        return _metadata_from_json(metadata)
    if isinstance(metadata, dict):
        return _string_metadata(metadata)
    return {}


def _fact_markers(facts: list[JsonObject]) -> set[str]:
    markers: set[str] = set()
    for fact in facts:
        for field_name in ("id", "subject", "object"):
            value = fact.get(field_name)
            if isinstance(value, str) and value != "":
                markers.add(value)
    return markers


def _dedupe_graph_context(
    graph: GraphContextResponse,
    long_term_markers: set[str],
) -> GraphContextResponse:
    if not long_term_markers or not graph.facts:
        return graph

    facts = [
        fact
        for fact in graph.facts
        if not _graph_fact_matches_markers(fact, long_term_markers)
    ]
    if len(facts) == len(graph.facts):
        return graph

    return GraphContextResponse(
        context="\n".join(_graph_fact_summary(fact) for fact in facts),
        facts=facts,
    )


def _graph_fact_matches_markers(fact: JsonObject, markers: set[str]) -> bool:
    return any(
        isinstance(value, str) and value in markers
        for field_name in ("id", "summary")
        if (value := fact.get(field_name)) is not None
    )


def _graph_fact_summary(fact: JsonObject) -> str:
    summary = fact.get("summary")
    if isinstance(summary, str):
        return summary
    return ""


def _metadata_from_json(metadata: str) -> dict[str, str]:
    try:
        parsed = _JSON_OBJECT_ADAPTER.validate_json(metadata)
    except ValidationError:
        return {}
    return _string_metadata(parsed)


def _metadata_fragments(metadata: dict[str, str]) -> list[JsonValue]:
    fragments: list[JsonValue] = []
    for key, value in metadata.items():
        if key != "space_id":
            fragments.append(f'"{key}": {json.dumps(value)}')
    return fragments


def _string_metadata(metadata: dict[str, JsonValue]) -> dict[str, str]:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, str) and value != ""
    }


def _format_provenance(metadata: dict[str, str]) -> str:
    provenance = {
        key: metadata[key]
        for key in sorted(metadata)
        if key in _PROVENANCE_FIELDS and metadata[key]
    }
    return ", ".join(f"{key}={value}" for key, value in provenance.items())


def _redacted_text(value: str) -> str:
    redacted = redact_secrets(value)
    if isinstance(redacted, str):
        return redacted
    return value


def _redacted_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _redacted_text(value)


def _redacted_object(value: JsonObject) -> JsonObject:
    redacted = redact_secrets(value)
    if isinstance(redacted, dict):
        return redacted
    return {}


_PROVENANCE_FIELDS = {
    "tenant_id",
    "agent_id",
    "session_id",
    "user_id",
    "visibility",
    "guild_id",
    "channel_id",
    "event_id",
    "idempotency_key",
    "event_type",
}

_FACT_SCOPE_FIELDS = {
    "tenant_id",
    "agent_id",
    "session_id",
    "user_id",
    "visibility",
    "guild_id",
    "channel_id",
}


def _append_context_section(
    sections: list[MemoryContextSection],
    source: str,
    content: str,
) -> None:
    if content:
        sections.append(MemoryContextSection(source=source, content=content))
