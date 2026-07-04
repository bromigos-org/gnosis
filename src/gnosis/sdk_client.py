"""Structural typing and construction for the neo4j-agent-memory client.

Everything gnosis assumes about the SDK client lives here: the protocol
shapes of the short-term/long-term/reasoning surfaces and the optional
capability protocols probed with ``isinstance`` at call time, plus the
builders that map gnosis :class:`~gnosis.settings.Settings` onto SDK
configuration and the small handle helpers (graph write access, embedding,
graph export conversion) that adapt SDK objects for the backend.
"""

from collections.abc import Awaitable, Sequence
from datetime import datetime
from typing import Final, Literal, Protocol, Self, TypedDict, cast, runtime_checkable
from uuid import UUID

from neo4j_agent_memory import MemoryConfig, MemorySettings, Neo4jConfig
from neo4j_agent_memory.llm.adapters.litellm import (
    LiteLLMEmbeddingProvider,
    LiteLLMProvider,
)
from neo4j_agent_memory.memory.long_term import EntityType
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolCallStatus, ToolStats
from neo4j_agent_memory.schema.models import EntityRef
from pydantic import SecretStr

from gnosis.backend_protocols import BackendCapabilityUnavailable
from gnosis.json_redaction import json_object, redacted_object
from gnosis.models import (
    EntityRecord,
    FactRecord,
    GraphExportNode,
    GraphExportRelationship,
    GraphExportRequest,
    GraphExportResponse,
    JsonObject,
    JsonValue,
    PreferenceRecord,
)
from gnosis.settings import Settings

_MEMORY_WRITE_UNAVAILABLE_DETAIL: Final[str] = "SDK graph writes are unavailable."


class MemoryConfigKwargs(TypedDict, total=False):
    multi_tenant: bool
    write_mode: Literal["sync", "buffered"]
    max_pending: int
    conversation_ttl_days: int | None
    audit_read: bool
    fact_deduplication_enabled: bool
    trace_embedding_enabled: bool


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


class LongTermFactMemory(Protocol):
    async def search_entities(
        self,
        query: str,
        *,
        entity_types: list[EntityType | str] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> object: ...

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> object: ...

    async def search_preferences(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> object: ...

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
    ) -> object: ...

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
    ) -> object: ...

    async def add_preference(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
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
    ) -> object: ...

    async def get_context(self, query: str, *, max_items: int) -> str: ...


class LongTermMemory(Protocol):
    async def search_entities(
        self,
        query: str,
        *,
        entity_types: list[EntityType | str] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[EntityRecord]: ...

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[FactRecord]: ...

    async def search_preferences(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[PreferenceRecord]: ...

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
    ) -> EntityRecord: ...

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
    ) -> FactRecord: ...

    async def add_preference(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
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
    ) -> PreferenceRecord: ...

    async def get_preferences_for(
        self,
        user_identifier: str,
        *,
        applies_to: object | None = None,
        active_only: bool = True,
        as_of: datetime | None = None,
    ) -> list[PreferenceRecord]: ...

    async def get_facts_about(
        self,
        subject: str,
        *,
        limit: int = 100,
    ) -> list[FactRecord]: ...

    async def link_entity_to_message(  # noqa: PLR0913 - Mirrors SDK API.
        self,
        entity: EntityRecord | UUID,
        message_id: UUID | str,
        *,
        confidence: float = 1.0,
        start_pos: int | None = None,
        end_pos: int | None = None,
        context: str | None = None,
    ) -> bool: ...

    async def link_entity_to_extractor(
        self,
        entity: EntityRecord | UUID,
        extractor_name: str,
        *,
        confidence: float = 1.0,
        extraction_time_ms: float | None = None,
    ) -> bool: ...

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
    async def list_traces(
        self,
        *,
        session_id: str | None = None,
        success_only: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SdkReasoningTrace]: ...
    async def get_trace(self, trace_id: UUID | str) -> SdkReasoningTrace | None: ...
    async def get_trace_with_steps(
        self,
        trace_id: UUID,
    ) -> SdkReasoningTrace | None: ...
    async def get_similar_traces(
        self,
        task: str,
        *,
        limit: int = 5,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[SdkReasoningTrace]: ...
    async def search_steps(
        self,
        query: str,
        *,
        limit: int = 10,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> Sequence[object]: ...
    async def get_tool_stats(
        self,
        tool_name: str | None = None,
    ) -> list[ToolStats]: ...


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
    def long_term(self) -> LongTermFactMemory: ...
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


@runtime_checkable
class StatsCapableMemoryClient(Protocol):
    def get_stats(self) -> Awaitable[object]: ...


@runtime_checkable
class BufferFlushCapableMemoryClient(Protocol):
    def flush(self) -> Awaitable[object]: ...


@runtime_checkable
class BufferPendingCapableMemoryClient(Protocol):
    def wait_for_pending(self) -> Awaitable[object]: ...


@runtime_checkable
class BufferErrorCapableMemoryClient(Protocol):
    @property
    def write_errors(self) -> Sequence[object]: ...


class GraphNodeLike(Protocol):
    id: str
    labels: Sequence[str]
    properties: object


class GraphRelationshipLike(Protocol):
    id: str
    type: str
    from_node: str
    to_node: str
    properties: object


class MemoryGraphLike(Protocol):
    nodes: Sequence[GraphNodeLike]
    relationships: Sequence[GraphRelationshipLike]
    metadata: object


@runtime_checkable
class GraphCapableMemoryClient(Protocol):
    def get_graph(
        self,
        *,
        memory_types: list[Literal["short_term", "long_term", "reasoning"]] | None,
        session_id: str | None,
        include_embeddings: bool,
        limit: int,
    ) -> Awaitable[MemoryGraphLike]: ...


@runtime_checkable
class GraphWriteQuery(Protocol):
    def execute_write(
        self,
        query: str,
        parameters: dict[str, JsonValue] | None = None,
    ) -> Awaitable[list[JsonObject]]: ...


@runtime_checkable
class TextEmbedder(Protocol):
    def embed(self, text: str) -> Awaitable[list[float]]: ...


def build_memory_settings(settings: Settings) -> MemorySettings:
    return MemorySettings(
        backend="bolt",
        neo4j=Neo4jConfig(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=SecretStr(settings.neo4j_password),
        ),
        llm=LiteLLMProvider(
            settings.gnosis_llm,
            api_base=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        ),
        embedding=LiteLLMEmbeddingProvider(
            litellm_embedding_model(settings.gnosis_embedding),
            dimensions=settings.gnosis_embedding_dimensions,
            api_base=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        ),
        memory=build_memory_config(settings),
    )


def build_memory_config(settings: Settings) -> MemoryConfig:
    config = MemoryConfigKwargs(multi_tenant=True)
    supported_fields = MemoryConfig.model_fields
    if "write_mode" in supported_fields:
        config["write_mode"] = settings.gnosis_write_mode
    if "max_pending" in supported_fields:
        config["max_pending"] = settings.gnosis_max_pending
    if "conversation_ttl_days" in supported_fields:
        config["conversation_ttl_days"] = settings.gnosis_conversation_ttl_days
    if "audit_read" in supported_fields:
        config["audit_read"] = settings.gnosis_audit_read
    if "fact_deduplication_enabled" in supported_fields:
        config["fact_deduplication_enabled"] = (
            settings.gnosis_fact_deduplication_enabled
        )
    if "trace_embedding_enabled" in supported_fields:
        config["trace_embedding_enabled"] = settings.gnosis_trace_embedding_enabled
    return MemoryConfig(**config)


def litellm_embedding_model(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


def memory_client_context(client: object) -> MemoryClientContext:
    return cast("MemoryClientContext", client)


def graph_write_query(client: MemoryClientContext) -> GraphWriteQuery:
    """Acquire the graph write handle used by memory update and delete.

    The runtime-protocol ``isinstance`` check alone is too strict for the
    installed SDK: since Python 3.12 it resolves members with
    ``inspect.getattr_static``, and ``neo4j-agent-memory==0.5.0`` returns a
    ``client.graph`` proxy that forwards ``execute_write`` to the same driver
    the read routes use only through dynamic ``__getattr__`` delegation. Fall
    back to a duck-typed check so that proxy stays usable.
    """
    graph: object = getattr(client, "graph", None)
    if isinstance(graph, GraphWriteQuery):
        return graph
    execute_write: object = getattr(graph, "execute_write", None)
    if graph is not None and callable(execute_write):
        return cast("GraphWriteQuery", graph)
    raise BackendCapabilityUnavailable(_MEMORY_WRITE_UNAVAILABLE_DETAIL)


async def memory_embedding(
    client: MemoryClientContext,
    text: str,
) -> list[JsonValue] | None:
    embedder: object = getattr(client.long_term, "embedder", None)
    if not isinstance(embedder, TextEmbedder):
        return None
    return [float(item) for item in await embedder.embed(text)]


def graph_export_response(
    request: GraphExportRequest,
    graph: MemoryGraphLike,
) -> GraphExportResponse:
    return GraphExportResponse(
        scope=request.scope,
        nodes=[_graph_export_node(node) for node in graph.nodes],
        relationships=[
            _graph_export_relationship(relationship)
            for relationship in graph.relationships
        ],
        metadata=redacted_object(json_object(graph.metadata)),
    )


def _graph_export_node(node: GraphNodeLike) -> GraphExportNode:
    return GraphExportNode(
        id=node.id,
        labels=list(node.labels),
        properties=redacted_object(json_object(node.properties)),
    )


def _graph_export_relationship(
    relationship: GraphRelationshipLike,
) -> GraphExportRelationship:
    return GraphExportRelationship(
        id=relationship.id,
        type=relationship.type,
        from_node=relationship.from_node,
        to_node=relationship.to_node,
        properties=redacted_object(json_object(relationship.properties)),
    )
