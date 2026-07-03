import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from os import environ
from typing import Self, cast

import pytest

_ = environ.setdefault("GNOSIS_TOKEN", "memory-access")
_ = environ.setdefault("GNOSIS_READ_OPERATOR_TOKEN", "read-operator-token")
_ = environ.setdefault("GNOSIS_EXPORT_OPERATOR_TOKEN", "export-operator-token")
_ = environ.setdefault("GNOSIS_WRITE_OPERATOR_TOKEN", "write-operator-token")
_ = environ.setdefault("GNOSIS_ADMIN_OPERATOR_TOKEN", "admin-operator-token")
_ = environ.setdefault("NEO4J_URI", "bolt://neo4j.local:7687")
_ = environ.setdefault("NEO4J_PASSWORD", "inert-password")
_ = environ.setdefault("LITELLM_BASE_URL", "http://litellm.local/v1")
_ = environ.setdefault("LITELLM_API_KEY", "inert-litellm-key")

from neo4j_agent_memory import MemorySettings  # noqa: E402
from neo4j_agent_memory.memory.long_term import Fact  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

from gnosis.backend import (  # noqa: E402
    BackendCapabilityUnavailable,
    BackendRequestError,
    MemoryClientContext,
    MemoryNotFoundError,
    Neo4jAgentMemoryBackend,
)
from gnosis.models import (  # noqa: E402
    BackendReadiness,
    ClientEvent,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    JsonObject,
    JsonValue,
    MemoryAddRequest,
    MemoryDeleteRequest,
    MemoryListRequest,
    MemoryMessage,
    MemoryScope,
    MemorySearchRequest,
    MemoryUpdateRequest,
    MemoryVisibility,
)
from gnosis.settings import Settings  # noqa: E402

_MEMORY_ID = "00000000-0000-0000-0000-0000000000aa"
_OTHER_MEMORY_ID = "00000000-0000-0000-0000-0000000000bb"
_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


@pytest.mark.anyio
async def test_add_memories_when_content_is_verbatim() -> None:
    # Given: a verbatim add request with caller metadata tags.
    client = FakeMemoryClient()
    backend = _backend(client)
    request = MemoryAddRequest(
        scope=_scope(),
        content="Cartman prefers cheesy poofs",
        infer=False,
        metadata={"topic": "snacks"},
    )

    # When: the memory is added.
    response = await backend.add_memories(request)

    # Then: a durable fact is written with scope tags and a stable id returns.
    assert len(response.results) == 1
    result = response.results[0]
    assert result.event == "ADD"
    assert result.memory_id == str(client.long_term.facts[0].id)
    assert result.content == "Cartman prefers cheesy poofs"
    assert result.metadata == {"topic": "snacks"}
    write = client.long_term.fact_writes[0]
    assert write.predicate == "memory"
    assert write.metadata["tenant_id"] == "bromigos"
    assert write.metadata["user_id"] == "789"
    assert write.metadata["agent_id"] == "pc-principal"
    assert write.metadata["topic"] == "snacks"
    assert client.short_term.messages == []


@pytest.mark.anyio
async def test_add_memories_when_messages_use_extraction_mode() -> None:
    # Given: a conversation pair synced with infer=true.
    client = FakeMemoryClient()
    backend = _backend(client)
    request = MemoryAddRequest(
        scope=_scope(),
        messages=[
            MemoryMessage(role="user", content="I love cheesy poofs"),
            MemoryMessage(role="assistant", content="Noted, cheesy poofs it is"),
        ],
    )

    # When: the memories are added.
    response = await backend.add_memories(request)

    # Then: each turn flows through the SDK message and fact add paths.
    assert [message.role for message in client.short_term.messages] == [
        "user",
        "assistant",
    ]
    assert [write.predicate for write in client.long_term.fact_writes] == [
        "said_user",
        "said_assistant",
    ]
    assert [result.event for result in response.results] == ["ADD", "ADD"]
    assert all(result.memory_id for result in response.results)


@pytest.mark.anyio
async def test_add_memories_when_sdk_deduplicates_the_fact() -> None:
    # Given: the SDK reports the fact as a deduplicated update.
    client = FakeMemoryClient()
    client.long_term.deduplicated = True
    backend = _backend(client)

    # When: the memory is added.
    response = await backend.add_memories(
        MemoryAddRequest(scope=_scope(), content="repeat", infer=False),
    )

    # Then: the caller sees an UPDATE event with the surviving id.
    assert response.results[0].event == "UPDATE"
    assert response.results[0].memory_id == str(client.long_term.facts[0].id)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "request_payload",
    [
        {"content": "x", "infer": True},
        {"messages": [{"role": "user", "content": "x"}], "infer": False},
        {"infer": True},
        {
            "content": "x",
            "messages": [{"role": "user", "content": "x"}],
            "infer": True,
        },
    ],
)
async def test_add_memories_when_mode_is_invalid(
    request_payload: dict[str, object],
) -> None:
    # Given: an add request mixing or omitting the two supported modes.
    backend = _backend(FakeMemoryClient())
    request = MemoryAddRequest.model_validate({"scope": _scope(), **request_payload})

    # When / Then: the backend rejects the request.
    with pytest.raises(BackendRequestError):
        _ = await backend.add_memories(request)


@pytest.mark.anyio
async def test_search_memories_when_results_cross_scopes() -> None:
    # Given: SDK search results from this user, another user, and low score.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("in scope", _scope_metadata() | {"similarity": 0.92, "topic": "snacks"}),
        _fact("other user", _scope_metadata(user_id="666") | {"similarity": 0.99}),
        _fact("low score", _scope_metadata() | {"similarity": 0.11}),
    ]
    backend = _backend(client)

    # When: the caller searches with a minimum score.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="poofs", min_score=0.5),
    )

    # Then: only in-scope results above the score cut return, redacted metadata.
    assert [result.content for result in response.results] == ["in scope"]
    assert response.results[0].score == 0.92
    assert response.results[0].metadata == {"topic": "snacks"}
    assert response.results[0].memory_id


@pytest.mark.anyio
async def test_search_memories_when_filters_narrow_results() -> None:
    # Given: two in-scope results with different metadata tags.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("snacks", _scope_metadata() | {"similarity": 0.9, "topic": "snacks"}),
        _fact("games", _scope_metadata() | {"similarity": 0.8, "topic": "games"}),
    ]
    backend = _backend(client)

    # When: the caller filters on a metadata tag.
    response = await backend.search_memories(
        MemorySearchRequest(
            scope=_scope(),
            query="poofs",
            filters={"metadata.topic": "games"},
        ),
    )

    # Then: only the matching record returns.
    assert [result.content for result in response.results] == ["games"]


@pytest.mark.anyio
async def test_search_memories_when_filters_are_invalid() -> None:
    # Given: a filter with an unknown field.
    backend = _backend(FakeMemoryClient())

    # When / Then: the request is rejected before any SDK call.
    with pytest.raises(BackendRequestError):
        _ = await backend.search_memories(
            MemorySearchRequest(
                scope=_scope(),
                query="poofs",
                filters={"session_id": "guild:1"},
            ),
        )


@pytest.mark.anyio
async def test_list_memories_when_pages_are_deterministic() -> None:
    # Given: three stored rows ordered by created_at desc, one out of scope.
    client = FakeMemoryClient()
    client.query.rows = [
        _memory_row(_MEMORY_ID, "newest", "2026-06-29T00:00:00+00:00"),
        _memory_row(
            _OTHER_MEMORY_ID,
            "other user",
            "2026-06-28T00:00:00+00:00",
            user_id="666",
        ),
        _memory_row(
            "00000000-0000-0000-0000-0000000000cc",
            "older",
            "2026-06-27T00:00:00+00:00",
        ),
    ]
    backend = _backend(client)

    # When: the caller lists the second page with page_size 1.
    response = await backend.list_memories(
        MemoryListRequest(scope=_scope(), page=2, page_size=1),
    )

    # Then: cross-scope rows are dropped and totals reflect scoped matches.
    assert response.total == 2
    assert response.page == 2
    assert response.page_size == 1
    assert [result.content for result in response.results] == ["older"]
    query, params = client.query.calls[0]
    assert "ORDER BY f.created_at DESC, f.id ASC" in query
    assert params is not None
    assert params["scope_fragments"] == ['"tenant_id": "bromigos"', '"user_id": "789"']


@pytest.mark.anyio
async def test_list_memories_when_filters_produce_narrowing_parameters() -> None:
    # Given: a metadata filter that narrows in Cypher and in the gateway.
    client = FakeMemoryClient()
    client.query.rows = [
        _memory_row(_MEMORY_ID, "snacks", "2026-06-29T00:00:00+00:00", topic="snacks"),
        _memory_row(
            _OTHER_MEMORY_ID,
            "games",
            "2026-06-28T00:00:00+00:00",
            topic="games",
        ),
    ]
    backend = _backend(client)

    # When: the caller lists with the filter DSL.
    response = await backend.list_memories(
        MemoryListRequest(scope=_scope(), filters={"metadata.topic": "snacks"}),
    )

    # Then: values ride as parameters and exact evaluation prunes the rest.
    assert [result.content for result in response.results] == ["snacks"]
    query, params = client.query.calls[0]
    assert "f.metadata CONTAINS $filter_0" in query
    assert params is not None
    assert params["filter_0"] == '"topic": "snacks"'


@pytest.mark.anyio
async def test_update_memory_when_record_is_in_scope() -> None:
    # Given: a stored memory owned by the request scope.
    client = FakeMemoryClient()
    client.query.rows = [_memory_row(_MEMORY_ID, "old", "2026-06-27T00:00:00+00:00")]
    write_rows: list[JsonObject] = [{"object": "new content"}]
    _graph(client).write_rows = write_rows
    backend = _backend(client)

    # When: the caller updates content and metadata.
    response = await backend.update_memory(
        _MEMORY_ID,
        MemoryUpdateRequest(
            scope=_scope(),
            content="new content",
            metadata={"topic": "games"},
        ),
    )

    # Then: the parameterized write keeps the id, scope tags, and embedding.
    assert response.memory_id == _MEMORY_ID
    assert response.content == "new content"
    assert response.event == "UPDATE"
    write_query, write_params = _graph(client).writes[0]
    assert "SET f.object" in write_query
    assert write_params is not None
    assert write_params["memory_id"] == _MEMORY_ID
    assert write_params["content"] == "new content"
    assert write_params["embedding"] == [0.1, 0.2]
    stored_metadata = _JSON_OBJECT_ADAPTER.validate_json(
        cast("str", write_params["metadata"]),
    )
    assert stored_metadata["topic"] == "games"
    assert stored_metadata["tenant_id"] == "bromigos"
    assert stored_metadata["user_id"] == "789"


@pytest.mark.anyio
async def test_update_memory_when_no_fields_are_provided() -> None:
    # Given: an update request without content or metadata.
    backend = _backend(FakeMemoryClient())

    # When / Then: the backend rejects the empty update.
    with pytest.raises(BackendRequestError):
        _ = await backend.update_memory(
            _MEMORY_ID,
            MemoryUpdateRequest(scope=_scope()),
        )


@pytest.mark.anyio
async def test_update_memory_when_record_belongs_to_another_scope() -> None:
    # Given: the stored memory is tagged to a different user.
    client = FakeMemoryClient()
    client.query.rows = [
        _memory_row(_MEMORY_ID, "secret", "2026-06-27T00:00:00+00:00", user_id="666"),
    ]
    backend = _backend(client)

    # When / Then: the memory is reported as not found without a write.
    with pytest.raises(MemoryNotFoundError):
        _ = await backend.update_memory(
            _MEMORY_ID,
            MemoryUpdateRequest(scope=_scope(), content="hijack"),
        )
    assert _graph(client).writes == []


@pytest.mark.anyio
async def test_delete_memory_when_record_is_in_scope() -> None:
    # Given: a stored memory owned by the request scope.
    client = FakeMemoryClient()
    client.query.rows = [_memory_row(_MEMORY_ID, "old", "2026-06-27T00:00:00+00:00")]
    backend = _backend(client)

    # When: the caller deletes the memory.
    response = await backend.delete_memory(
        _MEMORY_ID,
        MemoryDeleteRequest(scope=_scope()),
    )

    # Then: the node is detached and deleted by parameterized id.
    assert response.memory_id == _MEMORY_ID
    assert response.event == "DELETE"
    write_query, write_params = _graph(client).writes[0]
    assert "DETACH DELETE f" in write_query
    assert write_params == {"memory_id": _MEMORY_ID}


@pytest.mark.anyio
async def test_delete_memory_when_record_is_missing() -> None:
    # Given: no stored memory matches the id.
    client = FakeMemoryClient()
    backend = _backend(client)

    # When / Then: the delete reports not found without a write.
    with pytest.raises(MemoryNotFoundError):
        _ = await backend.delete_memory(
            _MEMORY_ID,
            MemoryDeleteRequest(scope=_scope()),
        )
    assert _graph(client).writes == []


@pytest.mark.anyio
async def test_delete_memory_when_sdk_has_no_graph_write_access() -> None:
    # Given: an SDK client without a graph write surface.
    client = FakeMemoryClient(graph=None)
    client.query.rows = [_memory_row(_MEMORY_ID, "old", "2026-06-27T00:00:00+00:00")]
    backend = _backend(client)

    # When / Then: the closest safe behavior is a capability error.
    with pytest.raises(BackendCapabilityUnavailable):
        _ = await backend.delete_memory(
            _MEMORY_ID,
            MemoryDeleteRequest(scope=_scope()),
        )


def _graph(client: "FakeMemoryClient") -> "FakeGraphWriter":
    assert client.graph is not None
    return client.graph


def _backend(client: "FakeMemoryClient") -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=FakeMemoryClientFactory(client),
        graph_store=FakeGraphStore(),
    )


def _scope() -> MemoryScope:
    return MemoryScope(
        tenant_id="bromigos",
        space_id="discord",
        agent_id="pc-principal",
        session_id="guild:123:channel:456",
        user_id="789",
        visibility=MemoryVisibility.PRIVATE_USER,
    )


def _scope_metadata(*, user_id: str = "789") -> dict[str, JsonValue]:
    return {
        "tenant_id": "bromigos",
        "space_id": "discord",
        "agent_id": "pc-principal",
        "session_id": "guild:123:channel:456",
        "user_id": user_id,
        "visibility": "private_user",
    }


def _fact(content: str, metadata: dict[str, JsonValue]) -> Fact:
    return Fact(
        subject="bromigos:discord:private_user:pc-principal:789",
        predicate="memory",
        object=content,
        created_at=datetime(2026, 6, 27, 1, 2, 3, tzinfo=UTC),
        metadata=dict(metadata),
    )


def _memory_row(
    memory_id: str,
    content: str,
    created_at: str,
    *,
    user_id: str = "789",
    topic: str | None = None,
) -> JsonObject:
    metadata = _scope_metadata(user_id=user_id)
    if topic is not None:
        metadata["topic"] = topic
    return {
        "id": memory_id,
        "subject": "bromigos:discord:private_user:pc-principal:789",
        "predicate": "memory",
        "object": content,
        "metadata": json.dumps(metadata),
        "created_at": created_at,
        "updated_at": None,
    }


@dataclass(frozen=True, slots=True)
class ShortTermWrite:
    session_id: str
    role: str
    content: str
    user_identifier: str
    metadata: dict[str, str]
    extract_entities: bool
    extract_relations: bool


@dataclass(frozen=True, slots=True)
class FactWrite:
    subject: str
    predicate: str
    object: str
    metadata: JsonObject


@dataclass(slots=True)
class FakeShortTermMemory:
    messages: list[ShortTermWrite] = field(default_factory=list)

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
        self.messages.append(
            ShortTermWrite(
                session_id=session_id,
                role=role,
                content=content,
                user_identifier=user_identifier,
                metadata=dict(metadata),
                extract_entities=extract_entities,
                extract_relations=extract_relations,
            ),
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
class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        _ = text
        return [0.1, 0.2]


@dataclass(slots=True)
class FakeLongTermMemory:
    facts: list[Fact] = field(default_factory=list)
    fact_writes: list[FactWrite] = field(default_factory=list)
    search_results: list[Fact] = field(default_factory=list)
    deduplicated: bool = False
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
    ) -> Fact:
        _ = generate_embedding
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=obj,
            created_at=datetime.now(UTC),
            metadata=dict(metadata or {}),
        )
        if self.deduplicated:
            fact.metadata["deduplicated"] = True
        self.facts.append(fact)
        self.fact_writes.append(
            FactWrite(
                subject=subject,
                predicate=predicate,
                object=obj,
                metadata=dict(metadata or {}),
            ),
        )
        return fact

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[Fact]:
        _ = (query, limit, threshold)
        return list(self.search_results)


@dataclass(slots=True)
class FakeCypherQuery:
    rows: list[JsonObject] = field(default_factory=list)
    calls: list[tuple[str, dict[str, JsonValue] | None]] = field(default_factory=list)

    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.calls.append((query, params))
        return list(self.rows)


@dataclass(slots=True)
class FakeGraphWriter:
    write_rows: list[JsonObject] = field(default_factory=list)
    writes: list[tuple[str, dict[str, JsonValue] | None]] = field(default_factory=list)

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.writes.append((query, parameters))
        return list(self.write_rows)


@dataclass(slots=True)
class FakeMemoryClient:
    short_term: FakeShortTermMemory = field(default_factory=FakeShortTermMemory)
    long_term: FakeLongTermMemory = field(default_factory=FakeLongTermMemory)
    query: FakeCypherQuery = field(default_factory=FakeCypherQuery)
    graph: FakeGraphWriter | None = field(default_factory=FakeGraphWriter)

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
class FakeMemoryClientFactory:
    client: FakeMemoryClient

    def __call__(self, settings: MemorySettings) -> MemoryClientContext:
        _ = settings
        client = cast("object", self.client)
        return cast("MemoryClientContext", client)


@dataclass(slots=True)
class FakeGraphStore:
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
        _ = request
        return GraphContextResponse(context="")
