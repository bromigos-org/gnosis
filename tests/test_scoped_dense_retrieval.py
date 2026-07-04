"""Scope-narrowed dense retrieval behind GNOSIS_SCOPED_DENSE_RETRIEVAL_ENABLED.

The SDK's search_facts ranks the fact vector index globally; in a store
holding many users with near-duplicate content (LongMemEval: one user per
question instance, haystack sessions shared between instances) the global
top-k crowds the requesting user out of the candidate pool. The scoped path
over-fetches the vector index and narrows to scope in-query.

Covers: flag-off byte-identity with the SDK dense path, the scoped vector
query's parameters and score contract, scope isolation on the returned rows,
and degradation to the SDK path when the embedder or the vector query fails.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from os import environ
from typing import Self, cast
from uuid import UUID

import pytest

_ = environ.setdefault("GNOSIS_TOKEN", "test-token")
_ = environ.setdefault("GNOSIS_READ_OPERATOR_TOKEN", "read-operator-token")
_ = environ.setdefault("GNOSIS_EXPORT_OPERATOR_TOKEN", "export-operator-token")
_ = environ.setdefault("GNOSIS_WRITE_OPERATOR_TOKEN", "write-operator-token")
_ = environ.setdefault("GNOSIS_ADMIN_OPERATOR_TOKEN", "admin-operator-token")
_ = environ.setdefault("NEO4J_URI", "bolt://neo4j.local:7687")
_ = environ.setdefault("NEO4J_PASSWORD", "inert-password")
_ = environ.setdefault("LITELLM_BASE_URL", "http://litellm.local/v1")
_ = environ.setdefault("LITELLM_API_KEY", "inert-litellm-key")

from neo4j.exceptions import Neo4jError  # noqa: E402
from neo4j_agent_memory import MemorySettings  # noqa: E402
from neo4j_agent_memory.memory.long_term import Fact  # noqa: E402

from gnosis.backend import (  # noqa: E402
    MemoryClientContext,
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
    MemoryContextRequest,
    MemoryScope,
    MemorySearchRequest,
    MemoryVisibility,
)
from gnosis.settings import Settings  # noqa: E402

_MEMORY_ID_ONE = "00000000-0000-0000-0000-0000000000aa"
_MEMORY_ID_TWO = "00000000-0000-0000-0000-0000000000bb"
_MEMORY_ID_THREE = "00000000-0000-0000-0000-0000000000cc"


@pytest.mark.anyio
async def test_search_flag_off_uses_the_sdk_dense_path_byte_identically() -> None:
    # Given: the default settings posture (scoped dense off) plus a vector
    # row that would surface if the scoped query ever ran.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Cartman prefers cheesy poofs", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [_scored_fact_row("scoped decoy", memory_id=_MEMORY_ID_TWO)]
    backend = _backend(client, scoped_dense_enabled=False)

    # When: the caller searches.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="poofs"),
    )

    # Then: no cypher ran, no embedding was requested, and the response is
    # byte-identical to the historical SDK-dense contract.
    assert client.query.calls == []
    assert client.long_term.embedder.embedded == []
    assert response.model_dump_json() == (
        '{"results":[{"memory_id":"00000000-0000-0000-0000-0000000000aa",'
        '"content":"Cartman prefers cheesy poofs","score":0.9,'
        '"metadata":{"topic":"snacks"},'
        '"created_at":"2026-06-27T01:02:03Z","updated_at":null}]}'
    )


@pytest.mark.anyio
async def test_search_scoped_dense_queries_the_vector_index_with_scope() -> None:
    # Given: the scoped flag on and one in-scope vector row.
    client = FakeMemoryClient()
    client.query.rows = [
        _scored_fact_row("in-scope fact", memory_id=_MEMORY_ID_ONE, score=0.83),
    ]
    backend = _backend(client, scoped_dense_enabled=True)

    # When: the caller searches.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="what fact?"),
    )

    # Then: the SDK global ranking was never asked, the vector query embedded
    # the query text and carried the scope fragments plus the over-fetch pool,
    # and the row's vector score became the response score.
    assert client.long_term.search_queries == []
    assert client.long_term.embedder.embedded == ["what fact?"]
    statement, params = client.query.calls[0]
    assert "db.index.vector.queryNodes" in statement
    assert "'fact_embedding_idx'" in statement
    assert params is not None
    assert params["embedding"] == [0.1, 0.2]
    assert params["vector_pool"] == 4000
    assert params["scope_fragments"] == [
        '"tenant_id": "bromigos"',
        '"user_id": "789"',
    ]
    assert params["candidate_limit"] == 100
    assert [(result.content, result.score) for result in response.results] == [
        ("in-scope fact", 0.83),
    ]


@pytest.mark.anyio
async def test_search_scoped_dense_keeps_scope_isolation_on_returned_rows() -> None:
    # Given: the vector query returns cross-tenant and cross-user rows next
    # to the in-scope hit (as if the storage filter were bypassed).
    client = FakeMemoryClient()
    client.query.rows = [
        _scored_fact_row(
            "other tenant secret",
            memory_id=_MEMORY_ID_TWO,
            metadata=_scope_metadata() | {"tenant_id": "other-tenant"},
        ),
        _scored_fact_row(
            "other user secret",
            memory_id=_MEMORY_ID_THREE,
            metadata=_scope_metadata() | {"user_id": "666"},
        ),
        _scored_fact_row("in-scope fact", memory_id=_MEMORY_ID_ONE),
    ]
    backend = _backend(client, scoped_dense_enabled=True)

    # When: the caller searches.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="secret?"),
    )

    # Then: the gateway scope re-check drops the cross-scope rows.
    assert [result.content for result in response.results] == ["in-scope fact"]


@pytest.mark.anyio
async def test_search_scoped_dense_vector_failure_degrades_to_sdk_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the vector query raises while the SDK ranking has results.
    client = FakeMemoryClient(query=FailingCypherQuery())
    client.long_term.search_results = [
        _fact("sdk fallback answer", memory_id=_MEMORY_ID_ONE),
    ]
    backend = _backend(client, scoped_dense_enabled=True)

    # When: the caller searches with the scoped flag on.
    with caplog.at_level(logging.WARNING):
        response = await backend.search_memories(
            MemorySearchRequest(scope=_scope(), query="anything"),
        )

    # Then: the read never fails - the SDK ranking answers and a structured
    # warning records the degradation.
    assert [result.content for result in response.results] == ["sdk fallback answer"]
    assert "scoped dense search failed" in caplog.text


@pytest.mark.anyio
async def test_search_scoped_dense_missing_embedder_degrades_to_sdk_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the SDK client exposes no embedder.
    client = FakeMemoryClient(
        long_term=FakeLongTermMemory(
            embedder=cast("FakeEmbedder", cast("object", None))
        ),
    )
    client.long_term.search_results = [
        _fact("sdk fallback answer", memory_id=_MEMORY_ID_ONE),
    ]
    backend = _backend(client, scoped_dense_enabled=True)

    # When: the caller searches with the scoped flag on.
    with caplog.at_level(logging.WARNING):
        response = await backend.search_memories(
            MemorySearchRequest(scope=_scope(), query="anything"),
        )

    # Then: the read degrades to the SDK ranking with a warning.
    assert [result.content for result in response.results] == ["sdk fallback answer"]
    assert "scoped dense search failed" in caplog.text
    assert client.query.calls == []


@pytest.mark.anyio
async def test_context_scoped_dense_feeds_the_long_term_section() -> None:
    # Given: the scoped flag on and one in-scope vector row.
    client = FakeMemoryClient()
    client.query.rows = [
        _scored_fact_row("routed scoped fact", memory_id=_MEMORY_ID_ONE),
    ]
    backend = _backend(client, scoped_dense_enabled=True)

    # When: context is assembled.
    response = await backend.get_memory_context(_context_request())

    # Then: the scoped candidate renders into the long-term facts section
    # and the SDK global ranking was never asked.
    long_term = [s for s in response.sections if s.source == "long_term_facts"]
    assert len(long_term) == 1
    assert "routed scoped fact" in long_term[0].content
    assert client.long_term.search_queries == []


@dataclass(slots=True)
class FakeEmbedder:
    embedded: list[str] = field(default_factory=list)

    async def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return [0.1, 0.2]


@dataclass(slots=True)
class FakeLongTermMemory:
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)
    search_results: list[Fact] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[Fact]:
        _ = (limit, threshold)
        self.search_queries.append(query)
        return list(self.search_results)

    async def get_context(self, query: str, *, max_items: int) -> str:
        _ = (query, max_items)
        return ""


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
class FailingCypherQuery:
    rows: list[JsonObject] = field(default_factory=list)
    calls: list[tuple[str, dict[str, JsonValue] | None]] = field(default_factory=list)

    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.calls.append((query, params))
        message = "vector index unavailable"
        raise Neo4jError(message)


@dataclass(slots=True)
class FakeMemoryClient:
    long_term: FakeLongTermMemory = field(default_factory=FakeLongTermMemory)
    query: FakeCypherQuery | FailingCypherQuery = field(
        default_factory=FakeCypherQuery,
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


def _backend(
    client: FakeMemoryClient,
    *,
    scoped_dense_enabled: bool,
) -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        Settings(gnosis_scoped_dense_retrieval_enabled=scoped_dense_enabled),
        memory_client_factory=FakeMemoryClientFactory(client),
        graph_store=FakeGraphStore(),
    )


def _context_request() -> MemoryContextRequest:
    return MemoryContextRequest(
        scope=_scope(),
        query="what happened?",
        include_short_term=False,
        include_reasoning=False,
        include_graph=False,
        max_items=8,
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


def _scope_metadata() -> dict[str, JsonValue]:
    return {
        "tenant_id": "bromigos",
        "space_id": "discord",
        "agent_id": "pc-principal",
        "session_id": "guild:123:channel:456",
        "user_id": "789",
        "visibility": "private_user",
    }


def _fact(content: str, *, memory_id: str, similarity: float = 0.9) -> Fact:
    return Fact(
        id=UUID(memory_id),
        subject="bromigos:discord:private_user:pc-principal:789",
        predicate="memory",
        object=content,
        created_at=datetime(2026, 6, 27, 1, 2, 3, tzinfo=UTC),
        metadata=_scope_metadata() | {"similarity": similarity, "topic": "snacks"},
    )


def _scored_fact_row(
    content: str,
    *,
    memory_id: str,
    metadata: dict[str, JsonValue] | None = None,
    score: float = 0.9,
) -> JsonObject:
    return {
        "id": memory_id,
        "subject": "bromigos:discord:private_user:pc-principal:789",
        "predicate": "memory",
        "object": content,
        "metadata": json.dumps(metadata or _scope_metadata()),
        "created_at": "2026-06-27T01:02:03+00:00",
        "updated_at": None,
        "score": score,
    }
