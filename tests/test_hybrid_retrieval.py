"""Hybrid lexical+dense retrieval behind GNOSIS_HYBRID_RETRIEVAL_ENABLED.

Covers the RRF fusion math, Lucene query sanitization, the lexical BM25 leg
of /v1/memories/search and context assembly, scope isolation on the lexical
path, dense-only degradation on full-text failures, and the pinned
byte-identical contract with the flag off.
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
from gnosis.memory_provider import (  # noqa: E402
    CREATE_FACT_OBJECT_FULLTEXT_INDEX_CYPHER,
    FACT_OBJECT_FULLTEXT_INDEX,
    StoredMemory,
    fuse_memory_rankings,
    sanitize_lucene_query,
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
_MEMORY_ID_FOUR = "00000000-0000-0000-0000-0000000000dd"


def test_rrf_fusion_orders_by_reciprocal_rank_sum() -> None:
    # Given: a dense ranking [A, B, C] and a lexical ranking [C, D].
    memory_a = _stored_memory(_MEMORY_ID_ONE, "a")
    memory_b = _stored_memory(_MEMORY_ID_TWO, "b")
    memory_c = _stored_memory(_MEMORY_ID_THREE, "c")
    memory_d = _stored_memory(_MEMORY_ID_FOUR, "d")

    # When: the rankings are fused with RRF (k=60).
    fused = fuse_memory_rankings(
        [memory_a, memory_b, memory_c],
        [memory_c, memory_d],
    )

    # Then: C wins (1/63 + 1/61), A follows (1/61), and the B/D tie at 1/62
    # keeps dense-first arrival order.
    assert [memory.memory_id for memory in fused] == [
        _MEMORY_ID_THREE,
        _MEMORY_ID_ONE,
        _MEMORY_ID_TWO,
        _MEMORY_ID_FOUR,
    ]


def test_rrf_fusion_keeps_dense_representative_for_shared_hits() -> None:
    # Given: the same memory id ranked by both legs with different metadata.
    dense = _stored_memory(_MEMORY_ID_ONE, "shared", similarity=0.92)
    lexical = _stored_memory(_MEMORY_ID_ONE, "shared", similarity=0.0)

    # When: the rankings are fused.
    fused = fuse_memory_rankings([dense], [lexical])

    # Then: one record survives and it is the dense one, so the vector
    # similarity score stays intact for the response contract.
    assert len(fused) == 1
    assert fused[0].metadata.get("similarity") == 0.92


def test_rrf_fusion_dedupes_within_one_ranking_keeping_best_rank() -> None:
    # Given: a lexical ranking that repeats one memory id.
    memory_a = _stored_memory(_MEMORY_ID_ONE, "a")
    memory_b = _stored_memory(_MEMORY_ID_TWO, "b")

    # When: the duplicate-bearing ranking is fused against a dense ranking
    # that only saw B.
    fused = fuse_memory_rankings([memory_b], [memory_a, memory_a, memory_b])

    # Then: A contributes once at its best rank (1/61) while B sums both legs
    # (1/61 + 1/62) and wins.
    assert [memory.memory_id for memory in fused] == [
        _MEMORY_ID_TWO,
        _MEMORY_ID_ONE,
    ]


@pytest.mark.parametrize(
    ("raw", "sanitized"),
    [
        ("foo AND bar)", "foo and bar\\)"),
        ('"unclosed', '\\"unclosed'),
        ("field:x", "field\\:x"),
        ("a OR b NOT c", "a or b not c"),
        ("wild*card? fuzz~ boost^2", "wild\\*card\\? fuzz\\~ boost\\^2"),
        ("path\\traversal/segment", "path\\\\traversal\\/segment"),
        ("range:[1 TO 2] {curly}", "range\\:\\[1 TO 2\\] \\{curly\\}"),
        ("plus+minus- and && or ||", "plus\\+minus\\- and \\&\\& or \\|\\|"),
        ("ANDROID ORCA NOTE", "ANDROID ORCA NOTE"),
        ("   ", ""),
    ],
)
def test_sanitize_lucene_query_neutralizes_injection(
    raw: str,
    sanitized: str,
) -> None:
    # Given/When/Then: every Lucene special character is escaped and bare
    # boolean operator words are defused, so user input never reaches the
    # full-text index as query syntax.
    assert sanitize_lucene_query(raw) == sanitized


@pytest.mark.anyio
async def test_search_memories_hybrid_surfaces_lexical_only_keyword_hit() -> None:
    # Given: dense ranking misses the keyword fact that BM25 finds.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Cartman prefers cheesy poofs", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [
        _fact_row("the Zephyrine-9 prototype shipped", memory_id=_MEMORY_ID_TWO),
    ]
    backend = _backend(client, hybrid_enabled=True)

    # When: the caller searches for the exact keyword.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="Zephyrine-9"),
    )

    # Then: the lexical-only hit fuses in with score 0.0 (RRF admitted it,
    # not vector similarity) while the dense hit keeps its vector score, the
    # index bootstrap ran once, and the lexical query was sanitized and
    # narrowed by the same tenant/user metadata fragments as other reads.
    assert [(result.content, result.score) for result in response.results] == [
        ("Cartman prefers cheesy poofs", 0.9),
        ("the Zephyrine-9 prototype shipped", 0.0),
    ]
    graph = client.graph
    assert graph is not None
    assert graph.writes == [(CREATE_FACT_OBJECT_FULLTEXT_INDEX_CYPHER, {})]
    statement, params = client.query.calls[0]
    assert f"db.index.fulltext.queryNodes('{FACT_OBJECT_FULLTEXT_INDEX}'" in statement
    assert params == {
        "query": "Zephyrine\\-9",
        "scope_fragments": ['"tenant_id": "bromigos"', '"user_id": "789"'],
        "candidate_limit": 100,
    }


@pytest.mark.anyio
async def test_search_memories_hybrid_creates_index_once_per_backend() -> None:
    # Given: a hybrid-enabled backend serving two searches.
    client = FakeMemoryClient()
    client.query.rows = [_fact_row("keyword note", memory_id=_MEMORY_ID_ONE)]
    backend = _backend(client, hybrid_enabled=True)

    # When: two searches run against the same backend instance.
    _ = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="keyword"),
    )
    _ = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="keyword"),
    )

    # Then: the idempotent index bootstrap ran exactly once.
    graph = client.graph
    assert graph is not None
    assert graph.writes == [(CREATE_FACT_OBJECT_FULLTEXT_INDEX_CYPHER, {})]
    assert len(client.query.calls) == 2


@pytest.mark.anyio
async def test_search_memories_hybrid_flag_off_keeps_todays_bytes() -> None:
    # Given: the default settings posture (hybrid off) and a lexical row
    # that would fuse in if the lexical leg ever ran.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Cartman prefers cheesy poofs", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [_fact_row("lexical decoy", memory_id=_MEMORY_ID_TWO)]
    backend = _backend(client, hybrid_enabled=False)

    # When: the caller searches.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="poofs"),
    )

    # Then: no lexical query and no index bootstrap run, and the response is
    # byte-identical to the pre-hybrid contract.
    assert client.query.calls == []
    graph = client.graph
    assert graph is not None
    assert graph.writes == []
    assert response.model_dump_json() == (
        '{"results":[{"memory_id":"00000000-0000-0000-0000-0000000000aa",'
        '"content":"Cartman prefers cheesy poofs","score":0.9,'
        '"metadata":{"topic":"snacks"},'
        '"created_at":"2026-06-27T01:02:03Z","updated_at":null}]}'
    )


@pytest.mark.anyio
async def test_search_memories_hybrid_fulltext_failure_degrades_to_dense(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the full-text query raises while the dense ranking has results.
    client = FakeMemoryClient(query=FailingCypherQuery())
    client.long_term.search_results = [
        _fact("dense answer", memory_id=_MEMORY_ID_ONE),
    ]
    backend = _backend(client, hybrid_enabled=True)

    # When: the caller searches with hybrid enabled.
    with caplog.at_level(logging.WARNING):
        response = await backend.search_memories(
            MemorySearchRequest(scope=_scope(), query="anything"),
        )

    # Then: the read never fails - the dense-only ranking answers and a
    # structured warning records the degradation.
    assert [result.content for result in response.results] == ["dense answer"]
    assert "lexical memory search failed" in caplog.text


@pytest.mark.anyio
async def test_search_memories_hybrid_missing_write_handle_degrades_to_dense(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the SDK exposes no graph write handle for the index bootstrap.
    client = FakeMemoryClient(graph=None)
    client.long_term.search_results = [
        _fact("dense answer", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [_fact_row("lexical hit", memory_id=_MEMORY_ID_TWO)]
    backend = _backend(client, hybrid_enabled=True)

    # When: the caller searches with hybrid enabled.
    with caplog.at_level(logging.WARNING):
        response = await backend.search_memories(
            MemorySearchRequest(scope=_scope(), query="anything"),
        )

    # Then: the bootstrap failure degrades to the dense-only ranking.
    assert [result.content for result in response.results] == ["dense answer"]
    assert "lexical memory search failed" in caplog.text


@pytest.mark.anyio
async def test_search_memories_lexical_path_keeps_scope_isolation() -> None:
    # Given: the lexical query returns cross-tenant and cross-user rows in
    # addition to the in-scope keyword hit.
    client = FakeMemoryClient()
    client.query.rows = [
        _fact_row(
            "other tenant secret",
            memory_id=_MEMORY_ID_TWO,
            metadata=_scope_metadata() | {"tenant_id": "other-tenant"},
        ),
        _fact_row(
            "other user secret",
            memory_id=_MEMORY_ID_THREE,
            metadata=_scope_metadata() | {"user_id": "666"},
        ),
        _fact_row("in-scope keyword note", memory_id=_MEMORY_ID_ONE),
    ]
    backend = _backend(client, hybrid_enabled=True)

    # When: the caller searches with hybrid enabled.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="keyword"),
    )

    # Then: the gateway scope re-check drops the cross-scope rows even though
    # the storage query returned them, so they can never fuse in.
    assert [result.content for result in response.results] == [
        "in-scope keyword note",
    ]


@pytest.mark.anyio
async def test_search_memories_min_score_applies_to_the_vector_score_only() -> None:
    # Given: one dense hit above the cut and one lexical-only hit (vector
    # score 0.0 by contract).
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("dense answer", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [_fact_row("lexical answer", memory_id=_MEMORY_ID_TWO)]
    backend = _backend(client, hybrid_enabled=True)

    # When: the caller searches with a minimum score.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="anything", min_score=0.5),
    )

    # Then: min_score keeps judging the vector score, so the lexical-only hit
    # (0.0) does not pass the cut.
    assert [result.content for result in response.results] == ["dense answer"]


@pytest.mark.anyio
async def test_context_hybrid_surfaces_lexical_only_fact() -> None:
    # Given: dense ranking misses the keyword fact and a cross-tenant lexical
    # row rides along.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Cartman prefers cheesy poofs", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [
        _fact_row("the Zephyrine-9 prototype shipped", memory_id=_MEMORY_ID_TWO),
        _fact_row(
            "other tenant secret",
            memory_id=_MEMORY_ID_THREE,
            metadata=_scope_metadata() | {"tenant_id": "other-tenant"},
        ),
    ]
    backend = _backend(client, hybrid_enabled=True)

    # When: combined memory context is assembled with a query.
    response = await backend.get_memory_context(_context_request())

    # Then: the lexical-only fact renders alongside the dense fact while the
    # cross-tenant row never leaves the service.
    content = response.sections[0].content
    assert "- [2026-06-27] Cartman prefers cheesy poofs" in content
    assert "- [2026-06-27] the Zephyrine-9 prototype shipped" in content
    assert "other tenant secret" not in content


@pytest.mark.anyio
async def test_context_hybrid_flag_off_keeps_todays_bytes() -> None:
    # Given: the default settings posture (hybrid off).
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Cartman prefers cheesy poofs", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [_fact_row("lexical decoy", memory_id=_MEMORY_ID_TWO)]
    backend = _backend(client, hybrid_enabled=False)

    # When: combined memory context is assembled.
    response = await backend.get_memory_context(_context_request())

    # Then: the lexical leg never runs and the response is byte-identical to
    # the pre-hybrid contract.
    assert client.query.calls == []
    assert response.model_dump_json() == (
        '{"sections":[{"source":"long_term_facts",'
        '"content":"### Long-Term Facts\\n'
        '- [2026-06-27] Cartman prefers cheesy poofs","facts":[]}]}'
    )


@dataclass(slots=True)
class FakeLongTermMemory:
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
        message = "full-text index unavailable"
        raise Neo4jError(message)


@dataclass(slots=True)
class FakeGraphWriter:
    writes: list[tuple[str, dict[str, JsonValue] | None]] = field(default_factory=list)

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.writes.append((query, parameters))
        return []


@dataclass(slots=True)
class FakeMemoryClient:
    long_term: FakeLongTermMemory = field(default_factory=FakeLongTermMemory)
    query: FakeCypherQuery | FailingCypherQuery = field(
        default_factory=FakeCypherQuery,
    )
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


def _backend(
    client: FakeMemoryClient,
    *,
    hybrid_enabled: bool,
) -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        Settings(gnosis_hybrid_retrieval_enabled=hybrid_enabled),
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


def _stored_memory(
    memory_id: str,
    content: str,
    *,
    similarity: float | None = None,
) -> StoredMemory:
    metadata = _scope_metadata()
    if similarity is not None:
        metadata["similarity"] = similarity
    return StoredMemory(
        memory_id=memory_id,
        subject="bromigos:discord:private_user:pc-principal:789",
        predicate="memory",
        content=content,
        metadata=metadata,
        created_at="2026-06-27T01:02:03+00:00",
        updated_at=None,
    )


def _fact(content: str, *, memory_id: str, similarity: float = 0.9) -> Fact:
    return Fact(
        id=UUID(memory_id),
        subject="bromigos:discord:private_user:pc-principal:789",
        predicate="memory",
        object=content,
        created_at=datetime(2026, 6, 27, 1, 2, 3, tzinfo=UTC),
        metadata=_scope_metadata() | {"similarity": similarity, "topic": "snacks"},
    )


def _fact_row(
    content: str,
    *,
    memory_id: str,
    metadata: dict[str, JsonValue] | None = None,
) -> JsonObject:
    return {
        "id": memory_id,
        "subject": "bromigos:discord:private_user:pc-principal:789",
        "predicate": "memory",
        "object": content,
        "metadata": json.dumps(metadata or _scope_metadata()),
        "created_at": "2026-06-27T01:02:03+00:00",
        "updated_at": None,
    }
