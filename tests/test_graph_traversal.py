"""Entity-anchored graph traversal behind GNOSIS_GRAPH_TRAVERSAL_ENABLED (T1).

Covers seed-phrase generation (normalization parity with entity writes,
possessives, punctuation, the bounded parameter list), the traversal leg of
context assembly (provenance facts fuse in, hold the reserved graph budget
slots, and re-check scope), dense-only degradation on read failures, and the
pinned byte-identical contract with the flag off.
"""

import json
import logging
from dataclasses import dataclass, field
from os import environ
from typing import Self, cast

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

from datetime import UTC, datetime  # noqa: E402
from uuid import UUID  # noqa: E402

from neo4j.exceptions import Neo4jError  # noqa: E402
from neo4j_agent_memory import MemorySettings  # noqa: E402
from neo4j_agent_memory.memory.long_term import Fact  # noqa: E402

from gnosis.backend import (  # noqa: E402
    MemoryClientContext,
    Neo4jAgentMemoryBackend,
)
from gnosis.entity_traversal import query_seed_candidates  # noqa: E402
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
    MemoryVisibility,
)
from gnosis.settings import Settings  # noqa: E402

_MEMORY_ID_ONE = "00000000-0000-0000-0000-0000000000aa"
_MEMORY_ID_TWO = "00000000-0000-0000-0000-0000000000bb"
_MEMORY_ID_THREE = "00000000-0000-0000-0000-0000000000cc"


def test_seed_candidates_pin_single_and_multi_word_entities() -> None:
    # Given/When: a benchmark-shaped multi-hop query.
    seeds = query_seed_candidates(
        "When did Caroline go to the LGBTQ support group?",
    )

    # Then: both the one-word mention and the exact multi-word entity name
    # appear, normalized exactly like entity names at write time.
    assert "caroline" in seeds
    assert "lgbtq support group" in seeds


def test_seed_candidates_strip_possessives_and_edge_punctuation() -> None:
    # Given/When: possessive and punctuated mentions.
    seeds = query_seed_candidates("What did Caroline's grandma think of Sweden?")

    # Then: the possessive contributes its bare form for the "caroline" node
    # and its raw form for the "caroline's grandma" node, and the trailing
    # question mark strips off the final mention.
    assert "caroline" in seeds
    assert "caroline's grandma" in seeds
    assert "sweden" in seeds


def test_seed_candidates_keep_interior_punctuation_variants() -> None:
    # Given/When: an entity name with interior punctuation.
    seeds = query_seed_candidates("Did Melanie attend the LGBTQ+ pride parade?")

    # Then: the raw variant preserves the plus sign so it pins the stored
    # "lgbtq+ pride parade" node by equality.
    assert "lgbtq+ pride parade" in seeds


def test_seed_candidates_empty_query_pins_nothing() -> None:
    assert query_seed_candidates("") == []
    assert query_seed_candidates("   ") == []


def test_seed_candidates_are_bounded() -> None:
    # Given/When: a pathologically long query.
    seeds = query_seed_candidates(" ".join(f"word{index}" for index in range(300)))

    # Then: the parameter list stays capped.
    assert len(seeds) <= 128


@pytest.mark.anyio
async def test_context_traversal_fuses_provenance_facts_when_enabled() -> None:
    # Given: dense retrieval finds one fact while the graph traversal read
    # returns the bridge-hop provenance fact dense ranking never surfaced.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Caroline went to a support group", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [
        _fact_row(
            "Caroline's support group meets at the Elm Street center",
            memory_id=_MEMORY_ID_TWO,
        ),
    ]
    backend = _backend(client, traversal_enabled=True)

    # When: context is assembled with a query naming an entity.
    response = await backend.get_memory_context(
        _context_request("Where does Caroline's support group meet?"),
    )

    # Then: the traversal fact renders alongside the dense fact, and exactly
    # one scope-pinned traversal read ran with the query's seeds.
    content = response.sections[0].content
    assert "Caroline went to a support group" in content
    assert "Caroline's support group meets at the Elm Street center" in content
    assert len(client.query.calls) == 1
    statement, params = client.query.calls[0]
    assert "RELATES*1..2" in statement
    assert params is not None
    assert params["tenant_id"] == "nolgia"
    assert params["user_id"] == "789"
    seeds = cast("list[str]", params["seeds"])
    assert "caroline" in seeds
    assert params["scope_fragments"] == [
        '"tenant_id": "nolgia"',
        '"agent_id": "nolgia-agent"',
        '"user_id": "789"',
        '"visibility": "private_user"',
    ]


@pytest.mark.anyio
async def test_context_traversal_flag_off_keeps_todays_bytes() -> None:
    # Given: the default settings posture (traversal off) and a traversal row
    # that would fuse in if the leg ever ran.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Caroline went to a support group", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [_fact_row("traversal decoy", memory_id=_MEMORY_ID_TWO)]
    backend = _backend(client, traversal_enabled=False)

    # When: context is assembled.
    response = await backend.get_memory_context(
        _context_request("Where does Caroline's support group meet?"),
    )

    # Then: no traversal read runs and the decoy never renders.
    assert client.query.calls == []
    assert "traversal decoy" not in response.sections[0].content


@pytest.mark.anyio
async def test_context_traversal_failure_degrades_to_dense_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the traversal read raises while the dense ranking has results.
    client = FakeMemoryClient(query=FailingCypherQuery())
    client.long_term.search_results = [
        _fact("dense answer", memory_id=_MEMORY_ID_ONE),
    ]
    backend = _backend(client, traversal_enabled=True)

    # When: context is assembled.
    with caplog.at_level(logging.WARNING):
        response = await backend.get_memory_context(
            _context_request("Where does Caroline's support group meet?"),
        )

    # Then: the read never fails - the dense-only context answers and a
    # structured warning records the degradation.
    assert "dense answer" in response.sections[0].content
    assert "entity traversal failed" in caplog.text


@pytest.mark.anyio
async def test_context_traversal_keeps_scope_isolation() -> None:
    # Given: the traversal read returns a cross-tenant row alongside the
    # in-scope provenance fact.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("dense answer", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [
        _fact_row(
            "other tenant secret",
            memory_id=_MEMORY_ID_THREE,
            metadata=_scope_metadata() | {"tenant_id": "other-tenant"},
        ),
        _fact_row("in-scope traversal fact", memory_id=_MEMORY_ID_TWO),
    ]
    backend = _backend(client, traversal_enabled=True)

    # When: context is assembled.
    response = await backend.get_memory_context(
        _context_request("Where does Caroline's support group meet?"),
    )

    # Then: the gateway scope re-check drops the cross-scope row even though
    # the storage query returned it.
    content = response.sections[0].content
    assert "in-scope traversal fact" in content
    assert "other tenant secret" not in content


@pytest.mark.anyio
async def test_context_traversal_survives_item_budget_over_full_dense_pool() -> None:
    # Given: dense retrieval fills the pool past the item budget and the
    # traversal read yields one provenance fact ranked after all of them.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact(
            f"dense filler fact number {index}",
            memory_id=f"00000000-0000-0000-0000-000000000{index:03d}",
        )
        for index in range(30)
    ]
    client.query.rows = [
        _fact_row("bridge-hop provenance fact", memory_id=_MEMORY_ID_TWO),
    ]
    backend = _backend(client, traversal_enabled=True)

    # When: context is assembled with a budget smaller than the dense pool.
    response = await backend.get_memory_context(
        _context_request("Where does Caroline's support group meet?"),
    )

    # Then: the traversal fact holds a reserved graph slot instead of being
    # cut, and the budget is respected.
    content = response.sections[0].content
    assert "bridge-hop provenance fact" in content
    fact_lines = [line for line in content.splitlines() if line.startswith("- ")]
    assert len(fact_lines) == 8


@dataclass(slots=True)
class FakeLongTermMemory:
    search_results: list[Fact] = field(default_factory=list)

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[Fact]:
        _ = (query, limit, threshold)
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
        message = "traversal read unavailable"
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
    traversal_enabled: bool,
) -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        Settings(gnosis_graph_traversal_enabled=traversal_enabled),
        memory_client_factory=FakeMemoryClientFactory(client),
        graph_store=FakeGraphStore(),
    )


def _context_request(query: str) -> MemoryContextRequest:
    return MemoryContextRequest(
        scope=_scope(),
        query=query,
        include_short_term=False,
        include_reasoning=False,
        include_graph=False,
        max_items=8,
    )


def _scope() -> MemoryScope:
    return MemoryScope(
        tenant_id="nolgia",
        space_id="discord",
        agent_id="nolgia-agent",
        session_id="guild:123:channel:456",
        user_id="789",
        visibility=MemoryVisibility.PRIVATE_USER,
    )


def _scope_metadata() -> dict[str, JsonValue]:
    return {
        "tenant_id": "nolgia",
        "space_id": "discord",
        "agent_id": "nolgia-agent",
        "session_id": "guild:123:channel:456",
        "user_id": "789",
        "visibility": "private_user",
    }


def _fact(content: str, *, memory_id: str, similarity: float = 0.9) -> Fact:
    return Fact(
        id=UUID(memory_id),
        subject="nolgia:discord:private_user:nolgia-agent:789",
        predicate="memory",
        object=content,
        created_at=datetime(2026, 6, 27, 1, 2, 3, tzinfo=UTC),
        metadata=_scope_metadata() | {"similarity": similarity},
    )


def _fact_row(
    content: str,
    *,
    memory_id: str,
    metadata: dict[str, JsonValue] | None = None,
) -> JsonObject:
    return {
        "id": memory_id,
        "subject": "nolgia:discord:private_user:nolgia-agent:789",
        "predicate": "fact",
        "object": content,
        "metadata": json.dumps(metadata or _scope_metadata()),
        "created_at": "2026-06-27T01:02:03+00:00",
        "updated_at": None,
    }
