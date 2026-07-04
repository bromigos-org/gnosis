"""Directed bridge traversal behind GNOSIS_BRIDGE_TRAVERSAL_ENABLED (T1-directed).

Covers namer-reply parsing (line/comma/bullet noise, NONE, the bounded name
list), the directed hop of context assembly (the namer sees hop-1's dense
evidence, query-named entities are filtered from the bridge list, mention
facts fuse in holding the reserved graph budget slots, scope re-checks), and
dense-only degradation when the namer or the read fails.
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
from gnosis.bridge_traversal import parse_bridge_names  # noqa: E402
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


def test_parse_bridge_names_accepts_lines_commas_and_bullets() -> None:
    assert parse_bridge_names("Rob") == ["Rob"]
    assert parse_bridge_names("Rob\nWest County") == ["Rob", "West County"]
    assert parse_bridge_names("Rob, West County") == ["Rob", "West County"]
    assert parse_bridge_names("- Rob\n- West County") == ["Rob", "West County"]
    assert parse_bridge_names('1. "Rob"\n2. \u2018West County\u2019') == [
        "Rob",
        "West County",
    ]


def test_parse_bridge_names_rejects_none_and_empty() -> None:
    assert parse_bridge_names(None) == []
    assert parse_bridge_names("") == []
    assert parse_bridge_names("NONE") == []
    assert parse_bridge_names("none") == []
    assert parse_bridge_names("  \n ") == []


def test_parse_bridge_names_dedupes_and_caps() -> None:
    # Given/When: a reply repeating names beyond the cap.
    names = parse_bridge_names("Rob\nrob\nWest County\nRome\nExtra One\nExtra Two")

    # Then: case-insensitive dedupe in reply order, capped at three.
    assert names == ["Rob", "West County", "Rome"]


@pytest.mark.anyio
async def test_context_bridge_hop_fuses_mention_facts_when_enabled() -> None:
    # Given: dense retrieval reveals "a colleague" while the namer resolves
    # the bridge entity and the mention read returns hop-2's fact.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("John went to yoga with a colleague", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [
        _fact_row(
            "Rob invited John to a beginner's yoga class", memory_id=_MEMORY_ID_TWO
        ),
    ]
    namer = RecordingBridgeNamer(reply="Rob")
    backend = _backend(client, bridge_enabled=True, namer=namer)

    # When: context is assembled with a query that never names the bridge.
    response = await backend.get_memory_context(
        _context_request("Who did John go to yoga with?"),
    )

    # Then: the bridge fact renders alongside the dense fact; the namer saw
    # the query plus hop-1's rendered evidence; the mention read was pinned
    # to the caller's scope with the normalized bridge name.
    content = response.sections[0].content
    assert "John went to yoga with a colleague" in content
    assert "Rob invited John to a beginner's yoga class" in content
    assert len(namer.calls) == 1
    query, evidence = namer.calls[0]
    assert query == "Who did John go to yoga with?"
    assert any("colleague" in line for line in evidence)
    assert len(client.query.calls) == 1
    statement, params = client.query.calls[0]
    assert "MENTIONS" in statement
    assert params is not None
    assert params["tenant_id"] == "bromigos"
    assert params["user_id"] == "789"
    assert params["bridges"] == ["rob"]


@pytest.mark.anyio
async def test_context_bridge_hop_flag_off_keeps_todays_bytes() -> None:
    # Given: the default settings posture (bridge hop off) and a decoy row.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("John went to yoga with a colleague", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [_fact_row("bridge decoy", memory_id=_MEMORY_ID_TWO)]
    namer = RecordingBridgeNamer(reply="Rob")
    backend = _backend(client, bridge_enabled=False, namer=namer)

    # When: context is assembled.
    response = await backend.get_memory_context(
        _context_request("Who did John go to yoga with?"),
    )

    # Then: no namer call, no mention read, no decoy.
    assert namer.calls == []
    assert client.query.calls == []
    assert "bridge decoy" not in response.sections[0].content


@pytest.mark.anyio
async def test_context_bridge_hop_filters_query_named_entities() -> None:
    # Given: the namer parrots an entity the query itself already names.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("John went to yoga with a colleague", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [_fact_row("decoy", memory_id=_MEMORY_ID_TWO)]
    namer = RecordingBridgeNamer(reply="John")
    backend = _backend(client, bridge_enabled=True, namer=namer)

    # When: context is assembled.
    response = await backend.get_memory_context(
        _context_request("Who did John go to yoga with?"),
    )

    # Then: the parroted name is not a bridge - no mention read runs.
    assert client.query.calls == []
    assert "decoy" not in response.sections[0].content


@pytest.mark.anyio
async def test_context_bridge_hop_no_bridge_reply_skips_the_read() -> None:
    # Given: the namer finds no bridge entity.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("dense answer", memory_id=_MEMORY_ID_ONE),
    ]
    client.query.rows = [_fact_row("decoy", memory_id=_MEMORY_ID_TWO)]
    backend = _backend(
        client,
        bridge_enabled=True,
        namer=RecordingBridgeNamer(reply="NONE"),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        _context_request("Who did John go to yoga with?"),
    )

    # Then: no mention read runs and the dense context stands alone.
    assert client.query.calls == []
    assert "dense answer" in response.sections[0].content


@pytest.mark.anyio
async def test_context_bridge_hop_namer_failure_degrades_to_dense_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the namer call raises.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("dense answer", memory_id=_MEMORY_ID_ONE),
    ]
    backend = _backend(client, bridge_enabled=True, namer=FailingBridgeNamer())

    # When: context is assembled.
    with caplog.at_level(logging.WARNING):
        response = await backend.get_memory_context(
            _context_request("Who did John go to yoga with?"),
        )

    # Then: the read never fails - dense-only context answers with a warning.
    assert "dense answer" in response.sections[0].content
    assert "bridge namer failed" in caplog.text


@pytest.mark.anyio
async def test_context_bridge_hop_read_failure_degrades_to_dense_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the namer answers but the mention read raises.
    client = FakeMemoryClient(query=FailingCypherQuery())
    client.long_term.search_results = [
        _fact("dense answer", memory_id=_MEMORY_ID_ONE),
    ]
    backend = _backend(
        client,
        bridge_enabled=True,
        namer=RecordingBridgeNamer(reply="Rob"),
    )

    # When: context is assembled.
    with caplog.at_level(logging.WARNING):
        response = await backend.get_memory_context(
            _context_request("Who did John go to yoga with?"),
        )

    # Then: dense-only context answers with a structured warning.
    assert "dense answer" in response.sections[0].content
    assert "bridge traversal read failed" in caplog.text


@pytest.mark.anyio
async def test_context_bridge_hop_keeps_scope_isolation() -> None:
    # Given: the mention read returns a cross-tenant row alongside the
    # in-scope bridge fact.
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
        _fact_row("in-scope bridge fact", memory_id=_MEMORY_ID_TWO),
    ]
    backend = _backend(
        client,
        bridge_enabled=True,
        namer=RecordingBridgeNamer(reply="Rob"),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        _context_request("Who did John go to yoga with?"),
    )

    # Then: the gateway scope re-check drops the cross-scope row.
    content = response.sections[0].content
    assert "in-scope bridge fact" in content
    assert "other tenant secret" not in content


@pytest.mark.anyio
async def test_context_bridge_hop_survives_item_budget_over_full_dense_pool() -> None:
    # Given: dense retrieval fills the pool past the item budget and the
    # bridge hop yields one mention fact ranked after all of them.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact(
            f"dense filler fact number {index}",
            memory_id=f"00000000-0000-0000-0000-000000000{index:03d}",
        )
        for index in range(30)
    ]
    client.query.rows = [
        _fact_row("bridge mention fact", memory_id=_MEMORY_ID_TWO),
    ]
    backend = _backend(
        client,
        bridge_enabled=True,
        namer=RecordingBridgeNamer(reply="Rob"),
    )

    # When: context is assembled with a budget smaller than the dense pool.
    response = await backend.get_memory_context(
        _context_request("Who did John go to yoga with?"),
    )

    # Then: the bridge fact holds a reserved graph slot instead of being
    # cut, and the budget is respected.
    content = response.sections[0].content
    assert "bridge mention fact" in content
    fact_lines = [line for line in content.splitlines() if line.startswith("- ")]
    assert len(fact_lines) == 8


@dataclass(slots=True)
class RecordingBridgeNamer:
    reply: str
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    async def name_bridges(self, query: str, evidence: list[str]) -> str | None:
        self.calls.append((query, list(evidence)))
        return self.reply


@dataclass(slots=True)
class FailingBridgeNamer:
    async def name_bridges(self, query: str, evidence: list[str]) -> str | None:
        _ = (query, evidence)
        message = "bridge namer unavailable"
        raise RuntimeError(message)


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
        message = "mention read unavailable"
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
    bridge_enabled: bool,
    namer: RecordingBridgeNamer | FailingBridgeNamer,
) -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        Settings(gnosis_bridge_traversal_enabled=bridge_enabled),
        memory_client_factory=FakeMemoryClientFactory(client),
        graph_store=FakeGraphStore(),
        bridge_namer=namer,
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
        "subject": "bromigos:discord:private_user:pc-principal:789",
        "predicate": "fact",
        "object": content,
        "metadata": json.dumps(metadata or _scope_metadata()),
        "created_at": "2026-06-27T01:02:03+00:00",
        "updated_at": None,
    }
