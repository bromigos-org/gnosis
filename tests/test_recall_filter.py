import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from os import environ
from typing import TYPE_CHECKING, Self, cast
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

import httpx  # noqa: E402
import httpx2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from neo4j_agent_memory import MemorySettings  # noqa: E402
from neo4j_agent_memory.memory.long_term import Fact  # noqa: E402
from openai import APIConnectionError  # noqa: E402

from gnosis.backend import (  # noqa: E402
    MemoryClientContext,
    Neo4jAgentMemoryBackend,
    RecallFilteringBackend,
)
from gnosis.main import create_app  # noqa: E402
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
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryVisibility,
)
from gnosis.recall_filter import (  # noqa: E402
    RecallSelection,
    keep_relevant_candidates,
)
from gnosis.settings import Settings  # noqa: E402

if TYPE_CHECKING:
    from gnosis.backend import MemoryBackend

_MEMORY_ID_ONE = "00000000-0000-0000-0000-0000000000aa"
_MEMORY_ID_TWO = "00000000-0000-0000-0000-0000000000bb"
_MEMORY_ID_THREE = "00000000-0000-0000-0000-0000000000cc"
_FEDERATION_TOKEN = "federation-inbound-token"
_PEER_TOKEN = "peer-outbound-token"
_PEER_BASE_URL = "http://gnosis-partner.gnosis-partner.svc.cluster.local:8080"


@pytest.mark.anyio
async def test_keep_relevant_candidates_keeps_original_rank_order() -> None:
    # Given: a filter that selects candidates out of order with noise indices:
    # duplicates, zero, negatives, and numbers beyond the candidate list.
    recall_filter = RecordingRecallFilter(
        selection=RecallSelection(kept_indices=[3, 1, 3, 99, 0, -2]),
    )

    # When: four ranked items are screened.
    kept = await keep_relevant_candidates(
        recall_filter,
        query="which ones?",
        items=["alpha", "beta", "gamma", "delta"],
        render=str,
        max_candidates=10,
    )

    # Then: invalid indices are ignored and kept items preserve rank order.
    assert kept == ["alpha", "gamma"]
    assert recall_filter.calls == [
        RecallCall(
            query="which ones?",
            candidates=["alpha", "beta", "gamma", "delta"],
        ),
    ]


@pytest.mark.anyio
async def test_keep_relevant_candidates_screens_only_the_top_window() -> None:
    # Given: more ranked items than the candidate window.
    recall_filter = RecordingRecallFilter(
        selection=RecallSelection(kept_indices=[2, 3]),
    )

    # When: five items are screened with a three-item window.
    kept = await keep_relevant_candidates(
        recall_filter,
        query="which ones?",
        items=["alpha", "beta", "gamma", "delta", "epsilon"],
        render=str,
        max_candidates=3,
    )

    # Then: only the window reaches the filter and below-window items drop.
    assert kept == ["beta", "gamma"]
    assert recall_filter.calls[0].candidates == ["alpha", "beta", "gamma"]


@pytest.mark.anyio
async def test_keep_relevant_candidates_empty_selection_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a filter misfire that keeps nothing.
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[]))

    # When: items are screened.
    with caplog.at_level(logging.WARNING, logger="gnosis.recall_filter"):
        kept = await keep_relevant_candidates(
            recall_filter,
            query="which ones?",
            items=["alpha", "beta"],
            render=str,
            max_candidates=10,
        )

    # Then: the unfiltered ranking survives and the misfire is logged.
    assert kept == ["alpha", "beta"]
    assert "kept no valid candidates" in caplog.text


@pytest.mark.anyio
async def test_keep_relevant_candidates_missing_parse_falls_back() -> None:
    # Given: a filter whose structured output produced no content.
    recall_filter = RecordingRecallFilter(selection=None)

    # When: items are screened.
    kept = await keep_relevant_candidates(
        recall_filter,
        query="which ones?",
        items=["alpha", "beta"],
        render=str,
        max_candidates=10,
    )

    # Then: the unfiltered ranking survives.
    assert kept == ["alpha", "beta"]


@pytest.mark.anyio
async def test_keep_relevant_candidates_llm_failure_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a filter whose LLM call fails at the transport.
    recall_filter = FailingRecallFilter()

    # When: items are screened.
    with caplog.at_level(logging.WARNING, logger="gnosis.recall_filter"):
        kept = await keep_relevant_candidates(
            recall_filter,
            query="which ones?",
            items=["alpha", "beta"],
            render=str,
            max_candidates=10,
        )

    # Then: the failure degrades to the unfiltered ranking and is logged.
    assert kept == ["alpha", "beta"]
    assert "recall filter failed" in caplog.text


@pytest.mark.anyio
async def test_keep_relevant_candidates_emits_observability_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a filter that keeps one of three candidates.
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[2]))

    # When: items are screened.
    with caplog.at_level(logging.INFO, logger="gnosis.recall_filter"):
        _ = await keep_relevant_candidates(
            recall_filter,
            query="which ones?",
            items=["alpha", "beta", "gamma"],
            render=str,
            max_candidates=10,
        )

    # Then: the structured log carries candidates_in and kept counts.
    applied = next(
        record for record in caplog.records if record.message == "recall filter applied"
    )
    counts = cast("dict[str, int]", vars(applied))
    assert counts["candidates_in"] == 3
    assert counts["kept"] == 1


@pytest.mark.anyio
async def test_search_memories_recall_filter_removes_wrong_content() -> None:
    # Given: three in-scope ranked candidates and a filter keeping the second.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Cartman prefers cheesy poofs", memory_id=_MEMORY_ID_ONE),
        _fact("Maria adopted a golden retriever", memory_id=_MEMORY_ID_TWO),
        _fact("the weather was rainy", memory_id=_MEMORY_ID_THREE),
    ]
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[2]))
    backend = _backend(client, recall_filter)

    # When: the caller searches with the filter enabled.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="what pet did Maria adopt?"),
    )

    # Then: only the kept candidate returns, and the filter saw all three as
    # dated one-line renderings alongside the query.
    assert [result.content for result in response.results] == [
        "Maria adopted a golden retriever",
    ]
    assert recall_filter.calls == [
        RecallCall(
            query="what pet did Maria adopt?",
            candidates=[
                "- [2026-06-27] Cartman prefers cheesy poofs",
                "- [2026-06-27] Maria adopted a golden retriever",
                "- [2026-06-27] the weather was rainy",
            ],
        ),
    ]


@pytest.mark.anyio
async def test_search_memories_recall_filter_screens_beyond_the_limit() -> None:
    # Given: more scoped candidates than the caller's limit.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("decoy one", memory_id=_MEMORY_ID_ONE),
        _fact("decoy two", memory_id=_MEMORY_ID_TWO),
        _fact("the relevant fact", memory_id=_MEMORY_ID_THREE),
    ]
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[3]))
    backend = _backend(client, recall_filter)

    # When: the caller searches with limit below the candidate count.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="the relevant thing?", limit=1),
    )

    # Then: the filter screened the full candidate window, so a relevant fact
    # ranked below the limit can still win the budget.
    assert [result.content for result in response.results] == ["the relevant fact"]
    assert len(recall_filter.calls[0].candidates) == 3


@pytest.mark.anyio
async def test_search_memories_recall_filter_failure_degrades_to_ranking() -> None:
    # Given: a filter whose LLM call fails.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("first", memory_id=_MEMORY_ID_ONE),
        _fact("second", memory_id=_MEMORY_ID_TWO),
    ]
    backend = _backend(client, FailingRecallFilter())

    # When: the caller searches with the filter enabled.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="anything"),
    )

    # Then: the unfiltered ranking answers.
    assert [result.content for result in response.results] == ["first", "second"]


@pytest.mark.anyio
async def test_search_memories_disabled_flag_keeps_todays_bytes() -> None:
    # Given: the default settings posture (filter off) and a filter double
    # that would keep nothing if it were ever consulted.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Cartman prefers cheesy poofs", memory_id=_MEMORY_ID_ONE),
    ]
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[]))
    backend = _backend(client, recall_filter, enabled=False)

    # When: the caller searches.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="poofs"),
    )

    # Then: the filter never runs and the response is byte-identical to the
    # pre-filter contract.
    assert recall_filter.calls == []
    assert response.model_dump_json() == (
        '{"results":[{"memory_id":"00000000-0000-0000-0000-0000000000aa",'
        '"content":"Cartman prefers cheesy poofs","score":0.9,'
        '"metadata":{"topic":"snacks"},'
        '"created_at":"2026-06-27T01:02:03Z","updated_at":null}]}'
    )


@pytest.mark.anyio
async def test_search_memories_with_peers_defers_filtering_to_the_route() -> None:
    # Given: a federated search request naming a peer.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("local fact", memory_id=_MEMORY_ID_ONE),
    ]
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[]))
    backend = _backend(client, recall_filter)

    # When: the backend handles the local leg of the federated search.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="poofs", peers=["partner"]),
    )

    # Then: the backend leaves filtering to the route's merged-set pass, so
    # the request costs one LLM call in total.
    assert recall_filter.calls == []
    assert [result.content for result in response.results] == ["local fact"]


@pytest.mark.anyio
async def test_context_recall_filter_removes_wrong_content_facts() -> None:
    # Given: ranked long-term candidates where a decoy outranks the answer.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("the weather was rainy this morning", memory_id=_MEMORY_ID_ONE),
        _fact("Maria adopted a golden retriever", memory_id=_MEMORY_ID_TWO),
    ]
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[2]))
    backend = _backend(client, recall_filter)

    # When: combined context is assembled with the filter enabled.
    response = await backend.get_memory_context(_context_request(max_items=1))

    # Then: the kept fact wins the item budget and the decoy never renders.
    content = response.sections[0].content
    assert "Maria adopted a golden retriever" in content
    assert "rainy" not in content
    assert recall_filter.calls[0].query == "what pet did Maria adopt?"


@pytest.mark.anyio
async def test_context_recall_filter_screens_recency_fallback_too() -> None:
    # Given: no similarity ranking, so candidates come from the recency read.
    client = FakeMemoryClient()
    client.query.rows = [
        {"f": _fact_row("newest note", memory_id=_MEMORY_ID_ONE)},
        {"f": _fact_row("the relevant note", memory_id=_MEMORY_ID_TWO)},
    ]
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[2]))
    backend = _backend(client, recall_filter)

    # When: combined context is assembled.
    response = await backend.get_memory_context(_context_request())

    # Then: the filter screens the recency candidates as well.
    content = response.sections[0].content
    assert "the relevant note" in content
    assert "newest note" not in content


@pytest.mark.anyio
async def test_context_recall_filter_empty_selection_keeps_top_k() -> None:
    # Given: a filter misfire that keeps nothing.
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("first fact", memory_id=_MEMORY_ID_ONE),
        _fact("second fact", memory_id=_MEMORY_ID_TWO),
    ]
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[]))
    backend = _backend(client, recall_filter)

    # When: combined context is assembled.
    response = await backend.get_memory_context(_context_request())

    # Then: the unfiltered top-k renders; the filter alone never empties recall.
    content = response.sections[0].content
    assert "first fact" in content
    assert "second fact" in content


@pytest.mark.anyio
async def test_context_disabled_flag_keeps_todays_bytes() -> None:
    # Given: the default settings posture (filter off).
    client = FakeMemoryClient()
    client.long_term.search_results = [
        _fact("Cartman prefers cheesy poofs", memory_id=_MEMORY_ID_ONE),
    ]
    recall_filter = RecordingRecallFilter(selection=RecallSelection(kept_indices=[]))
    backend = _backend(client, recall_filter, enabled=False)

    # When: combined context is assembled.
    response = await backend.get_memory_context(_context_request())

    # Then: the filter never runs and the response is byte-identical to the
    # pre-filter contract.
    assert recall_filter.calls == []
    assert response.model_dump_json() == (
        '{"sections":[{"source":"long_term_facts",'
        '"content":"### Long-Term Facts\\n'
        '- [2026-06-27] Cartman prefers cheesy poofs","facts":[]}]}'
    )


def test_federated_search_filters_the_merged_result_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a peer with shareable results and a backend whose recall filter
    # keeps only the remote record.
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json=MemorySearchResponse(
                results=[_memory_record("peer-memory-1", "peer strong", score=0.95)],
            ).model_dump(mode="json"),
        )

    backend = FilterRecordingBackend(kept_memory_ids={"peer-memory-1"})
    client = _federated_app_client(
        monkeypatch,
        backend,
        transport=httpx2.MockTransport(handler),
    )

    # When: a service caller searches across the peer.
    response = client.post(
        "/v1/memories/search",
        headers={"Authorization": f"Bearer {environ['GNOSIS_TOKEN']}"},
        json={
            "scope": _scope().model_dump(mode="json"),
            "query": "what snacks?",
            "peers": ["partner"],
        },
    )

    # Then: the filter ran once over the merged local+remote set and its
    # verdict shapes the response.
    assert response.status_code == 200
    assert backend.filter_calls == [
        ("what snacks?", ["peer-memory-1", _MEMORY_ID_ONE]),
    ]
    merged = MemorySearchResponse.model_validate_json(response.content)
    assert [(result.memory_id, result.origin) for result in merged.results] == [
        ("peer-memory-1", "partner"),
    ]


@dataclass(frozen=True, slots=True)
class RecallCall:
    query: str
    candidates: list[str]


@dataclass(slots=True)
class RecordingRecallFilter:
    selection: RecallSelection | None = None
    calls: list[RecallCall] = field(default_factory=list)

    async def select_candidates(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> RecallSelection | None:
        self.calls.append(RecallCall(query=query, candidates=list(candidates)))
        return self.selection


@dataclass(slots=True)
class FailingRecallFilter:
    async def select_candidates(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> RecallSelection | None:
        _ = (query, candidates)
        raise APIConnectionError(request=httpx.Request("POST", "http://litellm.test"))


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


@dataclass(slots=True)
class FakeCypherQuery:
    rows: list[JsonObject] = field(default_factory=list)

    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        _ = (query, params)
        return list(self.rows)


@dataclass(slots=True)
class FakeMemoryClient:
    long_term: FakeLongTermMemory = field(default_factory=FakeLongTermMemory)
    query: FakeCypherQuery = field(default_factory=FakeCypherQuery)

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


@dataclass(slots=True)
class FilterRecordingBackend:
    """Just enough backend surface for the federated search route."""

    kept_memory_ids: set[str] = field(default_factory=set)
    filter_calls: list[tuple[str, list[str]]] = field(default_factory=list)

    async def search_memories(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        _ = request
        return MemorySearchResponse(
            results=[_memory_record(_MEMORY_ID_ONE, "local fact", score=0.91)],
        )

    async def filter_recalled_memories(
        self,
        query: str,
        records: Sequence[MemoryRecord],
    ) -> list[MemoryRecord]:
        self.filter_calls.append(
            (query, [record.memory_id for record in records]),
        )
        return [
            record for record in records if record.memory_id in self.kept_memory_ids
        ]

    async def shutdown(self) -> None:
        return None


def _backend(
    client: FakeMemoryClient,
    recall_filter: RecordingRecallFilter | FailingRecallFilter,
    *,
    enabled: bool = True,
    candidates: int = 30,
) -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        Settings(
            gnosis_recall_filter_enabled=enabled,
            gnosis_recall_filter_candidates=candidates,
        ),
        memory_client_factory=FakeMemoryClientFactory(client),
        graph_store=FakeGraphStore(),
        recall_filter=recall_filter,
    )


def _federated_app_client(
    monkeypatch: pytest.MonkeyPatch,
    backend: FilterRecordingBackend,
    *,
    transport: httpx2.MockTransport,
) -> TestClient:
    monkeypatch.setenv(
        "GNOSIS_PEERS",
        json.dumps(
            [
                {
                    "name": "partner",
                    "base_url": _PEER_BASE_URL,
                    "direction": "both",
                    "remote_tenant_id": "partner",
                },
            ],
        ),
    )
    monkeypatch.setenv("GNOSIS_FEDERATION_TOKEN", _FEDERATION_TOKEN)
    monkeypatch.setenv("GNOSIS_PEER_PARTNER_TOKEN", _PEER_TOKEN)
    assert isinstance(backend, RecallFilteringBackend)
    return TestClient(
        create_app(
            settings_factory=Settings,
            backend=cast("MemoryBackend", cast("object", backend)),
            federation_transport=transport,
        ),
    )


def _context_request(*, max_items: int = 8) -> MemoryContextRequest:
    return MemoryContextRequest(
        scope=_scope(),
        query="what pet did Maria adopt?",
        include_short_term=False,
        include_reasoning=False,
        include_graph=False,
        max_items=max_items,
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


def _fact_row(content: str, *, memory_id: str) -> JsonObject:
    return {
        "id": memory_id,
        "subject": "bromigos:discord:private_user:pc-principal:789",
        "predicate": "memory",
        "object": content,
        "metadata": json.dumps(_scope_metadata()),
        "created_at": "2026-06-27T01:02:03+00:00",
        "updated_at": None,
    }


def _memory_record(
    memory_id: str,
    content: str,
    *,
    score: float | None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        content=content,
        score=score,
        metadata={"topic": "snacks", "shareable": True},
        created_at="2026-06-27T01:02:03+00:00",
    )
