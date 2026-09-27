"""Point-in-time reads (as_of), LLM-free requests and spaces, append-only adds.

The leak test for every read leg: a record observed *after* the read's moment
is stored next to one observed before it, the store hands both back to the
gateway (as if the in-Cypher narrowing were bypassed), and the response must
never contain the later one - not from dense retrieval, the SDK fallback,
BM25, the recency fallback, verbatim expansion, list, or a federated merge,
and read-time supersession must not let the later record displace the
earlier one. Graph legs, which walk edges that may come from later facts,
must not run at all. LLM-free requests must never reach a model.
"""

import json
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

from neo4j_agent_memory import MemorySettings  # noqa: E402
from neo4j_agent_memory.memory.long_term import Fact  # noqa: E402

from gnosis.backend import (  # noqa: E402
    BackendRequestError,
    MemoryClientContext,
    Neo4jAgentMemoryBackend,
)
from gnosis.memory_provider import (  # noqa: E402
    CREATE_APPEND_ONLY_MEMORY_CYPHER,
    LEXICAL_MEMORY_SEARCH_AS_OF_CYPHER,
    SCOPED_DENSE_MEMORY_SEARCH_AS_OF_CYPHER,
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
    MemoryContextRequest,
    MemoryListRequest,
    MemoryMessage,
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemoryVisibility,
    MessageRole,
    MessageWriteRequest,
)
from gnosis.point_in_time import (  # noqa: E402
    fact_visible_as_of,
    observed_epoch,
    parse_as_of,
    record_visible_as_of,
    visible_as_of,
)
from gnosis.query_router import RouteDecision, RouteVerdict  # noqa: E402
from gnosis.recall_filter import RecallSelection  # noqa: E402
from gnosis.reranker import RerankResult  # noqa: E402
from gnosis.settings import Settings  # noqa: E402
from gnosis.sufficiency import SufficiencyVerdict  # noqa: E402

_EARLY = "00000000-0000-0000-0000-00000000e001"
_LATE = "00000000-0000-0000-0000-00000000e002"
_EARLY_SEEN = "2021-03-01T12:00:00+00:00"
_LATE_SEEN = "2023-06-05T12:00:00+00:00"
_AS_OF = "2022-01-01T00:00:00Z"
_WRITTEN = "2026-09-27T10:00:00+00:00"  # both back-loaded years later


# ── the predicate ─────────────────────────────────────────────────────────


def test_parse_as_of_requires_an_offset() -> None:
    assert parse_as_of(_AS_OF) == datetime(2022, 1, 1, tzinfo=UTC)
    assert parse_as_of(None) is None
    with pytest.raises(ValueError, match="offset"):
        _ = parse_as_of("2022-01-01T00:00:00")
    with pytest.raises(ValueError, match="offset"):
        _ = parse_as_of("yesterday")


def test_observed_at_wins_over_the_write_time() -> None:
    as_of = parse_as_of(_AS_OF)
    # Written in 2026, observed in 2021: visible as of 2022.
    assert visible_as_of({"observed_at": _EARLY_SEEN}, _WRITTEN, as_of)
    # Observed in 2023: invisible, whenever it was written.
    assert not visible_as_of({"observed_at": _LATE_SEEN}, "2021-01-01T00:00:00Z", as_of)
    # No observed_at: the write time decides.
    assert visible_as_of({}, "2021-06-01T00:00:00+00:00", as_of)
    assert not visible_as_of({}, _WRITTEN, as_of)
    # No as_of: everything, as before.
    assert visible_as_of({"observed_at": _LATE_SEEN}, None, None)


def test_an_unknown_observation_time_fails_closed() -> None:
    as_of = parse_as_of(_AS_OF)
    assert not visible_as_of({"observed_at": "not a time"}, _WRITTEN, as_of)
    assert not visible_as_of({"observed_at": "2021-01-01T00:00:00"}, None, as_of)
    assert not visible_as_of({}, None, as_of)
    assert not fact_visible_as_of({"metadata": "not json"}, as_of)


def test_fact_and_record_predicates_read_every_metadata_shape() -> None:
    as_of = parse_as_of(_AS_OF)
    early = json.dumps({"observed_at": _EARLY_SEEN})
    assert fact_visible_as_of({"metadata": early, "created_at": _WRITTEN}, as_of)
    assert not fact_visible_as_of(
        {"metadata": {"observed_at": _LATE_SEEN}, "created_at": None},
        as_of,
    )
    record = MemoryRecord(
        memory_id=_LATE,
        content="late",
        metadata={"observed_at": _LATE_SEEN},
        created_at=_WRITTEN,
    )
    assert not record_visible_as_of(record, as_of)
    assert (
        observed_epoch({"observed_at": _EARLY_SEEN})
        == datetime(2021, 3, 1, 12, tzinfo=UTC).timestamp()
    )
    assert observed_epoch({}) is None


def test_as_of_decision_turns_off_every_leg_built_from_later_records() -> None:
    settings = Settings(
        gnosis_graphqa_fusion_enabled=True,
        gnosis_graph_traversal_enabled=True,
        gnosis_bridge_traversal_enabled=True,
    )
    decision = RouteDecision.for_route("knowledge_update", settings).for_as_of()
    assert not decision.graphqa_fusion
    assert not decision.graph_traversal
    assert not decision.bridge_traversal
    assert not decision.recency_injection_enabled
    assert not decision.filter_superseded
    llm_free = RouteDecision.from_settings(settings).without_llm()
    assert not llm_free.graphqa_fusion
    assert not llm_free.bridge_traversal
    assert not llm_free.sufficiency_check_enabled
    assert llm_free.graph_traversal  # zero LLM calls: stays as configured


# ── search ────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_search_as_of_hides_a_later_event_on_the_dense_leg() -> None:
    # Given: the vector query returns both events (narrowing bypassed).
    client = FakeMemoryClient()
    client.query.dense = [_row("early event", _EARLY), _row("late event", _LATE)]
    backend = _backend(client, Settings())

    # When: the caller searches as of 2022.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="event", as_of=_AS_OF),
    )

    # Then: only the 2021 event, and the scoped as-of query ran even with
    # scoped dense retrieval off - the SDK's global ranking was not asked.
    assert [r.content for r in response.results] == ["early event"]
    statement, params = client.query.calls[0]
    assert statement == SCOPED_DENSE_MEMORY_SEARCH_AS_OF_CYPHER
    assert params is not None
    assert params["as_of_epoch"] == datetime(2022, 1, 1, tzinfo=UTC).timestamp()
    assert "filter_superseded" not in params
    assert client.long_term.search_queries == []


@pytest.mark.anyio
async def test_search_as_of_still_filters_when_it_degrades_to_the_sdk() -> None:
    # Given: the vector query fails, so the SDK global ranking answers.
    client = FakeMemoryClient()
    client.query.fail_vector = True
    client.long_term.search_results = [
        _sdk_fact("late event", _LATE, _LATE_SEEN),
        _sdk_fact("early event", _EARLY, _EARLY_SEEN),
    ]
    backend = _backend(client, Settings())

    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="event", as_of=_AS_OF),
    )

    assert [r.content for r in response.results] == ["early event"]


@pytest.mark.anyio
async def test_search_as_of_hides_a_later_event_on_the_lexical_leg() -> None:
    # Given: hybrid retrieval on; only BM25 finds the later event.
    client = FakeMemoryClient()
    client.query.dense = [_row("early event", _EARLY)]
    client.query.lexical = [_row("late event", _LATE)]
    backend = _backend(client, Settings(gnosis_hybrid_retrieval_enabled=True))

    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="event", as_of=_AS_OF),
    )

    assert [r.content for r in response.results] == ["early event"]
    lexical = [s for s, _ in client.query.calls if "fulltext" in s]
    assert lexical == [LEXICAL_MEMORY_SEARCH_AS_OF_CYPHER]


@pytest.mark.anyio
async def test_supersession_as_of_compares_only_what_was_visible() -> None:
    # Given: two same-slot facts, "works at Google" (2021) superseded by
    # "works at NVIDIA" (2023), both returned by retrieval.
    client = FakeMemoryClient()
    slot: dict[str, JsonValue] = {
        "relation_slots": ["alice:works_at"],
        "entities": ["alice"],
    }
    client.query.dense = [
        _row(
            "Alice works at NVIDIA",
            _LATE,
            predicate="fact",
            extra=slot | {"event_date": "2023-06-05"},
        ),
        _row(
            "Alice works at Google",
            _EARLY,
            predicate="fact",
            extra=slot | {"event_date": "2021-03-01"},
        ),
    ]
    backend = _backend(
        client,
        Settings(
            gnosis_read_supersession_enabled=True,
            gnosis_scoped_dense_retrieval_enabled=True,
        ),
    )

    # When: searched now, and as of 2022.
    now = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="where does alice work?"),
    )
    then = await backend.search_memories(
        MemorySearchRequest(
            scope=_scope(), query="where does alice work?", as_of=_AS_OF
        ),
    )

    # Then: newest wins today; as of 2022 the later fact cannot displace
    # the earlier one, because it did not exist yet.
    assert [r.content for r in now.results] == ["Alice works at NVIDIA"]
    assert [r.content for r in then.results] == ["Alice works at Google"]


@pytest.mark.anyio
async def test_list_as_of_hides_later_events() -> None:
    client = FakeMemoryClient()
    client.query.listing = [_row("late event", _LATE), _row("early event", _EARLY)]
    backend = _backend(client, Settings())

    response = await backend.list_memories(
        MemoryListRequest(scope=_scope(), as_of=_AS_OF),
    )

    assert [r.content for r in response.results] == ["early event"]
    assert response.total == 1


@pytest.mark.anyio
async def test_a_naive_as_of_is_a_request_error() -> None:
    backend = _backend(FakeMemoryClient(), Settings())
    with pytest.raises(BackendRequestError, match="offset"):
        _ = await backend.search_memories(
            MemorySearchRequest(scope=_scope(), query="q", as_of="2022-01-01T00:00:00"),
        )


# ── context ───────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_context_as_of_hides_later_facts_and_skips_graph_legs() -> None:
    # Given: every graph leg switched on globally, and retrieval that
    # returns an earlier and a later event.
    client = FakeMemoryClient()
    client.query.dense = [_row("early event", _EARLY), _row("late event", _LATE)]
    graph = FakeGraphStore()
    settings = Settings(
        gnosis_graphqa_fusion_enabled=True,
        gnosis_graph_traversal_enabled=True,
        gnosis_entity_graph_enabled=True,
    )
    backend = _backend(client, settings, graph_store=graph)

    # When: context is assembled as of 2022.
    response = await backend.get_memory_context(_context_request(as_of=_AS_OF))

    # Then: the later event never renders, and no leg that walks the graph
    # ran (the graph store was not asked; no traversal Cypher executed).
    content = "\n".join(s.content for s in response.sections)
    assert "early event" in content
    assert "late event" not in content
    assert graph.requests == []
    assert not any("RELATES" in statement for statement, _ in client.query.calls)


@pytest.mark.anyio
async def test_context_as_of_recency_fallback_is_filtered() -> None:
    # Given: nothing ranks (no dense hit), so context falls back to recency,
    # and the most recently written fact was observed after the moment.
    client = FakeMemoryClient()
    client.query.recent = [
        {"f": _fact_dict("late event", _LATE, _LATE_SEEN)},
        {"f": _fact_dict("early event", _EARLY, _EARLY_SEEN)},
    ]
    backend = _backend(client, Settings())

    response = await backend.get_memory_context(_context_request(as_of=_AS_OF))

    content = "\n".join(s.content for s in response.sections)
    assert "early event" in content
    assert "late event" not in content


@pytest.mark.anyio
async def test_context_as_of_verbatim_expansion_renders_only_visible_turns() -> None:
    # Given: expansion on; an early extracted fact whose sources are an early
    # turn and a turn observed later.
    client = FakeMemoryClient()
    client.query.dense = [
        _row(
            "extracted: the listing happened",
            _EARLY,
            predicate="fact",
            extra={
                "extracted": True,
                "source_memory_ids": [
                    "00000000-0000-0000-0000-0000000000a1",
                    "00000000-0000-0000-0000-0000000000a2",
                ],
            },
        ),
    ]
    client.query.verbatim = [
        _row(
            "early source turn",
            "00000000-0000-0000-0000-0000000000a1",
            predicate="said_user",
        ),
        _row(
            "later source turn",
            "00000000-0000-0000-0000-0000000000a2",
            predicate="said_user",
            observed=_LATE_SEEN,
        ),
    ]
    backend = _backend(client, Settings(gnosis_fact_verbatim_expansion_enabled=True))

    response = await backend.get_memory_context(_context_request(as_of=_AS_OF))

    content = "\n".join(s.content for s in response.sections)
    assert "quote: early source turn" in content
    assert "later source turn" not in content


@pytest.mark.anyio
async def test_context_as_of_refuses_sections_that_are_not_point_in_time() -> None:
    backend = _backend(FakeMemoryClient(), Settings())
    with pytest.raises(BackendRequestError, match="long-term facts only"):
        _ = await backend.get_memory_context(
            MemoryContextRequest(scope=_scope(), query="q", as_of=_AS_OF),
        )


# ── LLM-free requests and spaces ──────────────────────────────────────────


@pytest.mark.anyio
async def test_an_llm_free_search_never_reaches_a_model() -> None:
    # Given: every LLM leg on globally.
    client = FakeMemoryClient()
    client.query.dense = [_row("early event", _EARLY)]
    models = FakeModels()
    backend = _backend(client, _all_llm_settings(), models=models)

    # When: the caller asks for an LLM-free search.
    response = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="event", use_llm=False),
    )

    # Then: results, and not one model call.
    assert [r.content for r in response.results] == ["early event"]
    assert models.calls == []


@pytest.mark.anyio
async def test_the_same_reads_with_llm_allowed_do_call_the_models() -> None:
    # The control for the two tests around it: the fakes are wired, so an
    # empty call log there means the LLM legs were skipped, not unreachable.
    client = FakeMemoryClient()
    client.query.dense = [_row("early event", _EARLY), _row("other event", _LATE)]
    models = FakeModels()
    backend = _backend(client, _all_llm_settings(), models=models)

    _ = await backend.search_memories(
        MemorySearchRequest(scope=_scope(), query="event")
    )
    _ = await backend.get_memory_context(_context_request())

    assert any(call.startswith("router:") for call in models.calls)
    assert any(call.startswith("recall:") for call in models.calls)


@pytest.mark.anyio
async def test_an_llm_free_context_never_reaches_a_model() -> None:
    client = FakeMemoryClient()
    client.query.dense = [_row("early event", _EARLY), _row("other event", _LATE)]
    models = FakeModels()
    graph = FakeGraphStore()
    backend = _backend(client, _all_llm_settings(), models=models, graph_store=graph)

    response = await backend.get_memory_context(_context_request(use_llm=False))

    assert "early event" in "\n".join(s.content for s in response.sections)
    assert models.calls == []
    assert graph.requests == []
    assert response.sufficiency is None


@pytest.mark.anyio
async def test_an_llm_free_space_overrides_the_request() -> None:
    # Given: the space is listed as LLM-free, and the request does not ask.
    client = FakeMemoryClient()
    client.query.dense = [_row("early event", _EARLY)]
    models = FakeModels()
    settings = _all_llm_settings().model_copy(
        update={"gnosis_llm_free_spaces": ["arbiter-signals"]},
    )
    backend = _backend(client, settings, models=models)

    _ = await backend.search_memories(
        MemorySearchRequest(scope=_scope(space_id="arbiter-signals"), query="event"),
    )
    with pytest.raises(BackendRequestError, match="LLM"):
        _ = await backend.get_memory_context(
            MemoryContextRequest(scope=_scope(space_id="arbiter-signals"), query="q"),
        )

    assert models.calls == []


@pytest.mark.anyio
async def test_an_llm_free_space_refuses_extraction_writes() -> None:
    client = FakeMemoryClient()
    settings = Settings(
        gnosis_fact_extraction_enabled=True,
        gnosis_llm_free_spaces=["arbiter-signals"],
    )
    backend = _backend(client, settings)
    scope = _scope(space_id="arbiter-signals")
    with pytest.raises(BackendRequestError, match="LLM"):
        _ = await backend.add_memories(
            MemoryAddRequest(
                scope=scope,
                messages=[MemoryMessage(role="user", content="lists AAA")],
            ),
        )
    with pytest.raises(BackendRequestError, match="LLM"):
        _ = await backend.add_message(
            MessageWriteRequest(scope=scope, role=MessageRole.USER, content="hi"),
        )
    assert client.long_term.added == []
    assert client.graph.writes == []


# ── append-only verbatim adds ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_append_only_add_bypasses_dedup_and_stamps_the_observation() -> None:
    client = FakeMemoryClient()
    backend = _backend(client, Settings(gnosis_fact_deduplication_enabled=True))

    result = await backend.add_memories(
        MemoryAddRequest(
            scope=_scope(),
            content="2021-03-01 okx: OKX to list AAA/USDT",
            infer=False,
            append_only=True,
            metadata={"observed_at": _EARLY_SEEN, "item_id": "okx:1"},
        ),
    )

    # Then: a direct CREATE, never the SDK's deduplicating add_fact, with the
    # observation time as the node property the as-of queries read.
    assert client.long_term.added == []
    statement, params = client.graph.writes[0]
    assert statement == CREATE_APPEND_ONLY_MEMORY_CYPHER
    assert params["observed_at_epoch"] == observed_epoch({"observed_at": _EARLY_SEEN})
    assert params["predicate"] == "memory"
    assert result.results[0].event == "ADD"
    stored = cast("dict[str, JsonValue]", json.loads(cast("str", params["metadata"])))
    assert stored["item_id"] == "okx:1"


@pytest.mark.anyio
async def test_append_only_requires_verbatim_content() -> None:
    backend = _backend(FakeMemoryClient(), Settings())
    with pytest.raises(BackendRequestError, match="verbatim"):
        _ = await backend.add_memories(
            MemoryAddRequest(
                scope=_scope(),
                messages=[MemoryMessage(role="user", content="x")],
                append_only=True,
            ),
        )


# ── a memory is reachable by the scope and ids that wrote it ──────────────


@pytest.mark.anyio
async def test_a_memory_is_found_by_the_scope_and_ids_that_wrote_it() -> None:
    # Given: a user_id and a join id of 24+ characters mixing letters and
    # digits - the shape the opaque-value redaction took for a credential, so
    # the memory was stored under "[REDACTED]" and no scope could reach it.
    user_id = "arbiter-roundtrip-1790524864"
    item_id = "okx-announcements-1790524864-aaa"
    scope = _scope().model_copy(update={"user_id": user_id})
    client = FakeMemoryClient()
    backend = _backend(client, Settings())

    _ = await backend.add_memories(
        MemoryAddRequest(
            scope=scope,
            content="OKX to list AAA/USDT (ref a1b2c3d4e5f6g7h8i9j0k1l2m3n4)",
            infer=False,
            append_only=True,
            metadata={"observed_at": _EARLY_SEEN, "item_id": item_id},
        ),
    )

    # Then: the scope and the id are stored as written; the content is not.
    _, params = client.graph.writes[0]
    stored = cast("dict[str, JsonValue]", json.loads(cast("str", params["metadata"])))
    assert stored["user_id"] == user_id
    assert stored["item_id"] == item_id
    assert params["object"] == "OKX to list AAA/USDT (ref [REDACTED])"

    # When: retrieval hands the stored row back, searched by that scope and
    # filtered by that id.
    client.query.dense = [
        {
            "id": _EARLY,
            "subject": params["subject"],
            "predicate": params["predicate"],
            "object": params["object"],
            "metadata": params["metadata"],
            "created_at": _WRITTEN,
            "updated_at": None,
            "score": 0.9,
        },
    ]
    response = await backend.search_memories(
        MemorySearchRequest(
            scope=scope,
            query="okx listing",
            filters={"metadata.item_id": item_id},
            as_of=_AS_OF,
            use_llm=False,
        ),
    )

    # Then: found, with its join id intact.
    assert [r.memory_id for r in response.results] == [_EARLY]
    assert response.results[0].metadata["item_id"] == item_id


# ── fakes ─────────────────────────────────────────────────────────────────


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
    added: list[str] = field(default_factory=list)

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

    async def add_fact(  # noqa: PLR0913 - mirrors the SDK.
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
    ) -> Fact:
        _ = (confidence, valid_from, valid_until, generate_embedding)
        self.added.append(obj)
        return Fact(
            id=UUID(_EARLY),
            subject=subject,
            predicate=predicate,
            object=obj,
            metadata=metadata or {},
        )

    async def get_context(self, query: str, *, max_items: int) -> str:
        _ = (query, max_items)
        return ""


@dataclass(slots=True)
class FakeCypherQuery:
    dense: list[JsonObject] = field(default_factory=list)
    lexical: list[JsonObject] = field(default_factory=list)
    recent: list[JsonObject] = field(default_factory=list)
    verbatim: list[JsonObject] = field(default_factory=list)
    listing: list[JsonObject] = field(default_factory=list)
    fail_vector: bool = False
    calls: list[tuple[str, dict[str, JsonValue] | None]] = field(default_factory=list)

    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.calls.append((query, params))
        if "db.index.vector.queryNodes" in query:
            if self.fail_vector:
                message = "vector index unavailable"
                raise RuntimeError(message)
            return list(self.dense)
        if "db.index.fulltext.queryNodes" in query:
            return list(self.lexical)
        if "f.id IN $memory_ids" in query:
            return list(self.verbatim)
        if "LIMIT $scan_limit" in query:
            return list(self.listing)
        if "$metadata_fragments" in query:
            return list(self.recent)
        return []


@dataclass(slots=True)
class FakeGraphWrite:
    writes: list[tuple[str, dict[str, JsonValue]]] = field(default_factory=list)

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.writes.append((query, dict(parameters or {})))
        return []


@dataclass(slots=True)
class FakeMemoryClient:
    long_term: FakeLongTermMemory = field(default_factory=FakeLongTermMemory)
    query: FakeCypherQuery = field(default_factory=FakeCypherQuery)
    graph: FakeGraphWrite = field(default_factory=FakeGraphWrite)

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
        return cast("MemoryClientContext", cast("object", self.client))


@dataclass(slots=True)
class FakeGraphStore:
    requests: list[GraphContextRequest] = field(default_factory=list)

    async def require_available(self) -> None:
        return None

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        return EventIngestResult(
            event_id=event.event_id, status=EventIngestStatus.ACCEPTED
        )

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        self.requests.append(request)
        return GraphContextResponse(context="")


@dataclass(slots=True)
class FakeModels:
    """Every LLM collaborator in one; records each call it would make."""

    calls: list[str] = field(default_factory=list)

    async def classify(self, query: str) -> RouteVerdict | None:
        self.calls.append(f"router:{query}")
        return RouteVerdict(route="multi_hop")

    async def select_candidates(
        self, query: str, candidates: object
    ) -> RecallSelection:
        _ = candidates
        self.calls.append(f"recall:{query}")
        return RecallSelection(kept_indices=[1])

    async def rerank(self, query: str, candidates: object) -> RerankResult:
        _ = candidates
        self.calls.append(f"rerank:{query}")
        return RerankResult(order=[0])

    async def assess(self, query: str, context: str) -> SufficiencyVerdict:
        _ = context
        self.calls.append(f"sufficiency:{query}")
        return SufficiencyVerdict(sufficient=True)

    async def name_bridges(self, query: str, evidence: list[str]) -> str | None:
        _ = evidence
        self.calls.append(f"bridge:{query}")
        return "somebody"


def _all_llm_settings() -> Settings:
    return Settings(
        gnosis_scoped_dense_retrieval_enabled=True,
        gnosis_adaptive_routing_enabled=True,
        gnosis_recall_filter_enabled=True,
        gnosis_rerank_enabled=True,
        gnosis_sufficiency_check_enabled=True,
        gnosis_graphqa_fusion_enabled=True,
        gnosis_bridge_traversal_enabled=True,
        gnosis_query_rewrite_enabled=True,
    )


def _backend(
    client: FakeMemoryClient,
    settings: Settings,
    *,
    models: FakeModels | None = None,
    graph_store: FakeGraphStore | None = None,
) -> Neo4jAgentMemoryBackend:
    llm = models or FakeModels()
    return Neo4jAgentMemoryBackend(
        settings,
        memory_client_factory=FakeMemoryClientFactory(client),
        graph_store=graph_store or FakeGraphStore(),
        recall_filter=llm,
        sufficiency_assessor=llm,
        reranker=llm,
        query_router=llm,
        bridge_namer=llm,
    )


def _scope(space_id: str = "arbiter-signals") -> MemoryScope:
    return MemoryScope(
        tenant_id="bromigos",
        space_id=space_id,
        agent_id="arbiter-news",
        session_id="events",
        user_id="arbiter-signals",
        visibility=MemoryVisibility.AGENT_SHARED,
    )


def _scope_metadata() -> dict[str, JsonValue]:
    scope = _scope()
    return {
        "tenant_id": scope.tenant_id,
        "space_id": scope.space_id,
        "agent_id": scope.agent_id,
        "session_id": scope.session_id,
        "user_id": scope.user_id,
        "visibility": scope.visibility.value,
    }


def _context_request(
    *,
    as_of: str | None = None,
    use_llm: bool = True,
) -> MemoryContextRequest:
    return MemoryContextRequest(
        scope=_scope(),
        query="what happened?",
        include_short_term=False,
        include_reasoning=False,
        include_graph=False,
        max_items=8,
        as_of=as_of,
        use_llm=use_llm,
    )


def _observed_for(memory_id: str) -> str:
    return _LATE_SEEN if memory_id == _LATE else _EARLY_SEEN


def _row(
    content: str,
    memory_id: str,
    *,
    predicate: str = "memory",
    extra: dict[str, JsonValue] | None = None,
    observed: str | None = None,
) -> JsonObject:
    metadata = _scope_metadata() | {"observed_at": observed or _observed_for(memory_id)}
    return {
        "id": memory_id,
        "subject": "bromigos:arbiter-signals:agent_shared:arbiter-news:arbiter-signals",
        "predicate": predicate,
        "object": content,
        "metadata": json.dumps(metadata | (extra or {})),
        "created_at": _WRITTEN,
        "updated_at": None,
        "score": 0.9,
    }


def _fact_dict(content: str, memory_id: str, observed: str) -> JsonObject:
    return {
        "id": memory_id,
        "subject": "bromigos:arbiter-signals:agent_shared:arbiter-news:arbiter-signals",
        "predicate": "memory",
        "object": content,
        "metadata": json.dumps(_scope_metadata() | {"observed_at": observed}),
        "created_at": _WRITTEN,
    }


def _sdk_fact(content: str, memory_id: str, observed: str) -> Fact:
    return Fact(
        id=UUID(memory_id),
        subject="bromigos:arbiter-signals:agent_shared:arbiter-news:arbiter-signals",
        predicate="memory",
        object=content,
        created_at=datetime(2026, 9, 27, 10, tzinfo=UTC),
        metadata=_scope_metadata() | {"observed_at": observed, "similarity": 0.9},
    )
