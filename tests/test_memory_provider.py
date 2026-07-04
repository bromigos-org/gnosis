import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from os import environ
from typing import Literal, Self, cast

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

import httpx  # noqa: E402
from neo4j_agent_memory import MemorySettings  # noqa: E402
from neo4j_agent_memory.memory.long_term import Fact  # noqa: E402
from openai import APIConnectionError  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

from gnosis.backend import (  # noqa: E402
    BackendCapabilityUnavailable,
    BackendRequestError,
    GraphWriteQuery,
    MemoryClientContext,
    MemoryNotFoundError,
    Neo4jAgentMemoryBackend,
)
from gnosis.fact_extraction import (  # noqa: E402
    ConversationTurn,
    MemoryUnit,
    MemoryUnitExtraction,
    MemoryUnitExtractor,
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
    MessageRole,
    MessageWriteRequest,
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
async def test_add_memories_extraction_writes_units_alongside_verbatim() -> None:
    # Given: a turn-pair add with the extraction flag on and an extractor
    # producing one dated and one undated memory unit.
    client = FakeMemoryClient()
    extractor = RecordingUnitExtractor(
        extraction=MemoryUnitExtraction(
            facts=[
                MemoryUnit(
                    text="Cartman ate cheesy poofs on 7 May 2023",
                    source_turn_ids=[1],
                    entities=["Cartman"],
                    event_date="2023-05-07",
                ),
                MemoryUnit(
                    text="Cartman prefers cheesy poofs over vegetables",
                    source_turn_ids=[1, 2],
                    entities=["Cartman"],
                    event_date=None,
                ),
            ],
        ),
    )
    backend = _backend(client, extractor, extraction_enabled=True)
    request = MemoryAddRequest(
        scope=_scope(),
        messages=[
            MemoryMessage(role="user", content="I ate cheesy poofs today"),
            MemoryMessage(role="assistant", content="Better than vegetables?"),
        ],
        metadata={"session_date": "2023-05-07", "topic": "snacks"},
    )

    # When: the memories are added.
    response = await backend.add_memories(request)

    # Then: the verbatim turn facts are still written through the SDK path.
    assert [write.predicate for write in client.long_term.fact_writes] == [
        "said_user",
        "said_assistant",
    ]
    verbatim_ids = [str(fact.id) for fact in client.long_term.facts]

    # Then: the extractor saw the session date and the new turns as speakers.
    assert extractor.calls == [
        ExtractionCall(
            conversation_date="2023-05-07",
            context_turns=(),
            new_turns=(
                ConversationTurn(speaker="user", content="I ate cheesy poofs today"),
                ConversationTurn(
                    speaker="assistant",
                    content="Better than vegetables?",
                ),
            ),
        ),
    ]

    # Then: each unit lands as a Fact via a direct dedup-bypassing CREATE
    # with the `fact` predicate, the scope subject, and its own embedding.
    writes = _graph(client).writes
    assert len(writes) == 2
    create_query, create_params = writes[0]
    assert "CREATE (f:Fact" in create_query
    assert create_params is not None
    assert create_params["predicate"] == "fact"
    assert create_params["subject"] == (
        "bromigos:discord:private_user:pc-principal:789"
    )
    assert create_params["object"] == "Cartman ate cheesy poofs on 7 May 2023"
    assert create_params["embedding"] == [0.1, 0.2]
    stored_metadata = _JSON_OBJECT_ADAPTER.validate_json(
        cast("str", create_params["metadata"]),
    )
    assert stored_metadata["extracted"] is True
    assert stored_metadata["extraction_version"] == "edu-v1"
    assert stored_metadata["extraction_model"] == "openai/gemma4"
    assert stored_metadata["event_date"] == "2023-05-07"
    assert stored_metadata["entities"] == ["Cartman"]
    assert stored_metadata["source_memory_ids"] == verbatim_ids
    assert stored_metadata["source_turn_ids"] == [1]
    assert stored_metadata["tenant_id"] == "bromigos"
    assert stored_metadata["user_id"] == "789"
    assert stored_metadata["session_id"] == "guild:123:channel:456"
    assert stored_metadata["topic"] == "snacks"
    assert stored_metadata["session_date"] == "2023-05-07"

    # Then: the response appends the extracted units after the verbatim
    # results, with real memory ids, ADD events, and public metadata only.
    assert [result.event for result in response.results] == ["ADD"] * 4
    assert [result.memory_id for result in response.results[:2]] == verbatim_ids
    extracted_result = response.results[2]
    assert extracted_result.memory_id == create_params["memory_id"]
    assert extracted_result.content == "Cartman ate cheesy poofs on 7 May 2023"
    assert extracted_result.metadata["extracted"] is True
    assert extracted_result.metadata["event_date"] == "2023-05-07"
    assert "tenant_id" not in extracted_result.metadata
    assert response.results[3].content == (
        "Cartman prefers cheesy poofs over vegetables"
    )


@pytest.mark.anyio
async def test_add_memories_extraction_never_fabricates_event_date() -> None:
    # Given: an extracted unit for an ongoing state with no resolvable date.
    client = FakeMemoryClient()
    extractor = RecordingUnitExtractor(
        extraction=MemoryUnitExtraction(
            facts=[
                MemoryUnit(
                    text="Cartman prefers cheesy poofs",
                    source_turn_ids=[1],
                    entities=["Cartman"],
                    event_date=None,
                ),
            ],
        ),
    )
    backend = _backend(client, extractor, extraction_enabled=True)

    # When: the turn is added.
    response = await backend.add_memories(
        MemoryAddRequest(
            scope=_scope(),
            messages=[MemoryMessage(role="user", content="I love cheesy poofs")],
        ),
    )

    # Then: the stored and returned metadata omit event_date entirely.
    _, create_params = _graph(client).writes[0]
    assert create_params is not None
    stored_metadata = _JSON_OBJECT_ADAPTER.validate_json(
        cast("str", create_params["metadata"]),
    )
    assert "event_date" not in stored_metadata
    assert "event_date" not in response.results[1].metadata


@pytest.mark.anyio
async def test_add_memories_extraction_flag_off_keeps_todays_bytes() -> None:
    # Given: the default settings posture (extraction off) and an extractor
    # double that would produce a unit if it were ever consulted.
    client = FakeMemoryClient()
    extractor = RecordingUnitExtractor(
        extraction=MemoryUnitExtraction(
            facts=[MemoryUnit(text="never written", source_turn_ids=[1])],
        ),
    )
    backend = _backend(client, extractor)

    # When: a turn-pair is added.
    response = await backend.add_memories(
        MemoryAddRequest(
            scope=_scope(),
            messages=[
                MemoryMessage(role="user", content="I love cheesy poofs"),
                MemoryMessage(role="assistant", content="Noted, cheesy poofs it is"),
            ],
            metadata={"topic": "snacks"},
        ),
    )

    # Then: no extraction call, no context read, no extra graph write, and
    # the response is byte-identical to the pre-extraction contract.
    assert extractor.calls == []
    assert client.query.calls == []
    assert _graph(client).writes == []
    verbatim_ids = [str(fact.id) for fact in client.long_term.facts]
    assert response.model_dump_json() == (
        '{"results":['
        f'{{"memory_id":"{verbatim_ids[0]}","content":"I love cheesy poofs",'
        '"event":"ADD","metadata":{"topic":"snacks"}},'
        f'{{"memory_id":"{verbatim_ids[1]}","content":"Noted, cheesy poofs it is",'
        '"event":"ADD","metadata":{"topic":"snacks"}}'
        "]}"
    )


@pytest.mark.anyio
async def test_add_memories_extraction_llm_failure_degrades_to_verbatim(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the extraction flag on and an extractor failing at the transport.
    client = FakeMemoryClient()
    backend = _backend(client, FailingUnitExtractor(), extraction_enabled=True)

    # When: a turn-pair is added.
    with caplog.at_level("WARNING", logger="gnosis.fact_extraction"):
        response = await backend.add_memories(
            MemoryAddRequest(
                scope=_scope(),
                messages=[
                    MemoryMessage(role="user", content="I love cheesy poofs"),
                    MemoryMessage(
                        role="assistant",
                        content="Noted, cheesy poofs it is",
                    ),
                ],
            ),
        )

    # Then: the add succeeds exactly as a verbatim-only ingest and the
    # failure leaves a structured warning.
    assert [result.event for result in response.results] == ["ADD", "ADD"]
    assert [write.predicate for write in client.long_term.fact_writes] == [
        "said_user",
        "said_assistant",
    ]
    assert _graph(client).writes == []
    assert "fact extraction failed" in caplog.text


@pytest.mark.anyio
async def test_add_memories_extraction_context_respects_configured_turns() -> None:
    # Given: stored session turns (newest first, as the query returns them)
    # and a two-turn context window.
    client = FakeMemoryClient()
    client.query.rows = [
        _turn_row(_MEMORY_ID, "assistant", "How was Tokyo?"),
        _turn_row(_OTHER_MEMORY_ID, "user", "I went to Tokyo"),
    ]
    extractor = RecordingUnitExtractor(extraction=MemoryUnitExtraction())
    backend = _backend(
        client,
        extractor,
        extraction_enabled=True,
        extraction_context_turns=2,
    )

    # When: the next turn is added.
    _ = await backend.add_memories(
        MemoryAddRequest(
            scope=_scope(),
            messages=[MemoryMessage(role="user", content="It was great")],
        ),
    )

    # Then: the context read is session-scoped and limited to the configured
    # window, fetched before the new turns were written.
    context_query, context_params = client.query.calls[0]
    assert "STARTS WITH $predicate_prefix" in context_query
    assert context_params is not None
    assert context_params["limit"] == 2
    assert context_params["predicate_prefix"] == "said_"
    assert context_params["scope_fragments"] == [
        '"tenant_id": "bromigos"',
        '"user_id": "789"',
        '"session_id": "guild:123:channel:456"',
    ]

    # Then: the extractor received the window in chronological order with
    # speakers recovered from the said_* predicates.
    assert extractor.calls[0].context_turns == (
        ConversationTurn(speaker="user", content="I went to Tokyo"),
        ConversationTurn(speaker="assistant", content="How was Tokyo?"),
    )
    assert extractor.calls[0].new_turns == (
        ConversationTurn(speaker="user", content="It was great"),
    )


@pytest.mark.anyio
async def test_add_memories_verbatim_content_mode_never_extracts() -> None:
    # Given: the extraction flag on and a verbatim infer=false add.
    client = FakeMemoryClient()
    extractor = RecordingUnitExtractor(extraction=MemoryUnitExtraction())
    backend = _backend(client, extractor, extraction_enabled=True)

    # When: the verbatim memory is added.
    response = await backend.add_memories(
        MemoryAddRequest(scope=_scope(), content="plain note", infer=False),
    )

    # Then: no extraction runs on non-inferring adds.
    assert extractor.calls == []
    assert _graph(client).writes == []
    assert len(response.results) == 1


@pytest.mark.anyio
async def test_add_message_extraction_extends_message_writes() -> None:
    # Given: the extraction flag on for the /v1/messages ingestion path.
    client = FakeMemoryClient()
    extractor = RecordingUnitExtractor(
        extraction=MemoryUnitExtraction(
            facts=[
                MemoryUnit(text="Stan adopted a dog", source_turn_ids=[1]),
            ],
        ),
    )
    backend = _backend(client, extractor, extraction_enabled=True)

    # When: a message is written.
    response = await backend.add_message(
        MessageWriteRequest(
            scope=_scope(),
            role=MessageRole.USER,
            content="I adopted a dog",
        ),
    )

    # Then: the verbatim said_user fact still lands and the extracted unit is
    # written with provenance back to it; the write dates against ingest time.
    assert response.accepted is True
    assert client.long_term.fact_writes[0].predicate == "said_user"
    _, create_params = _graph(client).writes[0]
    assert create_params is not None
    assert create_params["predicate"] == "fact"
    stored_metadata = _JSON_OBJECT_ADAPTER.validate_json(
        cast("str", create_params["metadata"]),
    )
    assert stored_metadata["source_memory_ids"] == [
        str(client.long_term.facts[0].id),
    ]
    assert extractor.calls[0].conversation_date == (
        datetime.now(UTC).date().isoformat()
    )


@pytest.mark.anyio
async def test_add_memories_background_mode_returns_verbatim_only_then_extracts() -> (
    None
):
    # Given: the extraction flag on in background mode.
    client = FakeMemoryClient()
    extractor = RecordingUnitExtractor(
        extraction=MemoryUnitExtraction(
            facts=[MemoryUnit(text="Stan adopted a dog", source_turn_ids=[1])],
        ),
    )
    backend = _backend(
        client,
        extractor,
        extraction_enabled=True,
        extraction_mode="background",
    )

    # When: a turn is added.
    response = await backend.add_memories(
        MemoryAddRequest(
            scope=_scope(),
            messages=[MemoryMessage(role="user", content="I adopted a dog")],
        ),
    )

    # Then: the response returns immediately with only the verbatim result -
    # the same shape as with extraction disabled - and no extraction has run.
    assert [result.event for result in response.results] == ["ADD"]
    assert response.results[0].content == "I adopted a dog"
    assert extractor.calls == []
    assert _graph(client).writes == []
    verbatim_ids = [str(fact.id) for fact in client.long_term.facts]

    # When: the background queue drains.
    await backend.shutdown()

    # Then: the extraction ran once and the unit landed with provenance back
    # to the already-written verbatim facts.
    assert len(extractor.calls) == 1
    _, create_params = _graph(client).writes[0]
    assert create_params is not None
    assert create_params["object"] == "Stan adopted a dog"
    stored_metadata = _JSON_OBJECT_ADAPTER.validate_json(
        cast("str", create_params["metadata"]),
    )
    assert stored_metadata["source_memory_ids"] == verbatim_ids
    diagnostics = backend.diagnostics(await backend.readiness())
    assert diagnostics.extraction_queue is not None
    assert diagnostics.extraction_queue.mode == "background"
    assert diagnostics.extraction_queue.processed == 1
    assert diagnostics.extraction_queue.pending == 0
    assert diagnostics.extraction_queue.failed == 0
    assert diagnostics.extraction_queue.dropped == 0


@pytest.mark.anyio
async def test_background_extraction_excludes_own_turns_from_context() -> None:
    # Given: background mode with a two-turn context window; by the time the
    # job runs, the just-added turn is already stored and would come back
    # first from the context query.
    client = FakeMemoryClient()
    extractor = RecordingUnitExtractor(extraction=MemoryUnitExtraction())
    backend = _backend(
        client,
        extractor,
        extraction_enabled=True,
        extraction_mode="background",
        extraction_context_turns=2,
    )
    _ = await backend.add_memories(
        MemoryAddRequest(
            scope=_scope(),
            messages=[MemoryMessage(role="user", content="It was great")],
        ),
    )
    new_turn_id = str(client.long_term.facts[0].id)
    client.query.rows = [
        _turn_row(new_turn_id, "user", "It was great"),
        _turn_row(_MEMORY_ID, "assistant", "How was Tokyo?"),
        _turn_row(_OTHER_MEMORY_ID, "user", "I went to Tokyo"),
    ]

    # When: the background job runs.
    await backend.shutdown()

    # Then: the context read over-fetched by the one excluded turn and the
    # extractor saw the window without the pair being extracted.
    context_query, context_params = client.query.calls[0]
    assert "STARTS WITH $predicate_prefix" in context_query
    assert context_params is not None
    assert context_params["limit"] == 3
    assert extractor.calls[0].context_turns == (
        ConversationTurn(speaker="user", content="I went to Tokyo"),
        ConversationTurn(speaker="assistant", content="How was Tokyo?"),
    )
    assert extractor.calls[0].new_turns == (
        ConversationTurn(speaker="user", content="It was great"),
    )


@pytest.mark.anyio
async def test_add_message_background_mode_defers_extraction() -> None:
    # Given: background mode on the /v1/messages hot path.
    client = FakeMemoryClient()
    extractor = RecordingUnitExtractor(
        extraction=MemoryUnitExtraction(
            facts=[MemoryUnit(text="Stan adopted a dog", source_turn_ids=[1])],
        ),
    )
    backend = _backend(
        client,
        extractor,
        extraction_enabled=True,
        extraction_mode="background",
    )

    # When: a message is written.
    response = await backend.add_message(
        MessageWriteRequest(
            scope=_scope(),
            role=MessageRole.USER,
            content="I adopted a dog",
        ),
    )

    # Then: the write is acknowledged with no extraction in the request path.
    assert response.accepted is True
    assert client.long_term.fact_writes[0].predicate == "said_user"
    assert extractor.calls == []
    assert _graph(client).writes == []

    # When: the queue drains on shutdown.
    await backend.shutdown()

    # Then: the extracted fact appears with provenance to the verbatim turn.
    _, create_params = _graph(client).writes[0]
    assert create_params is not None
    stored_metadata = _JSON_OBJECT_ADAPTER.validate_json(
        cast("str", create_params["metadata"]),
    )
    assert stored_metadata["source_memory_ids"] == [
        str(client.long_term.facts[0].id),
    ]


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


@pytest.mark.anyio
async def test_update_memory_when_sdk_wraps_writes_in_a_delegating_proxy() -> None:
    # Given: the deployed SDK shape - client.graph is a proxy that exposes
    # execute_write only through dynamic __getattr__ delegation, which the
    # runtime protocol isinstance check cannot see on Python 3.12+.
    writer = FakeGraphWriter(write_rows=[{"object": "new content"}])
    proxy = FakeGraphWriteProxy(inner=writer)
    assert not isinstance(proxy, GraphWriteQuery)
    client = FakeMemoryClient(graph=proxy)
    client.query.rows = [_memory_row(_MEMORY_ID, "old", "2026-06-27T00:00:00+00:00")]
    backend = _backend(client)

    # When: the caller updates a memory in scope.
    response = await backend.update_memory(
        _MEMORY_ID,
        MemoryUpdateRequest(scope=_scope(), content="new content"),
    )

    # Then: the update rides the proxied write handle instead of a 501.
    assert response.memory_id == _MEMORY_ID
    assert response.content == "new content"
    write_query, write_params = writer.writes[0]
    assert "SET f.object" in write_query
    assert write_params is not None
    assert write_params["memory_id"] == _MEMORY_ID


@pytest.mark.anyio
async def test_delete_memory_when_sdk_wraps_writes_in_a_delegating_proxy() -> None:
    # Given: the deployed SDK proxy shape around the graph write handle.
    writer = FakeGraphWriter()
    client = FakeMemoryClient(graph=FakeGraphWriteProxy(inner=writer))
    client.query.rows = [_memory_row(_MEMORY_ID, "old", "2026-06-27T00:00:00+00:00")]
    backend = _backend(client)

    # When: the caller deletes a memory in scope.
    response = await backend.delete_memory(
        _MEMORY_ID,
        MemoryDeleteRequest(scope=_scope()),
    )

    # Then: the delete rides the proxied write handle instead of a 501.
    assert response.memory_id == _MEMORY_ID
    write_query, write_params = writer.writes[0]
    assert "DETACH DELETE f" in write_query
    assert write_params == {"memory_id": _MEMORY_ID}


def _graph(client: "FakeMemoryClient") -> "FakeGraphWriter":
    graph = client.graph
    assert graph is not None
    if isinstance(graph, FakeGraphWriteProxy):
        return graph.inner
    return graph


def _backend(
    client: "FakeMemoryClient",
    fact_extractor: MemoryUnitExtractor | None = None,
    *,
    extraction_enabled: bool = False,
    extraction_context_turns: int = 10,
    extraction_mode: Literal["sync", "background"] = "sync",
) -> Neo4jAgentMemoryBackend:
    return Neo4jAgentMemoryBackend(
        Settings(
            gnosis_fact_extraction_enabled=extraction_enabled,
            gnosis_fact_extraction_context_turns=extraction_context_turns,
            gnosis_fact_extraction_mode=extraction_mode,
        ),
        memory_client_factory=FakeMemoryClientFactory(client),
        graph_store=FakeGraphStore(),
        fact_extractor=fact_extractor,
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


def _turn_row(memory_id: str, role: str, content: str) -> JsonObject:
    return {
        "id": memory_id,
        "subject": "bromigos:discord:private_user:pc-principal:789",
        "predicate": f"said_{role}",
        "object": content,
        "metadata": json.dumps(_scope_metadata()),
        "created_at": "2026-06-27T00:00:00+00:00",
        "updated_at": None,
    }


@dataclass(frozen=True, slots=True)
class ExtractionCall:
    conversation_date: str
    context_turns: tuple[ConversationTurn, ...]
    new_turns: tuple[ConversationTurn, ...]


@dataclass(slots=True)
class RecordingUnitExtractor:
    extraction: MemoryUnitExtraction | None = None
    calls: list[ExtractionCall] = field(default_factory=list)

    async def extract_units(
        self,
        *,
        conversation_date: str,
        context_turns: Sequence[ConversationTurn],
        new_turns: Sequence[ConversationTurn],
    ) -> MemoryUnitExtraction | None:
        self.calls.append(
            ExtractionCall(
                conversation_date=conversation_date,
                context_turns=tuple(context_turns),
                new_turns=tuple(new_turns),
            ),
        )
        return self.extraction


@dataclass(frozen=True, slots=True)
class FailingUnitExtractor:
    async def extract_units(
        self,
        *,
        conversation_date: str,
        context_turns: Sequence[ConversationTurn],
        new_turns: Sequence[ConversationTurn],
    ) -> MemoryUnitExtraction | None:
        _ = (conversation_date, context_turns, new_turns)
        raise APIConnectionError(request=httpx.Request("POST", "http://litellm.local"))


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
class FakeGraphWriteProxy:
    """Mirrors the SDK's ``_DeprecatedGraphProxy`` around the write handle.

    ``execute_write`` is reachable only through dynamic ``__getattr__``
    delegation, exactly like ``neo4j-agent-memory==0.5.0``'s bolt proxy.
    """

    inner: FakeGraphWriter

    def __getattr__(self, name: str) -> object:
        if name == "execute_write":
            return self.inner.execute_write
        raise AttributeError(name)


@dataclass(slots=True)
class FakeMemoryClient:
    short_term: FakeShortTermMemory = field(default_factory=FakeShortTermMemory)
    long_term: FakeLongTermMemory = field(default_factory=FakeLongTermMemory)
    query: FakeCypherQuery = field(default_factory=FakeCypherQuery)
    graph: FakeGraphWriter | FakeGraphWriteProxy | None = field(
        default_factory=FakeGraphWriter,
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
