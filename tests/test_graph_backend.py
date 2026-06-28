from dataclasses import dataclass

import pytest
from pydantic import SecretStr

from agents_memory.backend import Neo4jAgentMemoryBackend
from agents_memory.graph_cypher import upsert_parameters
from agents_memory.graph_events import plan_event
from agents_memory.graph_store import DirectNeo4jGraphStore, InMemoryGraphExecutor
from agents_memory.models import (
    ClientEvent,
    ClientEventActor,
    ClientEventSubject,
    ClientEventType,
    DiscordEventContext,
    EventIngestStatus,
    GraphContextRequest,
    MemoryScope,
    MemoryVisibility,
    SourceClient,
)
from agents_memory.settings import Settings


@pytest.mark.anyio
async def test_discord_message_upsert_and_context_retrieval() -> None:
    # Given: a direct graph store receives a Discord channel message event.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    backend = Neo4jAgentMemoryBackend(_settings(), graph_store=store)
    event = _message_event(
        _MessageEventValues(content="remember the plasma conduit"),
    )

    # When: the event is ingested and graph context is requested in the same scope.
    result = await backend.ingest_event(event)
    context = await backend.get_graph_context(
        GraphContextRequest(scope=event.scope, query="plasma", limit=4),
    )

    # Then: current graph state and structured facts are visible without Neo4j.
    assert result.status == EventIngestStatus.ACCEPTED
    assert context.context == "message message-999: remember the plasma conduit"
    assert context.facts == [
        {
            "id": "tenant:bromigos:message:message-999",
            "type": "message",
            "scope": "channel",
            "summary": "message message-999: remember the plasma conduit",
            "deleted": False,
        },
    ]


@pytest.mark.anyio
async def test_graph_context_does_not_cross_channel_scope() -> None:
    # Given: two channel-scoped events in the same tenant and guild.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    first = _message_event(
        _MessageEventValues(
            message_id="message-1",
            channel_id="channel-1",
            content="alpha",
        ),
    )
    second = _message_event(
        _MessageEventValues(
            message_id="message-2",
            channel_id="channel-2",
            content="beta",
        ),
    )
    _ = await store.ingest_event(first)
    _ = await store.ingest_event(second)

    # When: graph context is requested from the first channel scope.
    context = await store.get_context(
        GraphContextRequest(scope=first.scope, query="channel", limit=10),
    )

    # Then: facts from sibling channels never leak into the response.
    assert context.context == "message message-1: alpha"
    assert [fact["id"] for fact in context.facts] == [
        "tenant:bromigos:message:message-1",
    ]


@pytest.mark.anyio
async def test_duplicate_idempotency_key_is_noop_duplicate() -> None:
    # Given: the graph store has already accepted an event idempotency key.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    event = _message_event()
    accepted = await store.ingest_event(event)

    # When: the same event is delivered again.
    duplicate = await store.ingest_event(event)

    # Then: the duplicate is reported without changing graph state or history.
    assert accepted.status == EventIngestStatus.ACCEPTED
    assert duplicate.status == EventIngestStatus.DUPLICATE
    assert duplicate.reason == "idempotency key already ingested"
    assert await store.event_count() == 1


@pytest.mark.anyio
async def test_message_delete_creates_tombstone() -> None:
    # Given: a Discord message exists in graph state.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    created = _message_event(_MessageEventValues(content="remove me"))
    deleted = _message_event(
        _MessageEventValues(
            event_id="message_deleted:message-999",
            event_type=ClientEventType.MESSAGE_DELETED,
            idempotency_key="message_deleted:message-999",
            content="",
        ),
    )
    _ = await store.ingest_event(created)

    # When: a delete event is ingested.
    result = await store.ingest_event(deleted)
    context = await store.get_context(
        GraphContextRequest(scope=created.scope, query="message", limit=4),
    )

    # Then: current state is tombstoned while both events remain in history.
    assert result.status == EventIngestStatus.ACCEPTED
    assert context.facts == [
        {
            "id": "tenant:bromigos:message:message-999",
            "type": "message",
            "scope": "channel",
            "summary": "message message-999: deleted",
            "deleted": True,
        },
    ]
    assert await store.event_count() == 2


@pytest.mark.anyio
async def test_channel_rename_updates_current_state_preserving_history() -> None:
    # Given: a channel was created with its original name.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    created = _channel_event(name="general")
    renamed = _channel_event(
        event_id="channel_updated:channel-456",
        event_type=ClientEventType.CHANNEL_UPDATED,
        idempotency_key="channel_updated:channel-456",
        name="operations",
    )
    _ = await store.ingest_event(created)

    # When: the channel rename arrives.
    result = await store.ingest_event(renamed)
    context = await store.get_context(
        GraphContextRequest(scope=created.scope, query="operations", limit=4),
    )

    # Then: the current channel fact reflects the rename and both events remain.
    assert result.status == EventIngestStatus.ACCEPTED
    assert context.context == "channel channel-456: operations"
    assert context.facts[0]["summary"] == "channel channel-456: operations"
    assert await store.event_count() == 2


def test_upsert_parameters_serializes_message_payload_for_neo4j() -> None:
    # Given: a Discord backfill message carries structured payload metadata.
    event = _message_event().model_copy(
        update={
            "payload": {
                "message_id": "message-999",
                "channel_id": "channel-456",
                "guild_id": "guild-123",
                "content": "remember this",
                "attachment_count": 2,
                "source_marker": "backfill",
            },
        },
    )

    # When: the event is converted to Neo4j upsert parameters.
    parameters = upsert_parameters(plan_event(event))

    # Then: the payload property is a deterministic scalar, not a Neo4j MAP.
    assert parameters["payload"] == (
        '{"attachment_count":2,"channel_id":"channel-456","content":"remember this",'
        '"guild_id":"guild-123","message_id":"message-999","source_marker":"backfill"}'
    )
    assert not isinstance(parameters["payload"], dict)


def _scope(*, channel_id: str = "channel-456") -> MemoryScope:
    return MemoryScope(
        tenant_id="bromigos",
        space_id="discord",
        agent_id="pc-principal",
        session_id=f"guild:guild-123:channel:{channel_id}",
        user_id="user-789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="guild-123",
        channel_id=channel_id,
    )


def _settings() -> Settings:
    token = SecretStr("test-token").get_secret_value()
    neo4j_password = SecretStr("test-password").get_secret_value()
    return Settings(
        agents_memory_token=token,
        neo4j_uri="bolt://neo4j.local:7687",
        neo4j_password=neo4j_password,
        litellm_base_url="http://litellm.local/v1",
        litellm_api_key="test-litellm-key",
    )


@dataclass(frozen=True, slots=True)
class _MessageEventValues:
    event_id: str = "message_created:message-999"
    event_type: ClientEventType = ClientEventType.MESSAGE_CREATED
    idempotency_key: str = "message_created:message-999"
    message_id: str = "message-999"
    channel_id: str = "channel-456"
    content: str = "remember this"


def _message_event(
    values: _MessageEventValues | None = None,
) -> ClientEvent:
    event_values = values or _MessageEventValues()
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id=event_values.event_id,
        event_type=event_values.event_type,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key=event_values.idempotency_key,
        scope=_scope(channel_id=event_values.channel_id),
        actor=ClientEventActor(id="user-789", display_name="cartman", is_bot=False),
        subject=ClientEventSubject(
            id=event_values.message_id,
            type="message",
            parent_id=event_values.channel_id,
        ),
        payload={
            "message_id": event_values.message_id,
            "channel_id": event_values.channel_id,
            "guild_id": "guild-123",
            "content": event_values.content,
        },
        discord=DiscordEventContext(
            guild_id="guild-123",
            channel_id=event_values.channel_id,
            message_id=event_values.message_id,
        ),
    )


def _channel_event(
    *,
    event_id: str = "channel_created:channel-456",
    event_type: ClientEventType = ClientEventType.CHANNEL_CREATED,
    idempotency_key: str = "channel_created:channel-456",
    name: str,
) -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id=event_id,
        event_type=event_type,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key=idempotency_key,
        scope=_scope(),
        actor=ClientEventActor(id="system", display_name="discord", is_bot=True),
        subject=ClientEventSubject(id="channel-456", type="channel"),
        payload={
            "channel_id": "channel-456",
            "guild_id": "guild-123",
            "name": name,
        },
        discord=DiscordEventContext(guild_id="guild-123", channel_id="channel-456"),
    )
