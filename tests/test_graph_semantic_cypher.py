import pytest

from agents_memory.graph_cypher import UPSERT_EVENT_CYPHER, upsert_parameters
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
    JsonObject,
    MemoryScope,
    MemoryVisibility,
    SourceClient,
)


def test_discord_message_upsert_cypher_fans_out_semantic_nodes() -> None:
    # Given: a Discord message event is planned for Neo4j persistence.
    event = _message_event()

    # When: the generated writer Cypher and parameters are inspected.
    parameters = upsert_parameters(plan_event(event))

    # Then: the audit event remains and fixed-label semantic nodes are upserted.
    assert "MERGE (e:Event" in UPSERT_EVENT_CYPHER
    assert "MERGE (m:Message" in UPSERT_EVENT_CYPHER
    assert "MERGE (ch:Channel" in UPSERT_EVENT_CYPHER
    assert "MERGE (u:User" in UPSERT_EVENT_CYPHER
    assert "MERGE (g:Guild" in UPSERT_EVENT_CYPHER
    assert "MERGE (a:Agent" in UPSERT_EVENT_CYPHER
    assert "MERGE (c:Client" in UPSERT_EVENT_CYPHER
    assert "MERGE (t:Tenant" in UPSERT_EVENT_CYPHER
    assert "MERGE (u)-[:AUTHORED]->(m)" in UPSERT_EVENT_CYPHER
    assert "MERGE (m)-[:IN_CHANNEL]->(ch)" in UPSERT_EVENT_CYPHER
    assert "MERGE (ch)-[:IN_GUILD]->(g)" in UPSERT_EVENT_CYPHER
    assert parameters["message_node_id"] == "tenant:bromigos:message:message-999"
    assert parameters["channel_node_id"] == "tenant:bromigos:channel:channel-456"
    assert parameters["user_node_id"] == "tenant:bromigos:user:user-789"
    assert parameters["guild_node_id"] == "tenant:bromigos:guild:guild-123"
    assert parameters["tenant_node_id"] == "tenant:bromigos:tenant:bromigos"


def test_discord_link_and_attachment_upsert_cypher_fans_out_media_nodes() -> None:
    # Given: Discord link and attachment discovery events point at a parent message.
    link = _media_event(
        event_id="link_discovered:message-999:example",
        event_type=ClientEventType.LINK_DISCOVERED,
        subject_id="https://example.invalid/docs",
        subject_type="link",
        payload={"url": "https://example.invalid/docs", "message_id": "message-999"},
    )
    attachment = _media_event(
        event_id="attachment_discovered:message-999:file-1",
        event_type=ClientEventType.ATTACHMENT_DISCOVERED,
        subject_id="attachment-1",
        subject_type="attachment",
        payload={"filename": "photo.png", "message_id": "message-999"},
    )

    # When: Neo4j parameters are built for both event types.
    link_parameters = upsert_parameters(plan_event(link))
    attachment_parameters = upsert_parameters(plan_event(attachment))

    # Then: the writer can create typed media nodes and connect them to the message.
    assert "MERGE (l:Link" in UPSERT_EVENT_CYPHER
    assert "MERGE (att:Attachment" in UPSERT_EVENT_CYPHER
    assert "MERGE (l)-[:LINKED_FROM]->(m)" in UPSERT_EVENT_CYPHER
    assert "MERGE (att)-[:ATTACHED_TO]->(m)" in UPSERT_EVENT_CYPHER
    assert link_parameters["link_node_id"] == (
        "tenant:bromigos:link:https://example.invalid/docs"
    )
    assert link_parameters["message_node_id"] == "tenant:bromigos:message:message-999"
    assert attachment_parameters["attachment_node_id"] == (
        "tenant:bromigos:attachment:attachment-1"
    )
    assert attachment_parameters["message_node_id"] == (
        "tenant:bromigos:message:message-999"
    )


@pytest.mark.anyio
async def test_duplicate_replay_repairs_semantic_graph_state() -> None:
    # Given: the executor previously accepted an event before semantic state existed.
    executor = InMemoryGraphExecutor()
    store = DirectNeo4jGraphStore(executor=executor)
    event = _message_event()
    accepted = await store.ingest_event(event)

    # When: compatibility state is missing and the same payload is replayed.
    executor.clear_current_nodes_for_test()
    duplicate = await store.ingest_event(event)
    context = await store.get_context(
        GraphContextRequest(scope=event.scope, query="message", limit=4),
    )

    # Then: duplicate status is preserved while current graph state is repaired.
    assert accepted.status == EventIngestStatus.ACCEPTED
    assert duplicate.status == EventIngestStatus.DUPLICATE
    assert await store.event_count() == 1
    assert context.context == "message message-999: remember this"
    assert executor.semantic_node_ids_for_test() == {
        "tenant:bromigos:agent:pc-principal",
        "tenant:bromigos:channel:channel-456",
        "tenant:bromigos:client:discord",
        "tenant:bromigos:guild:guild-123",
        "tenant:bromigos:message:message-999",
        "tenant:bromigos:tenant:bromigos",
        "tenant:bromigos:user:user-789",
    }


def _scope() -> MemoryScope:
    return MemoryScope(
        tenant_id="bromigos",
        space_id="discord",
        agent_id="pc-principal",
        session_id="guild:guild-123:channel:channel-456",
        user_id="user-789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="guild-123",
        channel_id="channel-456",
    )


def _message_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id="message_created:message-999",
        event_type=ClientEventType.MESSAGE_CREATED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="message_created:message-999",
        scope=_scope(),
        actor=ClientEventActor(id="user-789", display_name="cartman", is_bot=False),
        subject=ClientEventSubject(
            id="message-999",
            type="message",
            parent_id="channel-456",
        ),
        payload={
            "message_id": "message-999",
            "channel_id": "channel-456",
            "guild_id": "guild-123",
            "content": "remember this",
        },
        discord=DiscordEventContext(
            guild_id="guild-123",
            channel_id="channel-456",
            message_id="message-999",
        ),
    )


def _media_event(
    *,
    event_id: str,
    event_type: ClientEventType,
    subject_id: str,
    subject_type: str,
    payload: JsonObject,
) -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id=event_id,
        event_type=event_type,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key=event_id,
        scope=_scope(),
        actor=ClientEventActor(id="user-789", display_name="cartman", is_bot=False),
        subject=ClientEventSubject(
            id=subject_id,
            type=subject_type,
            parent_id="message-999",
        ),
        payload=payload,
        discord=DiscordEventContext(
            guild_id="guild-123",
            channel_id="channel-456",
            message_id="message-999",
        ),
    )
