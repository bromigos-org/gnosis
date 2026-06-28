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


def test_discord_category_upsert_cypher_fans_out_channel_hierarchy() -> None:
    # Given: Discord category and child channel events are planned for Neo4j.
    category = _category_event()
    child = _channel_child_event()

    # When: Neo4j parameters are built for the hierarchy events.
    category_parameters = upsert_parameters(plan_event(category))
    child_parameters = upsert_parameters(plan_event(child))

    # Then: fixed category labels and containment relationships are available.
    assert "MERGE (cat:Category" in UPSERT_EVENT_CYPHER
    assert "MERGE (cat)-[:IN_GUILD]->(g)" in UPSERT_EVENT_CYPHER
    assert "MERGE (ch)-[:IN_CATEGORY]->(cat)" in UPSERT_EVENT_CYPHER
    assert category_parameters["category_node_id"] == (
        "tenant:bromigos:category:category-111"
    )
    assert category_parameters["has_category_subject"] is True
    assert child_parameters["category_node_id"] == (
        "tenant:bromigos:category:category-111"
    )
    assert child_parameters["has_channel_category"] is True


def test_discord_role_upsert_cypher_fans_out_guild_role_ownership() -> None:
    # Given: a Discord role event is planned for Neo4j persistence.
    event = _role_event()

    # When: Neo4j parameters are built for the role event.
    parameters = upsert_parameters(plan_event(event))

    # Then: roles are typed and owned by their guild without dynamic labels.
    assert "MERGE (r:Role" in UPSERT_EVENT_CYPHER
    assert "MERGE (g)-[:OWNS_ROLE]->(r)" in UPSERT_EVENT_CYPHER
    assert parameters["role_node_id"] == "tenant:bromigos:role:role-222"
    assert parameters["role_id"] == "role-222"
    assert parameters["role_name"] == "hall monitor"
    assert parameters["has_role"] is True


def test_discord_role_upsert_preserves_existing_name_when_payload_is_blank() -> None:
    event = _role_event().model_copy(
        update={"payload": {"role_id": "role-222", "guild_id": "guild-123", "name": ""}},
    )

    parameters = upsert_parameters(plan_event(event))

    assert parameters["role_name"] == ""
    assert "coalesce(nullif($role_name, ''), r.name)" in UPSERT_EVENT_CYPHER


def test_discord_member_update_cypher_fans_out_user_role_assignments() -> None:
    # Given: a Discord member update includes the member's current role IDs.
    event = _member_event()

    # When: Neo4j parameters are built for the member update.
    parameters = upsert_parameters(plan_event(event))

    # Then: the writer can repair user typing and role assignment edges.
    assert (
        "FOREACH (member_role_node_id IN $member_role_node_ids" in UPSERT_EVENT_CYPHER
    )
    assert "MERGE (member)-[:HAS_ROLE]->(role)" in UPSERT_EVENT_CYPHER
    assert parameters["member_user_node_id"] == "tenant:bromigos:user:user-789"
    assert parameters["member_role_node_ids"] == [
        "tenant:bromigos:role:role-222",
        "tenant:bromigos:role:role-333",
    ]
    assert parameters["has_member_roles"] is True


def test_pc_principal_topology_event_enum_accepts_new_event_types() -> None:
    # Given: PC-Principal emits explicit user and member-role topology facts.
    event_types = (
        "user_discovered",
        "member_role_assigned",
        "member_role_unassigned",
    )

    # When: those wire values are parsed through the strict event enum.
    parsed = tuple(ClientEventType(event_type) for event_type in event_types)

    # Then: only the known PC-Principal topology event types are accepted.
    assert parsed == (
        ClientEventType.USER_DISCOVERED,
        ClientEventType.MEMBER_ROLE_ASSIGNED,
        ClientEventType.MEMBER_ROLE_UNASSIGNED,
    )


def test_pc_principal_user_discovered_cypher_repairs_bot_typing() -> None:
    # Given: PC-Principal discovers a bot user before message history is replayed.
    event = _user_discovered_event()

    # When: Neo4j parameters are built for the discovered user.
    parameters = upsert_parameters(plan_event(event))

    # Then: the subject user is upserted as both User and Bot through fixed Cypher.
    assert "SET u:Bot" in UPSERT_EVENT_CYPHER
    assert parameters["node_type"] == "user"
    assert parameters["user_node_id"] == "tenant:bromigos:user:bot-007"
    assert parameters["actor_is_bot"] is True
    semantic_node_ids = parameters["semantic_node_ids"]
    assert isinstance(semantic_node_ids, list)
    assert "tenant:bromigos:bot:bot-007" in semantic_node_ids


def test_pc_principal_member_role_assigned_cypher_creates_current_has_role() -> None:
    # Given: PC-Principal emits an explicit role assignment fact.
    event = _member_role_event(ClientEventType.MEMBER_ROLE_ASSIGNED)

    # When: Neo4j parameters are built for the assignment.
    parameters = upsert_parameters(plan_event(event))

    # Then: the writer targets the member and role nodes for a current HAS_ROLE edge.
    assert "MERGE (member)-[:HAS_ROLE]->(role)" in UPSERT_EVENT_CYPHER
    assert parameters["node_id"] == "tenant:bromigos:user:user-789"
    assert parameters["member_user_node_id"] == "tenant:bromigos:user:user-789"
    assert parameters["member_role_node_ids"] == ["tenant:bromigos:role:role-222"]
    assert parameters["role_node_id"] == "tenant:bromigos:role:role-222"
    assert parameters["has_member_roles"] is True
    assert parameters["is_member_role_assignment"] is True
    assert parameters["is_member_role_unassignment"] is False


def test_pc_principal_member_role_unassigned_cypher_removes_current_has_role() -> None:
    # Given: PC-Principal emits an explicit role unassignment fact.
    event = _member_role_event(ClientEventType.MEMBER_ROLE_UNASSIGNED)

    # When: Neo4j parameters are built for the unassignment.
    parameters = upsert_parameters(plan_event(event))

    # Then: the writer deletes the current HAS_ROLE edge instead of recreating it.
    assert "DELETE current_role" in UPSERT_EVENT_CYPHER
    assert parameters["node_id"] == "tenant:bromigos:user:user-789"
    assert parameters["member_user_node_id"] == "tenant:bromigos:user:user-789"
    assert parameters["member_role_node_ids"] == []
    assert parameters["role_node_id"] == "tenant:bromigos:role:role-222"
    assert parameters["has_member_roles"] is False
    assert parameters["is_member_role_assignment"] is False
    assert parameters["is_member_role_unassignment"] is True


def test_discord_bot_actor_cypher_adds_queryable_bot_typing() -> None:
    # Given: a Discord event is authored by a bot actor.
    event = _message_event().model_copy(
        update={
            "actor": ClientEventActor(
                id="bot-007",
                display_name="PC Principal",
                is_bot=True,
            ),
        },
    )

    # When: Neo4j parameters are built for the bot-authored event.
    parameters = upsert_parameters(plan_event(event))

    # Then: the User node is additionally typed as Bot through fixed Cypher.
    assert "SET u:Bot" in UPSERT_EVENT_CYPHER
    assert parameters["user_node_id"] == "tenant:bromigos:user:bot-007"
    assert parameters["actor_is_bot"] is True


def test_discord_media_event_does_not_overwrite_parent_message_properties() -> None:
    # Given: link discovery points at an existing parent message.
    event = _media_event(
        event_id="link_discovered:message-999:example",
        event_type=ClientEventType.LINK_DISCOVERED,
        subject_id="https://example.invalid/docs",
        subject_type="link",
        payload={"url": "https://example.invalid/docs", "message_id": "message-999"},
    )

    # When: Neo4j parameters are built for the media event.
    parameters = upsert_parameters(plan_event(event))

    # Then: media links to the parent Message but cannot write Message content/state.
    assert parameters["has_message"] is True
    assert parameters["has_message_subject"] is False
    assert "WHEN $has_message_subject THEN [1] ELSE []" in UPSERT_EVENT_CYPHER


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


def _category_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id="channel_created:category-111",
        event_type=ClientEventType.CHANNEL_CREATED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="channel_created:category-111",
        scope=_scope(),
        actor=ClientEventActor(id="system", display_name="discord", is_bot=True),
        subject=ClientEventSubject(id="category-111", type="category"),
        payload={
            "channel_id": "category-111",
            "guild_id": "guild-123",
            "name": "School Board",
            "channel_type": "category",
        },
        discord=DiscordEventContext(guild_id="guild-123", channel_id="category-111"),
    )


def _channel_child_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id="channel_created:channel-456",
        event_type=ClientEventType.CHANNEL_CREATED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="channel_created:channel-456",
        scope=_scope(),
        actor=ClientEventActor(id="system", display_name="discord", is_bot=True),
        subject=ClientEventSubject(
            id="channel-456",
            type="channel",
            parent_id="category-111",
        ),
        payload={
            "channel_id": "channel-456",
            "guild_id": "guild-123",
            "name": "announcements",
            "category_id": "category-111",
        },
        discord=DiscordEventContext(guild_id="guild-123", channel_id="channel-456"),
    )


def _role_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id="role_created:role-222",
        event_type=ClientEventType.ROLE_CREATED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="role_created:role-222",
        scope=_scope(),
        actor=ClientEventActor(id="system", display_name="discord", is_bot=True),
        subject=ClientEventSubject(id="role-222", type="role"),
        payload={
            "role_id": "role-222",
            "guild_id": "guild-123",
            "name": "hall monitor",
        },
        discord=DiscordEventContext(guild_id="guild-123"),
    )


def _member_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id="member_updated:user-789",
        event_type=ClientEventType.MEMBER_UPDATED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="member_updated:user-789",
        scope=_scope(),
        actor=ClientEventActor(id="system", display_name="discord", is_bot=True),
        subject=ClientEventSubject(id="user-789", type="member"),
        payload={
            "user_id": "user-789",
            "guild_id": "guild-123",
            "display_name": "cartman",
            "roles": ["role-222", "role-333"],
            "previous_roles": ["role-222"],
        },
        discord=DiscordEventContext(guild_id="guild-123"),
    )


def _user_discovered_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id="user_discovered:bot-007",
        event_type=ClientEventType.USER_DISCOVERED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="user_discovered:bot-007",
        scope=_scope().model_copy(update={"visibility": MemoryVisibility.GUILD}),
        actor=ClientEventActor(id="bot-007", display_name="PC Principal", is_bot=True),
        subject=ClientEventSubject(id="bot-007", type="bot", parent_id="guild-123"),
        payload={
            "guild_id": "guild-123",
            "user_id": "bot-007",
            "display_name": "PC Principal",
            "is_bot": True,
            "user_type": "bot",
        },
        discord=DiscordEventContext(guild_id="guild-123"),
    )


def _member_role_event(event_type: ClientEventType) -> ClientEvent:
    return ClientEvent(
        tenant_id="bromigos",
        source_client=SourceClient.DISCORD,
        agent_id="pc-principal",
        event_id=f"{event_type.value}:user-789:role-222",
        event_type=event_type,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key=f"{event_type.value}:user-789:role-222",
        scope=_scope().model_copy(update={"visibility": MemoryVisibility.GUILD}),
        actor=ClientEventActor(id="user-789", display_name="cartman", is_bot=False),
        subject=ClientEventSubject(
            id=f"{event_type.value}:user-789:role-222",
            type="member_role_assignment",
            parent_id="user-789",
        ),
        payload={
            "guild_id": "guild-123",
            "user_id": "user-789",
            "member_id": "user-789",
            "role_id": "role-222",
        },
        discord=DiscordEventContext(guild_id="guild-123"),
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
