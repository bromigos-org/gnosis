import json
from collections.abc import Sequence

from agents_memory.graph_events import PlannedGraphEvent
from agents_memory.models import GraphContextRequest, JsonValue

type CypherParameters = dict[str, JsonValue]


def upsert_parameters(event: PlannedGraphEvent) -> CypherParameters:
    message_id = _message_id(event)
    link_id = event.event.subject.id if event.node.node_type == "link" else None
    attachment_id = (
        event.event.subject.id if event.node.node_type == "attachment" else None
    )
    return {
        "event_id": event.event.event_id,
        "event_type": event.event.event_type.value,
        "idempotency_key": event.event.idempotency_key,
        "occurred_at": event.event.occurred_at,
        "observed_at": event.event.observed_at,
        "node_id": event.node.id,
        "node_type": event.node.node_type,
        "tenant_id": event.event.tenant_id,
        "agent_id": event.event.agent_id,
        "user_id": event.node.scope.user_id,
        "guild_id": event.node.scope.guild_id,
        "channel_id": event.node.scope.channel_id,
        "visibility": event.node.scope.visibility.value,
        "summary": event.node.summary,
        "deleted": event.node.deleted,
        "payload": json.dumps(
            event.node.payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "supporting_node_ids": list(event.supporting_node_ids),
        "semantic_node_ids": list(event.semantic_node_ids),
        "tenant_node_id": _node_id(
            event.event.tenant_id,
            "tenant",
            event.event.tenant_id,
        ),
        "agent_node_id": _node_id(event.event.tenant_id, "agent", event.event.agent_id),
        "client_node_id": _node_id(
            event.event.tenant_id,
            "client",
            event.event.source_client.value,
        ),
        "guild_node_id": _optional_node_id(
            event.event.tenant_id,
            "guild",
            event.event.scope.guild_id,
        ),
        "channel_node_id": _optional_node_id(
            event.event.tenant_id,
            "channel",
            event.event.scope.channel_id,
        ),
        "thread_node_id": _optional_node_id(
            event.event.tenant_id,
            "thread",
            event.event.discord.thread_id if event.event.discord is not None else None,
        ),
        "user_node_id": _node_id(event.event.tenant_id, "user", event.event.actor.id),
        "message_node_id": _optional_node_id(
            event.event.tenant_id,
            "message",
            message_id,
        ),
        "link_node_id": _optional_node_id(event.event.tenant_id, "link", link_id),
        "attachment_node_id": _optional_node_id(
            event.event.tenant_id,
            "attachment",
            attachment_id,
        ),
        "subject_node_id": event.node.id,
        "subject_node_type": event.node.node_type,
        "actor_id": event.event.actor.id,
        "actor_display_name": event.event.actor.display_name,
        "actor_is_bot": event.event.actor.is_bot,
        "source_client": event.event.source_client.value,
        "message_id": message_id,
        "message_content": _string_payload(event.event.payload, "content"),
        "link_url": _string_payload(event.event.payload, "url"),
        "attachment_filename": _string_payload(event.event.payload, "filename"),
        "has_guild": event.event.scope.guild_id is not None,
        "has_channel": event.event.scope.channel_id is not None,
        "has_thread": event.event.discord is not None
        and event.event.discord.thread_id is not None,
        "has_message": message_id is not None,
        "has_link": link_id is not None,
        "has_attachment": attachment_id is not None,
    }


def context_parameters(request: GraphContextRequest) -> CypherParameters:
    return {
        "tenant_id": request.scope.tenant_id,
        "agent_id": request.scope.agent_id,
        "user_id": request.scope.user_id,
        "guild_id": request.scope.guild_id,
        "channel_id": request.scope.channel_id,
        "limit": request.limit,
    }


def is_duplicate_result(rows: Sequence[dict[str, JsonValue]]) -> bool:
    if not rows:
        return False
    return rows[0].get("duplicate", False) is True


UPSERT_EVENT_CYPHER = """
MERGE (e:Event {tenant_id: $tenant_id, idempotency_key: $idempotency_key})
ON CREATE SET e.event_id = $event_id, e.event_type = $event_type,
  e.occurred_at = $occurred_at, e.observed_at = $observed_at,
  e.created_at = datetime(), e.was_created = true
ON MATCH SET e.was_created = false
WITH e, e.was_created = false AS duplicate
REMOVE e.was_created
WITH e, duplicate
MERGE (n:GraphNode {id: $node_id})
SET n.tenant_id = $tenant_id, n.agent_id = $agent_id,
  n.user_id = $user_id, n.guild_id = $guild_id, n.channel_id = $channel_id,
  n.visibility = $visibility, n.type = $node_type, n.summary = $summary,
  n.deleted = $deleted, n.payload = $payload, n.updated_at = datetime()
MERGE (e)-[:AFFECTS]->(n)
MERGE (t:Tenant {id: $tenant_node_id})
SET t.tenant_id = $tenant_id, t.updated_at = datetime()
MERGE (a:Agent {id: $agent_node_id})
SET a.tenant_id = $tenant_id, a.agent_id = $agent_id, a.updated_at = datetime()
MERGE (c:Client {id: $client_node_id})
SET c.tenant_id = $tenant_id, c.name = $source_client, c.updated_at = datetime()
MERGE (t)-[:OWNS_AGENT]->(a)
MERGE (t)-[:OWNS_CLIENT]->(c)
MERGE (a)-[:USES_CLIENT]->(c)
MERGE (e)-[:AFFECTS]->(t)
MERGE (e)-[:AFFECTS]->(a)
MERGE (e)-[:AFFECTS]->(c)
FOREACH (_ IN CASE WHEN $has_guild THEN [1] ELSE [] END |
  MERGE (g:Guild {id: $guild_node_id})
  SET g.tenant_id = $tenant_id, g.guild_id = $guild_id, g.updated_at = datetime()
  MERGE (t)-[:OWNS_GUILD]->(g)
  MERGE (e)-[:AFFECTS]->(g)
)
FOREACH (_ IN CASE WHEN $has_channel THEN [1] ELSE [] END |
  MERGE (ch:Channel {id: $channel_node_id})
  SET ch.tenant_id = $tenant_id, ch.guild_id = $guild_id,
    ch.channel_id = $channel_id, ch.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(ch)
)
FOREACH (_ IN CASE WHEN $has_guild AND $has_channel THEN [1] ELSE [] END |
  MERGE (g:Guild {id: $guild_node_id})
  MERGE (ch:Channel {id: $channel_node_id})
  MERGE (ch)-[:IN_GUILD]->(g)
)
FOREACH (_ IN CASE WHEN $has_thread THEN [1] ELSE [] END |
  MERGE (th:Channel {id: $thread_node_id})
  SET th.tenant_id = $tenant_id, th.guild_id = $guild_id,
    th.channel_id = $channel_id, th.kind = 'thread', th.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(th)
)
FOREACH (_ IN CASE WHEN $actor_id <> '' THEN [1] ELSE [] END |
  MERGE (u:User {id: $user_node_id})
  SET u.tenant_id = $tenant_id, u.user_id = $actor_id,
    u.display_name = $actor_display_name, u.is_bot = $actor_is_bot,
    u.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(u)
)
FOREACH (_ IN CASE WHEN $has_message THEN [1] ELSE [] END |
  MERGE (m:Message {id: $message_node_id})
  SET m.tenant_id = $tenant_id, m.message_id = $message_id,
    m.summary = $summary, m.content = $message_content,
    m.deleted = $deleted, m.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(m)
)
FOREACH (_ IN CASE WHEN $has_message AND $actor_id <> '' THEN [1] ELSE [] END |
  MERGE (u:User {id: $user_node_id})
  MERGE (m:Message {id: $message_node_id})
  MERGE (u)-[:AUTHORED]->(m)
)
FOREACH (_ IN CASE WHEN $has_message AND $has_channel THEN [1] ELSE [] END |
  MERGE (m:Message {id: $message_node_id})
  MERGE (ch:Channel {id: $channel_node_id})
  MERGE (m)-[:IN_CHANNEL]->(ch)
)
FOREACH (_ IN CASE WHEN $has_link THEN [1] ELSE [] END |
  MERGE (l:Link {id: $link_node_id})
  SET l.tenant_id = $tenant_id, l.url = $link_url, l.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(l)
)
FOREACH (_ IN CASE WHEN $has_link AND $has_message THEN [1] ELSE [] END |
  MERGE (l:Link {id: $link_node_id})
  MERGE (m:Message {id: $message_node_id})
  MERGE (l)-[:LINKED_FROM]->(m)
)
FOREACH (_ IN CASE WHEN $has_attachment THEN [1] ELSE [] END |
  MERGE (att:Attachment {id: $attachment_node_id})
  SET att.tenant_id = $tenant_id, att.filename = $attachment_filename,
    att.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(att)
)
FOREACH (_ IN CASE WHEN $has_attachment AND $has_message THEN [1] ELSE [] END |
  MERGE (att:Attachment {id: $attachment_node_id})
  MERGE (m:Message {id: $message_node_id})
  MERGE (att)-[:ATTACHED_TO]->(m)
)
RETURN duplicate AS duplicate
"""


CONTEXT_CYPHER = """
MATCH (n:GraphNode {tenant_id: $tenant_id, agent_id: $agent_id})
WHERE n.visibility = 'global'
  OR n.visibility = 'tenant'
  OR n.visibility IN ['agent_private', 'agent_shared']
  OR (n.visibility = 'private_user' AND n.user_id = $user_id)
  OR (n.visibility = 'guild' AND n.guild_id = $guild_id)
  OR (
    n.visibility = 'channel'
    AND n.guild_id = $guild_id
    AND n.channel_id = $channel_id
  )
RETURN n.id AS id, n.type AS type, n.summary AS summary, n.deleted AS deleted
ORDER BY n.updated_at DESC
LIMIT $limit
"""


def _node_id(tenant_id: str, node_type: str, raw_id: str) -> str:
    return f"tenant:{tenant_id}:{node_type}:{raw_id}"


def _optional_node_id(tenant_id: str, node_type: str, raw_id: str | None) -> str | None:
    if raw_id is None:
        return None
    return _node_id(tenant_id, node_type, raw_id)


def _message_id(event: PlannedGraphEvent) -> str | None:
    if event.event.subject.type == "message":
        return event.event.subject.id
    if event.event.discord is not None and event.event.discord.message_id is not None:
        return event.event.discord.message_id
    value = event.event.payload.get("message_id")
    if isinstance(value, str):
        return value
    return None


def _string_payload(payload: dict[str, JsonValue], key: str) -> str:
    value = payload.get(key, "")
    if isinstance(value, str):
        return value
    return ""
