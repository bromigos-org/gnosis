import json
from collections.abc import Sequence

from agents_memory.graph_events import PlannedGraphEvent
from agents_memory.models import ClientEventType, GraphContextRequest, JsonValue

type CypherParameters = dict[str, JsonValue]


def upsert_parameters(event: PlannedGraphEvent) -> CypherParameters:
    message_id = _message_id(event)
    link_id = event.event.subject.id if event.node.node_type == "link" else None
    attachment_id = (
        event.event.subject.id if event.node.node_type == "attachment" else None
    )
    category_id = _category_id(event)
    channel_name = _channel_name(event)
    channel_kind = _channel_kind(event)
    role_id = _role_id(event)
    member_user_id = _member_user_id(event)
    member_role_ids = _member_role_ids(event)
    user_id = _user_id(event)
    is_bot_user = _is_bot_user(event)
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
        "user_node_id": _node_id(event.event.tenant_id, "user", user_id),
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
        "category_node_id": _optional_node_id(
            event.event.tenant_id,
            "category",
            category_id,
        ),
        "role_node_id": _optional_node_id(event.event.tenant_id, "role", role_id),
        "member_user_node_id": _optional_node_id(
            event.event.tenant_id,
            "user",
            member_user_id,
        ),
        "member_role_node_ids": [
            _node_id(event.event.tenant_id, "role", member_role_id)
            for member_role_id in member_role_ids
        ],
        "subject_node_id": event.node.id,
        "subject_node_type": event.node.node_type,
        "actor_id": event.event.actor.id,
        "actor_display_name": _display_name(event),
        "actor_is_bot": is_bot_user,
        "user_identity_id": user_id,
        "category_id": category_id,
        "category_name": _category_name(event),
        "channel_name": channel_name,
        "channel_kind": channel_kind,
        "role_id": role_id,
        "role_name": _string_payload(event.event.payload, "name"),
        "member_user_id": member_user_id,
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
        "has_message_subject": event.node.node_type == "message"
        and message_id is not None,
        "has_link": link_id is not None,
        "has_attachment": attachment_id is not None,
        "has_category": category_id is not None,
        "has_category_subject": event.node.node_type == "category"
        and category_id is not None,
        "has_channel_category": event.node.node_type == "channel"
        and category_id is not None,
        "has_role": role_id is not None,
        "has_member_user": member_user_id is not None,
        "has_member_roles": bool(member_role_ids),
        "has_member_role_snapshot": _has_member_role_snapshot(event),
        "has_user_identity": user_id != "",
        "is_member_role_assignment": event.event.event_type
        == ClientEventType.MEMBER_ROLE_ASSIGNED,
        "is_member_role_unassignment": event.event.event_type
        == ClientEventType.MEMBER_ROLE_UNASSIGNED,
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
    ch.channel_id = $channel_id,
    ch.name = coalesce(nullif($channel_name, ''), ch.name),
    ch.kind = coalesce(nullif($channel_kind, ''), ch.kind),
    ch.updated_at = datetime()
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
FOREACH (_ IN CASE WHEN $has_user_identity THEN [1] ELSE [] END |
  MERGE (u:User {id: $user_node_id})
  SET u.tenant_id = $tenant_id, u.user_id = $user_identity_id,
    u.display_name = $actor_display_name, u.is_bot = $actor_is_bot,
    u.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(u)
)
FOREACH (_ IN CASE WHEN $has_user_identity AND $actor_is_bot THEN [1] ELSE [] END |
  MERGE (u:User {id: $user_node_id})
  SET u:Bot
)
FOREACH (_ IN CASE WHEN $has_message_subject THEN [1] ELSE [] END |
  MERGE (m:Message {id: $message_node_id})
  SET m.tenant_id = $tenant_id, m.message_id = $message_id,
    m.summary = $summary, m.content = $message_content,
    m.deleted = $deleted, m.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(m)
)
FOREACH (_ IN CASE WHEN $has_message AND NOT $has_message_subject THEN [1] ELSE [] END |
  MERGE (m:Message {id: $message_node_id})
  SET m.tenant_id = $tenant_id, m.message_id = $message_id,
    m.updated_at = datetime()
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
FOREACH (_ IN CASE WHEN $has_category THEN [1] ELSE [] END |
  MERGE (cat:Category {id: $category_node_id})
  SET cat.tenant_id = $tenant_id, cat.guild_id = $guild_id,
    cat.category_id = $category_id,
    cat.name = coalesce(nullif($category_name, ''), cat.name),
    cat.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(cat)
)
FOREACH (_ IN CASE WHEN $has_category AND $has_guild THEN [1] ELSE [] END |
  MERGE (cat:Category {id: $category_node_id})
  MERGE (g:Guild {id: $guild_node_id})
  MERGE (cat)-[:IN_GUILD]->(g)
)
FOREACH (_ IN CASE WHEN $has_channel_category THEN [1] ELSE [] END |
  MERGE (ch:Channel {id: $channel_node_id})
  MERGE (cat:Category {id: $category_node_id})
  MERGE (ch)-[:IN_CATEGORY]->(cat)
)
FOREACH (_ IN CASE WHEN $has_role THEN [1] ELSE [] END |
  MERGE (r:Role {id: $role_node_id})
  SET r.tenant_id = $tenant_id, r.guild_id = $guild_id,
    r.role_id = $role_id,
    r.name = coalesce(nullif($role_name, ''), r.name),
    r.deleted = $deleted, r.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(r)
)
FOREACH (_ IN CASE WHEN $has_role AND $has_guild THEN [1] ELSE [] END |
  MERGE (g:Guild {id: $guild_node_id})
  MERGE (r:Role {id: $role_node_id})
  MERGE (g)-[:OWNS_ROLE]->(r)
)
FOREACH (_ IN CASE WHEN $has_member_user THEN [1] ELSE [] END |
  MERGE (member:User {id: $member_user_node_id})
  SET member.tenant_id = $tenant_id, member.user_id = $member_user_id,
    member.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(member)
)
FOREACH (member_role_node_id IN $member_role_node_ids |
  MERGE (member:User {id: $member_user_node_id})
  MERGE (role:Role {id: member_role_node_id})
  MERGE (member)-[:HAS_ROLE]->(role)
  MERGE (e)-[:AFFECTS]->(role)
)
WITH e, duplicate
OPTIONAL MATCH (:User {id: $member_user_node_id})-[stale_role:HAS_ROLE]->(
  stale_role_node:Role
)
FOREACH (_ IN CASE
  WHEN $has_member_role_snapshot
    AND stale_role IS NOT NULL
    AND stale_role_node.id IN $member_role_node_ids THEN []
  WHEN $has_member_role_snapshot AND stale_role IS NOT NULL THEN [1]
  ELSE []
END | DELETE stale_role)
WITH e, duplicate
OPTIONAL MATCH (:User {id: $member_user_node_id})-[current_role:HAS_ROLE]->(
  :Role {id: $role_node_id}
)
FOREACH (_ IN CASE
  WHEN $is_member_role_unassignment AND current_role IS NOT NULL THEN [1]
  ELSE []
END | DELETE current_role)
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


def _category_id(event: PlannedGraphEvent) -> str | None:
    if event.node.node_type == "category":
        return event.event.subject.id
    value = event.event.payload.get("category_id", event.event.subject.parent_id)
    if isinstance(value, str) and value != "":
        if event.event.event_type not in {
            ClientEventType.CHANNEL_CREATED,
            ClientEventType.CHANNEL_UPDATED,
            ClientEventType.CHANNEL_DELETED,
        }:
            return None
        return value
    return None


def _category_name(event: PlannedGraphEvent) -> str:
    if event.node.node_type == "category":
        return _string_payload(event.event.payload, "name")
    return _string_payload(event.event.payload, "category_name")


def _channel_name(event: PlannedGraphEvent) -> str:
    if event.node.node_type in {"category", "channel", "thread"}:
        return _string_payload(event.event.payload, "name")
    return ""


def _channel_kind(event: PlannedGraphEvent) -> str:
    if event.node.node_type == "category":
        return "category"
    if event.node.node_type == "thread":
        return "thread"
    return _normalize_channel_kind(event.event.payload.get("channel_type"))


def _normalize_channel_kind(value: JsonValue | None) -> str:
    match value:
        case str():
            return _normalize_channel_kind_text(value)
        case int():
            return _normalize_channel_kind_number(value)
        case _:
            return ""


def _normalize_channel_kind_text(value: str) -> str:
    normalized = value.strip().lower().removeprefix("guild_")
    match normalized:
        case "0" | "text":
            return "text"
        case "2" | "voice":
            return "voice"
        case "4" | "category":
            return "category"
        case (
            "10"
            | "11"
            | "12"
            | "15"
            | "thread"
            | "news_thread"
            | "public_thread"
            | "private_thread"
            | "forum_thread"
        ):
            return "thread"
        case _:
            return normalized


def _normalize_channel_kind_number(value: int) -> str:
    match value:
        case 0:
            return "text"
        case 2:
            return "voice"
        case 4:
            return "category"
        case 10 | 11 | 12 | 15:
            return "thread"
        case _:
            return str(value)


def _role_id(event: PlannedGraphEvent) -> str | None:
    if event.node.node_type == "role":
        return event.event.subject.id
    value = event.event.payload.get("role_id")
    if isinstance(value, str) and value != "":
        return value
    return None


def _member_user_id(event: PlannedGraphEvent) -> str | None:
    match event.event.event_type:
        case ClientEventType.MEMBER_UPDATED:
            value = event.event.payload.get("user_id", event.event.subject.id)
            if isinstance(value, str) and value != "":
                return value
        case (
            ClientEventType.MEMBER_ROLE_ASSIGNED
            | ClientEventType.MEMBER_ROLE_UNASSIGNED
        ):
            value = event.event.payload.get(
                "user_id", event.event.payload.get("member_id")
            )
            if isinstance(value, str) and value != "":
                return value
            if event.event.subject.parent_id is not None:
                return event.event.subject.parent_id
        case _:
            return None
    return None


def _member_role_ids(event: PlannedGraphEvent) -> list[str]:
    match event.event.event_type:
        case ClientEventType.MEMBER_ROLE_ASSIGNED:
            role_id = _role_id(event)
            if role_id is not None:
                return [role_id]
            return []
        case ClientEventType.MEMBER_ROLE_UNASSIGNED:
            return []
        case _:
            return _string_list_payload(event.event.payload, "roles")


def _has_member_role_snapshot(event: PlannedGraphEvent) -> bool:
    return (
        event.event.event_type == ClientEventType.MEMBER_UPDATED
        and isinstance(event.event.payload.get("roles"), list)
    )


def _user_id(event: PlannedGraphEvent) -> str:
    if event.event.event_type == ClientEventType.USER_DISCOVERED:
        value = event.event.payload.get("user_id")
        if isinstance(value, str) and value != "":
            return value
        return event.event.subject.id
    return event.event.actor.id


def _display_name(event: PlannedGraphEvent) -> str | None:
    value = event.event.payload.get("display_name")
    if isinstance(value, str) and value != "":
        return value
    return event.event.actor.display_name


def _is_bot_user(event: PlannedGraphEvent) -> bool:
    if event.event.actor.is_bot:
        return True
    is_bot = event.event.payload.get("is_bot", False)
    if is_bot is True:
        return True
    user_type = event.event.payload.get("user_type", event.event.subject.type)
    return user_type == "bot"


def _string_payload(payload: dict[str, JsonValue], key: str) -> str:
    value = payload.get(key, "")
    if isinstance(value, str):
        return value
    return ""


def _string_list_payload(payload: dict[str, JsonValue], key: str) -> list[str]:
    value = payload.get(key, [])
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item != ""]
    return []
