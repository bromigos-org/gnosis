from dataclasses import dataclass

from agents_memory.models import (
    ClientEvent,
    ClientEventType,
    GraphContextRequest,
    JsonObject,
    JsonValue,
    MemoryScope,
    MemoryVisibility,
    SourceClient,
)


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    node_type: str
    scope: MemoryScope
    summary: str
    deleted: bool
    payload: JsonObject


@dataclass(frozen=True, slots=True)
class PlannedGraphEvent:
    event: ClientEvent
    node: GraphNode
    supporting_node_ids: tuple[str, ...]


def plan_event(event: ClientEvent) -> PlannedGraphEvent:
    node_type = node_type_for(event)
    node = GraphNode(
        id=_node_id(event.tenant_id, node_type, event.subject.id),
        node_type=node_type,
        scope=event.scope,
        summary=_summary(event, node_type),
        deleted=is_delete_event(event.event_type),
        payload=event.payload,
    )
    return PlannedGraphEvent(
        event=event,
        node=node,
        supporting_node_ids=_supporting_node_ids(event),
    )


def node_type_for(event: ClientEvent) -> str:  # noqa: C901, PLR0911
    match event.event_type:
        case (
            ClientEventType.MESSAGE_CREATED
            | ClientEventType.MESSAGE_UPDATED
            | ClientEventType.MESSAGE_DELETED
        ):
            return "message"
        case ClientEventType.REACTION_ADDED | ClientEventType.REACTION_REMOVED:
            return "reaction"
        case (
            ClientEventType.CHANNEL_CREATED
            | ClientEventType.CHANNEL_UPDATED
            | ClientEventType.CHANNEL_DELETED
        ):
            return event.subject.type
        case (
            ClientEventType.THREAD_CREATED
            | ClientEventType.THREAD_UPDATED
            | ClientEventType.THREAD_DELETED
        ):
            return "thread"
        case (
            ClientEventType.ROLE_CREATED
            | ClientEventType.ROLE_UPDATED
            | ClientEventType.ROLE_DELETED
        ):
            return "role"
        case ClientEventType.MEMBER_UPDATED:
            return "user"
        case ClientEventType.ATTACHMENT_DISCOVERED:
            return "attachment"
        case ClientEventType.LINK_DISCOVERED:
            return "link"
        case ClientEventType.TOPIC_UPDATED:
            return "topic"
        case (
            ClientEventType.SKILL_PROPOSED
            | ClientEventType.SKILL_APPROVED
            | ClientEventType.SKILL_USED
        ):
            return "skill"


def context_allows_node(request: GraphContextRequest, node: GraphNode) -> bool:
    return _scope_allows(request.scope, node.scope)


def fact_from_node(node: GraphNode) -> JsonObject:
    return {
        "id": node.id,
        "type": node.node_type,
        "scope": node.scope.visibility.value,
        "summary": node.summary,
        "deleted": node.deleted,
    }


def node_from_row(row: dict[str, JsonValue], scope: MemoryScope) -> GraphNode:
    return GraphNode(
        id=_row_string(row, "id"),
        node_type=_row_string(row, "type"),
        scope=scope,
        summary=_row_string(row, "summary"),
        deleted=row.get("deleted", False) is True,
        payload={},
    )


def _summary(event: ClientEvent, node_type: str) -> str:
    if is_delete_event(event.event_type):
        return f"{node_type} {event.subject.id}: deleted"
    name = _string_payload(event.payload, "name")
    content = _string_payload(event.payload, "content")
    topic = _string_payload(event.payload, "topic")
    url = _string_payload(event.payload, "url")
    label = name or content or topic or url or event.subject.id
    return f"{node_type} {event.subject.id}: {label}"


def _supporting_node_ids(event: ClientEvent) -> tuple[str, ...]:
    ids = [
        _node_id(event.tenant_id, "tenant", event.tenant_id),
        _node_id(event.tenant_id, "agent", event.agent_id),
        _node_id(event.tenant_id, "client", event.source_client.value),
        _node_id(event.tenant_id, "event", event.event_id),
    ]
    _append_scoped_ids(ids, event)
    return tuple(ids)


def _append_scoped_ids(ids: list[str], event: ClientEvent) -> None:
    scope = event.scope
    if scope.guild_id is not None:
        ids.append(_node_id(event.tenant_id, "guild", scope.guild_id))
    if scope.channel_id is not None:
        ids.append(_node_id(event.tenant_id, "channel", scope.channel_id))
    if event.discord is not None and event.discord.thread_id is not None:
        ids.append(_node_id(event.tenant_id, "thread", event.discord.thread_id))
    category_id = _string_payload(event.payload, "category_id")
    if category_id != "":
        ids.append(_node_id(event.tenant_id, "category", category_id))
    if event.actor.id != "":
        ids.append(_node_id(event.tenant_id, "user", event.actor.id))
    match event.source_client:
        case SourceClient.DISCORD:
            pass


def _node_id(tenant_id: str, node_type: str, raw_id: str) -> str:
    return f"tenant:{tenant_id}:{node_type}:{raw_id}"


def _scope_allows(request: MemoryScope, candidate: MemoryScope) -> bool:
    return (
        request.tenant_id == candidate.tenant_id
        and request.agent_id == candidate.agent_id
        and _visibility_allows(request, candidate)
    )


def _visibility_allows(request: MemoryScope, candidate: MemoryScope) -> bool:
    match candidate.visibility:
        case MemoryVisibility.PRIVATE_USER:
            return request.user_id == candidate.user_id
        case MemoryVisibility.AGENT_PRIVATE | MemoryVisibility.AGENT_SHARED:
            return request.agent_id == candidate.agent_id
        case MemoryVisibility.CHANNEL:
            return (
                request.guild_id == candidate.guild_id
                and request.channel_id == candidate.channel_id
            )
        case MemoryVisibility.GUILD:
            return request.guild_id == candidate.guild_id
        case MemoryVisibility.TENANT:
            return request.tenant_id == candidate.tenant_id
        case MemoryVisibility.GLOBAL:
            return True


def is_delete_event(event_type: ClientEventType) -> bool:
    match event_type:
        case (
            ClientEventType.MESSAGE_DELETED
            | ClientEventType.CHANNEL_DELETED
            | ClientEventType.THREAD_DELETED
            | ClientEventType.ROLE_DELETED
            | ClientEventType.REACTION_REMOVED
        ):
            return True
        case (
            ClientEventType.MESSAGE_CREATED
            | ClientEventType.MESSAGE_UPDATED
            | ClientEventType.REACTION_ADDED
            | ClientEventType.CHANNEL_CREATED
            | ClientEventType.CHANNEL_UPDATED
            | ClientEventType.THREAD_CREATED
            | ClientEventType.THREAD_UPDATED
            | ClientEventType.ROLE_CREATED
            | ClientEventType.ROLE_UPDATED
            | ClientEventType.MEMBER_UPDATED
            | ClientEventType.ATTACHMENT_DISCOVERED
            | ClientEventType.LINK_DISCOVERED
            | ClientEventType.TOPIC_UPDATED
            | ClientEventType.SKILL_PROPOSED
            | ClientEventType.SKILL_APPROVED
            | ClientEventType.SKILL_USED
        ):
            return False


def _string_payload(payload: JsonObject, key: str) -> str:
    value: JsonValue = payload.get(key, "")
    if isinstance(value, str):
        return value
    return ""


def _row_string(row: dict[str, JsonValue], key: str) -> str:
    value = row.get(key, "")
    if isinstance(value, str):
        return value
    return ""
