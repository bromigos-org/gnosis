from dataclasses import dataclass

from gnosis.models import (
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
    semantic_node_ids: tuple[str, ...]


def plan_event(event: ClientEvent) -> PlannedGraphEvent:
    node_type = node_type_for(event)
    node_raw_id = _node_raw_id(event)
    node = GraphNode(
        id=_node_id(event.tenant_id, node_type, node_raw_id),
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
        semantic_node_ids=_semantic_node_ids(event),
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
            if event.subject.type == "category":
                return "category"
            if _string_payload(event.payload, "channel_type") == "category":
                return "category"
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
        case (
            ClientEventType.MEMBER_UPDATED
            | ClientEventType.USER_DISCOVERED
            | ClientEventType.MEMBER_ROLE_ASSIGNED
            | ClientEventType.MEMBER_ROLE_UNASSIGNED
        ):
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


def _semantic_node_ids(event: ClientEvent) -> tuple[str, ...]:
    ids = [
        _node_id(event.tenant_id, "tenant", event.tenant_id),
        _node_id(event.tenant_id, "agent", event.agent_id),
        _node_id(event.tenant_id, "client", event.source_client.value),
    ]
    _append_topology_ids(ids, event)
    _append_actor_ids(ids, event)
    _append_role_ids(ids, event)
    message_id = _message_id(event)
    if message_id != "":
        ids.append(_node_id(event.tenant_id, "message", message_id))
    if node_type_for(event) == "link":
        ids.append(_node_id(event.tenant_id, "link", event.subject.id))
    if node_type_for(event) == "attachment":
        ids.append(_node_id(event.tenant_id, "attachment", event.subject.id))
    return tuple(ids)


def _append_scoped_ids(ids: list[str], event: ClientEvent) -> None:
    _append_topology_ids(ids, event)
    _append_actor_ids(ids, event)
    _append_role_ids(ids, event)
    match event.source_client:
        case SourceClient.DISCORD:
            pass


def _append_topology_ids(ids: list[str], event: ClientEvent) -> None:
    scope = event.scope
    if scope.guild_id is not None:
        ids.append(_node_id(event.tenant_id, "guild", scope.guild_id))
    if scope.channel_id is not None:
        ids.append(_node_id(event.tenant_id, "channel", scope.channel_id))
    if event.discord is not None and event.discord.thread_id is not None:
        ids.append(_node_id(event.tenant_id, "thread", event.discord.thread_id))
    category_id = _category_id(event)
    if category_id != "":
        ids.append(_node_id(event.tenant_id, "category", category_id))


def _append_actor_ids(ids: list[str], event: ClientEvent) -> None:
    user_id = _user_id(event)
    if user_id != "":
        ids.append(_node_id(event.tenant_id, "user", user_id))
        if _is_bot_user(event):
            ids.append(_node_id(event.tenant_id, "bot", user_id))


def _append_role_ids(ids: list[str], event: ClientEvent) -> None:
    role_id = _role_id(event)
    if role_id != "":
        ids.append(_node_id(event.tenant_id, "role", role_id))
    member_user_id = _member_user_id(event)
    if member_user_id != "":
        ids.append(_node_id(event.tenant_id, "user", member_user_id))
    ids.extend(
        _node_id(event.tenant_id, "role", role_id) for role_id in _role_ids(event)
    )


def _node_id(tenant_id: str, node_type: str, raw_id: str) -> str:
    return f"tenant:{tenant_id}:{node_type}:{raw_id}"


def _node_raw_id(event: ClientEvent) -> str:
    match event.event_type:
        case (
            ClientEventType.MEMBER_ROLE_ASSIGNED
            | ClientEventType.MEMBER_ROLE_UNASSIGNED
        ):
            member_user_id = _member_user_id(event)
            return member_user_id or event.subject.id
        case _:
            return event.subject.id


def _message_id(event: ClientEvent) -> str:
    if event.subject.type == "message":
        return event.subject.id
    if event.discord is not None and event.discord.message_id is not None:
        return event.discord.message_id
    return _string_payload(event.payload, "message_id")


def _category_id(event: ClientEvent) -> str:
    if node_type_for(event) == "category":
        return event.subject.id
    category_id = _string_payload(event.payload, "category_id")
    if category_id != "":
        return category_id
    if event.event_type in {
        ClientEventType.CHANNEL_CREATED,
        ClientEventType.CHANNEL_UPDATED,
        ClientEventType.CHANNEL_DELETED,
    }:
        return event.subject.parent_id or ""
    return ""


def _role_id(event: ClientEvent) -> str:
    if node_type_for(event) == "role":
        return event.subject.id
    return _string_payload(event.payload, "role_id")


def _member_user_id(event: ClientEvent) -> str:
    match event.event_type:
        case ClientEventType.MEMBER_UPDATED:
            user_id = _string_payload(event.payload, "user_id")
            return user_id or event.subject.id
        case (
            ClientEventType.MEMBER_ROLE_ASSIGNED
            | ClientEventType.MEMBER_ROLE_UNASSIGNED
        ):
            user_id = _string_payload(event.payload, "user_id")
            member_id = _string_payload(event.payload, "member_id")
            return user_id or member_id or event.subject.parent_id or ""
        case _:
            return ""


def _role_ids(event: ClientEvent) -> tuple[str, ...]:
    match event.event_type:
        case ClientEventType.MEMBER_ROLE_ASSIGNED:
            role_id = _role_id(event)
            if role_id != "":
                return (role_id,)
            return ()
        case _:
            value = event.payload.get("roles", [])
            if isinstance(value, list):
                return tuple(
                    item for item in value if isinstance(item, str) and item != ""
                )
            return ()


def _user_id(event: ClientEvent) -> str:
    if event.event_type == ClientEventType.USER_DISCOVERED:
        user_id = _string_payload(event.payload, "user_id")
        return user_id or event.subject.id
    return event.actor.id


def _is_bot_user(event: ClientEvent) -> bool:
    if event.actor.is_bot:
        return True
    is_bot = event.payload.get("is_bot", False)
    if is_bot is True:
        return True
    return (
        event.subject.type == "bot"
        or _string_payload(event.payload, "user_type") == "bot"
    )


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
            | ClientEventType.USER_DISCOVERED
            | ClientEventType.MEMBER_ROLE_ASSIGNED
            | ClientEventType.MEMBER_ROLE_UNASSIGNED
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
