from gnosis.graph_events import PlannedGraphEvent
from gnosis.graph_upsert_values import string_list_payload
from gnosis.models import ClientEventType


def node_id(tenant_id: str, node_type: str, raw_id: str) -> str:
    return f"tenant:{tenant_id}:{node_type}:{raw_id}"


def optional_node_id(tenant_id: str, node_type: str, raw_id: str | None) -> str | None:
    if raw_id is None:
        return None
    return node_id(tenant_id, node_type, raw_id)


def message_id(event: PlannedGraphEvent) -> str | None:
    if event.event.subject.type == "message":
        return event.event.subject.id
    if event.event.discord is not None and event.event.discord.message_id is not None:
        return event.event.discord.message_id
    value = event.event.payload.get("message_id")
    if isinstance(value, str):
        return value
    return None


def category_id(event: PlannedGraphEvent) -> str | None:
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


def role_id(event: PlannedGraphEvent) -> str | None:
    if event.node.node_type == "role":
        return event.event.subject.id
    value = event.event.payload.get("role_id")
    if isinstance(value, str) and value != "":
        return value
    return None


def member_user_id(event: PlannedGraphEvent) -> str | None:
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


def member_role_ids(event: PlannedGraphEvent) -> list[str]:
    match event.event.event_type:
        case ClientEventType.MEMBER_ROLE_ASSIGNED:
            role_id_value = role_id(event)
            if role_id_value is not None:
                return [role_id_value]
            return []
        case ClientEventType.MEMBER_ROLE_UNASSIGNED:
            return []
        case _:
            return string_list_payload(event.event.payload, "roles")


def user_id(event: PlannedGraphEvent) -> str:
    if event.event.event_type == ClientEventType.USER_DISCOVERED:
        value = event.event.payload.get("user_id")
        if isinstance(value, str) and value != "":
            return value
        return event.event.subject.id
    return event.event.actor.id
