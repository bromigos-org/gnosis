from agents_memory.graph_events import PlannedGraphEvent
from agents_memory.models import ClientEventType, JsonValue


def category_name(event: PlannedGraphEvent) -> str:
    if event.node.node_type == "category":
        return string_payload(event.event.payload, "name")
    return string_payload(event.event.payload, "category_name")


def channel_name(event: PlannedGraphEvent) -> str:
    if event.node.node_type in {"category", "channel", "thread"}:
        return string_payload(event.event.payload, "name")
    return ""


def channel_kind(event: PlannedGraphEvent) -> str:
    if event.node.node_type == "category":
        return "category"
    if event.node.node_type == "thread":
        return "thread"
    return normalize_channel_kind(event.event.payload.get("channel_type"))


def normalize_channel_kind(value: JsonValue | None) -> str:
    match value:
        case str():
            return normalize_channel_kind_text(value)
        case int():
            return normalize_channel_kind_number(value)
        case _:
            return ""


def normalize_channel_kind_text(value: str) -> str:
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


def normalize_channel_kind_number(value: int) -> str:
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


def has_member_role_snapshot(event: PlannedGraphEvent) -> bool:
    return event.event.event_type == ClientEventType.MEMBER_UPDATED and isinstance(
        event.event.payload.get("roles"), list
    )


def has_member_identity_snapshot(event: PlannedGraphEvent) -> bool:
    return event.event.event_type == ClientEventType.MEMBER_UPDATED and isinstance(
        event.event.payload.get("is_bot"), bool
    )


def display_name(event: PlannedGraphEvent) -> str | None:
    value = event.event.payload.get("display_name")
    if isinstance(value, str) and value != "":
        return value
    return event.event.actor.display_name


def is_bot_user(event: PlannedGraphEvent) -> bool:
    if event.event.event_type == ClientEventType.USER_DISCOVERED:
        is_bot = event.event.payload.get("is_bot", False)
        if isinstance(is_bot, bool):
            return is_bot
        user_type = event.event.payload.get("user_type", event.event.subject.type)
        return user_type == "bot"
    if event.event.actor.is_bot:
        return True
    is_bot = event.event.payload.get("is_bot", False)
    if is_bot is True:
        return True
    user_type = event.event.payload.get("user_type", event.event.subject.type)
    return user_type == "bot"


def member_is_bot(event: PlannedGraphEvent) -> bool:
    is_bot = event.event.payload.get("is_bot", False)
    if isinstance(is_bot, bool):
        return is_bot
    return string_payload(event.event.payload, "user_type") == "bot"


def string_payload(payload: dict[str, JsonValue], key: str) -> str:
    value = payload.get(key, "")
    if isinstance(value, str):
        return value
    return ""


def string_list_payload(payload: dict[str, JsonValue], key: str) -> list[str]:
    value = payload.get(key, [])
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item != ""]
    return []
