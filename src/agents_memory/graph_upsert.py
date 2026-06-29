import json

from agents_memory.graph_events import PlannedGraphEvent
from agents_memory.graph_types import CypherParameters
from agents_memory.graph_upsert_ids import (
    category_id,
    member_role_ids,
    member_user_id,
    message_id,
    node_id,
    optional_node_id,
    role_id,
    user_id,
)
from agents_memory.graph_upsert_values import (
    category_name,
    channel_kind,
    channel_name,
    display_name,
    has_member_identity_snapshot,
    has_member_role_snapshot,
    is_bot_user,
    member_is_bot,
    string_payload,
)
from agents_memory.models import ClientEventType


def upsert_parameters(event: PlannedGraphEvent) -> CypherParameters:
    message_id_value = message_id(event)
    link_id = event.event.subject.id if event.node.node_type == "link" else None
    attachment_id = (
        event.event.subject.id if event.node.node_type == "attachment" else None
    )
    category_id_value = category_id(event)
    channel_name_value = channel_name(event)
    channel_kind_value = channel_kind(event)
    role_id_value = role_id(event)
    member_user_id_value = member_user_id(event)
    member_is_bot_value = member_is_bot(event)
    member_role_id_values = member_role_ids(event)
    user_id_value = user_id(event)
    is_bot_user_value = is_bot_user(event)
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
        "node_embedding": None,
        "deleted": event.node.deleted,
        "payload": json.dumps(
            event.node.payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "supporting_node_ids": list(event.supporting_node_ids),
        "semantic_node_ids": list(event.semantic_node_ids),
        "tenant_node_id": node_id(
            event.event.tenant_id,
            "tenant",
            event.event.tenant_id,
        ),
        "agent_node_id": node_id(event.event.tenant_id, "agent", event.event.agent_id),
        "client_node_id": node_id(
            event.event.tenant_id,
            "client",
            event.event.source_client.value,
        ),
        "guild_node_id": optional_node_id(
            event.event.tenant_id,
            "guild",
            event.event.scope.guild_id,
        ),
        "channel_node_id": optional_node_id(
            event.event.tenant_id,
            "channel",
            event.event.scope.channel_id,
        ),
        "thread_node_id": optional_node_id(
            event.event.tenant_id,
            "thread",
            event.event.discord.thread_id if event.event.discord is not None else None,
        ),
        "user_node_id": node_id(event.event.tenant_id, "user", user_id_value),
        "message_node_id": optional_node_id(
            event.event.tenant_id,
            "message",
            message_id_value,
        ),
        "link_node_id": optional_node_id(event.event.tenant_id, "link", link_id),
        "attachment_node_id": optional_node_id(
            event.event.tenant_id,
            "attachment",
            attachment_id,
        ),
        "category_node_id": optional_node_id(
            event.event.tenant_id,
            "category",
            category_id_value,
        ),
        "role_node_id": optional_node_id(event.event.tenant_id, "role", role_id_value),
        "member_user_node_id": optional_node_id(
            event.event.tenant_id,
            "user",
            member_user_id_value,
        ),
        "member_role_node_ids": [
            node_id(event.event.tenant_id, "role", member_role_id)
            for member_role_id in member_role_id_values
        ],
        "subject_node_id": event.node.id,
        "subject_node_type": event.node.node_type,
        "actor_id": event.event.actor.id,
        "actor_display_name": display_name(event),
        "actor_is_bot": is_bot_user_value,
        "user_identity_id": user_id_value,
        "category_id": category_id_value,
        "category_name": category_name(event),
        "channel_name": channel_name_value,
        "channel_kind": channel_kind_value,
        "role_id": role_id_value,
        "role_name": string_payload(event.event.payload, "name"),
        "member_user_id": member_user_id_value,
        "member_is_bot": member_is_bot_value,
        "member_user_type": "bot" if member_is_bot_value else "user",
        "source_client": event.event.source_client.value,
        "message_id": message_id_value,
        "message_content": string_payload(event.event.payload, "content"),
        "link_url": string_payload(event.event.payload, "url"),
        "attachment_filename": string_payload(event.event.payload, "filename"),
        "has_guild": event.event.scope.guild_id is not None,
        "has_channel": event.event.scope.channel_id is not None,
        "has_thread": event.event.discord is not None
        and event.event.discord.thread_id is not None,
        "has_message": message_id_value is not None,
        "has_message_subject": event.node.node_type == "message"
        and message_id_value is not None,
        "has_link": link_id is not None,
        "has_attachment": attachment_id is not None,
        "has_category": category_id_value is not None,
        "has_category_subject": event.node.node_type == "category"
        and category_id_value is not None,
        "has_channel_category": event.node.node_type == "channel"
        and category_id_value is not None,
        "has_role": role_id_value is not None,
        "has_member_user": member_user_id_value is not None,
        "has_member_identity_snapshot": has_member_identity_snapshot(event),
        "has_member_roles": bool(member_role_id_values),
        "has_member_role_snapshot": has_member_role_snapshot(event),
        "has_user_identity": user_id_value != "",
        "is_member_role_assignment": event.event.event_type
        == ClientEventType.MEMBER_ROLE_ASSIGNED,
        "is_member_role_unassignment": event.event.event_type
        == ClientEventType.MEMBER_ROLE_UNASSIGNED,
    }
