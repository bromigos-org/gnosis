import re
from collections.abc import Sequence
from dataclasses import dataclass
from re import Pattern
from typing import Final

from gnosis.graph_events import GraphNode
from gnosis.graph_types import CypherParameters
from gnosis.models import ClientEvent, ClientEventType, GraphContextRequest, JsonValue

TOP_ACTIVE_CHANNELS_CYPHER: Final = """
MATCH (u:User {tenant_id: $tenant_id})-[:AUTHORED]->(m:Message)-[:IN_CHANNEL]->(
  ch:Channel
)
WHERE ($guild_id IS NULL OR ch.guild_id = $guild_id)
  AND (
    toLower(
      replace(replace(replace(coalesce(u.user_id, ''), ' ', ''), '-', ''), '_', '')
    )
      CONTAINS $user_token
    OR toLower(
      replace(replace(replace(coalesce(u.display_name, ''), ' ', ''), '-', ''), '_', '')
    )
      CONTAINS $user_token
  )
WITH u, ch, count(m) AS message_count
ORDER BY message_count DESC, coalesce(ch.name, ch.channel_id) ASC
WITH collect({user: u, channel: ch, message_count: message_count})[..$limit] AS rows
UNWIND range(0, size(rows) - 1) AS index
WITH rows[index] AS row, index + 1 AS rank
WITH row.user AS u, row.channel AS ch, row.message_count AS message_count, rank,
  coalesce(u.display_name, u.user_id) AS user_name,
  coalesce(ch.name, ch.channel_id) AS channel_name
RETURN 'aggregate:' + $tenant_id + ':' + coalesce($guild_id, 'global') + ':'
  + u.user_id + ':' + ch.channel_id AS id,
  'channel_activity' AS type,
  user_name + ' active channel #' + toString(rank) + ': ' + channel_name + ' ('
    + toString(message_count)
    + CASE message_count WHEN 1 THEN ' message)' ELSE ' messages)' END AS summary,
  false AS deleted,
  rank AS rank,
  u.user_id AS user_id,
  user_name AS user_display_name,
  ch.channel_id AS channel_id,
  channel_name AS channel_name,
  message_count AS message_count
"""

_ACTIVITY_WORDS: Final = frozenset(("active", "activity"))
_CHANNEL_WORDS: Final = frozenset(("channel", "channels"))
_TOP_WORDS: Final = frozenset(("top", "most"))
_MENTION_PATTERN: Final[Pattern[str]] = re.compile(r"@([A-Za-z0-9_-]+)")
_WORD_PATTERN: Final[Pattern[str]] = re.compile(r"[A-Za-z0-9_@-]+")


@dataclass(frozen=True, slots=True)
class ChannelActivity:
    user_id: str
    user_display_name: str
    channel_id: str
    channel_name: str
    message_count: int


def is_top_active_channels_request(request: GraphContextRequest) -> bool:
    words = frozenset(_WORD_PATTERN.findall(request.query.casefold()))
    return (
        bool(words & _TOP_WORDS)
        and bool(words & _ACTIVITY_WORDS)
        and bool(words & _CHANNEL_WORDS)
        and _query_user_token(request.query) != ""
    )


def top_active_channel_parameters(request: GraphContextRequest) -> CypherParameters:
    return {
        "tenant_id": request.scope.tenant_id,
        "agent_id": request.scope.agent_id,
        "guild_id": request.scope.guild_id,
        "limit": request.limit,
        "user_token": _query_user_token(request.query),
    }


def top_active_channel_nodes(
    request: GraphContextRequest,
    events: Sequence[ClientEvent],
) -> Sequence[GraphNode]:
    user_token = _query_user_token(request.query)
    if user_token == "":
        return ()
    channels = _channel_names(events)
    activities = _rank_channel_activity(request, events, channels, user_token)
    return tuple(
        _activity_node(request, activity, rank) for rank, activity in activities
    )


def _rank_channel_activity(
    request: GraphContextRequest,
    events: Sequence[ClientEvent],
    channels: dict[str, str],
    user_token: str,
) -> tuple[tuple[int, ChannelActivity], ...]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for event in events:
        if not _event_matches_activity_request(request, event, user_token):
            continue
        channel_id = event.scope.channel_id
        if channel_id is None:
            continue
        user_name = event.actor.display_name or event.actor.id
        key = (
            event.actor.id,
            user_name,
            channel_id,
            channels.get(channel_id, channel_id),
        )
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(
        (
            ChannelActivity(
                user_id=user_id,
                user_display_name=user_name,
                channel_id=channel_id,
                channel_name=channel_name,
                message_count=count,
            )
            for (user_id, user_name, channel_id, channel_name), count in counts.items()
        ),
        key=lambda activity: (-activity.message_count, activity.channel_name),
    )
    return tuple(enumerate(ranked[: request.limit], start=1))


def _event_matches_activity_request(
    request: GraphContextRequest,
    event: ClientEvent,
    user_token: str,
) -> bool:
    return (
        event.event_type == ClientEventType.MESSAGE_CREATED
        and event.tenant_id == request.scope.tenant_id
        and event.agent_id == request.scope.agent_id
        and event.scope.guild_id == request.scope.guild_id
        and _user_matches(event, user_token)
    )


def _user_matches(event: ClientEvent, user_token: str) -> bool:
    return user_token in _normalized(event.actor.id) or user_token in _normalized(
        event.actor.display_name or "",
    )


def _channel_names(events: Sequence[ClientEvent]) -> dict[str, str]:
    names: dict[str, str] = {}
    for event in events:
        if event.event_type not in {
            ClientEventType.CHANNEL_CREATED,
            ClientEventType.CHANNEL_UPDATED,
        }:
            continue
        channel_id = _string_payload(event.payload, "channel_id")
        name = _string_payload(event.payload, "name")
        if channel_id != "" and name != "":
            names[channel_id] = name
    return names


def _activity_node(
    request: GraphContextRequest,
    activity: ChannelActivity,
    rank: int,
) -> GraphNode:
    message_label = "message" if activity.message_count == 1 else "messages"
    summary = (
        f"{activity.user_display_name} active channel #{rank}: "
        f"{activity.channel_name} ({activity.message_count} {message_label})"
    )
    return GraphNode(
        id=(
            f"aggregate:{request.scope.tenant_id}:{request.scope.guild_id}:"
            f"{activity.user_id}:{activity.channel_id}"
        ),
        node_type="channel_activity",
        scope=request.scope,
        summary=summary,
        deleted=False,
        payload={
            "rank": rank,
            "user_id": activity.user_id,
            "user_display_name": activity.user_display_name,
            "channel_id": activity.channel_id,
            "channel_name": activity.channel_name,
            "message_count": activity.message_count,
        },
    )


def _query_user_token(query: str) -> str:
    mentions: list[str] = _MENTION_PATTERN.findall(query)
    if mentions:
        return _normalized(mentions[0])
    return ""


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _string_payload(payload: dict[str, JsonValue], key: str) -> str:
    value = payload.get(key, "")
    if isinstance(value, str):
        return value
    return ""
