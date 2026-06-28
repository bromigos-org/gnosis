import json
from collections.abc import Sequence

from agents_memory.graph_events import PlannedGraphEvent
from agents_memory.models import GraphContextRequest, JsonValue

type CypherParameters = dict[str, JsonValue]


def upsert_parameters(event: PlannedGraphEvent) -> CypherParameters:
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
  e.created_at = datetime()
WITH e, e.event_id <> $event_id AS duplicate
FOREACH (_ IN CASE WHEN duplicate THEN [] ELSE [1] END |
  MERGE (n:GraphNode {id: $node_id})
  SET n.tenant_id = $tenant_id, n.agent_id = $agent_id,
    n.user_id = $user_id, n.guild_id = $guild_id, n.channel_id = $channel_id,
    n.visibility = $visibility, n.type = $node_type, n.summary = $summary,
    n.deleted = $deleted, n.payload = $payload, n.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(n)
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
