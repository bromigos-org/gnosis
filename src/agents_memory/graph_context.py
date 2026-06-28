from collections.abc import Sequence

from agents_memory.graph_types import CypherParameters
from agents_memory.models import GraphContextRequest, JsonValue


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
