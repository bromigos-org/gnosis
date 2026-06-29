from collections.abc import Sequence
from typing import Final

from agents_memory.graph_types import CypherParameters, vector_parameter
from agents_memory.models import GraphContextRequest, JsonValue


def context_parameters(
    request: GraphContextRequest,
    query_embedding: Sequence[float] | None = None,
) -> CypherParameters:
    parameters: CypherParameters = {
        "tenant_id": request.scope.tenant_id,
        "agent_id": request.scope.agent_id,
        "user_id": request.scope.user_id,
        "guild_id": request.scope.guild_id,
        "channel_id": request.scope.channel_id,
        "limit": request.limit,
    }
    if query_embedding is None:
        return parameters
    parameters["query_embedding"] = vector_parameter(query_embedding)
    parameters["vector_limit"] = request.limit * 4
    return parameters


def is_duplicate_result(rows: Sequence[dict[str, JsonValue]]) -> bool:
    if not rows:
        return False
    return rows[0].get("duplicate", False) is True


CONTEXT_SCOPE_PREDICATE: Final[str] = """
n.visibility = 'global'
  OR n.visibility = 'tenant'
  OR n.visibility IN ['agent_private', 'agent_shared']
  OR (n.visibility = 'private_user' AND n.user_id = $user_id)
  OR (n.visibility = 'guild' AND n.guild_id = $guild_id)
  OR (
    n.visibility = 'channel'
    AND n.guild_id = $guild_id
    AND n.channel_id = $channel_id
  )
"""


CONTEXT_CYPHER = f"""
MATCH (n:GraphNode {{tenant_id: $tenant_id, agent_id: $agent_id}})
WHERE {CONTEXT_SCOPE_PREDICATE}
RETURN n.id AS id, n.type AS type, n.summary AS summary, n.deleted AS deleted
ORDER BY n.updated_at DESC
LIMIT $limit
"""


SEMANTIC_CONTEXT_CYPHER = f"""
CALL {{
  CALL db.index.vector.queryNodes(
    'graph_node_embedding',
    $vector_limit,
    $query_embedding
  ) YIELD node, score
  WITH node AS n, score
  WHERE n:GraphNode
    AND n.tenant_id = $tenant_id
    AND n.agent_id = $agent_id
    AND ({CONTEXT_SCOPE_PREDICATE})
  RETURN n, score, 0 AS retrieval_rank
  UNION
  MATCH (n:GraphNode {{tenant_id: $tenant_id, agent_id: $agent_id}})
  WHERE {CONTEXT_SCOPE_PREDICATE}
  RETURN n, 0.0 AS score, 1 AS retrieval_rank
}}
WITH n,
  max(score) AS score,
  min(retrieval_rank) AS retrieval_rank,
  max(n.updated_at) AS updated_at
RETURN n.id AS id, n.type AS type, n.summary AS summary, n.deleted AS deleted
ORDER BY retrieval_rank ASC, score DESC, updated_at DESC
LIMIT $limit
"""
