import pytest

from gnosis.graph_query_qa import GraphQueryPlan
from gnosis.graph_query_validation import (
    GraphQueryValidationError,
    SafeGraphQueryValidator,
)
from gnosis.models import GraphContextRequest, MemoryScope, MemoryVisibility


def test_validator_accepts_scoped_read_only_query() -> None:
    # Given: a generated query uses only approved labels and scope parameters.
    request = GraphContextRequest(scope=_scope(), query="Which roles exist?", limit=5)
    plan = GraphQueryPlan(
        cypher="""
        MATCH (r:Role {tenant_id: $tenant_id})
        WHERE r.guild_id = $guild_id
        RETURN r.id AS id, 'graph_query' AS type,
          coalesce(r.name, r.role_id) AS summary, false AS deleted
        ORDER BY summary ASC
        LIMIT $limit
        """,
        parameters={},
        answer_kind="roles_by_guild",
    )

    # When: gnosis validates the query before execution.
    validated = SafeGraphQueryValidator().validate(plan, request)

    # Then: runtime scope parameters are injected and the generated query is kept.
    assert validated.cypher == plan.cypher
    assert validated.parameters["tenant_id"] == "bromigos"
    assert validated.parameters["guild_id"] == "guild-123"
    assert validated.parameters["limit"] == 5


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) DETACH DELETE n RETURN n LIMIT $limit",
        "MERGE (n:Message {tenant_id: $tenant_id}) RETURN n LIMIT $limit",
        "CALL db.labels() YIELD label RETURN label LIMIT $limit",
        "MATCH (n:Message {tenant_id: 'bromigos'}) RETURN n LIMIT $limit",
        "MATCH (n:Message {tenant_id: $tenant_id}) RETURN n",
        "MATCH (n:Secret {tenant_id: $tenant_id}) RETURN n LIMIT $limit",
    ],
)
def test_validator_rejects_unsafe_generated_cypher(cypher: str) -> None:
    # Given: a generated query violates one graph QA safety rule.
    request = GraphContextRequest(scope=_scope(), query="anything", limit=5)
    plan = GraphQueryPlan(
        cypher=cypher,
        parameters={},
        answer_kind="unsafe",
    )

    # When / Then: validation blocks the query before Neo4j can execute it.
    with pytest.raises(GraphQueryValidationError):
        _ = SafeGraphQueryValidator().validate(plan, request)


@pytest.mark.parametrize(
    "cypher",
    [
        """
        MATCH (t:Tenant {tenant_id: $tenant_id})
        WITH t
        MATCH (m:Message)
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        WITH $tenant_id AS tenant_id
        MATCH (m:Message)
        WHERE tenant_id = $tenant_id
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        MATCH (m:Message {tenant_id: $tenant_id})
        WHERE m.guild_id = $guild_id
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
    ],
)
def test_validator_rejects_scope_bypass_queries(cypher: str) -> None:
    # Given: generated Cypher mentions scope without applying full runtime scope.
    request = GraphContextRequest(scope=_scope(), query="messages", limit=5)
    plan = GraphQueryPlan(cypher=cypher, parameters={}, answer_kind="messages")

    # When / Then: validation blocks the query before Neo4j can execute it.
    with pytest.raises(GraphQueryValidationError):
        _ = SafeGraphQueryValidator().validate(plan, request)


@pytest.mark.parametrize(
    "cypher",
    [
        """
        MATCH (m:Message {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        MATCH (m)-[r:SECRET_REL]->(u:User)
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        MATCH (m:`Secret` {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        MATCH (m:Message {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        RETURN m.id AS id, 'graph_query' AS type,
          m.`secret` AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        MATCH (m:Message {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        RETURN m.id AS id, 'graph_query' AS type,
          properties(m) AS summary, false AS deleted
        LIMIT $limit
        """,
    ],
)
def test_validator_rejects_unsupported_schema_syntax(cypher: str) -> None:
    # Given: generated Cypher uses syntax outside the regex allowlist contract.
    request = GraphContextRequest(scope=_scope(), query="messages", limit=5)
    plan = GraphQueryPlan(cypher=cypher, parameters={}, answer_kind="messages")

    # When / Then: validation rejects the unsupported schema access.
    with pytest.raises(GraphQueryValidationError):
        _ = SafeGraphQueryValidator().validate(plan, request)


def test_validator_accepts_fully_scoped_message_query() -> None:
    # Given: a message query applies every runtime scope field to the read alias.
    request = GraphContextRequest(scope=_scope(), query="messages", limit=5)
    plan = GraphQueryPlan(
        cypher="""
        MATCH (m:Message {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        ORDER BY m.updated_at DESC
        LIMIT $limit
        """,
        parameters={},
        answer_kind="messages_by_channel",
    )

    # When: gnosis validates the generated query.
    validated = SafeGraphQueryValidator().validate(plan, request)

    # Then: the query survives with trusted runtime scope parameters.
    assert validated.parameters["channel_id"] == "channel-456"


def _scope() -> MemoryScope:
    return MemoryScope(
        tenant_id="bromigos",
        space_id="discord",
        agent_id="pc-principal",
        session_id="guild:guild-123:channel:channel-456",
        user_id="user-789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="guild-123",
        channel_id="channel-456",
    )
