import httpx
import pytest
from openai import APIConnectionError

from gnosis.graph_query_execution import plan_graph_query
from gnosis.graph_query_qa import GraphQueryPlan, proxy_model_name
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


def test_proxy_model_name_strips_litellm_openai_prefix() -> None:
    # Given: the SDK-facing setting uses litellm provider-prefixed names.
    # Then: the raw OpenAI proxy client receives the bare model id.
    assert proxy_model_name("openai/gpt-5.5") == "gpt-5.5"
    assert proxy_model_name("openai/gemma4") == "gemma4"
    assert proxy_model_name("gpt-5.5") == "gpt-5.5"


class _ExplodingPlanner:
    async def plan_query(self, request: GraphContextRequest) -> GraphQueryPlan | None:
        del request
        raise APIConnectionError(request=httpx.Request("POST", "http://litellm.test"))


@pytest.mark.anyio
async def test_plan_graph_query_swallows_openai_errors() -> None:
    # Given: the upstream proxy rejects or drops the planner call.
    request = GraphContextRequest(scope=_scope(), query="Which roles exist?", limit=5)

    # When: gnosis plans a graph query.
    plan = await plan_graph_query(_ExplodingPlanner(), request)

    # Then: planner failures degrade to no graph context, never an error.
    assert plan is None


class _StructuredOutputViolatingPlanner:
    async def plan_query(self, request: GraphContextRequest) -> GraphQueryPlan | None:
        del request
        # The planner LLM returns prose + fenced JSON instead of the structured
        # contract; `.parse()` raises pydantic ValidationError.
        _ = GraphQueryPlan.model_validate_json("I don't have direct Neo4j access")
        return None


@pytest.mark.anyio
async def test_plan_graph_query_swallows_structured_output_violations() -> None:
    # Given: the planner LLM ignores the structured-output contract.
    request = GraphContextRequest(scope=_scope(), query="Which roles exist?", limit=5)

    # When: gnosis plans a graph query.
    plan = await plan_graph_query(_StructuredOutputViolatingPlanner(), request)

    # Then: it degrades to no graph context rather than 500-ing the caller.
    assert plan is None
