from typing import Final

GRAPH_SCHEMA_CYPHER: Final[tuple[str, ...]] = (
    """
    MATCH (e:Event)
    WHERE e.tenant_id IS NOT NULL AND e.idempotency_key IS NOT NULL
    WITH e.tenant_id AS tenant_id, e.idempotency_key AS idempotency_key,
      collect(e) AS events
    WHERE size(events) > 1
    WITH tenant_id, idempotency_key, head(events) AS keep, tail(events) AS duplicates
    UNWIND duplicates AS duplicate
    OPTIONAL MATCH (duplicate)-[:AFFECTS]->(target)
    FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END |
      MERGE (keep)-[:AFFECTS]->(target)
    )
    DETACH DELETE duplicate
    """,
    """
    CREATE CONSTRAINT event_idempotency IF NOT EXISTS
    FOR (e:Event) REQUIRE (e.tenant_id, e.idempotency_key) IS UNIQUE
    """,
    """
    MATCH (n:GraphNode)
    WHERE n.id IS NOT NULL
    WITH n.id AS graph_node_id, collect(n) AS nodes
    WHERE size(nodes) > 1
    WITH graph_node_id, head(nodes) AS keep, tail(nodes) AS duplicates
    UNWIND duplicates AS duplicate
    OPTIONAL MATCH (source)-[:AFFECTS]->(duplicate)
    FOREACH (_ IN CASE WHEN source IS NULL THEN [] ELSE [1] END |
      MERGE (source)-[:AFFECTS]->(keep)
    )
    DETACH DELETE duplicate
    """,
    """
    CREATE CONSTRAINT graph_node_id IF NOT EXISTS
    FOR (n:GraphNode) REQUIRE n.id IS UNIQUE
    """,
    """
    MATCH (n:GraphNode {type: 'message'})
    WHERE n.tenant_id IS NOT NULL
      AND n.id IS NOT NULL
      AND n.user_id IS NOT NULL
      AND n.channel_id IS NOT NULL
      AND NOT n:Message
    MATCH (m:Message {id: n.id})
    WHERE elementId(m) <> elementId(n)
    SET m.tenant_id = coalesce(m.tenant_id, n.tenant_id),
      m.message_id = coalesce(m.message_id, last(split(n.id, ':'))),
      m.updated_at = datetime()
    WITH n, m
    OPTIONAL MATCH (e:Event)-[:AFFECTS]->(n)
    WITH n, m, collect(e) AS events
    FOREACH (event IN events | MERGE (event)-[:AFFECTS]->(m))
    MERGE (u:User {id: 'tenant:' + n.tenant_id + ':user:' + n.user_id})
    SET u.tenant_id = n.tenant_id,
      u.user_id = coalesce(u.user_id, n.user_id),
      u.updated_at = datetime()
    MERGE (ch:Channel {id: 'tenant:' + n.tenant_id + ':channel:' + n.channel_id})
    SET ch.tenant_id = n.tenant_id,
      ch.guild_id = coalesce(ch.guild_id, n.guild_id),
      ch.channel_id = coalesce(ch.channel_id, n.channel_id),
      ch.updated_at = datetime()
    MERGE (u)-[:AUTHORED]->(m)
    MERGE (m)-[:IN_CHANNEL]->(ch)
    DETACH DELETE n
    """,
    """
    MATCH (n:GraphNode {type: 'message'})
    WHERE n.tenant_id IS NOT NULL
      AND n.id IS NOT NULL
      AND n.user_id IS NOT NULL
      AND n.channel_id IS NOT NULL
      AND NOT n:Message
    OPTIONAL MATCH (existing:Message {id: n.id})
    WITH n, existing
    WHERE existing IS NULL
    SET n:Message,
      n.message_id = coalesce(n.message_id, last(split(n.id, ':'))),
      n.updated_at = datetime()
    WITH n AS m
    MERGE (u:User {id: 'tenant:' + m.tenant_id + ':user:' + m.user_id})
    SET u.tenant_id = m.tenant_id,
      u.user_id = coalesce(u.user_id, m.user_id),
      u.updated_at = datetime()
    MERGE (ch:Channel {id: 'tenant:' + m.tenant_id + ':channel:' + m.channel_id})
    SET ch.tenant_id = m.tenant_id,
      ch.guild_id = coalesce(ch.guild_id, m.guild_id),
      ch.channel_id = coalesce(ch.channel_id, m.channel_id),
      ch.updated_at = datetime()
    MERGE (u)-[:AUTHORED]->(m)
    MERGE (m)-[:IN_CHANNEL]->(ch)
    """,
    """
    CREATE CONSTRAINT tenant_id IF NOT EXISTS
    FOR (t:Tenant) REQUIRE t.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT agent_id IF NOT EXISTS
    FOR (a:Agent) REQUIRE a.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT client_id IF NOT EXISTS
    FOR (c:Client) REQUIRE c.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT guild_id IF NOT EXISTS
    FOR (g:Guild) REQUIRE g.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT channel_id IF NOT EXISTS
    FOR (ch:Channel) REQUIRE ch.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT user_id IF NOT EXISTS
    FOR (u:User) REQUIRE u.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT message_id IF NOT EXISTS
    FOR (m:Message) REQUIRE m.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT role_id IF NOT EXISTS
    FOR (r:Role) REQUIRE r.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT category_id IF NOT EXISTS
    FOR (cat:Category) REQUIRE cat.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT link_id IF NOT EXISTS
    FOR (l:Link) REQUIRE l.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT attachment_id IF NOT EXISTS
    FOR (att:Attachment) REQUIRE att.id IS UNIQUE
    """,
    """
    CREATE INDEX graph_node_scope IF NOT EXISTS
    FOR (n:GraphNode) ON (n.tenant_id, n.agent_id, n.visibility)
    """,
    """
    CREATE INDEX graph_node_updated_at IF NOT EXISTS
    FOR (n:GraphNode) ON (n.updated_at)
    """,
)


def graph_vector_schema_cypher(dimensions: int) -> str:
    return f"""
    CREATE VECTOR INDEX graph_node_embedding IF NOT EXISTS
    FOR (n:GraphNode) ON (n.embedding)
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {dimensions},
      `vector.similarity_function`: 'cosine'
    }}}}
    """
