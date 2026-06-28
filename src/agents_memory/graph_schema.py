from typing import Final

GRAPH_SCHEMA_CYPHER: Final[tuple[str, ...]] = (
    """
    CREATE CONSTRAINT event_idempotency IF NOT EXISTS
    FOR (e:Event) REQUIRE (e.tenant_id, e.idempotency_key) IS UNIQUE
    """,
    """
    CREATE CONSTRAINT graph_node_id IF NOT EXISTS
    FOR (n:GraphNode) REQUIRE n.id IS UNIQUE
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
