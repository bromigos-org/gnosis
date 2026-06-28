from typing import Final

UPSERT_EVENT_CYPHER: Final = """
MERGE (e:Event {tenant_id: $tenant_id, idempotency_key: $idempotency_key})
ON CREATE SET e.event_id = $event_id, e.event_type = $event_type,
  e.occurred_at = $occurred_at, e.observed_at = $observed_at,
  e.created_at = datetime(), e.was_created = true
ON MATCH SET e.was_created = false
WITH e, e.was_created = false AS duplicate
REMOVE e.was_created
WITH e, duplicate
MERGE (n:GraphNode {id: $node_id})
SET n.tenant_id = $tenant_id, n.agent_id = $agent_id,
  n.user_id = $user_id, n.guild_id = $guild_id, n.channel_id = $channel_id,
  n.visibility = $visibility, n.type = $node_type, n.summary = $summary,
  n.deleted = $deleted, n.payload = $payload, n.updated_at = datetime()
MERGE (e)-[:AFFECTS]->(n)
MERGE (t:Tenant {id: $tenant_node_id})
SET t.tenant_id = $tenant_id, t.updated_at = datetime()
MERGE (a:Agent {id: $agent_node_id})
SET a.tenant_id = $tenant_id, a.agent_id = $agent_id, a.updated_at = datetime()
MERGE (c:Client {id: $client_node_id})
SET c.tenant_id = $tenant_id, c.name = $source_client, c.updated_at = datetime()
MERGE (t)-[:OWNS_AGENT]->(a)
MERGE (t)-[:OWNS_CLIENT]->(c)
MERGE (a)-[:USES_CLIENT]->(c)
MERGE (e)-[:AFFECTS]->(t)
MERGE (e)-[:AFFECTS]->(a)
MERGE (e)-[:AFFECTS]->(c)
FOREACH (_ IN CASE WHEN $has_guild THEN [1] ELSE [] END |
  MERGE (g:Guild {id: $guild_node_id})
  SET g.tenant_id = $tenant_id, g.guild_id = $guild_id, g.updated_at = datetime()
  MERGE (t)-[:OWNS_GUILD]->(g)
  MERGE (e)-[:AFFECTS]->(g)
)
FOREACH (_ IN CASE WHEN $has_channel THEN [1] ELSE [] END |
  MERGE (ch:Channel {id: $channel_node_id})
  SET ch.tenant_id = $tenant_id, ch.guild_id = $guild_id,
    ch.channel_id = $channel_id,
    ch.name = coalesce(nullif($channel_name, ''), ch.name),
    ch.kind = coalesce(nullif($channel_kind, ''), ch.kind),
    ch.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(ch)
)
FOREACH (_ IN CASE WHEN $has_guild AND $has_channel THEN [1] ELSE [] END |
  MERGE (g:Guild {id: $guild_node_id})
  MERGE (ch:Channel {id: $channel_node_id})
  MERGE (ch)-[:IN_GUILD]->(g)
)
FOREACH (_ IN CASE WHEN $has_thread THEN [1] ELSE [] END |
  MERGE (th:Channel {id: $thread_node_id})
  SET th.tenant_id = $tenant_id, th.guild_id = $guild_id,
    th.channel_id = $channel_id, th.kind = 'thread', th.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(th)
)
FOREACH (_ IN CASE WHEN $has_user_identity THEN [1] ELSE [] END |
  MERGE (u:User {id: $user_node_id})
  SET u.tenant_id = $tenant_id, u.user_id = $user_identity_id,
    u.display_name = $actor_display_name, u.is_bot = $actor_is_bot,
    u.user_type = CASE WHEN $actor_is_bot THEN 'bot' ELSE 'user' END,
    u.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(u)
)
FOREACH (_ IN CASE WHEN $has_user_identity AND $actor_is_bot THEN [1] ELSE [] END |
  MERGE (u:User {id: $user_node_id})
  SET u:Bot
)
FOREACH (_ IN CASE WHEN $has_user_identity AND NOT $actor_is_bot THEN [1] ELSE [] END |
  MERGE (u:User {id: $user_node_id})
  REMOVE u:Bot
)
FOREACH (_ IN CASE WHEN $has_message_subject THEN [1] ELSE [] END |
  MERGE (m:Message {id: $message_node_id})
  SET m.tenant_id = $tenant_id, m.message_id = $message_id,
    m.summary = $summary, m.content = $message_content,
    m.deleted = $deleted, m.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(m)
)
FOREACH (_ IN CASE WHEN $has_message AND NOT $has_message_subject THEN [1] ELSE [] END |
  MERGE (m:Message {id: $message_node_id})
  SET m.tenant_id = $tenant_id, m.message_id = $message_id,
    m.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(m)
)
FOREACH (_ IN CASE WHEN $has_message AND $actor_id <> '' THEN [1] ELSE [] END |
  MERGE (u:User {id: $user_node_id})
  MERGE (m:Message {id: $message_node_id})
  MERGE (u)-[:AUTHORED]->(m)
)
FOREACH (_ IN CASE WHEN $has_message AND $has_channel THEN [1] ELSE [] END |
  MERGE (m:Message {id: $message_node_id})
  MERGE (ch:Channel {id: $channel_node_id})
  MERGE (m)-[:IN_CHANNEL]->(ch)
)
FOREACH (_ IN CASE WHEN $has_link THEN [1] ELSE [] END |
  MERGE (l:Link {id: $link_node_id})
  SET l.tenant_id = $tenant_id, l.url = $link_url, l.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(l)
)
FOREACH (_ IN CASE WHEN $has_link AND $has_message THEN [1] ELSE [] END |
  MERGE (l:Link {id: $link_node_id})
  MERGE (m:Message {id: $message_node_id})
  MERGE (l)-[:LINKED_FROM]->(m)
)
FOREACH (_ IN CASE WHEN $has_attachment THEN [1] ELSE [] END |
  MERGE (att:Attachment {id: $attachment_node_id})
  SET att.tenant_id = $tenant_id, att.filename = $attachment_filename,
    att.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(att)
)
FOREACH (_ IN CASE WHEN $has_attachment AND $has_message THEN [1] ELSE [] END |
  MERGE (att:Attachment {id: $attachment_node_id})
  MERGE (m:Message {id: $message_node_id})
  MERGE (att)-[:ATTACHED_TO]->(m)
)
FOREACH (_ IN CASE WHEN $has_category THEN [1] ELSE [] END |
  MERGE (cat:Category {id: $category_node_id})
  SET cat.tenant_id = $tenant_id, cat.guild_id = $guild_id,
    cat.category_id = $category_id,
    cat.name = coalesce(nullif($category_name, ''), cat.name),
    cat.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(cat)
)
FOREACH (_ IN CASE WHEN $has_category AND $has_guild THEN [1] ELSE [] END |
  MERGE (cat:Category {id: $category_node_id})
  MERGE (g:Guild {id: $guild_node_id})
  MERGE (cat)-[:IN_GUILD]->(g)
)
FOREACH (_ IN CASE WHEN $has_channel_category THEN [1] ELSE [] END |
  MERGE (ch:Channel {id: $channel_node_id})
  MERGE (cat:Category {id: $category_node_id})
  MERGE (ch)-[:IN_CATEGORY]->(cat)
)
FOREACH (_ IN CASE WHEN $has_role THEN [1] ELSE [] END |
  MERGE (r:Role {id: $role_node_id})
  SET r.tenant_id = $tenant_id, r.guild_id = $guild_id,
    r.role_id = $role_id,
    r.name = coalesce(nullif($role_name, ''), r.name),
    r.deleted = $deleted, r.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(r)
)
FOREACH (_ IN CASE WHEN $has_role AND $has_guild THEN [1] ELSE [] END |
  MERGE (g:Guild {id: $guild_node_id})
  MERGE (r:Role {id: $role_node_id})
  MERGE (g)-[:OWNS_ROLE]->(r)
)
FOREACH (_ IN CASE WHEN $has_member_user THEN [1] ELSE [] END |
  MERGE (member:User {id: $member_user_node_id})
  SET member.tenant_id = $tenant_id, member.user_id = $member_user_id,
    member.updated_at = datetime()
  MERGE (e)-[:AFFECTS]->(member)
)
FOREACH (_ IN CASE WHEN $has_member_identity_snapshot THEN [1] ELSE [] END |
  MERGE (member:User {id: $member_user_node_id})
  SET member.display_name = $actor_display_name,
    member.is_bot = $member_is_bot,
    member.user_type = $member_user_type,
    member.updated_at = datetime()
)
FOREACH (_ IN CASE
  WHEN $has_member_identity_snapshot AND $member_is_bot THEN [1]
  ELSE []
END |
  MERGE (member:User {id: $member_user_node_id})
  SET member:Bot
)
FOREACH (_ IN CASE
  WHEN $has_member_identity_snapshot AND NOT $member_is_bot THEN [1]
  ELSE []
END |
  MERGE (member:User {id: $member_user_node_id})
  REMOVE member:Bot
)
FOREACH (member_role_node_id IN $member_role_node_ids |
  MERGE (member:User {id: $member_user_node_id})
  MERGE (role:Role {id: member_role_node_id})
  MERGE (member)-[:HAS_ROLE]->(role)
  MERGE (e)-[:AFFECTS]->(role)
)
WITH e, duplicate
OPTIONAL MATCH (:User {id: $member_user_node_id})-[stale_role:HAS_ROLE]->(
  stale_role_node:Role
)
FOREACH (_ IN CASE
  WHEN $has_member_role_snapshot
    AND stale_role IS NOT NULL
    AND stale_role_node.id IN $member_role_node_ids THEN []
  WHEN $has_member_role_snapshot AND stale_role IS NOT NULL THEN [1]
  ELSE []
END | DELETE stale_role)
WITH e, duplicate
OPTIONAL MATCH (:User {id: $member_user_node_id})-[current_role:HAS_ROLE]->(
  :Role {id: $role_node_id}
)
FOREACH (_ IN CASE
  WHEN $is_member_role_unassignment AND current_role IS NOT NULL THEN [1]
  ELSE []
END | DELETE current_role)
RETURN duplicate AS duplicate
"""
