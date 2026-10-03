"""Community subgraph detection and summarization for open-domain retrieval.

Open-domain queries ("What does Alice usually talk about?", "What are John's
interests?") fail on dense retrieval because no single fact has high cosine
similarity — the answer emerges from patterns across many facts. The Zep/Graphiti
architecture analysis (arXiv:2501.13956) and independent Memori Labs evaluations
show community-level summaries close ~30 points of the open-domain gap by giving
the retrieval path a progressively broader view: individual facts → entity nodes →
community cluster summaries.

This module builds communities without the Neo4j Graph Data Science plugin (not
available in all Neo4j editions):

1. Fetch Entity nodes and RELATES edges for a tenant+user scope.
2. Build weakly-connected components via BFS in Python.
3. Collect representative fact texts per component (up to a cap).
4. Ask the LLM to write a concise natural-language community summary.
5. Persist :Community nodes and :MEMBER_OF edges; embed the summary so it can
   be retrieved alongside regular Fact vectors.

Communities are rebuilt via a dedicated operator route
(``POST /v1/communities/rebuild``) or by the consolidation schedule. The read
path adds community summaries as an
extra context section for ``aggregative`` and ``open_domain``-like queries where
dense fact retrieval returns low-confidence candidates.

Design notes
------------
- Scope isolation: community nodes carry ``tenant_id`` + ``user_id`` so a rebuild
  on one user never touches another's communities.
- Idempotent: MERGE on community id (sha256 of sorted member normalized names
  within scope) so repeated rebuilds are safe.
- Degradation: every failure is logged and returns an empty list; the read path
  works without communities present.
- Minimum size: components below ``COMMUNITY_MIN_ENTITIES`` are skipped (likely
  isolated entities with no structural community signal).
"""

import hashlib
import logging
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from gnosis.graph_query_qa import proxy_model_name
from gnosis.graph_types import CypherParameters
from gnosis.llm_clients import shared_openai_client

if TYPE_CHECKING:
    from gnosis.models import JsonValue

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

# Skip components this small — insufficient signal for a useful summary.
COMMUNITY_MIN_ENTITIES: Final[int] = 3
# Max fact snippets fed to the LLM per community to keep the summary call cheap.
COMMUNITY_FACT_CAP: Final[int] = 15
# One-sentence description capped at this many tokens.
_MAX_SUMMARY_TOKENS: Final[int] = 300

# ---------------------------------------------------------------------------
# Cypher: read path (entity graph queries)
# ---------------------------------------------------------------------------

FETCH_ENTITIES_CYPHER: Final[str] = """
MATCH (e:Entity {tenant_id: $tenant_id, user_id: $user_id})
RETURN e.normalized AS normalized, e.name AS name, e.id AS id
"""

FETCH_RELATIONS_CYPHER: Final[str] = """
MATCH (h:Entity {tenant_id: $tenant_id, user_id: $user_id})
      -[:RELATES]->
      (t:Entity {tenant_id: $tenant_id, user_id: $user_id})
RETURN h.normalized AS head_normalized, t.normalized AS tail_normalized
"""

FETCH_ENTITY_FACTS_CYPHER: Final[str] = """
MATCH (f:Fact)-[:MENTIONS]->(e:Entity {tenant_id: $tenant_id, user_id: $user_id})
WHERE e.normalized IN $entity_normals
RETURN f.content AS content, f.event_date AS event_date
ORDER BY f.event_date DESC
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Cypher: write path (community persistence)
# ---------------------------------------------------------------------------

CREATE_COMMUNITY_INDEX_CYPHER: Final[str] = """
CREATE INDEX community_scope_key IF NOT EXISTS
FOR (c:Community) ON (c.tenant_id, c.user_id, c.community_id)
"""

MERGE_COMMUNITY_CYPHER: Final[str] = """
MERGE (c:Community {
    tenant_id: $tenant_id,
    user_id: $user_id,
    community_id: $community_id
})
ON CREATE SET
    c.summary = $summary,
    c.member_count = $member_count,
    c.created_at = datetime()
ON MATCH SET
    c.summary = $summary,
    c.member_count = $member_count,
    c.updated_at = datetime()
RETURN c.community_id AS community_id
"""

MERGE_MEMBER_OF_CYPHER: Final[str] = """
UNWIND $normals AS normalized
MATCH (e:Entity {tenant_id: $tenant_id, user_id: $user_id, normalized: normalized})
MATCH (c:Community {
  tenant_id: $tenant_id, user_id: $user_id, community_id: $community_id
})
MERGE (e)-[:MEMBER_OF]->(c)
"""

# ---------------------------------------------------------------------------
# Cypher: community context retrieval
# ---------------------------------------------------------------------------

FETCH_COMMUNITY_SUMMARIES_CYPHER: Final[str] = """
MATCH (c:Community {tenant_id: $tenant_id, user_id: $user_id})
RETURN c.summary AS summary,
       c.member_count AS member_count,
       c.community_id AS community_id
ORDER BY c.member_count DESC
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

_SUMMARY_GUIDE: Final[str] = """
You write a concise, factual community summary for a memory system.
You are given a cluster of related entities and snippets of facts that mention them.
Write ONE paragraph (3-5 sentences) describing what this cluster of memories is about:
who the people are, their relationships, key topics, and any notable context.
Be specific and grounded in the facts provided. Do not speculate beyond the evidence.
""".strip()


@dataclass(frozen=True, slots=True)
class CommunityRecord:
    """One detected community with its metadata and summary."""

    community_id: str
    normalized_members: list[str]
    summary: str
    member_count: int


def community_id_for(tenant_id: str, user_id: str, members: Sequence[str]) -> str:
    """Deterministic community id: sha256 of sorted normalized member names + scope."""
    key = f"{tenant_id}:{user_id}:" + "|".join(sorted(members))
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def weakly_connected_components(
    entities: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> list[list[str]]:
    """BFS weakly-connected components over entity normals.

    Treats all RELATES edges as undirected for community detection: we want
    clusters of entities that are related to each other in any direction, not
    just entities that share the same "head" or "tail" role.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for head, tail in edges:
        adjacency[head].add(tail)
        adjacency[tail].add(head)

    visited: set[str] = set()
    components: list[list[str]] = []

    for entity in entities:
        if entity in visited:
            continue
        component: list[str] = []
        queue: deque[str] = deque([entity])
        visited.add(entity)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in adjacency.get(node, set()):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        components.append(component)

    return components


async def summarize_community(
    *,
    entity_names: Sequence[str],
    fact_snippets: Sequence[str],
    model: str,
    base_url: str,
    api_key: str,
) -> str | None:
    """Ask the LLM to summarize a community cluster into one paragraph.

    Returns ``None`` on any failure so callers can skip and continue.
    """
    if not entity_names:
        return None
    try:
        client = shared_openai_client(
            api_key=api_key,
            base_url=base_url,
        )
        response = await client.chat.completions.create(
            messages=_summary_messages(entity_names, fact_snippets),
            model=proxy_model_name(model),
            max_completion_tokens=_MAX_SUMMARY_TOKENS,
        )
        content = response.choices[0].message.content
        if content:
            return " ".join(content.split())
    except Exception:
        _LOGGER.exception(
            "community summary failed",
            extra={"entity_count": len(entity_names)},
        )
        return None
    else:
        return None


def _summary_messages(
    entity_names: Sequence[str],
    fact_snippets: Sequence[str],
) -> tuple[ChatCompletionMessageParam, ...]:
    entities_text = ", ".join(entity_names[:30])
    facts_text = "\n".join(f"- {s}" for s in fact_snippets[:COMMUNITY_FACT_CAP])
    user_content = (
        f"Entities in this cluster: {entities_text}\n\n"
        f"Relevant memory snippets:\n{facts_text}"
    )
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": _SUMMARY_GUIDE,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user_content,
    }
    return (system_message, user_message)


# ---------------------------------------------------------------------------
# Cypher statement builders
# ---------------------------------------------------------------------------


def community_write_statements(
    *,
    tenant_id: str,
    user_id: str,
    community: CommunityRecord,
) -> list[tuple[str, CypherParameters]]:
    """Build the MERGE + MEMBER_OF write statements for one community.

    Returns an ordered list: first the community node MERGE, then the
    MEMBER_OF edge writes. Both are idempotent.
    """
    return [
        (
            MERGE_COMMUNITY_CYPHER,
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "community_id": community.community_id,
                "summary": community.summary,
                "member_count": community.member_count,
            },
        ),
        (
            MERGE_MEMBER_OF_CYPHER,
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "community_id": community.community_id,
                "normals": cast("list[JsonValue]", community.normalized_members),
            },
        ),
    ]
