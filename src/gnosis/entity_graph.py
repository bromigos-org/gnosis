"""Entity-relationship graph materialization for multi-hop QA.

Behind ``GNOSIS_ENTITY_GRAPH_ENABLED``, each extracted edu-v1 memory unit also
materializes a traversable knowledge graph next to its ``:Fact`` node
(HippoRAG-2 / Graphiti approach, arXiv 2405.14831):

* an ``:Entity`` node per named entity, deduplicated by normalized name within
  ``tenant_id`` + ``user_id`` scope - entities are never merged across tenants
  or users;
* a ``(:Fact)-[:MENTIONS]->(:Entity)`` edge linking each fact to the entities
  it names;
* a directed ``(:Entity)-[:RELATES {relation, fact_id, event_date}]->(:Entity)``
  edge per extracted ``(head, relation, tail)`` triple, so entities are
  connected and multi-hop questions can be answered by graph traversal.

Every write is scope-tagged and parameterized. This module is a pure builder of
``(cypher, parameters)`` statements; the backend runs them through the same
graph write handle the direct ``:Fact`` writes use and degrades to "no graph
materialized" with a structured warning on any failure, never failing the add.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from gnosis.graph_types import CypherParameters

if TYPE_CHECKING:
    from gnosis.models import JsonValue

# Composite range index over the scope-keyed entity dedup properties so the
# MERGE resolves by lookup instead of a label scan. A range index (not a
# uniqueness constraint) keeps this portable across Neo4j editions; MERGE
# itself provides the find-or-create dedup. Idempotent; created once per
# backend before the first entity write, through the graph write handle the
# fact writes use.
CREATE_ENTITY_SCOPE_INDEX_CYPHER: Final[str] = """
CREATE INDEX entity_scope_key IF NOT EXISTS
FOR (e:Entity) ON (e.tenant_id, e.user_id, e.normalized)
"""

# MERGE one :Entity per named entity within tenant+user scope and link the fact
# to it. The scope keys live in the MERGE pattern so a name can never collide
# across tenants or users; display name and id are set only on first create.
MERGE_ENTITY_MENTIONS_CYPHER: Final[str] = """
MATCH (f:Fact {id: $fact_id})
UNWIND $entities AS entity
MERGE (e:Entity {
    tenant_id: $tenant_id,
    user_id: $user_id,
    normalized: entity.normalized
})
ON CREATE SET e.id = entity.id, e.name = entity.name, e.created_at = datetime()
MERGE (f)-[:MENTIONS]->(e)
"""

# Connect the entities of a fact with the extracted directed relations. Both
# endpoints are matched within scope, so a relation can never bridge scopes;
# fact_id in the MERGE key keeps one edge per (relation, fact) for provenance.
MERGE_ENTITY_RELATIONS_CYPHER: Final[str] = """
UNWIND $relations AS rel
MATCH (head:Entity {
    tenant_id: $tenant_id,
    user_id: $user_id,
    normalized: rel.head
})
MATCH (tail:Entity {
    tenant_id: $tenant_id,
    user_id: $user_id,
    normalized: rel.tail
})
MERGE (head)-[r:RELATES {relation: rel.relation, fact_id: $fact_id}]->(tail)
ON CREATE SET r.event_date = $event_date, r.created_at = datetime()
"""


@dataclass(frozen=True, slots=True)
class RelationTriple:
    """One directed ``(head, relation, tail)`` triple for a RELATES edge."""

    head: str
    relation: str
    tail: str


def normalize_entity_name(name: str) -> str:
    """Scope-local dedup key for an entity name.

    Case-folds and collapses internal whitespace so ``"New York"`` and ``"new
    york"`` merge into one node while preserving the first-seen display name.
    """
    return " ".join(name.split()).casefold()


def entity_id(tenant_id: str, user_id: str, normalized: str) -> str:
    """Deterministic, scope-encoding id for an entity node."""
    return f"tenant:{tenant_id}:user:{user_id}:entity:{normalized}"


def entity_graph_statements(  # noqa: PLR0913 - One argument per scope/graph input.
    *,
    tenant_id: str,
    user_id: str,
    fact_id: str,
    entities: Sequence[str],
    relations: Sequence[RelationTriple],
    event_date: str | None,
) -> list[tuple[str, CypherParameters]]:
    """Build the scope-tagged entity-graph writes for one extracted fact.

    Returns an ordered list of ``(cypher, parameters)`` statements: first the
    MENTIONS write that MERGEs every named entity (the union of the unit's
    entities and any relation endpoints, so no relation dangles), then - when
    there are valid triples - the RELATES write connecting them. Returns an
    empty list when the fact names no entities, so a flag-on write with nothing
    to materialize adds no statements.
    """
    named = _deduped_entities(entities, relations)
    if not named:
        return []
    known = {normalized for normalized, _ in named}
    entity_rows: list[JsonValue] = [
        {
            "normalized": normalized,
            "name": display,
            "id": entity_id(tenant_id, user_id, normalized),
        }
        for normalized, display in named
    ]
    statements: list[tuple[str, CypherParameters]] = [
        (
            MERGE_ENTITY_MENTIONS_CYPHER,
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "fact_id": fact_id,
                "entities": entity_rows,
            },
        ),
    ]
    relation_rows: list[JsonValue] = [
        {"head": head, "relation": label, "tail": tail}
        for head, label, tail in _deduped_relations(relations, known)
    ]
    if relation_rows:
        statements.append(
            (
                MERGE_ENTITY_RELATIONS_CYPHER,
                {
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "fact_id": fact_id,
                    "event_date": event_date,
                    "relations": relation_rows,
                },
            ),
        )
    return statements


def _deduped_entities(
    entities: Sequence[str],
    relations: Sequence[RelationTriple],
) -> list[tuple[str, str]]:
    """Deduplicated ``(normalized, display)`` pairs for the fact's entities.

    The union of the unit's ``entities`` and every relation endpoint is taken
    so a relation head or tail the model omitted from ``entities`` still gets a
    node to link; blank names are dropped and first-seen display name wins.
    """
    names: list[str] = [*entities]
    for relation in relations:
        names.extend((relation.head, relation.tail))
    pairs: dict[str, str] = {}
    for name in names:
        display = name.strip()
        if not display:
            continue
        normalized = normalize_entity_name(display)
        if not normalized or normalized in pairs:
            continue
        pairs[normalized] = display
    return list(pairs.items())


def _deduped_relations(
    relations: Sequence[RelationTriple],
    known: set[str],
) -> list[tuple[str, str, str]]:
    """Normalized, self-loop-free ``(head, relation, tail)`` triples for edges.

    A triple contributes an edge only when head, relation, and tail are all
    non-blank, head and tail normalize to distinct entities, and both endpoints
    are among the entities MERGEd for this fact.
    """
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations:
        head = normalize_entity_name(relation.head.strip())
        tail = normalize_entity_name(relation.tail.strip())
        label = " ".join(relation.relation.split())
        if not head or not tail or not label or head == tail:
            continue
        if head not in known or tail not in known:
            continue
        key = (head, label, tail)
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    return rows
