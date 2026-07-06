from typing import cast

from gnosis.entity_graph import (
    MERGE_ENTITY_MENTIONS_CYPHER,
    MERGE_ENTITY_RELATIONS_CYPHER,
    RelationTriple,
    entity_graph_statements,
    entity_id,
    normalize_entity_name,
)
from gnosis.graph_types import CypherParameters


def _rows(params: CypherParameters, key: str) -> list[dict[str, str]]:
    """Narrow a JsonValue param list of row dicts for assertion purposes."""
    return cast("list[dict[str, str]]", params[key])


def test_normalize_entity_name_folds_case_and_collapses_whitespace() -> None:
    # Then: casing and internal whitespace are normalized for scope-local dedup.
    assert normalize_entity_name("  New   York ") == "new york"
    assert normalize_entity_name("Alice") == normalize_entity_name("alice")


def test_entity_id_encodes_scope_and_normalized_name() -> None:
    # Then: the id is deterministic and encodes tenant + user + entity.
    assert entity_id("nolgia", "789", "alice") == (
        "tenant:nolgia:user:789:entity:alice"
    )


def test_statements_materialize_scoped_mentions_and_relations() -> None:
    # Given: a fact naming two entities linked by one directed triple.
    statements = entity_graph_statements(
        tenant_id="nolgia",
        user_id="789",
        fact_id="fact-1",
        entities=["Alice", "Acme Corp"],
        relations=[RelationTriple(head="Alice", relation="works at", tail="Acme Corp")],
        event_date="2023-05-07",
    )

    # Then: a MENTIONS write MERGEs both scope-keyed entities and links the fact.
    assert len(statements) == 2
    mentions_query, mentions_params = statements[0]
    assert mentions_query == MERGE_ENTITY_MENTIONS_CYPHER
    assert mentions_params["tenant_id"] == "nolgia"
    assert mentions_params["user_id"] == "789"
    assert mentions_params["fact_id"] == "fact-1"
    assert _rows(mentions_params, "entities") == [
        {
            "normalized": "alice",
            "name": "Alice",
            "id": "tenant:nolgia:user:789:entity:alice",
        },
        {
            "normalized": "acme corp",
            "name": "Acme Corp",
            "id": "tenant:nolgia:user:789:entity:acme corp",
        },
    ]

    # Then: a RELATES write connects the two entities by normalized name,
    # carrying the relation, the fact id, and the fact's event date.
    relations_query, relations_params = statements[1]
    assert relations_query == MERGE_ENTITY_RELATIONS_CYPHER
    assert relations_params["tenant_id"] == "nolgia"
    assert relations_params["user_id"] == "789"
    assert relations_params["fact_id"] == "fact-1"
    assert relations_params["event_date"] == "2023-05-07"
    assert _rows(relations_params, "relations") == [
        {"head": "alice", "relation": "works at", "tail": "acme corp"},
    ]


def test_statements_add_relation_endpoints_missing_from_entities() -> None:
    # Given: a triple whose tail was not listed in the unit's entities.
    statements = entity_graph_statements(
        tenant_id="nolgia",
        user_id="789",
        fact_id="fact-2",
        entities=["Alice"],
        relations=[RelationTriple(head="Alice", relation="knows", tail="Bob")],
        event_date=None,
    )

    # Then: the union of entities and endpoints is MERGEd so no edge dangles.
    _, mentions_params = statements[0]
    normalized = {row["normalized"] for row in _rows(mentions_params, "entities")}
    assert normalized == {"alice", "bob"}
    _, relations_params = statements[1]
    assert relations_params["event_date"] is None
    assert _rows(relations_params, "relations") == [
        {"head": "alice", "relation": "knows", "tail": "bob"},
    ]


def test_statements_drop_self_loops_blank_and_unbalanced_triples() -> None:
    # Given: triples that a knowledge graph must not materialize as edges.
    statements = entity_graph_statements(
        tenant_id="nolgia",
        user_id="789",
        fact_id="fact-3",
        entities=["Alice", "Bob", ""],
        relations=[
            RelationTriple(head="Alice", relation="is", tail="Alice"),
            RelationTriple(head="Alice", relation="", tail="Bob"),
            RelationTriple(head="", relation="knows", tail="Bob"),
            RelationTriple(head="Alice", relation="knows", tail="Bob"),
            RelationTriple(head="alice", relation="knows", tail="bob"),
        ],
        event_date=None,
    )

    # Then: only the one valid, deduplicated, non-self-loop triple survives,
    # and the blank entity name is dropped from the MERGE set.
    _, mentions_params = statements[0]
    assert [row["normalized"] for row in _rows(mentions_params, "entities")] == [
        "alice",
        "bob",
    ]
    _, relations_params = statements[1]
    assert _rows(relations_params, "relations") == [
        {"head": "alice", "relation": "knows", "tail": "bob"},
    ]


def test_statements_keep_relation_endpoints_as_entities() -> None:
    # Given: a triple whose endpoints never appear in the entities list.
    statements = entity_graph_statements(
        tenant_id="nolgia",
        user_id="789",
        fact_id="fact-4",
        entities=["Alice"],
        relations=[RelationTriple(head="Carol", relation="knows", tail="Dave")],
        event_date=None,
    )

    # Then: the endpoints are still MERGEd as entities (they are named in the
    # relation) so the edge is materializable, never silently dropped.
    _, mentions_params = statements[0]
    assert {row["normalized"] for row in _rows(mentions_params, "entities")} == {
        "alice",
        "carol",
        "dave",
    }
    _, relations_params = statements[1]
    assert _rows(relations_params, "relations") == [
        {"head": "carol", "relation": "knows", "tail": "dave"},
    ]


def test_statements_without_entities_materialize_nothing() -> None:
    # Given: a unit that names no entities and states no relations.
    statements = entity_graph_statements(
        tenant_id="nolgia",
        user_id="789",
        fact_id="fact-5",
        entities=[],
        relations=[],
        event_date=None,
    )

    # Then: there is nothing to materialize.
    assert statements == []


def test_statements_with_only_entities_emit_mentions_only() -> None:
    # Given: a unit that names entities but states no relationship.
    statements = entity_graph_statements(
        tenant_id="nolgia",
        user_id="789",
        fact_id="fact-6",
        entities=["Alice", "Tokyo"],
        relations=[],
        event_date=None,
    )

    # Then: only the MENTIONS write is produced.
    assert len(statements) == 1
    assert statements[0][0] == MERGE_ENTITY_MENTIONS_CYPHER


def test_statements_isolate_entities_by_scope() -> None:
    # Given: the same entity name under two different user scopes.
    first = entity_graph_statements(
        tenant_id="nolgia",
        user_id="alice",
        fact_id="fact-a",
        entities=["Paris"],
        relations=[],
        event_date=None,
    )
    second = entity_graph_statements(
        tenant_id="nolgia",
        user_id="bob",
        fact_id="fact-b",
        entities=["Paris"],
        relations=[],
        event_date=None,
    )

    # Then: the scope-keyed ids differ, so the MERGE can never merge the two
    # users' entities into one node.
    assert _rows(first[0][1], "entities")[0]["id"] == (
        "tenant:nolgia:user:alice:entity:paris"
    )
    assert _rows(second[0][1], "entities")[0]["id"] == (
        "tenant:nolgia:user:bob:entity:paris"
    )
