"""Entity-anchored graph traversal for multi-hop context retrieval (T1).

Run 10 (entity-graph benchmark, 2026-07-04) proved a materialized entity
graph is inert without something driving the traversal: the LLM-planned
Cypher route left multi-hop exactly flat. The docs/multihop-techniques.md
prior (self-ask arXiv 2210.03350, IRCoT arXiv 2212.10509, HippoRAG-2 arXiv
2405.14831) says the missing piece is resolving the *bridge* entity - the
one that answers hop 1 but never appears in the query text, so no dense or
lexical ranking of the query can surface hop 2's evidence.

Behind ``GNOSIS_GRAPH_TRAVERSAL_ENABLED`` (default off), this module drives
the graph deterministically, with zero extra LLM calls:

1. **Pin seeds.** Every 1..4-word phrase of the query is normalized exactly
   like entity names at write time (``normalize_entity_name``), so query
   mentions match ``:Entity`` nodes by equality - no fuzzy matching, no
   Lucene, no injection surface.
2. **Traverse.** A fixed parameterized Cypher expands 1-2 ``RELATES`` hops
   from every pinned seed - reaching the bridge entity and its neighborhood
   without naming it - and follows edge provenance (``fact_id``) back to the
   dated extracted ``:Fact`` units that asserted each relationship.
3. **Fuse.** The provenance facts join the context candidates as
   graph-derived candidates, holding the reserved graph slots of the item
   budget (they carry their real dated content, so they render exactly like
   dense-ranked facts).

Every read is scope-pinned twice: seeds match only within
``tenant_id`` + ``user_id`` (the entity dedup scope) and each provenance
fact re-checks the caller's metadata scope fragments in-query, with the
gateway re-checking scope again on the deserialized rows.
"""

from typing import Final

from gnosis.entity_graph import normalize_entity_name
from gnosis.graph_types import CypherParameters
from gnosis.models import JsonValue

# Longest seed phrase, in words. Entity names are short noun phrases; the
# longest observed on the benchmark store is four words.
_MAX_SEED_WORDS: Final[int] = 4

# Upper bound on seed phrases sent to the traversal query, so a pathological
# query cannot build an unbounded parameter list.
_MAX_SEEDS: Final[int] = 128

# Expand 1-2 RELATES hops from every entity the query names and follow each
# traversed edge's provenance back to the dated extracted fact that asserted
# it. Shorter paths rank first (a 1-hop neighbor is stronger evidence than a
# 2-hop one), newest facts first within a depth. Both endpoints of the
# traversal stay inside the tenant+user entity scope by construction (RELATES
# never bridges scopes at write time) and every provenance fact re-checks the
# caller's scope fragments before it can surface.
ENTITY_TRAVERSAL_CYPHER: Final[str] = """
MATCH (seed:Entity {tenant_id: $tenant_id, user_id: $user_id})
WHERE seed.normalized IN $seeds
MATCH path = (seed)-[:RELATES*1..2]-(:Entity)
UNWIND relationships(path) AS edge
WITH edge.fact_id AS fact_id, min(length(path)) AS depth
MATCH (f:Fact {id: fact_id})
WHERE f.metadata IS NOT NULL
  AND all(fragment IN $scope_fragments WHERE f.metadata CONTAINS fragment)
RETURN f.id AS id,
       f.subject AS subject,
       f.predicate AS predicate,
       f.object AS object,
       f.metadata AS metadata,
       toString(f.created_at) AS created_at,
       toString(f.updated_at) AS updated_at
ORDER BY depth ASC, created_at DESC, id ASC
LIMIT $limit
"""

_POSSESSIVE_SUFFIXES: Final[tuple[str, ...]] = ("'s", "\u2019s")

# Only sentence punctuation strips off word edges - characters that end or
# quote a sentence, never characters that can end an entity's own name
# ("lgbtq+" keeps its plus sign so it still pins "lgbtq+ pride parade").
_EDGE_PUNCTUATION: Final[str] = ".,;:!?\"'()[]{}\u2018\u2019\u201c\u201d"


def _cleaned_word(word: str) -> str:
    """One query word with possessive suffix and sentence punctuation removed.

    ``Caroline's`` cleans to ``caroline`` and ``library?`` to ``library``,
    matching how those mentions were named as entities at write time. Only
    sentence punctuation strips off the edges, so name-bearing punctuation
    survives; the raw phrase variant still covers names with interior
    apostrophes (``caroline's grandma``).
    """
    lowered = word.casefold().strip(_EDGE_PUNCTUATION)
    for suffix in _POSSESSIVE_SUFFIXES:
        lowered = lowered.removesuffix(suffix)
    return lowered


def query_seed_candidates(query: str) -> list[str]:
    """Normalized query phrases that can pin ``:Entity`` seed nodes.

    Every contiguous 1..4-word phrase of the query, normalized with the same
    ``normalize_entity_name`` used at entity write time so a query mention
    matches its node by plain equality. Each phrase also contributes a
    cleaned variant with possessives and edge punctuation removed
    ("Caroline's" pins ``caroline``; a trailing "library?" pins
    ``library``), deduplicated in first-seen order and capped so the
    traversal parameter list stays bounded.
    """
    words = query.split()
    seeds: dict[str, None] = {}
    for start in range(len(words)):
        for width in range(1, _MAX_SEED_WORDS + 1):
            if start + width > len(words):
                break
            raw = " ".join(words[start : start + width])
            cleaned = " ".join(
                cleaned_word
                for word in words[start : start + width]
                if (cleaned_word := _cleaned_word(word))
            )
            for variant in (raw, cleaned):
                normalized = normalize_entity_name(variant)
                if normalized:
                    seeds.setdefault(normalized, None)
            if len(seeds) >= _MAX_SEEDS:
                return list(seeds)
    return list(seeds)


def traversal_parameters(
    *,
    tenant_id: str,
    user_id: str,
    seeds: list[str],
    scope_fragments: list[JsonValue],
    limit: int,
) -> CypherParameters:
    """Parameters for one scope-pinned ``ENTITY_TRAVERSAL_CYPHER`` read."""
    return {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "seeds": list(seeds),
        "scope_fragments": scope_fragments,
        "limit": limit,
    }
