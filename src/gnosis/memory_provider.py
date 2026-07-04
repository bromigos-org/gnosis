"""Record mapping and Cypher for the /v1/memories provider surface.

Provider memories are stored as SDK long-term ``Fact`` nodes. Scope fields are
write-side tags inside the fact metadata JSON, so reads narrow with
parameterized metadata fragments and the gateway re-checks scope and filter
semantics on the deserialized records before anything leaves the service.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final

from pydantic import BaseModel, TypeAdapter, ValidationError

from gnosis.memory_filters import MemoryFilterFields
from gnosis.models import (
    JsonObject,
    JsonValue,
    MemoryAddEvent,
    MemoryRecord,
    MemoryScope,
)
from gnosis.redaction import redact_secrets

_JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)
_SCOPE_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "tenant_id",
        "space_id",
        "agent_id",
        "session_id",
        "user_id",
        "visibility",
        "guild_id",
        "channel_id",
    },
)
_PRIVATE_METADATA_KEYS: Final[frozenset[str]] = _SCOPE_METADATA_KEYS | {
    "similarity",
    "deduplicated",
}

VERBATIM_MEMORY_PREDICATE: Final[str] = "memory"
TURN_MEMORY_PREDICATE_PREFIX: Final[str] = "said_"
EXTRACTED_FACT_PREDICATE: Final[str] = "fact"

# Standard Reciprocal Rank Fusion constant (EverMemOS uses the same k=60).
RRF_K: Final[int] = 60

# Lucene query syntax that user input must never inject: every special
# character (https://lucene.apache.org/ escaping rules) plus the bare boolean
# operator words, which are only operators when uppercase.
_LUCENE_SPECIAL_CHARS: Final[frozenset[str]] = frozenset('+-&|!(){}[]^"~*?:\\/')
_LUCENE_OPERATOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b(?:AND|OR|NOT)\b")

FACT_OBJECT_FULLTEXT_INDEX: Final[str] = "fact_object_fulltext"

# Neo4j 5.x full-text (Lucene/BM25) index over stored fact content. The SDK
# owns the rest of the Fact schema, so this one gateway-owned index is created
# idempotently through the same graph write handle the direct Fact writes use.
CREATE_FACT_OBJECT_FULLTEXT_INDEX_CYPHER: Final[str] = f"""
CREATE FULLTEXT INDEX {FACT_OBJECT_FULLTEXT_INDEX} IF NOT EXISTS
FOR (f:Fact) ON EACH [f.object]
"""

MEMORY_RETURN_CYPHER: Final[str] = """
RETURN f.id AS id,
       f.subject AS subject,
       f.predicate AS predicate,
       f.object AS object,
       f.metadata AS metadata,
       toString(f.created_at) AS created_at,
       toString(f.updated_at) AS updated_at
"""

LOOKUP_MEMORY_CYPHER: Final[str] = f"""
MATCH (f:Fact {{id: $memory_id}})
{MEMORY_RETURN_CYPHER}
LIMIT 1
"""

LOOKUP_LATEST_MEMORY_CYPHER: Final[str] = f"""
MATCH (f:Fact)
WHERE f.subject = $subject
  AND f.predicate = $predicate
  AND f.object = $object
  AND f.metadata IS NOT NULL
  AND all(fragment IN $scope_fragments WHERE f.metadata CONTAINS fragment)
{MEMORY_RETURN_CYPHER}
ORDER BY f.created_at DESC
LIMIT 1
"""

# Recent verbatim turn facts for one session, newest first. The extraction
# context window reads these because they are exactly the turns the add path
# writes, with the speaker role recoverable from the ``said_*`` predicate.
RECENT_TURN_MEMORIES_CYPHER: Final[str] = f"""
MATCH (f:Fact)
WHERE f.subject = $subject
  AND f.predicate STARTS WITH $predicate_prefix
  AND f.metadata IS NOT NULL
  AND all(fragment IN $scope_fragments WHERE f.metadata CONTAINS fragment)
{MEMORY_RETURN_CYPHER}
ORDER BY f.created_at DESC, f.id DESC
LIMIT $limit
"""

# Extracted-fact writes create the node directly instead of going through the
# SDK's add_fact, because add_fact's write-time dedup can silently swallow a
# near-duplicate into an existing fact - for extracted units two same-day
# duplicates are harmless, but a swallowed distinct dated event is a lost
# answer. The property shape mirrors the SDK's CREATE_FACT query.
CREATE_MEMORY_CYPHER: Final[str] = """
CREATE (f:Fact {
    id: $memory_id,
    subject: $subject,
    predicate: $predicate,
    object: $object,
    confidence: 1.0,
    embedding: $embedding,
    created_at: datetime(),
    metadata: $metadata
})
"""

UPDATE_MEMORY_CYPHER: Final[str] = """
MATCH (f:Fact {id: $memory_id})
SET f.object = coalesce($content, f.object),
    f.metadata = coalesce($metadata, f.metadata),
    f.embedding = CASE
        WHEN $embedding IS NULL THEN f.embedding ELSE $embedding END,
    f.updated_at = datetime()
RETURN f.object AS object
"""

DELETE_MEMORY_CYPHER: Final[str] = """
MATCH (f:Fact {id: $memory_id})
DETACH DELETE f
"""

# Lexical (BM25) candidates for hybrid retrieval. Scoped exactly like the
# other provider reads: parameterized metadata fragments narrow in-query and
# the gateway re-checks scope on the deserialized records afterwards. The
# query string must already be Lucene-sanitized (``sanitize_lucene_query``).
LEXICAL_MEMORY_SEARCH_CYPHER: Final[str] = f"""
CALL db.index.fulltext.queryNodes('{FACT_OBJECT_FULLTEXT_INDEX}', $query)
YIELD node AS f, score
WHERE f.metadata IS NOT NULL
  AND all(fragment IN $scope_fragments WHERE f.metadata CONTAINS fragment)
{MEMORY_RETURN_CYPHER}
ORDER BY score DESC
LIMIT $candidate_limit
"""


def list_memories_cypher(narrowing_fragment: str) -> str:
    return f"""
MATCH (f:Fact)
WHERE f.metadata IS NOT NULL
  AND all(fragment IN $scope_fragments WHERE f.metadata CONTAINS fragment)
  AND {narrowing_fragment}
{MEMORY_RETURN_CYPHER}
ORDER BY f.created_at DESC, f.id ASC
LIMIT $scan_limit
"""


@dataclass(frozen=True, slots=True)
class StoredMemory:
    memory_id: str
    subject: str
    predicate: str
    content: str
    metadata: JsonObject
    created_at: str | None
    updated_at: str | None


def scope_read_fragments(scope: MemoryScope) -> list[JsonValue]:
    return [
        _metadata_json_fragment("tenant_id", scope.tenant_id),
        _metadata_json_fragment("user_id", scope.user_id),
    ]


def session_read_fragments(scope: MemoryScope) -> list[JsonValue]:
    return [
        *scope_read_fragments(scope),
        _metadata_json_fragment("session_id", scope.session_id),
    ]


def sanitize_lucene_query(query: str) -> str:
    """Neutralize Lucene query syntax in user-supplied search text.

    Every Lucene special character is backslash-escaped and the bare boolean
    operator words (AND/OR/NOT) are lowercased - the index analyzer lowercases
    terms anyway, so matching is unchanged - which means user input can never
    inject Lucene operators, field selectors, or unbalanced syntax. An empty
    result tells the caller to skip the lexical leg entirely.
    """
    escaped = "".join(
        f"\\{char}" if char in _LUCENE_SPECIAL_CHARS else char for char in query
    )
    return _LUCENE_OPERATOR_PATTERN.sub(
        lambda match: match.group(0).lower(),
        escaped,
    ).strip()


def fuse_memory_rankings(
    dense: Sequence[StoredMemory],
    lexical: Sequence[StoredMemory],
    *,
    k: int = RRF_K,
) -> list[StoredMemory]:
    """Fuse the dense and lexical candidate rankings with standard RRF.

    Each ranking contributes ``1 / (k + rank)`` per memory (rank is 1-based
    within that ranking; on a duplicate id the best rank wins) and the summed
    scores decide the fused order. When both rankings carry the same memory
    the dense record is kept as the representative so its vector similarity
    survives into the response; ties keep dense-first arrival order.
    """
    fused_scores: dict[str, float] = {}
    representatives: dict[str, StoredMemory] = {}
    for ranking in (dense, lexical):
        rank = 0
        seen: set[str] = set()
        for memory in ranking:
            if memory.memory_id in seen:
                continue
            seen.add(memory.memory_id)
            rank += 1
            fused_scores[memory.memory_id] = fused_scores.get(
                memory.memory_id,
                0.0,
            ) + 1.0 / (k + rank)
            _ = representatives.setdefault(memory.memory_id, memory)
    arrival_order = {
        memory_id: position for position, memory_id in enumerate(fused_scores)
    }
    return [
        representatives[memory_id]
        for memory_id in sorted(
            fused_scores,
            key=lambda memory_id: (-fused_scores[memory_id], arrival_order[memory_id]),
        )
    ]


def lexical_stored_memory(memory: StoredMemory) -> StoredMemory:
    """Tag a lexical candidate with a zero vector-similarity score.

    Keeps the response contract: ``score`` stays the vector similarity when
    the dense ranking saw the record (the dense representative wins fusion
    dedupe), while a purely lexical hit surfaces with score 0.0 because RRF
    rank, not similarity, admitted it.
    """
    return replace(memory, metadata={**memory.metadata, "similarity": 0.0})


def stored_memory_from_sdk(record: object) -> StoredMemory | None:
    return _stored_memory(_sdk_payload(record))


def stored_memories_from_sdk(records: object) -> list[StoredMemory]:
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        return []
    return [
        memory
        for record in records
        if (memory := stored_memory_from_sdk(record)) is not None
    ]


def stored_memory_from_row(row: JsonObject) -> StoredMemory | None:
    return _stored_memory(row)


def memory_matches_scope(memory: StoredMemory, scope: MemoryScope) -> bool:
    return (
        memory.metadata.get("tenant_id") == scope.tenant_id
        and memory.metadata.get("user_id") == scope.user_id
    )


def memory_filter_fields(memory: StoredMemory) -> MemoryFilterFields:
    return MemoryFilterFields(
        user_id=_metadata_text(memory.metadata, "user_id"),
        agent_id=_metadata_text(memory.metadata, "agent_id"),
        created_at=_parsed_timestamp(memory.created_at),
        metadata=memory.metadata,
    )


def memory_score(memory: StoredMemory) -> float | None:
    score = memory.metadata.get("similarity")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return min(max(float(score), 0.0), 1.0)


def memory_add_event(memory: StoredMemory) -> MemoryAddEvent:
    if memory.metadata.get("deduplicated") is True:
        return "UPDATE"
    return "ADD"


def public_memory_metadata(memory: StoredMemory) -> JsonObject:
    metadata = {
        key: value
        for key, value in memory.metadata.items()
        if key not in _PRIVATE_METADATA_KEYS
    }
    return _redacted_object(metadata)


def memory_record(memory: StoredMemory, *, include_score: bool) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory.memory_id,
        content=_redacted_text(memory.content),
        score=memory_score(memory) if include_score else None,
        metadata=public_memory_metadata(memory),
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


def merged_memory_metadata(memory: StoredMemory, update: JsonObject) -> str:
    merged = dict(memory.metadata) | _redacted_object(update)
    for key in _SCOPE_METADATA_KEYS:
        if key in memory.metadata:
            merged[key] = memory.metadata[key]
        else:
            _ = merged.pop(key, None)
    return json.dumps(merged)


def _stored_memory(payload: JsonObject) -> StoredMemory | None:
    memory_id = payload.get("id")
    subject = payload.get("subject")
    predicate = payload.get("predicate")
    content = payload.get("object")
    if not (
        isinstance(memory_id, str)
        and memory_id
        and isinstance(subject, str)
        and isinstance(predicate, str)
        and isinstance(content, str)
        and content
    ):
        return None
    return StoredMemory(
        memory_id=memory_id,
        subject=subject,
        predicate=predicate,
        content=content,
        metadata=_memory_metadata(payload.get("metadata")),
        created_at=_optional_timestamp_text(payload.get("created_at")),
        updated_at=_optional_timestamp_text(payload.get("updated_at")),
    )


def _sdk_payload(record: object) -> JsonObject:
    if isinstance(record, BaseModel):
        try:
            return _json_object(record.model_dump(mode="json", exclude={"embedding"}))
        except ValueError:
            return {}
    try:
        return _json_object(vars(record))
    except TypeError:
        return _json_object(record)


def _memory_metadata(value: JsonValue | None) -> JsonObject:
    if isinstance(value, dict):
        return _json_object(value)
    if isinstance(value, str):
        try:
            return _JSON_OBJECT_ADAPTER.validate_json(value)
        except ValidationError:
            return {}
    return {}


def _metadata_text(metadata: JsonObject, key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _optional_timestamp_text(value: JsonValue | None) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _parsed_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _metadata_json_fragment(key: str, value: str) -> str:
    return f"{json.dumps(key)}: {json.dumps(value)}"


def _json_object(value: object) -> JsonObject:
    try:
        return _JSON_OBJECT_ADAPTER.validate_python(value)
    except ValidationError:
        return {}


def _redacted_text(value: str) -> str:
    redacted = redact_secrets(value)
    if isinstance(redacted, str):
        return redacted
    return value


def _redacted_object(value: JsonObject) -> JsonObject:
    redacted = redact_secrets(value)
    if isinstance(redacted, dict):
        return redacted
    return {}
