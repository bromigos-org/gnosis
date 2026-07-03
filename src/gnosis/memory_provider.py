"""Record mapping and Cypher for the /v1/memories provider surface.

Provider memories are stored as SDK long-term ``Fact`` nodes. Scope fields are
write-side tags inside the fact metadata JSON, so reads narrow with
parameterized metadata fragments and the gateway re-checks scope and filter
semantics on the deserialized records before anything leaves the service.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
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
