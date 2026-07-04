"""Scope identity, scoped write metadata, and provider record redaction.

One place defines how a :class:`~gnosis.models.MemoryScope` becomes SDK
identifiers and stored metadata: the composite ``user_identifier`` key, the
scope metadata stamped on every write, the scope-narrowed filters for
provider searches, and the redacted views of entity/fact/preference records
returned to callers. Every read re-check and every write path must agree on
these shapes, so they live together.
"""

from typing import Final

from pydantic import BaseModel

from gnosis.json_redaction import (
    json_object,
    redacted_object,
    redacted_optional_text,
    redacted_text,
    validated_json_object,
)
from gnosis.models import (
    EntityRecord,
    FactRecord,
    JsonObject,
    MemoryScope,
    PreferenceRecord,
)

SCOPE_METADATA_KEYS: Final[frozenset[str]] = frozenset(
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


def session_id(scope: MemoryScope) -> str:
    return scope.session_id


def user_identifier(scope: MemoryScope) -> str:
    return (
        f"{scope.tenant_id}:{scope.space_id}:{scope.visibility.value}:"
        f"{scope.agent_id}:{scope.user_id}"
    )


def scope_metadata(scope: MemoryScope) -> dict[str, str]:
    metadata = {
        "tenant_id": scope.tenant_id,
        "space_id": scope.space_id,
        "agent_id": scope.agent_id,
        "session_id": scope.session_id,
        "user_id": scope.user_id,
        "visibility": scope.visibility.value,
    }
    if scope.guild_id is not None:
        metadata["guild_id"] = scope.guild_id
    if scope.channel_id is not None:
        metadata["channel_id"] = scope.channel_id
    return metadata


def scope_json_metadata(scope: MemoryScope) -> JsonObject:
    return validated_json_object(scope_metadata(scope))


def scoped_filters(scope: MemoryScope, metadata: JsonObject) -> JsonObject:
    return validated_json_object(metadata | scope_metadata(scope))


def write_metadata(
    scope: MemoryScope,
    metadata: JsonObject,
    provenance: object | None,
) -> JsonObject:
    base = metadata | scope_metadata(scope)
    if isinstance(provenance, BaseModel):
        base |= json_object(provenance.model_dump(exclude_none=True))
    return redacted_object(validated_json_object(base))


def reasoning_write_metadata(scope: MemoryScope, metadata: JsonObject) -> JsonObject:
    return redacted_object(
        validated_json_object(metadata | scope_metadata(scope)),
    )


def record_matches_filters(metadata: JsonObject, filters: JsonObject) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


def memory_edit_audit(memory_id: str, scope: MemoryScope) -> dict[str, str]:
    return {
        "memory_id": memory_id,
        "tenant_id": scope.tenant_id,
        "agent_id": scope.agent_id,
        "user_id": scope.user_id,
    }


def redacted_entity(record: EntityRecord) -> EntityRecord:
    return record.model_copy(
        update={
            "description": redacted_optional_text(record.description),
            "attributes": redacted_object(record.attributes),
            "metadata": redacted_object(record.metadata),
        },
    )


def redacted_fact(record: FactRecord) -> FactRecord:
    return record.model_copy(
        update={
            "subject": redacted_text(record.subject),
            "predicate": redacted_text(record.predicate),
            "object": redacted_text(record.object),
            "metadata": redacted_object(record.metadata),
        },
    )


def redacted_preference(record: PreferenceRecord) -> PreferenceRecord:
    return record.model_copy(
        update={
            "preference": redacted_text(record.preference),
            "context": redacted_optional_text(record.context),
            "metadata": redacted_object(record.metadata),
        },
    )
