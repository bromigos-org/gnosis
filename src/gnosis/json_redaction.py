"""Shared JSON-object coercion and redaction primitives.

Small typed wrappers over :func:`gnosis.redaction.redact_secrets` and one
shared ``TypeAdapter`` for the ``JsonObject`` contract type. Every backend
seam (context assembly, operator flows, reasoning views, SDK payload
conversion) renders untrusted or SDK-shaped values through these helpers so
redaction and JSON coercion behave identically everywhere.
"""

import base64
import hashlib
import json
from typing import Final

from pydantic import TypeAdapter, ValidationError

from gnosis.models import JsonObject, JsonValue
from gnosis.redaction import redact_secrets

JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)


def json_object(value: object) -> JsonObject:
    """Coerce any value to a ``JsonObject``, degrading to empty on mismatch."""
    try:
        return JSON_OBJECT_ADAPTER.validate_python(value)
    except ValidationError:
        return {}


def validated_json_object(value: object) -> JsonObject:
    """Validate to a ``JsonObject``, raising ``ValidationError`` on mismatch."""
    return JSON_OBJECT_ADAPTER.validate_python(value)


def json_compatible_object(value: object) -> JsonObject:
    """Coerce arbitrary objects through a JSON round-trip when needed."""
    parsed = json_object(value)
    if parsed:
        return parsed
    try:
        return JSON_OBJECT_ADAPTER.validate_json(json.dumps(value, default=str))
    except (TypeError, ValidationError):
        return {}


def canonical_json(value: JsonObject) -> str:
    """Render a deterministic JSON encoding for hashing and signing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def hash_json(value: JsonObject) -> str:
    """SHA-256 hex digest of the canonical JSON encoding."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def urlsafe_b64encode(value: bytes) -> str:
    """URL-safe base64 without padding, for compact token payloads."""
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def urlsafe_b64decode(value: str) -> bytes:
    """Decode URL-safe base64 produced by :func:`urlsafe_b64encode`."""
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def redacted_text(value: str) -> str:
    """Redact secrets in one string, preserving the ``str`` type."""
    redacted = redact_secrets(value)
    if isinstance(redacted, str):
        return redacted
    return value


def redacted_optional_text(value: str | None) -> str | None:
    """Redact secrets in an optional string."""
    if value is None:
        return None
    return redacted_text(value)


def redacted_object(value: JsonObject) -> JsonObject:
    """Redact secrets in a JSON object, preserving the object type."""
    redacted = redact_secrets(value)
    if isinstance(redacted, dict):
        return redacted
    return {}


def string_metadata(metadata: dict[str, JsonValue]) -> dict[str, str]:
    """Keep only the non-empty string members of a metadata object."""
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, str) and value != ""
    }


def metadata_from_json(metadata: str) -> dict[str, str]:
    """Parse serialized metadata to its string members, empty on bad JSON."""
    try:
        parsed = JSON_OBJECT_ADAPTER.validate_json(metadata)
    except ValidationError:
        return {}
    return string_metadata(parsed)
