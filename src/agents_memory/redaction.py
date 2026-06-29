from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents_memory.models import JsonValue

_REDACTED = "[REDACTED]"
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_KEY|APIKEY|PASSWORD|SECRET|TOKEN)[A-Z0-9_]*)=(['\"]?)([^\s'\",;]+)\2"
)
_DISCORD_TOKEN_PATTERN = re.compile(
    r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b",
)
_SK_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_OPAQUE_VALUE_PATTERN = re.compile(
    r"\b(?=[A-Za-z0-9+/_=-]{24,}\b)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9+/_=-]+\b",
)
_SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}


def redact_secrets(value: JsonValue) -> JsonValue:
    match value:
        case str():
            return _redact_text(value)
        case bool() | int() | float() | None:
            return value
        case list():
            return [redact_secrets(item) for item in value]
        case dict():
            return {
                key: _REDACTED if _is_sensitive_key(key) else redact_secrets(item)
                for key, item in value.items()
            }


def _redact_text(value: str) -> str:
    assignment = _redact_assignment(value)
    if assignment is not None:
        return assignment
    if _is_full_match(value, _BEARER_PATTERN):
        return _REDACTED
    if _is_full_match(value, _DISCORD_TOKEN_PATTERN):
        return _REDACTED
    if _is_full_match(value, _SK_PATTERN):
        return _REDACTED
    if _is_full_match(value, _OPAQUE_VALUE_PATTERN):
        return _REDACTED
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}={_REDACTED}",
        value,
    )
    redacted = _BEARER_PATTERN.sub(_REDACTED, redacted)
    redacted = _DISCORD_TOKEN_PATTERN.sub(_REDACTED, redacted)
    redacted = _SK_PATTERN.sub(_REDACTED, redacted)
    return _OPAQUE_VALUE_PATTERN.sub(_REDACTED, redacted)


def _redact_assignment(value: str) -> str | None:
    if match := _SECRET_ASSIGNMENT_PATTERN.fullmatch(value.strip()):
        return f"{match.group(1)}={_REDACTED}"
    return None


def _is_full_match(value: str, pattern: re.Pattern[str]) -> bool:
    return bool(pattern.fullmatch(value.strip()))


def _is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        key,
    )
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    return (
        any(part in _SENSITIVE_KEY_NAMES for part in normalized.split("_") if part)
        or normalized in _SENSITIVE_KEY_NAMES
    )
