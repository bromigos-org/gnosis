from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gnosis.models import JsonValue

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
# Identifiers are opaque by construction: scope fields (``user_id``,
# ``session_id``, ...), memory and source ids, a caller's join ids. The
# opaque-value pattern takes any 24+ character mix of letters and digits for
# a credential, and an identifier rewritten to ``[REDACTED]`` makes its record
# unreachable by the very scope or filter that wrote it. Values under an
# identifier key therefore skip that one heuristic; the explicit credential
# shapes (Bearer, ``sk-``, Discord tokens, ``KEY=value``) and the sensitive
# key names still apply, and free text is redacted as before. ``*_key`` names
# are deliberately not identifiers: api, access, private and signing keys are
# named that way.
_IDENTIFIER_KEY_NAMES = frozenset({"id", "ids", "uuid", "uuids", "visibility"})
_IDENTIFIER_KEY_SUFFIXES = ("_id", "_ids", "_uuid", "_uuids")


def redact_secrets(value: JsonValue) -> JsonValue:
    match value:
        case str():
            return _redact_text(value)
        case bool() | int() | float() | None:
            return value
        case list():
            return [redact_secrets(item) for item in value]
        case dict():
            return {key: _redact_member(key, item) for key, item in value.items()}


def is_identifier_key(key: object) -> bool:
    """Whether a member's name marks its value as an identifier.

    ``id``, ``ids``, ``uuid``, ``*_id``, ``*_ids``, ``*_uuid`` (camelCase and
    kebab-case alike) and ``visibility``; never a sensitive name such as
    ``token_id`` or ``secret_id``, which stays fully redacted.
    """
    if not isinstance(key, str) or _is_sensitive_key(key):
        return False
    normalized = _normalized_key(key)
    return normalized in _IDENTIFIER_KEY_NAMES or normalized.endswith(
        _IDENTIFIER_KEY_SUFFIXES,
    )


def _redact_member(key: str, value: JsonValue) -> JsonValue:
    if _is_sensitive_key(key):
        return _REDACTED
    if is_identifier_key(key):
        return _redact_identifier(value)
    return redact_secrets(value)


def _redact_identifier(value: JsonValue) -> JsonValue:
    match value:
        case str():
            return _redact_text(value, opaque=False)
        case list():
            return [_redact_identifier(item) for item in value]
        case _:
            return redact_secrets(value)


def _redact_text(value: str, *, opaque: bool = True) -> str:
    assignment = _redact_assignment(value)
    if assignment is not None:
        return assignment
    patterns = [_BEARER_PATTERN, _DISCORD_TOKEN_PATTERN, _SK_PATTERN]
    if opaque:
        patterns.append(_OPAQUE_VALUE_PATTERN)
    if any(_is_full_match(value, pattern) for pattern in patterns):
        return _REDACTED
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}={_REDACTED}",
        value,
    )
    for pattern in patterns:
        redacted = pattern.sub(_REDACTED, redacted)
    return redacted


def _redact_assignment(value: str) -> str | None:
    if match := _SECRET_ASSIGNMENT_PATTERN.fullmatch(value.strip()):
        return f"{match.group(1)}={_REDACTED}"
    return None


def _is_full_match(value: str, pattern: re.Pattern[str]) -> bool:
    return bool(pattern.fullmatch(value.strip()))


def _normalized_key(key: str) -> str:
    normalized = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        key,
    )
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def _is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = _normalized_key(key)
    return (
        any(part in _SENSITIVE_KEY_NAMES for part in normalized.split("_") if part)
        or normalized in _SENSITIVE_KEY_NAMES
    )
