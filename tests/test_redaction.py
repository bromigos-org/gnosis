from typing import TYPE_CHECKING

from gnosis.models import MemoryScope, MemoryVisibility
from gnosis.redaction import is_identifier_key, redact_secrets
from gnosis.scope_policy import SCOPE_METADATA_KEYS, write_metadata

if TYPE_CHECKING:
    from gnosis.models import JsonObject

# 28 characters mixing letters and digits: the opaque-value pattern's shape.
_DIGIT_RUN_ID = "arbiter-roundtrip-1790524864"
_OPAQUE = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4"


def test_redacts_nested_tool_payload_secrets() -> None:
    payload: JsonObject = {
        "query": "hello",
        "headers": {
            "Authorization": "alpha",
            "X-Trace": "trace-42",
        },
        "environment": [
            {"password": "bravo"},
            {"token": "charlie"},
        ],
        "note": "visible-note",
    }

    redacted = redact_secrets(payload)

    assert redacted == {
        "query": "hello",
        "headers": {
            "Authorization": "[REDACTED]",
            "X-Trace": "trace-42",
        },
        "environment": [
            {"password": "[REDACTED]"},
            {"token": "[REDACTED]"},
        ],
        "note": "visible-note",
    }


def test_redaction_never_returns_original_secret() -> None:
    candidate = "delta"
    payload: JsonObject = {
        "payload": {"authorization": candidate, "token": "echo"},
    }

    redacted = redact_secrets(payload)

    assert candidate not in repr(redacted)
    assert "echo" not in repr(redacted)
    assert redacted == {
        "payload": {
            "authorization": "[REDACTED]",
            "token": "[REDACTED]",
        },
    }


def test_redacts_sensitive_key_name_variants() -> None:
    payload: JsonObject = {
        "apiKey": "foxtrot",
        "refresh-token": "golf",
        "client_secret": "hotel",
    }

    redacted = redact_secrets(payload)

    assert redacted == {
        "apiKey": "[REDACTED]",
        "refresh-token": "[REDACTED]",
        "client_secret": "[REDACTED]",
    }


# ── identifiers ───────────────────────────────────────────────────────────
#
# An identifier rewritten to "[REDACTED]" makes its record unreachable by the
# scope or filter that wrote it: scope checks and metadata filters compare the
# stored value with the caller's. Identifiers skip the opaque-value heuristic
# only; content, sensitive names and explicit credential shapes do not.


def test_opaque_free_text_is_still_redacted() -> None:
    assert redact_secrets(_DIGIT_RUN_ID) == "[REDACTED]"
    assert redact_secrets(f"see {_OPAQUE} here") == "see [REDACTED] here"


def test_scope_fields_keep_their_values() -> None:
    scope: JsonObject = {
        "tenant_id": "bromigos",
        "space_id": "arbiter-signals-1790524864-test",
        "agent_id": "arbiter-news-agent-1790524864",
        "session_id": "session-2026-09-27-1790524864",
        "user_id": _DIGIT_RUN_ID,
        "visibility": "agent_shared",
        "guild_id": "guild-1790524864-1790524864",
        "channel_id": "channel-1790524864-1790524864",
    }

    assert redact_secrets(scope) == scope


def test_every_scope_metadata_key_is_an_identifier() -> None:
    assert all(is_identifier_key(key) for key in SCOPE_METADATA_KEYS)


def test_identifier_members_keep_their_values() -> None:
    payload: JsonObject = {
        "id": "00000000-0000-0000-0000-00000000e001",
        "item_id": "okx-announcements-1790524864-listing",
        "itemId": "binance-listings-BTCUSDT-1790524864",
        "source-memory-ids": [
            "00000000-0000-0000-0000-00000000e001",
            "00000000-0000-0000-0000-00000000e002",
        ],
        "record_uuid": "0f8fad5b-d9cb-469f-a165-70867728950e",
    }

    assert redact_secrets(payload) == payload


def test_other_members_are_still_redacted() -> None:
    payload: JsonObject = {
        "note": _OPAQUE,
        "content": f"ref {_OPAQUE}",
        "item_id": _OPAQUE,
        "nested": {"body": _OPAQUE, "run_id": _OPAQUE},
    }

    assert redact_secrets(payload) == {
        "note": "[REDACTED]",
        "content": "ref [REDACTED]",
        "item_id": _OPAQUE,
        "nested": {"body": "[REDACTED]", "run_id": _OPAQUE},
    }


def test_identifiers_still_lose_explicit_credentials() -> None:
    payload: JsonObject = {
        "user_id": "Bearer abcdefghijklmnop",
        "agent_id": "sk-abcdefghijklmnopqrstu",
        "run_id": "API_KEY=abc123",
        "session_ids": ["ok-1", "sk-abcdefghijklmnopqrstu"],
    }

    assert redact_secrets(payload) == {
        "user_id": "[REDACTED]",
        "agent_id": "[REDACTED]",
        "run_id": "API_KEY=[REDACTED]",
        "session_ids": ["ok-1", "[REDACTED]"],
    }


def test_sensitive_names_win_over_identifier_suffixes() -> None:
    payload: JsonObject = {"token_id": "t-1", "secretId": "s-1", "auth_id": "a-1"}

    assert redact_secrets(payload) == {
        "token_id": "[REDACTED]",
        "secretId": "[REDACTED]",
        "auth_id": "[REDACTED]",
    }


def test_key_names_are_not_identifiers() -> None:
    access = "a1" * 13  # opaque-shaped (24+ letters and digits), not a credential
    payload: JsonObject = {"access_key": access, "signing_key": access}

    assert redact_secrets(payload) == {
        "access_key": "[REDACTED]",
        "signing_key": "[REDACTED]",
    }


def test_member_names_are_never_rewritten() -> None:
    name = "k1790524864abcdefghijklmnop"

    redacted = redact_secrets({name: "v", "nested": {name: "w"}})

    assert redacted == {name: "v", "nested": {name: "w"}}


def test_write_metadata_keeps_the_scope_it_is_read_back_by() -> None:
    scope = MemoryScope(
        tenant_id="bromigos",
        space_id="arbiter-signals",
        agent_id="arbiter-news",
        session_id="events",
        user_id=_DIGIT_RUN_ID,
        visibility=MemoryVisibility.AGENT_SHARED,
    )

    stored = write_metadata(
        scope,
        {"item_id": "okx-announcements-1790524864-listing", "note": _OPAQUE},
        None,
    )

    assert stored["user_id"] == _DIGIT_RUN_ID
    assert stored["item_id"] == "okx-announcements-1790524864-listing"
    assert stored["note"] == "[REDACTED]"
