from typing import TYPE_CHECKING

from gnosis.redaction import redact_secrets

if TYPE_CHECKING:
    from gnosis.models import JsonObject


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
