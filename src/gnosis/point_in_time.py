"""Point-in-time reads: what the store knew as of a moment.

A read with ``as_of`` sees a memory only if it was *observed* at or before
that moment. A memory's observation time is its caller-supplied
``metadata.observed_at`` (ISO-8601 with an offset) when present - a client
that back-loads history writes years after the fact and says when each record
was first seen - else its ``created_at`` (when gnosis wrote it). A memory
whose observation time cannot be read is invisible to an as-of read: an
unknown time fails closed.

Everything in the read path that could let a later record shape an earlier
answer is neutralized for an as-of read (``backend.py``):

- every candidate leg (dense, BM25, recency fallback) is filtered by this
  predicate before fusion, supersession and the item budget, so read-time
  newest-wins only ever compares records visible as of then;
- the write-time structural supersession filter (``valid_to``) is not
  applied, because ``valid_to`` is stamped when a *later* fact is written;
- recency injection, graph-QA fusion, entity and bridge traversal and
  community summaries are skipped: they walk edges or summaries built from
  records that may postdate ``as_of``, and filtering their output after the
  walk cannot undo a path chosen through a later edge;
- facts-to-verbatim expansion only renders source turns visible as of then.

The in-Cypher ``$as_of_epoch`` clause (``memory_provider.py``) only narrows
the candidate pool so old moments still fill a budget; this module's
predicate, applied in the gateway, is the authority.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Final

from gnosis.json_redaction import metadata_from_json
from gnosis.memory_provider import StoredMemory
from gnosis.models import JsonObject, JsonValue, MemoryRecord

OBSERVED_AT_KEY: Final[str] = "observed_at"


def parse_as_of(value: str | None) -> datetime | None:
    """Parse an ``as_of`` request value; it must carry a UTC offset.

    Raises ``ValueError`` on a naive or malformed timestamp: a point-in-time
    read with an ambiguous moment is refused rather than guessed.
    """
    if value is None:
        return None
    parsed = _parse_aware(value)
    if parsed is None:
        message = f"as_of must be an ISO-8601 timestamp with an offset: {value!r}"
        raise ValueError(message)
    return parsed


def observed_time(
    metadata: Mapping[str, JsonValue],
    created_at: str | None,
) -> datetime | None:
    """A record's observation time: ``metadata.observed_at``, else created_at."""
    stamped = metadata.get(OBSERVED_AT_KEY)
    if isinstance(stamped, str) and stamped:
        return _parse_aware(stamped)
    if created_at:
        return _parse_aware(created_at)
    return None


def observed_epoch(metadata: Mapping[str, JsonValue]) -> float | None:
    """``metadata.observed_at`` as Unix seconds, for the node property."""
    stamped = metadata.get(OBSERVED_AT_KEY)
    if not isinstance(stamped, str) or not stamped:
        return None
    parsed = _parse_aware(stamped)
    return parsed.timestamp() if parsed is not None else None


def visible_as_of(
    metadata: Mapping[str, JsonValue],
    created_at: str | None,
    as_of: datetime | None,
) -> bool:
    """Whether a record may be seen by a read as of ``as_of`` (None: always)."""
    if as_of is None:
        return True
    observed = observed_time(metadata, created_at)
    return observed is not None and observed <= as_of


def memory_visible_as_of(memory: StoredMemory, as_of: datetime | None) -> bool:
    """``visible_as_of`` for a provider-surface stored memory."""
    return visible_as_of(memory.metadata, memory.created_at, as_of)


def record_visible_as_of(record: MemoryRecord, as_of: datetime | None) -> bool:
    """``visible_as_of`` for a public record (federated search merges)."""
    return visible_as_of(record.metadata, record.created_at, as_of)


def fact_visible_as_of(fact: JsonObject, as_of: datetime | None) -> bool:
    """``visible_as_of`` for a context-assembly fact dict."""
    if as_of is None:
        return True
    metadata = fact.get("metadata")
    parsed: Mapping[str, JsonValue]
    if isinstance(metadata, dict):
        parsed = metadata
    elif isinstance(metadata, str):
        parsed = metadata_from_json(metadata)
    else:
        parsed = {}
    created_at = fact.get("created_at")
    return visible_as_of(
        parsed,
        created_at if isinstance(created_at, str) else None,
        as_of,
    )


def as_of_epoch(as_of: datetime | None) -> float | None:
    return as_of.timestamp() if as_of is not None else None


def _parse_aware(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed
