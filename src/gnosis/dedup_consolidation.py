"""Operator dedup and consolidation flows: tokens, staleness, idempotency.

The `/v1/memories/dedup/*` and `/v1/memories/consolidation/*` operator
surfaces follow the same two-phase contract: a dry-run returns a report
plus an HMAC-signed, TTL-bounded token binding the exact reviewed state;
the apply call must present that token, matching candidate/report state,
and an idempotency key. This module holds the SDK capability protocols for
those surfaces, the in-memory state records, and every pure helper (token
mint/verify, fingerprints, snapshot hashes, report sanitization, and the
SDK operation dispatch). The request orchestration stays on
:class:`gnosis.backend.Neo4jAgentMemoryBackend`.
"""

import binascii
import hashlib
import hmac
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol, assert_never, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ValidationError

from gnosis.backend_protocols import BackendRequestError
from gnosis.json_redaction import (
    JSON_OBJECT_ADAPTER,
    canonical_json,
    hash_json,
    json_compatible_object,
    json_object,
    redacted_object,
    redacted_optional_text,
    urlsafe_b64decode,
    urlsafe_b64encode,
)
from gnosis.models import (
    ConsolidationApplyRequest,
    ConsolidationApplyResponse,
    ConsolidationDryRunRequest,
    ConsolidationOperationName,
    DedupApplyRequest,
    DedupApplyResponse,
    DedupCandidate,
    DedupEntitySnapshot,
    DedupOperationName,
    JsonObject,
    JsonValue,
    MemoryScope,
)
from gnosis.settings import Settings

DEDUP_UNAVAILABLE_DETAIL: Final[str] = "SDK deduplication is unavailable."
DEDUP_APPLY_REQUIRED_DETAIL: Final[str] = (
    "Deduplication apply requests require apply=true."
)
_DEDUP_TOKEN_DETAIL: Final[str] = "Deduplication dry-run token is invalid or expired."  # noqa: S105
_DEDUP_STALE_DETAIL: Final[str] = "Deduplication candidate is stale."
DEDUP_IDEMPOTENCY_DETAIL: Final[str] = (
    "Idempotency key was already used for a different deduplication request."
)
DEDUP_TOKEN_TTL: Final[timedelta] = timedelta(minutes=15)
_DEDUP_TOKEN_PARTS: Final[int] = 2
_DEDUP_PENDING_TOKEN: Final[str] = "pending"  # noqa: S105
CONSOLIDATION_UNAVAILABLE_DETAIL: Final[str] = "SDK consolidation is unavailable."
CONSOLIDATION_APPLY_REQUIRED_DETAIL: Final[str] = (
    "Consolidation apply requests require apply=true."
)
_CONSOLIDATION_TOKEN_DETAIL: Final[str] = (
    "Consolidation dry-run token is invalid or expired."  # noqa: S105
)
_CONSOLIDATION_STALE_DETAIL: Final[str] = "Consolidation dry-run report is stale."
CONSOLIDATION_IDEMPOTENCY_DETAIL: Final[str] = (
    "Idempotency key was already used for a different consolidation request."
)
CONSOLIDATION_TOKEN_TTL: Final[timedelta] = timedelta(minutes=15)
UNSAFE_CONSOLIDATION_REPORT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "chain_of_thought",
        "credentials",
        "cypher",
        "embedding",
        "embeddings",
        "password",
        "prompt",
        "raw_prompt",
        "rustfs_credentials",
        "secret",
        "thought",
        "token",
    },
)


@runtime_checkable
class DedupCapableLongTermMemory(Protocol):
    def get_deduplication_stats(self) -> Awaitable[object]: ...
    def find_potential_duplicates(
        self,
        *,
        limit: int = 100,
    ) -> Awaitable[list[tuple[object, object, float]]]: ...
    def review_duplicate(
        self,
        source_id: UUID,
        target_id: UUID,
        *,
        confirm: bool,
    ) -> Awaitable[bool]: ...
    def merge_duplicate_entities(
        self,
        source_id: UUID,
        target_id: UUID,
    ) -> Awaitable[tuple[object, object] | None]: ...


@runtime_checkable
class ConsolidationMemory(Protocol):
    def archive_expired_conversations(
        self,
        *,
        ttl_days: int | None = None,
        dry_run: bool = True,
    ) -> Awaitable[object]: ...

    def dedupe_entities(
        self,
        *,
        similarity_threshold: float = 0.95,
        max_pairs: int = 10000,
        dry_run: bool = True,
    ) -> Awaitable[object]: ...

    def detect_superseded_preferences(
        self,
        *,
        user_identifier: str | None = None,
        similarity_threshold: float = 0.92,
        dry_run: bool = True,
    ) -> Awaitable[object]: ...

    def summarize_long_traces(
        self,
        *,
        min_steps: int = 20,
        max_traces: int = 1000,
        dry_run: bool = True,
    ) -> Awaitable[object]: ...


@runtime_checkable
class ConsolidationCapableMemoryClient(Protocol):
    @property
    def consolidation(self) -> ConsolidationMemory: ...


@dataclass(frozen=True, slots=True)
class DedupCandidateState:
    candidate_id: str
    version: int
    scope: MemoryScope
    source_id: UUID
    target_id: UUID
    graph_snapshot_hash: str


@dataclass(frozen=True, slots=True)
class DedupTokenClaims:
    scope: MemoryScope
    candidate_id: str
    candidate_version: int
    graph_snapshot_hash: str
    operation: DedupOperationName
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class DedupIdempotencyRecord:
    request_fingerprint: str
    response: DedupApplyResponse


@dataclass(frozen=True, slots=True)
class ConsolidationDryRunState:
    scope: MemoryScope
    operation: ConsolidationOperationName
    graph_snapshot_hash: str
    request_fingerprint: str


@dataclass(frozen=True, slots=True)
class ConsolidationTokenClaims:
    scope: MemoryScope
    operation: ConsolidationOperationName
    graph_snapshot_hash: str
    request_fingerprint: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ConsolidationIdempotencyRecord:
    request_fingerprint: str
    response: ConsolidationApplyResponse


def dedup_stats_payload(stats: object) -> JsonObject:
    if isinstance(stats, BaseModel):
        return json_object(stats.model_dump(mode="json"))
    try:
        return json_compatible_object(vars(stats))
    except TypeError:
        return json_compatible_object(stats)


def dedup_candidate(
    source: object,
    target: object,
    similarity: float,
) -> DedupCandidate:
    source_snapshot = _dedup_entity_snapshot(source)
    target_snapshot = _dedup_entity_snapshot(target)
    fingerprint = hash_json(
        {
            "source_id": source_snapshot.id,
            "target_id": target_snapshot.id,
            "similarity": similarity,
        },
    )
    return DedupCandidate(
        candidate_id=f"dedup-{fingerprint[:24]}",
        version=1,
        source=source_snapshot,
        target=target_snapshot,
        similarity=similarity,
        reject_dry_run_token=_DEDUP_PENDING_TOKEN,
        merge_dry_run_token=_DEDUP_PENDING_TOKEN,
    )


def _dedup_entity_snapshot(entity: object) -> DedupEntitySnapshot:
    record = _dedup_entity_payload(entity)
    return DedupEntitySnapshot(
        id=_required_text(record, "id"),
        name=_required_text(record, "name"),
        type=_required_text(record, "type"),
        subtype=_optional_text(record, "subtype"),
        description=redacted_optional_text(_optional_text(record, "description")),
        confidence=_optional_float(record, "confidence", 1.0),
        aliases=_string_list(record.get("aliases")),
        attributes=redacted_object(_json_member_object(record, "attributes")),
        metadata=redacted_object(_json_member_object(record, "metadata")),
    )


def _dedup_entity_payload(entity: object) -> JsonObject:
    if isinstance(entity, BaseModel):
        return json_object(entity.model_dump(mode="json"))
    try:
        return json_compatible_object(vars(entity))
    except TypeError:
        return json_compatible_object(entity)


def dedup_snapshot_hash(
    scope: MemoryScope,
    candidates: list[DedupCandidate],
) -> str:
    return hash_json(
        {
            "scope": json_object(scope.model_dump(mode="json")),
            "candidates": [
                candidate.model_dump(
                    mode="json",
                    exclude={"reject_dry_run_token", "merge_dry_run_token"},
                )
                for candidate in candidates
            ],
        },
    )


def dedup_token(
    settings: Settings,
    claims: DedupTokenClaims,
) -> str:
    payload = json_object(
        {
            "scope": claims.scope.model_dump(mode="json"),
            "candidate_id": claims.candidate_id,
            "candidate_version": claims.candidate_version,
            "graph_snapshot_hash": claims.graph_snapshot_hash,
            "operation": claims.operation,
            "expires_at": claims.expires_at.isoformat(),
        },
    )
    payload_bytes = canonical_json(payload).encode()
    encoded_payload = urlsafe_b64encode(payload_bytes)
    signature = hmac.new(
        settings.gnosis_admin_operator_token.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded_payload}.{signature}"


def require_current_dedup_candidate(
    request: DedupApplyRequest,
    state: DedupCandidateState | None,
) -> DedupCandidateState:
    if state is None:
        raise BackendRequestError(_DEDUP_STALE_DETAIL)
    if state.version != request.candidate_version:
        raise BackendRequestError(_DEDUP_STALE_DETAIL)
    if state.scope != request.scope:
        raise BackendRequestError(_DEDUP_STALE_DETAIL)
    if state.graph_snapshot_hash != request.graph_snapshot_hash:
        raise BackendRequestError(_DEDUP_STALE_DETAIL)
    return state


def require_dedup_token(settings: Settings, request: DedupApplyRequest) -> None:
    payload = _dedup_token_payload(settings, request.dry_run_token)
    expected = json_object(
        {
            "scope": request.scope.model_dump(mode="json"),
            "candidate_id": request.candidate_id,
            "candidate_version": request.candidate_version,
            "graph_snapshot_hash": request.graph_snapshot_hash,
            "operation": request.operation,
        },
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BackendRequestError(_DEDUP_TOKEN_DETAIL)
    expires_at = _optional_text(payload, "expires_at")
    if expires_at is None:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL)
    try:
        expiry = datetime.fromisoformat(expires_at)
    except (ValueError, binascii.Error) as error:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL) from error
    if expiry <= datetime.now(UTC):
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL)


def _dedup_token_payload(settings: Settings, token: str) -> JsonObject:
    parts = token.split(".", maxsplit=1)
    if len(parts) != _DEDUP_TOKEN_PARTS:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL)
    encoded_payload, signature = parts
    try:
        payload_bytes = urlsafe_b64decode(encoded_payload)
    except ValueError as error:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL) from error
    expected_signature = hmac.new(
        settings.gnosis_admin_operator_token.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL)
    try:
        return JSON_OBJECT_ADAPTER.validate_json(payload_bytes)
    except ValidationError as error:
        raise BackendRequestError(_DEDUP_TOKEN_DETAIL) from error


async def run_consolidation_operation(
    consolidation: ConsolidationMemory,
    request: ConsolidationDryRunRequest | ConsolidationApplyRequest,
    *,
    dry_run: bool,
) -> object:
    match request.operation:
        case "archive_expired_conversations":
            return await consolidation.archive_expired_conversations(
                ttl_days=request.ttl_days,
                dry_run=dry_run,
            )
        case "dedupe_entities":
            return await consolidation.dedupe_entities(
                similarity_threshold=request.similarity_threshold or 0.95,
                max_pairs=request.max_pairs or 10000,
                dry_run=dry_run,
            )
        case "detect_superseded_preferences":
            return await consolidation.detect_superseded_preferences(
                user_identifier=request.user_identifier,
                similarity_threshold=request.similarity_threshold or 0.92,
                dry_run=dry_run,
            )
        case "summarize_long_traces":
            return await consolidation.summarize_long_traces(
                min_steps=request.min_steps or 20,
                max_traces=request.max_traces or 1000,
                dry_run=dry_run,
            )
    assert_never(request.operation)


def safe_consolidation_report(report: object) -> JsonObject:
    if isinstance(report, BaseModel):
        payload = json_object(report.model_dump(mode="json"))
    else:
        try:
            payload = json_compatible_object(vars(report))
        except TypeError:
            payload = json_compatible_object(report)
    return _strip_unsafe_consolidation_fields(redacted_object(payload))


def _strip_unsafe_consolidation_fields(value: JsonObject) -> JsonObject:
    safe: JsonObject = {}
    for key, item in value.items():
        normalized = key.casefold()
        if normalized in UNSAFE_CONSOLIDATION_REPORT_KEYS:
            continue
        safe[key] = _strip_unsafe_consolidation_value(item)
    return safe


def _strip_unsafe_consolidation_value(value: JsonValue) -> JsonValue:
    match value:
        case dict():
            return _strip_unsafe_consolidation_fields(json_object(value))
        case list():
            return [_strip_unsafe_consolidation_value(item) for item in value]
        case _:
            return value


def consolidation_request_fingerprint(
    request: ConsolidationDryRunRequest | ConsolidationApplyRequest,
) -> str:
    return hash_json(
        json_object(
            request.model_dump(
                mode="json",
                exclude={
                    "audit",
                    "apply",
                    "dry_run_token",
                    "graph_snapshot_hash",
                    "idempotency_key",
                },
            ),
        ),
    )


def consolidation_apply_fingerprint(request: ConsolidationApplyRequest) -> str:
    return hash_json(
        json_object(
            request.model_dump(mode="json", exclude={"idempotency_key"}),
        ),
    )


def consolidation_token(settings: Settings, claims: ConsolidationTokenClaims) -> str:
    payload = json_object(
        {
            "scope": claims.scope.model_dump(mode="json"),
            "operation": claims.operation,
            "graph_snapshot_hash": claims.graph_snapshot_hash,
            "request_fingerprint": claims.request_fingerprint,
            "expires_at": claims.expires_at.isoformat(),
        },
    )
    payload_bytes = canonical_json(payload).encode()
    encoded_payload = urlsafe_b64encode(payload_bytes)
    signature = hmac.new(
        settings.gnosis_admin_operator_token.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded_payload}.{signature}"


def require_current_consolidation_dry_run(
    request: ConsolidationApplyRequest,
    state: ConsolidationDryRunState | None,
) -> ConsolidationDryRunState:
    if state is None:
        raise BackendRequestError(_CONSOLIDATION_STALE_DETAIL)
    if state.scope != request.scope:
        raise BackendRequestError(_CONSOLIDATION_STALE_DETAIL)
    if state.operation != request.operation:
        raise BackendRequestError(_CONSOLIDATION_STALE_DETAIL)
    if state.request_fingerprint != consolidation_request_fingerprint(request):
        raise BackendRequestError(_CONSOLIDATION_STALE_DETAIL)
    return state


def require_consolidation_token(
    settings: Settings,
    request: ConsolidationApplyRequest,
    state: ConsolidationDryRunState,
) -> None:
    payload = _consolidation_token_payload(settings, request.dry_run_token)
    expected = json_object(
        {
            "scope": request.scope.model_dump(mode="json"),
            "operation": request.operation,
            "graph_snapshot_hash": request.graph_snapshot_hash,
            "request_fingerprint": state.request_fingerprint,
        },
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)
    expires_at = _optional_text(payload, "expires_at")
    if expires_at is None:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)
    try:
        expiry = datetime.fromisoformat(expires_at)
    except (ValueError, binascii.Error) as error:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL) from error
    if expiry <= datetime.now(UTC):
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)


def _consolidation_token_payload(settings: Settings, token: str) -> JsonObject:
    parts = token.split(".", maxsplit=1)
    if len(parts) != _DEDUP_TOKEN_PARTS:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)
    encoded_payload, signature = parts
    try:
        payload_bytes = urlsafe_b64decode(encoded_payload)
    except ValueError as error:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL) from error
    expected_signature = hmac.new(
        settings.gnosis_admin_operator_token.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL)
    try:
        return JSON_OBJECT_ADAPTER.validate_json(payload_bytes)
    except ValidationError as error:
        raise BackendRequestError(_CONSOLIDATION_TOKEN_DETAIL) from error


async def apply_dedup_operation(
    long_term: DedupCapableLongTermMemory,
    request: DedupApplyRequest,
    state: DedupCandidateState,
) -> JsonObject:
    match request.operation:
        case "reject":
            rejected = await long_term.review_duplicate(
                state.source_id,
                state.target_id,
                confirm=False,
            )
            return {"rejected": rejected}
        case "merge":
            merged = await long_term.merge_duplicate_entities(
                state.source_id,
                state.target_id,
            )
            return {"merged": merged is not None}


def _required_text(record: JsonObject, field_name: str) -> str:
    value = record.get(field_name)
    if isinstance(value, str) and value:
        return value
    raise BackendRequestError(DEDUP_UNAVAILABLE_DETAIL)


def _optional_text(record: JsonObject, field_name: str) -> str | None:
    value = record.get(field_name)
    if isinstance(value, str) and value:
        return value
    return None


def _optional_float(record: JsonObject, field_name: str, default: float) -> float:
    value = record.get(field_name)
    if isinstance(value, int | float):
        return float(value)
    return default


def _string_list(value: JsonValue | None) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _json_member_object(record: JsonObject, field_name: str) -> JsonObject:
    value = record.get(field_name)
    if isinstance(value, dict):
        return json_object(value)
    return {}
