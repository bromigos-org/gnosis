"""Cross-gnosis memory federation.

One peer concept backs both federation directions: ``GNOSIS_PEERS`` names the
sovereign deployments this instance may talk to, promote pushes explicitly
shareable memories to a peer, and federated search pulls shareable memories
from peers into one merged, origin-tagged result set. Outbound calls always
ride per-peer bearer tokens (``GNOSIS_PEER_<NAME>_TOKEN``); the remote side
authenticates them as federated callers and enforces shareable-only reads.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from os import environ
from typing import Final

import httpx2 as httpx

from gnosis.models import (
    JsonObject,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryPeerError,
    MemoryPromoteCandidate,
    MemoryPromotedRecord,
    MemoryPromoteFailure,
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryVisibility,
)
from gnosis.redaction import redact_secrets
from gnosis.settings import PeerConfig, Settings

type FederationTransport = httpx.AsyncBaseTransport

LOCAL_ORIGIN: Final[str] = "local"
SHAREABLE_FILTER_FIELD: Final[str] = "metadata.shareable"

_PROMOTE_SPACE_ID: Final[str] = "federation"
_PROMOTE_SESSION_ID: Final[str] = "promote"
_PROMOTE_TIMEOUT_SECONDS: Final[float] = 15.0
_PROMOTE_MAX_CONCURRENCY: Final[int] = 4
_SEARCH_TIMEOUT_SECONDS: Final[float] = 10.0
_EMPTY_ADD_RESULT_DETAIL: Final[str] = "peer returned no add result"


class UnknownPeerError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail: str = detail
        super().__init__(detail)


class PeerNotAllowedError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail: str = detail
        super().__init__(detail)


class PeerTokenUnavailableError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail: str = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class PromoteOutcome:
    promoted: list[MemoryPromotedRecord]
    failed: list[MemoryPromoteFailure]


def shareable_filters(filters: JsonObject | None) -> JsonObject:
    """Conjoin the mandatory shareable-consent filter onto caller filters."""
    conjunct: JsonObject = {SHAREABLE_FILTER_FIELD: True}
    if filters is None:
        return conjunct
    return {"AND": [filters, conjunct]}


def merged_search_results(
    local: Sequence[MemoryRecord],
    remote: Sequence[MemoryRecord],
    limit: int,
) -> list[MemoryRecord]:
    """Interleave local and remote results by score descending, origin-tagged."""
    combined = [record.model_copy(update={"origin": LOCAL_ORIGIN}) for record in local]
    combined.extend(remote)
    combined.sort(key=_result_score, reverse=True)
    return combined[:limit]


class FederationGateway:
    def __init__(
        self,
        settings: Settings,
        transport: FederationTransport | None = None,
    ) -> None:
        self._local_tenant_id: str = settings.gnosis_tenant_id
        self._peers: dict[str, PeerConfig] = {
            peer.name: peer for peer in settings.gnosis_peers
        }
        self._peer_tokens: dict[str, str] = {
            peer.name: environ.get(peer.token_env_var, "")
            for peer in settings.gnosis_peers
        }
        self._transport: FederationTransport | None = transport

    def require_push_peer(self, name: str) -> PeerConfig:
        peer = self._require_peer(name)
        if not peer.allows_push():
            detail = f"peer {peer.name} does not allow push"
            raise PeerNotAllowedError(detail)
        return peer

    def require_pull_peer(self, name: str) -> PeerConfig:
        peer = self._require_peer(name)
        if not peer.allows_pull():
            detail = f"peer {peer.name} does not allow pull"
            raise PeerNotAllowedError(detail)
        return peer

    async def promote(
        self,
        peer: PeerConfig,
        scope: MemoryScope,
        candidates: Sequence[MemoryPromoteCandidate],
    ) -> PromoteOutcome:
        token = self._require_peer_token(peer)
        promoted_at = datetime.now(UTC).isoformat()
        semaphore = asyncio.Semaphore(_PROMOTE_MAX_CONCURRENCY)
        async with self._peer_client(token, _PROMOTE_TIMEOUT_SECONDS) as client:
            outcomes = await asyncio.gather(
                *[
                    self._push_candidate(
                        client,
                        semaphore,
                        peer,
                        scope,
                        candidate,
                        promoted_at,
                    )
                    for candidate in candidates
                ],
            )
        return PromoteOutcome(
            promoted=[
                outcome
                for outcome in outcomes
                if isinstance(outcome, MemoryPromotedRecord)
            ],
            failed=[
                outcome
                for outcome in outcomes
                if isinstance(outcome, MemoryPromoteFailure)
            ],
        )

    async def search_peers(
        self,
        request: MemorySearchRequest,
    ) -> tuple[list[MemoryRecord], list[MemoryPeerError]]:
        peers = [self.require_pull_peer(name) for name in request.peers]
        results: list[MemoryRecord] = []
        errors: list[MemoryPeerError] = []
        outcomes = await asyncio.gather(
            *[self._search_peer(peer, request) for peer in peers],
        )
        for outcome in outcomes:
            if isinstance(outcome, MemoryPeerError):
                errors.append(outcome)
            else:
                results.extend(outcome)
        return results, errors

    async def _push_candidate(  # noqa: PLR0913 - mirrors the fan-out inputs.
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        peer: PeerConfig,
        scope: MemoryScope,
        candidate: MemoryPromoteCandidate,
        promoted_at: str,
    ) -> MemoryPromotedRecord | MemoryPromoteFailure:
        request = MemoryAddRequest(
            scope=_promoted_scope(peer, scope, self._local_tenant_id),
            content=_redacted_text(candidate.content),
            infer=False,
            metadata=_promoted_metadata(
                candidate,
                self._local_tenant_id,
                promoted_at,
            ),
        )
        try:
            async with semaphore:
                response = await client.post(
                    f"{peer.base_url}/v1/memories",
                    json=request.model_dump(mode="json"),
                )
            _ = response.raise_for_status()
            results = MemoryAddResponse.model_validate_json(response.content).results
        except (httpx.HTTPError, ValueError) as error:
            return MemoryPromoteFailure(
                source_memory_id=candidate.memory_id,
                error=_peer_error_text(error),
            )
        if not results:
            return MemoryPromoteFailure(
                source_memory_id=candidate.memory_id,
                error=_EMPTY_ADD_RESULT_DETAIL,
            )
        return MemoryPromotedRecord(
            source_memory_id=candidate.memory_id,
            peer_memory_id=results[0].memory_id,
            event=results[0].event,
        )

    async def _search_peer(
        self,
        peer: PeerConfig,
        request: MemorySearchRequest,
    ) -> list[MemoryRecord] | MemoryPeerError:
        try:
            token = self._require_peer_token(peer)
        except PeerTokenUnavailableError as error:
            return MemoryPeerError(peer=peer.name, error=error.detail)
        try:
            async with self._peer_client(token, _SEARCH_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{peer.base_url}/v1/memories/search",
                    json=_remote_search_payload(request, peer),
                )
            _ = response.raise_for_status()
            remote = MemorySearchResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValueError) as error:
            return MemoryPeerError(peer=peer.name, error=_peer_error_text(error))
        return [
            record.model_copy(update={"origin": peer.name}) for record in remote.results
        ]

    def _require_peer(self, name: str) -> PeerConfig:
        peer = self._peers.get(name)
        if peer is None:
            detail = f"unknown peer: {name}"
            raise UnknownPeerError(detail)
        return peer

    def _require_peer_token(self, peer: PeerConfig) -> str:
        token = self._peer_tokens.get(peer.name, "")
        if not token:
            detail = (
                f"outbound token for peer {peer.name} is not configured; "
                f"set {peer.token_env_var}"
            )
            raise PeerTokenUnavailableError(detail)
        return token

    def _peer_client(self, token: str, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"},
            transport=self._transport,
        )


def _promoted_scope(
    peer: PeerConfig,
    scope: MemoryScope,
    local_tenant_id: str,
) -> MemoryScope:
    return MemoryScope(
        tenant_id=peer.remote_tenant_id,
        space_id=_PROMOTE_SPACE_ID,
        agent_id=f"gnosis:{local_tenant_id}",
        session_id=_PROMOTE_SESSION_ID,
        user_id=scope.user_id,
        visibility=MemoryVisibility.PRIVATE_USER,
    )


def _promoted_metadata(
    candidate: MemoryPromoteCandidate,
    local_tenant_id: str,
    promoted_at: str,
) -> JsonObject:
    """Merge provenance over the candidate metadata, without re-sharing.

    ``shareable`` is stripped so promotion never transitively re-shares a
    memory from the receiving tenant; the peer decides its own consent tags.
    Caller metadata is redacted before the gateway-generated provenance keys
    are merged in, so provenance ids survive the opaque-value redaction.
    """
    metadata = _redacted_object(
        {key: value for key, value in candidate.metadata.items() if key != "shareable"},
    )
    metadata["promoted_from"] = local_tenant_id
    metadata["source_memory_id"] = candidate.memory_id
    metadata["promoted_at"] = promoted_at
    return metadata


def _remote_search_payload(
    request: MemorySearchRequest,
    peer: PeerConfig,
) -> JsonObject:
    remote_scope = request.scope.model_copy(
        update={"tenant_id": peer.remote_tenant_id},
    )
    remote_request = request.model_copy(
        update={"scope": remote_scope, "peers": []},
    )
    return remote_request.model_dump(mode="json")


def _result_score(record: MemoryRecord) -> float:
    if record.score is None:
        return 0.0
    return record.score


def _peer_error_text(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"peer responded with HTTP {error.response.status_code}"
    if isinstance(error, httpx.TimeoutException):
        return "peer request timed out"
    return type(error).__name__


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
