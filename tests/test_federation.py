import json
from collections.abc import Callable
from dataclasses import dataclass, field
from os import environ
from typing import TYPE_CHECKING, cast

import pytest

_ = environ.setdefault("GNOSIS_TOKEN", "test-token")
_ = environ.setdefault("GNOSIS_READ_OPERATOR_TOKEN", "read-operator-token")
_ = environ.setdefault("GNOSIS_EXPORT_OPERATOR_TOKEN", "export-operator-token")
_ = environ.setdefault("GNOSIS_WRITE_OPERATOR_TOKEN", "write-operator-token")
_ = environ.setdefault("GNOSIS_ADMIN_OPERATOR_TOKEN", "admin-operator-token")
_ = environ.setdefault("NEO4J_URI", "bolt://neo4j.neo4j.svc.cluster.local:7687")
_ = environ.setdefault("NEO4J_PASSWORD", "test-password")
_ = environ.setdefault(
    "LITELLM_BASE_URL",
    "http://litellm.litellm.svc.cluster.local:4000/v1",
)
_ = environ.setdefault("LITELLM_API_KEY", "test-litellm-key")

import httpx2 as httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from pydantic_settings import SettingsError  # noqa: E402

from gnosis.federation import (  # noqa: E402
    FederationGateway,
    PeerNotAllowedError,
    UnknownPeerError,
)
from gnosis.main import create_app  # noqa: E402
from gnosis.models import (  # noqa: E402
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryAddResult,
    MemoryListRequest,
    MemoryListResponse,
    MemoryPeerError,
    MemoryPromotedRecord,
    MemoryPromoteFailure,
    MemoryPromoteResponse,
    MemoryRecord,
    MemorySearchRequest,
    MemorySearchResponse,
)
from gnosis.settings import PeerConfig, Settings  # noqa: E402

if TYPE_CHECKING:
    from gnosis.backend import MemoryBackend

_FEDERATION_TOKEN = "federation-inbound-token"
_PEER_TOKEN = "peer-outbound-token"
_PEER_BASE_URL = "http://gnosis-partner.gnosis-partner.svc.cluster.local:8080"
_MEMORY_ID = "00000000-0000-0000-0000-0000000000aa"
_OTHER_MEMORY_ID = "00000000-0000-0000-0000-0000000000bb"


def test_peer_settings_parse_and_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a two-peer registry in the deployment environment.
    monkeypatch.setenv(
        "GNOSIS_PEERS",
        json.dumps(
            [
                _peer_payload(),
                _peer_payload(name="lab", direction="pull", remote_tenant_id="lab"),
            ],
        ),
    )

    # When: settings load at startup.
    settings = Settings()

    # Then: peers are validated into typed configs with direction semantics.
    partner, lab = settings.gnosis_peers
    assert partner.name == "partner"
    assert partner.base_url == _PEER_BASE_URL
    assert partner.remote_tenant_id == "partner"
    assert partner.token_env_var == "GNOSIS_PEER_PARTNER_TOKEN"
    assert partner.allows_push()
    assert partner.allows_pull()
    assert lab.allows_pull()
    assert not lab.allows_push()


def test_peer_settings_default_to_no_peers_and_disabled_federation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: no federation environment is configured.
    monkeypatch.delenv("GNOSIS_PEERS", raising=False)
    monkeypatch.delenv("GNOSIS_FEDERATION_TOKEN", raising=False)

    # When: settings load.
    settings = Settings()

    # Then: federation stays off in both directions.
    assert settings.gnosis_peers == []
    assert settings.gnosis_federation_token == ""


def test_peer_settings_reject_duplicate_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two peers whose names collide after env-var normalization.
    monkeypatch.setenv(
        "GNOSIS_PEERS",
        json.dumps([_peer_payload(name="partner"), _peer_payload(name="Partner")]),
    )

    # When / Then: startup validation fails loudly.
    with pytest.raises(ValidationError, match="duplicate peer name"):
        _ = Settings()


def test_peer_settings_reject_unknown_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a peer with a direction outside the contract.
    monkeypatch.setenv(
        "GNOSIS_PEERS",
        json.dumps([_peer_payload(direction="sideways")]),
    )

    # When / Then: startup validation fails loudly.
    with pytest.raises(ValidationError):
        _ = Settings()


def test_peer_settings_reject_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a peer registry that is not valid JSON.
    monkeypatch.setenv("GNOSIS_PEERS", "not-json")

    # When / Then: startup fails instead of silently running unfederated.
    with pytest.raises(SettingsError):
        _ = Settings()


def test_peer_registry_enforces_existence_and_direction() -> None:
    # Given: a gateway with one push-only and one pull-only peer.
    gateway = FederationGateway(
        Settings(
            gnosis_peers=[
                PeerConfig(
                    name="pushy",
                    base_url=_PEER_BASE_URL,
                    direction="push",
                    remote_tenant_id="pushy",
                ),
                PeerConfig(
                    name="pully",
                    base_url=_PEER_BASE_URL,
                    direction="pull",
                    remote_tenant_id="pully",
                ),
            ],
        ),
    )

    # When / Then: unknown peers and direction mismatches are rejected.
    assert gateway.require_push_peer("pushy").name == "pushy"
    assert gateway.require_pull_peer("pully").name == "pully"
    with pytest.raises(UnknownPeerError, match="unknown peer: ghost"):
        _ = gateway.require_push_peer("ghost")
    with pytest.raises(PeerNotAllowedError, match="does not allow pull"):
        _ = gateway.require_pull_peer("pushy")
    with pytest.raises(PeerNotAllowedError, match="does not allow push"):
        _ = gateway.require_push_peer("pully")


def test_federated_search_injects_shareable_conjunct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a caller authenticated with the federation token class.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend)

    # When: the federated caller searches with its own filters.
    response = client.post(
        "/v1/memories/search",
        headers=_federation_header(),
        json={
            "scope": _scope_payload(),
            "query": "what snacks?",
            "filters": {"metadata.topic": "snacks"},
        },
    )

    # Then: the server conjoins the mandatory shareable-consent filter.
    assert response.status_code == 200
    assert backend.memory_search_requests[0].filters == {
        "AND": [{"metadata.topic": "snacks"}, {"metadata.shareable": True}],
    }


def test_federated_search_without_filters_still_requires_shareable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a federated caller with no filters of its own.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend)

    # When: the federated caller searches without filters.
    response = client.post(
        "/v1/memories/search",
        headers=_federation_header(),
        json={"scope": _scope_payload(), "query": "anything"},
    )

    # Then: the shareable filter is mandatory, not optional.
    assert response.status_code == 200
    assert backend.memory_search_requests[0].filters == {"metadata.shareable": True}


def test_federated_list_injects_shareable_conjunct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a federated caller listing memories.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend)

    # When: the federated caller lists with its own filters.
    response = client.post(
        "/v1/memories/list",
        headers=_federation_header(),
        json={"scope": _scope_payload(), "filters": {"metadata.topic": "snacks"}},
    )

    # Then: the server conjoins the mandatory shareable-consent filter.
    assert response.status_code == 200
    assert backend.memory_list_requests[0].filters == {
        "AND": [{"metadata.topic": "snacks"}, {"metadata.shareable": True}],
    }


def test_federated_search_rejects_peer_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a federated caller trying to trigger transitive fan-out.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend)

    # When: the federated caller names peers in its search.
    response = client.post(
        "/v1/memories/search",
        headers=_federation_header(),
        json={"scope": _scope_payload(), "query": "loop", "peers": ["partner"]},
    )

    # Then: fan-out is refused so federation cannot loop between instances.
    assert response.status_code == 403
    assert backend.memory_search_requests == []


def test_federated_add_requires_promotion_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a federated caller writing memories.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend)

    # When: the write lacks promoted_from provenance.
    rejected = client.post(
        "/v1/memories",
        headers=_federation_header(),
        json={
            "scope": _scope_payload(),
            "content": "smuggled",
            "infer": False,
        },
    )

    # Then: the write is refused before the backend runs.
    assert rejected.status_code == 403
    assert "promoted_from" in rejected.json()["detail"]
    assert backend.memory_add_requests == []

    # When: the write carries promoted_from provenance.
    accepted = client.post(
        "/v1/memories",
        headers=_federation_header(),
        json={
            "scope": _scope_payload(),
            "content": "promoted memory",
            "infer": False,
            "metadata": {"promoted_from": "bromigos"},
        },
    )

    # Then: the promoted write is accepted.
    assert accepted.status_code == 200
    assert backend.memory_add_requests[0].metadata["promoted_from"] == "bromigos"


def test_federation_token_is_rejected_on_non_memory_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a federated caller probing beyond the memory provider surface.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend)
    attempts = [
        client.post(
            "/v1/messages",
            headers=_federation_header(),
            json={"scope": _scope_payload(), "role": "user", "content": "hi"},
        ),
        client.post(
            "/v1/memory/context",
            headers=_federation_header(),
            json={"scope": _scope_payload(), "query": "what matters?"},
        ),
        client.patch(
            f"/v1/memories/{_MEMORY_ID}",
            headers=_federation_header(),
            json={"scope": _scope_payload(), "content": "rewritten"},
        ),
        client.request(
            "DELETE",
            f"/v1/memories/{_MEMORY_ID}",
            headers=_federation_header(),
            json={"scope": _scope_payload()},
        ),
        client.post(
            "/v1/memories/promote",
            headers=_federation_header(),
            json={"scope": _scope_payload(), "peer": "partner"},
        ),
    ]

    # When / Then: every non-memory route answers 403 for the token class.
    for response in attempts:
        assert response.status_code == 403, response.request.url
        assert (
            response.json()["detail"]
            == "federation token is not authorized for this route"
        )


def test_federation_token_is_unauthorized_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an instance without an inbound federation token configured.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend, federation_token="")

    # When: a caller presents a would-be federation token.
    response = client.post(
        "/v1/memories/search",
        headers={"Authorization": f"Bearer {_FEDERATION_TOKEN}"},
        json={"scope": _scope_payload(), "query": "anything"},
    )

    # Then: inbound federation is disabled, not silently matched.
    assert response.status_code == 401


def test_promote_dry_run_returns_only_shareable_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a service caller reviewing what a promotion would push.
    calls: list[httpx.Request] = []
    backend = FederationRecordingBackend()
    client = _federated_app_client(
        monkeypatch,
        backend,
        transport=httpx.MockTransport(_recording_handler(calls)),
    )

    # When: the caller runs the default review-first promotion.
    response = client.post(
        "/v1/memories/promote",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "peer": "partner",
            "filters": {"metadata.topic": "snacks"},
            "limit": 25,
        },
    )

    # Then: shareable-only candidates return and nothing leaves the instance.
    assert response.status_code == 200
    assert response.json() == {
        "peer": "partner",
        "count": 1,
        "dry_run": True,
        "candidates": [
            {
                "memory_id": _MEMORY_ID,
                "content": "remember this",
                "metadata": {"topic": "snacks", "shareable": True},
            },
        ],
        "promoted": [],
        "failed": [],
    }
    listing = backend.memory_list_requests[0]
    assert listing.filters == {
        "AND": [{"metadata.topic": "snacks"}, {"metadata.shareable": True}],
    }
    assert listing.page_size == 25
    assert calls == []


def test_promote_real_run_posts_provenance_and_reports_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two shareable candidates and a peer that accepts only the first.
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        add_request = MemoryAddRequest.model_validate_json(request.content)
        if add_request.metadata.get("source_memory_id") == _OTHER_MEMORY_ID:
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(
            200,
            json=MemoryAddResponse(
                results=[
                    MemoryAddResult(
                        memory_id="peer-memory-1",
                        content="remember this",
                        event="ADD",
                    ),
                ],
            ).model_dump(mode="json"),
        )

    backend = FederationRecordingBackend()
    backend.memory_list = MemoryListResponse(
        results=[
            _memory_record(_MEMORY_ID, "remember this", score=None),
            _memory_record(_OTHER_MEMORY_ID, "also share this", score=None),
        ],
        total=2,
        page=1,
        page_size=50,
    )
    client = _federated_app_client(
        monkeypatch,
        backend,
        transport=httpx.MockTransport(handler),
    )

    # When: the caller applies the promotion for real.
    response = client.post(
        "/v1/memories/promote",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "peer": "partner", "dry_run": False},
    )

    # Then: the manifest reports the applied push and the partial failure.
    assert response.status_code == 200
    manifest = MemoryPromoteResponse.model_validate_json(response.content)
    assert manifest.dry_run is False
    assert manifest.count == 2
    assert manifest.promoted == [
        MemoryPromotedRecord(
            source_memory_id=_MEMORY_ID,
            peer_memory_id="peer-memory-1",
            event="ADD",
        ),
    ]
    assert manifest.failed == [
        MemoryPromoteFailure(
            source_memory_id=_OTHER_MEMORY_ID,
            error="peer responded with HTTP 500",
        ),
    ]

    # Then: each push is a verbatim add with mapped scope and provenance.
    assert len(calls) == 2
    accepted = MemoryAddRequest.model_validate_json(calls[0].content)
    assert str(calls[0].url) == f"{_PEER_BASE_URL}/v1/memories"
    assert calls[0].headers["Authorization"] == f"Bearer {_PEER_TOKEN}"
    assert accepted.scope.tenant_id == "partner"
    assert accepted.scope.space_id == "federation"
    assert accepted.scope.agent_id == "gnosis:bromigos"
    assert accepted.scope.session_id == "promote"
    assert accepted.scope.user_id == "789"
    assert accepted.scope.visibility == "private_user"
    assert accepted.infer is False
    assert accepted.content == "remember this"
    assert accepted.metadata["promoted_from"] == "bromigos"
    assert accepted.metadata["source_memory_id"] == _MEMORY_ID
    assert accepted.metadata["topic"] == "snacks"
    assert "shareable" not in accepted.metadata
    promoted_at = accepted.metadata["promoted_at"]
    assert isinstance(promoted_at, str)
    assert promoted_at


def test_promote_rejects_unknown_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a promotion aimed at a peer outside the registry.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend)

    # When: the caller promotes to the unknown peer.
    response = client.post(
        "/v1/memories/promote",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "peer": "ghost"},
    )

    # Then: the request is rejected before any read happens.
    assert response.status_code == 400
    assert response.json()["detail"] == "unknown peer: ghost"
    assert backend.memory_list_requests == []


def test_promote_rejects_pull_only_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a peer configured for pull-only federation.
    backend = FederationRecordingBackend()
    client = _federated_app_client(
        monkeypatch,
        backend,
        peers=[_peer_payload(direction="pull")],
    )

    # When: the caller tries to push to it.
    response = client.post(
        "/v1/memories/promote",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "peer": "partner"},
    )

    # Then: the direction policy is enforced.
    assert response.status_code == 403
    assert response.json()["detail"] == "peer partner does not allow push"


def test_promote_real_run_requires_outbound_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a push-capable peer without an outbound token configured.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend, peer_token=None)

    # When: the caller applies a promotion for real.
    response = client.post(
        "/v1/memories/promote",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "peer": "partner", "dry_run": False},
    )

    # Then: the misconfiguration is reported clearly instead of half-running.
    assert response.status_code == 503
    assert "GNOSIS_PEER_PARTNER_TOKEN" in response.json()["detail"]


def test_search_with_peers_merges_by_score_with_origin_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a peer holding one stronger and one weaker shareable match.
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json=MemorySearchResponse(
                results=[
                    _memory_record("peer-memory-1", "peer strong", score=0.95),
                    _memory_record("peer-memory-2", "peer weak", score=0.11),
                ],
            ).model_dump(mode="json"),
        )

    backend = FederationRecordingBackend()
    client = _federated_app_client(
        monkeypatch,
        backend,
        transport=httpx.MockTransport(handler),
    )

    # When: a service caller searches across the peer.
    response = client.post(
        "/v1/memories/search",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "query": "what snacks?",
            "peers": ["partner"],
        },
    )

    # Then: results interleave by score descending with origin tags.
    assert response.status_code == 200
    merged = MemorySearchResponse.model_validate_json(response.content)
    assert [
        (result.memory_id, result.origin, result.score) for result in merged.results
    ] == [
        ("peer-memory-1", "partner", 0.95),
        (_MEMORY_ID, "local", 0.91),
        ("peer-memory-2", "partner", 0.11),
    ]
    assert b'"peer_errors"' not in response.content

    # Then: the fan-out query maps the scope tenant and never forwards peers.
    remote = MemorySearchRequest.model_validate_json(calls[0].content)
    assert str(calls[0].url) == f"{_PEER_BASE_URL}/v1/memories/search"
    assert calls[0].headers["Authorization"] == f"Bearer {_PEER_TOKEN}"
    assert remote.scope.tenant_id == "partner"
    assert remote.scope.user_id == "789"
    assert remote.query == "what snacks?"
    assert remote.peers == []
    assert b'"peers"' not in calls[0].content


def test_search_with_peers_caps_merged_results_at_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: more combined matches than the caller's limit.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=MemorySearchResponse(
                results=[
                    _memory_record("peer-memory-1", "peer strong", score=0.95),
                    _memory_record("peer-memory-2", "peer weak", score=0.11),
                ],
            ).model_dump(mode="json"),
        )

    backend = FederationRecordingBackend()
    client = _federated_app_client(
        monkeypatch,
        backend,
        transport=httpx.MockTransport(handler),
    )

    # When: the caller searches with limit 2.
    response = client.post(
        "/v1/memories/search",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "query": "what snacks?",
            "limit": 2,
            "peers": ["partner"],
        },
    )

    # Then: only the top-scored results survive the cap.
    assert response.status_code == 200
    merged = MemorySearchResponse.model_validate_json(response.content)
    assert [result.memory_id for result in merged.results] == [
        "peer-memory-1",
        _MEMORY_ID,
    ]


def test_search_with_peers_degrades_gracefully_on_peer_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a peer that times out.
    def handler(_request: httpx.Request) -> httpx.Response:
        detail = "peer took too long"
        raise httpx.ConnectTimeout(detail)

    backend = FederationRecordingBackend()
    client = _federated_app_client(
        monkeypatch,
        backend,
        transport=httpx.MockTransport(handler),
    )

    # When: a service caller searches across the peer.
    response = client.post(
        "/v1/memories/search",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "query": "what snacks?",
            "peers": ["partner"],
        },
    )

    # Then: local results still answer and the peer failure is reported.
    assert response.status_code == 200
    merged = MemorySearchResponse.model_validate_json(response.content)
    assert [result.origin for result in merged.results] == ["local"]
    assert merged.peer_errors == [
        MemoryPeerError(peer="partner", error="peer request timed out"),
    ]


def test_search_with_unknown_peer_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a search naming a peer outside the registry.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend)

    # When: the caller searches across the unknown peer.
    response = client.post(
        "/v1/memories/search",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "anything", "peers": ["ghost"]},
    )

    # Then: the request is rejected before any fan-out.
    assert response.status_code == 400
    assert response.json()["detail"] == "unknown peer: ghost"
    assert backend.memory_search_requests == []


def test_search_with_push_only_peer_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a peer configured for push-only federation.
    backend = FederationRecordingBackend()
    client = _federated_app_client(
        monkeypatch,
        backend,
        peers=[_peer_payload(direction="push")],
    )

    # When: the caller tries to pull from it.
    response = client.post(
        "/v1/memories/search",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "anything", "peers": ["partner"]},
    )

    # Then: the direction policy is enforced.
    assert response.status_code == 403
    assert response.json()["detail"] == "peer partner does not allow pull"


def test_search_without_peers_keeps_existing_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a plain non-federated search.
    backend = FederationRecordingBackend()
    client = _federated_app_client(monkeypatch, backend)

    # When: the caller searches without naming peers.
    response = client.post(
        "/v1/memories/search",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "what snacks?"},
    )

    # Then: the response shape is unchanged - no origin, no peer_errors.
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "memory_id": _MEMORY_ID,
                "content": "remember this",
                "score": 0.91,
                "metadata": {"topic": "snacks", "shareable": True},
                "created_at": "2026-06-27T01:02:03+00:00",
                "updated_at": None,
            },
        ],
    }


def _peer_payload(
    *,
    name: str = "partner",
    direction: str = "both",
    remote_tenant_id: str = "partner",
) -> dict[str, str]:
    return {
        "name": name,
        "base_url": _PEER_BASE_URL,
        "direction": direction,
        "remote_tenant_id": remote_tenant_id,
    }


def _federated_app_client(  # noqa: PLR0913 - mirrors the federation env knobs.
    monkeypatch: pytest.MonkeyPatch,
    backend: "FederationRecordingBackend",
    *,
    transport: httpx.MockTransport | None = None,
    peers: list[dict[str, str]] | None = None,
    federation_token: str = _FEDERATION_TOKEN,
    peer_token: str | None = _PEER_TOKEN,
) -> TestClient:
    monkeypatch.setenv("GNOSIS_PEERS", json.dumps(peers or [_peer_payload()]))
    monkeypatch.setenv("GNOSIS_FEDERATION_TOKEN", federation_token)
    if peer_token is None:
        monkeypatch.delenv("GNOSIS_PEER_PARTNER_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GNOSIS_PEER_PARTNER_TOKEN", peer_token)
    return TestClient(
        create_app(
            settings_factory=Settings,
            backend=cast("MemoryBackend", cast("object", backend)),
            federation_transport=transport,
        ),
    )


def _recording_handler(
    calls: list[httpx.Request],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"results": []})

    return handler


def _auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {environ['GNOSIS_TOKEN']}"}


def _federation_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {_FEDERATION_TOKEN}"}


def _scope_payload() -> dict[str, str]:
    return {
        "tenant_id": "bromigos",
        "space_id": "discord",
        "agent_id": "pc-principal",
        "session_id": "guild:123:channel:456",
        "user_id": "789",
        "visibility": "channel",
    }


def _memory_record(
    memory_id: str,
    content: str,
    *,
    score: float | None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        content=content,
        score=score,
        metadata={"topic": "snacks", "shareable": True},
        created_at="2026-06-27T01:02:03+00:00",
    )


@dataclass(slots=True)
class FederationRecordingBackend:
    """Just enough MemoryBackend surface for the federation routes."""

    memory_search: MemorySearchResponse = field(
        default_factory=lambda: MemorySearchResponse(
            results=[_memory_record(_MEMORY_ID, "remember this", score=0.91)],
        ),
    )
    memory_list: MemoryListResponse = field(
        default_factory=lambda: MemoryListResponse(
            results=[_memory_record(_MEMORY_ID, "remember this", score=None)],
            total=1,
            page=1,
            page_size=50,
        ),
    )
    memory_add: MemoryAddResponse = field(
        default_factory=lambda: MemoryAddResponse(
            results=[
                MemoryAddResult(
                    memory_id=_MEMORY_ID,
                    content="remember this",
                    event="ADD",
                ),
            ],
        ),
    )
    memory_search_requests: list[MemorySearchRequest] = field(default_factory=list)
    memory_list_requests: list[MemoryListRequest] = field(default_factory=list)
    memory_add_requests: list[MemoryAddRequest] = field(default_factory=list)

    async def add_memories(self, request: MemoryAddRequest) -> MemoryAddResponse:
        self.memory_add_requests.append(request)
        return self.memory_add

    async def search_memories(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        self.memory_search_requests.append(request)
        return self.memory_search

    async def list_memories(self, request: MemoryListRequest) -> MemoryListResponse:
        self.memory_list_requests.append(request)
        return self.memory_list

    async def shutdown(self) -> None:
        return
