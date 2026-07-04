import logging
from dataclasses import dataclass, field
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Self
from uuid import UUID

import pytest
from neo4j_agent_memory.memory.long_term import EntityType
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolCallStatus, ToolStats
from neo4j_agent_memory.schema.models import EntityRef
from pydantic import TypeAdapter, ValidationError

environ["GNOSIS_TOKEN"] = "test-token"
environ["GNOSIS_READ_OPERATOR_TOKEN"] = "read-operator-token"
environ["GNOSIS_EXPORT_OPERATOR_TOKEN"] = "export-operator-token"
environ["GNOSIS_WRITE_OPERATOR_TOKEN"] = "write-operator-token"
environ["GNOSIS_ADMIN_OPERATOR_TOKEN"] = "admin-operator-token"
environ["NEO4J_URI"] = "bolt://neo4j.neo4j.svc.cluster.local:7687"
environ["NEO4J_PASSWORD"] = "test-password"
environ["LITELLM_BASE_URL"] = "http://litellm.litellm.svc.cluster.local:4000/v1"
environ["LITELLM_API_KEY"] = "test-litellm-key"

from fastapi.testclient import TestClient
from neo4j_agent_memory import MemorySettings

from gnosis.backend import (
    BackendCapabilityUnavailable,
    BackendRequestError,
    MemoryClientContext,
    MemoryNotFoundError,
    Neo4jAgentMemoryBackend,
)
from gnosis.main import create_app
from gnosis.models import (
    BackendReadiness,
    BufferFlushResponse,
    BufferStatus,
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ClientEventType,
    ConsolidationApplyRequest,
    ConsolidationApplyResponse,
    ConsolidationDryRunRequest,
    ConsolidationDryRunResponse,
    ContextRequest,
    ContextResponse,
    DedupApplyRequest,
    DedupApplyResponse,
    DedupCandidate,
    DedupCandidateRequest,
    DedupCandidateResponse,
    DedupEntitySnapshot,
    DedupOperatorAudit,
    DedupStatsRequest,
    DedupStatsResponse,
    DiagnosticsConfig,
    DiagnosticsResponse,
    EntityRecord,
    EntitySearchRequest,
    EntitySearchResponse,
    EntityWriteRequest,
    EventIngestResult,
    EventIngestStatus,
    ExtractionCandidate,
    ExtractionPreviewMetrics,
    ExtractionPreviewProvenance,
    ExtractionPreviewRequest,
    ExtractionPreviewResponse,
    FactRecord,
    FactSearchRequest,
    FactSearchResponse,
    FactWriteRequest,
    GraphContextRequest,
    GraphContextResponse,
    GraphExportNode,
    GraphExportRelationship,
    GraphExportRequest,
    GraphExportResponse,
    JsonObject,
    JsonValue,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryAddResult,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSection,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryListRequest,
    MemoryListResponse,
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
    MemoryVisibility,
    MessageWriteRequest,
    MessageWriteResponse,
    PreferenceRecord,
    PreferenceSearchRequest,
    PreferenceSearchResponse,
    PreferenceWriteRequest,
    ReasoningContextRequest,
    ReasoningContextResponse,
    ReasoningSimilarTracesRequest,
    ReasoningSimilarTracesResponse,
    ReasoningStepRecord,
    ReasoningStepRequest,
    ReasoningStepResponse,
    ReasoningStepSearchRequest,
    ReasoningStepSearchResponse,
    ReasoningToolCallRequest,
    ReasoningToolCallResponse,
    ReasoningToolStatsRecord,
    ReasoningToolStatsRequest,
    ReasoningToolStatsResponse,
    ReasoningTraceCompleteRequest,
    ReasoningTraceCompleteResponse,
    ReasoningTraceDetailRequest,
    ReasoningTraceDetailResponse,
    ReasoningTraceListRequest,
    ReasoningTraceListResponse,
    ReasoningTraceStartRequest,
    ReasoningTraceStartResponse,
    ReasoningTraceSummary,
    SdkStatsRequest,
    SdkStatsResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillRecord,
    SkillStatus,
    SkillUsage,
    default_event_visibility,
    default_skill_visibility,
)
from gnosis.settings import Settings

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_JSON_OBJECTS_ADAPTER: TypeAdapter[list[JsonObject]] = TypeAdapter(list[JsonObject])
_MEMORY_CONTEXT_CONTRACT_FIXTURE = (
    Path(__file__).parent / "testdata" / "memory_context_enriched_contract.json"
)


def test_health_when_called_without_auth() -> None:
    # Given: an app configured with a memory token.
    client = TestClient(create_app(settings_factory=_settings))

    # When: a client calls the unauthenticated health endpoint.
    response = client.get("/health")

    # Then: the service reports healthy.
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_backend_is_available_without_auth() -> None:
    # Given: an app whose graph backend can bootstrap and connect.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: Kubernetes calls the unauthenticated readiness endpoint.
    response = client.get("/ready")

    # Then: readiness succeeds without requiring bearer credentials.
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert backend.readiness_checks == 1


def test_ready_when_backend_is_unavailable_without_auth() -> None:
    # Given: an app whose graph backend cannot connect.
    backend = RecordingBackend(backend_available=False)
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: Kubernetes calls the unauthenticated readiness endpoint.
    response = client.get("/ready")

    # Then: readiness fails shallowly without leaking connection details.
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


def test_diagnostics_requires_bearer_auth() -> None:
    # Given: diagnostics exposes backend configuration readiness details.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller omits credentials.
    response = client.get("/v1/diagnostics")

    # Then: diagnostics remains protected.
    assert response.status_code == 401


def test_all_memory_endpoints_require_bearer_token() -> None:
    client = TestClient(
        create_app(settings_factory=_settings, backend=RecordingBackend()),
    )

    assert client.get("/v1/diagnostics").status_code == 401
    assert (
        client.request(
            "GET",
            "/v1/memory/stats",
            json={"scope": _scope_payload()},
        ).status_code
        == 401
    )
    for path, payload in _protected_post_endpoint_payloads():
        response = client.post(path, json=payload)
        assert response.status_code == 401, path

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_add_memories_returns_stable_ids() -> None:
    # Given: an authenticated caller adding a verbatim memory.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the memory is added.
    response = client.post(
        "/v1/memories",
        headers=_auth_header(),
        json=_memory_add_payload(),
    )

    # Then: the add result carries a stable memory id and event.
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "memory_id": "00000000-0000-0000-0000-0000000000aa",
                "content": "remember this",
                "event": "ADD",
                "metadata": {},
            },
        ],
    }
    assert backend.memory_add_requests[0].content == "Cartman prefers cheesy poofs"
    assert backend.memory_add_requests[0].infer is False


def test_add_memories_rejects_foreign_tenant_scope() -> None:
    # Given: a caller holding a token for another tenant's scope.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller adds a memory with a foreign tenant scope.
    response = client.post(
        "/v1/memories",
        headers=_auth_header(),
        json=_memory_add_payload(scope=_scope_payload(tenant_id="evil")),
    )

    # Then: the request is rejected before the backend runs.
    assert response.status_code == 403
    assert backend.memory_add_requests == []


def test_add_memories_maps_mode_errors_to_bad_request() -> None:
    # Given: the backend rejects the add mode combination.
    backend = RecordingBackend()
    backend.memory_add_error = BackendRequestError(
        "Provide messages with infer=true or content with infer=false.",
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller sends the invalid combination.
    response = client.post(
        "/v1/memories",
        headers=_auth_header(),
        json=_memory_add_payload(),
    )

    # Then: the caller receives a 400 with the policy detail.
    assert response.status_code == 400
    assert "infer" in response.json()["detail"]


def test_search_memories_returns_ranked_results() -> None:
    # Given: an authenticated caller searching long-term memories.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller searches with filters and a score floor.
    response = client.post(
        "/v1/memories/search",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "query": "what snacks?",
            "filters": {"metadata.topic": "snacks"},
            "limit": 5,
            "min_score": 0.5,
        },
    )

    # Then: scored results return and the backend saw the full request.
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "memory_id": "00000000-0000-0000-0000-0000000000aa",
                "content": "remember this",
                "score": 0.91,
                "metadata": {"topic": "snacks"},
                "created_at": "2026-06-27T01:02:03+00:00",
                "updated_at": None,
            },
        ],
    }
    request = backend.memory_search_requests[0]
    assert request.limit == 5
    assert request.min_score == 0.5
    assert request.filters == {"metadata.topic": "snacks"}


def test_search_memories_rejects_foreign_tenant_scope() -> None:
    # Given: a caller holding a token for another tenant's scope.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller searches with a foreign tenant scope.
    response = client.post(
        "/v1/memories/search",
        headers=_auth_header(),
        json={"scope": _scope_payload(tenant_id="evil"), "query": "secrets"},
    )

    # Then: the request is rejected before the backend runs.
    assert response.status_code == 403
    assert backend.memory_search_requests == []


def test_list_memories_returns_deterministic_page() -> None:
    # Given: an authenticated caller listing memories.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller lists page two.
    response = client.post(
        "/v1/memories/list",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "page": 2, "page_size": 10},
    )

    # Then: paging metadata is echoed with the results.
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "memory_id": "00000000-0000-0000-0000-0000000000aa",
                "content": "remember this",
                "metadata": {"topic": "snacks"},
                "created_at": "2026-06-27T01:02:03+00:00",
                "updated_at": None,
            },
        ],
        "total": 1,
        "page": 2,
        "page_size": 10,
    }


def test_list_memories_rejects_foreign_tenant_scope() -> None:
    # Given: a caller holding a token for another tenant's scope.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller lists with a foreign tenant scope.
    response = client.post(
        "/v1/memories/list",
        headers=_auth_header(),
        json={"scope": _scope_payload(tenant_id="evil")},
    )

    # Then: the request is rejected before the backend runs.
    assert response.status_code == 403
    assert backend.memory_list_requests == []


def test_update_memory_is_disabled_by_default() -> None:
    # Given: the default safe rollout posture.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller attempts a memory update.
    response = client.patch(
        "/v1/memories/00000000-0000-0000-0000-0000000000aa",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "content": "rewritten"},
    )

    # Then: the edit surface stays disabled with a clear message.
    assert response.status_code == 403
    assert response.json()["detail"] == "Memory editing is disabled by service policy."
    assert backend.memory_update_requests == []


def test_update_memory_when_edit_flag_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: memory editing is explicitly enabled.
    monkeypatch.setenv("GNOSIS_MEMORY_EDIT_ENABLED", "true")
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller updates a memory in scope.
    response = client.patch(
        "/v1/memories/00000000-0000-0000-0000-0000000000aa",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "content": "rewritten"},
    )

    # Then: the update succeeds with an UPDATE event.
    assert response.status_code == 200
    assert response.json() == {
        "memory_id": "00000000-0000-0000-0000-0000000000aa",
        "content": "rewritten",
        "event": "UPDATE",
    }
    memory_id, request = backend.memory_update_requests[0]
    assert memory_id == "00000000-0000-0000-0000-0000000000aa"
    assert request.content == "rewritten"


def test_update_memory_returns_not_found_outside_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: editing is enabled but the memory is not in the caller's scope.
    monkeypatch.setenv("GNOSIS_MEMORY_EDIT_ENABLED", "true")
    backend = RecordingBackend()
    backend.missing_memory_ids.add("00000000-0000-0000-0000-0000000000bb")
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller updates the out-of-scope memory.
    response = client.patch(
        "/v1/memories/00000000-0000-0000-0000-0000000000bb",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "content": "hijack"},
    )

    # Then: the caller learns nothing beyond not-found.
    assert response.status_code == 404
    assert response.json()["detail"] == "memory not found in scope"


def test_update_memory_rejects_foreign_tenant_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: editing is enabled but the scope belongs to another tenant.
    monkeypatch.setenv("GNOSIS_MEMORY_EDIT_ENABLED", "true")
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller updates with a foreign tenant scope.
    response = client.patch(
        "/v1/memories/00000000-0000-0000-0000-0000000000aa",
        headers=_auth_header(),
        json={"scope": _scope_payload(tenant_id="evil"), "content": "hijack"},
    )

    # Then: the request is rejected before the backend runs.
    assert response.status_code == 403
    assert backend.memory_update_requests == []


def test_delete_memory_is_disabled_by_default() -> None:
    # Given: the default safe rollout posture.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller attempts a memory delete.
    response = client.request(
        "DELETE",
        "/v1/memories/00000000-0000-0000-0000-0000000000aa",
        headers=_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: the edit surface stays disabled with a clear message.
    assert response.status_code == 403
    assert response.json()["detail"] == "Memory editing is disabled by service policy."
    assert backend.memory_delete_requests == []


def test_delete_memory_when_edit_flag_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: memory editing is explicitly enabled.
    monkeypatch.setenv("GNOSIS_MEMORY_EDIT_ENABLED", "true")
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller deletes a memory in scope.
    response = client.request(
        "DELETE",
        "/v1/memories/00000000-0000-0000-0000-0000000000aa",
        headers=_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: the delete succeeds with a DELETE event.
    assert response.status_code == 200
    assert response.json() == {
        "memory_id": "00000000-0000-0000-0000-0000000000aa",
        "event": "DELETE",
    }


def test_delete_memory_returns_not_found_outside_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: deleting is enabled but the memory is not in the caller's scope.
    monkeypatch.setenv("GNOSIS_MEMORY_EDIT_ENABLED", "true")
    backend = RecordingBackend()
    backend.missing_memory_ids.add("00000000-0000-0000-0000-0000000000bb")
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller deletes the out-of-scope memory.
    response = client.request(
        "DELETE",
        "/v1/memories/00000000-0000-0000-0000-0000000000bb",
        headers=_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: the caller learns nothing beyond not-found.
    assert response.status_code == 404
    assert response.json()["detail"] == "memory not found in scope"


def test_delete_memory_rejects_foreign_tenant_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: deleting is enabled but the scope belongs to another tenant.
    monkeypatch.setenv("GNOSIS_MEMORY_EDIT_ENABLED", "true")
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the caller deletes with a foreign tenant scope.
    response = client.request(
        "DELETE",
        "/v1/memories/00000000-0000-0000-0000-0000000000aa",
        headers=_auth_header(),
        json={"scope": _scope_payload(tenant_id="evil")},
    )

    # Then: the request is rejected before the backend runs.
    assert response.status_code == 403
    assert backend.memory_delete_requests == []


def test_diagnostics_returns_safe_readiness_details() -> None:
    # Given: an authenticated operator and available backend.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: diagnostics are requested.
    response = client.get("/v1/diagnostics", headers=_auth_header())

    # Then: tenant, non-secret config, and backend readiness are returned.
    assert response.status_code == 200
    assert response.json() == {
        "tenant_id": "bromigos",
        "config": {
            "neo4j_uri": "bolt://neo4j.neo4j.svc.cluster.local:7687",
            "neo4j_username": "neo4j",
            "litellm_base_url": "http://litellm.litellm.svc.cluster.local:4000/v1",
            "gnosis_llm": "openai/gemma4",
            "gnosis_embedding": "local-qwen3-embedding-0.6b",
            "gnosis_embedding_dimensions": 1024,
            "gnosis_audit_read": False,
            "gnosis_conversation_ttl_days": None,
            "gnosis_write_mode": "sync",
            "gnosis_max_pending": 200,
            "gnosis_fact_deduplication_enabled": True,
            "gnosis_trace_embedding_enabled": True,
            "gnosis_extract_entities_enabled": False,
            "gnosis_extract_relations_enabled": False,
            "gnosis_extraction_preview_enabled": False,
            "gnosis_extraction_batch_size": 25,
            "gnosis_extraction_max_concurrency": 1,
            "gnosis_extraction_chunk_size": 4000,
            "gnosis_extraction_chunk_overlap": 200,
            "gnosis_fact_extraction_enabled": False,
            "gnosis_fact_extraction_model": "",
            "gnosis_fact_extraction_context_turns": 10,
            "gnosis_ocr_enabled": False,
            "gnosis_ocr_model": "",
            "gnosis_ocr_max_image_bytes": 0,
            "gnosis_rustfs_enabled": False,
            "gnosis_rustfs_bucket": "",
            "gnosis_rustfs_prefix": "",
            "gnosis_rustfs_endpoint": "",
            "gnosis_rustfs_retention_days": None,
            "gnosis_prompt_entities_enabled": False,
            "gnosis_prompt_preferences_enabled": False,
            "gnosis_prompt_reasoning_enabled": False,
            "gnosis_consolidation_schedule_enabled": False,
        },
        "backend": {"graph": "ready", "schema": "ready", "buffer": "ready"},
    }


def test_diagnostics_reflects_memory_feature_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: extraction discovery is enabled in environment policy only.
    monkeypatch.setenv("GNOSIS_EXTRACT_ENTITIES_ENABLED", "true")
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )
    client = TestClient(create_app(settings_factory=Settings, backend=backend))

    # When: diagnostics are requested and a message is written.
    diagnostics = client.get("/v1/diagnostics", headers=_auth_header())
    write = client.post(
        "/v1/messages",
        headers=_auth_header(),
        json=_message_payload(),
    )

    # Then: diagnostics show the policy switch, while writes keep extraction off.
    assert diagnostics.status_code == 200
    assert diagnostics.json()["config"]["gnosis_extract_entities_enabled"] is True
    assert write.status_code == 200
    assert fake_client.short_term.messages[0].extract_entities is False


def test_write_message_when_token_is_missing() -> None:
    # Given: an app with protected memory endpoints.
    client = TestClient(create_app(settings_factory=_settings))

    # When: a client writes memory without credentials.
    response = client.post("/v1/messages", json=_message_payload())

    # Then: the API rejects the request.
    assert response.status_code == 401


def test_write_message_when_request_is_scoped() -> None:
    # Given: an app with a fake memory backend.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client writes a scoped message.
    response = client.post(
        "/v1/messages",
        headers=_auth_header(),
        json=_message_payload(),
    )

    # Then: the API accepts and forwards the typed request.
    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    assert backend.messages[0].scope.tenant_id == "bromigos"


def test_write_message_when_scope_is_outside_policy() -> None:
    # Given: an app configured for the bromigos tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller tries to write memory into another tenant.
    response = client.post(
        "/v1/messages",
        headers=_auth_header(),
        json=_message_payload(scope=_scope_payload(tenant_id="evil-corp")),
    )

    # Then: the API rejects the request before reaching the backend.
    assert response.status_code == 403
    assert backend.messages == []


def test_preview_extraction_when_request_is_scoped() -> None:
    # Given: an app with a fake memory backend that supports dry-run previews.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client requests extraction preview for scoped raw text.
    response = client.post(
        "/v1/memory/extraction/preview",
        headers=_auth_header(),
        json=_extraction_preview_payload(),
    )

    # Then: the API returns preview data without using the message write path.
    assert response.status_code == 200
    assert response.json()["metrics"]["documents"] == 1
    assert backend.preview_requests == [
        ExtractionPreviewRequest.model_validate(_extraction_preview_payload()),
    ]
    assert backend.messages == []


def test_preview_extraction_when_scope_is_outside_policy() -> None:
    # Given: an app configured for the bromigos tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller previews extraction for another tenant.
    response = client.post(
        "/v1/memory/extraction/preview",
        headers=_auth_header(),
        json=_extraction_preview_payload(
            scope=_scope_payload(tenant_id="evil-corp"),
        ),
    )

    # Then: the API rejects the request before reaching the backend.
    assert response.status_code == 403
    assert backend.preview_requests == []


def test_preview_extraction_maps_backend_policy_error_to_bad_request() -> None:
    # Given: the backend rejects a preview according to extraction policy.
    preview_error = BackendRequestError(
        "Extraction preview is disabled by service policy.",
    )
    backend = RecordingBackend(
        preview_error=preview_error,
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a scoped preview request reaches the backend.
    response = client.post(
        "/v1/memory/extraction/preview",
        headers=_auth_header(),
        json=_extraction_preview_payload(),
    )

    # Then: the API exposes a client-correctable 400 without accepting the preview.
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Extraction preview is disabled by service policy.",
    }
    assert backend.preview_requests == [
        ExtractionPreviewRequest.model_validate(_extraction_preview_payload()),
    ]


def test_existing_message_write_contract_still_works() -> None:
    # Given: the deployed message write endpoint and response shape.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an existing client writes a scoped mention-memory message.
    response = client.post(
        "/v1/messages",
        headers=_auth_header(),
        json=_message_payload(),
    )

    # Then: the legacy wire contract stays accepted-only and tenant scoped.
    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    assert backend.messages == [MessageWriteRequest.model_validate(_message_payload())]


def test_existing_context_contract_still_works() -> None:
    # Given: the deployed context endpoint and response shape.
    backend = RecordingBackend(context="remembered facts")
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an existing client requests mention-memory context.
    response = client.post(
        "/v1/context",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "what matters?", "limit": 4},
    )

    # Then: the legacy wire contract stays context-only and tenant scoped.
    assert response.status_code == 200
    assert response.json() == {"context": "remembered facts"}
    assert backend.context_requests == [
        ContextRequest.model_validate(
            {"scope": _scope_payload(), "query": "what matters?", "limit": 4},
        ),
    ]


def test_memory_context_returns_labeled_combined_sections() -> None:
    # Given: an app with a backend that can compose official-style memory context.
    backend = RecordingBackend(
        memory_context=MemoryContextResponse(
            sections=[
                MemoryContextSection(
                    source="short_term",
                    content="recent channel chat",
                ),
                MemoryContextSection(
                    source="long_term_preferences_entities",
                    content="### User Preferences\n- concise answers",
                ),
                MemoryContextSection(
                    source="reasoning",
                    content="### Similar Past Tasks\n- prior Discord reply",
                ),
                MemoryContextSection(
                    source="graph",
                    content="Cartman mentioned memory alignment",
                    facts=[{"kind": "message", "id": "message-999"}],
                ),
            ],
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client requests the combined memory context endpoint.
    response = client.post(
        "/v1/memory/context",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "query": "what matters?",
            "max_items": 4,
            "graph_limit": 3,
        },
    )

    # Then: the endpoint returns labeled sections and forwards the typed request.
    assert response.status_code == 200
    assert response.json() == {
        "sections": [
            {"source": "short_term", "content": "recent channel chat", "facts": []},
            {
                "source": "long_term_preferences_entities",
                "content": "### User Preferences\n- concise answers",
                "facts": [],
            },
            {
                "source": "reasoning",
                "content": "### Similar Past Tasks\n- prior Discord reply",
                "facts": [],
            },
            {
                "source": "graph",
                "content": "Cartman mentioned memory alignment",
                "facts": [{"kind": "message", "id": "message-999"}],
            },
        ],
    }
    assert backend.memory_context_requests == [
        MemoryContextRequest.model_validate(
            {
                "scope": _scope_payload(),
                "query": "what matters?",
                "max_items": 4,
                "graph_limit": 3,
            },
        ),
    ]


def test_memory_context_contract_fixture_serializes_enriched_sections() -> None:
    # Given: the shared fixture documents enriched memory-type sections.
    fixture = _MEMORY_CONTEXT_CONTRACT_FIXTURE.read_text()

    # When: gnosis validates and serializes the response contract.
    response = MemoryContextResponse.model_validate_json(fixture)
    serialized_response = MemoryContextResponse.model_validate(
        response.model_dump(mode="json"),
    )

    # Then: all current enriched memory types remain in service order.
    assert [section.memory_type for section in serialized_response.sections] == [
        "short_term",
        "long_term",
        "entities",
        "preferences",
        "dedup_notice",
        "reasoning",
        "similar_traces",
    ]
    assert "Short-term continuity" in serialized_response.sections[0].content
    assert "Durable preference" in serialized_response.sections[3].content
    assert "prior successful pattern" in serialized_response.sections[5].content


def test_memory_context_response_contract_fixture_rejects_missing_content() -> None:
    # Given: a fixture-like section with a required response field missing.
    payload: JsonObject = {
        "sections": [
            {
                "source": "conversation_buffer",
                "memory_type": "short_term",
                "facts": [],
            },
        ],
    }

    # When/Then: contract validation fails clearly at the missing field.
    with pytest.raises(ValidationError, match="content"):
        _ = MemoryContextResponse.model_validate(payload)


def test_memory_context_requires_auth() -> None:
    # Given: the combined memory context endpoint is protected.
    client = TestClient(
        create_app(settings_factory=_settings, backend=RecordingBackend()),
    )

    # When: a client omits credentials.
    response = client.post(
        "/v1/memory/context",
        json={"scope": _scope_payload(), "query": "what matters?"},
    )

    # Then: bearer auth is required before backend access.
    assert response.status_code == 401


def test_memory_context_when_scope_is_outside_policy() -> None:
    # Given: an app configured for the bromigos tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller tries to read context from another tenant.
    response = client.post(
        "/v1/memory/context",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(tenant_id="evil-corp"),
            "query": "what matters?",
        },
    )

    # Then: the API rejects the request before reaching the backend.
    assert response.status_code == 403
    assert backend.memory_context_requests == []


def test_memory_context_rejects_unknown_fields() -> None:
    # Given: the combined memory context request contract forbids extra fields.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client sends an unsupported request field.
    response = client.post(
        "/v1/memory/context",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "query": "what matters?",
            "unsupported": "ignored?",
        },
    )

    # Then: validation rejects it before backend access.
    assert response.status_code == 422
    assert backend.memory_context_requests == []


def test_legacy_context_remains_short_term_only() -> None:
    # Given: the deployed context endpoint still has the legacy backend response.
    backend = RecordingBackend(context="short-term only")
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an existing client requests legacy context.
    response = client.post(
        "/v1/context",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "what matters?", "limit": 4},
    )

    # Then: legacy shape remains context-only and does not call combined context.
    assert response.status_code == 200
    assert response.json() == {"context": "short-term only"}
    assert backend.context_requests == [
        ContextRequest.model_validate(
            {"scope": _scope_payload(), "query": "what matters?", "limit": 4},
        ),
    ]
    assert backend.memory_context_requests == []


def test_legacy_context_signals_deprecation_headers() -> None:
    # Given: the deprecated legacy context endpoint.
    backend = RecordingBackend(context="short-term only")
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an existing client requests legacy context.
    response = client.post(
        "/v1/context",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "what matters?", "limit": 4},
    )

    # Then: the response advertises the deprecation and its successor route.
    assert response.status_code == 200
    assert response.headers["Deprecation"] == "true"
    assert response.headers["Link"] == '</v1/memory/context>; rel="successor-version"'


def test_legacy_context_is_marked_deprecated_in_openapi() -> None:
    # Given: an app exposing both the legacy and successor context routes.
    client = TestClient(
        create_app(settings_factory=_settings, backend=RecordingBackend()),
    )

    # When: the OpenAPI schema is fetched.
    schema = _JSON_OBJECT_ADAPTER.validate_json(client.get("/openapi.json").text)
    paths = TypeAdapter(dict[str, dict[str, JsonObject]]).validate_python(
        schema["paths"],
    )

    # Then: only the legacy route is flagged deprecated.
    assert paths["/v1/context"]["post"]["deprecated"] is True
    assert "deprecated" not in paths["/v1/memory/context"]["post"]


def test_legacy_context_warns_once_per_process(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the deprecated legacy context endpoint on a fresh app.
    backend = RecordingBackend(context="short-term only")
    client = TestClient(create_app(settings_factory=_settings, backend=backend))
    payload = {"scope": _scope_payload(), "query": "what matters?", "limit": 4}

    # When: the legacy endpoint is called repeatedly.
    with caplog.at_level(logging.WARNING, logger="gnosis.main"):
        first = client.post("/v1/context", headers=_auth_header(), json=payload)
        second = client.post("/v1/context", headers=_auth_header(), json=payload)

    # Then: exactly one structured deprecation warning is emitted.
    assert first.status_code == 200
    assert second.status_code == 200
    warnings = [record for record in caplog.records if record.name == "gnosis.main"]
    assert len(warnings) == 1
    assert warnings[0].__dict__["deprecated_route"] == "/v1/context"
    assert warnings[0].__dict__["successor_route"] == "/v1/memory/context"


def test_write_event_when_token_is_missing() -> None:
    # Given: an app with protected event endpoints.
    client = TestClient(create_app(settings_factory=_settings))

    # When: a client writes a client event without credentials.
    response = client.post("/v1/events", json=_client_event_payload())

    # Then: the API rejects the request.
    assert response.status_code == 401


def test_new_event_endpoint_requires_auth() -> None:
    # Given: the structured event endpoint is protected.
    client = TestClient(create_app(settings_factory=_settings))

    # When: a client omits the bearer token.
    response = client.post("/v1/events", json=_client_event_payload())

    # Then: the API rejects the request before persistence.
    assert response.status_code == 401


def test_write_event_when_scope_is_outside_policy() -> None:
    # Given: an app configured for the bromigos tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller tries to write a client event into another tenant.
    response = client.post(
        "/v1/events",
        headers=_auth_header(),
        json=_client_event_payload(scope=_scope_payload(tenant_id="evil-corp")),
    )

    # Then: the API rejects the request before reaching the backend.
    assert response.status_code == 403
    assert backend.events == []


def test_write_event_when_tenant_differs_from_scope_is_rejected() -> None:
    # Given: an event whose authorized scope tenant differs from its write tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))
    payload = _client_event_payload()
    payload["tenant_id"] = "evil-corp"

    # When: the caller submits the event with valid bearer credentials.
    response = client.post("/v1/events", headers=_auth_header(), json=payload)

    # Then: the API rejects the cross-tenant write before persistence.
    assert response.status_code == 403
    assert backend.events == []


def test_write_event_when_backend_returns_duplicate() -> None:
    # Given: an app with a fake memory backend that has already seen the event.
    backend = RecordingBackend(duplicate_event_ids={"discord-message-999"})
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client writes the same event again.
    response = client.post(
        "/v1/events",
        headers=_auth_header(),
        json=_client_event_payload(),
    )

    # Then: the idempotent backend result is passed through unchanged.
    assert response.status_code == 200
    assert response.json() == {
        "event_id": "discord-message-999",
        "status": "duplicate",
        "reason": None,
    }


def test_write_event_when_payload_has_unknown_field() -> None:
    # Given: a structured event request with an undeclared top-level field.
    payload = _client_event_payload()
    payload["extra"] = "reject me"
    client = TestClient(
        create_app(settings_factory=_settings, backend=RecordingBackend()),
    )

    # When: the malformed payload crosses the FastAPI/Pydantic boundary.
    response = client.post("/v1/events", headers=_auth_header(), json=payload)

    # Then: model strictness rejects unknown fields.
    assert response.status_code == 422


def test_write_event_batch_when_one_event_fails_policy() -> None:
    # Given: an app with a fake memory backend.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: one event is valid and one event targets another tenant.
    response = client.post(
        "/v1/events/batch",
        headers=_auth_header(),
        json={
            "events": [
                _client_event_payload(),
                _client_event_payload(
                    event_type="reaction_added",
                    scope=_scope_payload(tenant_id="evil-corp"),
                    subject={"id": "reaction-1", "type": "reaction"},
                ),
            ],
        },
    )

    # Then: the batch returns per-event results instead of a 500.
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "event_id": "discord-message-999",
                "status": "accepted",
                "reason": None,
            },
            {
                "event_id": "discord-message-999",
                "status": "rejected",
                "reason": "scope is not authorized for this token",
            },
        ],
    }
    assert [event.tenant_id for event in backend.events] == ["bromigos"]


def test_write_event_batch_when_event_tenant_differs_from_scope_rejects_event() -> None:
    # Given: a batch with one valid event and one cross-tenant event.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))
    invalid_event = _client_event_payload(
        event_type="reaction_added",
        subject={"id": "reaction-1", "type": "reaction"},
    )
    invalid_event["tenant_id"] = "evil-corp"

    # When: the caller submits the mixed batch.
    response = client.post(
        "/v1/events/batch",
        headers=_auth_header(),
        json={"events": [_client_event_payload(), invalid_event]},
    )

    # Then: only the cross-tenant event is rejected and never reaches the backend.
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "event_id": "discord-message-999",
                "status": "accepted",
                "reason": None,
            },
            {
                "event_id": "discord-message-999",
                "status": "rejected",
                "reason": "event tenant does not match scope tenant",
            },
        ],
    }
    assert [event.tenant_id for event in backend.events] == ["bromigos"]


def test_get_graph_context_when_request_is_scoped() -> None:
    # Given: an app with a fake memory backend.
    backend = RecordingBackend(context="graph facts")
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client requests graph context with scope metadata.
    response = client.post(
        "/v1/graph/context",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "what matters?", "limit": 4},
    )

    # Then: the scoped graph context is returned.
    assert response.status_code == 200
    assert response.json() == {"context": "graph facts", "facts": []}
    assert backend.graph_context_requests[0].scope.agent_id == "pc-principal"


def test_get_graph_context_when_payload_has_unknown_field() -> None:
    # Given: a graph context request with an undeclared field.
    client = TestClient(
        create_app(settings_factory=_settings, backend=RecordingBackend()),
    )

    # When: the malformed payload crosses the API boundary.
    response = client.post(
        "/v1/graph/context",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "query": "what matters?",
            "limit": 4,
            "extra": "reject me",
        },
    )

    # Then: model strictness rejects unknown fields.
    assert response.status_code == 422


def test_sdk_stats_requires_read_operator_token() -> None:
    # Given: SDK stats are configured for read operators only.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a normal memory client token requests SDK stats.
    response = client.request(
        "GET",
        "/v1/memory/stats",
        headers=_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: the token class is rejected before backend access.
    assert response.status_code == 403
    assert backend.sdk_stats_requests == []


def test_sdk_stats_requires_authentication() -> None:
    # Given: SDK stats are configured for authenticated read operators only.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller omits credentials.
    response = client.request(
        "GET",
        "/v1/memory/stats",
        json={"scope": _scope_payload()},
    )

    # Then: authentication fails before backend access.
    assert response.status_code == 401
    assert backend.sdk_stats_requests == []


def test_sdk_stats_rejects_export_operator_token() -> None:
    # Given: SDK stats are limited to read operators.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an export-only operator requests stats.
    response = client.request(
        "GET",
        "/v1/memory/stats",
        headers=_export_operator_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: the wrong operator class is rejected before backend access.
    assert response.status_code == 403
    assert backend.sdk_stats_requests == []


def test_sdk_stats_when_scope_is_outside_policy() -> None:
    # Given: a read operator token scoped to the bromigos tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the operator requests another tenant's stats.
    response = client.post(
        "/v1/sdk/stats",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(tenant_id="evil-corp")},
    )

    # Then: tenant policy is enforced before backend access.
    assert response.status_code == 403
    assert backend.sdk_stats_requests == []


def test_sdk_stats_redacts_sensitive_values() -> None:
    # Given: the backend returns SDK stats that include accidental secrets.
    backend = RecordingBackend(
        sdk_stats=SdkStatsResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            stats={
                "nodes": 7,
                "token": "memory-token-sentinel",
                "nested": {"api_key": "sk-1234567890abcdef"},
            },
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator requests scoped SDK stats.
    response = client.post(
        "/v1/sdk/stats",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: typed stats are returned without leaking secret-looking fields.
    assert response.status_code == 200
    assert response.json() == {
        "scope": _scope_payload(),
        "stats": {
            "nodes": 7,
            "token": "[REDACTED]",
            "nested": {"api_key": "[REDACTED]"},
        },
    }
    assert backend.sdk_stats_requests == [
        SdkStatsRequest(scope=MemoryScope.model_validate(_scope_payload())),
    ]


def test_memory_stats_get_uses_tenant_scope_without_request_body() -> None:
    # Given: the canonical memory stats route is a GET endpoint.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator requests tenant-level stats without a request body.
    response = client.get("/v1/memory/stats", headers=_read_operator_auth_header())

    # Then: the endpoint succeeds without relying on a brittle GET body.
    assert response.status_code == 200
    assert response.json() == {
        "scope": _tenant_scope_payload(),
        "stats": {"nodes": 0},
    }
    assert backend.sdk_stats_requests == [
        SdkStatsRequest(scope=MemoryScope.model_validate(_tenant_scope_payload())),
    ]


def test_sdk_stats_legacy_alias_redacts_sensitive_values() -> None:
    # Given: the legacy SDK stats route remains a compatibility alias.
    backend = RecordingBackend(
        sdk_stats=SdkStatsResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            stats={"token": "memory-token-sentinel"},
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator requests stats through the legacy alias.
    response = client.post(
        "/v1/sdk/stats",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: the alias preserves the same typed and redacted behavior.
    assert response.status_code == 200
    assert response.json() == {
        "scope": _scope_payload(),
        "stats": {"token": "[REDACTED]"},
    }
    assert backend.sdk_stats_requests == [
        SdkStatsRequest(scope=MemoryScope.model_validate(_scope_payload())),
    ]


def test_buffer_flush_requires_admin_operator_token() -> None:
    # Given: buffer flush is an admin-only write-buffer control.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator attempts to flush buffered writes.
    response = client.post(
        "/v1/memory/buffer/flush",
        headers=_read_operator_auth_header(),
    )

    # Then: admin authorization is required before backend access.
    assert response.status_code == 403
    assert backend.buffer_flushes == 0


def test_buffer_flush_returns_status_for_admin_operator() -> None:
    # Given: the backend can flush the SDK write buffer.
    backend = RecordingBackend(
        buffer_flush=BufferFlushResponse(
            flushed=True,
            status=BufferStatus(
                write_mode="buffered",
                max_pending=7,
                pending_writes=0,
                write_errors=0,
                status="ready",
            ),
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an admin operator flushes buffered writes.
    response = client.post(
        "/v1/memory/buffer/flush",
        headers=_admin_operator_auth_header(),
    )

    # Then: the response exposes only counters and non-secret buffer policy.
    assert response.status_code == 200
    assert response.json() == {
        "flushed": True,
        "status": {
            "write_mode": "buffered",
            "max_pending": 7,
            "pending_writes": 0,
            "write_errors": 0,
            "status": "ready",
        },
    }
    assert backend.buffer_flushes == 1


def test_graph_export_requires_export_operator_token() -> None:
    # Given: graph export is configured for export operators only.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read-only operator attempts export.
    response = client.post(
        "/v1/memory/graph/export",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "limit": 10},
    )

    # Then: the wrong operator class is rejected before backend access.
    assert response.status_code == 403
    assert backend.graph_export_requests == []


def test_graph_export_requires_authentication() -> None:
    # Given: graph export is configured for authenticated export operators only.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller omits credentials.
    response = client.post(
        "/v1/memory/graph/export",
        json={"scope": _scope_payload(), "limit": 10},
    )

    # Then: authentication fails before backend access.
    assert response.status_code == 401
    assert backend.graph_export_requests == []


def test_graph_export_validates_max_limit() -> None:
    # Given: graph export enforces a service max limit at the API boundary.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an export operator requests more than the max limit.
    response = client.post(
        "/v1/memory/graph/export",
        headers=_export_operator_auth_header(),
        json={"scope": _scope_payload(), "limit": 1001},
    )

    # Then: validation rejects the request before backend access.
    assert response.status_code == 422
    assert backend.graph_export_requests == []


def test_graph_export_when_scope_is_outside_policy() -> None:
    # Given: an export operator token scoped to the bromigos tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the operator requests another tenant's graph.
    response = client.post(
        "/v1/memory/graph/export",
        headers=_export_operator_auth_header(),
        json={"scope": _scope_payload(tenant_id="evil-corp"), "limit": 10},
    )

    # Then: tenant policy is enforced before backend export.
    assert response.status_code == 403
    assert backend.graph_export_requests == []


def test_graph_export_returns_scoped_redacted_graph() -> None:
    # Given: the backend returns a scoped graph containing secret-looking fields.
    backend = RecordingBackend(
        graph_export=GraphExportResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            nodes=[
                GraphExportNode(
                    id="message-1",
                    labels=["Message"],
                    properties={
                        "tenant_id": "bromigos",
                        "token": "abc123secretTOKEN456",
                    },
                ),
            ],
            relationships=[
                GraphExportRelationship(
                    id="rel-1",
                    type="MENTIONS",
                    from_node="message-1",
                    to_node="entity-1",
                    properties={"api_key": "sk-1234567890abcdef"},
                ),
            ],
            metadata={"limit": 10},
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an export operator requests a scoped graph export.
    response = client.post(
        "/v1/memory/graph/export",
        headers=_export_operator_auth_header(),
        json={"scope": _scope_payload(), "limit": 10},
    )

    # Then: the export is typed, scoped, and redacted.
    assert response.status_code == 200
    assert response.json() == {
        "scope": _scope_payload(),
        "nodes": [
            {
                "id": "message-1",
                "labels": ["Message"],
                "properties": {"tenant_id": "bromigos", "token": "[REDACTED]"},
            },
        ],
        "relationships": [
            {
                "id": "rel-1",
                "type": "MENTIONS",
                "from_node": "message-1",
                "to_node": "entity-1",
                "properties": {"api_key": "[REDACTED]"},
            },
        ],
        "metadata": {"limit": 10},
    }
    assert backend.graph_export_requests == [
        GraphExportRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            limit=10,
        ),
    ]


def test_graph_export_maps_unavailable_sdk_capability() -> None:
    # Given: the backend reports the SDK graph capability is unavailable.
    backend = RecordingBackend(
        graph_export_error=BackendCapabilityUnavailable(
            "SDK graph export is unavailable.",
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an export operator requests graph export.
    response = client.post(
        "/v1/memory/graph/export",
        headers=_export_operator_auth_header(),
        json={"scope": _scope_payload(), "limit": 10},
    )

    # Then: the API returns a deterministic capability-unavailable response.
    assert response.status_code == 501
    assert response.json() == {
        "detail": "capability_unavailable",
        "message": "SDK graph export is unavailable.",
    }
    assert backend.graph_export_requests == [
        GraphExportRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            limit=10,
        ),
    ]


def test_graph_export_legacy_alias_returns_scoped_redacted_graph() -> None:
    # Given: the legacy graph export route remains a compatibility alias.
    backend = RecordingBackend(
        graph_export=GraphExportResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            nodes=[
                GraphExportNode(
                    id="message-1",
                    labels=["Message"],
                    properties={"api_key": "sk-1234567890abcdef"},
                ),
            ],
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an export operator requests graph export through the legacy alias.
    response = client.post(
        "/v1/graph/export",
        headers=_export_operator_auth_header(),
        json={"scope": _scope_payload(), "limit": 10},
    )

    # Then: the alias preserves the same typed and redacted behavior.
    assert response.status_code == 200
    assert response.json() == {
        "scope": _scope_payload(),
        "nodes": [
            {
                "id": "message-1",
                "labels": ["Message"],
                "properties": {"api_key": "[REDACTED]"},
            },
        ],
        "relationships": [],
        "metadata": {},
    }
    assert backend.graph_export_requests == [
        GraphExportRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            limit=10,
        ),
    ]


def test_dedup_stats_requires_read_operator_token() -> None:
    # Given: dedup stats are limited to read operators.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a normal bearer token attempts to read dedup stats.
    response = client.request(
        "GET",
        "/v1/memory/dedup/stats",
        headers=_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: the wrong token class is rejected before backend access.
    assert response.status_code == 403
    assert backend.dedup_stats_requests == []


def test_dedup_stats_returns_redacted_counters() -> None:
    # Given: dedup stats include accidental secret-looking fields.
    backend = RecordingBackend(
        dedup_stats=DedupStatsResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            stats={"pending_reviews": 3, "api_key": "sk-1234567890abcdef"},
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator requests scoped dedup stats.
    response = client.request(
        "GET",
        "/v1/memory/dedup/stats",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: stats are returned through the typed redaction boundary.
    assert response.status_code == 200
    assert response.json() == {
        "scope": _scope_payload(),
        "stats": {"pending_reviews": 3, "api_key": "[REDACTED]"},
    }
    assert backend.dedup_stats_requests == [
        DedupStatsRequest(scope=MemoryScope.model_validate(_scope_payload())),
    ]


def test_dedup_candidates_require_read_operator_and_scope() -> None:
    # Given: candidate dry-runs are read-operator scoped.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: callers use the wrong token class and wrong scope.
    wrong_token = client.post(
        "/v1/memory/dedup/candidates",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "limit": 5},
    )
    wrong_scope = client.post(
        "/v1/memory/dedup/candidates",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(tenant_id="evil-corp"), "limit": 5},
    )

    # Then: policy rejects both before backend access.
    assert wrong_token.status_code == 403
    assert wrong_scope.status_code == 403
    assert backend.dedup_candidate_requests == []


def test_dedup_candidates_return_dry_run_report() -> None:
    # Given: the backend has a candidate dry-run report.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator requests duplicate candidates.
    response = client.post(
        "/v1/memory/dedup/candidates",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "limit": 5},
    )

    # Then: the API returns operation-specific dry-run tokens without applying work.
    assert response.status_code == 200
    assert response.json()["candidates"][0]["reject_dry_run_token"] == "reject-token"
    assert response.json()["candidates"][0]["merge_dry_run_token"] == "merge-token"
    assert response.json()["graph_snapshot_hash"] == "snapshot-1"
    assert backend.dedup_candidate_requests == [
        DedupCandidateRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            limit=5,
        ),
    ]
    assert backend.dedup_apply_requests == []


def test_dedup_apply_requires_admin_operator_token() -> None:
    # Given: apply is limited to admin operators.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator attempts to apply a reject.
    response = client.post(
        "/v1/memory/dedup/apply",
        headers=_read_operator_auth_header(),
        json=_dedup_apply_payload(),
    )

    # Then: admin authorization is required before backend access.
    assert response.status_code == 403
    assert backend.dedup_apply_requests == []


def test_dedup_apply_maps_request_errors_to_bad_request() -> None:
    # Given: backend dry-run validation rejects an apply request.
    backend = RecordingBackend(
        dedup_apply_error=BackendRequestError(
            "Deduplication dry-run token is invalid.",
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an admin operator submits the invalid apply request.
    response = client.post(
        "/v1/memory/dedup/apply",
        headers=_admin_operator_auth_header(),
        json=_dedup_apply_payload(),
    )

    # Then: the API reports a typed 400 without hiding the backend reason.
    assert response.status_code == 400
    assert response.json() == {"detail": "Deduplication dry-run token is invalid."}


def test_dedup_apply_requires_explicit_apply_flag() -> None:
    # Given: the apply contract requires explicit apply=true.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the admin request omits the apply flag.
    payload = _dedup_apply_payload()
    del payload["apply"]
    response = client.post(
        "/v1/memory/dedup/apply",
        headers=_admin_operator_auth_header(),
        json=payload,
    )

    # Then: boundary validation rejects the request before backend access.
    assert response.status_code == 422
    assert backend.dedup_apply_requests == []


def test_dedup_apply_returns_typed_audit_response() -> None:
    # Given: an admin operator has a valid dry-run token and audit reason.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the reject operation is applied.
    response = client.post(
        "/v1/memory/dedup/apply",
        headers=_admin_operator_auth_header(),
        json=_dedup_apply_payload(),
    )

    # Then: the typed apply response includes the operator audit fields.
    assert response.status_code == 200
    assert response.json() == {
        "scope": _scope_payload(),
        "operation": "reject",
        "candidate_id": "dedup-1",
        "candidate_version": 1,
        "applied": True,
        "result": {"rejected": True},
        "audit": {"operator_id": "admin-1", "reason": "not duplicate", "ticket": None},
    }
    assert backend.dedup_apply_requests == [
        DedupApplyRequest.model_validate(_dedup_apply_payload()),
    ]


def test_consolidation_dry_run_requires_read_operator_and_scope() -> None:
    # Given: consolidation dry-runs are read-operator scoped.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: callers use the wrong token class and wrong scope.
    wrong_token = client.post(
        "/v1/memory/consolidation/dry-run",
        headers=_auth_header(),
        json=_consolidation_dry_run_payload(),
    )
    wrong_scope = client.post(
        "/v1/memory/consolidation/dry-run",
        headers=_read_operator_auth_header(),
        json=_consolidation_dry_run_payload(
            scope=_scope_payload(tenant_id="evil-corp"),
        ),
    )

    # Then: policy rejects both before backend access.
    assert wrong_token.status_code == 403
    assert wrong_scope.status_code == 403
    assert backend.consolidation_dry_run_requests == []


def test_consolidation_dry_run_returns_redacted_report_without_apply() -> None:
    # Given: the backend has a consolidation dry-run report with a redacted token.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator requests a consolidation dry-run.
    response = client.post(
        "/v1/memory/consolidation/dry-run",
        headers=_read_operator_auth_header(),
        json=_consolidation_dry_run_payload(),
    )

    # Then: the API returns a dry-run report and does not apply consolidation.
    assert response.status_code == 200
    assert response.json() == {
        "scope": _scope_payload(),
        "operation": "dedupe_entities",
        "dry_run": True,
        "report": {"kind": "dedupe_entities", "token": "[REDACTED]"},
        "graph_snapshot_hash": "consolidation-snapshot-1",
        "dry_run_token": "consolidation-token",
        "expires_at": "2026-06-29T00:15:00+00:00",
    }
    assert backend.consolidation_dry_run_requests == [
        ConsolidationDryRunRequest.model_validate(_consolidation_dry_run_payload()),
    ]
    assert backend.consolidation_apply_requests == []


def test_consolidation_apply_requires_admin_operator_token() -> None:
    # Given: consolidation apply is limited to admin operators.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator attempts to apply consolidation.
    response = client.post(
        "/v1/memory/consolidation/apply",
        headers=_read_operator_auth_header(),
        json=_consolidation_apply_payload(),
    )

    # Then: admin authorization is required before backend access.
    assert response.status_code == 403
    assert backend.consolidation_apply_requests == []


def test_consolidation_apply_maps_request_errors_to_bad_request() -> None:
    # Given: backend dry-run validation rejects a consolidation apply request.
    backend = RecordingBackend(
        consolidation_apply_error=BackendRequestError(
            "Consolidation dry-run token is invalid.",
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: an admin operator submits the invalid apply request.
    response = client.post(
        "/v1/memory/consolidation/apply",
        headers=_admin_operator_auth_header(),
        json=_consolidation_apply_payload(),
    )

    # Then: the API reports a typed 400 without hiding the backend reason.
    assert response.status_code == 400
    assert response.json() == {"detail": "Consolidation dry-run token is invalid."}


def test_consolidation_apply_returns_typed_audit_response() -> None:
    # Given: an admin operator has a valid consolidation dry-run token.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the consolidation operation is applied.
    response = client.post(
        "/v1/memory/consolidation/apply",
        headers=_admin_operator_auth_header(),
        json=_consolidation_apply_payload(),
    )

    # Then: the typed apply response includes the operator audit fields.
    assert response.status_code == 200
    assert response.json() == {
        "scope": _scope_payload(),
        "operation": "dedupe_entities",
        "applied": True,
        "result": {"merged": 2},
        "audit": {"operator_id": "admin-1", "reason": "reviewed", "ticket": None},
    }
    assert backend.consolidation_apply_requests == [
        ConsolidationApplyRequest.model_validate(_consolidation_apply_payload()),
    ]


def test_entity_search_requires_read_operator_and_scope() -> None:
    # Given: entity search is limited to scoped read operators.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a normal bearer token attempts entity search.
    wrong_token = client.post(
        "/v1/memory/entities/search",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "Cartman"},
    )
    wrong_scope = client.post(
        "/v1/memory/entities/search",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(tenant_id="evil-corp"), "query": "Cartman"},
    )

    # Then: token class and tenant scope are enforced before backend access.
    assert wrong_token.status_code == 403
    assert wrong_scope.status_code == 403
    assert backend.entity_search_requests == []


def test_scoped_search_routes_return_typed_results() -> None:
    # Given: the backend has scoped long-term records for all search surfaces.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator searches entities, facts, and preferences.
    entity_response = client.post(
        "/v1/memory/entities/search",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "query": "Cartman", "limit": 5},
    )
    fact_response = client.post(
        "/v1/memory/facts/search",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "query": "snacks", "limit": 5},
    )
    preference_response = client.post(
        "/v1/memory/preferences/search",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "query": "concise", "category": "style"},
    )

    # Then: typed responses are returned and backend receives scoped requests.
    assert entity_response.status_code == 200
    assert entity_response.json()["entities"] == [
        {
            "id": None,
            "name": "Cartman",
            "type": "PERSON",
            "subtype": None,
            "description": None,
            "confidence": 1.0,
            "aliases": [],
            "attributes": {},
            "metadata": {},
            "provenance": None,
        },
    ]
    assert fact_response.status_code == 200
    assert fact_response.json()["facts"] == [
        {
            "id": None,
            "subject": "Cartman",
            "predicate": "prefers",
            "object": "snacks",
            "confidence": 1.0,
            "metadata": {},
            "provenance": None,
        },
    ]
    assert preference_response.status_code == 200
    assert preference_response.json()["preferences"] == [
        {
            "id": None,
            "category": "style",
            "preference": "concise answers",
            "context": None,
            "confidence": 1.0,
            "user_identifier": None,
            "metadata": {},
            "provenance": None,
        },
    ]
    assert backend.entity_search_requests == [
        EntitySearchRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            query="Cartman",
            limit=5,
        ),
    ]
    assert backend.fact_search_requests == [
        FactSearchRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            query="snacks",
            limit=5,
        ),
    ]
    assert backend.preference_search_requests == [
        PreferenceSearchRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            query="concise",
            category="style",
        ),
    ]


def test_write_routes_require_write_operator_token() -> None:
    # Given: direct long-term writes are limited to write operators.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read-only operator attempts a fact write.
    response = client.post(
        "/v1/memory/facts",
        headers=_read_operator_auth_header(),
        json={
            "scope": _scope_payload(),
            "subject": "Cartman",
            "predicate": "prefers",
            "object": "snacks",
        },
    )

    # Then: the wrong operator class is rejected before backend access.
    assert response.status_code == 403
    assert backend.fact_writes == []


def test_scoped_write_routes_return_typed_records() -> None:
    # Given: direct long-term write APIs are available to scoped write operators.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a write operator adds an entity, fact, and preference.
    entity_response = client.post(
        "/v1/memory/entities",
        headers=_write_operator_auth_header(),
        json={"scope": _scope_payload(), "name": "Cartman", "type": "PERSON"},
    )
    fact_response = client.post(
        "/v1/memory/facts",
        headers=_write_operator_auth_header(),
        json={
            "scope": _scope_payload(),
            "subject": "Cartman",
            "predicate": "prefers",
            "object": "snacks",
            "confidence": 0.8,
        },
    )
    preference_response = client.post(
        "/v1/memory/preferences",
        headers=_write_operator_auth_header(),
        json={
            "scope": _scope_payload(),
            "category": "style",
            "preference": "concise answers",
        },
    )

    # Then: each route returns a typed record and records the scoped write request.
    assert entity_response.status_code == 200
    assert entity_response.json() == {
        "id": None,
        "name": "Cartman",
        "type": "PERSON",
        "subtype": None,
        "description": None,
        "confidence": 1.0,
        "aliases": [],
        "attributes": {},
        "metadata": {},
        "provenance": None,
    }
    assert fact_response.status_code == 200
    assert fact_response.json() == {
        "id": None,
        "subject": "Cartman",
        "predicate": "prefers",
        "object": "snacks",
        "confidence": 0.8,
        "metadata": {},
        "provenance": None,
    }
    assert preference_response.status_code == 200
    assert preference_response.json() == {
        "id": None,
        "category": "style",
        "preference": "concise answers",
        "context": None,
        "confidence": 1.0,
        "user_identifier": None,
        "metadata": {},
        "provenance": None,
    }
    assert backend.entity_writes == [
        EntityWriteRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            name="Cartman",
            type="PERSON",
        ),
    ]
    assert backend.fact_writes == [
        FactWriteRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            subject="Cartman",
            predicate="prefers",
            object="snacks",
            confidence=0.8,
        ),
    ]
    assert backend.preference_writes == [
        PreferenceWriteRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            category="style",
            preference="concise answers",
        ),
    ]


def test_skill_endpoints_when_requests_are_scoped() -> None:
    # Given: an app with a fake skill-capable backend.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client lists, proposes, and records skill usage.
    list_response = client.post(
        "/v1/skills",
        headers=_auth_header(),
        json={"tenant_id": "bromigos", "agent_id": "pc-principal"},
    )
    proposal_response = client.post(
        "/v1/skills/proposals",
        headers=_auth_header(),
        json=_skill_proposal_payload(),
    )
    usage_response = client.post(
        "/v1/skills/usage",
        headers=_auth_header(),
        json=_skill_usage_payload(),
    )

    # Then: each endpoint returns the deployed typed success contract.
    assert list_response.status_code == 200
    assert list_response.json() == {
        "skills": [
            {
                "skill_id": "skill-1",
                "tenant_id": "bromigos",
                "agent_id": "pc-principal",
                "name": "Summarize",
                "description": "Summarize channels",
                "status": "approved",
                "scope": "agent_shared",
                "metadata": {"reviewed": True},
            },
        ],
    }
    assert proposal_response.status_code == 200
    assert proposal_response.json() == _skill_proposal_payload()
    assert usage_response.status_code == 200
    assert usage_response.json() == {"accepted": True}


def test_skill_endpoint_when_token_is_missing() -> None:
    # Given: protected skill endpoints.
    client = TestClient(
        create_app(settings_factory=_settings, backend=RecordingBackend()),
    )

    # When: a client lists skills without credentials.
    response = client.post(
        "/v1/skills",
        json={"tenant_id": "bromigos", "agent_id": "pc-principal"},
    )

    # Then: bearer auth is required.
    assert response.status_code == 401


def test_skill_endpoint_when_tenant_is_outside_policy() -> None:
    # Given: an app configured for the bromigos tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a caller lists skills for another tenant.
    response = client.post(
        "/v1/skills",
        headers=_auth_header(),
        json={"tenant_id": "evil-corp", "agent_id": "pc-principal"},
    )

    # Then: the API rejects the request before reaching the backend.
    assert response.status_code == 403
    assert backend.skill_list_requests == []


@pytest.mark.parametrize(
    ("path", "payload", "backend_collection"),
    [
        (
            "/v1/skills",
            {"tenant_id": "evil-corp", "agent_id": "pc-principal"},
            "skill_list_requests",
        ),
        (
            "/v1/skills/proposals",
            {
                "proposal_id": "proposal-1",
                "tenant_id": "evil-corp",
                "agent_id": "pc-principal",
                "proposed_by": "789",
                "name": "Summarize",
                "description": "Summarize channels",
                "scope": "agent_shared",
                "metadata": {"source": "test"},
            },
            "skill_proposals",
        ),
        (
            "/v1/skills/usage",
            {
                "skill_id": "skill-1",
                "tenant_id": "evil-corp",
                "agent_id": "pc-principal",
                "used_by": "789",
                "used_at": "2026-06-27T01:02:05Z",
                "scope": "agent_shared",
                "metadata": {"outcome": "ok"},
            },
            "skill_usages",
        ),
    ],
)
def test_skill_endpoints_reject_cross_tenant_requests(
    path: str,
    payload: dict[str, object],
    backend_collection: str,
) -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(path, headers=_auth_header(), json=payload)

    assert response.status_code == 403
    assert getattr(backend, backend_collection) == []


def test_skill_endpoint_when_payload_has_unknown_field() -> None:
    # Given: a skill proposal with an undeclared field.
    payload = _skill_proposal_payload()
    payload["unknown"] = "reject me"
    client = TestClient(
        create_app(settings_factory=_settings, backend=RecordingBackend()),
    )

    # When: the malformed payload crosses the API boundary.
    response = client.post(
        "/v1/skills/proposals",
        headers=_auth_header(),
        json=payload,
    )

    # Then: model strictness rejects unknown fields.
    assert response.status_code == 422


def test_skills_are_not_returned_as_reasoning_context() -> None:
    backend = RecordingBackend(
        reasoning_context=ReasoningContextResponse(
            context="reasoning trace summary",
            traces=[
                {
                    "trace_id": "trace-1",
                    "task": "discord_reply",
                    "metadata": {"skill_name": "Summarize"},
                },
            ],
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/context",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "what reasoning applies?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "context": "reasoning trace summary",
        "traces": [
            {
                "trace_id": "trace-1",
                "task": "discord_reply",
                "metadata": {"skill_name": "Summarize"},
            },
        ],
    }
    assert backend.reasoning_context_requests == [
        ReasoningContextRequest.model_validate(
            {"scope": _scope_payload(), "query": "what reasoning applies?"},
        ),
    ]
    assert backend.skill_list_requests == []


def test_reasoning_context_redacts_inert_sensitive_placeholders() -> None:
    backend = RecordingBackend(
        reasoning_context=ReasoningContextResponse(
            context="internal note placeholder value",
            traces=[
                {
                    "trace_id": "trace-1",
                    "metadata": {"secret": "placeholder-value"},
                    "steps": [
                        {
                            "step_id": "step-1",
                            "observation": "found placeholder value",
                        },
                    ],
                },
            ],
        ),
    )
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/context",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "redaction check"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "context": "internal note placeholder value",
        "traces": [
            {
                "trace_id": "trace-1",
                "metadata": {"secret": "[REDACTED]"},
                "steps": [
                    {"step_id": "step-1", "observation": "found placeholder value"},
                ],
            },
        ],
    }


def test_reasoning_context_rejects_scope_outside_policy() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/context",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(tenant_id="evil-corp"),
            "query": "what reasoning applies?",
        },
    )

    assert response.status_code == 403
    assert backend.reasoning_context_requests == []


def test_reasoning_context_rejects_unknown_fields() -> None:
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    response = client.post(
        "/v1/reasoning/context",
        headers=_auth_header(),
        json={
            "scope": _scope_payload(),
            "query": "what reasoning applies?",
            "unexpected": "reject me",
        },
    )

    assert response.status_code == 422
    assert backend.reasoning_context_requests == []


def test_reasoning_trace_reads_require_read_operator_token() -> None:
    # Given: reasoning trace reads expose operator diagnostics only.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a normal bearer token attempts to list reasoning traces.
    response = client.post(
        "/v1/reasoning/traces/list",
        headers=_auth_header(),
        json={"scope": _scope_payload()},
    )

    # Then: the wrong token class is rejected before backend access.
    assert response.status_code == 403
    assert backend.reasoning_trace_list_requests == []


def test_reasoning_trace_reads_reject_scope_outside_policy() -> None:
    # Given: a read operator token scoped to the configured tenant.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the operator requests another tenant's reasoning traces.
    response = client.post(
        "/v1/reasoning/traces/list",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(tenant_id="evil-corp")},
    )

    # Then: tenant policy rejects the request before backend access.
    assert response.status_code == 403
    assert backend.reasoning_trace_list_requests == []


def test_reasoning_trace_list_returns_operator_read_contract() -> None:
    # Given: the backend has scoped trace summaries.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator lists traces.
    response = client.post(
        "/v1/reasoning/traces/list",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "success_only": True, "limit": 10},
    )

    # Then: typed summaries are returned and the request is recorded.
    assert response.status_code == 200
    assert response.json()["traces"] == [
        {
            "trace_id": "trace-1",
            "session_id": "guild:123:channel:456",
            "task": "discord_reply",
            "outcome": None,
            "success": True,
            "started_at": None,
            "completed_at": None,
            "metadata": {"channel_id": "456"},
        },
    ]
    assert backend.reasoning_trace_list_requests == [
        ReasoningTraceListRequest.model_validate(
            {"scope": _scope_payload(), "success_only": True, "limit": 10},
        ),
    ]


def test_reasoning_trace_detail_rejects_path_body_mismatch() -> None:
    # Given: trace detail reads require path/body consistency.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: the path trace id differs from the typed body.
    response = client.post(
        "/v1/reasoning/traces/other-trace/detail",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "trace_id": "trace-1"},
    )

    # Then: the request is rejected before backend access.
    assert response.status_code == 400
    assert backend.reasoning_trace_detail_requests == []


def test_reasoning_trace_detail_returns_steps_without_hidden_thoughts() -> None:
    # Given: the backend response contains only public reasoning step fields.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator requests trace detail.
    response = client.post(
        "/v1/reasoning/traces/trace-1/detail",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "trace_id": "trace-1"},
    )

    # Then: steps are exposed without chain-of-thought or embeddings.
    assert response.status_code == 200
    payload = _JSON_OBJECT_ADAPTER.validate_python(response.json())
    trace = _JSON_OBJECT_ADAPTER.validate_python(payload["trace"])
    steps = _JSON_OBJECTS_ADAPTER.validate_python(payload["steps"])
    assert trace["trace_id"] == "trace-1"
    assert steps == [
        {
            "step_id": "step-1",
            "trace_id": "trace-1",
            "step_number": 1,
            "action": "get_memory_context",
            "observation": "combined memory returned",
            "tool_calls": [],
            "metadata": {"safe": "kept"},
        },
    ]
    assert "thought" not in str(payload).casefold()
    assert "embedding" not in str(payload).casefold()


def test_reasoning_similarity_search_and_stats_use_read_operator() -> None:
    # Given: the backend has read-operator reasoning search/stat surfaces.
    backend = RecordingBackend()
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a read operator calls the search and stats routes.
    similar = client.post(
        "/v1/reasoning/traces/similar",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "task": "discord reply"},
    )
    steps = client.post(
        "/v1/reasoning/steps/search",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "query": "lookup"},
    )
    stats = client.post(
        "/v1/reasoning/tools/stats",
        headers=_read_operator_auth_header(),
        json={"scope": _scope_payload(), "tool_name": "memory.get_context"},
    )

    # Then: each route returns its typed operator read response.
    assert similar.status_code == 200
    assert similar.json()["traces"][0]["trace_id"] == "trace-1"
    assert steps.status_code == 200
    assert steps.json()["steps"][0]["step_id"] == "step-1"
    assert stats.status_code == 200
    assert stats.json()["tools"][0]["name"] == "memory.get_context"
    assert backend.reasoning_similar_trace_requests[0].task == "discord reply"
    assert backend.reasoning_step_search_requests[0].query == "lookup"
    assert backend.reasoning_tool_stats_requests[0].tool_name == "memory.get_context"


def test_get_context_when_request_is_scoped() -> None:
    # Given: an app with a fake memory backend.
    backend = RecordingBackend(context="remembered facts")
    client = TestClient(create_app(settings_factory=_settings, backend=backend))

    # When: a client requests context with scope metadata.
    response = client.post(
        "/v1/context",
        headers=_auth_header(),
        json={"scope": _scope_payload(), "query": "what matters?", "limit": 4},
    )

    # Then: the scoped context is returned.
    assert response.status_code == 200
    assert response.json() == {"context": "remembered facts"}
    assert backend.context_requests[0].scope.agent_id == "pc-principal"


@pytest.mark.anyio
async def test_neo4j_backend_writes_short_and_long_term_memory() -> None:
    # Given: a Neo4j backend wired to a fake memory client.
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
    )

    # When: a scoped user message is written.
    response = await backend.add_message(
        MessageWriteRequest.model_validate(_message_payload()),
    )

    # Then: short-term conversation and long-term fact stores both receive scope.
    assert response.accepted is True
    assert fake_client.short_term.messages[0].user_identifier == (
        "bromigos:discord:channel:pc-principal:789"
    )
    assert fake_client.long_term.facts[0].metadata["session_id"] == (
        "guild:123:channel:456"
    )


@pytest.mark.anyio
async def test_neo4j_backend_context_uses_scoped_short_term_only() -> None:
    # Given: a Neo4j backend wired to a fake memory client.
    fake_client = RecordingMemoryClient(context="scoped short-term context")
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
    )

    # When: scoped context is requested.
    response = await backend.get_context(
        ContextRequest.model_validate(
            {"scope": _scope_payload(), "query": "what matters?", "limit": 4},
        ),
    )

    # Then: retrieval is constrained to the scoped session metadata filter.
    assert response.context == "scoped short-term context"
    assert (
        fake_client.short_term.context_queries[0].metadata_filters["user_id"] == "789"
    )
    assert fake_client.long_term.context_queries == []


@pytest.mark.anyio
async def test_neo4j_backend_legacy_context_delegates_to_memory_context() -> None:
    # Given: identical fake memory clients behind the legacy and successor paths.
    legacy_client = RecordingMemoryClient(context="scoped short-term context")
    successor_client = RecordingMemoryClient(context="scoped short-term context")
    legacy_backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(legacy_client),
    )
    successor_backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(successor_client),
    )

    # When: a legacy request and its equivalent combined request are issued.
    legacy_response = await legacy_backend.get_context(
        ContextRequest.model_validate(
            {"scope": _scope_payload(), "query": "what matters?", "limit": 4},
        ),
    )
    successor_response = await successor_backend.get_memory_context(
        MemoryContextRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            query="what matters?",
            include_long_term=False,
            include_reasoning=False,
            include_graph=False,
            max_items=4,
        ),
    )

    # Then: both paths issue the same short-term backend call and content.
    assert (
        legacy_client.short_term.context_queries
        == successor_client.short_term.context_queries
    )
    assert legacy_response.context == "scoped short-term context"
    assert successor_response.sections == [
        MemoryContextSection(
            source="short_term",
            content="scoped short-term context",
        ),
    ]


@pytest.mark.anyio
async def test_neo4j_backend_memory_context_uses_graph_context_path() -> None:
    # Given: combined context is backed by a fake graph store with structured facts.
    fake_client = RecordingMemoryClient()
    graph_store = RecordingGraphStore(
        context="message message-999: visible graph note",
        facts=[
            {
                "id": "tenant:bromigos:message:message-999",
                "type": "message",
                "scope": "channel",
                "summary": "message message-999: visible graph note",
                "deleted": False,
            },
        ],
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=graph_store,
    )

    # When: graph-enabled combined context is requested.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            query="visible graph note",
            include_short_term=False,
            include_long_term=False,
            include_reasoning=False,
            graph_limit=2,
        ),
    )

    # Then: the same GraphContextRequest path feeds the combined graph section.
    assert response.sections == [
        MemoryContextSection(
            source="graph",
            content="message message-999: visible graph note",
            facts=[
                {
                    "id": "tenant:bromigos:message:message-999",
                    "type": "message",
                    "scope": "channel",
                    "summary": "message message-999: visible graph note",
                    "deleted": False,
                },
            ],
        ),
    ]
    assert graph_store.context_requests == [
        GraphContextRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            query="visible graph note",
            limit=2,
        ),
    ]


@pytest.mark.anyio
async def test_neo4j_backend_memory_context_dedupes_graph_facts() -> None:
    # Given: a local Fact already represents one graph node summary.
    fact: JsonObject = {
        "subject": "tenant:bromigos:message:message-999",
        "predicate": "discord.message_created",
        "object": "message message-999: duplicate graph note",
        "metadata": {
            "tenant_id": "bromigos",
            "agent_id": "pc-principal",
            "session_id": "guild:123:channel:456",
            "user_id": "789",
            "visibility": "channel",
            "guild_id": "123",
            "channel_id": "456",
        },
    }
    fake_client = RecordingMemoryClient(query=RecordingQuery(rows=[{"f": fact}]))
    graph_store = RecordingGraphStore(
        context=(
            "message message-999: duplicate graph note\n"
            "message message-1000: unique graph note"
        ),
        facts=[
            {
                "id": "tenant:bromigos:message:message-999",
                "type": "message",
                "scope": "channel",
                "summary": "message message-999: duplicate graph note",
                "deleted": False,
            },
            {
                "id": "tenant:bromigos:message:message-1000",
                "type": "message",
                "scope": "channel",
                "summary": "message message-1000: unique graph note",
                "deleted": False,
            },
        ],
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(gnosis_prompt_entities_enabled=True),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=graph_store,
    )

    # When: combined context contains both long-term facts and graph facts.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=MemoryScope.model_validate(_scope_payload()),
            query="graph note",
            include_short_term=False,
            include_reasoning=False,
            graph_limit=4,
        ),
    )

    # Then: the duplicate graph fact is not repeated in the graph section.
    assert [section.source for section in response.sections] == [
        "long_term_facts",
        "long_term_preferences_entities",
        "graph",
    ]
    graph_section = response.sections[2]
    assert "message message-999: duplicate graph note" not in graph_section.content
    assert graph_section.content == "message message-1000: unique graph note"
    assert [fact["id"] for fact in graph_section.facts] == [
        "tenant:bromigos:message:message-1000",
    ]


def test_client_event_model_accepts_discord_message() -> None:
    # Given: a Discord message event with the canonical client envelope.
    payload = _client_event_payload()

    # When: the event crosses the API model boundary.
    event = ClientEvent.model_validate(payload)

    # Then: Pydantic parses strict enums, nested context, and arbitrary payload JSON.
    assert event.source_client == "discord"
    assert event.event_type == ClientEventType.MESSAGE_CREATED
    assert event.discord is not None
    assert event.discord.message_id == "message-999"
    assert event.payload["content"] == "remember this"


def test_client_event_model_rejects_unknown_fields() -> None:
    # Given: a client event containing an undeclared future field.
    payload = _client_event_payload()
    payload["unknown"] = "not allowed"

    # When / Then: the contract rejects the extra field at the boundary.
    with pytest.raises(ValidationError):
        _ = ClientEvent.model_validate(payload)


def test_client_event_model_rejects_malformed_enum() -> None:
    # Given: a client event with an unsupported event type.
    payload = _client_event_payload()
    payload["event_type"] = "message_created_by_typo"

    # When / Then: enum validation rejects the malformed value.
    with pytest.raises(ValidationError):
        _ = ClientEvent.model_validate(payload)


def test_existing_message_and_context_models_still_serialize() -> None:
    # Given: deployed message and context models.
    message = MessageWriteRequest.model_validate(_message_payload())
    context = ContextRequest.model_validate(
        {"scope": _scope_payload(), "query": "what matters?", "limit": 4},
    )

    # When: current clients serialize them.
    message_payload = message.model_dump(mode="json")
    context_payload = context.model_dump(mode="json")

    # Then: existing wire shapes remain unchanged.
    assert message_payload == _message_payload()
    assert context_payload == {
        "scope": _scope_payload(),
        "query": "what matters?",
        "limit": 4,
    }


def test_memory_context_model_accepts_official_style_toggles() -> None:
    payload = {
        "scope": _scope_payload(),
        "query": "what matters?",
        "include_short_term": True,
        "include_long_term": True,
        "include_reasoning": True,
        "include_graph": False,
        "max_items": 8,
        "graph_limit": 3,
    }

    request = MemoryContextRequest.model_validate(payload)
    response = MemoryContextResponse.model_validate(
        {
            "sections": [
                {
                    "source": "short_term",
                    "content": "remembered facts",
                    "facts": [{"kind": "message", "value": "hello"}],
                },
                {
                    "source": "long_term",
                    "content": "user prefers concise answers",
                },
            ],
        },
    )

    assert request.model_dump(mode="json") == payload
    assert response.model_dump(mode="json") == {
        "sections": [
            {
                "source": "short_term",
                "content": "remembered facts",
                "facts": [{"kind": "message", "value": "hello"}],
            },
            {
                "source": "long_term",
                "content": "user prefers concise answers",
                "facts": [],
            },
        ],
    }


def test_reasoning_models_reject_unknown_chain_of_thought_field() -> None:
    start_payload = {
        "scope": _scope_payload(),
        "session_id": "guild:123:channel:456",
        "task": "discord_reply",
        "chain_of_thought": "private internal reasoning",
    }
    step_payload = {
        "scope": _scope_payload(),
        "trace_id": "trace-1",
        "thought": "private internal reasoning",
        "chain_of_thought": "private internal reasoning",
    }
    tool_call_payload = {
        "scope": _scope_payload(),
        "trace_id": "trace-1",
        "step_id": "step-1",
        "tool_name": "memory.get_context",
        "arguments": {"query": "what matters?"},
        "chain_of_thought": "private internal reasoning",
    }

    with pytest.raises(ValidationError):
        _ = ReasoningTraceStartRequest.model_validate(start_payload)
    with pytest.raises(ValidationError):
        _ = ReasoningStepRequest.model_validate(step_payload)
    with pytest.raises(ValidationError):
        _ = ReasoningToolCallRequest.model_validate(tool_call_payload)


def test_reasoning_lifecycle_models_serialize_stable_ids() -> None:
    start_request = ReasoningTraceStartRequest.model_validate(
        {
            "scope": _scope_payload(),
            "session_id": "guild:123:channel:456",
            "task": "discord_reply",
            "metadata": {"channel_id": "456"},
        },
    )
    start_response = ReasoningTraceStartResponse.model_validate(
        {
            "trace_id": "trace-1",
            "session_id": "guild:123:channel:456",
            "task": "discord_reply",
        },
    )
    step_request = ReasoningStepRequest.model_validate(
        {
            "scope": _scope_payload(),
            "trace_id": "trace-1",
            "step_number": 1,
            "action": "get_memory_context",
            "observation": "combined memory returned",
        },
    )
    step_response = ReasoningStepResponse.model_validate(
        {
            "step_id": "step-1",
            "trace_id": "trace-1",
            "step_number": 1,
        },
    )
    tool_call_request = ReasoningToolCallRequest.model_validate(
        {
            "scope": _scope_payload(),
            "trace_id": "trace-1",
            "step_id": "step-1",
            "tool_name": "memory.get_context",
            "arguments": {"query": "what matters?"},
            "status": "success",
            "touched_entities": [
                {"id": "entity-1", "name": "cartman", "type": "user"},
            ],
        },
    )
    tool_call_response = ReasoningToolCallResponse.model_validate(
        {
            "tool_call_id": "tool-call-1",
            "trace_id": "trace-1",
            "step_id": "step-1",
        },
    )
    complete_request = ReasoningTraceCompleteRequest.model_validate(
        {
            "scope": _scope_payload(),
            "trace_id": "trace-1",
            "outcome": "sent reply",
            "success": True,
        },
    )
    complete_response = ReasoningTraceCompleteResponse.model_validate(
        {
            "trace_id": "trace-1",
            "success": True,
        },
    )
    reasoning_request = ReasoningContextRequest.model_validate(
        {
            "scope": _scope_payload(),
            "query": "what matters?",
            "include_traces": True,
            "include_steps": True,
            "include_tool_calls": True,
            "max_items": 8,
        },
    )
    reasoning_response = ReasoningContextResponse.model_validate(
        {"context": "similar trace context", "traces": []},
    )

    assert start_request.model_dump(mode="json") == {
        "scope": _scope_payload(),
        "session_id": "guild:123:channel:456",
        "task": "discord_reply",
        "metadata": {"channel_id": "456"},
        "triggered_by_message_id": None,
        "user_identifier": None,
    }
    assert start_response.model_dump(mode="json") == {
        "trace_id": "trace-1",
        "session_id": "guild:123:channel:456",
        "task": "discord_reply",
    }
    assert step_request.model_dump(mode="json") == {
        "scope": _scope_payload(),
        "trace_id": "trace-1",
        "step_number": 1,
        "action": "get_memory_context",
        "observation": "combined memory returned",
        "metadata": {},
    }
    assert step_response.model_dump(mode="json") == {
        "step_id": "step-1",
        "trace_id": "trace-1",
        "step_number": 1,
    }
    assert tool_call_request.model_dump(mode="json") == {
        "scope": _scope_payload(),
        "trace_id": "trace-1",
        "step_id": "step-1",
        "tool_name": "memory.get_context",
        "arguments": {"query": "what matters?"},
        "result": None,
        "status": "success",
        "duration_ms": None,
        "error": None,
        "message_id": None,
        "touched_entities": [
            {"id": "entity-1", "name": "cartman", "type": "user"},
        ],
    }
    assert tool_call_response.model_dump(mode="json") == {
        "tool_call_id": "tool-call-1",
        "trace_id": "trace-1",
        "step_id": "step-1",
    }
    assert complete_request.model_dump(mode="json") == {
        "scope": _scope_payload(),
        "trace_id": "trace-1",
        "outcome": "sent reply",
        "success": True,
        "metadata": {},
    }
    assert complete_response.model_dump(mode="json") == {
        "trace_id": "trace-1",
        "success": True,
        "outcome": None,
        "completed_at": None,
    }
    assert reasoning_request.model_dump(mode="json") == {
        "scope": _scope_payload(),
        "query": "what matters?",
        "include_traces": True,
        "include_steps": True,
        "include_tool_calls": True,
        "max_items": 8,
    }
    assert reasoning_response.model_dump(mode="json") == {
        "context": "similar trace context",
        "traces": [],
    }


def test_privacy_defaults_match_discord_scope_policy() -> None:
    # Given: representative DM, channel, topology, and skill records.
    dm_event = ClientEvent.model_validate(
        _client_event_payload(scope=_scope_payload(guild_id=None, channel_id=None)),
    )
    channel_event = ClientEvent.model_validate(_client_event_payload())
    topology_event = ClientEvent.model_validate(
        _client_event_payload(
            event_type="channel_updated",
            subject={"id": "456", "type": "channel"},
        ),
    )
    skill = SkillRecord(
        skill_id="skill-1",
        tenant_id="bromigos",
        agent_id="pc-principal",
        name="Summarize channel",
        description="Summarizes visible Discord channel context for review.",
        status=SkillStatus.APPROVED,
    )

    # When / Then: default helpers keep privacy scoped by surface.
    assert default_event_visibility(dm_event) == MemoryVisibility.PRIVATE_USER
    assert default_event_visibility(channel_event) == MemoryVisibility.CHANNEL
    assert default_event_visibility(topology_event) == MemoryVisibility.GUILD
    assert skill.scope == MemoryVisibility.AGENT_SHARED
    assert default_skill_visibility() == MemoryVisibility.AGENT_SHARED
    assert default_skill_visibility(MemoryVisibility.GLOBAL) == MemoryVisibility.GLOBAL


@dataclass(slots=True)
class RecordingBackend:
    context: str = ""
    memory_context: MemoryContextResponse = field(
        default_factory=MemoryContextResponse,
    )
    reasoning_context: ReasoningContextResponse = field(
        default_factory=lambda: ReasoningContextResponse(context="reasoning context"),
    )
    reasoning_trace_list: ReasoningTraceListResponse = field(
        default_factory=lambda: ReasoningTraceListResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            traces=[
                ReasoningTraceSummary(
                    trace_id="trace-1",
                    session_id="guild:123:channel:456",
                    task="discord_reply",
                    success=True,
                    metadata={"channel_id": "456"},
                ),
            ],
        ),
    )
    reasoning_trace_detail: ReasoningTraceDetailResponse = field(
        default_factory=lambda: ReasoningTraceDetailResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            trace=ReasoningTraceSummary(
                trace_id="trace-1",
                session_id="guild:123:channel:456",
                task="discord_reply",
                success=True,
            ),
            steps=[
                ReasoningStepRecord(
                    step_id="step-1",
                    trace_id="trace-1",
                    step_number=1,
                    action="get_memory_context",
                    observation="combined memory returned",
                    metadata={"safe": "kept"},
                ),
            ],
        ),
    )
    reasoning_similar_traces: ReasoningSimilarTracesResponse = field(
        default_factory=lambda: ReasoningSimilarTracesResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            traces=[
                ReasoningTraceSummary(
                    trace_id="trace-1",
                    session_id="guild:123:channel:456",
                    task="discord_reply",
                ),
            ],
        ),
    )
    reasoning_step_search: ReasoningStepSearchResponse = field(
        default_factory=lambda: ReasoningStepSearchResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            steps=[
                ReasoningStepRecord(
                    step_id="step-1",
                    trace_id="trace-1",
                    step_number=1,
                    action="lookup",
                    observation="found relevant context",
                ),
            ],
        ),
    )
    reasoning_tool_stats: ReasoningToolStatsResponse = field(
        default_factory=lambda: ReasoningToolStatsResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            tools=[
                ReasoningToolStatsRecord(
                    name="memory.get_context",
                    total_calls=3,
                    successful_calls=2,
                    failed_calls=1,
                    success_rate=0.67,
                ),
            ],
        ),
    )
    sdk_stats: SdkStatsResponse = field(
        default_factory=lambda: SdkStatsResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            stats={"nodes": 0},
        ),
    )
    dedup_stats: DedupStatsResponse = field(
        default_factory=lambda: DedupStatsResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            stats={"pending_reviews": 0},
        ),
    )
    dedup_candidates: DedupCandidateResponse = field(
        default_factory=lambda: DedupCandidateResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            candidates=[
                DedupCandidate(
                    candidate_id="dedup-1",
                    version=1,
                    source=DedupEntitySnapshot(
                        id="00000000-0000-0000-0000-000000000001",
                        name="Cartman",
                        type="PERSON",
                        metadata={"api_key": "[REDACTED]"},
                    ),
                    target=DedupEntitySnapshot(
                        id="00000000-0000-0000-0000-000000000002",
                        name="Eric Cartman",
                        type="PERSON",
                    ),
                    similarity=0.94,
                    reject_dry_run_token="reject-token",  # noqa: S106
                    merge_dry_run_token="merge-token",  # noqa: S106
                ),
            ],
            graph_snapshot_hash="snapshot-1",
            expires_at="2026-06-29T00:15:00+00:00",
        ),
    )
    dedup_apply: DedupApplyResponse = field(
        default_factory=lambda: DedupApplyResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            operation="reject",
            candidate_id="dedup-1",
            candidate_version=1,
            applied=True,
            result={"rejected": True},
            audit=DedupOperatorAudit(operator_id="admin-1", reason="not duplicate"),
        ),
    )
    consolidation_dry_run: ConsolidationDryRunResponse = field(
        default_factory=lambda: ConsolidationDryRunResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            operation="dedupe_entities",
            dry_run=True,
            report={"kind": "dedupe_entities", "token": "[REDACTED]"},
            graph_snapshot_hash="consolidation-snapshot-1",
            dry_run_token="consolidation-token",  # noqa: S106
            expires_at="2026-06-29T00:15:00+00:00",
        ),
    )
    consolidation_apply: ConsolidationApplyResponse = field(
        default_factory=lambda: ConsolidationApplyResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
            operation="dedupe_entities",
            applied=True,
            result={"merged": 2},
            audit=DedupOperatorAudit(operator_id="admin-1", reason="reviewed"),
        ),
    )
    graph_export: GraphExportResponse = field(
        default_factory=lambda: GraphExportResponse(
            scope=MemoryScope.model_validate(_scope_payload()),
        ),
    )
    buffer_flush: BufferFlushResponse = field(
        default_factory=lambda: BufferFlushResponse(
            flushed=True,
            status=BufferStatus(
                write_mode="sync",
                max_pending=200,
                pending_writes=None,
                write_errors=0,
                status="ready",
            ),
        ),
    )
    entity_search: EntitySearchResponse = field(
        default_factory=lambda: EntitySearchResponse(
            entities=[EntityRecord(name="Cartman", type="PERSON")],
        ),
    )
    fact_search: FactSearchResponse = field(
        default_factory=lambda: FactSearchResponse(
            facts=[
                FactRecord(
                    subject="Cartman",
                    predicate="prefers",
                    object="snacks",
                ),
            ],
        ),
    )
    preference_search: PreferenceSearchResponse = field(
        default_factory=lambda: PreferenceSearchResponse(
            preferences=[
                PreferenceRecord(category="style", preference="concise answers"),
            ],
        ),
    )
    memory_add: MemoryAddResponse = field(
        default_factory=lambda: MemoryAddResponse(
            results=[
                MemoryAddResult(
                    memory_id="00000000-0000-0000-0000-0000000000aa",
                    content="remember this",
                    event="ADD",
                ),
            ],
        ),
    )
    memory_search: MemorySearchResponse = field(
        default_factory=lambda: MemorySearchResponse(
            results=[
                MemoryRecord(
                    memory_id="00000000-0000-0000-0000-0000000000aa",
                    content="remember this",
                    score=0.91,
                    metadata={"topic": "snacks"},
                    created_at="2026-06-27T01:02:03+00:00",
                ),
            ],
        ),
    )
    memory_list: MemoryListResponse = field(
        default_factory=lambda: MemoryListResponse(
            results=[
                MemoryRecord(
                    memory_id="00000000-0000-0000-0000-0000000000aa",
                    content="remember this",
                    metadata={"topic": "snacks"},
                    created_at="2026-06-27T01:02:03+00:00",
                ),
            ],
            total=1,
            page=1,
            page_size=50,
        ),
    )
    backend_available: bool = True
    messages: list[MessageWriteRequest] = field(default_factory=list)
    context_requests: list[ContextRequest] = field(default_factory=list)
    memory_context_requests: list[MemoryContextRequest] = field(default_factory=list)
    memory_add_requests: list[MemoryAddRequest] = field(default_factory=list)
    memory_search_requests: list[MemorySearchRequest] = field(default_factory=list)
    memory_list_requests: list[MemoryListRequest] = field(default_factory=list)
    memory_update_requests: list[tuple[str, MemoryUpdateRequest]] = field(
        default_factory=list,
    )
    memory_delete_requests: list[tuple[str, MemoryDeleteRequest]] = field(
        default_factory=list,
    )
    memory_add_error: BackendRequestError | None = None
    missing_memory_ids: set[str] = field(default_factory=set)
    events: list[ClientEvent] = field(default_factory=list)
    graph_context_requests: list[GraphContextRequest] = field(default_factory=list)
    sdk_stats_requests: list[SdkStatsRequest] = field(default_factory=list)
    dedup_stats_requests: list[DedupStatsRequest] = field(default_factory=list)
    dedup_candidate_requests: list[DedupCandidateRequest] = field(default_factory=list)
    dedup_apply_requests: list[DedupApplyRequest] = field(default_factory=list)
    consolidation_dry_run_requests: list[ConsolidationDryRunRequest] = field(
        default_factory=list,
    )
    consolidation_apply_requests: list[ConsolidationApplyRequest] = field(
        default_factory=list,
    )
    graph_export_requests: list[GraphExportRequest] = field(default_factory=list)
    entity_search_requests: list[EntitySearchRequest] = field(default_factory=list)
    fact_search_requests: list[FactSearchRequest] = field(default_factory=list)
    preference_search_requests: list[PreferenceSearchRequest] = field(
        default_factory=list,
    )
    entity_writes: list[EntityWriteRequest] = field(default_factory=list)
    fact_writes: list[FactWriteRequest] = field(default_factory=list)
    preference_writes: list[PreferenceWriteRequest] = field(default_factory=list)
    trace_starts: list[ReasoningTraceStartRequest] = field(default_factory=list)
    steps: list[ReasoningStepRequest] = field(default_factory=list)
    tool_calls: list[ReasoningToolCallRequest] = field(default_factory=list)
    completions: list[ReasoningTraceCompleteRequest] = field(default_factory=list)
    reasoning_context_requests: list[ReasoningContextRequest] = field(
        default_factory=list,
    )
    reasoning_trace_list_requests: list[ReasoningTraceListRequest] = field(
        default_factory=list,
    )
    reasoning_trace_detail_requests: list[ReasoningTraceDetailRequest] = field(
        default_factory=list,
    )
    reasoning_similar_trace_requests: list[ReasoningSimilarTracesRequest] = field(
        default_factory=list,
    )
    reasoning_step_search_requests: list[ReasoningStepSearchRequest] = field(
        default_factory=list,
    )
    reasoning_tool_stats_requests: list[ReasoningToolStatsRequest] = field(
        default_factory=list,
    )
    preview_requests: list[ExtractionPreviewRequest] = field(default_factory=list)
    preview_error: BackendRequestError | None = None
    dedup_apply_error: BackendRequestError | None = None
    consolidation_apply_error: BackendRequestError | None = None
    graph_export_error: BackendCapabilityUnavailable | None = None
    duplicate_event_ids: set[str] = field(default_factory=set)
    skill_usages: list[SkillUsage] = field(default_factory=list)
    skill_list_requests: list[SkillListRequest] = field(default_factory=list)
    skill_proposals: list[SkillProposal] = field(default_factory=list)
    buffer_flushes: int = 0
    shutdowns: int = 0
    readiness_checks: int = 0

    async def readiness(self) -> BackendReadiness:
        self.readiness_checks += 1
        if self.backend_available:
            return BackendReadiness(graph="ready", schema="ready")
        return BackendReadiness(graph="unavailable", schema="unavailable")

    async def buffer_status(self) -> BufferStatus:
        return self.buffer_flush.status

    async def flush_buffer(self) -> BufferFlushResponse:
        self.buffer_flushes += 1
        return self.buffer_flush

    async def shutdown(self) -> None:
        self.shutdowns += 1

    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse:
        settings = Settings()
        return DiagnosticsResponse(
            tenant_id=settings.gnosis_tenant_id,
            config=DiagnosticsConfig(
                neo4j_uri="bolt://neo4j.neo4j.svc.cluster.local:7687",
                neo4j_username="neo4j",
                litellm_base_url="http://litellm.litellm.svc.cluster.local:4000/v1",
                gnosis_llm="openai/gemma4",
                gnosis_embedding="local-qwen3-embedding-0.6b",
                gnosis_embedding_dimensions=1024,
                gnosis_audit_read=settings.gnosis_audit_read,
                gnosis_conversation_ttl_days=settings.gnosis_conversation_ttl_days,
                gnosis_write_mode=settings.gnosis_write_mode,
                gnosis_max_pending=settings.gnosis_max_pending,
                gnosis_fact_deduplication_enabled=(
                    settings.gnosis_fact_deduplication_enabled
                ),
                gnosis_trace_embedding_enabled=settings.gnosis_trace_embedding_enabled,
                gnosis_extract_entities_enabled=(
                    settings.gnosis_extract_entities_enabled
                ),
                gnosis_extract_relations_enabled=(
                    settings.gnosis_extract_relations_enabled
                ),
                gnosis_extraction_preview_enabled=(
                    settings.gnosis_extraction_preview_enabled
                ),
                gnosis_extraction_batch_size=settings.gnosis_extraction_batch_size,
                gnosis_extraction_max_concurrency=(
                    settings.gnosis_extraction_max_concurrency
                ),
                gnosis_extraction_chunk_size=settings.gnosis_extraction_chunk_size,
                gnosis_extraction_chunk_overlap=(
                    settings.gnosis_extraction_chunk_overlap
                ),
                gnosis_fact_extraction_enabled=(
                    settings.gnosis_fact_extraction_enabled
                ),
                gnosis_fact_extraction_model=settings.gnosis_fact_extraction_model,
                gnosis_fact_extraction_context_turns=(
                    settings.gnosis_fact_extraction_context_turns
                ),
                gnosis_ocr_enabled=settings.gnosis_ocr_enabled,
                gnosis_ocr_model=settings.gnosis_ocr_model,
                gnosis_ocr_max_image_bytes=settings.gnosis_ocr_max_image_bytes,
                gnosis_rustfs_enabled=settings.gnosis_rustfs_enabled,
                gnosis_rustfs_bucket=settings.gnosis_rustfs_bucket,
                gnosis_rustfs_prefix=settings.gnosis_rustfs_prefix,
                gnosis_rustfs_endpoint=settings.gnosis_rustfs_endpoint,
                gnosis_rustfs_retention_days=settings.gnosis_rustfs_retention_days,
                gnosis_prompt_entities_enabled=settings.gnosis_prompt_entities_enabled,
                gnosis_prompt_preferences_enabled=(
                    settings.gnosis_prompt_preferences_enabled
                ),
                gnosis_prompt_reasoning_enabled=settings.gnosis_prompt_reasoning_enabled,
                gnosis_consolidation_schedule_enabled=(
                    settings.gnosis_consolidation_schedule_enabled
                ),
            ),
            backend=readiness,
        )

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        self.messages.append(request)
        return MessageWriteResponse(accepted=True)

    async def preview_extraction(
        self,
        request: ExtractionPreviewRequest,
    ) -> ExtractionPreviewResponse:
        self.preview_requests.append(request)
        if self.preview_error is not None:
            raise self.preview_error
        return ExtractionPreviewResponse(
            candidates=[
                ExtractionCandidate(
                    kind="text_chunk",
                    text=request.raw_text_documents[0].text,
                    source_id=request.raw_text_documents[0].source_id,
                    confidence=1.0,
                ),
            ],
            metrics=ExtractionPreviewMetrics(
                documents=len(request.raw_text_documents),
                chunks=len(request.raw_text_documents),
                ocr_images=len(request.ocr_image_references),
                rustfs_objects=len(request.rustfs_source_references),
                batch_size=25,
                max_concurrency=1,
            ),
            provenance=ExtractionPreviewProvenance(
                source_ids=[
                    document.source_id for document in request.raw_text_documents
                ],
                rustfs_objects=request.rustfs_source_references,
            ),
            extract_entities=request.extract_entities is True,
            extract_relations=request.extract_relations is True,
        )

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        self.context_requests.append(request)
        return ContextResponse(context=self.context)

    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        self.memory_context_requests.append(request)
        return self.memory_context

    async def add_memories(self, request: MemoryAddRequest) -> MemoryAddResponse:
        self.memory_add_requests.append(request)
        if self.memory_add_error is not None:
            raise self.memory_add_error
        return self.memory_add

    async def search_memories(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        self.memory_search_requests.append(request)
        return self.memory_search

    async def list_memories(self, request: MemoryListRequest) -> MemoryListResponse:
        self.memory_list_requests.append(request)
        return self.memory_list.model_copy(
            update={"page": request.page, "page_size": request.page_size},
        )

    async def update_memory(
        self,
        memory_id: str,
        request: MemoryUpdateRequest,
    ) -> MemoryUpdateResponse:
        self.memory_update_requests.append((memory_id, request))
        if memory_id in self.missing_memory_ids:
            raise MemoryNotFoundError
        return MemoryUpdateResponse(
            memory_id=memory_id,
            content=request.content or "remember this",
        )

    async def delete_memory(
        self,
        memory_id: str,
        request: MemoryDeleteRequest,
    ) -> MemoryDeleteResponse:
        self.memory_delete_requests.append((memory_id, request))
        if memory_id in self.missing_memory_ids:
            raise MemoryNotFoundError
        return MemoryDeleteResponse(memory_id=memory_id)

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        self.events.append(event)
        if event.event_id in self.duplicate_event_ids:
            return EventIngestResult(
                event_id=event.event_id,
                status=EventIngestStatus.DUPLICATE,
            )
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def ingest_events(
        self,
        request: ClientEventBatchRequest,
    ) -> ClientEventBatchResponse:
        results = [await self.ingest_event(event) for event in request.events]
        return ClientEventBatchResponse(results=results)

    async def get_graph_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse:
        self.graph_context_requests.append(request)
        return GraphContextResponse(context=self.context)

    async def get_sdk_stats(self, request: SdkStatsRequest) -> SdkStatsResponse:
        self.sdk_stats_requests.append(request)
        return self.sdk_stats.model_copy(update={"scope": request.scope})

    async def get_dedup_stats(
        self,
        request: DedupStatsRequest,
    ) -> DedupStatsResponse:
        self.dedup_stats_requests.append(request)
        return self.dedup_stats

    async def find_dedup_candidates(
        self,
        request: DedupCandidateRequest,
    ) -> DedupCandidateResponse:
        self.dedup_candidate_requests.append(request)
        return self.dedup_candidates

    async def apply_dedup_candidate(
        self,
        request: DedupApplyRequest,
    ) -> DedupApplyResponse:
        self.dedup_apply_requests.append(request)
        if self.dedup_apply_error is not None:
            raise self.dedup_apply_error
        return self.dedup_apply

    async def dry_run_consolidation(
        self,
        request: ConsolidationDryRunRequest,
    ) -> ConsolidationDryRunResponse:
        self.consolidation_dry_run_requests.append(request)
        return self.consolidation_dry_run

    async def apply_consolidation(
        self,
        request: ConsolidationApplyRequest,
    ) -> ConsolidationApplyResponse:
        self.consolidation_apply_requests.append(request)
        if self.consolidation_apply_error is not None:
            raise self.consolidation_apply_error
        return self.consolidation_apply

    async def export_graph(self, request: GraphExportRequest) -> GraphExportResponse:
        self.graph_export_requests.append(request)
        if self.graph_export_error is not None:
            raise self.graph_export_error
        return self.graph_export

    async def search_entities(
        self,
        request: EntitySearchRequest,
    ) -> EntitySearchResponse:
        self.entity_search_requests.append(request)
        return self.entity_search

    async def search_facts(self, request: FactSearchRequest) -> FactSearchResponse:
        self.fact_search_requests.append(request)
        return self.fact_search

    async def search_preferences(
        self,
        request: PreferenceSearchRequest,
    ) -> PreferenceSearchResponse:
        self.preference_search_requests.append(request)
        return self.preference_search

    async def add_entity(self, request: EntityWriteRequest) -> EntityRecord:
        self.entity_writes.append(request)
        return EntityRecord(
            name=request.name,
            type=request.type,
            subtype=request.subtype,
            description=request.description,
            confidence=request.confidence,
            aliases=request.aliases,
            attributes=request.attributes,
            metadata=request.metadata,
            provenance=request.provenance,
        )

    async def add_fact(self, request: FactWriteRequest) -> FactRecord:
        self.fact_writes.append(request)
        return FactRecord(
            subject=request.subject,
            predicate=request.predicate,
            object=request.object,
            confidence=request.confidence,
            metadata=request.metadata,
            provenance=request.provenance,
        )

    async def add_preference(
        self,
        request: PreferenceWriteRequest,
    ) -> PreferenceRecord:
        self.preference_writes.append(request)
        return PreferenceRecord(
            category=request.category,
            preference=request.preference,
            context=request.context,
            confidence=request.confidence,
            user_identifier=request.user_identifier,
            metadata=request.metadata,
            provenance=request.provenance,
        )

    async def list_skills(self, request: SkillListRequest) -> SkillListResponse:
        self.skill_list_requests.append(request)
        return SkillListResponse(
            skills=[
                SkillRecord(
                    skill_id="skill-1",
                    tenant_id=request.tenant_id,
                    agent_id=request.agent_id,
                    name="Summarize",
                    description="Summarize channels",
                    status=SkillStatus.APPROVED,
                    metadata={"reviewed": True},
                ),
            ],
        )

    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal:
        self.skill_proposals.append(proposal)
        return proposal

    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult:
        self.skill_usages.append(usage)
        return EventIngestResult(
            event_id=usage.skill_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def start_reasoning_trace(
        self,
        request: ReasoningTraceStartRequest,
    ) -> ReasoningTraceStartResponse:
        self.trace_starts.append(request)
        return ReasoningTraceStartResponse(
            trace_id="trace-1",
            session_id=request.session_id,
            task=request.task,
        )

    async def add_reasoning_step(
        self,
        request: ReasoningStepRequest,
    ) -> ReasoningStepResponse:
        self.steps.append(request)
        return ReasoningStepResponse(
            step_id="step-1",
            trace_id=request.trace_id,
            step_number=1,
        )

    async def record_reasoning_tool_call(
        self,
        request: ReasoningToolCallRequest,
    ) -> ReasoningToolCallResponse:
        self.tool_calls.append(request)
        return ReasoningToolCallResponse(
            tool_call_id="tool-call-1",
            trace_id=request.trace_id,
            step_id=request.step_id,
        )

    async def complete_reasoning_trace(
        self,
        request: ReasoningTraceCompleteRequest,
    ) -> ReasoningTraceCompleteResponse:
        self.completions.append(request)
        return ReasoningTraceCompleteResponse(
            trace_id=request.trace_id,
            success=request.success,
            outcome=request.outcome,
            completed_at="2026-06-28T01:02:03Z",
        )

    async def get_reasoning_context(
        self,
        request: ReasoningContextRequest,
    ) -> ReasoningContextResponse:
        self.reasoning_context_requests.append(request)
        return self.reasoning_context

    async def list_reasoning_traces(
        self,
        request: ReasoningTraceListRequest,
    ) -> ReasoningTraceListResponse:
        self.reasoning_trace_list_requests.append(request)
        return self.reasoning_trace_list

    async def get_reasoning_trace(
        self,
        request: ReasoningTraceDetailRequest,
    ) -> ReasoningTraceDetailResponse:
        self.reasoning_trace_detail_requests.append(request)
        return self.reasoning_trace_detail

    async def find_similar_reasoning_traces(
        self,
        request: ReasoningSimilarTracesRequest,
    ) -> ReasoningSimilarTracesResponse:
        self.reasoning_similar_trace_requests.append(request)
        return self.reasoning_similar_traces

    async def search_reasoning_steps(
        self,
        request: ReasoningStepSearchRequest,
    ) -> ReasoningStepSearchResponse:
        self.reasoning_step_search_requests.append(request)
        return self.reasoning_step_search

    async def get_reasoning_tool_stats(
        self,
        request: ReasoningToolStatsRequest,
    ) -> ReasoningToolStatsResponse:
        self.reasoning_tool_stats_requests.append(request)
        return self.reasoning_tool_stats


@dataclass(frozen=True, slots=True)
class ShortTermWrite:
    user_identifier: str
    metadata: dict[str, str]
    extract_entities: bool
    extract_relations: bool


@dataclass(frozen=True, slots=True)
class LongTermFactWrite:
    metadata: JsonObject
    generate_embedding: bool


@dataclass(frozen=True, slots=True)
class ShortTermContextQuery:
    query: str
    session_id: str
    max_messages: int
    metadata_filters: dict[str, str]


@dataclass(slots=True)
class RecordingShortTermMemory:
    context: str = ""
    messages: list[ShortTermWrite] = field(default_factory=list)
    context_queries: list[ShortTermContextQuery] = field(default_factory=list)

    async def add_message(  # noqa: PLR0913
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        user_identifier: str,
        metadata: dict[str, str],
        extract_entities: bool,
        extract_relations: bool,
    ) -> None:
        _ = (session_id, role, content, extract_entities, extract_relations)
        self.messages.append(
            ShortTermWrite(
                user_identifier=user_identifier,
                metadata=metadata,
                extract_entities=extract_entities,
                extract_relations=extract_relations,
            ),
        )

    async def get_context(
        self,
        query: str,
        *,
        session_id: str,
        max_messages: int,
        metadata_filters: dict[str, str],
    ) -> str:
        self.context_queries.append(
            ShortTermContextQuery(
                query=query,
                session_id=session_id,
                max_messages=max_messages,
                metadata_filters=metadata_filters,
            ),
        )
        return self.context


@dataclass(slots=True)
class RecordingLongTermMemory:
    facts: list[LongTermFactWrite] = field(default_factory=list)
    context_queries: list[str] = field(default_factory=list)
    context: str = "unscoped long-term context"

    async def search_entities(
        self,
        query: str,
        *,
        entity_types: list[EntityType | str] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[EntityRecord]:
        _ = (query, entity_types, limit, threshold)
        return []

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[FactRecord]:
        _ = (query, limit, threshold)
        return []

    async def search_preferences(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[PreferenceRecord]:
        _ = (query, category, limit, threshold)
        return []

    async def add_entity(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        name: str,
        entity_type: EntityType | str,
        *,
        subtype: str | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
        attributes: JsonObject | None = None,
        resolve: bool = True,
        generate_embedding: bool = True,
        deduplicate: bool = True,
        geocode: bool = True,
        enrich: bool = True,
        coordinates: tuple[float, float] | None = None,
        metadata: JsonObject | None = None,
    ) -> EntityRecord:
        _ = (
            name,
            entity_type,
            subtype,
            description,
            aliases,
            attributes,
            resolve,
            generate_embedding,
            deduplicate,
            geocode,
            enrich,
            coordinates,
            metadata,
        )
        return EntityRecord(name=name, type=str(entity_type))

    async def add_fact(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
    ) -> FactRecord:
        _ = (subject, predicate, obj, confidence, valid_from, valid_until)
        self.facts.append(
            LongTermFactWrite(
                metadata=metadata or {},
                generate_embedding=generate_embedding,
            ),
        )
        return FactRecord(subject=subject, predicate=predicate, object=obj)

    async def add_preference(  # noqa: PLR0913 - Mirrors SDK API.
        self,
        category: str,
        preference: str,
        *,
        context: str | None = None,
        confidence: float = 1.0,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
        user_identifier: str | None = None,
        applies_to: object | None = None,
    ) -> PreferenceRecord:
        _ = (
            category,
            preference,
            context,
            confidence,
            generate_embedding,
            metadata,
            user_identifier,
            applies_to,
        )
        return PreferenceRecord(category=category, preference=preference)

    async def get_preferences_for(
        self,
        user_identifier: str,
        *,
        applies_to: object | None = None,
        active_only: bool = True,
        as_of: datetime | None = None,
    ) -> list[PreferenceRecord]:
        _ = (user_identifier, applies_to, active_only, as_of)
        return []

    async def get_facts_about(
        self,
        subject: str,
        *,
        limit: int = 100,
    ) -> list[FactRecord]:
        _ = (subject, limit)
        return []

    async def link_entity_to_message(  # noqa: PLR0913 - Mirrors SDK API.
        self,
        entity: EntityRecord | UUID,
        message_id: UUID | str,
        *,
        confidence: float = 1.0,
        start_pos: int | None = None,
        end_pos: int | None = None,
        context: str | None = None,
    ) -> bool:
        _ = (entity, message_id, confidence, start_pos, end_pos, context)
        return True

    async def link_entity_to_extractor(
        self,
        entity: EntityRecord | UUID,
        extractor_name: str,
        *,
        confidence: float = 1.0,
        extraction_time_ms: float | None = None,
    ) -> bool:
        _ = (entity, extractor_name, confidence, extraction_time_ms)
        return True

    async def get_context(self, query: str, *, max_items: int) -> str:
        _ = max_items
        self.context_queries.append(query)
        return self.context


@dataclass(slots=True)
class RecordingReasoningMemory:
    context: str = ""
    context_queries: list[str] = field(default_factory=list)

    async def get_context(self, query: str, *, max_traces: int) -> str:
        _ = max_traces
        self.context_queries.append(query)
        return self.context

    async def start_trace(  # noqa: PLR0913
        self,
        session_id: str,
        task: str,
        *,
        generate_embedding: bool,
        metadata: JsonObject | None,
        triggered_by_message_id: str | None,
        user_identifier: str,
    ) -> SdkReasoningTrace:
        _ = (generate_embedding, metadata, triggered_by_message_id, user_identifier)
        return SdkReasoningTrace(session_id=session_id, task=task)

    async def add_step(  # noqa: PLR0913
        self,
        trace_id: UUID,
        *,
        thought: None,
        action: str | None,
        observation: str | None,
        generate_embedding: bool,
        metadata: JsonObject | None,
    ) -> SdkReasoningStep:
        _ = (thought, action, observation, generate_embedding, metadata)
        return SdkReasoningStep(trace_id=trace_id, step_number=1)

    async def record_tool_call(  # noqa: PLR0913
        self,
        step_id: UUID,
        tool_name: str,
        arguments: JsonObject,
        *,
        result: JsonValue | None,
        status: ToolCallStatus,
        duration_ms: int | None,
        error: str | None,
        message_id: str | None,
        touched_entities: list[EntityRef],
    ) -> ToolCall:
        _ = (result, message_id, touched_entities)
        return ToolCall(
            step_id=step_id,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            duration_ms=duration_ms,
            error=error,
        )

    async def complete_trace(
        self,
        trace_id: UUID,
        *,
        outcome: str | None,
        success: bool | None,
        generate_step_embeddings: bool,
    ) -> SdkReasoningTrace:
        _ = generate_step_embeddings
        return SdkReasoningTrace(
            id=trace_id,
            session_id="guild:123:channel:456",
            task="discord_reply",
            outcome=outcome,
            success=success,
        )

    async def list_traces(
        self,
        *,
        session_id: str | None = None,
        success_only: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SdkReasoningTrace]:
        _ = (session_id, success_only, limit, offset)
        return []

    async def get_trace(self, trace_id: UUID | str) -> SdkReasoningTrace | None:
        _ = trace_id
        return None

    async def get_trace_with_steps(self, trace_id: UUID) -> SdkReasoningTrace | None:
        _ = trace_id
        return None

    async def get_similar_traces(
        self,
        task: str,
        *,
        limit: int = 5,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[SdkReasoningTrace]:
        _ = (task, limit, success_only, threshold)
        return []

    async def search_steps(
        self,
        query: str,
        *,
        limit: int = 10,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[object]:
        _ = (query, limit, success_only, threshold)
        return []

    async def get_tool_stats(self, tool_name: str | None = None) -> list[ToolStats]:
        _ = tool_name
        return []


@dataclass(frozen=True, slots=True)
class CypherCall:
    statement: str
    parameters: dict[str, JsonValue]


@dataclass(slots=True)
class RecordingQuery:
    rows: list[JsonObject] = field(default_factory=list)
    cypher_calls: list[CypherCall] = field(default_factory=list)

    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.cypher_calls.append(CypherCall(statement=query, parameters=params or {}))
        return self.rows


@dataclass(slots=True)
class RecordingMemoryClient:
    context: str = ""
    query: "RecordingQuery" = field(default_factory=RecordingQuery)
    short_term: RecordingShortTermMemory = field(init=False)
    long_term: RecordingLongTermMemory = field(default_factory=RecordingLongTermMemory)
    reasoning: RecordingReasoningMemory = field(
        default_factory=RecordingReasoningMemory,
    )

    def __post_init__(self) -> None:
        self.short_term = RecordingShortTermMemory(context=self.context)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


@dataclass(slots=True)
class RecordingGraphStore:
    context: str = ""
    facts: list[JsonObject] = field(default_factory=list)
    context_requests: list[GraphContextRequest] = field(default_factory=list)

    async def require_available(self) -> None:
        return None

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        self.context_requests.append(request)
        return GraphContextResponse(context=self.context, facts=self.facts)


@dataclass(frozen=True, slots=True)
class MemoryClientFactory:
    client: RecordingMemoryClient

    def __call__(self, settings: MemorySettings) -> MemoryClientContext:
        _ = settings
        return self.client


def _settings() -> Settings:
    return Settings()


def _json_object(value: object) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def _auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _read_operator_auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer read-operator-token"}


def _write_operator_auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer write-operator-token"}


def _export_operator_auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer export-operator-token"}


def _admin_operator_auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer admin-operator-token"}


def _protected_post_endpoint_payloads() -> list[tuple[str, dict[str, object]]]:
    payloads: list[tuple[str, dict[str, object]]] = [
        (
            "/v1/messages",
            {"scope": _scope_payload(), "role": "user", "content": "remember this"},
        ),
        ("/v1/memory/extraction/preview", _extraction_preview_payload()),
        ("/v1/memories", _memory_add_payload()),
        (
            "/v1/memories/search",
            {"scope": _scope_payload(), "query": "what matters?"},
        ),
        ("/v1/memories/list", {"scope": _scope_payload()}),
        (
            "/v1/memories/promote",
            {"scope": _scope_payload(), "peer": "nolgia"},
        ),
        ("/v1/events", _client_event_payload()),
        ("/v1/events/batch", {"events": [_client_event_payload()]}),
        ("/v1/context", {"scope": _scope_payload(), "query": "what matters?"}),
        (
            "/v1/memory/context",
            {"scope": _scope_payload(), "query": "what matters?"},
        ),
        (
            "/v1/graph/context",
            {"scope": _scope_payload(), "query": "what matters?"},
        ),
        ("/v1/sdk/stats", {"scope": _scope_payload()}),
        ("/v1/memory/consolidation/dry-run", _consolidation_dry_run_payload()),
        ("/v1/memory/consolidation/apply", _consolidation_apply_payload()),
        ("/v1/memory/graph/export", {"scope": _scope_payload(), "limit": 10}),
        ("/v1/skills", {"tenant_id": "bromigos", "agent_id": "pc-principal"}),
        ("/v1/skills/proposals", _skill_proposal_payload()),
        ("/v1/skills/usage", _skill_usage_payload()),
        (
            "/v1/reasoning/traces",
            {
                "scope": _scope_payload(),
                "session_id": "guild:123:channel:456",
                "task": "discord_reply",
            },
        ),
        (
            "/v1/reasoning/traces/trace-1/steps",
            {
                "scope": _scope_payload(),
                "trace_id": "trace-1",
                "action": "get_memory_context",
            },
        ),
        (
            "/v1/reasoning/steps/step-1/tool-calls",
            {
                "scope": _scope_payload(),
                "trace_id": "trace-1",
                "step_id": "step-1",
                "tool_name": "memory.get_context",
                "status": "success",
            },
        ),
        (
            "/v1/reasoning/traces/trace-1/complete",
            {"scope": _scope_payload(), "trace_id": "trace-1", "success": True},
        ),
        (
            "/v1/reasoning/context",
            {"scope": _scope_payload(), "query": "what reasoning applies?"},
        ),
    ]
    return payloads


def _scope_payload(
    *,
    tenant_id: str = "bromigos",
    guild_id: str | None = "123",
    channel_id: str | None = "456",
) -> dict[str, str]:
    payload = {
        "tenant_id": tenant_id,
        "space_id": "discord",
        "agent_id": "pc-principal",
        "session_id": "guild:123:channel:456",
        "user_id": "789",
        "visibility": "channel",
    }
    if guild_id is not None:
        payload["guild_id"] = guild_id
    if channel_id is not None:
        payload["channel_id"] = channel_id
    return payload


def _tenant_scope_payload() -> dict[str, str]:
    return {
        "tenant_id": "bromigos",
        "space_id": "tenant",
        "agent_id": "operator",
        "session_id": "tenant:bromigos",
        "user_id": "operator",
        "visibility": "tenant",
    }


def _message_payload(
    *,
    scope: dict[str, str] | None = None,
) -> dict[str, str | dict[str, str]]:
    return {
        "scope": scope or _scope_payload(),
        "role": "user",
        "content": "remember this",
    }


def _memory_add_payload(
    *,
    scope: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "scope": scope or _scope_payload(),
        "content": "Cartman prefers cheesy poofs",
        "infer": False,
        "metadata": {"topic": "snacks"},
    }


def _dedup_apply_payload() -> dict[str, object]:
    return {
        "scope": _scope_payload(),
        "apply": True,
        "operation": "reject",
        "candidate_id": "dedup-1",
        "candidate_version": 1,
        "graph_snapshot_hash": "snapshot-1",
        "dry_run_token": "reject-token",
        "idempotency_key": "dedup-apply-1",
        "audit": {"operator_id": "admin-1", "reason": "not duplicate"},
    }


def _consolidation_dry_run_payload(
    *,
    scope: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "scope": scope or _scope_payload(),
        "operation": "dedupe_entities",
        "similarity_threshold": 0.91,
        "max_pairs": 25,
    }


def _consolidation_apply_payload() -> dict[str, object]:
    return {
        "scope": _scope_payload(),
        "apply": True,
        "operation": "dedupe_entities",
        "graph_snapshot_hash": "consolidation-snapshot-1",
        "dry_run_token": "consolidation-token",
        "idempotency_key": "consolidation-apply-1",
        "audit": {"operator_id": "admin-1", "reason": "reviewed"},
        "similarity_threshold": 0.91,
        "max_pairs": 25,
    }


def _extraction_preview_payload(
    *,
    scope: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "scope": scope or _scope_payload(),
        "raw_text_documents": [
            {
                "source_id": "doc-1",
                "text": "Kenny prefers concise plans.",
            },
        ],
        "extract_entities": True,
    }


def _client_event_payload(
    *,
    scope: dict[str, str] | None = None,
    event_type: str = "message_created",
    subject: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "tenant_id": "bromigos",
        "source_client": "discord",
        "agent_id": "pc-principal",
        "event_id": "discord-message-999",
        "event_type": event_type,
        "occurred_at": "2026-06-27T01:02:03Z",
        "observed_at": "2026-06-27T01:02:04Z",
        "idempotency_key": "discord:message:message-999:create",
        "scope": scope or _scope_payload(),
        "actor": {
            "id": "789",
            "display_name": "cartman",
            "is_bot": False,
        },
        "subject": subject or {"id": "message-999", "type": "message"},
        "payload": {
            "content": "remember this",
            "payload_version": 1,
        },
        "discord": {
            "guild_id": "123",
            "channel_id": "456",
            "message_id": "message-999",
        },
    }


def _skill_proposal_payload(*, tenant_id: str = "bromigos") -> dict[str, object]:
    return {
        "proposal_id": "proposal-1",
        "tenant_id": tenant_id,
        "agent_id": "pc-principal",
        "proposed_by": "789",
        "name": "Summarize",
        "description": "Summarize channels",
        "scope": "agent_shared",
        "metadata": {"source": "test"},
    }


def _skill_usage_payload(*, tenant_id: str = "bromigos") -> dict[str, object]:
    return {
        "skill_id": "skill-1",
        "tenant_id": tenant_id,
        "agent_id": "pc-principal",
        "used_by": "789",
        "used_at": "2026-06-27T01:02:05Z",
        "scope": "agent_shared",
        "metadata": {"outcome": "ok"},
    }
