from dataclasses import dataclass, field
from os import environ
from typing import Self
from uuid import UUID

import pytest
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolCallStatus
from neo4j_agent_memory.schema.models import EntityRef
from pydantic import ValidationError

environ["AGENTS_MEMORY_TOKEN"] = "test-token"
environ["NEO4J_URI"] = "bolt://neo4j.neo4j.svc.cluster.local:7687"
environ["NEO4J_PASSWORD"] = "test-password"
environ["LITELLM_BASE_URL"] = "http://litellm.litellm.svc.cluster.local:4000/v1"
environ["LITELLM_API_KEY"] = "test-litellm-key"

from fastapi.testclient import TestClient
from neo4j_agent_memory import MemorySettings

from agents_memory.backend import MemoryClientContext, Neo4jAgentMemoryBackend
from agents_memory.main import create_app
from agents_memory.models import (
    BackendReadiness,
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ClientEventType,
    ContextRequest,
    ContextResponse,
    DiagnosticsConfig,
    DiagnosticsResponse,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    JsonObject,
    JsonValue,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSection,
    MemoryScope,
    MemoryVisibility,
    MessageWriteRequest,
    MessageWriteResponse,
    ReasoningContextRequest,
    ReasoningContextResponse,
    ReasoningStepRequest,
    ReasoningStepResponse,
    ReasoningToolCallRequest,
    ReasoningToolCallResponse,
    ReasoningTraceCompleteRequest,
    ReasoningTraceCompleteResponse,
    ReasoningTraceStartRequest,
    ReasoningTraceStartResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillRecord,
    SkillStatus,
    SkillUsage,
    default_event_visibility,
    default_skill_visibility,
)
from agents_memory.settings import Settings


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
    for path, payload in _protected_post_endpoint_payloads():
        response = client.post(path, json=payload)
        assert response.status_code == 401, path

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


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
            "memory_llm": "openai/gemma4",
            "memory_embedding": "local-qwen3-embedding-0.6b",
            "memory_embedding_dimensions": 1024,
        },
        "backend": {"graph": "ready", "schema": "ready"},
    }


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
        _settings(),
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
        "trace_id": "trace-1",
        "thought": "private internal reasoning",
        "chain_of_thought": "private internal reasoning",
    }
    tool_call_payload = {
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
    backend_available: bool = True
    messages: list[MessageWriteRequest] = field(default_factory=list)
    context_requests: list[ContextRequest] = field(default_factory=list)
    memory_context_requests: list[MemoryContextRequest] = field(default_factory=list)
    events: list[ClientEvent] = field(default_factory=list)
    graph_context_requests: list[GraphContextRequest] = field(default_factory=list)
    trace_starts: list[ReasoningTraceStartRequest] = field(default_factory=list)
    steps: list[ReasoningStepRequest] = field(default_factory=list)
    tool_calls: list[ReasoningToolCallRequest] = field(default_factory=list)
    completions: list[ReasoningTraceCompleteRequest] = field(default_factory=list)
    reasoning_context_requests: list[ReasoningContextRequest] = field(
        default_factory=list,
    )
    duplicate_event_ids: set[str] = field(default_factory=set)
    skill_usages: list[SkillUsage] = field(default_factory=list)
    skill_list_requests: list[SkillListRequest] = field(default_factory=list)
    skill_proposals: list[SkillProposal] = field(default_factory=list)
    readiness_checks: int = 0

    async def readiness(self) -> BackendReadiness:
        self.readiness_checks += 1
        if self.backend_available:
            return BackendReadiness(graph="ready", schema="ready")
        return BackendReadiness(graph="unavailable", schema="unavailable")

    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse:
        return DiagnosticsResponse(
            tenant_id="bromigos",
            config=DiagnosticsConfig(
                neo4j_uri="bolt://neo4j.neo4j.svc.cluster.local:7687",
                neo4j_username="neo4j",
                litellm_base_url="http://litellm.litellm.svc.cluster.local:4000/v1",
                memory_llm="openai/gemma4",
                memory_embedding="local-qwen3-embedding-0.6b",
                memory_embedding_dimensions=1024,
            ),
            backend=readiness,
        )

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        self.messages.append(request)
        return MessageWriteResponse(accepted=True)

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        self.context_requests.append(request)
        return ContextResponse(context=self.context)

    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        self.memory_context_requests.append(request)
        return self.memory_context

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


@dataclass(frozen=True, slots=True)
class ShortTermWrite:
    user_identifier: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class LongTermFactWrite:
    metadata: dict[str, str]
    generate_embedding: bool


@dataclass(frozen=True, slots=True)
class ShortTermContextQuery:
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
            ShortTermWrite(user_identifier=user_identifier, metadata=metadata),
        )

    async def get_context(
        self,
        query: str,
        *,
        session_id: str,
        max_messages: int,
        metadata_filters: dict[str, str],
    ) -> str:
        _ = (query, session_id, max_messages)
        self.context_queries.append(
            ShortTermContextQuery(metadata_filters=metadata_filters),
        )
        return self.context


@dataclass(slots=True)
class RecordingLongTermMemory:
    facts: list[LongTermFactWrite] = field(default_factory=list)
    context_queries: list[str] = field(default_factory=list)
    context: str = "unscoped long-term context"

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        metadata: dict[str, str],
        generate_embedding: bool,
    ) -> None:
        _ = (subject, predicate, obj)
        self.facts.append(
            LongTermFactWrite(
                metadata=metadata,
                generate_embedding=generate_embedding,
            ),
        )

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


def _auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _protected_post_endpoint_payloads() -> list[tuple[str, dict[str, object]]]:
    payloads: list[tuple[str, dict[str, object]]] = [
        (
            "/v1/messages",
            {"scope": _scope_payload(), "role": "user", "content": "remember this"},
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
            {"trace_id": "trace-1", "action": "get_memory_context"},
        ),
        (
            "/v1/reasoning/steps/step-1/tool-calls",
            {
                "trace_id": "trace-1",
                "step_id": "step-1",
                "tool_name": "memory.get_context",
                "status": "success",
            },
        ),
        (
            "/v1/reasoning/traces/trace-1/complete",
            {"trace_id": "trace-1", "success": True},
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


def _message_payload(
    *,
    scope: dict[str, str] | None = None,
) -> dict[str, str | dict[str, str]]:
    return {
        "scope": scope or _scope_payload(),
        "role": "user",
        "content": "remember this",
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
